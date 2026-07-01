#!/usr/bin/env python3
"""Unified incremental quality analysis for mission smooth LeRobot datasets."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cv2 = None


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_DATASET_DIR = MISSION_DIR / "smooth"
DEFAULT_OUTPUT_DIR = MISSION_DIR / "quality"
DEFAULT_RAW_ROOT = MISSION_DIR / "raw"

QUALITY_ANALYZER_VERSION = "quality.v4_state_action_rmse"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMPACT_DATE_RE = re.compile(r"(20\d{6})")

ACTION_EEF_SLICE = slice(0, 9)
ACTION_HAND_SLICE = slice(9, 19)
STATE_ARM_SLICE = slice(0, 7)
STATE_EEF_SLICE = slice(7, 16)
STATE_HAND_SLICE = slice(16, 26)
STATE_ACTION_DIM_PAIRS = tuple(
    [(f"eef_{idx}", STATE_EEF_SLICE.start + idx, ACTION_EEF_SLICE.start + idx) for idx in range(STATE_EEF_SLICE.stop - STATE_EEF_SLICE.start)]
    + [
        (f"hand_{idx}", STATE_HAND_SLICE.start + idx, ACTION_HAND_SLICE.start + idx)
        for idx in range(STATE_HAND_SLICE.stop - STATE_HAND_SLICE.start)
    ]
)
LAG_SEARCH_FRAMES = 20
DEFAULT_VIDEO_SAMPLE_FRAMES = 24


def _major_minor(version: str) -> tuple[int, int]:
    parts = re.match(r"(\d+)\.(\d+)", version)
    if not parts:
        return (0, 0)
    return (int(parts.group(1)), int(parts.group(2)))


MATPLOTLIB_BOXPLOT_LABEL_KEY = "tick_labels" if _major_minor(matplotlib.__version__) >= (3, 9) else "labels"


@dataclass(frozen=True)
class EpisodeRecord:
    row: dict[str, Any]
    episode_index: int
    source_date: str
    data_path: Path
    video_paths: dict[str, Path]
    fingerprint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--force", action="store_true", help="Recompute all episodes even if fingerprints match.")
    parser.add_argument("--skip-video", action="store_true", help="Skip expensive video quality metrics.")
    parser.add_argument("--skip-timing", action="store_true", help="Skip raw-log timing quality metrics.")
    parser.add_argument(
        "--video-samples",
        type=int,
        default=DEFAULT_VIDEO_SAMPLE_FRAMES,
        help="Maximum sampled frames per episode video stream for video quality metrics.",
    )
    parser.add_argument(
        "--embedding-backend",
        help="Optional backend as module:function. Callable receives an episode context dict and returns embeddings.",
    )
    parser.add_argument(
        "--object-detector-backend",
        help="Optional backend as module:function. Callable receives RGB frame and context, returns visibility score or dict.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def compact_date_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def iso_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = DATE_RE.search(value)
    return match.group(0) if match else None


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("teleop_stack_metadata")
    return value if isinstance(value, dict) else {}


def row_value(row: dict[str, Any], key: str) -> Any:
    return row.get(key, metadata(row).get(key))


def source_date(row: dict[str, Any]) -> str:
    candidates = [
        row_value(row, "raw_episode_id"),
        row_value(row, "raw_episode_dir"),
        row_value(row, "source_dataset_name"),
        row_value(row, "source_dataset"),
        row_value(row, "data_path"),
        row.get("created_at_utc"),
        row_value(row, "trimmed_at_utc"),
        row_value(row, "smoothed_at_utc"),
    ]
    for value in candidates:
        if not isinstance(value, str):
            continue
        compact = COMPACT_DATE_RE.search(value)
        if compact:
            return compact_date_to_iso(compact.group(1))
        date = iso_date(value)
        if date:
            return date
    return "unknown"


def data_path_for_episode(dataset_dir: Path, info: dict[str, Any], row: dict[str, Any]) -> Path:
    meta = metadata(row)
    for key in ("data_path", "output_data_path", "smooth_data_path"):
        value = meta.get(key, row.get(key))
        if isinstance(value, str):
            return dataset_dir / value
    episode_index = int(row["episode_index"])
    chunk_size = int(info.get("chunks_size", 1000))
    pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    return dataset_dir / pattern.format(
        episode_chunk=episode_index // max(1, chunk_size),
        episode_index=episode_index,
    )


def video_paths_for_episode(dataset_dir: Path, info: dict[str, Any], row: dict[str, Any]) -> dict[str, Path]:
    meta = metadata(row)
    raw_paths = meta.get("video_paths") or row.get("video_paths") or meta.get("output_video_paths")
    if isinstance(raw_paths, dict):
        return {str(key): dataset_dir / str(value) for key, value in raw_paths.items()}
    pattern = info.get("video_path")
    if not isinstance(pattern, str):
        return {}
    episode_index = int(row["episode_index"])
    chunk_size = int(info.get("chunks_size", 1000))
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    video_features = [key for key, spec in features.items() if isinstance(spec, dict) and spec.get("dtype") == "video"]
    return {
        feature_key: dataset_dir
        / pattern.format(
            episode_chunk=episode_index // max(1, chunk_size),
            episode_index=episode_index,
            video_key=feature_key,
        )
        for feature_key in video_features
    }


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str, separators=(",", ":"))


def episode_fingerprint(
    row: dict[str, Any],
    data_path: Path,
    video_paths: dict[str, Path],
    analysis_signature: dict[str, Any],
) -> str:
    payload = {
        "version": QUALITY_ANALYZER_VERSION,
        "analysis_signature": analysis_signature,
        "episode_index": row.get("episode_index"),
        "length": row.get("length"),
        "tasks": row.get("tasks"),
        "metadata": metadata(row),
        "data": file_stat(data_path),
        "videos": {key: file_stat(path) for key, path in sorted(video_paths.items())},
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def discover_episodes(dataset_dir: Path, analysis_signature: dict[str, Any]) -> tuple[dict[str, Any], list[EpisodeRecord]]:
    info = read_json(dataset_dir / "meta" / "info.json")
    rows = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    records: list[EpisodeRecord] = []
    for row in rows:
        episode_index = int(row["episode_index"])
        data_path = data_path_for_episode(dataset_dir, info, row)
        video_paths = video_paths_for_episode(dataset_dir, info, row)
        records.append(
            EpisodeRecord(
                row=row,
                episode_index=episode_index,
                source_date=source_date(row),
                data_path=data_path,
                video_paths=video_paths,
                fingerprint=episode_fingerprint(row, data_path, video_paths, analysis_signature),
            )
        )
    return info, records


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise ValueError(f"Column {column_name!r} has inconsistent vector widths")
    return np.vstack(values)


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def robust_high_outlier_mask(series: pd.Series, z_threshold: float = 3.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return pd.Series(False, index=series.index)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 1e-12:
        p95 = float(np.percentile(finite, 95))
        return values > p95 if p95 > median else pd.Series(False, index=series.index)
    robust_z = 0.6745 * (values - median) / mad
    return robust_z > z_threshold


def robust_low_outlier_mask(series: pd.Series, z_threshold: float = 3.5) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return pd.Series(False, index=series.index)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 1e-12:
        p05 = float(np.percentile(finite, 5))
        return values < p05 if p05 < median else pd.Series(False, index=series.index)
    robust_z = 0.6745 * (median - values) / mad
    return robust_z > z_threshold


def add_stats(row: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    for key, value in finite_summary(values).items():
        row[f"{prefix}_{key}"] = value


def l2(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def derivative(values: np.ndarray, fps: float, order: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    for _ in range(order):
        if out.shape[0] < 2:
            return np.zeros((0, out.shape[1] if out.ndim == 2 else 0), dtype=np.float64)
        out = np.diff(out, axis=0) * fps
    return out


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def vector_metrics(row: dict[str, Any], prefix: str, values: np.ndarray, fps: float) -> None:
    if values.size == 0:
        for name in ("velocity", "acceleration", "jerk"):
            add_stats(row, f"{prefix}_{name}_l2", np.asarray([]))
        return
    add_stats(row, f"{prefix}_velocity_l2", l2(derivative(values, fps, 1)))
    add_stats(row, f"{prefix}_acceleration_l2", l2(derivative(values, fps, 2)))
    add_stats(row, f"{prefix}_jerk_l2", l2(derivative(values, fps, 3)))


def hand_metrics(row: dict[str, Any], hand_state: np.ndarray, hand_action: np.ndarray, fps: float) -> None:
    values = hand_state if hand_state.size else hand_action
    finite = values[np.isfinite(values)] if values.size else np.asarray([], dtype=np.float64)
    row["finger_angle_min"] = float(np.min(finite)) if finite.size else math.nan
    row["finger_angle_max"] = float(np.max(finite)) if finite.size else math.nan
    row["finger_angle_mean"] = float(np.mean(finite)) if finite.size else math.nan
    row["finger_angle_std"] = float(np.std(finite)) if finite.size else math.nan
    vel = derivative(values, fps, 1)
    acc = derivative(values, fps, 2)
    row["finger_velocity_p95"] = float(np.percentile(np.abs(vel[np.isfinite(vel)]), 95)) if vel.size else math.nan
    row["finger_acceleration_p95"] = float(np.percentile(np.abs(acc[np.isfinite(acc)]), 95)) if acc.size else math.nan
    row["zero_ratio"] = float(np.mean(np.abs(finite) <= 1e-6)) if finite.size else math.nan
    row["saturation_ratio"] = float(np.mean(np.abs(finite) >= 0.98)) if finite.size else math.nan
    if values.shape[0] >= 2:
        mean_velocity = np.mean(derivative(values, fps, 1), axis=1)
        threshold = max(1e-6, float(np.percentile(np.abs(mean_velocity), 75)))
        row["closing_duration"] = float(longest_true_run(mean_velocity > threshold) / fps)
    else:
        row["closing_duration"] = 0.0
    tail = values[int(values.shape[0] * 0.8) :] if values.shape[0] else values
    row["holding_stability"] = float(np.nanmean(np.nanstd(tail, axis=0))) if tail.size else math.nan


def normalize_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    std = float(np.std(values))
    if std <= 1e-12:
        return values * 0.0
    return (values - float(np.mean(values))) / std


def estimate_lag(action: np.ndarray, state: np.ndarray, max_lag: int = LAG_SEARCH_FRAMES) -> tuple[int | None, float | None]:
    action_sig = normalize_signal(l2(np.diff(action, axis=0))) if action.shape[0] > 1 else np.asarray([])
    state_sig = normalize_signal(l2(np.diff(state, axis=0))) if state.shape[0] > 1 else np.asarray([])
    n = min(action_sig.size, state_sig.size)
    if n < 4:
        return None, None
    action_sig = action_sig[:n]
    state_sig = state_sig[:n]
    best_lag: int | None = None
    best_corr: float | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = action_sig[-lag:]
            s = state_sig[: a.size]
        elif lag > 0:
            a = action_sig[:-lag]
            s = state_sig[lag:]
        else:
            a = action_sig
            s = state_sig
        if a.size < 4 or s.size < 4:
            continue
        corr = float(np.corrcoef(a, s)[0, 1])
        if not math.isfinite(corr):
            continue
        if best_corr is None or corr > best_corr:
            best_lag = lag
            best_corr = corr
    return best_lag, best_corr


def _resolve_raw_episode_dir(raw_root: Path, meta: dict[str, Any]) -> Path | None:
    raw_dir = meta.get("raw_episode_dir")
    if isinstance(raw_dir, str):
        path = Path(raw_dir)
        if path.is_dir():
            return path
        match = re.search(r"(20\d{6})", raw_dir)
        episode_id = Path(raw_dir).name
        capture = Path(raw_dir).parent.parent.name if "/episodes/" in raw_dir else None
        if match and capture:
            candidate = raw_root / compact_date_to_iso(match.group(1)) / capture / "episodes" / episode_id
            if candidate.is_dir():
                return candidate
    raw_episode_id = meta.get("raw_episode_id")
    if isinstance(raw_episode_id, str):
        match = COMPACT_DATE_RE.search(raw_episode_id)
        search_roots = [raw_root / compact_date_to_iso(match.group(1))] if match else [raw_root]
        for root in search_roots:
            if not root.exists():
                continue
            found = list(root.glob(f"*/episodes/{raw_episode_id}"))
            if found:
                return found[0]
    return None


def timing_metrics(record: EpisodeRecord, action: np.ndarray, state: np.ndarray, raw_root: Path, fps: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_index": record.episode_index,
        "source_date": record.source_date,
        "timing_status": "unavailable",
        "mean_dt_video_state": math.nan,
        "std_dt_video_state": math.nan,
        "max_abs_time_offset": math.nan,
        "action_state_lag": math.nan,
        "action_state_lag_s": math.nan,
        "action_state_lag_corr": math.nan,
        "camera_frame_jitter": math.nan,
    }
    lag, corr = estimate_lag(action, state)
    if lag is not None:
        row["action_state_lag"] = int(lag)
        row["action_state_lag_s"] = float(lag / fps)
        row["action_state_lag_corr"] = float(corr) if corr is not None else math.nan

    raw_episode_dir = _resolve_raw_episode_dir(raw_root, metadata(record.row))
    if raw_episode_dir is None:
        row["timing_reason"] = "raw_episode_not_found"
        return row
    frames_path = raw_episode_dir / "frames.jsonl"
    video_path = raw_episode_dir / "video.jsonl"
    if not frames_path.exists():
        row["timing_reason"] = "frames_jsonl_missing"
        return row
    frames = read_jsonl(frames_path)
    video_records = read_jsonl(video_path) if video_path.exists() else []
    video_state_dts: list[float] = []
    max_offsets: list[float] = []
    for frame in frames:
        ts = frame.get("monotonic_ts_s")
        if not isinstance(ts, (int, float)):
            continue
        videos = frame.get("videos")
        if not isinstance(videos, list):
            continue
        for video in videos:
            if not isinstance(video, dict):
                continue
            vts = video.get("monotonic_ts_s")
            if isinstance(vts, (int, float)):
                offset = float(vts) - float(ts)
                video_state_dts.append(offset)
                max_offsets.append(abs(offset))
    camera_jitter_values: list[float] = []
    by_camera: dict[str, list[float]] = {}
    for rec in video_records:
        camera = rec.get("camera_name")
        ts = rec.get("monotonic_ts_s")
        if isinstance(camera, str) and isinstance(ts, (int, float)):
            by_camera.setdefault(camera, []).append(float(ts))
    for values in by_camera.values():
        values = sorted(values)
        if len(values) >= 3:
            dt = np.diff(np.asarray(values, dtype=np.float64))
            camera_jitter_values.extend(np.abs(dt - float(np.median(dt))).tolist())
    if video_state_dts:
        offsets = np.asarray(video_state_dts, dtype=np.float64)
        row["timing_status"] = "ok"
        row["timing_reason"] = "raw_logs"
        row["mean_dt_video_state"] = float(np.mean(offsets))
        row["std_dt_video_state"] = float(np.std(offsets))
        row["max_abs_time_offset"] = float(np.max(np.asarray(max_offsets, dtype=np.float64)))
    else:
        row["timing_reason"] = "video_state_offsets_unavailable"
    row["camera_frame_jitter"] = float(np.mean(camera_jitter_values)) if camera_jitter_values else math.nan
    return row


def video_frame_count(path: Path) -> int | None:
    if cv2 is None or not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count >= 0 else None


def sample_indices(count: int, max_samples: int = DEFAULT_VIDEO_SAMPLE_FRAMES) -> list[int]:
    if count <= 0:
        return []
    if count <= max_samples:
        return list(range(count))
    return sorted(set(int(round(v)) for v in np.linspace(0, count - 1, max_samples)))


def load_backend(spec: str | None) -> Callable[..., Any] | None:
    if not spec:
        return None
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise RuntimeError(f"Backend spec must be module:function, got {spec!r}")
    module = importlib.import_module(module_name)
    backend = getattr(module, attr)
    if not callable(backend):
        raise RuntimeError(f"Backend is not callable: {spec}")
    return backend


def call_object_detector(detector: Callable[..., Any] | None, frame_rgb: np.ndarray, context: dict[str, Any]) -> float | None:
    if detector is None:
        return None
    result = detector(frame_rgb, context)
    if isinstance(result, dict):
        value = result.get("object_visibility_ratio", result.get("visibility", result.get("score")))
    else:
        value = result
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def video_quality(
    record: EpisodeRecord,
    detector: Callable[..., Any] | None,
    skip_video: bool,
    video_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_key, path in sorted(record.video_paths.items()):
        base = {
            "episode_index": record.episode_index,
            "source_date": record.source_date,
            "video_key": video_key,
            "video_path": str(path),
            "object_visibility_status": "skipped" if detector is None else "ok",
        }
        count = video_frame_count(path)
        base["video_frames"] = count if count is not None else math.nan
        if skip_video:
            base["video_status"] = "skipped"
            rows.append(base)
            continue
        if cv2 is None:
            base["video_status"] = "unavailable"
            base["video_reason"] = "cv2_missing"
            rows.append(base)
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened() or count is None:
            base["video_status"] = "error"
            base["video_reason"] = "open_failed"
            rows.append(base)
            continue
        blur: list[float] = []
        brightness: list[float] = []
        over: list[float] = []
        under: list[float] = []
        frame_diff: list[float] = []
        flow_mag: list[float] = []
        visibility: list[float] = []
        prev_gray: np.ndarray | None = None
        prev_small: np.ndarray | None = None
        try:
            for frame_index in sample_indices(count, max(1, int(video_samples))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                brightness.append(float(np.mean(gray)))
                over.append(float(np.mean(gray >= 245)))
                under.append(float(np.mean(gray <= 10)))
                small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
                if prev_small is not None:
                    frame_diff.append(float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32)))))
                if prev_gray is not None:
                    prev_flow = cv2.resize(prev_gray, (160, 120), interpolation=cv2.INTER_AREA)
                    flow = cv2.calcOpticalFlowFarneback(prev_flow, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
                    flow_mag.append(float(np.mean(np.linalg.norm(flow, axis=2))))
                score = call_object_detector(
                    detector,
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    {"episode_index": record.episode_index, "video_key": video_key, "frame_index": frame_index},
                )
                if score is not None:
                    visibility.append(score)
                prev_gray = gray
                prev_small = small
        finally:
            cap.release()
        base.update(
            {
                "video_status": "ok",
                "sampled_frames": len(blur),
                "blur_score": float(np.mean(blur)) if blur else math.nan,
                "brightness_mean": float(np.mean(brightness)) if brightness else math.nan,
                "brightness_std": float(np.std(brightness)) if brightness else math.nan,
                "overexposure_ratio": float(np.mean(over)) if over else math.nan,
                "underexposure_ratio": float(np.mean(under)) if under else math.nan,
                "frame_difference": float(np.mean(frame_diff)) if frame_diff else math.nan,
                "optical_flow_magnitude": float(np.mean(flow_mag)) if flow_mag else math.nan,
                "object_visibility_ratio": float(np.mean(visibility)) if visibility else math.nan,
            }
        )
        rows.append(base)
    return rows


def compact_series(values: np.ndarray, max_points: int = 240) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    if arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, max_points).astype(int)
        arr = arr[idx]
    return [float(v) if math.isfinite(float(v)) else math.nan for v in arr]


def state_action_comparison(state: np.ndarray, action: np.ndarray) -> dict[str, Any]:
    n = min(state.shape[0], action.shape[0])
    labels: list[str] = []
    state_series: list[list[float]] = []
    action_series: list[list[float]] = []
    rmse_by_dim: list[float] = []
    finite_diffs: list[np.ndarray] = []
    for label, state_idx, action_idx in STATE_ACTION_DIM_PAIRS:
        if state.shape[1] <= state_idx or action.shape[1] <= action_idx or n <= 0:
            continue
        state_values = state[:n, state_idx]
        action_values = action[:n, action_idx]
        diff = state_values - action_values
        finite = diff[np.isfinite(diff)]
        rmse = float(np.sqrt(np.mean(np.square(finite)))) if finite.size else math.nan
        labels.append(label)
        state_series.append(compact_series(state_values))
        action_series.append(compact_series(action_values))
        rmse_by_dim.append(rmse)
        if finite.size:
            finite_diffs.append(finite)
    if finite_diffs:
        all_diffs = np.concatenate(finite_diffs)
        overall_rmse = float(np.sqrt(np.mean(np.square(all_diffs))))
    else:
        overall_rmse = math.nan
    finite_rmse = np.asarray([value for value in rmse_by_dim if math.isfinite(value)], dtype=np.float64)
    max_dim = labels[int(np.nanargmax(rmse_by_dim))] if finite_rmse.size else ""
    return {
        "labels": labels,
        "state_series": state_series,
        "action_series": action_series,
        "rmse_by_dim": rmse_by_dim,
        "overall_rmse": overall_rmse,
        "max_dim": max_dim,
    }


def process_episode(
    record: EpisodeRecord,
    *,
    raw_root: Path,
    skip_video: bool,
    skip_timing: bool,
    object_detector: Callable[..., Any] | None,
    embedding_backend: Callable[..., Any] | None,
    fps: float,
    video_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    episode_row: dict[str, Any] = {
        "episode_index": record.episode_index,
        "source_date": record.source_date,
        "expected_frames": int(record.row.get("length", -1)),
        "data_path": str(record.data_path),
        "fingerprint": record.fingerprint,
    }
    curves: dict[str, Any] = {"episode_index": record.episode_index, "source_date": record.source_date}
    embedding_row: dict[str, Any] = {
        "episode_index": record.episode_index,
        "source_date": record.source_date,
        "embedding_status": "skipped" if embedding_backend is None else "ok",
    }
    timing_row: dict[str, Any]
    video_rows: list[dict[str, Any]]
    try:
        df = pd.read_parquet(record.data_path)
        action = stack_vector_column(df["action"], "action") if "action" in df else np.zeros((0, 0))
        state = stack_vector_column(df["observation.state"], "observation.state") if "observation.state" in df else np.zeros((0, 0))
        timestamps = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df else np.asarray([])
        episode_row.update(
            {
                "status": "ok",
                "parquet_rows": int(len(df)),
                "state_frames": int(state.shape[0]),
                "action_frames": int(action.shape[0]),
                "timestamp_frames": int(timestamps.shape[0]),
                "state_dim": int(state.shape[1]) if state.ndim == 2 else 0,
                "action_dim": int(action.shape[1]) if action.ndim == 2 else 0,
                "frame_count_match": bool(len(df) == int(record.row.get("length", len(df))) == state.shape[0] == action.shape[0]),
                "finite_state": bool(np.isfinite(state).all()) if state.size else False,
                "finite_action": bool(np.isfinite(action).all()) if action.size else False,
            }
        )
        video_counts = {key: video_frame_count(path) for key, path in sorted(record.video_paths.items())}
        episode_row["video_frames_by_key"] = json.dumps(video_counts, sort_keys=True)
        episode_row["video_frame_count_match"] = bool(video_counts) and all(count == len(df) for count in video_counts.values() if count is not None)

        vector_metrics(episode_row, "arm_joint", state[:, STATE_ARM_SLICE] if state.shape[1] >= 7 else np.zeros((0, 0)), fps)
        vector_metrics(episode_row, "eef_state", state[:, STATE_EEF_SLICE] if state.shape[1] >= 16 else np.zeros((0, 0)), fps)
        vector_metrics(episode_row, "hand_state", state[:, STATE_HAND_SLICE] if state.shape[1] >= 26 else np.zeros((0, 0)), fps)
        vector_metrics(episode_row, "eef_action", action[:, ACTION_EEF_SLICE] if action.shape[1] >= 9 else np.zeros((0, 0)), fps)
        vector_metrics(episode_row, "hand_action", action[:, ACTION_HAND_SLICE] if action.shape[1] >= 19 else np.zeros((0, 0)), fps)
        hand_metrics(
            episode_row,
            state[:, STATE_HAND_SLICE] if state.shape[1] >= 26 else np.zeros((0, 0)),
            action[:, ACTION_HAND_SLICE] if action.shape[1] >= 19 else np.zeros((0, 0)),
            fps,
        )
        if action.shape[0] >= 2:
            add_stats(episode_row, "action_step_l2", l2(np.diff(action, axis=0)))
        if state.shape[0] >= 2:
            add_stats(episode_row, "state_step_l2", l2(np.diff(state, axis=0)))
        comparison = state_action_comparison(state, action)
        episode_row["state_action_rmse"] = comparison["overall_rmse"]
        episode_row["state_action_comparable_dims"] = len(comparison["labels"])
        episode_row["state_action_rmse_max_dim"] = comparison["max_dim"]
        add_stats(episode_row, "state_action_rmse_dim", np.asarray(comparison["rmse_by_dim"], dtype=np.float64))

        curves.update(
            {
                "length": int(len(df)),
                "timestamp": compact_series(timestamps),
                "action_l2": compact_series(l2(action)),
                "state_l2": compact_series(l2(state)),
                "hand_state_mean": compact_series(np.mean(state[:, STATE_HAND_SLICE], axis=1) if state.shape[1] >= 26 else np.asarray([])),
                "eef_xyz_x": compact_series(state[:, 7] if state.shape[1] > 7 else np.asarray([])),
                "eef_xyz_y": compact_series(state[:, 8] if state.shape[1] > 8 else np.asarray([])),
                "eef_xyz_z": compact_series(state[:, 9] if state.shape[1] > 9 else np.asarray([])),
                "state_action_dim_labels": comparison["labels"],
                "state_action_state": comparison["state_series"],
                "state_action_action": comparison["action_series"],
                "state_action_rmse_by_dim": comparison["rmse_by_dim"],
                "state_action_rmse": comparison["overall_rmse"],
                "state_action_rmse_max_dim": comparison["max_dim"],
            }
        )
        timing_row = (
            {
                "episode_index": record.episode_index,
                "source_date": record.source_date,
                "timing_status": "skipped",
            }
            if skip_timing
            else timing_metrics(record, action, state, raw_root, fps)
        )
        video_rows = video_quality(record, object_detector, skip_video, video_samples)

        if embedding_backend is not None:
            result = embedding_backend(
                {
                    "record": record,
                    "dataframe": df,
                    "action": action,
                    "state": state,
                    "video_paths": record.video_paths,
                }
            )
            if isinstance(result, dict):
                for key, value in result.items():
                    if key.endswith("embedding") or key in {"video", "state_action", "language"}:
                        embedding_row[f"{key}_embedding"] = json.dumps(np.asarray(value, dtype=float).reshape(-1).tolist())
                    else:
                        embedding_row[key] = value
            else:
                embedding_row["embedding_status"] = "error"
                embedding_row["embedding_reason"] = "backend_returned_non_dict"
    except Exception as exc:  # Keep one bad episode from aborting a whole batch.
        episode_row.update({"status": "error", "error": str(exc)})
        timing_row = {"episode_index": record.episode_index, "source_date": record.source_date, "timing_status": "error", "timing_reason": str(exc)}
        video_rows = []
    processed = {
        "episode_index": record.episode_index,
        "fingerprint": record.fingerprint,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": episode_row.get("status", "unknown"),
    }
    return episode_row, video_rows, timing_row, embedding_row, curves, processed


def previous_processed_map(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        try:
            latest[int(row["episode_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return latest


def filter_cached(df: pd.DataFrame, keep: set[int]) -> pd.DataFrame:
    if df.empty or "episode_index" not in df:
        return pd.DataFrame()
    return df[df["episode_index"].astype(int).isin(keep)].copy()


def plot_length_distribution(episodes: pd.DataFrame, out_path: Path) -> None:
    if episodes.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    dates = sorted(episodes["source_date"].astype(str).unique())
    data = [episodes.loc[episodes["source_date"].astype(str) == date, "parquet_rows"].dropna().to_numpy() for date in dates]
    axes[0].boxplot(data, **{MATPLOTLIB_BOXPLOT_LABEL_KEY: dates}, showfliers=False)
    axes[0].set_title("Episode length by source date")
    axes[0].set_ylabel("frames")
    by_date = episodes.groupby("source_date").size().reindex(dates)
    axes[1].bar(dates, by_date.to_numpy())
    axes[1].set_title("Episodes by source date")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metric_boxplots(episodes: pd.DataFrame, out_path: Path) -> None:
    if episodes.empty:
        return
    metrics = [
        ("state_action_rmse", "State/action RMSE"),
        ("state_action_rmse_dim_p95", "State/action dimension RMSE p95"),
        ("action_step_l2_p95", "Action step L2 p95"),
        ("state_step_l2_p95", "State step L2 p95"),
        ("arm_joint_jerk_l2_p95", "Arm jerk p95"),
        ("finger_velocity_p95", "Finger velocity p95"),
    ]
    present = [(col, title) for col, title in metrics if col in episodes]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 4 * len(present)), constrained_layout=True)
    if len(present) == 1:
        axes = [axes]
    dates = sorted(episodes["source_date"].astype(str).unique())
    for axis, (col, title) in zip(axes, present):
        data = [episodes.loc[episodes["source_date"].astype(str) == date, col].dropna().to_numpy() for date in dates]
        axis.boxplot(data, **{MATPLOTLIB_BOXPLOT_LABEL_KEY: dates}, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_video_quality(videos: pd.DataFrame, out_path: Path) -> None:
    if videos.empty or "brightness_mean" not in videos:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for key, group in videos.groupby("video_key"):
        axes[0].hist(group["blur_score"].dropna(), bins=30, alpha=0.45, label=str(key))
        axes[1].hist(group["brightness_mean"].dropna(), bins=30, alpha=0.45, label=str(key))
    axes[0].set_title("Blur score distribution")
    axes[1].set_title("Brightness distribution")
    for axis in axes:
        axis.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_bad_episode_summary(episodes: pd.DataFrame, out_path: Path) -> None:
    if episodes.empty or "quality_score" not in episodes:
        return
    by_date = episodes.groupby("source_date").agg(
        episodes=("episode_index", "count"),
        bad=("issue_count", lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())),
        mean_score=("quality_score", "mean"),
    ).reset_index()
    by_date["bad_ratio"] = by_date["bad"] / by_date["episodes"].replace(0, np.nan)
    worst = episodes.sort_values(["quality_score", "issue_count"], ascending=[True, False]).head(20)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    axes[0].bar(by_date["source_date"].astype(str), by_date["bad_ratio"].fillna(0.0), color="#e45756")
    axes[0].set_title("Bad episode ratio by source date")
    axes[0].set_ylabel("ratio")
    axes[0].set_ylim(0, 1)
    axes[1].barh(worst["episode_index"].astype(str), worst["quality_score"], color="#f58518")
    axes[1].invert_yaxis()
    axes[1].set_title("Worst episodes by quality score")
    axes[1].set_xlabel("score, lower is worse")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_timing_quality(timing: pd.DataFrame, out_path: Path) -> None:
    if timing.empty:
        return
    cols = [
        ("mean_dt_video_state", "mean dt video-state"),
        ("std_dt_video_state", "std dt video-state"),
        ("max_abs_time_offset", "max abs time offset"),
        ("action_state_lag", "action-state lag frames"),
        ("camera_frame_jitter", "camera frame jitter"),
    ]
    present = [(col, title) for col, title in cols if col in timing]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(14, 3.3 * len(present)), constrained_layout=True)
    if len(present) == 1:
        axes = [axes]
    dates = sorted(timing["source_date"].astype(str).unique()) if "source_date" in timing else ["all"]
    for axis, (col, title) in zip(axes, present):
        data = [timing.loc[timing["source_date"].astype(str) == date, col].dropna().to_numpy() for date in dates]
        axis.boxplot(data, **{MATPLOTLIB_BOXPLOT_LABEL_KEY: dates}, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hand_quality(episodes: pd.DataFrame, out_path: Path) -> None:
    if episodes.empty:
        return
    cols = [
        ("finger_angle_mean", "finger angle mean"),
        ("finger_angle_std", "finger angle std"),
        ("finger_velocity_p95", "finger velocity p95"),
        ("finger_acceleration_p95", "finger acceleration p95"),
        ("zero_ratio", "zero ratio"),
        ("saturation_ratio", "saturation ratio"),
        ("closing_duration", "closing duration"),
        ("holding_stability", "holding stability"),
    ]
    present = [(col, title) for col, title in cols if col in episodes]
    if not present:
        return
    rows = math.ceil(len(present) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    dates = sorted(episodes["source_date"].astype(str).unique())
    for axis, (col, title) in zip(axes_flat, present):
        data = [episodes.loc[episodes["source_date"].astype(str) == date, col].dropna().to_numpy() for date in dates]
        axis.boxplot(data, **{MATPLOTLIB_BOXPLOT_LABEL_KEY: dates}, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    for axis in axes_flat[len(present):]:
        axis.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cross_modal_relations(episodes: pd.DataFrame, videos: pd.DataFrame, timing: pd.DataFrame, out_path: Path) -> None:
    if episodes.empty:
        return
    video_summary = videos.groupby("episode_index").mean(numeric_only=True).reset_index() if not videos.empty else pd.DataFrame()
    merged = episodes.merge(video_summary, on="episode_index", how="left", suffixes=("", "_video")) if not video_summary.empty else episodes.copy()
    if not timing.empty:
        merged = merged.merge(timing[[c for c in ["episode_index", "action_state_lag", "max_abs_time_offset"] if c in timing]], on="episode_index", how="left", suffixes=("", "_timing"))
    specs = [
        ("finger_velocity_p95", "frame_difference", "finger velocity vs frame difference"),
        ("finger_velocity_p95", "optical_flow_magnitude", "finger velocity vs optical flow"),
        ("eef_state_velocity_l2_p95", "optical_flow_magnitude", "EEF velocity vs optical flow"),
        ("action_step_l2_p95", "action_state_lag", "action jump vs estimated lag"),
    ]
    present = [(x, y, title) for x, y, title in specs if x in merged and y in merged]
    if not present:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for axis, (x, y, title) in zip(axes_flat, present):
        for date, group in merged.groupby("source_date"):
            axis.scatter(group[x], group[y], s=24, alpha=0.75, label=str(date))
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(title)
    if present:
        axes_flat[0].legend(fontsize=8)
    for axis in axes_flat[len(present):]:
        axis.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_issue_codes(issues: pd.DataFrame, out_path: Path) -> None:
    if issues.empty or "code" not in issues:
        return
    counts = issues["code"].value_counts().head(20)
    fig, axis = plt.subplots(figsize=(12, max(4, 0.35 * len(counts))), constrained_layout=True)
    axis.barh(counts.index.astype(str), counts.to_numpy(), color="#e45756")
    axis.invert_yaxis()
    axis.set_title("Most common quality issue codes")
    axis.set_xlabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_issue_heatmap(issues: pd.DataFrame, out_path: Path) -> None:
    if issues.empty or not {"source_date", "code"}.issubset(issues.columns):
        return
    pivot = pd.crosstab(issues["code"].astype(str), issues["source_date"].astype(str))
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:24]]
    fig, axis = plt.subplots(figsize=(max(9, 1.0 * len(pivot.columns) + 5), max(5, 0.35 * len(pivot.index) + 2)), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns.astype(str), rotation=30, ha="right")
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index.astype(str))
    axis.set_title("Issue heatmap by source date")
    for y in range(len(pivot.index)):
        for x in range(len(pivot.columns)):
            value = int(pivot.iloc[y, x])
            if value:
                axis.text(x, y, str(value), ha="center", va="center", fontsize=8, color="#111827")
    fig.colorbar(image, ax=axis, label="issue count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_issue_timeline(episodes: pd.DataFrame, issues: pd.DataFrame, out_path: Path) -> None:
    if episodes.empty or "episode_index" not in episodes:
        return
    ordered = episodes.sort_values(["source_date", "episode_index"]).copy()
    issue_counts = issues.groupby("episode_index").size() if not issues.empty and "episode_index" in issues else pd.Series(dtype=float)
    error_counts = (
        issues[issues["severity"] == "error"].groupby("episode_index").size()
        if not issues.empty and {"episode_index", "severity"}.issubset(issues.columns)
        else pd.Series(dtype=float)
    )
    ordered["plot_issue_count"] = ordered["episode_index"].map(issue_counts).fillna(0).astype(int)
    ordered["plot_error_count"] = ordered["episode_index"].map(error_counts).fillna(0).astype(int)
    colors = np.where(ordered["plot_error_count"] > 0, "#d62728", np.where(ordered["plot_issue_count"] > 0, "#f58518", "#54a24b"))
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, constrained_layout=True)
    x = np.arange(len(ordered))
    axes[0].bar(x, ordered["quality_score"].fillna(100.0), color=colors, width=0.9)
    axes[0].axhline(80, color="#f58518", linestyle="--", linewidth=1)
    axes[0].axhline(60, color="#d62728", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("quality score")
    axes[0].set_title("Episode quality timeline, sorted by date and episode")
    axes[1].bar(x, ordered["plot_issue_count"], color=colors, width=0.9)
    axes[1].set_ylabel("issue count")
    axes[1].set_xlabel("episode")
    tick_positions: list[int] = []
    tick_labels: list[str] = []
    for date, group in ordered.groupby("source_date", sort=False):
        position = int(group.index.to_series().map({idx: pos for pos, idx in enumerate(ordered.index)}).iloc[0])
        tick_positions.append(position)
        tick_labels.append(str(date))
        for axis in axes:
            axis.axvline(position - 0.5, color="#d7dde5", linewidth=0.8)
    axes[1].set_xticks(tick_positions, tick_labels, rotation=30, ha="right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def add_issue(
    issues: list[dict[str, Any]],
    episode_index: int,
    source_date: str,
    severity: str,
    code: str,
    message: str,
    metric: str | None = None,
    value: Any | None = None,
) -> None:
    issues.append(
        {
            "episode_index": int(episode_index),
            "source_date": source_date,
            "severity": severity,
            "code": code,
            "message": message,
            "metric": metric,
            "value": make_json_safe(value),
        }
    )


def build_issue_rows(episodes: pd.DataFrame, videos: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []
    if not episodes.empty:
        length_low = robust_low_outlier_mask(episodes.get("parquet_rows", pd.Series(dtype=float)))
        length_high = robust_high_outlier_mask(episodes.get("parquet_rows", pd.Series(dtype=float)))
        for idx, row in episodes.iterrows():
            episode_index = int(row.get("episode_index", -1))
            date = str(row.get("source_date", "unknown"))
            if row.get("status") != "ok":
                add_issue(issues, episode_index, date, "error", "episode_error", str(row.get("error", "episode processing failed")))
            if row.get("frame_count_match") is False:
                add_issue(issues, episode_index, date, "error", "state_action_frame_mismatch", "state/action/parquet frame counts do not match", "frame_count_match", False)
            if row.get("video_frame_count_match") is False:
                add_issue(issues, episode_index, date, "error", "video_frame_mismatch", "video frame count does not match parquet rows", "video_frame_count_match", False)
            if row.get("finite_state") is False:
                add_issue(issues, episode_index, date, "error", "nonfinite_state", "observation.state contains non-finite values")
            if row.get("finite_action") is False:
                add_issue(issues, episode_index, date, "error", "nonfinite_action", "action contains non-finite values")
            if idx in length_low.index and bool(length_low.loc[idx]):
                add_issue(issues, episode_index, date, "warning", "short_episode_outlier", "episode is unusually short for this dataset", "parquet_rows", row.get("parquet_rows"))
            if idx in length_high.index and bool(length_high.loc[idx]):
                add_issue(issues, episode_index, date, "warning", "long_episode_outlier", "episode is unusually long for this dataset", "parquet_rows", row.get("parquet_rows"))
        high_metrics = {
            "action_step_l2_p95": "large_action_jump",
            "state_step_l2_p95": "large_state_jump",
            "arm_joint_jerk_l2_p95": "large_arm_joint_jerk",
            "eef_state_jerk_l2_p95": "large_eef_jerk",
            "hand_state_jerk_l2_p95": "large_hand_jerk",
            "finger_velocity_p95": "large_finger_velocity",
            "finger_acceleration_p95": "large_finger_acceleration",
            "saturation_ratio": "hand_saturation",
        }
        for metric, code in high_metrics.items():
            if metric not in episodes:
                continue
            mask = robust_high_outlier_mask(episodes[metric])
            if metric == "saturation_ratio":
                mask = pd.to_numeric(episodes[metric], errors="coerce") > 0.10
            for _, row in episodes.loc[mask.fillna(False)].iterrows():
                add_issue(issues, int(row["episode_index"]), str(row.get("source_date", "unknown")), "warning", code, f"{metric} is unusually high", metric, row.get(metric))

    if not timing.empty:
        for _, row in timing.iterrows():
            episode_index = int(row.get("episode_index", -1))
            date = str(row.get("source_date", "unknown"))
            if row.get("timing_status") not in {"ok", "skipped"}:
                add_issue(issues, episode_index, date, "warning", "timing_unavailable", str(row.get("timing_reason", "timing metrics unavailable")))
            max_offset = row.get("max_abs_time_offset")
            if isinstance(max_offset, (int, float)) and math.isfinite(float(max_offset)) and float(max_offset) > 0.12:
                add_issue(issues, episode_index, date, "warning", "large_time_offset", "video/state time offset is large", "max_abs_time_offset", max_offset)
            jitter = row.get("camera_frame_jitter")
            if isinstance(jitter, (int, float)) and math.isfinite(float(jitter)) and float(jitter) > 0.015:
                add_issue(issues, episode_index, date, "warning", "camera_jitter", "camera frame interval jitter is high", "camera_frame_jitter", jitter)
        if "action_state_lag" in timing:
            lag_values = pd.to_numeric(timing["action_state_lag"], errors="coerce")
            if lag_values.notna().sum() >= 8 and float(lag_values.std(skipna=True)) > 4.0:
                for _, row in timing.loc[lag_values.sub(lag_values.median(skipna=True)).abs() > 6.0].iterrows():
                    add_issue(issues, int(row["episode_index"]), str(row.get("source_date", "unknown")), "warning", "unstable_action_state_lag", "episode lag differs strongly from dataset median", "action_state_lag", row.get("action_state_lag"))

    if not videos.empty:
        low_blur_mask = robust_low_outlier_mask(videos["blur_score"]) if "blur_score" in videos else pd.Series(False, index=videos.index)
        low_brightness_mask = robust_low_outlier_mask(videos["brightness_mean"]) if "brightness_mean" in videos else pd.Series(False, index=videos.index)
        high_brightness_mask = robust_high_outlier_mask(videos["brightness_mean"]) if "brightness_mean" in videos else pd.Series(False, index=videos.index)
        high_flow_mask = robust_high_outlier_mask(videos["optical_flow_magnitude"]) if "optical_flow_magnitude" in videos else pd.Series(False, index=videos.index)
        for _, row in videos.iterrows():
            episode_index = int(row.get("episode_index", -1))
            date = str(row.get("source_date", "unknown"))
            video_key = str(row.get("video_key", "video"))
            if row.get("video_status") not in {"ok", "skipped"}:
                add_issue(issues, episode_index, date, "error", "video_error", f"{video_key}: video quality scan failed", "video_status", row.get("video_status"))
            blur = row.get("blur_score")
            if isinstance(blur, (int, float)) and math.isfinite(float(blur)) and float(blur) < 20.0:
                add_issue(issues, episode_index, date, "warning", "low_blur_score", f"{video_key}: image appears blurry", "blur_score", blur)
            elif row.name in low_blur_mask.index and bool(low_blur_mask.loc[row.name]):
                add_issue(issues, episode_index, date, "warning", "relative_low_blur", f"{video_key}: blur score is low relative to the dataset", "blur_score", blur)
            bright = row.get("brightness_mean")
            if isinstance(bright, (int, float)) and math.isfinite(float(bright)) and (float(bright) < 25.0 or float(bright) > 230.0):
                add_issue(issues, episode_index, date, "warning", "bad_brightness", f"{video_key}: brightness is extreme", "brightness_mean", bright)
            elif row.name in low_brightness_mask.index and bool(low_brightness_mask.loc[row.name]):
                add_issue(issues, episode_index, date, "warning", "relative_dark_video", f"{video_key}: brightness is low relative to the dataset", "brightness_mean", bright)
            elif row.name in high_brightness_mask.index and bool(high_brightness_mask.loc[row.name]):
                add_issue(issues, episode_index, date, "warning", "relative_bright_video", f"{video_key}: brightness is high relative to the dataset", "brightness_mean", bright)
            flow = row.get("optical_flow_magnitude")
            if row.name in high_flow_mask.index and bool(high_flow_mask.loc[row.name]):
                add_issue(issues, episode_index, date, "warning", "relative_high_optical_flow", f"{video_key}: optical flow is high relative to the dataset", "optical_flow_magnitude", flow)
            for metric, code, threshold in (
                ("overexposure_ratio", "overexposed_video", 0.35),
                ("underexposure_ratio", "underexposed_video", 0.35),
            ):
                value = row.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > threshold:
                    add_issue(issues, episode_index, date, "warning", code, f"{video_key}: {metric} is high", metric, value)
    return pd.DataFrame(issues)


def apply_quality_scores(episodes: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return episodes
    scored = episodes.copy()
    scored["error_count"] = 0
    scored["warning_count"] = 0
    scored["issue_count"] = 0
    scored["quality_score"] = 100.0
    scored["issue_summary"] = ""
    if issues.empty:
        return scored
    grouped = issues.groupby("episode_index")
    for episode_index, rows in grouped:
        errors = int((rows["severity"] == "error").sum())
        warnings = int((rows["severity"] == "warning").sum())
        score = max(0.0, 100.0 - errors * 35.0 - warnings * 8.0)
        codes = rows["code"].astype(str).tolist()
        mask = scored["episode_index"].astype(int) == int(episode_index)
        scored.loc[mask, "error_count"] = errors
        scored.loc[mask, "warning_count"] = warnings
        scored.loc[mask, "issue_count"] = errors + warnings
        scored.loc[mask, "quality_score"] = score
        scored.loc[mask, "issue_summary"] = ", ".join(codes[:6])
    return scored


def _embedding_matrix(df: pd.DataFrame, column: str) -> tuple[list[int], np.ndarray] | None:
    if df.empty or column not in df:
        return None
    rows: list[np.ndarray] = []
    episode_ids: list[int] = []
    for _, row in df.iterrows():
        value = row.get(column)
        if not isinstance(value, str) or not value:
            continue
        try:
            arr = np.asarray(json.loads(value), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size:
            rows.append(arr)
            episode_ids.append(int(row["episode_index"]))
    if not rows:
        return None
    width = rows[0].shape[0]
    rows = [row for row in rows if row.shape[0] == width]
    return episode_ids[: len(rows)], np.vstack(rows)


def reduce_embeddings(embeddings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    status: dict[str, str] = {}
    outputs: list[pd.DataFrame] = []
    for kind, column in (
        ("video", "video_embedding"),
        ("state_action", "state_action_embedding"),
        ("language", "language_embedding"),
    ):
        matrix_info = _embedding_matrix(embeddings, column)
        if matrix_info is None:
            status[kind] = "skipped:no_embeddings"
            continue
        episode_ids, matrix = matrix_info
        if matrix.shape[0] < 3:
            status[kind] = "skipped:not_enough_points"
            continue
        reducers: dict[str, Any] = {}
        try:
            umap_mod = importlib.import_module("umap")
            reducers["umap"] = umap_mod.UMAP(n_components=2, random_state=42)
        except ModuleNotFoundError:
            status[f"{kind}_umap"] = "skipped:umap_missing"
        try:
            sklearn_tsne = importlib.import_module("sklearn.manifold")
            perplexity = max(2, min(30, matrix.shape[0] - 1))
            reducers["tsne"] = sklearn_tsne.TSNE(n_components=2, random_state=42, perplexity=perplexity, init="random")
        except ModuleNotFoundError:
            status[f"{kind}_tsne"] = "skipped:sklearn_missing"
        for reducer_name, reducer in reducers.items():
            coords = reducer.fit_transform(matrix)
            outputs.append(
                pd.DataFrame(
                    {
                        "episode_index": episode_ids,
                        "embedding_kind": kind,
                        "reducer": reducer_name,
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                    }
                )
            )
            status[f"{kind}_{reducer_name}"] = "ok"
    if outputs:
        return pd.concat(outputs, ignore_index=True), status
    return pd.DataFrame(), status


def plot_embedding_reductions(reduced: pd.DataFrame, episodes: pd.DataFrame, out_dir: Path) -> list[str]:
    if reduced.empty:
        return []
    merged = reduced.merge(episodes[["episode_index", "source_date"]], on="episode_index", how="left")
    paths: list[str] = []
    for (kind, reducer), group in merged.groupby(["embedding_kind", "reducer"]):
        fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        for date, date_group in group.groupby("source_date"):
            axis.scatter(date_group["x"], date_group["y"], s=24, label=str(date), alpha=0.8)
        axis.set_title(f"{kind} {reducer}")
        axis.legend(fontsize=8)
        path = out_dir / f"embedding_{kind}_{reducer}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.name))
    return paths


def make_json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return make_json_safe(float(value))
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    return value


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = df.head(limit).to_dict(orient="records") if limit is not None else df.to_dict(orient="records")
    return [make_json_safe(row) for row in records]


def build_report_payload(
    *,
    dataset_dir: Path,
    output_dir: Path,
    episodes: pd.DataFrame,
    videos: pd.DataFrame,
    timing: pd.DataFrame,
    issues: pd.DataFrame,
    embeddings: pd.DataFrame,
    curves: list[dict[str, Any]],
    run_summary: dict[str, Any],
    embedding_status: dict[str, str],
    plot_files: list[str],
) -> dict[str, Any]:
    merged = episodes.merge(timing, on=["episode_index", "source_date"], how="left", suffixes=("", "_timing")) if not timing.empty else episodes.copy()
    video_summary = videos.groupby("episode_index").mean(numeric_only=True).reset_index() if not videos.empty else pd.DataFrame()
    if not video_summary.empty:
        merged = merged.merge(video_summary, on="episode_index", how="left", suffixes=("", "_video"))
    issue_count = int(len(issues)) if not issues.empty else 0
    bad_episode_count = int((episodes.get("issue_count", pd.Series(dtype=float)).fillna(0) > 0).sum()) if not episodes.empty else 0
    by_date = episodes.groupby("source_date").agg(
        episodes=("episode_index", "count"),
        frames=("parquet_rows", "sum"),
        mean_length=("parquet_rows", "mean"),
        bad_episodes=("issue_count", lambda values: int((pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum())),
        mean_quality_score=("quality_score", "mean"),
    ).reset_index() if not episodes.empty else pd.DataFrame()
    if not by_date.empty:
        by_date["bad_ratio"] = by_date["bad_episodes"] / by_date["episodes"].replace(0, np.nan)
    worst = episodes.sort_values(["quality_score", "issue_count"], ascending=[True, False]).head(30) if not episodes.empty and "quality_score" in episodes else pd.DataFrame()
    issue_counts = issues["code"].value_counts().rename_axis("code").reset_index(name="count") if not issues.empty and "code" in issues else pd.DataFrame()
    return {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": make_json_safe(
            {
                **run_summary,
                "episodes": int(len(episodes)),
                "frames": int(episodes["parquet_rows"].fillna(0).sum()) if "parquet_rows" in episodes else 0,
                "source_dates": int(episodes["source_date"].nunique()) if "source_date" in episodes else 0,
                "issues": issue_count,
                "bad_episodes": bad_episode_count,
                "bad_episode_ratio": float(bad_episode_count / len(episodes)) if len(episodes) else 0.0,
                "embedding_status": embedding_status,
            }
        ),
        "date_summary": dataframe_records(by_date),
        "worst_episodes": dataframe_records(worst),
        "issue_counts": dataframe_records(issue_counts),
        "issues": dataframe_records(issues),
        "episodes": dataframe_records(merged),
        "videos": dataframe_records(videos),
        "embeddings": dataframe_records(embeddings),
        "curves": [make_json_safe(row) for row in curves],
        "plots": plot_files,
    }


def write_html(output_dir: Path, payload: dict[str, Any]) -> None:
    plot_imgs = "\n".join(
        f'<figure><img src="plots/{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in payload.get("plots", [])
    )
    payload_json = json.dumps(make_json_safe(payload), ensure_ascii=False)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mission Quality Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; background: #fbfcfd; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; max-width: 1100px; }}
    .card {{ border: 1px solid #d7dde5; border-radius: 8px; padding: 12px; background: #fff; }}
    .card.alert {{ border-color: #f2a65a; background: #fff7ed; }}
    .card.danger {{ border-color: #e45756; background: #fff1f2; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 26px; font-weight: 650; margin-top: 4px; }}
    figure {{ margin: 18px 0; }} img {{ max-width: 100%; border: 1px solid #d7dde5; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    tr.issue-error td, td.issue-error {{ background: #fff1f2; color: #9f1239; font-weight: 650; }}
    tr.issue-warning td, td.issue-warning {{ background: #fff7ed; color: #9a3412; }}
    td.score-bad {{ background: #fff1f2; color: #9f1239; font-weight: 700; }}
    td.score-warn {{ background: #fff7ed; color: #9a3412; font-weight: 650; }}
    td.count-bad {{ color: #9f1239; font-weight: 700; }}
    select {{ padding: 6px 8px; min-width: 320px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 8px 0 12px; }}
    .metric-pill {{ border: 1px solid #d7dde5; border-radius: 6px; padding: 6px 10px; background: #fff; font-size: 13px; }}
    .muted {{ color: #667085; }}
    .legend {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin: 4px 0 10px; font-size: 13px; color: #46515f; }}
    .swatch {{ display: inline-block; width: 28px; height: 0; border-top: 3px solid currentColor; vertical-align: middle; margin-right: 5px; }}
    .swatch.dashed {{ border-top-style: dashed; }}
    .state-action-columns {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; align-items: start; }}
    .dimension-column {{ display: grid; gap: 10px; }}
    .dimension-column h3 {{ margin-bottom: 2px; }}
    .dimension-card {{ border: 1px solid #d7dde5; border-radius: 6px; padding: 8px; background: #fff; }}
    .dimension-card canvas {{ width: 100%; max-width: none; height: 150px; border: 0; border-radius: 0; }}
    @media (max-width: 900px) {{ .state-action-columns {{ grid-template-columns: 1fr; }} }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr); gap: 18px; align-items: start; }}
    canvas {{ width: 100%; max-width: 860px; height: 260px; border: 1px solid #e5e7eb; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Mission Quality Report</h1>
  <p>Dataset: <code>{html.escape(str(payload.get('dataset_dir', '')))}</code></p>
  <div id="summary" class="summary"></div>
  <h2>Bad Data First</h2>
  <div class="grid">
    <div><h3>Worst Episodes</h3><div id="worst-episodes"></div></div>
    <div><h3>Issue Codes</h3><div id="issue-codes"></div></div>
  </div>
  <h2>Overview Plots</h2>
  {plot_imgs}
  <h2>Date Summary</h2>
  <div id="date-summary"></div>
  <h2>Episode Explorer</h2>
  <label for="episode-select">Episode</label>
  <select id="episode-select"></select>
  <div class="grid">
    <div><h3>Metrics</h3><div id="episode-metrics"></div></div>
    <div><h3>Curves</h3><canvas id="curve-canvas" width="860" height="300"></canvas></div>
  </div>
  <h2>State / Action Dimension Comparison</h2>
  <div id="state-action-compare">
    <div class="controls">
      <span id="state-action-summary" class="metric-pill"></span>
    </div>
    <div class="legend">
      <span style="color:#2563eb"><span class="swatch"></span>state</span>
      <span style="color:#dc2626"><span class="swatch dashed"></span>action</span>
    </div>
    <div class="state-action-columns">
      <div class="dimension-column">
        <h3>EEF</h3>
        <div id="state-action-eef"></div>
      </div>
      <div class="dimension-column">
        <h3>Hand</h3>
        <div id="state-action-hand"></div>
      </div>
    </div>
  </div>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const summary = payload.summary || {{}};
    const summaryEl = document.getElementById('summary');
    for (const key of ['episodes','frames','source_dates','bad_episodes','bad_episode_ratio','issues','computed','reused']) {{
      const div = document.createElement('div');
      const value = key === 'bad_episode_ratio' ? ((summary[key] ?? 0) * 100).toFixed(1) + '%' : (summary[key] ?? 0);
      const numeric = Number(summary[key] ?? 0);
      div.className = 'card' + ((key === 'bad_episodes' || key === 'issues') && numeric > 0 ? ' alert' : '') + (key === 'bad_episode_ratio' && numeric >= 0.25 ? ' danger' : '');
      div.innerHTML = `<div class="label">${{key}}</div><div class="value">${{value}}</div>`;
      summaryEl.appendChild(div);
    }}
    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function cellClass(c, v) {{
      if (c === 'severity' && v === 'error') return 'issue-error';
      if (c === 'severity' && v === 'warning') return 'issue-warning';
      if (c === 'quality_score' && Number(v) < 60) return 'score-bad';
      if (c === 'quality_score' && Number(v) < 85) return 'score-warn';
      if ((c === 'issue_count' || c === 'error_count' || c === 'warning_count') && Number(v) > 0) return 'count-bad';
      return '';
    }}
    function table(rows, cols) {{
      if (!rows || rows.length === 0) return '<p>No rows.</p>';
      cols = cols || Object.keys(rows[0]);
      return '<table><thead><tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr class="' + (r.severity === 'error' ? 'issue-error' : (r.severity === 'warning' ? 'issue-warning' : '')) + '">' +
          cols.map(c => `<td class="${{cellClass(c, r[c])}}">${{esc(r[c])}}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
    }}
    function formatNumber(value) {{
      const num = Number(value);
      if (!Number.isFinite(num)) return '';
      if (Math.abs(num) >= 1000 || (Math.abs(num) > 0 && Math.abs(num) < 0.001)) return num.toExponential(3);
      return num.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
    }}
    document.getElementById('worst-episodes').innerHTML = table(payload.worst_episodes || [], ['episode_index','source_date','quality_score','issue_count','error_count','warning_count','issue_summary']);
    document.getElementById('issue-codes').innerHTML = table(payload.issue_counts || [], ['code','count']);
    document.getElementById('date-summary').innerHTML = table(payload.date_summary, ['source_date','episodes','bad_episodes','bad_ratio','mean_quality_score','frames','mean_length']);
    const episodes = (payload.episodes || []).slice().sort((a, b) =>
      (Number(a.quality_score ?? 100) - Number(b.quality_score ?? 100)) ||
      (Number(b.issue_count ?? 0) - Number(a.issue_count ?? 0)) ||
      (Number(a.episode_index ?? 0) - Number(b.episode_index ?? 0))
    );
    const issuesByEpisode = new Map();
    (payload.issues || []).forEach(issue => {{
      const key = Number(issue.episode_index);
      if (!issuesByEpisode.has(key)) issuesByEpisode.set(key, []);
      issuesByEpisode.get(key).push(issue);
    }});
    const curves = new Map((payload.curves || []).map(c => [Number(c.episode_index), c]));
    const select = document.getElementById('episode-select');
    episodes.forEach(ep => {{
      const opt = document.createElement('option'); opt.value = ep.episode_index;
      const prefix = Number(ep.issue_count ?? 0) > 0 ? '[BAD] ' : '';
      opt.textContent = `${{prefix}}episode ${{ep.episode_index}} | score=${{ep.quality_score ?? 100}} | issues=${{ep.issue_count ?? 0}} | ${{ep.source_date}}`;
      select.appendChild(opt);
    }});
    function drawCurve(curve) {{
      const canvas = document.getElementById('curve-canvas'); const ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height); ctx.font = '13px system-ui';
      const series = [['action_l2','#f58518'], ['state_l2','#4c78a8'], ['hand_state_mean','#54a24b']];
      ctx.fillText('action_l2 / state_l2 / hand_state_mean', 12, 18);
      series.forEach(([name, color]) => {{
        const arr = (curve && curve[name]) || [];
        if (!arr.length) return;
        const finite = arr.filter(v => Number.isFinite(v)); if (!finite.length) return;
        const min = Math.min(...finite), max = Math.max(...finite), span = Math.max(1e-9, max-min);
        ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        arr.forEach((v, i) => {{
          const x = 30 + i * (canvas.width - 50) / Math.max(1, arr.length - 1);
          const y = canvas.height - 24 - ((v - min) / span) * (canvas.height - 58);
          if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }});
        ctx.stroke();
      }});
    }}
    function finiteNumbers(arr) {{
      return (arr || []).map(Number).filter(v => Number.isFinite(v));
    }}
    function drawStateActionPanel(canvas, label, stateArr, actionArr, rmse) {{
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = '13px system-ui';
      const finite = finiteNumbers(stateArr).concat(finiteNumbers(actionArr));
      if (!finite.length) {{
        ctx.fillStyle = '#667085';
        ctx.fillText(`${{label}} | no finite values`, 12, 22);
        return;
      }}
      const min = Math.min(...finite), max = Math.max(...finite), span = Math.max(1e-9, max - min);
      const left = 42, right = 12, top = 30, bottom = 24;
      ctx.strokeStyle = '#d7dde5'; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, canvas.height - bottom); ctx.lineTo(canvas.width - right, canvas.height - bottom); ctx.stroke();
      ctx.fillStyle = '#1f2933';
      ctx.fillText(`${{label}} | RMSE=${{formatNumber(rmse)}}`, 12, 19);
      ctx.fillStyle = '#667085';
      ctx.font = '11px system-ui';
      ctx.fillText(formatNumber(max), 6, top + 4);
      ctx.fillText(formatNumber(min), 6, canvas.height - bottom);
      function drawLine(arr, color, dash) {{
        if (!arr || !arr.length) return;
        ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.setLineDash(dash);
        let started = false;
        arr.forEach((raw, i) => {{
          const v = Number(raw);
          if (!Number.isFinite(v)) return;
          const x = left + i * (canvas.width - left - right) / Math.max(1, arr.length - 1);
          const y = canvas.height - bottom - ((v - min) / span) * (canvas.height - top - bottom);
          if (!started) {{ ctx.moveTo(x, y); started = true; }} else {{ ctx.lineTo(x, y); }}
        }});
        ctx.stroke();
        ctx.setLineDash([]);
      }}
      drawLine(stateArr, '#2563eb', []);
      drawLine(actionArr, '#dc2626', [8, 6]);
    }}
    function appendStateActionPanel(parent, label, stateArr, actionArr, rmse) {{
      const card = document.createElement('div');
      card.className = 'dimension-card';
      const canvas = document.createElement('canvas');
      canvas.width = 560;
      canvas.height = 180;
      card.appendChild(canvas);
      parent.appendChild(card);
      drawStateActionPanel(canvas, label, stateArr, actionArr, rmse);
    }}
    function showStateActionComparison(curve) {{
      const labels = (curve && curve.state_action_dim_labels) || [];
      const rmses = ((curve && curve.state_action_rmse_by_dim) || []).map(Number);
      const stateSeries = (curve && curve.state_action_state) || [];
      const actionSeries = (curve && curve.state_action_action) || [];
      const eefEl = document.getElementById('state-action-eef');
      const handEl = document.getElementById('state-action-hand');
      eefEl.innerHTML = '';
      handEl.innerHTML = '';
      const summary = document.getElementById('state-action-summary');
      summary.textContent = labels.length
        ? `overall RMSE=${{formatNumber(curve.state_action_rmse)}} | max dim=${{curve.state_action_rmse_max_dim || ''}}`
        : 'No comparable state/action dimensions';
      labels.forEach((label, i) => {{
        const parent = label.startsWith('eef_') ? eefEl : handEl;
        appendStateActionPanel(parent, label, stateSeries[i] || [], actionSeries[i] || [], rmses[i]);
      }});
      if (!eefEl.children.length) eefEl.innerHTML = '<p class="muted">No EEF dimensions.</p>';
      if (!handEl.children.length) handEl.innerHTML = '<p class="muted">No hand dimensions.</p>';
    }}
    function showEpisode() {{
      const id = Number(select.value);
      const ep = episodes.find(e => Number(e.episode_index) === id) || {{}};
      const episodeIssues = issuesByEpisode.get(id) || [];
      const cols = ['episode_index','source_date','quality_score','issue_count','issue_summary','status','parquet_rows','expected_frames','frame_count_match','video_frame_count_match','state_action_rmse','state_action_rmse_dim_mean','state_action_rmse_dim_p95','state_action_rmse_dim_max','state_action_rmse_max_dim','timing_status','mean_dt_video_state','action_state_lag','camera_frame_jitter','finger_velocity_p95','arm_joint_jerk_l2_p95','blur_score','brightness_mean'];
      document.getElementById('episode-metrics').innerHTML = '<h4>Issues</h4>' + table(episodeIssues, ['severity','code','message','metric','value']) + '<h4>Metrics</h4>' + table([ep], cols) + '<pre>' + JSON.stringify(ep, null, 2) + '</pre>';
      const curve = curves.get(id);
      drawCurve(curve);
      showStateActionComparison(curve);
    }}
    select.addEventListener('change', showEpisode);
    if (episodes.length) showEpisode();
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    cache_dir = output_dir / "cache"
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    logs_dir = output_dir / "logs"
    for path in (cache_dir, data_dir, plots_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    analysis_signature = {
        "skip_video": bool(args.skip_video),
        "skip_timing": bool(args.skip_timing),
        "video_samples": max(1, int(args.video_samples)),
        "embedding_backend": args.embedding_backend or None,
        "object_detector_backend": args.object_detector_backend or None,
    }
    info, records = discover_episodes(dataset_dir, analysis_signature)
    fps = float(info.get("fps", 10))
    processed_path = cache_dir / "processed_episodes.jsonl"
    previous = previous_processed_map(processed_path)
    reusable = {
        record.episode_index
        for record in records
        if not args.force and previous.get(record.episode_index, {}).get("fingerprint") == record.fingerprint
    }

    episode_cache = filter_cached(load_table(cache_dir / "episode_metrics.parquet"), reusable)
    video_cache = filter_cached(load_table(cache_dir / "video_metrics.parquet"), reusable)
    timing_cache = filter_cached(load_table(cache_dir / "timing_metrics.parquet"), reusable)
    embedding_cache = filter_cached(load_table(cache_dir / "embedding_metrics.parquet"), reusable)
    curves_cache = [row for row in read_jsonl(cache_dir / "episode_curves.jsonl") if int(row.get("episode_index", -1)) in reusable]

    to_compute = [record for record in records if record.episode_index not in reusable]
    object_detector = load_backend(args.object_detector_backend)
    embedding_backend = load_backend(args.embedding_backend)

    computed_episode_rows: list[dict[str, Any]] = []
    computed_video_rows: list[dict[str, Any]] = []
    computed_timing_rows: list[dict[str, Any]] = []
    computed_embedding_rows: list[dict[str, Any]] = []
    computed_curves: list[dict[str, Any]] = []
    processed_rows: list[dict[str, Any]] = []

    worker_count = max(1, int(args.max_workers))
    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = [
        executor.submit(
            process_episode,
            record,
            raw_root=raw_root,
            skip_video=bool(args.skip_video),
            skip_timing=bool(args.skip_timing),
            object_detector=object_detector,
            embedding_backend=embedding_backend,
            fps=fps,
            video_samples=max(1, int(args.video_samples)),
        )
        for record in to_compute
    ]
    try:
        for future in as_completed(futures):
            episode_row, video_rows, timing_row, embedding_row, curves, processed = future.result()
            computed_episode_rows.append(episode_row)
            computed_video_rows.extend(video_rows)
            computed_timing_rows.append(timing_row)
            computed_embedding_rows.append(embedding_row)
            computed_curves.append(curves)
            processed_rows.append(processed)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    episode_df = pd.concat([episode_cache, pd.DataFrame(computed_episode_rows)], ignore_index=True) if not episode_cache.empty or computed_episode_rows else pd.DataFrame()
    video_df = pd.concat([video_cache, pd.DataFrame(computed_video_rows)], ignore_index=True) if not video_cache.empty or computed_video_rows else pd.DataFrame()
    timing_df = pd.concat([timing_cache, pd.DataFrame(computed_timing_rows)], ignore_index=True) if not timing_cache.empty or computed_timing_rows else pd.DataFrame()
    embedding_df = pd.concat([embedding_cache, pd.DataFrame(computed_embedding_rows)], ignore_index=True) if not embedding_cache.empty or computed_embedding_rows else pd.DataFrame()
    curves = curves_cache + computed_curves

    for df in (episode_df, video_df, timing_df, embedding_df):
        if not df.empty and "episode_index" in df:
            df.sort_values(["episode_index"], inplace=True)
            df.reset_index(drop=True, inplace=True)

    issue_df = build_issue_rows(episode_df, video_df, timing_df)
    episode_df = apply_quality_scores(episode_df, issue_df)

    save_table(episode_df, cache_dir / "episode_metrics.parquet")
    save_table(video_df, cache_dir / "video_metrics.parquet")
    save_table(timing_df, cache_dir / "timing_metrics.parquet")
    save_table(embedding_df, cache_dir / "embedding_metrics.parquet")
    save_table(issue_df, cache_dir / "issue_rows.parquet")
    if not episode_df.empty:
        episode_df.sort_values(["quality_score", "issue_count", "episode_index"], ascending=[True, False, True]).head(100).to_csv(
            cache_dir / "worst_episodes.csv",
            index=False,
        )
    if not issue_df.empty:
        issue_df.sort_values(["source_date", "episode_index", "severity", "code"]).to_csv(cache_dir / "issue_rows.csv", index=False)
    write_jsonl(cache_dir / "episode_curves.jsonl", sorted(curves, key=lambda row: int(row.get("episode_index", -1))))

    processed_latest = [
        {
            "episode_index": record.episode_index,
            "fingerprint": record.fingerprint,
            "processed_at_utc": previous.get(record.episode_index, {}).get("processed_at_utc"),
            "status": previous.get(record.episode_index, {}).get("status", "reused"),
        }
        for record in records
        if record.episode_index in reusable
    ] + processed_rows
    write_jsonl(processed_path, sorted(processed_latest, key=lambda row: int(row["episode_index"])))

    plot_files: list[str] = []
    plot_bad_episode_summary(episode_df, plots_dir / "bad_episode_summary.png")
    plot_length_distribution(episode_df, plots_dir / "length_by_date.png")
    plot_metric_boxplots(episode_df, plots_dir / "metric_distributions_by_date.png")
    plot_timing_quality(timing_df, plots_dir / "timing_quality.png")
    plot_hand_quality(episode_df, plots_dir / "hand_quality.png")
    plot_video_quality(video_df, plots_dir / "video_quality.png")
    plot_cross_modal_relations(episode_df, video_df, timing_df, plots_dir / "cross_modal_relations.png")
    plot_issue_codes(issue_df, plots_dir / "issue_codes.png")
    plot_issue_heatmap(issue_df, plots_dir / "issue_heatmap_by_date.png")
    plot_issue_timeline(episode_df, issue_df, plots_dir / "issue_timeline.png")
    for name in (
        "bad_episode_summary.png",
        "issue_timeline.png",
        "issue_heatmap_by_date.png",
        "length_by_date.png",
        "metric_distributions_by_date.png",
        "timing_quality.png",
        "hand_quality.png",
        "video_quality.png",
        "cross_modal_relations.png",
        "issue_codes.png",
    ):
        if (plots_dir / name).exists():
            plot_files.append(name)

    reduced_embeddings, embedding_status = reduce_embeddings(embedding_df)
    if embedding_backend is None:
        embedding_status["backend"] = "skipped:not_configured"
    if object_detector is None:
        embedding_status["object_visibility"] = "skipped:not_configured"
    plot_files.extend(plot_embedding_reductions(reduced_embeddings, episode_df, plots_dir))

    run_summary = {
        "computed": len(to_compute),
        "reused": len(reusable),
        "skipped": 0,
        "force": bool(args.force),
        "skip_video": bool(args.skip_video),
        "skip_timing": bool(args.skip_timing),
        "video_samples": max(1, int(args.video_samples)),
        "analysis_signature": analysis_signature,
        "analyzer_version": QUALITY_ANALYZER_VERSION,
    }
    payload = build_report_payload(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        episodes=episode_df,
        videos=video_df,
        timing=timing_df,
        issues=issue_df,
        embeddings=reduced_embeddings,
        curves=curves,
        run_summary=run_summary,
        embedding_status=embedding_status,
        plot_files=plot_files,
    )
    write_json(data_dir / "report_payload.json", payload)
    write_html(output_dir, payload)
    log_path = logs_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(log_path, payload["summary"])
    print(json.dumps({**payload["summary"], "report": str(output_dir / "index.html")}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
