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

QUALITY_ANALYZER_VERSION = "quality.v1"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMPACT_DATE_RE = re.compile(r"(20\d{6})")

ACTION_EEF_SLICE = slice(0, 9)
ACTION_HAND_SLICE = slice(9, 19)
STATE_ARM_SLICE = slice(0, 7)
STATE_EEF_SLICE = slice(7, 16)
STATE_HAND_SLICE = slice(16, 26)
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
        for _, row in videos.iterrows():
            episode_index = int(row.get("episode_index", -1))
            date = str(row.get("source_date", "unknown"))
            video_key = str(row.get("video_key", "video"))
            if row.get("video_status") not in {"ok", "skipped"}:
                add_issue(issues, episode_index, date, "error", "video_error", f"{video_key}: video quality scan failed", "video_status", row.get("video_status"))
            blur = row.get("blur_score")
            if isinstance(blur, (int, float)) and math.isfinite(float(blur)) and float(blur) < 20.0:
                add_issue(issues, episode_index, date, "warning", "low_blur_score", f"{video_key}: image appears blurry", "blur_score", blur)
            bright = row.get("brightness_mean")
            if isinstance(bright, (int, float)) and math.isfinite(float(bright)) and (float(bright) < 25.0 or float(bright) > 230.0):
                add_issue(issues, episode_index, date, "warning", "bad_brightness", f"{video_key}: brightness is extreme", "brightness_mean", bright)
            for metric, code, threshold in (
                ("overexposure_ratio", "overexposed_video", 0.10),
                ("underexposure_ratio", "underexposed_video", 0.10),
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
    issue_count = 0
    if "status" in episodes:
        issue_count += int((episodes["status"] != "ok").sum())
    if "frame_count_match" in episodes:
        issue_count += int((episodes["frame_count_match"] == False).sum())  # noqa: E712
    if "video_frame_count_match" in episodes:
        issue_count += int((episodes["video_frame_count_match"] == False).sum())  # noqa: E712
    by_date = episodes.groupby("source_date").agg(
        episodes=("episode_index", "count"),
        frames=("parquet_rows", "sum"),
        mean_length=("parquet_rows", "mean"),
    ).reset_index() if not episodes.empty else pd.DataFrame()
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
                "embedding_status": embedding_status,
            }
        ),
        "date_summary": dataframe_records(by_date),
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
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; max-width: 1100px; }}
    .card {{ border: 1px solid #d7dde5; border-radius: 8px; padding: 12px; background: #fff; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 26px; font-weight: 650; margin-top: 4px; }}
    figure {{ margin: 18px 0; }} img {{ max-width: 100%; border: 1px solid #d7dde5; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    select {{ padding: 6px 8px; min-width: 320px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; border-radius: 6px; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr); gap: 18px; align-items: start; }}
    canvas {{ width: 100%; max-width: 860px; height: 260px; border: 1px solid #e5e7eb; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Mission Quality Report</h1>
  <p>Dataset: <code>{html.escape(str(payload.get('dataset_dir', '')))}</code></p>
  <div id="summary" class="summary"></div>
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
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const summary = payload.summary || {{}};
    const summaryEl = document.getElementById('summary');
    for (const key of ['episodes','frames','source_dates','issues','computed','reused','skipped']) {{
      const div = document.createElement('div'); div.className = 'card';
      div.innerHTML = `<div class="label">${{key}}</div><div class="value">${{summary[key] ?? 0}}</div>`;
      summaryEl.appendChild(div);
    }}
    function table(rows, cols) {{
      if (!rows || rows.length === 0) return '<p>No rows.</p>';
      cols = cols || Object.keys(rows[0]);
      return '<table><thead><tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + cols.map(c => `<td>${{r[c] ?? ''}}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
    }}
    document.getElementById('date-summary').innerHTML = table(payload.date_summary, ['source_date','episodes','frames','mean_length']);
    const episodes = payload.episodes || [];
    const curves = new Map((payload.curves || []).map(c => [Number(c.episode_index), c]));
    const select = document.getElementById('episode-select');
    episodes.forEach(ep => {{
      const opt = document.createElement('option'); opt.value = ep.episode_index;
      opt.textContent = `episode ${{ep.episode_index}} | ${{ep.source_date}} | frames=${{ep.parquet_rows ?? ''}}`;
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
    function showEpisode() {{
      const id = Number(select.value);
      const ep = episodes.find(e => Number(e.episode_index) === id) || {{}};
      const cols = ['episode_index','source_date','status','parquet_rows','expected_frames','frame_count_match','video_frame_count_match','timing_status','mean_dt_video_state','action_state_lag','camera_frame_jitter','finger_velocity_p95','arm_joint_jerk_l2_p95','blur_score','brightness_mean'];
      document.getElementById('episode-metrics').innerHTML = table([ep], cols) + '<pre>' + JSON.stringify(ep, null, 2) + '</pre>';
      drawCurve(curves.get(id));
    }}
    select.addEventListener('change', showEpisode); if (episodes.length) showEpisode();
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

    save_table(episode_df, cache_dir / "episode_metrics.parquet")
    save_table(video_df, cache_dir / "video_metrics.parquet")
    save_table(timing_df, cache_dir / "timing_metrics.parquet")
    save_table(embedding_df, cache_dir / "embedding_metrics.parquet")
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
    plot_length_distribution(episode_df, plots_dir / "length_by_date.png")
    plot_metric_boxplots(episode_df, plots_dir / "metric_distributions_by_date.png")
    plot_video_quality(video_df, plots_dir / "video_quality.png")
    for name in ("length_by_date.png", "metric_distributions_by_date.png", "video_quality.png"):
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
