#!/usr/bin/env python3
"""Analyze smooth LeRobot data quality across all source dates.

The report is intentionally episode-centric: one CSV row per episode, one CSV
row per video stream, date-level aggregates, issue rows, and overview plots with
stable source-date colors. It complements the narrower per-date smooth quality
script by making distribution shifts between capture dates easy to inspect.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cv2 = None

from plot_config_relative_action_curves import groot_relative_groups, l2, rot6d_to_matrix, rot6d_angle_deg


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_DATASET_DIR = MISSION_DIR / "smooth"
DEFAULT_OUTPUT_DIR = MISSION_DIR / "smooth_quality" / "all_dates"
DEFAULT_RELATIVE_STATS = MISSION_DIR / "prepared_smooth" / "meta" / "relative_stats.json"

ACTION_DIM = 19
STATE_DIM = 26
ACTION_EEF_SLICE = slice(0, 9)
ACTION_HAND_SLICE = slice(9, 19)
STATE_ARM_SLICE = slice(0, 7)
STATE_EEF_SLICE = slice(7, 16)
STATE_HAND_SLICE = slice(16, 26)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMPACT_DATE_RE = re.compile(r"(20\d{6})")
HORIZONS = (1, 2, 4, 8, 16, 32)
DATE_COLORS = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#59a14f",
)
ISSUE_COLUMNS = [
    "severity",
    "scope",
    "code",
    "message",
    "episode_index",
    "source_date",
    "video_key",
    "path",
    "count",
    "expected",
    "actual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relative-stats", type=Path, default=DEFAULT_RELATIVE_STATS)
    parser.add_argument("--max-horizon", type=int, default=32)
    parser.add_argument("--video-samples", type=int, default=5)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.025)
    parser.add_argument("--video-duration-tolerance", type=float, default=0.35)
    parser.add_argument("--min-episode-frames", type=int, default=10)
    parser.add_argument("--skip-video-scan", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def source_date(row: dict[str, Any]) -> str:
    meta = metadata(row)
    candidates = [
        meta.get("raw_episode_id"),
        meta.get("raw_episode_dir"),
        meta.get("source_dataset"),
        meta.get("source_dataset_name"),
        meta.get("data_path"),
        row.get("created_at_utc"),
        meta.get("trimmed_at_utc"),
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


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    bad = [idx for idx, value in enumerate(values) if value.shape[0] != width]
    if bad:
        raise ValueError(f"Column {column_name!r} has inconsistent vector widths; first bad row={bad[0]}")
    return np.vstack(values)


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"n": 0, "mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def add_prefixed(row: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    for key, value in finite_stats(values).items():
        row[f"{prefix}_{key}"] = value


def rot6d_orthogonality(rot6d: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    if values.size == 0:
        return {
            "rot6d_col0_norm_abs_error_p99": math.nan,
            "rot6d_col1_norm_abs_error_p99": math.nan,
            "rot6d_col01_abs_dot_p99": math.nan,
        }
    col0 = values[:, :3]
    col1 = values[:, 3:6]
    stats: dict[str, float | int] = {}
    for name, data in (
        ("rot6d_col0_norm_abs_error", np.abs(np.linalg.norm(col0, axis=1) - 1.0)),
        ("rot6d_col1_norm_abs_error", np.abs(np.linalg.norm(col1, axis=1) - 1.0)),
        ("rot6d_col01_abs_dot", np.abs(np.sum(col0 * col1, axis=1))),
    ):
        for key, value in finite_stats(data).items():
            stats[f"{name}_{key}"] = value
    return stats


def rot6d_step_angle_deg(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    if values.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    matrices = rot6d_to_matrix(values)
    relative = np.einsum("nij,njk->nik", np.transpose(matrices[:-1], (0, 2, 1)), matrices[1:])
    return Rotation.from_matrix(relative).magnitude() * 180.0 / np.pi


def path_metrics(prefix: str, xyz: np.ndarray, fps: float, row: dict[str, Any]) -> None:
    if xyz.shape[0] < 2:
        add_prefixed(row, f"{prefix}_step_xyz_l2_mm", np.asarray([]))
        return
    step = np.diff(xyz[:, :3], axis=0)
    step_mm = l2(step) * 1000.0
    add_prefixed(row, f"{prefix}_step_xyz_l2_mm", step_mm)
    add_prefixed(row, f"{prefix}_velocity_xyz_l2_mm_s", step_mm * fps)
    if step_mm.size:
        row[f"{prefix}_path_length_xyz_mm"] = float(np.sum(step_mm))
        row[f"{prefix}_displacement_xyz_mm"] = float(np.linalg.norm(xyz[-1, :3] - xyz[0, :3]) * 1000.0)
        row[f"{prefix}_straightness"] = float(row[f"{prefix}_displacement_xyz_mm"] / max(row[f"{prefix}_path_length_xyz_mm"], 1e-8))
        row[f"{prefix}_pause_ratio_step_lt_1mm"] = float(np.mean(step_mm < 1.0))
    if xyz.shape[0] >= 3:
        accel = l2(np.diff(xyz[:, :3], n=2, axis=0)) * 1000.0 * fps * fps
        add_prefixed(row, f"{prefix}_acceleration_xyz_l2_mm_s2", accel)
    if xyz.shape[0] >= 4:
        jerk = l2(np.diff(xyz[:, :3], n=3, axis=0)) * 1000.0 * fps * fps * fps
        add_prefixed(row, f"{prefix}_jerk_xyz_l2_mm_s3", jerk)


def load_relative_bounds(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    raw = read_json(path)
    try:
        low = np.concatenate(
            [
                np.asarray(raw["eef_9d"]["q01"], dtype=np.float64),
                np.asarray(raw["hand_joint_target"]["q01"], dtype=np.float64),
                np.asarray(raw["arm_joint_target"]["q01"], dtype=np.float64),
            ],
            axis=1,
        )
        high = np.concatenate(
            [
                np.asarray(raw["eef_9d"]["q99"], dtype=np.float64),
                np.asarray(raw["hand_joint_target"]["q99"], dtype=np.float64),
                np.asarray(raw["arm_joint_target"]["q99"], dtype=np.float64),
            ],
            axis=1,
        )
    except KeyError:
        return None
    return low, high


def concat_relative(groups: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([groups["eef_9d"], groups["hand_joint_target"], groups["arm_joint_target"]], axis=1)


def normalized_q01_q99(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return 2.0 * (values - low) / np.maximum(high - low, 1e-8) - 1.0


def relative_metrics(
    *,
    row: dict[str, Any],
    action: np.ndarray,
    state: np.ndarray,
    max_horizon: int,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    action_eef = action[:, ACTION_EEF_SLICE]
    action_hand = action[:, ACTION_HAND_SLICE]
    state_arm = state[:, STATE_ARM_SLICE]
    state_eef = state[:, STATE_EEF_SLICE]
    state_hand = state[:, STATE_HAND_SLICE]
    for horizon in [h for h in HORIZONS if h <= max_horizon]:
        delta = horizon - 1
        groups = groot_relative_groups(action_eef, action_hand, state_arm, state_eef, state_hand, delta)
        if groups["eef_9d"].shape[0] == 0:
            continue
        add_prefixed(row, f"relative_h{horizon}_eef_xyz_l2_mm", l2(groups["eef_9d"][:, :3]) * 1000.0)
        add_prefixed(row, f"relative_h{horizon}_eef_rot_angle_deg", rot6d_angle_deg(groups["eef_9d"][:, 3:9]))
        add_prefixed(row, f"relative_h{horizon}_hand_l2", l2(groups["hand_joint_target"]))
        add_prefixed(row, f"relative_h{horizon}_arm_l2", l2(groups["arm_joint_target"]))
        relative = concat_relative(groups)
        add_prefixed(row, f"relative_h{horizon}_all_l2", l2(relative))
        if bounds is None:
            continue
        low, high = bounds
        if horizon > low.shape[0]:
            continue
        normalized = normalized_q01_q99(relative, low[horizon - 1], high[horizon - 1])
        clipped = np.abs(normalized) > 1.0
        row[f"relative_h{horizon}_q01q99_clip_count"] = int(np.count_nonzero(clipped))
        row[f"relative_h{horizon}_q01q99_clip_ratio"] = float(np.mean(clipped)) if clipped.size else 0.0
        group_slices = {
            "eef": slice(0, 9),
            "hand": slice(9, 19),
            "arm": slice(19, 26),
        }
        for group, slc in group_slices.items():
            group_mask = clipped[:, slc]
            row[f"relative_h{horizon}_{group}_q01q99_clip_ratio"] = float(np.mean(group_mask)) if group_mask.size else 0.0


def parse_frame_rate(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            return None if denominator_float == 0.0 else float(numerator) / denominator_float
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def ffprobe_video(ffprobe: str, video_path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams = json.loads(result.stdout.decode("utf-8")).get("streams") or []
    return streams[0] if streams else {}


def sample_video_quality(video_path: Path, sample_count: int) -> dict[str, float | int]:
    if cv2 is None or sample_count <= 0 or not video_path.exists():
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"sample_opened": 0}
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        return {"sample_opened": 1, "sampled_frames": 0}
    indices = set(np.linspace(0, max(0, frame_count - 1), num=min(sample_count, frame_count), dtype=int).tolist())
    blur_values: list[float] = []
    brightness_values: list[float] = []
    contrast_values: list[float] = []
    frame_diffs: list[float] = []
    previous_gray: np.ndarray | None = None
    current_index = 0
    while current_index < frame_count and len(blur_values) < len(indices):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if current_index not in indices:
            current_index += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_values.append(float(np.mean(gray)))
        contrast_values.append(float(np.std(gray)))
        if previous_gray is not None and previous_gray.shape == gray.shape:
            frame_diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32)))))
        previous_gray = gray
        current_index += 1
    capture.release()
    result: dict[str, float | int] = {"sample_opened": 1, "sampled_frames": len(blur_values)}
    for prefix, values in (
        ("laplacian_var", blur_values),
        ("brightness", brightness_values),
        ("contrast", contrast_values),
        ("sample_frame_absdiff", frame_diffs),
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.size:
            result[f"{prefix}_mean"] = float(np.mean(array))
            result[f"{prefix}_min"] = float(np.min(array))
            result[f"{prefix}_max"] = float(np.max(array))
    return result


def add_issue(issues: list[dict[str, Any]], severity: str, scope: str, code: str, message: str, **extra: Any) -> None:
    row = {"severity": severity, "scope": scope, "code": code, "message": message}
    row.update(extra)
    issues.append(row)


def analyze_episode(
    *,
    dataset_dir: Path,
    info: dict[str, Any],
    episode_row: dict[str, Any],
    max_horizon: int,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    timestamp_tolerance: float,
    min_episode_frames: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_index = int(episode_row["episode_index"])
    date = source_date(episode_row)
    data_path = data_path_for_episode(dataset_dir, info, episode_row)
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "source_date": date,
        "declared_length": int(episode_row.get("length", 0)),
        "data_path": str(data_path.relative_to(dataset_dir) if data_path.is_relative_to(dataset_dir) else data_path),
        "status": "ok",
    }
    if row["declared_length"] < min_episode_frames:
        add_issue(issues, "error", "episode", "short_episode", "Episode is shorter than the minimum usable length", episode_index=episode_index, source_date=date, frames=row["declared_length"])
    if not data_path.exists():
        row["status"] = "error"
        add_issue(issues, "error", "parquet", "missing_parquet", "Parquet file is missing", episode_index=episode_index, source_date=date, path=str(data_path))
        return row
    try:
        df = pd.read_parquet(data_path)
    except Exception as exc:
        row["status"] = "error"
        row["read_error"] = str(exc)
        add_issue(issues, "error", "parquet", "read_failed", str(exc), episode_index=episode_index, source_date=date)
        return row

    row["frames"] = int(len(df))
    fps = float(info.get("fps", 10) or 10)
    row["duration_s"] = float(len(df) / fps) if fps > 0 else math.nan
    if len(df) != row["declared_length"]:
        add_issue(issues, "error", "parquet", "length_mismatch", "Parquet rows differ from episodes.jsonl length", episode_index=episode_index, source_date=date, declared=row["declared_length"], actual=len(df))

    missing_columns = [column for column in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "next.done") if column not in df.columns]
    row["missing_columns"] = ",".join(missing_columns)
    for column in missing_columns:
        add_issue(issues, "error", "parquet", "missing_column", f"Required column {column!r} is missing", episode_index=episode_index, source_date=date)
    if "action" not in df.columns or "observation.state" not in df.columns:
        row["status"] = "error"
        return row

    try:
        action = stack_vector_column(df["action"], "action")
        state = stack_vector_column(df["observation.state"], "observation.state")
    except Exception as exc:
        row["status"] = "error"
        add_issue(issues, "error", "parquet", "vector_shape_error", str(exc), episode_index=episode_index, source_date=date)
        return row

    row["action_dim"] = int(action.shape[1]) if action.ndim == 2 else 0
    row["state_dim"] = int(state.shape[1]) if state.ndim == 2 else 0
    row["action_nonfinite"] = int(np.size(action) - np.count_nonzero(np.isfinite(action)))
    row["state_nonfinite"] = int(np.size(state) - np.count_nonzero(np.isfinite(state)))
    if row["action_dim"] != ACTION_DIM:
        add_issue(issues, "error", "parquet", "action_dim_mismatch", "Expected 19D action", episode_index=episode_index, source_date=date, actual=row["action_dim"])
    if row["state_dim"] != STATE_DIM:
        add_issue(issues, "error", "parquet", "state_dim_mismatch", "Expected 26D observation.state", episode_index=episode_index, source_date=date, actual=row["state_dim"])
    if row["action_nonfinite"]:
        add_issue(issues, "error", "parquet", "nonfinite_action", "Action contains NaN or Inf", episode_index=episode_index, source_date=date, count=row["action_nonfinite"])
    if row["state_nonfinite"]:
        add_issue(issues, "error", "parquet", "nonfinite_state", "Observation state contains NaN or Inf", episode_index=episode_index, source_date=date, count=row["state_nonfinite"])

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        dt = np.diff(timestamps) if timestamps.size > 1 else np.asarray([], dtype=np.float64)
        row["timestamp_nonfinite"] = int(timestamps.size - np.count_nonzero(np.isfinite(timestamps)))
        add_prefixed(row, "timestamp_dt", dt)
        if row["timestamp_nonfinite"]:
            add_issue(issues, "error", "parquet", "nonfinite_timestamp", "Timestamp contains NaN or Inf", episode_index=episode_index, source_date=date)
        if np.any(dt < -1e-6):
            add_issue(issues, "error", "parquet", "timestamp_not_monotonic", "Timestamp decreases", episode_index=episode_index, source_date=date)
        if fps > 0 and dt.size:
            expected_dt = 1.0 / fps
            row["timestamp_dt_max_abs_error_s"] = float(np.max(np.abs(dt - expected_dt)))
            row["timestamp_bad_dt_count"] = int(np.count_nonzero(np.abs(dt - expected_dt) > timestamp_tolerance))
            if row["timestamp_bad_dt_count"]:
                add_issue(issues, "warning", "parquet", "timestamp_gap", "Timestamp spacing differs from fps", episode_index=episode_index, source_date=date, count=row["timestamp_bad_dt_count"], max_abs_error=row["timestamp_dt_max_abs_error_s"])

    if "frame_index" in df.columns and len(df):
        frame_index = df["frame_index"].to_numpy(dtype=np.int64)
        row["frame_index_contiguous"] = bool(np.array_equal(frame_index, np.arange(len(df), dtype=np.int64)))
        if not row["frame_index_contiguous"]:
            add_issue(issues, "warning", "parquet", "frame_index_not_contiguous", "frame_index should be 0..length-1", episode_index=episode_index, source_date=date)
    if "episode_index" in df.columns and len(df):
        unique_episode_indices = sorted(set(int(value) for value in df["episode_index"].to_numpy()))
        row["parquet_episode_indices"] = ",".join(str(value) for value in unique_episode_indices)
        if unique_episode_indices != [episode_index]:
            add_issue(issues, "error", "parquet", "episode_index_mismatch", "episode_index column differs from metadata", episode_index=episode_index, source_date=date, actual=unique_episode_indices)
    if "next.done" in df.columns and len(df):
        done = df["next.done"].to_numpy(dtype=bool)
        row["done_true_count"] = int(np.count_nonzero(done))
        row["done_only_last_frame"] = bool(done[-1] and not np.any(done[:-1]))
        if not row["done_only_last_frame"]:
            add_issue(issues, "warning", "parquet", "done_column_unexpected", "next.done should be true only on final row", episode_index=episode_index, source_date=date, true_count=row["done_true_count"])

    add_prefixed(row, "action_abs", np.abs(action))
    add_prefixed(row, "state_abs", np.abs(state))
    path_metrics("action", action[:, :3], fps, row)
    path_metrics("state", state[:, STATE_EEF_SLICE.start : STATE_EEF_SLICE.start + 3], fps, row)
    if action.shape[0] > 1:
        action_diff = np.diff(action, axis=0)
        state_diff = np.diff(state, axis=0)
        add_prefixed(row, "action_step_l2", l2(action_diff))
        add_prefixed(row, "state_step_l2", l2(state_diff))
        add_prefixed(row, "action_rot6d_step_angle_deg", rot6d_step_angle_deg(action[:, 3:9]))
        add_prefixed(row, "state_eef_rot6d_step_angle_deg", rot6d_step_angle_deg(state[:, 10:16]))
        add_prefixed(row, "hand_action_step_l2", l2(action_diff[:, ACTION_HAND_SLICE]))
        add_prefixed(row, "hand_state_step_l2", l2(state_diff[:, STATE_HAND_SLICE]))
    row.update({f"action_{key}": value for key, value in rot6d_orthogonality(action[:, 3:9]).items()})
    row.update({f"state_eef_{key}": value for key, value in rot6d_orthogonality(state[:, 10:16]).items()})
    if action.shape[0] and state.shape[0] and action.shape[1] == ACTION_DIM and state.shape[1] == STATE_DIM:
        n = min(action.shape[0], state.shape[0])
        add_prefixed(row, "eef_action_state_xyz_error_l2_mm", l2(action[:n, :3] - state[:n, 7:10]) * 1000.0)
        add_prefixed(row, "hand_action_state_error_l2", l2(action[:n, 9:19] - state[:n, 16:26]))
        relative_metrics(row=row, action=action[:n], state=state[:n], max_horizon=max_horizon, bounds=bounds)

    if float(row.get("action_step_xyz_l2_mm_max", 0) or 0) > 30.0:
        add_issue(issues, "warning", "motion", "large_action_xyz_jump", "Action EEF step exceeds 30 mm", episode_index=episode_index, source_date=date, max_mm=row.get("action_step_xyz_l2_mm_max"))
    if float(row.get("state_step_xyz_l2_mm_max", 0) or 0) > 30.0:
        add_issue(issues, "warning", "motion", "large_state_xyz_jump", "State EEF step exceeds 30 mm", episode_index=episode_index, source_date=date, max_mm=row.get("state_step_xyz_l2_mm_max"))
    if float(row.get("action_rot6d_col01_abs_dot_p99", 0) or 0) > 0.05:
        add_issue(issues, "warning", "motion", "action_rot6d_not_orthogonal", "Action rot6d columns are not close to orthogonal", episode_index=episode_index, source_date=date, p99=row.get("action_rot6d_col01_abs_dot_p99"))
    if float(row.get("relative_h32_q01q99_clip_ratio", 0) or 0) > 0.10:
        add_issue(issues, "warning", "relative_action", "high_h32_clip_ratio", "More than 10% of h32 relative action entries exceed prepared q01/q99 bounds", episode_index=episode_index, source_date=date, ratio=row.get("relative_h32_q01q99_clip_ratio"))
    return row


def analyze_video(
    *,
    dataset_dir: Path,
    info: dict[str, Any],
    episode_row: dict[str, Any],
    video_key: str,
    video_path: Path,
    sample_count: int,
    duration_tolerance: float,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_index = int(episode_row["episode_index"])
    date = source_date(episode_row)
    fps = float(info.get("fps", 0) or 0)
    expected_duration = int(episode_row.get("length", 0)) / fps if fps > 0 else math.nan
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "source_date": date,
        "video_key": video_key,
        "video_path": str(video_path.relative_to(dataset_dir) if video_path.is_relative_to(dataset_dir) else video_path),
        "exists": video_path.exists(),
        "expected_duration_s": expected_duration,
        "status": "ok",
    }
    if not video_path.exists():
        row["status"] = "error"
        add_issue(issues, "error", "video", "missing_video", "Video file is missing", episode_index=episode_index, source_date=date, video_key=video_key, path=str(video_path))
        return row
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        row["status"] = "skipped"
        row["reason"] = "ffprobe not found"
    else:
        try:
            stream = ffprobe_video(ffprobe, video_path)
            row.update(
                {
                    "codec_name": stream.get("codec_name"),
                    "pix_fmt": stream.get("pix_fmt"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "avg_fps": parse_frame_rate(stream.get("avg_frame_rate")),
                    "duration_s": float(stream.get("duration", "nan")) if stream.get("duration") is not None else math.nan,
                    "nb_read_frames": int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0),
                    "browser_safe": stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p",
                }
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            add_issue(issues, "error", "video", "ffprobe_failed", str(exc), episode_index=episode_index, source_date=date, video_key=video_key)
    row.update(sample_video_quality(video_path, sample_count))

    if row.get("browser_safe") is False:
        add_issue(issues, "warning", "video", "browser_unsafe_codec", "Video is not h264/yuv420p", episode_index=episode_index, source_date=date, video_key=video_key)
    duration = float(row.get("duration_s", math.nan) or math.nan)
    if math.isfinite(expected_duration) and math.isfinite(duration) and abs(duration - expected_duration) > duration_tolerance:
        add_issue(issues, "warning", "video", "duration_mismatch", "Video duration differs from episode length/fps", episode_index=episode_index, source_date=date, video_key=video_key, expected=expected_duration, actual=duration)
    if float(row.get("laplacian_var_mean", math.inf) or math.inf) < 20.0:
        add_issue(issues, "warning", "video", "low_sharpness", "Sampled video frames have low Laplacian variance", episode_index=episode_index, source_date=date, video_key=video_key, laplacian_var_mean=row.get("laplacian_var_mean"))
    return row


def aggregate_numeric(group: pd.DataFrame, column: str, prefix: str, row: dict[str, Any]) -> None:
    if column not in group:
        return
    values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return
    row[f"{prefix}_mean"] = float(np.mean(values))
    row[f"{prefix}_p50"] = float(np.percentile(values, 50))
    row[f"{prefix}_p95"] = float(np.percentile(values, 95))
    row[f"{prefix}_max"] = float(np.max(values))


def build_date_summary(episodes: pd.DataFrame, videos: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in episodes.groupby("source_date", dropna=False):
        row: dict[str, Any] = {
            "source_date": date,
            "episodes": int(len(group)),
            "frames": int(pd.to_numeric(group.get("frames"), errors="coerce").fillna(0).sum()),
        }
        for column in (
            "duration_s",
            "action_step_xyz_l2_mm_p95",
            "action_acceleration_xyz_l2_mm_s2_p95",
            "action_jerk_xyz_l2_mm_s3_p95",
            "state_step_xyz_l2_mm_p95",
            "hand_action_step_l2_p95",
            "relative_h32_all_l2_p95",
            "relative_h32_q01q99_clip_ratio",
            "eef_action_state_xyz_error_l2_mm_p95",
            "hand_action_state_error_l2_p95",
        ):
            aggregate_numeric(group, column, column, row)
        if not issues.empty:
            date_issues = issues[issues.get("source_date") == date]
            row["errors"] = int((date_issues.get("severity") == "error").sum()) if "severity" in date_issues else 0
            row["warnings"] = int((date_issues.get("severity") == "warning").sum()) if "severity" in date_issues else 0
        if not videos.empty:
            date_videos = videos[videos.get("source_date") == date]
            aggregate_numeric(date_videos, "laplacian_var_mean", "video_laplacian_var", row)
            row["video_checks"] = int(len(date_videos))
            row["video_browser_unsafe"] = int((date_videos.get("browser_safe") == False).sum()) if "browser_safe" in date_videos else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("source_date")


def robust_outlier_scores(metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = metrics.copy()
    usable = [column for column in columns if column in result]
    if not usable:
        result["robust_outlier_score"] = 0.0
        return result
    values = result[usable].apply(pd.to_numeric, errors="coerce")
    med = values.median(axis=0, skipna=True)
    mad = (values - med).abs().median(axis=0, skipna=True).replace(0, np.nan)
    z = 0.6745 * (values - med) / mad
    result["robust_outlier_score"] = z.abs().max(axis=1, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["robust_outlier_top_metric"] = z.abs().idxmax(axis=1).fillna("")

    centered = values.fillna(values.median(axis=0, skipna=True)).to_numpy(dtype=np.float64)
    if centered.shape[0] >= 2 and centered.shape[1] >= 2:
        centered = centered - np.mean(centered, axis=0, keepdims=True)
        scale = np.std(centered, axis=0, keepdims=True)
        centered = centered / np.maximum(scale, 1e-8)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt[:2].T
        result["pca_0"] = projected[:, 0]
        result["pca_1"] = projected[:, 1] if projected.shape[1] > 1 else 0.0
    else:
        result["pca_0"] = 0.0
        result["pca_1"] = 0.0
    return result


def date_color_map(dates: list[str]) -> dict[str, str]:
    return {date: DATE_COLORS[index % len(DATE_COLORS)] for index, date in enumerate(sorted(dates))}


def plot_date_summary(date_summary: pd.DataFrame, colors: dict[str, str], out_path: Path) -> None:
    if date_summary.empty:
        return
    dates = date_summary["source_date"].astype(str).tolist()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].bar(dates, date_summary["episodes"], color=[colors.get(date, "#4c78a8") for date in dates])
    axes[0].set_title("Episodes by source date")
    axes[0].set_ylabel("episodes")
    axes[1].bar(dates, date_summary["frames"], color=[colors.get(date, "#4c78a8") for date in dates])
    axes[1].set_title("Frames by source date")
    axes[1].set_ylabel("frames")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_distributions(metrics: pd.DataFrame, colors: dict[str, str], out_path: Path) -> None:
    plot_specs = [
        ("frames", "Episode frames"),
        ("action_step_xyz_l2_mm_p95", "Action EEF step p95 (mm)"),
        ("action_acceleration_xyz_l2_mm_s2_p95", "Action accel p95 (mm/s^2)"),
        ("action_jerk_xyz_l2_mm_s3_p95", "Action jerk p95 (mm/s^3)"),
        ("relative_h32_all_l2_p95", "Relative h32 all L2 p95"),
        ("relative_h32_q01q99_clip_ratio", "Relative h32 q01/q99 clip ratio"),
        ("eef_action_state_xyz_error_l2_mm_p95", "EEF action-state error p95 (mm)"),
        ("robust_outlier_score", "Robust outlier score"),
    ]
    present = [(column, title) for column, title in plot_specs if column in metrics]
    if not present:
        return
    dates = sorted(str(date) for date in metrics["source_date"].dropna().unique())
    fig, axes = plt.subplots(math.ceil(len(present) / 2), 2, figsize=(16, 4.4 * math.ceil(len(present) / 2)), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    rng = np.random.default_rng(7)
    for axis, (column, title) in zip(axes_flat, present):
        data = [pd.to_numeric(metrics.loc[metrics["source_date"].astype(str) == date, column], errors="coerce").dropna().to_numpy() for date in dates]
        boxplot_kwargs = {"patch_artist": True, "showfliers": False}
        try:
            box = axis.boxplot(data, tick_labels=dates, **boxplot_kwargs)
        except TypeError:
            box = axis.boxplot(data, labels=dates, **boxplot_kwargs)
        for patch, date in zip(box["boxes"], dates):
            patch.set_facecolor(colors.get(date, "#4c78a8"))
            patch.set_alpha(0.35)
        for index, (date, values) in enumerate(zip(dates, data), start=1):
            if values.size:
                jitter = rng.normal(0.0, 0.035, size=values.size)
                axis.scatter(np.full(values.size, index) + jitter, values, s=18, alpha=0.78, color=colors.get(date, "#4c78a8"), edgecolors="none")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    for axis in axes_flat[len(present) :]:
        axis.axis("off")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_outliers(metrics: pd.DataFrame, colors: dict[str, str], out_path: Path) -> None:
    if not {"pca_0", "pca_1", "source_date"}.issubset(metrics.columns):
        return
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for date, group in metrics.groupby("source_date"):
        axis.scatter(group["pca_0"], group["pca_1"], s=45, alpha=0.82, label=str(date), color=colors.get(str(date), "#4c78a8"))
    top = metrics.sort_values("robust_outlier_score", ascending=False).head(12)
    for _, row in top.iterrows():
        axis.annotate(str(int(row["episode_index"])), (float(row["pca_0"]), float(row["pca_1"])), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axis.set_title("Episode metric PCA, colored by source date")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, title="source date")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_video_quality(videos: pd.DataFrame, colors: dict[str, str], out_path: Path) -> None:
    if videos.empty or "laplacian_var_mean" not in videos:
        return
    dates = sorted(str(date) for date in videos["source_date"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "laplacian_var_mean", "Video sharpness: Laplacian variance mean"),
        (axes[1], "brightness_mean", "Video brightness mean"),
    ):
        if column not in videos:
            axis.axis("off")
            continue
        for date in dates:
            values = pd.to_numeric(videos.loc[videos["source_date"].astype(str) == date, column], errors="coerce").dropna()
            axis.scatter([date] * len(values), values, color=colors.get(date, "#4c78a8"), alpha=0.75, s=22)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def table_html(df: pd.DataFrame, columns: list[str], limit: int = 40) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    cols = [column for column in columns if column in df.columns]
    rows = []
    for _, row in df.head(limit).iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in cols) + "</tr>")
    if len(df) > limit:
        rows.append(f"<tr><td colspan=\"{len(cols)}\">Showing first {limit} of {len(df)} rows.</td></tr>")
    return "<table><tr>" + "".join(f"<th>{html.escape(column)}</th>" for column in cols) + "</tr>" + "".join(rows) + "</table>"


def write_html_report(
    *,
    summary: dict[str, Any],
    output_dir: Path,
    date_summary: pd.DataFrame,
    outliers: pd.DataFrame,
    issues: pd.DataFrame,
) -> None:
    image_names = [
        "date_summary.png",
        "metric_distributions_by_source_date.png",
        "episode_metric_pca_by_source_date.png",
        "video_quality_by_source_date.png",
    ]
    images = "\n".join(
        f'<h2>{html.escape(name)}</h2><img src="{name}" alt="{html.escape(name)}">'
        for name in image_names
        if (output_dir / name).exists()
    )
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smooth Quality All Dates</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    code {{ white-space: pre-wrap; }}
    img {{ max-width: 1400px; width: 100%; border: 1px solid #d9dee7; border-radius: 6px; }}
    table {{ border-collapse: collapse; margin: 14px 0 28px; min-width: 980px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; max-width: 1100px; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 6px; padding: 10px 12px; background: #fafbfc; }}
    .label {{ color: #52606d; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 20px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Smooth Quality Report: All Source Dates</h1>
  <p>Dataset: <code>{html.escape(summary['dataset_dir'])}</code></p>
  <div class="summary">
    <div class="card"><div class="label">Episodes</div><div class="value">{summary['episodes']}</div></div>
    <div class="card"><div class="label">Frames</div><div class="value">{summary['frames']}</div></div>
    <div class="card"><div class="label">Dates</div><div class="value">{summary['source_dates']}</div></div>
    <div class="card"><div class="label">Errors</div><div class="value">{summary['errors']}</div></div>
    <div class="card"><div class="label">Warnings</div><div class="value">{summary['warnings']}</div></div>
  </div>
  {images}
  <h2>Date Summary</h2>
  {table_html(date_summary, ['source_date', 'episodes', 'frames', 'errors', 'warnings', 'action_step_xyz_l2_mm_p95_mean', 'relative_h32_q01q99_clip_ratio_mean', 'video_laplacian_var_mean'])}
  <h2>Top Outlier Episodes</h2>
  {table_html(outliers, ['episode_index', 'source_date', 'frames', 'robust_outlier_score', 'robust_outlier_top_metric', 'action_step_xyz_l2_mm_p95', 'relative_h32_q01q99_clip_ratio', 'eef_action_state_xyz_error_l2_mm_p95'])}
  <h2>Issues</h2>
  {table_html(issues, ['severity', 'scope', 'code', 'episode_index', 'source_date', 'video_key', 'message'])}
  <h2>Files</h2>
  <ul>
    <li><code>episode_quality_metrics.csv</code></li>
    <li><code>date_quality_summary.csv</code></li>
    <li><code>video_quality.csv</code></li>
    <li><code>issue_rows.csv</code></li>
    <li><code>outlier_rankings.csv</code></li>
    <li><code>summary.json</code></li>
  </ul>
</body>
</html>
"""
    (output_dir / "smooth_quality_all_dates_report.html").write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = read_json(dataset_dir / "meta" / "info.json")
    episode_rows = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if not episode_rows:
        raise RuntimeError(f"No episode metadata found under {dataset_dir / 'meta'}")
    bounds = load_relative_bounds(args.relative_stats.expanduser().resolve())
    max_horizon = min(max(1, int(args.max_horizon)), 32)
    issues: list[dict[str, Any]] = []

    episode_metrics = [
        analyze_episode(
            dataset_dir=dataset_dir,
            info=info,
            episode_row=row,
            max_horizon=max_horizon,
            bounds=bounds,
            timestamp_tolerance=float(args.timestamp_tolerance),
            min_episode_frames=max(1, int(args.min_episode_frames)),
            issues=issues,
        )
        for row in episode_rows
    ]

    video_rows: list[dict[str, Any]] = []
    if not args.skip_video_scan:
        for row in episode_rows:
            for video_key, video_path in video_paths_for_episode(dataset_dir, info, row).items():
                video_rows.append(
                    analyze_video(
                        dataset_dir=dataset_dir,
                        info=info,
                        episode_row=row,
                        video_key=video_key,
                        video_path=video_path,
                        sample_count=max(0, int(args.video_samples)),
                        duration_tolerance=float(args.video_duration_tolerance),
                        issues=issues,
                    )
                )

    metrics = pd.DataFrame(episode_metrics).sort_values("episode_index")
    outlier_columns = [
        "frames",
        "action_step_xyz_l2_mm_p95",
        "action_acceleration_xyz_l2_mm_s2_p95",
        "action_jerk_xyz_l2_mm_s3_p95",
        "state_step_xyz_l2_mm_p95",
        "hand_action_step_l2_p95",
        "relative_h32_all_l2_p95",
        "relative_h32_q01q99_clip_ratio",
        "eef_action_state_xyz_error_l2_mm_p95",
        "hand_action_state_error_l2_p95",
    ]
    metrics = robust_outlier_scores(metrics, outlier_columns)
    videos = pd.DataFrame(video_rows)
    issue_df = pd.DataFrame(issues, columns=ISSUE_COLUMNS if not issues else None)
    date_summary = build_date_summary(metrics, videos, issue_df)
    outliers = metrics.sort_values("robust_outlier_score", ascending=False)

    colors = date_color_map([str(date) for date in metrics["source_date"].dropna().unique()])
    plot_date_summary(date_summary, colors, output_dir / "date_summary.png")
    plot_metric_distributions(metrics, colors, output_dir / "metric_distributions_by_source_date.png")
    plot_pca_outliers(metrics, colors, output_dir / "episode_metric_pca_by_source_date.png")
    plot_video_quality(videos, colors, output_dir / "video_quality_by_source_date.png")

    metrics.to_csv(output_dir / "episode_quality_metrics.csv", index=False)
    date_summary.to_csv(output_dir / "date_quality_summary.csv", index=False)
    videos.to_csv(output_dir / "video_quality.csv", index=False)
    issue_df.to_csv(output_dir / "issue_rows.csv", index=False)
    outliers.to_csv(output_dir / "outlier_rankings.csv", index=False)

    errors = int((issue_df.get("severity") == "error").sum()) if not issue_df.empty and "severity" in issue_df else 0
    warnings = int((issue_df.get("severity") == "warning").sum()) if not issue_df.empty and "severity" in issue_df else 0
    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "relative_stats": str(args.relative_stats.expanduser().resolve()) if args.relative_stats.exists() else None,
        "episodes": int(len(metrics)),
        "frames": int(pd.to_numeric(metrics.get("frames"), errors="coerce").fillna(0).sum()),
        "source_dates": sorted(str(date) for date in metrics["source_date"].dropna().unique()),
        "date_colors": colors,
        "max_horizon": int(max_horizon),
        "video_checks": int(len(videos)),
        "errors": errors,
        "warnings": warnings,
        "top_outliers": outliers[["episode_index", "source_date", "robust_outlier_score", "robust_outlier_top_metric"]].head(12).to_dict(orient="records"),
        "tables": {
            "episodes": str((output_dir / "episode_quality_metrics.csv").resolve()),
            "dates": str((output_dir / "date_quality_summary.csv").resolve()),
            "videos": str((output_dir / "video_quality.csv").resolve()),
            "issues": str((output_dir / "issue_rows.csv").resolve()),
            "outliers": str((output_dir / "outlier_rankings.csv").resolve()),
        },
        "plots": {
            "date_summary": str((output_dir / "date_summary.png").resolve()),
            "metric_distributions": str((output_dir / "metric_distributions_by_source_date.png").resolve()),
            "pca": str((output_dir / "episode_metric_pca_by_source_date.png").resolve()),
            "video_quality": str((output_dir / "video_quality_by_source_date.png").resolve()),
        },
        "report": str((output_dir / "smooth_quality_all_dates_report.html").resolve()),
    }
    write_json(output_dir / "summary.json", summary)
    write_html_report(summary=summary, output_dir=output_dir, date_summary=date_summary, outliers=outliers, issues=issue_df)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
