#!/usr/bin/env python3
"""Run batch-level quality checks for a LeRobot v2 dataset."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_DATASET_DIR = MISSION_DIR / "smooth"
DEFAULT_OUTPUT_ROOT = MISSION_DIR / "batch_quality"
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMPACT_DATE_RE = re.compile(r"(20\d{6})")
REQUIRED_META_FILES = ("info.json", "episodes.jsonl", "modality.json", "tasks.jsonl")
REQUIRED_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "next.done",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date", help="Check one batch date, formatted as YYYY-MM-DD.")
    parser.add_argument(
        "--date-field",
        choices=("source", "created", "trimmed", "smoothed", "any"),
        default="source",
        help="Which episode provenance date should match --date.",
    )
    parser.add_argument("--all-dates", action="store_true", help="Write one report per discovered date.")
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.25)
    parser.add_argument("--video-duration-tolerance", type=float, default=0.35)
    parser.add_argument("--max-video-checks", type=int, default=-1)
    parser.add_argument("--skip-video-check", action="store_true")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    if key in row:
        return row[key]
    return metadata(row).get(key)


def source_date(row: dict[str, Any]) -> str | None:
    candidates = (
        "raw_episode_id",
        "raw_episode_dir",
        "source_dataset_name",
        "source_dataset",
        "source_path",
        "data_path",
    )
    for key in candidates:
        value = row_value(row, key)
        if not isinstance(value, str):
            continue
        compact = COMPACT_DATE_RE.search(value)
        if compact:
            return compact_date_to_iso(compact.group(1))
        date = iso_date(value)
        if date:
            return date
    return iso_date(row_value(row, "created_at_utc"))


def row_dates(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        "source": source_date(row),
        "created": iso_date(row_value(row, "created_at_utc")),
        "trimmed": iso_date(row_value(row, "trimmed_at_utc")),
        "smoothed": iso_date(row_value(row, "smoothed_at_utc")),
    }


def row_matches_date(row: dict[str, Any], date: str, date_field: str) -> bool:
    dates = row_dates(row)
    if date_field == "any":
        return date in {value for value in dates.values() if value}
    return dates.get(date_field) == date


def discovered_dates(rows: list[dict[str, Any]], date_field: str) -> list[str]:
    dates: set[str] = set()
    for row in rows:
        values = row_dates(row)
        if date_field == "any":
            dates.update(value for value in values.values() if value)
        elif values.get(date_field):
            dates.add(str(values[date_field]))
    return sorted(dates)


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    scope: str,
    code: str,
    message: str,
    **extra: Any,
) -> None:
    row = {"severity": severity, "scope": scope, "code": code, "message": message}
    row.update(extra)
    issues.append(row)


def split_keys(spec: str) -> list[str]:
    return [part for part in spec.split("/") if part]


def flatten_finite_issues(value: Any, path: str, issues: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_finite_issues(child, f"{path}/{key}", issues)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            flatten_finite_issues(child, f"{path}[{idx}]", issues)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            issues.append(f"{path}={value}")


def vector_width(feature: dict[str, Any] | None) -> int | None:
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if isinstance(shape, list) and len(shape) == 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    return None


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    bad = [idx for idx, value in enumerate(values) if value.shape[0] != width]
    if bad:
        raise ValueError(f"Column {column_name!r} has inconsistent vector widths; first bad row={bad[0]}")
    return np.vstack(values)


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
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


def prefixed(prefix: str, stats: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def l2(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def row_episode_index(row: dict[str, Any]) -> int:
    return int(row["episode_index"])


def data_path_for_episode(dataset_dir: Path, info: dict[str, Any], row: dict[str, Any]) -> Path:
    meta = metadata(row)
    for key in ("data_path", "output_data_path", "smooth_data_path", "source_data_path"):
        value = meta.get(key, row.get(key))
        if isinstance(value, str):
            return dataset_dir / value
    episode_index = row_episode_index(row)
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
    episode_index = row_episode_index(row)
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // max(1, chunk_size)
    video_features = {
        key: spec
        for key, spec in info.get("features", {}).items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    }
    return {
        feature_key: dataset_dir
        / pattern.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key=feature_key,
        )
        for feature_key in video_features
    }


def expected_video_info(info: dict[str, Any], video_key: str) -> dict[str, Any]:
    features = info.get("features", {})
    candidates = [video_key]
    if not str(video_key).startswith("observation.images."):
        candidates.append(f"observation.images.{video_key}")
    for key in candidates:
        feature = features.get(key)
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            return feature.get("info", {}) if isinstance(feature.get("info"), dict) else {}
    return {}


def parse_frame_rate(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_float = float(denominator)
            return None if denominator_float == 0 else float(numerator) / denominator_float
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
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,duration,nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams") or []
    return streams[0] if streams else {}


def check_video(
    *,
    dataset_dir: Path,
    info: dict[str, Any],
    episode_row: dict[str, Any],
    video_key: str,
    video_path: Path,
    ffprobe: str | None,
    duration_tolerance: float,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_index = row_episode_index(episode_row)
    length = int(episode_row.get("length", 0))
    expected_fps = float(info.get("fps", 0) or 0)
    expected_duration = length / expected_fps if expected_fps > 0 else math.nan
    expected = expected_video_info(info, video_key)
    row: dict[str, Any] = {
        "episode_index": episode_index,
        "video_key": video_key,
        "video_path": str(video_path.relative_to(dataset_dir) if video_path.is_relative_to(dataset_dir) else video_path),
        "exists": video_path.exists(),
        "status": "ok",
        "expected_duration_s": expected_duration,
    }
    if not video_path.exists():
        row["status"] = "error"
        add_issue(issues, "error", "video", "missing_video", "Video file is missing", episode_index=episode_index, video_key=video_key, path=str(video_path))
        return row
    if ffprobe is None:
        row["status"] = "skipped"
        row["reason"] = "ffprobe not found"
        return row

    try:
        stream = ffprobe_video(ffprobe, video_path)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        add_issue(issues, "error", "video", "ffprobe_failed", str(exc), episode_index=episode_index, video_key=video_key, path=str(video_path))
        return row

    fps = parse_frame_rate(stream.get("avg_frame_rate"))
    duration = float(stream.get("duration", "nan")) if stream.get("duration") is not None else math.nan
    nb_frames = int(stream.get("nb_read_frames", 0) or 0)
    row.update(
        {
            "codec_name": stream.get("codec_name"),
            "pix_fmt": stream.get("pix_fmt"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "avg_fps": fps,
            "duration_s": duration,
            "nb_read_frames": nb_frames,
            "browser_safe": stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p",
        }
    )

    if row["browser_safe"] is False:
        add_issue(issues, "warning", "video", "browser_unsafe_codec", "Video is not h264/yuv420p", episode_index=episode_index, video_key=video_key)
    for field, expected_key in (("width", "video.width"), ("height", "video.height")):
        expected_value = expected.get(expected_key)
        if expected_value is not None and stream.get(field) != expected_value:
            add_issue(
                issues,
                "error",
                "video",
                "video_shape_mismatch",
                f"Video {field} does not match info.json",
                episode_index=episode_index,
                video_key=video_key,
                expected=expected_value,
                actual=stream.get(field),
            )
    if expected_fps > 0 and fps is not None and abs(fps - expected_fps) > 0.05:
        add_issue(issues, "warning", "video", "fps_mismatch", "Video fps differs from info.json", episode_index=episode_index, video_key=video_key, expected=expected_fps, actual=fps)
    if math.isfinite(expected_duration) and math.isfinite(duration) and abs(duration - expected_duration) > duration_tolerance:
        add_issue(issues, "warning", "video", "duration_mismatch", "Video duration differs from episode length/fps", episode_index=episode_index, video_key=video_key, expected=expected_duration, actual=duration)
    if nb_frames > 0 and length > 0 and abs(nb_frames - length) > max(2, math.ceil(expected_fps * duration_tolerance)):
        add_issue(issues, "warning", "video", "frame_count_mismatch", "Video frame count differs from episode length", episode_index=episode_index, video_key=video_key, expected=length, actual=nb_frames)
    return row


def check_parquet_episode(
    *,
    dataset_dir: Path,
    info: dict[str, Any],
    episode_row: dict[str, Any],
    min_length: int,
    timestamp_tolerance: float,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_index = row_episode_index(episode_row)
    data_path = data_path_for_episode(dataset_dir, info, episode_row)
    relative_path = str(data_path.relative_to(dataset_dir) if data_path.is_relative_to(dataset_dir) else data_path)
    metric: dict[str, Any] = {
        "episode_index": episode_index,
        "source_date": row_dates(episode_row).get("source"),
        "declared_length": int(episode_row.get("length", 0)),
        "data_path": relative_path,
        "parquet_exists": data_path.exists(),
        "status": "ok",
    }
    if int(episode_row.get("length", 0)) < min_length:
        add_issue(issues, "error", "episode", "short_episode", "Episode length is below --min-length", episode_index=episode_index, length=episode_row.get("length"))
    if not data_path.exists():
        metric["status"] = "error"
        add_issue(issues, "error", "parquet", "missing_parquet", "Parquet file is missing", episode_index=episode_index, path=str(data_path))
        return metric

    try:
        df = pd.read_parquet(data_path)
    except Exception as exc:
        metric["status"] = "error"
        metric["read_error"] = str(exc)
        add_issue(issues, "error", "parquet", "read_failed", str(exc), episode_index=episode_index, path=str(data_path))
        return metric

    metric["rows"] = int(len(df))
    if len(df) != int(episode_row.get("length", -1)):
        add_issue(issues, "error", "parquet", "length_mismatch", "Parquet row count differs from episodes.jsonl", episode_index=episode_index, declared=episode_row.get("length"), actual=len(df))

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    metric["missing_columns"] = ",".join(missing_columns)
    for column in missing_columns:
        add_issue(issues, "error", "parquet", "missing_column", f"Required column {column!r} is missing", episode_index=episode_index, column=column)

    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    for column in ("action", "observation.state"):
        if column not in df.columns:
            continue
        try:
            values = stack_vector_column(df[column], column)
        except Exception as exc:
            add_issue(issues, "error", "parquet", "vector_shape_error", str(exc), episode_index=episode_index, column=column)
            continue
        width = values.shape[1] if values.ndim == 2 else 0
        expected_width = vector_width(features.get(column))
        metric[f"{column}_width"] = width
        metric[f"{column}_nonfinite"] = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        if expected_width is not None and width != expected_width:
            add_issue(issues, "error", "parquet", "vector_width_mismatch", f"{column} width differs from info.json", episode_index=episode_index, column=column, expected=expected_width, actual=width)
        if metric[f"{column}_nonfinite"]:
            add_issue(issues, "error", "parquet", "nonfinite_vector", f"{column} contains NaN or Inf", episode_index=episode_index, column=column, count=metric[f"{column}_nonfinite"])
        diff = np.diff(values, axis=0) if values.shape[0] > 1 else np.zeros((0, width), dtype=np.float64)
        metric.update(prefixed(column.replace(".", "_") + "_abs", finite_summary(np.abs(values))))
        metric.update(prefixed(column.replace(".", "_") + "_step_l2", finite_summary(l2(diff))))

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        dt = np.diff(timestamps) if timestamps.size > 1 else np.asarray([], dtype=np.float64)
        metric["timestamp_nonfinite"] = int(timestamps.size - np.count_nonzero(np.isfinite(timestamps)))
        metric.update(prefixed("timestamp_dt", finite_summary(dt)))
        if metric["timestamp_nonfinite"]:
            add_issue(issues, "error", "parquet", "nonfinite_timestamp", "timestamp contains NaN or Inf", episode_index=episode_index)
        if np.any(dt < -1e-6):
            add_issue(issues, "error", "parquet", "timestamp_not_monotonic", "timestamp decreases within episode", episode_index=episode_index)
        fps = float(info.get("fps", 0) or 0)
        if fps > 0 and dt.size:
            expected_dt = 1.0 / fps
            bad_dt = np.abs(dt - expected_dt) > timestamp_tolerance
            metric["timestamp_bad_dt_count"] = int(np.count_nonzero(bad_dt))
            if metric["timestamp_bad_dt_count"]:
                add_issue(issues, "warning", "parquet", "timestamp_gap", "timestamp step differs from expected fps", episode_index=episode_index, count=metric["timestamp_bad_dt_count"], expected_dt=expected_dt)

    if "frame_index" in df.columns and len(df):
        frame_index = df["frame_index"].to_numpy(dtype=np.int64)
        expected = np.arange(len(df), dtype=np.int64)
        metric["frame_index_min"] = int(frame_index.min())
        metric["frame_index_max"] = int(frame_index.max())
        if not np.array_equal(frame_index, expected):
            add_issue(issues, "warning", "parquet", "frame_index_not_contiguous", "frame_index is not 0..length-1", episode_index=episode_index)

    if "episode_index" in df.columns and len(df):
        unique_episode_indices = sorted(set(int(value) for value in df["episode_index"].to_numpy()))
        metric["parquet_episode_indices"] = ",".join(str(value) for value in unique_episode_indices)
        if unique_episode_indices != [episode_index]:
            add_issue(issues, "error", "parquet", "episode_index_mismatch", "episode_index column does not match episodes.jsonl", episode_index=episode_index, actual=unique_episode_indices)

    if "next.done" in df.columns and len(df):
        done = df["next.done"].to_numpy(dtype=bool)
        metric["done_true_count"] = int(np.count_nonzero(done))
        if not done[-1] or np.any(done[:-1]):
            add_issue(issues, "warning", "parquet", "done_column_unexpected", "next.done should be false except final row", episode_index=episode_index, true_count=metric["done_true_count"])

    return metric


def check_meta(dataset_dir: Path, info: dict[str, Any], rows: list[dict[str, Any]], issues: list[dict[str, Any]]) -> dict[str, Any]:
    meta_dir = dataset_dir / "meta"
    for name in REQUIRED_META_FILES:
        if not (meta_dir / name).exists():
            add_issue(issues, "error", "meta", "missing_meta_file", f"meta/{name} is missing", path=str(meta_dir / name))
    stats_path = meta_dir / "stats.json"
    if not stats_path.exists():
        add_issue(issues, "warning", "meta", "missing_stats", "meta/stats.json is missing")
    else:
        finite_issues: list[str] = []
        flatten_finite_issues(read_json(stats_path), "stats", finite_issues)
        for issue in finite_issues[:50]:
            add_issue(issues, "error", "meta", "nonfinite_stats", "stats.json contains NaN or Inf", path=issue)
        if len(finite_issues) > 50:
            add_issue(issues, "error", "meta", "nonfinite_stats_truncated", "More nonfinite stats entries were omitted", count=len(finite_issues) - 50)

    episode_indices = [row_episode_index(row) for row in rows if "episode_index" in row]
    duplicates = [index for index, count in Counter(episode_indices).items() if count > 1]
    if duplicates:
        add_issue(issues, "error", "meta", "duplicate_episode_index", "episodes.jsonl contains duplicate episode_index values", indices=duplicates)
    if episode_indices and sorted(episode_indices) != list(range(min(episode_indices), max(episode_indices) + 1)):
        add_issue(issues, "warning", "meta", "noncontiguous_episode_indices", "episode_index values are not contiguous", first=min(episode_indices), last=max(episode_indices), count=len(episode_indices))

    total_episodes = info.get("total_episodes")
    if total_episodes is not None and int(total_episodes) != len(rows):
        add_issue(issues, "error", "meta", "total_episodes_mismatch", "info.json total_episodes differs from episodes.jsonl", expected=total_episodes, actual=len(rows))
    total_frames = info.get("total_frames")
    declared_frames = sum(int(row.get("length", 0)) for row in rows)
    if total_frames is not None and int(total_frames) != declared_frames:
        add_issue(issues, "error", "meta", "total_frames_mismatch", "info.json total_frames differs from episodes.jsonl lengths", expected=total_frames, actual=declared_frames)
    if float(info.get("fps", 0) or 0) <= 0:
        add_issue(issues, "error", "meta", "invalid_fps", "info.json fps must be positive", fps=info.get("fps"))

    return {
        "dataset_dir": str(dataset_dir),
        "meta_dir": str(meta_dir),
        "episodes_jsonl_rows": len(rows),
        "declared_frames": int(declared_frames),
        "fps": info.get("fps"),
        "chunks_size": info.get("chunks_size"),
    }


def plot_episode_metrics(episode_metrics: pd.DataFrame, out_path: Path) -> None:
    if episode_metrics.empty:
        return
    x = episode_metrics["episode_index"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True, constrained_layout=True)
    axes[0].bar(x, episode_metrics.get("rows", episode_metrics["declared_length"]), color="#4c78a8", alpha=0.85)
    axes[0].set_ylabel("rows")
    axes[0].set_title("Episode lengths")
    if "action_step_l2_p95" in episode_metrics:
        axes[1].plot(x, episode_metrics["action_step_l2_p95"], marker="o", color="#f58518")
    axes[1].set_ylabel("p95")
    axes[1].set_title("Action adjacent-step L2")
    if "observation_state_step_l2_p95" in episode_metrics:
        axes[2].plot(x, episode_metrics["observation_state_step_l2_p95"], marker="o", color="#54a24b")
    axes[2].set_ylabel("p95")
    axes[2].set_title("State adjacent-step L2")
    axes[2].set_xlabel("episode_index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def table_rows(rows: list[dict[str, Any]], columns: list[str], limit: int = 200) -> str:
    body = []
    for row in rows[:limit]:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    if len(rows) > limit:
        body.append(f"<tr><td colspan=\"{len(columns)}\">Showing first {limit} of {len(rows)} rows.</td></tr>")
    return "".join(body)


def write_html_report(summary: dict[str, Any], out_dir: Path, issue_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], video_rows: list[dict[str, Any]]) -> None:
    issue_columns = ["severity", "scope", "code", "episode_index", "video_key", "message"]
    episode_columns = ["episode_index", "source_date", "declared_length", "rows", "status", "action_nonfinite", "observation.state_nonfinite", "timestamp_bad_dt_count"]
    video_columns = ["episode_index", "video_key", "exists", "status", "codec_name", "pix_fmt", "avg_fps", "duration_s", "nb_read_frames", "browser_safe"]
    plot_html = '<img src="episode_metrics.png" alt="episode metrics">' if (out_dir / "episode_metrics.png").exists() else ""
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LeRobot Batch Quality {html.escape(str(summary['batch']))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    code {{ white-space: pre-wrap; }}
    img {{ width: 100%; max-width: 1300px; border: 1px solid #d9dee7; border-radius: 6px; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; min-width: 980px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 18px 0; max-width: 1200px; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 6px; padding: 10px 12px; background: #fafbfc; }}
    .label {{ color: #52606d; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 20px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>LeRobot Batch Quality Report</h1>
  <p>Dataset: <code>{html.escape(summary['dataset_dir'])}</code><br>Batch: <strong>{html.escape(str(summary['batch']))}</strong></p>
  <div class="summary">
    <div class="card"><div class="label">Episodes</div><div class="value">{summary['episodes']}</div></div>
    <div class="card"><div class="label">Frames</div><div class="value">{summary['frames']}</div></div>
    <div class="card"><div class="label">Errors</div><div class="value">{summary['errors']}</div></div>
    <div class="card"><div class="label">Warnings</div><div class="value">{summary['warnings']}</div></div>
  </div>
  <h2>Episode Metrics</h2>
  {plot_html}
  <h2>Issues</h2>
  <table><tr>{''.join(f'<th>{column}</th>' for column in issue_columns)}</tr>{table_rows(issue_rows, issue_columns)}</table>
  <h2>Episodes</h2>
  <table><tr>{''.join(f'<th>{column}</th>' for column in episode_columns)}</tr>{table_rows(episode_rows, episode_columns)}</table>
  <h2>Videos</h2>
  <table><tr>{''.join(f'<th>{column}</th>' for column in video_columns)}</tr>{table_rows(video_rows, video_columns)}</table>
  <h2>Files</h2>
  <ul>
    <li><code>summary.json</code></li>
    <li><code>episode_checks.csv</code></li>
    <li><code>issue_rows.csv</code></li>
    <li><code>video_checks.csv</code></li>
  </ul>
</body>
</html>
"""
    (out_dir / "batch_quality_report.html").write_text(html_doc, encoding="utf-8")


def check_batch(
    *,
    dataset_dir: Path,
    output_root: Path,
    all_rows: list[dict[str, Any]],
    info: dict[str, Any],
    batch_date: str | None,
    date_field: str,
    min_length: int,
    timestamp_tolerance: float,
    video_duration_tolerance: float,
    max_video_checks: int,
    skip_video_check: bool,
) -> dict[str, Any]:
    if batch_date is None:
        selected_rows = list(all_rows)
        batch_name = "all"
    else:
        selected_rows = [row for row in all_rows if row_matches_date(row, batch_date, date_field)]
        batch_name = f"{date_field}_{batch_date}"
    if not selected_rows:
        raise RuntimeError(f"No episodes matched batch {batch_name!r}")

    out_dir = output_root / batch_name
    out_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    meta_summary = check_meta(dataset_dir, info, all_rows, issues)

    episode_metrics = [
        check_parquet_episode(
            dataset_dir=dataset_dir,
            info=info,
            episode_row=row,
            min_length=min_length,
            timestamp_tolerance=timestamp_tolerance,
            issues=issues,
        )
        for row in selected_rows
    ]

    ffprobe = shutil.which("ffprobe")
    video_rows: list[dict[str, Any]] = []
    if not skip_video_check:
        checked = 0
        for row in selected_rows:
            for video_key, video_path in video_paths_for_episode(dataset_dir, info, row).items():
                if max_video_checks >= 0 and checked >= max_video_checks:
                    break
                video_rows.append(
                    check_video(
                        dataset_dir=dataset_dir,
                        info=info,
                        episode_row=row,
                        video_key=video_key,
                        video_path=video_path,
                        ffprobe=ffprobe,
                        duration_tolerance=video_duration_tolerance,
                        issues=issues,
                    )
                )
                checked += 1
            if max_video_checks >= 0 and checked >= max_video_checks:
                break

    episode_df = pd.DataFrame(episode_metrics).sort_values("episode_index")
    issue_df = pd.DataFrame(issues)
    video_df = pd.DataFrame(video_rows)
    episode_df.to_csv(out_dir / "episode_checks.csv", index=False)
    issue_df.to_csv(out_dir / "issue_rows.csv", index=False)
    video_df.to_csv(out_dir / "video_checks.csv", index=False)
    plot_episode_metrics(episode_df, out_dir / "episode_metrics.png")

    errors = sum(1 for issue in issues if issue.get("severity") == "error")
    warnings = sum(1 for issue in issues if issue.get("severity") == "warning")
    summary = {
        **meta_summary,
        "batch": batch_name,
        "date": batch_date,
        "date_field": date_field,
        "output_dir": str(out_dir),
        "episodes": int(len(selected_rows)),
        "frames": int(sum(int(row.get("length", 0)) for row in selected_rows)),
        "errors": int(errors),
        "warnings": int(warnings),
        "video_checks": int(len(video_rows)),
        "status": "fail" if errors else "pass_with_warnings" if warnings else "pass",
        "tables": {
            "episodes": str((out_dir / "episode_checks.csv").resolve()),
            "issues": str((out_dir / "issue_rows.csv").resolve()),
            "videos": str((out_dir / "video_checks.csv").resolve()),
        },
        "report": str((out_dir / "batch_quality_report.html").resolve()),
    }
    write_json(out_dir / "summary.json", summary)
    write_html_report(summary, out_dir, issues, episode_metrics, video_rows)
    return summary


def main() -> int:
    args = parse_args()
    if args.date and not DATE_RE.fullmatch(args.date):
        raise ValueError("--date must be formatted as YYYY-MM-DD")
    if args.date and args.all_dates:
        raise ValueError("Use either --date or --all-dates, not both")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Expected LeRobot dataset meta files under {dataset_dir / 'meta'}")
    info = read_json(info_path)
    rows = read_jsonl(episodes_path)

    if args.all_dates:
        dates = discovered_dates(rows, args.date_field)
        if not dates:
            raise RuntimeError(f"No dates discovered for date_field={args.date_field!r}")
    else:
        dates = [args.date] if args.date else [None]

    summaries = [
        check_batch(
            dataset_dir=dataset_dir,
            output_root=output_root,
            all_rows=rows,
            info=info,
            batch_date=date,
            date_field=args.date_field,
            min_length=max(1, int(args.min_length)),
            timestamp_tolerance=float(args.timestamp_tolerance),
            video_duration_tolerance=float(args.video_duration_tolerance),
            max_video_checks=int(args.max_video_checks),
            skip_video_check=bool(args.skip_video_check),
        )
        for date in dates
    ]

    index = {
        "dataset_dir": str(dataset_dir),
        "date_field": args.date_field,
        "batches": summaries,
        "output_root": str(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
