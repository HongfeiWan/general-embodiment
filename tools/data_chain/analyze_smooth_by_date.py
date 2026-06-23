#!/usr/bin/env python3
"""Analyze smooth LeRobot episodes selected by capture or processing date."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config_relative_action_curves import groot_relative_groups, l2, rot6d_angle_deg


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_SMOOTH_DIR = MISSION_DIR / "smooth"
DEFAULT_RELATIVE_STATS = MISSION_DIR / "prepared_smooth" / "meta" / "relative_stats.json"
DEFAULT_OUTPUT_ROOT = MISSION_DIR / "smooth_quality"
ACTION_EEF_SLICE = slice(0, 9)
ACTION_HAND_SLICE = slice(9, 19)
STATE_ARM_SLICE = slice(0, 7)
STATE_EEF_SLICE = slice(7, 16)
STATE_HAND_SLICE = slice(16, 26)
MAX_HORIZON = 32

DIM_NAMES = (
    ["eef_x", "eef_y", "eef_z"]
    + [f"eef_rot6d_{idx}" for idx in range(6)]
    + [
        "thumb_cmc_pitch",
        "thumb_cmc_yaw",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
        "index_mcp_roll",
        "ring_mcp_roll",
        "pinky_mcp_roll",
        "thumb_cmc_roll",
    ]
    + [f"arm_joint_{idx}" for idx in range(7)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smooth-dir", type=Path, default=DEFAULT_SMOOTH_DIR)
    parser.add_argument("--date", required=True, help="Date to analyze, formatted as YYYY-MM-DD.")
    parser.add_argument(
        "--date-field",
        choices=("source", "created", "smoothed", "any"),
        default="source",
        help="Which date provenance field should match --date.",
    )
    parser.add_argument("--relative-stats", type=Path, default=DEFAULT_RELATIVE_STATS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-horizon", type=int, default=MAX_HORIZON)
    parser.add_argument("--max-video-checks", type=int, default=-1)
    parser.add_argument("--skip-video-check", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def iso_date(value: object) -> str | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    date = value[:10]
    return date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else None


def compact_date_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def source_date(row: dict[str, Any]) -> str | None:
    candidates = [
        row.get("raw_episode_id"),
        row.get("raw_episode_dir"),
        row.get("source_dataset_name"),
        row.get("source_dataset"),
        row.get("source_path"),
    ]
    for value in candidates:
        if not isinstance(value, str):
            continue
        match = re.search(r"(20\d{6})", value)
        if match:
            return compact_date_to_iso(match.group(1))
    return iso_date(row.get("created_at_utc"))


def row_dates(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        "source": source_date(row),
        "created": iso_date(row.get("created_at_utc")),
        "smoothed": iso_date(row.get("smoothed_at_utc")),
    }


def row_matches_date(row: dict[str, Any], date: str, date_field: str) -> bool:
    dates = row_dates(row)
    if date_field == "any":
        return date in set(value for value in dates.values() if value is not None)
    return dates.get(date_field) == date


def smooth_episode_index(row: dict[str, Any]) -> int:
    for key in ("output_episode_index", "smooth_episode_index", "episode_index"):
        if key in row:
            return int(row[key])
    raise KeyError(f"No smooth episode index in manifest row: {row}")


def smooth_data_path(smooth_dir: Path, row: dict[str, Any]) -> Path:
    for key in ("output_data_path", "smooth_data_path"):
        value = row.get(key)
        if isinstance(value, str):
            return smooth_dir / value
    episode_index = smooth_episode_index(row)
    matches = list((smooth_dir / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet found for smooth episode {episode_index}")
    return matches[0]


def video_paths(smooth_dir: Path, row: dict[str, Any]) -> dict[str, Path]:
    raw_paths = row.get("output_video_paths") or row.get("smooth_video_paths") or {}
    if not isinstance(raw_paths, dict):
        return {}
    return {str(key): smooth_dir / str(value) for key, value in raw_paths.items()}


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        raise ValueError(f"Column {column_name!r} is empty")
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise ValueError(f"Column {column_name!r} has inconsistent vector sizes")
    return np.vstack(values)


def read_episode(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    action = stack_vector_column(df["action"], "action")
    state = stack_vector_column(df["observation.state"], "observation.state")
    return df, action, state


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
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


def action_group(dim: int) -> str:
    if dim < 3:
        return "eef_xyz"
    if dim < 9:
        return "eef_rot6d"
    if dim < 19:
        return "hand_joint_target"
    return "arm_joint_target"


def concat_relative(groups: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([groups["eef_9d"], groups["hand_joint_target"], groups["arm_joint_target"]], axis=1)


def load_relative_bounds(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
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
    return low, high


def normalize_q01_q99(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    denom = np.maximum(high - low, 1e-8)
    return 2.0 * (values - low) / denom - 1.0


def episode_quality_rows(
    *,
    episode_index: int,
    dates: dict[str, str | None],
    parquet_path: Path,
    max_horizon: int,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    df, action, state = read_episode(parquet_path)
    if action.shape[1] != 19:
        raise RuntimeError(f"Expected 19D action in {parquet_path}, got {action.shape[1]}D")

    action_diff = np.diff(action, axis=0) if action.shape[0] > 1 else np.zeros((0, action.shape[1]))
    row: dict[str, Any] = {
        "episode_index": int(episode_index),
        "frames": int(action.shape[0]),
        "source_date": dates.get("source"),
        "created_date": dates.get("created"),
        "smoothed_date": dates.get("smoothed"),
        "parquet_path": str(parquet_path),
    }
    row.update(prefixed("action_abs", finite_summary(np.abs(action))))
    row.update(prefixed("action_step_l2", finite_summary(l2(action_diff))))
    row.update(prefixed("eef_xyz_step_l2_mm", finite_summary(l2(action_diff[:, :3]) * 1000.0)))
    row.update(prefixed("hand_step_l2", finite_summary(l2(action_diff[:, 9:19]))))

    action_eef = action[:, ACTION_EEF_SLICE]
    action_hand = action[:, ACTION_HAND_SLICE]
    state_arm = state[:, STATE_ARM_SLICE]
    state_eef = state[:, STATE_EEF_SLICE]
    state_hand = state[:, STATE_HAND_SLICE]
    relative_chunks: dict[str, list[np.ndarray]] = {
        "eef_xyz_mm": [],
        "eef_rot_deg": [],
        "hand": [],
        "arm": [],
    }
    clipped_rows: list[dict[str, Any]] = []
    if bounds is not None:
        low_all, high_all = bounds
    else:
        low_all = high_all = np.zeros((max_horizon, 26), dtype=np.float64)

    max_episode_horizon = min(max_horizon, max(1, action.shape[0] - 1))
    for horizon in range(1, max_episode_horizon + 1):
        delta = horizon - 1
        groups = groot_relative_groups(
            action_eef,
            action_hand,
            state_arm,
            state_eef,
            state_hand,
            delta,
        )
        if groups["eef_9d"].shape[0] == 0:
            continue
        relative = concat_relative(groups)
        relative_chunks["eef_xyz_mm"].append(l2(groups["eef_9d"][:, :3]) * 1000.0)
        relative_chunks["eef_rot_deg"].append(rot6d_angle_deg(groups["eef_9d"][:, 3:9]))
        relative_chunks["hand"].append(l2(groups["hand_joint_target"]))
        relative_chunks["arm"].append(l2(groups["arm_joint_target"]))

        if bounds is None or horizon > low_all.shape[0]:
            continue
        normalized = normalize_q01_q99(relative, low_all[horizon - 1], high_all[horizon - 1])
        clipped_mask = np.abs(normalized) > 1.0
        start_indices, dims = np.nonzero(clipped_mask)
        frame_values = df["frame_index"].to_numpy() if "frame_index" in df else np.arange(action.shape[0])
        timestamps = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df else np.full(action.shape[0], np.nan)
        for start_idx, dim in zip(start_indices.tolist(), dims.tolist()):
            target_idx = start_idx + delta
            normalized_value = float(normalized[start_idx, dim])
            clipped_rows.append(
                {
                    "episode_index": int(episode_index),
                    "source_date": dates.get("source"),
                    "start_row": int(start_idx),
                    "start_frame_index": int(frame_values[start_idx]),
                    "start_timestamp": float(timestamps[start_idx]),
                    "horizon": int(horizon),
                    "target_row": int(target_idx),
                    "target_frame_index": int(frame_values[target_idx]),
                    "target_timestamp": float(timestamps[target_idx]),
                    "action_dim": int(dim),
                    "action_group": action_group(dim),
                    "action_name": DIM_NAMES[dim],
                    "relative_value": float(relative[start_idx, dim]),
                    "q01": float(low_all[horizon - 1, dim]),
                    "q99": float(high_all[horizon - 1, dim]),
                    "normalized_unclipped": normalized_value,
                    "clip_side": "high" if normalized_value > 1.0 else "low",
                    "excess_over_boundary": float(abs(normalized_value) - 1.0),
                }
            )

    for key, chunks in relative_chunks.items():
        values = np.concatenate(chunks, axis=0) if chunks else np.asarray([], dtype=np.float64)
        row.update(prefixed(f"relative_{key}", finite_summary(values)))

    return row, clipped_rows, {key: np.concatenate(chunks, axis=0) if chunks else np.asarray([]) for key, chunks in relative_chunks.items()}


def ffprobe_video(ffprobe: str, video_path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    return {
        "video_path": str(video_path),
        "exists": video_path.exists(),
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": stream.get("duration"),
        "browser_safe": stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p",
    }


def collect_video_checks(
    *,
    rows: list[dict[str, Any]],
    smooth_dir: Path,
    max_video_checks: int,
) -> list[dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return [{"status": "skipped", "reason": "ffprobe not found"}]
    checks: list[dict[str, Any]] = []
    limit = len(rows) if max_video_checks < 0 else min(max_video_checks, len(rows))
    for row in rows[:limit]:
        episode_index = smooth_episode_index(row)
        for video_key, path in video_paths(smooth_dir, row).items():
            try:
                check = ffprobe_video(ffprobe, path)
                check.update({"episode_index": episode_index, "video_key": video_key, "status": "ok"})
            except Exception as exc:
                check = {
                    "episode_index": episode_index,
                    "video_key": video_key,
                    "video_path": str(path),
                    "exists": path.exists(),
                    "status": "error",
                    "error": str(exc),
                    "browser_safe": False,
                }
            checks.append(check)
    return checks


def plot_episode_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    x = metrics["episode_index"].to_numpy()
    axes[0].bar(x, metrics["frames"], color="#4c78a8", alpha=0.85)
    axes[0].set_ylabel("frames")
    axes[0].set_title("Selected smooth episode lengths")
    axes[1].plot(x, metrics["action_step_l2_p95"], marker="o", color="#f58518")
    axes[1].set_ylabel("p95 L2")
    axes[1].set_title("Adjacent action step L2")
    axes[2].plot(x, metrics["relative_eef_xyz_mm_p95"], marker="o", label="eef xyz mm")
    axes[2].plot(x, metrics["relative_hand_p95"], marker="o", label="hand")
    axes[2].plot(x, metrics["relative_arm_p95"], marker="o", label="arm")
    axes[2].set_title("Relative action p95 by episode")
    axes[2].set_xlabel("smooth episode index")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_clipped_summary(clipped: pd.DataFrame, out_path: Path) -> None:
    if clipped.empty:
        return
    by_group = clipped.groupby(["action_group", "clip_side"]).size().unstack(fill_value=0)
    by_episode = clipped.groupby("episode_index").size().sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    by_group.plot(kind="bar", stacked=True, ax=axes[0], color=["#54a24b", "#e45756"])
    axes[0].set_title("q01/q99 clipped relative-action rows by group")
    axes[0].set_ylabel("rows")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(by_episode.index.to_numpy(), by_episode.to_numpy(), color="#b279a2", alpha=0.88)
    axes[1].set_title("q01/q99 clipped rows by episode")
    axes[1].set_xlabel("smooth episode index")
    axes[1].set_ylabel("rows")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_html_report(summary: dict[str, Any], out_dir: Path) -> None:
    clipped_plot = out_dir / "clipped_summary.png"
    clipped_section = (
        f'<h2>q01/q99 Clipped Relative Actions</h2><img src="{clipped_plot.name}" />'
        if clipped_plot.exists()
        else "<h2>q01/q99 Clipped Relative Actions</h2><p>No clipped rows found, or relative stats were unavailable.</p>"
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smooth Quality Report {summary['date']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    img {{ max-width: 1200px; width: 100%; border: 1px solid #d9dee7; border-radius: 6px; }}
    code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Smooth Quality Report</h1>
  <p>Date: <strong>{summary['date']}</strong> ({summary['date_field']})</p>
  <p>Dataset: <code>{summary['smooth_dir']}</code></p>
  <p>Episodes: {summary['episodes']} | frames: {summary['frames']} | clipped rows: {summary['clipped_rows']}</p>
  <h2>Episode Metrics</h2>
  <img src="episode_quality_summary.png" />
  {clipped_section}
  <h2>Files</h2>
  <ul>
    <li><code>episode_quality_metrics.csv</code></li>
    <li><code>q01_q99_clipped_relative_rows.csv</code></li>
    <li><code>video_checks.csv</code></li>
    <li><code>summary.json</code></li>
  </ul>
</body>
</html>
"""
    (out_dir / "smooth_quality_report.html").write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        raise ValueError("--date must be formatted as YYYY-MM-DD")

    smooth_dir = args.smooth_dir.expanduser().resolve()
    manifest_rows = read_jsonl(smooth_dir / "meta" / "trim_manifest.jsonl")
    if not manifest_rows:
        manifest_rows = read_jsonl(smooth_dir / "meta" / "source_trimmed_map.jsonl")
    selected_rows = [row for row in manifest_rows if row_matches_date(row, args.date, args.date_field)]
    if not selected_rows:
        raise RuntimeError(f"No smooth episodes matched date={args.date!r} date_field={args.date_field!r}")

    output_name = f"{args.date_field}_{args.date}"
    out_dir = args.output_root.expanduser().resolve() / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = load_relative_bounds(args.relative_stats.expanduser().resolve())
    max_horizon = min(MAX_HORIZON, max(1, int(args.max_horizon)))
    episode_rows: list[dict[str, Any]] = []
    clipped_rows: list[dict[str, Any]] = []

    for manifest_row in selected_rows:
        episode_index = smooth_episode_index(manifest_row)
        dates = row_dates(manifest_row)
        parquet_path = smooth_data_path(smooth_dir, manifest_row)
        episode_row, episode_clipped, _relative_chunks = episode_quality_rows(
            episode_index=episode_index,
            dates=dates,
            parquet_path=parquet_path,
            max_horizon=max_horizon,
            bounds=bounds,
        )
        episode_rows.append(episode_row)
        clipped_rows.extend(episode_clipped)

    episode_metrics = pd.DataFrame(episode_rows).sort_values("episode_index")
    clipped = pd.DataFrame(clipped_rows)
    video_checks = (
        []
        if args.skip_video_check
        else collect_video_checks(
            rows=selected_rows,
            smooth_dir=smooth_dir,
            max_video_checks=int(args.max_video_checks),
        )
    )

    episode_metrics.to_csv(out_dir / "episode_quality_metrics.csv", index=False)
    clipped.to_csv(out_dir / "q01_q99_clipped_relative_rows.csv", index=False)
    pd.DataFrame(video_checks).to_csv(out_dir / "video_checks.csv", index=False)
    plot_episode_metrics(episode_metrics, out_dir / "episode_quality_summary.png")
    plot_clipped_summary(clipped, out_dir / "clipped_summary.png")

    video_failures = [row for row in video_checks if row.get("status") != "ok" or row.get("browser_safe") is False]
    summary = {
        "date": args.date,
        "date_field": args.date_field,
        "smooth_dir": str(smooth_dir),
        "relative_stats": str(args.relative_stats.expanduser().resolve()) if args.relative_stats.exists() else None,
        "output_dir": str(out_dir),
        "episodes": int(len(episode_metrics)),
        "frames": int(episode_metrics["frames"].sum()),
        "max_horizon": int(max_horizon),
        "clipped_rows": int(len(clipped)),
        "video_checks": int(len(video_checks)),
        "video_failures": int(len(video_failures)),
        "tables": {
            "episode_metrics": str((out_dir / "episode_quality_metrics.csv").resolve()),
            "clipped_rows": str((out_dir / "q01_q99_clipped_relative_rows.csv").resolve()),
            "video_checks": str((out_dir / "video_checks.csv").resolve()),
        },
        "plots": {
            "episode_quality": str((out_dir / "episode_quality_summary.png").resolve()),
            "clipped_summary": str((out_dir / "clipped_summary.png").resolve()) if (out_dir / "clipped_summary.png").exists() else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html_report(summary, out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
