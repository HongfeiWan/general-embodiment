#!/usr/bin/env python3
"""Plot smooth index_mcp_roll first-10 mean vs max for all episodes."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_DATASET_DIR = MISSION_DIR / "smooth"
DEFAULT_OUTPUT_DIR = MISSION_DIR / "quality" / "smooth_index_mcp_roll_first10_mean_vs_max"
ROLL_SUFFIX = "index_mcp_roll"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--leading-frames", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = (info.get("features") or {}).get(key)
    names = feature.get("names") if isinstance(feature, dict) else None
    return [str(name) for name in names] if isinstance(names, list) else []


def resolve_roll_indices(info: dict[str, Any]) -> tuple[int, int]:
    state_names = feature_names(info, "observation.state")
    action_names = feature_names(info, "action")
    state_name = f"hand_joint_pos.{ROLL_SUFFIX}"
    action_name = f"hand_joint_target.{ROLL_SUFFIX}"
    if state_name not in state_names:
        raise RuntimeError(f"Missing observation.state feature: {state_name}")
    if action_name not in action_names:
        raise RuntimeError(f"Missing action feature: {action_name}")
    return state_names.index(state_name), action_names.index(action_name)


def data_path_for_episode(dataset_dir: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    metadata = episode.get("teleop_stack_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("data_path"), str):
        path = dataset_dir / str(metadata["data_path"])
        if path.exists():
            return path
    episode_index = int(episode["episode_index"])
    chunks_size = max(1, int(info.get("chunks_size", 1000)))
    pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    path = dataset_dir / pattern.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)
    if path.exists():
        return path
    matches = sorted(dataset_dir.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing parquet for episode {episode_index}: {path}")


def stack_vector_column(series: pd.Series, column: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise RuntimeError(f"Inconsistent vector width in {column}")
    return np.vstack(values)


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def episode_rows(
    dataset_dir: Path,
    info: dict[str, Any],
    state_index: int,
    action_index: int,
    leading_frames: int,
) -> list[dict[str, Any]]:
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes.sort(key=lambda row: int(row.get("episode_index", 0)))
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        data_path = data_path_for_episode(dataset_dir, info, episode)
        df = pd.read_parquet(data_path, columns=["observation.state", "action"])
        state = stack_vector_column(df["observation.state"], "observation.state")
        action = stack_vector_column(df["action"], "action")
        frames = int(min(len(state), len(action)))
        state_roll = state[:frames, state_index]
        action_roll = action[:frames, action_index]
        leading = max(1, min(int(leading_frames), frames))
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frames": frames,
            "leading_frames": leading,
            "data_path": str(data_path.relative_to(dataset_dir)),
            "state_first10_mean": finite_mean(state_roll[:leading]),
            "state_max": finite_max(state_roll),
            "action_first10_mean": finite_mean(action_roll[:leading]),
            "action_max": finite_max(action_roll),
        }
        row["state_action_first10_mean_delta"] = row["action_first10_mean"] - row["state_first10_mean"]
        row["state_action_max_delta"] = row["action_max"] - row["state_max"]
        rows.append(row)
    return rows


def add_reference_lines(axis: plt.Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return
    mins = [float(np.min(x_values[finite])), float(np.min(y_values[finite]))]
    maxs = [float(np.max(x_values[finite])), float(np.max(y_values[finite]))]
    lo = min(mins)
    hi = max(maxs)
    axis.plot([lo, hi], [lo, hi], color="#667085", linewidth=1.0, linestyle="--", alpha=0.7, label="y=x")


def write_plot(rows: list[dict[str, Any]], output_path: Path, leading_frames: int) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    specs = [
        ("state", axes[0], "#2563eb"),
        ("action", axes[1], "#dc2626"),
    ]
    for prefix, axis, color in specs:
        x = df[f"{prefix}_first10_mean"].to_numpy(dtype=np.float64)
        y = df[f"{prefix}_max"].to_numpy(dtype=np.float64)
        scatter = axis.scatter(
            x,
            y,
            c=df["episode_index"].to_numpy(dtype=np.float64),
            cmap="viridis",
            s=26,
            alpha=0.82,
            edgecolors="none",
            label=prefix,
        )
        add_reference_lines(axis, x, y)
        axis.set_title(f"{prefix} {ROLL_SUFFIX}: first {leading_frames} mean vs max")
        axis.set_xlabel(f"first {leading_frames} frames mean")
        axis.set_ylabel("episode max")
        axis.grid(alpha=0.18)
        axis.legend(loc="best")
        cbar = fig.colorbar(scatter, ax=axis)
        cbar.set_label("episode_index")
    fig.suptitle("Smooth Dataset index_mcp_roll Leading Mean vs Episode Max", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def stats_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {"episodes": int(len(df)), "columns": {}}
    for prefix in ("state", "action"):
        x_col = f"{prefix}_first10_mean"
        y_col = f"{prefix}_max"
        x = df[x_col].to_numpy(dtype=np.float64)
        y = df[y_col].to_numpy(dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            argmax_y = int(np.nanargmax(y))
            corr = float(np.corrcoef(x[finite], y[finite])[0, 1]) if np.count_nonzero(finite) >= 2 else None
        else:
            argmax_y = -1
            corr = None
        summary["columns"][prefix] = {
            "x_mean": float(np.nanmean(x)) if np.any(np.isfinite(x)) else None,
            "x_min": float(np.nanmin(x)) if np.any(np.isfinite(x)) else None,
            "x_max": float(np.nanmax(x)) if np.any(np.isfinite(x)) else None,
            "y_mean": float(np.nanmean(y)) if np.any(np.isfinite(y)) else None,
            "y_min": float(np.nanmin(y)) if np.any(np.isfinite(y)) else None,
            "y_max": float(np.nanmax(y)) if np.any(np.isfinite(y)) else None,
            "y_argmax_episode_index": int(df.iloc[argmax_y]["episode_index"]) if argmax_y >= 0 else None,
            "x_y_corr": corr,
        }
    return summary


def write_html(output_path: Path, plot_path: Path, csv_path: Path, summary: dict[str, Any], leading_frames: int) -> None:
    stat_rows = []
    for prefix, stats in summary["columns"].items():
        corr_text = "" if stats["x_y_corr"] is None else f"{stats['x_y_corr']:.6g}"
        stat_rows.append(
            "<tr>"
            f"<td>{html.escape(prefix)}</td>"
            f"<td>{stats['x_min']:.6g}</td>"
            f"<td>{stats['x_mean']:.6g}</td>"
            f"<td>{stats['x_max']:.6g}</td>"
            f"<td>{stats['y_min']:.6g}</td>"
            f"<td>{stats['y_mean']:.6g}</td>"
            f"<td>{stats['y_max']:.6g}</td>"
            f"<td>{stats['y_argmax_episode_index']}</td>"
            f"<td>{corr_text}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Smooth index_mcp_roll First-{leading_frames} Mean vs Max</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; background: #fbfcfd; }}
    img {{ width: 100%; max-width: 1400px; border: 1px solid #d7dde5; border-radius: 6px; background: #fff; }}
    table {{ border-collapse: collapse; margin-top: 18px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Smooth index_mcp_roll First-{leading_frames} Mean vs Max</h1>
  <p>x-axis is the mean of the first <code>{leading_frames}</code> frames in each episode; y-axis is the episode maximum. CSV: <code>{html.escape(csv_path.name)}</code>.</p>
  <img src="{html.escape(plot_path.name)}" alt="index_mcp_roll first mean vs max scatter">
  <table>
    <tr><th>series</th><th>x min</th><th>x mean</th><th>x max</th><th>y min</th><th>y mean</th><th>y max</th><th>y argmax episode</th><th>corr</th></tr>
    {''.join(stat_rows)}
  </table>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    leading_frames = max(1, int(args.leading_frames))
    info = read_json(dataset_dir / "meta" / "info.json")
    state_index, action_index = resolve_roll_indices(info)
    rows = episode_rows(dataset_dir, info, state_index, action_index, leading_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "index_mcp_roll_first10_mean_vs_max.csv"
    plot_path = output_dir / "index_mcp_roll_first10_mean_vs_max.png"
    html_path = output_dir / "index_mcp_roll_first10_mean_vs_max.html"
    summary_path = output_dir / "index_mcp_roll_first10_mean_vs_max_summary.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_plot(rows, plot_path, leading_frames)
    summary = stats_summary(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(html_path, plot_path, csv_path, summary, leading_frames)

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "episodes": len(rows),
                "leading_frames": leading_frames,
                "csv": str(csv_path),
                "plot": str(plot_path),
                "html": str(html_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
