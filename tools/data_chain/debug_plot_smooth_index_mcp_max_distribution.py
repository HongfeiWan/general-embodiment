#!/usr/bin/env python3
"""Plot per-episode max-value distributions for smooth index_mcp hand joints."""

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
DEFAULT_OUTPUT_DIR = MISSION_DIR / "quality" / "smooth_index_mcp_max_distribution"
INDEX_MCP_SUFFIXES = ("index_mcp_pitch", "index_mcp_roll")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bins",
        type=int,
        default=36,
        help="Histogram bin count.",
    )
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


def resolve_indices(info: dict[str, Any]) -> dict[str, dict[str, int]]:
    state_names = feature_names(info, "observation.state")
    action_names = feature_names(info, "action")
    indices: dict[str, dict[str, int]] = {"state": {}, "action": {}}
    for suffix in INDEX_MCP_SUFFIXES:
        state_name = f"hand_joint_pos.{suffix}"
        action_name = f"hand_joint_target.{suffix}"
        if state_name not in state_names:
            raise RuntimeError(f"Missing observation.state feature: {state_name}")
        if action_name not in action_names:
            raise RuntimeError(f"Missing action feature: {action_name}")
        indices["state"][suffix] = state_names.index(state_name)
        indices["action"][suffix] = action_names.index(action_name)
    return indices


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


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def episode_rows(dataset_dir: Path, info: dict[str, Any], indices: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes.sort(key=lambda row: int(row.get("episode_index", 0)))
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        data_path = data_path_for_episode(dataset_dir, info, episode)
        df = pd.read_parquet(data_path, columns=["observation.state", "action"])
        state = stack_vector_column(df["observation.state"], "observation.state")
        action = stack_vector_column(df["action"], "action")
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frames": int(min(len(state), len(action))),
            "data_path": str(data_path.relative_to(dataset_dir)),
        }
        for suffix in INDEX_MCP_SUFFIXES:
            state_values = state[:, indices["state"][suffix]]
            action_values = action[:, indices["action"][suffix]]
            row[f"state_{suffix}_max"] = finite_max(state_values)
            row[f"action_{suffix}_max"] = finite_max(action_values)
            row[f"state_action_{suffix}_max_delta"] = (
                row[f"action_{suffix}_max"] - row[f"state_{suffix}_max"]
            )
        rows.append(row)
    return rows


def write_plot(rows: list[dict[str, Any]], output_path: Path, bins: int) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    pairs = [
        ("index_mcp_pitch", axes[0, 0]),
        ("index_mcp_roll", axes[0, 1]),
    ]
    colors = {
        "state": "#2563eb",
        "action": "#dc2626",
    }
    for suffix, axis in pairs:
        state_col = f"state_{suffix}_max"
        action_col = f"action_{suffix}_max"
        state_values = df[state_col].to_numpy(dtype=np.float64)
        action_values = df[action_col].to_numpy(dtype=np.float64)
        axis.hist(state_values[np.isfinite(state_values)], bins=bins, alpha=0.62, label="state max", color=colors["state"])
        axis.hist(action_values[np.isfinite(action_values)], bins=bins, alpha=0.52, label="action max", color=colors["action"])
        axis.set_title(f"{suffix} per-episode max distribution")
        axis.set_xlabel("max value")
        axis.set_ylabel("episodes")
        axis.grid(alpha=0.18)
        axis.legend()

    for suffix, axis in (("index_mcp_pitch", axes[1, 0]), ("index_mcp_roll", axes[1, 1])):
        state_col = f"state_{suffix}_max"
        action_col = f"action_{suffix}_max"
        delta_col = f"state_action_{suffix}_max_delta"
        episode = df["episode_index"].to_numpy(dtype=np.int64)
        axis.scatter(episode, df[state_col], s=14, color=colors["state"], alpha=0.78, label="state max")
        axis.scatter(episode, df[action_col], s=14, color=colors["action"], alpha=0.72, label="action max")
        twin = axis.twinx()
        twin.plot(episode, df[delta_col], color="#7c3aed", linewidth=1.2, alpha=0.65, label="action-state max delta")
        axis.set_title(f"{suffix} max by episode")
        axis.set_xlabel("episode_index")
        axis.set_ylabel("max value")
        twin.set_ylabel("action - state max")
        axis.grid(alpha=0.18)
        lines, labels = axis.get_legend_handles_labels()
        twin_lines, twin_labels = twin.get_legend_handles_labels()
        axis.legend(lines + twin_lines, labels + twin_labels, loc="best")

    fig.suptitle("Smooth Dataset index_mcp Max Value Distribution", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def stats_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {"episodes": int(len(df)), "columns": {}}
    for suffix in INDEX_MCP_SUFFIXES:
        for prefix in ("state", "action"):
            column = f"{prefix}_{suffix}_max"
            values = df[column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary["columns"][column] = {
                "mean": float(np.mean(finite)) if finite.size else None,
                "p50": float(np.quantile(finite, 0.50)) if finite.size else None,
                "p95": float(np.quantile(finite, 0.95)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
                "argmax_episode_index": int(df.iloc[int(np.nanargmax(values))]["episode_index"]) if finite.size else None,
            }
    return summary


def write_html(output_path: Path, plot_path: Path, csv_path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for column, stats in summary["columns"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(column)}</td>"
            f"<td>{stats['mean']:.6g}</td>"
            f"<td>{stats['p50']:.6g}</td>"
            f"<td>{stats['p95']:.6g}</td>"
            f"<td>{stats['max']:.6g}</td>"
            f"<td>{stats['argmax_episode_index']}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Smooth index_mcp Max Distribution</title>
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
  <h1>Smooth index_mcp Max Distribution</h1>
  <p>Episodes: <code>{summary['episodes']}</code>. CSV: <code>{html.escape(csv_path.name)}</code>.</p>
  <img src="{html.escape(plot_path.name)}" alt="index_mcp max distribution">
  <table>
    <tr><th>column</th><th>mean</th><th>p50</th><th>p95</th><th>max</th><th>argmax episode</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    info = read_json(dataset_dir / "meta" / "info.json")
    indices = resolve_indices(info)
    rows = episode_rows(dataset_dir, info, indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "index_mcp_max_by_episode.csv"
    plot_path = output_dir / "index_mcp_max_distribution.png"
    html_path = output_dir / "index_mcp_max_distribution.html"
    summary_path = output_dir / "index_mcp_max_summary.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_plot(rows, plot_path, bins=max(1, int(args.bins)))
    summary = stats_summary(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(html_path, plot_path, csv_path, summary)

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "episodes": len(rows),
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
