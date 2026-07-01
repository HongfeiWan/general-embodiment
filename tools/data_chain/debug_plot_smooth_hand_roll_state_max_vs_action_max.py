#!/usr/bin/env python3
"""Plot smooth hand roll state-max vs action-max distributions."""

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
DEFAULT_OUTPUT_DIR = MISSION_DIR / "quality" / "smooth_hand_roll_state_max_vs_action_max"
ROLL_SUFFIXES = (
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)
ROLL_COLORS = {
    "index_mcp_roll": "#2563eb",
    "ring_mcp_roll": "#059669",
    "pinky_mcp_roll": "#dc2626",
    "thumb_cmc_roll": "#f97316",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = (info.get("features") or {}).get(key)
    names = feature.get("names") if isinstance(feature, dict) else None
    return [str(name) for name in names] if isinstance(names, list) else []


def resolve_roll_indices(info: dict[str, Any]) -> dict[str, dict[str, int]]:
    state_names = feature_names(info, "observation.state")
    action_names = feature_names(info, "action")
    indices: dict[str, dict[str, int]] = {"state": {}, "action": {}}
    for suffix in ROLL_SUFFIXES:
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
        frames = int(min(len(state), len(action)))
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frames": frames,
            "data_path": str(data_path.relative_to(dataset_dir)),
        }
        for suffix in ROLL_SUFFIXES:
            state_roll = state[:frames, indices["state"][suffix]]
            action_roll = action[:frames, indices["action"][suffix]]
            row[f"state_{suffix}_max"] = finite_max(state_roll)
            row[f"action_{suffix}_max"] = finite_max(action_roll)
            row[f"action_minus_state_{suffix}_max"] = (
                row[f"action_{suffix}_max"] - row[f"state_{suffix}_max"]
            )
        rows.append(row)
    return rows


def add_y_equals_x(axis: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    lo = min(float(np.min(x[finite])), float(np.min(y[finite])))
    hi = max(float(np.max(x[finite])), float(np.max(y[finite])))
    axis.plot([lo, hi], [lo, hi], color="#667085", linestyle="--", linewidth=1.0, alpha=0.72, label="y=x")


def plot_one(axis: plt.Axes, df: pd.DataFrame, suffix: str) -> None:
    x = df[f"state_{suffix}_max"].to_numpy(dtype=np.float64)
    y = df[f"action_{suffix}_max"].to_numpy(dtype=np.float64)
    scatter = axis.scatter(
        x,
        y,
        c=df["episode_index"].to_numpy(dtype=np.float64),
        cmap="viridis",
        s=24,
        alpha=0.82,
        edgecolors="none",
    )
    add_y_equals_x(axis, x, y)
    axis.set_title(suffix)
    axis.set_xlabel("state max")
    axis.set_ylabel("action max")
    axis.grid(alpha=0.18)
    axis.legend(loc="best")
    return scatter


def write_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), constrained_layout=True)
    axes_flat = axes.ravel()
    scatter = None
    for axis, suffix in zip(axes_flat[:4], ROLL_SUFFIXES):
        scatter = plot_one(axis, df, suffix)

    combined_axis = axes_flat[4]
    all_x: list[float] = []
    all_y: list[float] = []
    for suffix in ROLL_SUFFIXES:
        x = df[f"state_{suffix}_max"].to_numpy(dtype=np.float64)
        y = df[f"action_{suffix}_max"].to_numpy(dtype=np.float64)
        all_x.extend(x[np.isfinite(x)].tolist())
        all_y.extend(y[np.isfinite(y)].tolist())
        combined_axis.scatter(
            x,
            y,
            s=24,
            alpha=0.76,
            color=ROLL_COLORS[suffix],
            edgecolors="none",
            label=suffix,
        )
    add_y_equals_x(combined_axis, np.asarray(all_x, dtype=np.float64), np.asarray(all_y, dtype=np.float64))
    combined_axis.set_title("all hand roll dimensions")
    combined_axis.set_xlabel("state max")
    combined_axis.set_ylabel("action max")
    combined_axis.grid(alpha=0.18)
    combined_axis.legend(loc="best", fontsize=8)

    delta_axis = axes_flat[5]
    episode = df["episode_index"].to_numpy(dtype=np.int64)
    for suffix in ROLL_SUFFIXES:
        delta_axis.scatter(
            episode,
            df[f"action_minus_state_{suffix}_max"],
            s=20,
            alpha=0.72,
            color=ROLL_COLORS[suffix],
            edgecolors="none",
            label=suffix,
        )
    delta_axis.axhline(0.0, color="#667085", linestyle="--", linewidth=1.0)
    delta_axis.set_title("action max - state max by episode")
    delta_axis.set_xlabel("episode_index")
    delta_axis.set_ylabel("action max - state max")
    delta_axis.grid(alpha=0.18)
    delta_axis.legend(loc="best", fontsize=8)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes_flat[:4].tolist(), shrink=0.82)
        cbar.set_label("episode_index")
    fig.suptitle("Smooth Dataset Hand Roll: State Max vs Action Max", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def stats_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {"episodes": int(len(df)), "rolls": {}}
    for suffix in ROLL_SUFFIXES:
        x = df[f"state_{suffix}_max"].to_numpy(dtype=np.float64)
        y = df[f"action_{suffix}_max"].to_numpy(dtype=np.float64)
        delta = y - x
        finite = np.isfinite(x) & np.isfinite(y)
        corr = float(np.corrcoef(x[finite], y[finite])[0, 1]) if np.count_nonzero(finite) >= 2 else None
        argmax_delta = int(np.nanargmax(delta)) if np.any(np.isfinite(delta)) else -1
        summary["rolls"][suffix] = {
            "state_max_mean": float(np.nanmean(x)) if np.any(np.isfinite(x)) else None,
            "state_max_max": float(np.nanmax(x)) if np.any(np.isfinite(x)) else None,
            "action_max_mean": float(np.nanmean(y)) if np.any(np.isfinite(y)) else None,
            "action_max_max": float(np.nanmax(y)) if np.any(np.isfinite(y)) else None,
            "action_minus_state_mean": float(np.nanmean(delta)) if np.any(np.isfinite(delta)) else None,
            "action_minus_state_max": float(np.nanmax(delta)) if np.any(np.isfinite(delta)) else None,
            "argmax_delta_episode_index": int(df.iloc[argmax_delta]["episode_index"]) if argmax_delta >= 0 else None,
            "state_action_corr": corr,
        }
    return summary


def fmt_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return html.escape(str(value))


def write_html(output_path: Path, plot_path: Path, csv_path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for suffix, stats in summary["rolls"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(suffix)}</td>"
            f"<td>{fmt_cell(stats['state_max_mean'])}</td>"
            f"<td>{fmt_cell(stats['state_max_max'])}</td>"
            f"<td>{fmt_cell(stats['action_max_mean'])}</td>"
            f"<td>{fmt_cell(stats['action_max_max'])}</td>"
            f"<td>{fmt_cell(stats['action_minus_state_mean'])}</td>"
            f"<td>{fmt_cell(stats['action_minus_state_max'])}</td>"
            f"<td>{fmt_cell(stats['argmax_delta_episode_index'])}</td>"
            f"<td>{fmt_cell(stats['state_action_corr'])}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Smooth Hand Roll State Max vs Action Max</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; background: #fbfcfd; }}
    img {{ width: 100%; max-width: 1500px; border: 1px solid #d7dde5; border-radius: 6px; background: #fff; }}
    table {{ border-collapse: collapse; margin-top: 18px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Smooth Hand Roll State Max vs Action Max</h1>
  <p>x-axis is per-episode state max; y-axis is per-episode action max. CSV: <code>{html.escape(csv_path.name)}</code>.</p>
  <img src="{html.escape(plot_path.name)}" alt="hand roll state max vs action max scatter">
  <table>
    <tr><th>roll</th><th>state max mean</th><th>state max max</th><th>action max mean</th><th>action max max</th><th>delta mean</th><th>delta max</th><th>argmax delta episode</th><th>corr</th></tr>
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
    indices = resolve_roll_indices(info)
    rows = episode_rows(dataset_dir, info, indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hand_roll_state_max_vs_action_max.csv"
    plot_path = output_dir / "hand_roll_state_max_vs_action_max.png"
    html_path = output_dir / "hand_roll_state_max_vs_action_max.html"
    summary_path = output_dir / "hand_roll_state_max_vs_action_max_summary.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_plot(rows, plot_path)
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
