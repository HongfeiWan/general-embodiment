#!/usr/bin/env python3
"""Plot trimmed-vs-smooth overlays for the 19D Nero mission2 action commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
GROUPS = {
    "eef_xyz": (0, 3, "#1f77b4"),
    "eef_rot6d": (3, 9, "#2ca02c"),
    "hand": (9, 19, "#d62728"),
}


def _short_name(name: str) -> str:
    return (
        name.replace("arm_eef_pos_target.", "eef.")
        .replace("arm_eef_rot6d_target.", "rot6d.")
        .replace("hand_joint_target.", "hand.")
    )


def _load_action(parquet_path: Path) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["action"])
    return np.stack(df["action"].to_numpy()).astype(np.float64)


def _plot_episode(
    episode_index: int,
    raw: np.ndarray,
    smooth: np.ndarray,
    action_names: list[str],
    out_path: Path,
) -> dict[str, float | int]:
    diff = smooth - raw
    fig, axes = plt.subplots(5, 4, figsize=(20, 14), sharex=True)
    axes_flat = axes.reshape(-1)
    x = np.arange(raw.shape[0])

    for dim in range(raw.shape[1]):
        ax = axes_flat[dim]
        group_color = next(color for _, (start, end, color) in GROUPS.items() if start <= dim < end)
        ax.plot(x, raw[:, dim], color="#9a9a9a", linewidth=1.0, alpha=0.75, label="trimmed")
        ax.plot(x, smooth[:, dim], color=group_color, linewidth=1.25, alpha=0.95, label="smooth")
        ax.fill_between(x, raw[:, dim], smooth[:, dim], color=group_color, alpha=0.10)
        ax.set_title(f"{dim:02d} {_short_name(action_names[dim])}", fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[raw.shape[1] :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False)
    fig.suptitle(
        (
            f"Episode {episode_index:06d} action overlay: trimmed vs smooth "
            f"| mean_abs={np.mean(np.abs(diff)):.6f}, max_abs={np.max(np.abs(diff)):.6f}"
        ),
        fontsize=14,
    )
    fig.supxlabel("frame index")
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    metrics: dict[str, float | int] = {
        "episode_index": int(episode_index),
        "frames": int(raw.shape[0]),
        "mean_abs_change": float(np.mean(np.abs(diff))),
        "max_abs_change": float(np.max(np.abs(diff))),
        "rms_change": float(np.sqrt(np.mean(diff**2))),
    }
    for group, (start, end, _) in GROUPS.items():
        metrics[f"{group}_mean_abs_change"] = float(np.mean(np.abs(diff[:, start:end])))
        metrics[f"{group}_max_abs_change"] = float(np.max(np.abs(diff[:, start:end])))
    return metrics


def _plot_all_episode_summary(
    raw_all: np.ndarray,
    smooth_all: np.ndarray,
    action_names: list[str],
    out_path: Path,
) -> pd.DataFrame:
    diff = smooth_all - raw_all
    per_dim = pd.DataFrame(
        {
            "dim": np.arange(raw_all.shape[1], dtype=int),
            "name": action_names,
            "mean_abs_change": np.mean(np.abs(diff), axis=0),
            "max_abs_change": np.max(np.abs(diff), axis=0),
            "rms_change": np.sqrt(np.mean(diff**2, axis=0)),
            "raw_std": np.std(raw_all, axis=0),
            "smooth_std": np.std(smooth_all, axis=0),
        }
    )

    colors = []
    for dim in per_dim["dim"]:
        colors.append(next(color for _, (start, end, color) in GROUPS.items() if start <= dim < end))

    labels = [f"{int(row.dim):02d}\n{_short_name(str(row.name))}" for row in per_dim.itertuples()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    axes[0].bar(x, per_dim["mean_abs_change"], color=colors, alpha=0.85)
    axes[0].set_ylabel("mean |smooth-trimmed|")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, per_dim["max_abs_change"], color=colors, alpha=0.85)
    axes[1].set_ylabel("max |smooth-trimmed|")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(x, per_dim["raw_std"], color="#9a9a9a", alpha=0.65, label="trimmed std")
    axes[2].bar(x, per_dim["smooth_std"], color="#111111", alpha=0.35, label="smooth std")
    axes[2].set_ylabel("std over all frames")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=55, ha="right", fontsize=8)

    fig.suptitle("All episodes action smoothing summary", fontsize=14)
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return per_dim


def _plot_concatenated_overlay(
    raw_all: np.ndarray,
    smooth_all: np.ndarray,
    action_names: list[str],
    episode_boundaries: list[int],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(22, 15), sharex=True)
    axes_flat = axes.reshape(-1)
    x = np.arange(raw_all.shape[0])
    boundaries = np.asarray(episode_boundaries[1:-1], dtype=int)

    for dim in range(raw_all.shape[1]):
        ax = axes_flat[dim]
        group_color = next(color for _, (start, end, color) in GROUPS.items() if start <= dim < end)
        ax.plot(x, raw_all[:, dim], color="#b0b0b0", linewidth=0.55, alpha=0.65)
        ax.plot(x, smooth_all[:, dim], color=group_color, linewidth=0.65, alpha=0.90)
        for boundary in boundaries:
            ax.axvline(boundary, color="#000000", alpha=0.08, linewidth=0.6)
        ax.set_title(f"{dim:02d} {_short_name(action_names[dim])}", fontsize=9)
        ax.grid(True, alpha=0.20, linewidth=0.5)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[raw_all.shape[1] :]:
        ax.axis("off")

    fig.suptitle("All episodes concatenated action overlay: trimmed gray, smooth colored", fontsize=14)
    fig.supxlabel("concatenated frame index")
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trimmed", type=Path, default=MISSION_DIR / "trimmed")
    parser.add_argument("--smooth", type=Path, default=MISSION_DIR / "smooth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MISSION_DIR / "action_smooth_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = args.output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((args.trimmed / "meta" / "info.json").read_text(encoding="utf-8"))
    action_names = info["features"]["action"]["names"]

    trimmed_paths = sorted((args.trimmed / "data").rglob("*.parquet"))
    smooth_paths = sorted((args.smooth / "data").rglob("*.parquet"))
    if len(trimmed_paths) != len(smooth_paths):
        raise RuntimeError(f"Episode count mismatch: {len(trimmed_paths)} vs {len(smooth_paths)}")

    episode_metrics = []
    raw_chunks = []
    smooth_chunks = []
    episode_boundaries = [0]

    for trimmed_path, smooth_path in zip(trimmed_paths, smooth_paths):
        episode_index = int(trimmed_path.stem.split("_")[-1])
        raw = _load_action(trimmed_path)
        smooth = _load_action(smooth_path)
        if raw.shape != smooth.shape:
            raise RuntimeError(f"Shape mismatch for episode {episode_index}: {raw.shape} vs {smooth.shape}")
        if raw.shape[1] != 19:
            raise RuntimeError(f"Expected 19D action, got {raw.shape[1]}D in {trimmed_path}")

        out_path = episode_dir / f"episode_{episode_index:06d}_action_overlay.png"
        metrics = _plot_episode(episode_index, raw, smooth, action_names, out_path)
        metrics["plot"] = str(out_path)
        episode_metrics.append(metrics)
        raw_chunks.append(raw)
        smooth_chunks.append(smooth)
        episode_boundaries.append(episode_boundaries[-1] + raw.shape[0])

    raw_all = np.concatenate(raw_chunks, axis=0)
    smooth_all = np.concatenate(smooth_chunks, axis=0)
    per_dim = _plot_all_episode_summary(
        raw_all,
        smooth_all,
        action_names,
        args.output_dir / "action_smoothing_per_dim_summary.png",
    )
    _plot_concatenated_overlay(
        raw_all,
        smooth_all,
        action_names,
        episode_boundaries,
        args.output_dir / "all_episodes_action_overlay.png",
    )

    per_dim.to_csv(args.output_dir / "action_smoothing_per_dim_metrics.csv", index=False)
    pd.DataFrame(episode_metrics).to_csv(args.output_dir / "action_smoothing_episode_metrics.csv", index=False)

    summary = {
        "trimmed": str(args.trimmed.resolve()),
        "smooth": str(args.smooth.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "episodes": len(episode_metrics),
        "frames": int(raw_all.shape[0]),
        "action_dim": int(raw_all.shape[1]),
        "mean_abs_change": float(np.mean(np.abs(smooth_all - raw_all))),
        "max_abs_change": float(np.max(np.abs(smooth_all - raw_all))),
        "episode_plot_dir": str(episode_dir.resolve()),
        "summary_plot": str((args.output_dir / "action_smoothing_per_dim_summary.png").resolve()),
        "concatenated_overlay": str((args.output_dir / "all_episodes_action_overlay.png").resolve()),
    }
    (args.output_dir / "action_smoothing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
