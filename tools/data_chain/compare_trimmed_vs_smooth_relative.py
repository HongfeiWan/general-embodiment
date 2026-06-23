#!/usr/bin/env python3
"""Compare config-relative actions between trimmed and smooth mission2 datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config_relative_action_curves import (
    HORIZONS,
    l2,
    read_episode,
    relative_groups,
    rot6d_angle_deg,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
RELATIVE_NAMES = (
    [
        "eef_9d.x",
        "eef_9d.y",
        "eef_9d.z",
        "eef_9d.rot6d.r00",
        "eef_9d.rot6d.r10",
        "eef_9d.rot6d.r20",
        "eef_9d.rot6d.r01",
        "eef_9d.rot6d.r11",
        "eef_9d.rot6d.r21",
    ]
    + [
        "hand.thumb_cmc_pitch",
        "hand.thumb_cmc_yaw",
        "hand.index_mcp_pitch",
        "hand.middle_mcp_pitch",
        "hand.ring_mcp_pitch",
        "hand.pinky_mcp_pitch",
        "hand.index_mcp_roll",
        "hand.ring_mcp_roll",
        "hand.pinky_mcp_roll",
        "hand.thumb_cmc_roll",
    ]
    + [f"arm_joint.{idx}" for idx in range(7)]
)


GROUP_SLICES = {
    "eef_9d": slice(0, 9),
    "eef_xyz": slice(0, 3),
    "eef_rot6d": slice(3, 9),
    "hand_joint_target": slice(9, 19),
    "arm_joint_target": slice(19, 26),
}


GROUP_COLORS = {
    "eef_9d": "#1f77b4",
    "eef_xyz": "#1f77b4",
    "eef_rot6d": "#2ca02c",
    "hand_joint_target": "#d62728",
    "arm_joint_target": "#9467bd",
}


def finite_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {key: math.nan for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def concat_relative(groups: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            groups["eef_9d"],
            groups["hand_joint_target"],
            groups["arm_joint_target"],
        ],
        axis=1,
    )


def compute_episode_relative(path: Path, horizon: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    action_eef, action_hand, state_arm, _timestamps = read_episode(path)
    groups = relative_groups(action_eef, action_hand, state_arm, horizon)
    return concat_relative(groups), groups


def plot_h1_overlay(
    *,
    episode_index: int,
    trimmed_rel: np.ndarray,
    smooth_rel: np.ndarray,
    out_path: Path,
) -> None:
    diff = smooth_rel - trimmed_rel
    fig, axes = plt.subplots(6, 5, figsize=(24, 18), sharex=True)
    axes_flat = axes.reshape(-1)
    x = np.arange(trimmed_rel.shape[0])

    for dim in range(trimmed_rel.shape[1]):
        ax = axes_flat[dim]
        if dim < 9:
            color = GROUP_COLORS["eef_9d"]
        elif dim < 19:
            color = GROUP_COLORS["hand_joint_target"]
        else:
            color = GROUP_COLORS["arm_joint_target"]
        ax.plot(x, trimmed_rel[:, dim], color="#9a9a9a", linewidth=1.0, alpha=0.72, label="trimmed")
        ax.plot(x, smooth_rel[:, dim], color=color, linewidth=1.2, alpha=0.95, label="smooth")
        ax.fill_between(x, trimmed_rel[:, dim], smooth_rel[:, dim], color=color, alpha=0.10)
        ax.set_title(f"{dim:02d} {RELATIVE_NAMES[dim]}", fontsize=8)
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[trimmed_rel.shape[1] :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False)
    fig.suptitle(
        (
            f"Episode {episode_index:06d} h1 config-relative action overlay "
            f"| mean_abs_diff={np.mean(np.abs(diff)):.6f}, max_abs_diff={np.max(np.abs(diff)):.6f}"
        ),
        fontsize=14,
    )
    fig.supxlabel("start timestep t")
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h1_group_curves(
    *,
    episode_index: int,
    trimmed_groups: dict[str, np.ndarray],
    smooth_groups: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 14), constrained_layout=True)
    x = np.arange(trimmed_groups["eef_9d"].shape[0])

    series = [
        (
            "EEF h1 relative XYZ L2 (mm)",
            l2(trimmed_groups["eef_9d"][:, :3]) * 1000.0,
            l2(smooth_groups["eef_9d"][:, :3]) * 1000.0,
            "mm",
        ),
        (
            "EEF h1 relative rotation angle (deg)",
            rot6d_angle_deg(trimmed_groups["eef_9d"][:, 3:9]),
            rot6d_angle_deg(smooth_groups["eef_9d"][:, 3:9]),
            "deg",
        ),
        (
            "Hand h1 relative target L2",
            l2(trimmed_groups["hand_joint_target"]),
            l2(smooth_groups["hand_joint_target"]),
            "L2",
        ),
        (
            "Arm h1 relative alias L2",
            l2(trimmed_groups["arm_joint_target"]),
            l2(smooth_groups["arm_joint_target"]),
            "L2",
        ),
    ]

    for row, (title, trimmed_values, smooth_values, ylabel) in enumerate(series):
        ax = axes[row, 0]
        ax.plot(x, trimmed_values, color="#9a9a9a", linewidth=1.0, alpha=0.75, label="trimmed")
        ax.plot(x, smooth_values, color="tab:blue", linewidth=1.2, alpha=0.95, label="smooth")
        ax.fill_between(x, trimmed_values, smooth_values, color="tab:blue", alpha=0.10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        diff_ax = axes[row, 1]
        diff_ax.plot(x, smooth_values - trimmed_values, color="tab:red", linewidth=1.0)
        diff_ax.axhline(0.0, color="#333333", alpha=0.35, linewidth=0.8)
        diff_ax.set_title(f"{title}: smooth - trimmed")
        diff_ax.set_ylabel(ylabel)
        diff_ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("start timestep t")

    fig.suptitle(f"Episode {episode_index:06d} h1 group-relative comparison", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h1_summary(per_dim: pd.DataFrame, episode_metrics: pd.DataFrame, out_dir: Path) -> None:
    colors = []
    for dim in per_dim["dim"]:
        if dim < 9:
            colors.append(GROUP_COLORS["eef_9d"])
        elif dim < 19:
            colors.append(GROUP_COLORS["hand_joint_target"])
        else:
            colors.append(GROUP_COLORS["arm_joint_target"])
    labels = [f"{int(row.dim):02d}\n{row.name}" for row in per_dim.itertuples()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=False)
    axes[0].bar(x, per_dim["h1_mean_abs_diff"], color=colors, alpha=0.85)
    axes[0].set_title("h1 relative 26D: per-dimension mean |smooth - trimmed|")
    axes[0].set_ylabel("mean abs diff")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=55, ha="right", fontsize=8)

    axes[1].bar(x, per_dim["h1_max_abs_diff"], color=colors, alpha=0.85)
    axes[1].set_title("h1 relative 26D: per-dimension max |smooth - trimmed|")
    axes[1].set_ylabel("max abs diff")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=55, ha="right", fontsize=8)

    epi_x = episode_metrics["episode_index"].to_numpy()
    axes[2].plot(
        epi_x,
        episode_metrics["h1_eef_9d_l2_diff_p95"],
        marker="o",
        label="eef_9d p95 diff",
    )
    axes[2].plot(
        epi_x,
        episode_metrics["h1_hand_joint_target_l2_diff_p95"],
        marker="o",
        label="hand p95 diff",
    )
    axes[2].plot(
        epi_x,
        episode_metrics["h1_arm_joint_target_l2_diff_p95"],
        marker="o",
        label="arm p95 diff",
    )
    axes[2].set_title("h1 group L2 diff p95 by episode")
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("p95 L2 diff")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / "h1_relative_diff_summary.png", dpi=150)
    plt.close(fig)


def plot_horizon_summary(horizon_metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    groups = [
        ("eef_xyz_mm", "EEF XYZ diff p95 (mm)"),
        ("eef_rot_angle_deg", "EEF rotation angle diff p95 (deg)"),
        ("hand_joint_target", "Hand relative diff p95"),
        ("arm_joint_target", "Arm relative diff p95"),
    ]
    for ax, (key, title) in zip(axes, groups):
        subset = horizon_metrics[horizon_metrics["group"] == key]
        ax.plot(subset["horizon"], subset["l2_diff_p50"], marker="o", label="p50")
        ax.plot(subset["horizon"], subset["l2_diff_p95"], marker="o", label="p95")
        ax.plot(subset["horizon"], subset["l2_diff_max"], marker="o", label="max")
        ax.set_title(title)
        ax.set_ylabel("diff")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("horizon")
    fig.savefig(out_dir / "relative_diff_by_horizon.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trimmed", type=Path, default=MISSION_DIR / "trimmed")
    parser.add_argument("--smooth", type=Path, default=MISSION_DIR / "smooth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MISSION_DIR / "relative_smooth_comparison",
    )
    parser.add_argument("--max-episode-plots", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    episode_dir = out_dir / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)

    trimmed_paths = sorted((args.trimmed / "data").glob("chunk-*/*.parquet"))
    smooth_paths = sorted((args.smooth / "data").glob("chunk-*/*.parquet"))
    if len(trimmed_paths) != len(smooth_paths):
        raise RuntimeError(f"Episode count mismatch: {len(trimmed_paths)} vs {len(smooth_paths)}")

    plot_limit = len(trimmed_paths) if args.max_episode_plots < 0 else args.max_episode_plots
    episode_rows: list[dict[str, float | int | str]] = []
    h1_trimmed_chunks: list[np.ndarray] = []
    h1_smooth_chunks: list[np.ndarray] = []
    horizon_values: dict[tuple[int, str], list[np.ndarray]] = {}

    for idx, (trimmed_path, smooth_path) in enumerate(zip(trimmed_paths, smooth_paths)):
        episode_index = int(trimmed_path.stem.split("_")[-1])
        trimmed_h1, trimmed_h1_groups = compute_episode_relative(trimmed_path, horizon=1)
        smooth_h1, smooth_h1_groups = compute_episode_relative(smooth_path, horizon=1)
        if trimmed_h1.shape != smooth_h1.shape:
            raise RuntimeError(
                f"h1 shape mismatch for episode {episode_index}: "
                f"{trimmed_h1.shape} vs {smooth_h1.shape}"
            )

        h1_diff = smooth_h1 - trimmed_h1
        row: dict[str, float | int | str] = {
            "episode_index": episode_index,
            "h1_frames": int(trimmed_h1.shape[0]),
            "h1_mean_abs_diff": float(np.mean(np.abs(h1_diff))),
            "h1_max_abs_diff": float(np.max(np.abs(h1_diff))),
            "h1_rms_diff": float(np.sqrt(np.mean(h1_diff**2))),
        }
        for group_name, group_slice in (
            ("eef_9d", GROUP_SLICES["eef_9d"]),
            ("eef_xyz", GROUP_SLICES["eef_xyz"]),
            ("eef_rot6d", GROUP_SLICES["eef_rot6d"]),
            ("hand_joint_target", GROUP_SLICES["hand_joint_target"]),
            ("arm_joint_target", GROUP_SLICES["arm_joint_target"]),
        ):
            values = l2(h1_diff[:, group_slice])
            stats = finite_summary(values)
            for stat_name, stat_value in stats.items():
                row[f"h1_{group_name}_l2_diff_{stat_name}"] = stat_value

        episode_rows.append(row)
        h1_trimmed_chunks.append(trimmed_h1)
        h1_smooth_chunks.append(smooth_h1)

        if idx < plot_limit:
            plot_h1_overlay(
                episode_index=episode_index,
                trimmed_rel=trimmed_h1,
                smooth_rel=smooth_h1,
                out_path=episode_dir / f"episode_{episode_index:06d}_h1_relative_26d_overlay.png",
            )
            plot_h1_group_curves(
                episode_index=episode_index,
                trimmed_groups=trimmed_h1_groups,
                smooth_groups=smooth_h1_groups,
                out_path=episode_dir / f"episode_{episode_index:06d}_h1_group_relative_curves.png",
            )

        for horizon in HORIZONS:
            trimmed_rel, trimmed_groups = compute_episode_relative(trimmed_path, horizon=horizon)
            smooth_rel, smooth_groups = compute_episode_relative(smooth_path, horizon=horizon)
            if trimmed_rel.shape[0] == 0:
                continue
            diff = smooth_rel - trimmed_rel
            horizon_values.setdefault((horizon, "relative26"), []).append(l2(diff))
            horizon_values.setdefault((horizon, "eef_9d"), []).append(l2(diff[:, GROUP_SLICES["eef_9d"]]))
            horizon_values.setdefault((horizon, "hand_joint_target"), []).append(
                l2(diff[:, GROUP_SLICES["hand_joint_target"]])
            )
            horizon_values.setdefault((horizon, "arm_joint_target"), []).append(
                l2(diff[:, GROUP_SLICES["arm_joint_target"]])
            )
            eef_xyz_diff_mm = (
                l2(smooth_groups["eef_9d"][:, :3] - trimmed_groups["eef_9d"][:, :3]) * 1000.0
            )
            eef_rot_angle_diff_deg = np.abs(
                rot6d_angle_deg(smooth_groups["eef_9d"][:, 3:9])
                - rot6d_angle_deg(trimmed_groups["eef_9d"][:, 3:9])
            )
            horizon_values.setdefault((horizon, "eef_xyz_mm"), []).append(eef_xyz_diff_mm)
            horizon_values.setdefault((horizon, "eef_rot_angle_deg"), []).append(
                eef_rot_angle_diff_deg
            )

    h1_trimmed_all = np.concatenate(h1_trimmed_chunks, axis=0)
    h1_smooth_all = np.concatenate(h1_smooth_chunks, axis=0)
    h1_diff_all = h1_smooth_all - h1_trimmed_all

    per_dim = pd.DataFrame(
        {
            "dim": np.arange(26, dtype=int),
            "name": RELATIVE_NAMES,
            "h1_mean_abs_diff": np.mean(np.abs(h1_diff_all), axis=0),
            "h1_max_abs_diff": np.max(np.abs(h1_diff_all), axis=0),
            "h1_rms_diff": np.sqrt(np.mean(h1_diff_all**2, axis=0)),
            "trimmed_h1_std": np.std(h1_trimmed_all, axis=0),
            "smooth_h1_std": np.std(h1_smooth_all, axis=0),
        }
    )
    episode_metrics = pd.DataFrame(episode_rows)

    horizon_rows: list[dict[str, float | int | str]] = []
    for (horizon, group), chunks in sorted(horizon_values.items()):
        values = np.concatenate(chunks, axis=0)
        stats = finite_summary(values)
        horizon_rows.append(
            {
                "horizon": int(horizon),
                "group": group,
                "l2_diff_mean": stats["mean"],
                "l2_diff_p50": stats["p50"],
                "l2_diff_p95": stats["p95"],
                "l2_diff_p99": stats["p99"],
                "l2_diff_max": stats["max"],
                "samples": int(values.shape[0]),
            }
        )
    horizon_metrics = pd.DataFrame(horizon_rows)

    per_dim.to_csv(out_dir / "h1_relative_per_dim_diff.csv", index=False)
    episode_metrics.to_csv(out_dir / "h1_relative_episode_diff.csv", index=False)
    horizon_metrics.to_csv(out_dir / "relative_diff_by_horizon.csv", index=False)

    plot_h1_summary(per_dim, episode_metrics, out_dir)
    plot_horizon_summary(horizon_metrics, out_dir)

    summary = {
        "trimmed": str(args.trimmed.resolve()),
        "smooth": str(args.smooth.resolve()),
        "output_dir": str(out_dir),
        "semantics": {
            "h1": "delta_index=1, relative action from t to t+1",
            "eef_9d": "EndEffectorPose action(t+h) relative to action(t), XYZ_ROT6D",
            "hand_joint_target": "action.hand_joint_target(t+h) - action.hand_joint_target(t)",
            "arm_joint_target": "observation.state.arm_joint_pos(t+h) - observation.state.arm_joint_pos(t)",
        },
        "episodes": int(len(episode_rows)),
        "h1_samples": int(h1_diff_all.shape[0]),
        "h1_action_dim": int(h1_diff_all.shape[1]),
        "h1_mean_abs_diff": float(np.mean(np.abs(h1_diff_all))),
        "h1_max_abs_diff": float(np.max(np.abs(h1_diff_all))),
        "h1_eef_9d_l2_diff_p95": float(
            episode_metrics["h1_eef_9d_l2_diff_p95"].mean()
        ),
        "h1_hand_joint_target_l2_diff_p95": float(
            episode_metrics["h1_hand_joint_target_l2_diff_p95"].mean()
        ),
        "h1_arm_joint_target_l2_diff_p95": float(
            episode_metrics["h1_arm_joint_target_l2_diff_p95"].mean()
        ),
        "plots": {
            "h1_summary": str((out_dir / "h1_relative_diff_summary.png").resolve()),
            "horizon_summary": str((out_dir / "relative_diff_by_horizon.png").resolve()),
            "episode_dir": str(episode_dir.resolve()),
        },
        "tables": {
            "h1_per_dim": str((out_dir / "h1_relative_per_dim_diff.csv").resolve()),
            "h1_episode": str((out_dir / "h1_relative_episode_diff.csv").resolve()),
            "by_horizon": str((out_dir / "relative_diff_by_horizon.csv").resolve()),
        },
    }
    (out_dir / "relative_smooth_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
