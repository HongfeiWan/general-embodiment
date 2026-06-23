#!/usr/bin/env python3
"""Plot future relative action curves used by nero_right_l10_multiview_modality_config."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_DATASET_DIR = MISSION_DIR / "trimmed"
DEFAULT_OUTPUT_DIR = MISSION_DIR / "config_relative_action_curves"

ACTION_EEF_SLICE = slice(0, 9)
ACTION_HAND_SLICE = slice(9, 19)
STATE_ARM_SLICE = slice(0, 7)
STATE_EEF_SLICE = slice(7, 16)
STATE_HAND_SLICE = slice(16, 26)
HORIZONS = (1, 2, 4, 8, 16, 32)
CURVE_HORIZONS = (1, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-horizon", type=int, default=32)
    parser.add_argument("--max-episode-plots", type=int, default=-1)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        raise ValueError(f"Column {column_name!r} is empty")
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise ValueError(f"Column {column_name!r} has inconsistent vector sizes")
    return np.vstack(values)


def episode_index_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: math.nan for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def l2(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(values, axis=1)


def rot6d_angle_deg(rot6d: np.ndarray) -> np.ndarray:
    if rot6d.size == 0:
        return np.asarray([], dtype=np.float64)
    angles = []
    for row in rot6d:
        matrix = rot6d_to_matrix(np.asarray(row, dtype=np.float64).reshape(1, 6))[0]
        angles.append(float(Rotation.from_matrix(matrix).magnitude() * 180.0 / np.pi))
    return np.asarray(angles, dtype=np.float64)


def read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    action = stack_vector_column(df["action"], "action")
    state = stack_vector_column(df["observation.state"], "observation.state")
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    return action[:, ACTION_EEF_SLICE], action[:, ACTION_HAND_SLICE], state[:, STATE_ARM_SLICE], timestamps


def read_episode_full_state(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    action = stack_vector_column(df["action"], "action")
    state = stack_vector_column(df["observation.state"], "observation.state")
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    return (
        action[:, ACTION_EEF_SLICE],
        action[:, ACTION_HAND_SLICE],
        state[:, STATE_ARM_SLICE],
        state[:, STATE_EEF_SLICE],
        state[:, STATE_HAND_SLICE],
        timestamps,
    )


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    first_row = values[:, 0:3]
    second_row = values[:, 3:6]
    r1 = first_row / np.maximum(np.linalg.norm(first_row, axis=1, keepdims=True), 1e-12)
    second_orthogonal = second_row - np.sum(r1 * second_row, axis=1, keepdims=True) * r1
    r2 = second_orthogonal / np.maximum(
        np.linalg.norm(second_orthogonal, axis=1, keepdims=True), 1e-12
    )
    r3 = np.cross(r1, r2)
    matrices = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, :] = r1
    matrices[:, 1, :] = r2
    matrices[:, 2, :] = r3
    return matrices


def matrix_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    return np.concatenate([matrices[:, 0, :], matrices[:, 1, :]], axis=1)


def relative_eef(action_eef: np.ndarray, horizon: int) -> np.ndarray:
    n = action_eef.shape[0] - horizon
    if n <= 0:
        return np.zeros((0, 9), dtype=np.float64)
    ref_pos = np.asarray(action_eef[:-horizon, :3], dtype=np.float64)
    target_pos = np.asarray(action_eef[horizon:, :3], dtype=np.float64)
    ref_rot = rot6d_to_matrix(action_eef[:-horizon, 3:9])
    target_rot = rot6d_to_matrix(action_eef[horizon:, 3:9])
    ref_inv = np.transpose(ref_rot, (0, 2, 1))
    relative_pos = np.einsum("nij,nj->ni", ref_inv, target_pos - ref_pos)
    relative_rot = np.einsum("nij,njk->nik", ref_inv, target_rot)
    return np.concatenate([relative_pos, matrix_to_rot6d(relative_rot)], axis=1)


def relative_groups(
    action_eef: np.ndarray,
    action_hand: np.ndarray,
    state_arm: np.ndarray,
    horizon: int,
) -> dict[str, np.ndarray]:
    n = min(action_eef.shape[0], action_hand.shape[0], state_arm.shape[0]) - horizon
    if n <= 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return {"eef_9d": empty, "hand_joint_target": empty, "arm_joint_target": empty}
    # Exact config semantics:
    # eef_9d: action(t+h) relative to action(t)
    # hand_joint_target: hand action(t+h) relative to hand action(t)
    # arm_joint_target: state arm_joint_pos(t+h) relative to state arm_joint_pos(t)
    return {
        "eef_9d": relative_eef(action_eef, horizon),
        "hand_joint_target": action_hand[horizon:] - action_hand[:-horizon],
        "arm_joint_target": state_arm[horizon:] - state_arm[:-horizon],
    }


def groot_relative_eef(state_eef: np.ndarray, action_eef: np.ndarray) -> np.ndarray:
    n = min(state_eef.shape[0], action_eef.shape[0])
    if n <= 0:
        return np.zeros((0, 9), dtype=np.float64)
    ref_pos = np.asarray(state_eef[:n, :3], dtype=np.float64)
    target_pos = np.asarray(action_eef[:n, :3], dtype=np.float64)
    ref_rot = rot6d_to_matrix(state_eef[:n, 3:9])
    target_rot = rot6d_to_matrix(action_eef[:n, 3:9])
    ref_inv = np.transpose(ref_rot, (0, 2, 1))
    relative_pos = np.einsum("nij,nj->ni", ref_inv, target_pos - ref_pos)
    relative_rot = np.einsum("nij,njk->nik", ref_inv, target_rot)
    return np.concatenate([relative_pos, matrix_to_rot6d(relative_rot)], axis=1)


def groot_relative_groups(
    action_eef: np.ndarray,
    action_hand: np.ndarray,
    state_arm: np.ndarray,
    state_eef: np.ndarray,
    state_hand: np.ndarray,
    delta: int,
) -> dict[str, np.ndarray]:
    n = min(
        action_eef.shape[0],
        action_hand.shape[0],
        state_arm.shape[0],
        state_eef.shape[0],
        state_hand.shape[0],
    ) - delta
    if n <= 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return {"eef_9d": empty, "hand_joint_target": empty, "arm_joint_target": empty}
    return {
        "eef_9d": groot_relative_eef(state_eef[:n], action_eef[delta : delta + n]),
        "hand_joint_target": action_hand[delta : delta + n] - state_hand[:n],
        "arm_joint_target": state_arm[delta : delta + n] - state_arm[:n],
    }


def plot_episode_curves(
    *,
    out_path: Path,
    episode_index: int,
    action_eef: np.ndarray,
    action_hand: np.ndarray,
    state_arm: np.ndarray,
    timestamps: np.ndarray,
    max_horizon: int,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), constrained_layout=True)
    fig.suptitle(
        f"Episode {episode_index:06d} config-relative future action curves",
        fontsize=14,
    )
    horizons = [h for h in CURVE_HORIZONS if h <= max_horizon and action_eef.shape[0] > h]

    axes[0].set_title("EEF relative translation L2, exact EndEffectorPose relative (mm)")
    axes[1].set_title("EEF relative rotation angle (deg), exact EndEffectorPose relative")
    axes[2].set_title("Hand relative target L2")
    axes[3].set_title("Arm joint target alias relative L2")
    axes[4].set_title("All groups relative L2 heatmap by horizon")

    heat_rows: list[np.ndarray] = []
    heat_labels: list[str] = []
    for horizon in horizons:
        groups = relative_groups(action_eef, action_hand, state_arm, horizon)
        t = timestamps[: groups["eef_9d"].shape[0]]
        axes[0].plot(t, l2(groups["eef_9d"][:, :3]) * 1000.0, label=f"h{horizon}")
        axes[1].plot(t, rot6d_angle_deg(groups["eef_9d"][:, 3:9]), label=f"h{horizon}")
        axes[2].plot(t, l2(groups["hand_joint_target"]), label=f"h{horizon}")
        axes[3].plot(t, l2(groups["arm_joint_target"]), label=f"h{horizon}")

    for horizon in range(1, max_horizon + 1):
        if action_eef.shape[0] <= horizon:
            continue
        groups = relative_groups(action_eef, action_hand, state_arm, horizon)
        heat_rows.extend(
            [
                l2(groups["eef_9d"][:, :3]) * 1000.0,
                l2(groups["eef_9d"][:, 3:9]),
                l2(groups["hand_joint_target"]),
                l2(groups["arm_joint_target"]),
            ]
        )
        heat_labels.extend(
            [
                f"h{horizon} eef_xyz_mm",
                f"h{horizon} eef_rot_deg",
                f"h{horizon} hand",
                f"h{horizon} arm",
            ]
        )

    for axis in axes[:4]:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
        axis.set_xlabel("timestamp (s)")
        axis.set_ylabel("relative L2")

    if heat_rows:
        max_len = max(row.shape[0] for row in heat_rows)
        heat = np.full((len(heat_rows), max_len), np.nan, dtype=np.float64)
        for idx, row in enumerate(heat_rows):
            heat[idx, : row.shape[0]] = row
        finite = heat[np.isfinite(heat)]
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        image = axes[4].imshow(heat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
        axes[4].set_yticks(range(len(heat_labels)))
        axes[4].set_yticklabels(heat_labels, fontsize=6)
        axes[4].set_xlabel("start timestep")
        fig.colorbar(image, ax=axes[4], shrink=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_episode_heatmaps(
    *,
    out_path: Path,
    episode_index: int,
    action_eef: np.ndarray,
    action_hand: np.ndarray,
    state_arm: np.ndarray,
    max_horizon: int,
) -> None:
    groups_for_heatmap = [
        ("EEF XYZ relative translation L2 (mm)", "eef_xyz_mm"),
        ("EEF relative rotation angle (deg)", "eef_rot_deg"),
        ("Hand target relative L2", "hand"),
        ("Arm joint alias relative L2", "arm"),
    ]
    heatmaps: dict[str, np.ndarray] = {}
    max_steps = max(0, action_eef.shape[0] - 1)
    horizons = [h for h in range(1, max_horizon + 1) if action_eef.shape[0] > h]
    if not horizons:
        return

    for _, key in groups_for_heatmap:
        heatmaps[key] = np.full((len(horizons), max_steps), np.nan, dtype=np.float64)

    for row_idx, horizon in enumerate(horizons):
        groups = relative_groups(action_eef, action_hand, state_arm, horizon)
        values_by_key = {
            "eef_xyz_mm": l2(groups["eef_9d"][:, :3]) * 1000.0,
            "eef_rot_deg": rot6d_angle_deg(groups["eef_9d"][:, 3:9]),
            "hand": l2(groups["hand_joint_target"]),
            "arm": l2(groups["arm_joint_target"]),
        }
        for key, values in values_by_key.items():
            heatmaps[key][row_idx, : values.shape[0]] = values

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"Episode {episode_index:06d} config-relative action heatmaps",
        fontsize=14,
    )
    y_tick_values = [1, 8, 16, 24, 32]
    y_tick_positions = [horizons.index(h) for h in y_tick_values if h in horizons]
    y_tick_labels = [str(h) for h in y_tick_values if h in horizons]

    for axis, (title, key) in zip(axes, groups_for_heatmap):
        heat = heatmaps[key]
        finite = heat[np.isfinite(heat)]
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        image = axis.imshow(
            heat,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_ylabel("horizon")
        axis.set_yticks(y_tick_positions)
        axis.set_yticklabels(y_tick_labels)
        axis.set_xlabel("start timestep")
        fig.colorbar(image, ax=axis, shrink=0.85)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overview(metrics: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "eef_xyz_h32_l2_p95_mm", "EEF relative XYZ h32 p95 (mm)"),
        (axes[1], "eef_rot_angle_h32_p95_deg", "EEF relative rotation h32 p95 (deg)"),
        (axes[2], "hand_h32_l2_p95", "Hand relative h32 p95"),
        (axes[3], "arm_h32_l2_p95", "Arm alias relative h32 p95"),
    ):
        episodes = [row["episode_index"] for row in metrics if key in row and math.isfinite(row[key])]
        values = [row[key] for row in metrics if key in row and math.isfinite(row[key])]
        axis.bar(episodes, values, color="tab:blue", alpha=0.75)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_dir / "config_relative_h32_overview.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    max_horizon = max(1, int(args.max_horizon))
    output_dir.mkdir(parents=True, exist_ok=True)
    read_json(dataset_dir / "meta" / "info.json")
    episodes_meta = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")

    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    plot_limit = len(parquet_files) if args.max_episode_plots < 0 else int(args.max_episode_plots)
    metrics: list[dict[str, Any]] = []

    for path in parquet_files:
        episode_index = episode_index_from_path(path)
        action_eef, action_hand, state_arm, timestamps = read_episode(path)
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frames": int(action_eef.shape[0]),
            "duration_s": float(timestamps[-1] - timestamps[0]) if timestamps.size else math.nan,
        }
        meta = episodes_meta[episode_index] if episode_index < len(episodes_meta) else {}
        if isinstance(meta, dict):
            row["raw_episode_id"] = meta.get("teleop_stack_metadata", {}).get("raw_episode_id")

        for horizon in HORIZONS:
            if horizon > max_horizon or action_eef.shape[0] <= horizon:
                continue
            groups = relative_groups(action_eef, action_hand, state_arm, horizon)
            values_by_name = {
                "eef_xyz": l2(groups["eef_9d"][:, :3]) * 1000.0,
                "eef_rot6d": l2(groups["eef_9d"][:, 3:9]),
                "eef_rot_angle": rot6d_angle_deg(groups["eef_9d"][:, 3:9]),
                "hand": l2(groups["hand_joint_target"]),
                "arm": l2(groups["arm_joint_target"]),
                "relative26_proxy": np.sqrt(
                    l2(groups["eef_9d"]) ** 2
                    + l2(groups["hand_joint_target"]) ** 2
                    + l2(groups["arm_joint_target"]) ** 2
                ),
            }
            for name, values in values_by_name.items():
                stats = finite_stats(values)
                suffix = "_mm" if name == "eef_xyz" else "_deg" if name == "eef_rot_angle" else ""
                for stat_name, stat_value in stats.items():
                    row[f"{name}_h{horizon}_l2_{stat_name}{suffix}"] = stat_value
        metrics.append(row)

        if episode_index < plot_limit:
            plot_episode_curves(
                out_path=output_dir / f"episode_{episode_index:06d}_config_relative_curves.png",
                episode_index=episode_index,
                action_eef=action_eef,
                action_hand=action_hand,
                state_arm=state_arm,
                timestamps=timestamps,
                max_horizon=max_horizon,
            )
            plot_episode_heatmaps(
                out_path=output_dir / f"episode_{episode_index:06d}_config_relative_heatmaps.png",
                episode_index=episode_index,
                action_eef=action_eef,
                action_hand=action_hand,
                state_arm=state_arm,
                max_horizon=max_horizon,
            )

    csv_path = output_dir / "config_relative_action_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in metrics for key in row}))
        writer.writeheader()
        writer.writerows(metrics)

    summary = {
        "dataset_dir": str(dataset_dir),
        "episode_count": len(metrics),
        "max_horizon": max_horizon,
        "semantics": {
            "eef_9d": "action(t+h) relative to action(t), XYZ_ROT6D",
            "hand_joint_target": "action.hand_joint_target(t+h) - action.hand_joint_target(t)",
            "arm_joint_target": "observation.state.arm_joint_pos(t+h) - observation.state.arm_joint_pos(t)",
        },
        "top_eef_xyz_h32_l2_p95_mm": sorted(
            [row for row in metrics if "eef_xyz_h32_l2_p95_mm" in row],
            key=lambda row: float(row["eef_xyz_h32_l2_p95_mm"]),
            reverse=True,
        )[:8],
        "top_relative26_proxy_h32_l2_p95": sorted(
            [row for row in metrics if "relative26_proxy_h32_l2_p95" in row],
            key=lambda row: float(row["relative26_proxy_h32_l2_p95"]),
            reverse=True,
        )[:8],
    }
    (output_dir / "config_relative_action_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_overview(metrics, output_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"Episodes: {len(metrics)}")
    print(f"Metrics: {csv_path}")
    print(f"Summary: {output_dir / 'config_relative_action_summary.json'}")
    print(f"Overview: {output_dir / 'config_relative_h32_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
