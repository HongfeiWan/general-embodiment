#!/usr/bin/env python3
"""Create a smoothed copy of the mission2 LeRobot dataset.

Only the raw 19D absolute command in the parquet ``action`` column is smoothed.
Videos, metadata, observations, and task annotations are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
ACTION_DIM = 19
XYZ_SLICE = slice(0, 3)
ROT6D_SLICE = slice(3, 9)
HAND_SLICE = slice(9, 19)


def _odd_window(length: int, requested: int, polyorder: int) -> int | None:
    window = min(int(requested), int(length))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return None
    return window


def _hampel_1d(values: np.ndarray, window: int, threshold: float) -> tuple[np.ndarray, int]:
    filtered = values.astype(np.float64, copy=True)
    n = int(filtered.shape[0])
    half = int(window) // 2
    replacements = 0

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        neighborhood = filtered[start:end]
        median = float(np.median(neighborhood))
        mad = float(np.median(np.abs(neighborhood - median)))
        if mad <= 1e-12:
            continue
        sigma = 1.4826 * mad
        if abs(float(filtered[i]) - median) > threshold * sigma:
            filtered[i] = median
            replacements += 1

    return filtered, replacements


def _smooth_scalar_channels(
    values: np.ndarray,
    *,
    hampel_window: int,
    hampel_threshold: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    replacements = 0
    window = _odd_window(values.shape[0], savgol_window, savgol_polyorder)

    for dim in range(values.shape[1]):
        cleaned, dim_replacements = _hampel_1d(values[:, dim], hampel_window, hampel_threshold)
        replacements += dim_replacements
        if window is not None:
            cleaned = savgol_filter(
                cleaned,
                window_length=window,
                polyorder=savgol_polyorder,
                mode="interp",
            )
        smoothed[:, dim] = cleaned

    return smoothed, replacements


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
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


def _matrix_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    return np.concatenate([matrices[:, 0, :], matrices[:, 1, :]], axis=1)


def _smooth_rot6d(
    rot6d: np.ndarray,
    *,
    hampel_window: int,
    hampel_threshold: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, int]:
    matrices = _rot6d_to_matrix(np.asarray(rot6d, dtype=np.float64))
    rotations = Rotation.from_matrix(matrices)
    reference = rotations[0]
    relative_rotvec = (reference.inv() * rotations).as_rotvec()

    smoothed_rotvec, replacements = _smooth_scalar_channels(
        relative_rotvec,
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    smoothed_rotations = reference * Rotation.from_rotvec(smoothed_rotvec)
    return _matrix_to_rot6d(smoothed_rotations.as_matrix()), replacements


def _smooth_action_array(
    actions: np.ndarray,
    *,
    hampel_window: int,
    hampel_threshold: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    original = np.asarray(actions, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action array shape (T, {ACTION_DIM}), got {original.shape}")

    smoothed = original.copy()
    xyz_smoothed, xyz_replacements = _smooth_scalar_channels(
        original[:, XYZ_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    rot6d_smoothed, rot_replacements = _smooth_rot6d(
        original[:, ROT6D_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    hand_smoothed, hand_replacements = _smooth_scalar_channels(
        original[:, HAND_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )

    smoothed[:, XYZ_SLICE] = xyz_smoothed
    smoothed[:, ROT6D_SLICE] = rot6d_smoothed
    smoothed[:, HAND_SLICE] = hand_smoothed

    diff = smoothed - original
    metrics: dict[str, float | int] = {
        "frames": int(original.shape[0]),
        "hampel_replacements_xyz": int(xyz_replacements),
        "hampel_replacements_rotvec": int(rot_replacements),
        "hampel_replacements_hand": int(hand_replacements),
        "mean_abs_change": float(np.mean(np.abs(diff))),
        "max_abs_change": float(np.max(np.abs(diff))),
        "mean_abs_change_xyz": float(np.mean(np.abs(diff[:, XYZ_SLICE]))),
        "mean_abs_change_rot6d": float(np.mean(np.abs(diff[:, ROT6D_SLICE]))),
        "mean_abs_change_hand": float(np.mean(np.abs(diff[:, HAND_SLICE]))),
        "max_abs_change_xyz": float(np.max(np.abs(diff[:, XYZ_SLICE]))),
        "max_abs_change_rot6d": float(np.max(np.abs(diff[:, ROT6D_SLICE]))),
        "max_abs_change_hand": float(np.max(np.abs(diff[:, HAND_SLICE]))),
    }
    return smoothed.astype(np.float32), metrics


def _copy_dataset(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def smooth_dataset(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve()
    destination = args.destination.resolve()
    _copy_dataset(source, destination, overwrite=args.overwrite)

    episode_metrics = []
    parquet_paths = sorted((destination / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {destination / 'data'}")

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        smoothed_actions, metrics = _smooth_action_array(
            actions,
            hampel_window=args.hampel_window,
            hampel_threshold=args.hampel_threshold,
            savgol_window=args.savgol_window,
            savgol_polyorder=args.savgol_polyorder,
        )
        df["action"] = list(smoothed_actions)
        df.to_parquet(parquet_path, index=False)

        metrics["episode_file"] = str(parquet_path.relative_to(destination))
        episode_metrics.append(metrics)

    total_frames = sum(int(m["frames"]) for m in episode_metrics)
    summary: dict[str, object] = {
        "source": str(source),
        "destination": str(destination),
        "action_slices": {
            "eef_xyz": [0, 3],
            "eef_rot6d": [3, 9],
            "hand_joint_target": [9, 19],
        },
        "fps_assumption": 10,
        "hampel_window": int(args.hampel_window),
        "hampel_threshold_sigma": float(args.hampel_threshold),
        "savgol_window": int(args.savgol_window),
        "savgol_polyorder": int(args.savgol_polyorder),
        "episodes": len(episode_metrics),
        "total_frames": total_frames,
        "total_hampel_replacements_xyz": sum(
            int(m["hampel_replacements_xyz"]) for m in episode_metrics
        ),
        "total_hampel_replacements_rotvec": sum(
            int(m["hampel_replacements_rotvec"]) for m in episode_metrics
        ),
        "total_hampel_replacements_hand": sum(
            int(m["hampel_replacements_hand"]) for m in episode_metrics
        ),
        "mean_abs_change": float(
            sum(float(m["mean_abs_change"]) * int(m["frames"]) for m in episode_metrics)
            / max(total_frames, 1)
        ),
        "max_abs_change": float(max(float(m["max_abs_change"]) for m in episode_metrics)),
        "episode_metrics": episode_metrics,
    }

    summary_path = destination.parent / "smooth_action_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=MISSION_DIR / "trimmed",
        help="Source LeRobot dataset directory.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=MISSION_DIR / "smooth",
        help="Destination dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace destination if it exists.")
    parser.add_argument("--hampel-window", type=int, default=5)
    parser.add_argument("--hampel-threshold", type=float, default=3.0)
    parser.add_argument("--savgol-window", type=int, default=7)
    parser.add_argument("--savgol-polyorder", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = smooth_dataset(parse_args())
    print(json.dumps({k: v for k, v in summary.items() if k != "episode_metrics"}, indent=2))


if __name__ == "__main__":
    main()
