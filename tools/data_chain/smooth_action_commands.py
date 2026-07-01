#!/usr/bin/env python3
"""Create a smoothed copy of the mission2 LeRobot dataset.

The 19D absolute command in ``action`` and the 26D ``observation.state`` vector
are smoothed per episode. Videos, task annotations, and non-vector columns are
copied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
STATE_DIM = 26
STATE_ARM_JOINT_SLICE = slice(0, 7)
STATE_XYZ_SLICE = slice(7, 10)
STATE_ROT6D_SLICE = slice(10, 16)
STATE_HAND_SLICE = slice(16, 26)


@dataclass(frozen=True)
class SmoothConfig:
    hampel_window: int
    hampel_threshold: float
    savgol_window: int
    savgol_polyorder: int
    smooth_action: bool
    smooth_state: bool


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


def _stack_vector_column(series: pd.Series, column: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise ValueError(f"Inconsistent vector widths in {column}")
    return np.vstack(values)


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
        "action_hampel_replacements_xyz": int(xyz_replacements),
        "action_hampel_replacements_rotvec": int(rot_replacements),
        "action_hampel_replacements_hand": int(hand_replacements),
        "action_mean_abs_change": float(np.mean(np.abs(diff))),
        "action_max_abs_change": float(np.max(np.abs(diff))),
        "action_mean_abs_change_xyz": float(np.mean(np.abs(diff[:, XYZ_SLICE]))),
        "action_mean_abs_change_rot6d": float(np.mean(np.abs(diff[:, ROT6D_SLICE]))),
        "action_mean_abs_change_hand": float(np.mean(np.abs(diff[:, HAND_SLICE]))),
        "action_max_abs_change_xyz": float(np.max(np.abs(diff[:, XYZ_SLICE]))),
        "action_max_abs_change_rot6d": float(np.max(np.abs(diff[:, ROT6D_SLICE]))),
        "action_max_abs_change_hand": float(np.max(np.abs(diff[:, HAND_SLICE]))),
    }
    return smoothed.astype(np.float32), metrics


def _smooth_state_array(
    states: np.ndarray,
    *,
    hampel_window: int,
    hampel_threshold: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    original = np.asarray(states, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != STATE_DIM:
        raise ValueError(f"Expected observation.state array shape (T, {STATE_DIM}), got {original.shape}")

    smoothed = original.copy()
    arm_smoothed, arm_replacements = _smooth_scalar_channels(
        original[:, STATE_ARM_JOINT_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    xyz_smoothed, xyz_replacements = _smooth_scalar_channels(
        original[:, STATE_XYZ_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    rot6d_smoothed, rot_replacements = _smooth_rot6d(
        original[:, STATE_ROT6D_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    hand_smoothed, hand_replacements = _smooth_scalar_channels(
        original[:, STATE_HAND_SLICE],
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )

    smoothed[:, STATE_ARM_JOINT_SLICE] = arm_smoothed
    smoothed[:, STATE_XYZ_SLICE] = xyz_smoothed
    smoothed[:, STATE_ROT6D_SLICE] = rot6d_smoothed
    smoothed[:, STATE_HAND_SLICE] = hand_smoothed

    diff = smoothed - original
    metrics: dict[str, float | int] = {
        "state_hampel_replacements_arm_joint": int(arm_replacements),
        "state_hampel_replacements_xyz": int(xyz_replacements),
        "state_hampel_replacements_rotvec": int(rot_replacements),
        "state_hampel_replacements_hand": int(hand_replacements),
        "state_mean_abs_change": float(np.mean(np.abs(diff))),
        "state_max_abs_change": float(np.max(np.abs(diff))),
        "state_mean_abs_change_arm_joint": float(np.mean(np.abs(diff[:, STATE_ARM_JOINT_SLICE]))),
        "state_mean_abs_change_xyz": float(np.mean(np.abs(diff[:, STATE_XYZ_SLICE]))),
        "state_mean_abs_change_rot6d": float(np.mean(np.abs(diff[:, STATE_ROT6D_SLICE]))),
        "state_mean_abs_change_hand": float(np.mean(np.abs(diff[:, STATE_HAND_SLICE]))),
        "state_max_abs_change_arm_joint": float(np.max(np.abs(diff[:, STATE_ARM_JOINT_SLICE]))),
        "state_max_abs_change_xyz": float(np.max(np.abs(diff[:, STATE_XYZ_SLICE]))),
        "state_max_abs_change_rot6d": float(np.max(np.abs(diff[:, STATE_ROT6D_SLICE]))),
        "state_max_abs_change_hand": float(np.max(np.abs(diff[:, STATE_HAND_SLICE]))),
    }
    return smoothed.astype(np.float32), metrics


def _copy_ignore(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if ".bak_" in name or name.endswith(".tmp")]


def _copy_dataset(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=_copy_ignore)


def _smooth_parquet(parquet_path: Path, destination: Path, config: SmoothConfig) -> dict[str, float | int | str]:
    df = pd.read_parquet(parquet_path)
    metrics: dict[str, float | int | str] = {
        "episode_file": str(parquet_path.relative_to(destination)),
        "frames": int(len(df)),
    }
    changed = False

    if config.smooth_action:
        if "action" not in df.columns:
            raise ValueError(f"{parquet_path} is missing action")
        actions = _stack_vector_column(df["action"], "action")
        smoothed_actions, action_metrics = _smooth_action_array(
            actions,
            hampel_window=config.hampel_window,
            hampel_threshold=config.hampel_threshold,
            savgol_window=config.savgol_window,
            savgol_polyorder=config.savgol_polyorder,
        )
        df["action"] = [row.astype(np.float32) for row in smoothed_actions]
        metrics.update(action_metrics)
        changed = True

    if config.smooth_state:
        if "observation.state" not in df.columns:
            raise ValueError(f"{parquet_path} is missing observation.state")
        states = _stack_vector_column(df["observation.state"], "observation.state")
        smoothed_states, state_metrics = _smooth_state_array(
            states,
            hampel_window=config.hampel_window,
            hampel_threshold=config.hampel_threshold,
            savgol_window=config.savgol_window,
            savgol_polyorder=config.savgol_polyorder,
        )
        df["observation.state"] = [row.astype(np.float32) for row in smoothed_states]
        metrics.update(state_metrics)
        changed = True

    if changed:
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=parquet_path.stem + ".tmp.",
            dir=str(parquet_path.parent),
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        try:
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(parquet_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return metrics


def _run_smoothing(
    parquet_paths: list[Path],
    *,
    destination: Path,
    config: SmoothConfig,
    workers: int,
) -> list[dict[str, float | int | str]]:
    if workers <= 1:
        return [_smooth_parquet(path, destination, config) for path in parquet_paths]

    rows: list[dict[str, float | int | str]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_smooth_parquet, path, destination, config) for path in parquet_paths]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: str(row["episode_file"]))


def _vector_stats(matrix: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.nanmean(matrix, axis=0).astype(float).tolist(),
        "std": np.nanstd(matrix, axis=0).astype(float).tolist(),
        "min": np.nanmin(matrix, axis=0).astype(float).tolist(),
        "max": np.nanmax(matrix, axis=0).astype(float).tolist(),
        "q01": np.nanquantile(matrix, 0.01, axis=0).astype(float).tolist(),
        "q99": np.nanquantile(matrix, 0.99, axis=0).astype(float).tolist(),
    }


def _vector_fingerprint(matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    return "sha256:" + hashlib.sha256(contiguous.tobytes()).hexdigest()


def _recompute_vector_stats(destination: Path, columns: tuple[str, ...]) -> dict[str, Any]:
    stats_path = destination / "meta" / "stats.json"
    if not stats_path.exists():
        return {"updated": False, "reason": "missing meta/stats.json"}

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    chunks: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    for parquet_path in sorted((destination / "data").rglob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=list(columns))
        for column in columns:
            chunks[column].append(_stack_vector_column(df[column], column))

    fingerprints = stats.setdefault("__fingerprints__", {})
    for column, matrices in chunks.items():
        if not matrices:
            continue
        matrix = np.vstack(matrices)
        stats[column] = _vector_stats(matrix)
        if isinstance(fingerprints, dict):
            fingerprints[column] = _vector_fingerprint(matrix)

    stats_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"updated": True, "columns": list(columns)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _update_smoothed_metadata(destination: Path, smoothed_at_utc: str) -> bool:
    episodes_path = destination / "meta" / "episodes.jsonl"
    rows = _read_jsonl(episodes_path)
    if not rows:
        return False
    for row in rows:
        row["smoothed_at_utc"] = smoothed_at_utc
        metadata = row.setdefault("teleop_stack_metadata", {})
        if isinstance(metadata, dict):
            metadata["smoothed_at_utc"] = smoothed_at_utc
    _write_jsonl(episodes_path, rows)
    return True


def _weighted_mean(episode_metrics: list[dict[str, Any]], key: str, total_frames: int) -> float:
    return float(
        sum(float(m.get(key, 0.0)) * int(m["frames"]) for m in episode_metrics)
        / max(total_frames, 1)
    )


def _max_metric(episode_metrics: list[dict[str, Any]], key: str) -> float:
    values = [float(m[key]) for m in episode_metrics if key in m]
    return float(max(values)) if values else 0.0


def _sum_metric(episode_metrics: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(m.get(key, 0)) for m in episode_metrics))


def smooth_dataset(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve()
    destination = args.destination.resolve()
    _copy_dataset(source, destination, overwrite=args.overwrite)

    parquet_paths = sorted((destination / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {destination / 'data'}")

    workers = int(args.workers) if int(args.workers) > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(parquet_paths)))
    config = SmoothConfig(
        hampel_window=int(args.hampel_window),
        hampel_threshold=float(args.hampel_threshold),
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        smooth_action=not bool(args.no_action),
        smooth_state=not bool(args.no_state),
    )
    if not config.smooth_action and not config.smooth_state:
        raise ValueError("Nothing to smooth: both --no-action and --no-state were provided.")

    episode_metrics = _run_smoothing(
        parquet_paths,
        destination=destination,
        config=config,
        workers=workers,
    )
    smoothed_at_utc = datetime.now(timezone.utc).isoformat()
    metadata_updated = _update_smoothed_metadata(destination, smoothed_at_utc)
    stats_columns = []
    if config.smooth_action:
        stats_columns.append("action")
    if config.smooth_state:
        stats_columns.append("observation.state")
    stats_update = _recompute_vector_stats(destination, tuple(stats_columns))

    total_frames = sum(int(m["frames"]) for m in episode_metrics)
    summary: dict[str, object] = {
        "source": str(source),
        "destination": str(destination),
        "workers": workers,
        "smoothed_at_utc": smoothed_at_utc,
        "smoothed_columns": [
            column
            for column, enabled in (
                ("action", config.smooth_action),
                ("observation.state", config.smooth_state),
            )
            if enabled
        ],
        "action_slices": {
            "eef_xyz": [0, 3],
            "eef_rot6d": [3, 9],
            "hand_joint_target": [9, 19],
        },
        "state_slices": {
            "arm_joint_pos": [0, 7],
            "eef_xyz": [7, 10],
            "eef_rot6d": [10, 16],
            "hand_joint_pos": [16, 26],
        },
        "fps_assumption": 10,
        "hampel_window": int(args.hampel_window),
        "hampel_threshold_sigma": float(args.hampel_threshold),
        "savgol_window": int(args.savgol_window),
        "savgol_polyorder": int(args.savgol_polyorder),
        "episodes": len(episode_metrics),
        "total_frames": total_frames,
        "metadata_updated": metadata_updated,
        "stats_update": stats_update,
        "episode_metrics": episode_metrics,
    }
    if config.smooth_action:
        summary.update(
            {
                "total_action_hampel_replacements_xyz": _sum_metric(
                    episode_metrics, "action_hampel_replacements_xyz"
                ),
                "total_action_hampel_replacements_rotvec": _sum_metric(
                    episode_metrics, "action_hampel_replacements_rotvec"
                ),
                "total_action_hampel_replacements_hand": _sum_metric(
                    episode_metrics, "action_hampel_replacements_hand"
                ),
                "action_mean_abs_change": _weighted_mean(
                    episode_metrics, "action_mean_abs_change", total_frames
                ),
                "action_max_abs_change": _max_metric(episode_metrics, "action_max_abs_change"),
                "action_mean_abs_change_xyz": _weighted_mean(
                    episode_metrics, "action_mean_abs_change_xyz", total_frames
                ),
                "action_mean_abs_change_rot6d": _weighted_mean(
                    episode_metrics, "action_mean_abs_change_rot6d", total_frames
                ),
                "action_mean_abs_change_hand": _weighted_mean(
                    episode_metrics, "action_mean_abs_change_hand", total_frames
                ),
                "action_max_abs_change_xyz": _max_metric(
                    episode_metrics, "action_max_abs_change_xyz"
                ),
                "action_max_abs_change_rot6d": _max_metric(
                    episode_metrics, "action_max_abs_change_rot6d"
                ),
                "action_max_abs_change_hand": _max_metric(
                    episode_metrics, "action_max_abs_change_hand"
                ),
                # Backward-compatible aliases for older notes/scripts that read this summary.
                "total_hampel_replacements_xyz": _sum_metric(
                    episode_metrics, "action_hampel_replacements_xyz"
                ),
                "total_hampel_replacements_rotvec": _sum_metric(
                    episode_metrics, "action_hampel_replacements_rotvec"
                ),
                "total_hampel_replacements_hand": _sum_metric(
                    episode_metrics, "action_hampel_replacements_hand"
                ),
                "mean_abs_change": _weighted_mean(
                    episode_metrics, "action_mean_abs_change", total_frames
                ),
                "max_abs_change": _max_metric(episode_metrics, "action_max_abs_change"),
            }
        )
    if config.smooth_state:
        summary.update(
            {
                "total_state_hampel_replacements_arm_joint": _sum_metric(
                    episode_metrics, "state_hampel_replacements_arm_joint"
                ),
                "total_state_hampel_replacements_xyz": _sum_metric(
                    episode_metrics, "state_hampel_replacements_xyz"
                ),
                "total_state_hampel_replacements_rotvec": _sum_metric(
                    episode_metrics, "state_hampel_replacements_rotvec"
                ),
                "total_state_hampel_replacements_hand": _sum_metric(
                    episode_metrics, "state_hampel_replacements_hand"
                ),
                "state_mean_abs_change": _weighted_mean(
                    episode_metrics, "state_mean_abs_change", total_frames
                ),
                "state_max_abs_change": _max_metric(episode_metrics, "state_max_abs_change"),
                "state_mean_abs_change_arm_joint": _weighted_mean(
                    episode_metrics, "state_mean_abs_change_arm_joint", total_frames
                ),
                "state_mean_abs_change_xyz": _weighted_mean(
                    episode_metrics, "state_mean_abs_change_xyz", total_frames
                ),
                "state_mean_abs_change_rot6d": _weighted_mean(
                    episode_metrics, "state_mean_abs_change_rot6d", total_frames
                ),
                "state_mean_abs_change_hand": _weighted_mean(
                    episode_metrics, "state_mean_abs_change_hand", total_frames
                ),
                "state_max_abs_change_arm_joint": _max_metric(
                    episode_metrics, "state_max_abs_change_arm_joint"
                ),
                "state_max_abs_change_xyz": _max_metric(
                    episode_metrics, "state_max_abs_change_xyz"
                ),
                "state_max_abs_change_rot6d": _max_metric(
                    episode_metrics, "state_max_abs_change_rot6d"
                ),
                "state_max_abs_change_hand": _max_metric(
                    episode_metrics, "state_max_abs_change_hand"
                ),
            }
        )

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count for parquet smoothing. 0 uses all CPU cores.",
    )
    parser.add_argument("--no-action", action="store_true", help="Do not smooth the action column.")
    parser.add_argument("--no-state", action="store_true", help="Do not smooth observation.state.")
    return parser.parse_args()


def main() -> None:
    summary = smooth_dataset(parse_args())
    print(json.dumps({k: v for k, v in summary.items() if k != "episode_metrics"}, indent=2))


if __name__ == "__main__":
    main()
