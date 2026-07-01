#!/usr/bin/env python3
"""Inspect and repair LeRobot EEF 9D state/action rotation conventions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIMMED = REPO_ROOT / "missions" / "nero" / "mission2" / "trimmed"
DEFAULT_LEROBOT_ROOT = REPO_ROOT / "missions" / "nero" / "mission2" / "lerobot_v2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "missions" / "nero" / "mission2" / "quality" / "eef9d_alignment"

ROT6D_ROW_MAJOR = ("r00", "r01", "r02", "r10", "r11", "r12")
ROT6D_COLUMN_MAJOR = ("r00", "r10", "r20", "r01", "r11", "r21")
ACTION_TO_STATE_LEFT = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
ACTION_TO_STATE_RIGHT = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
EXPECTED_LEFT_NAME = "+z,+x,+y"
EXPECTED_RIGHT_NAME = "-y,-z,+x"
EXPECTED_ACTION_ROT6D_MAPPING = ["r22", "-r20", "-r21", "r02", "-r00", "-r01"]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_dir: Path
    info: dict[str, Any]
    modality: dict[str, Any]
    episodes: list[dict[str, Any]]
    state_eef: slice
    action_eef: slice
    state_rot: slice
    action_rot: slice
    state_pos: slice
    action_pos: slice
    state_rot_convention: str
    action_rot_convention: str
    action_already_in_state_frame: bool


@dataclass(frozen=True)
class EpisodeTask:
    dataset_dir: Path
    dataset_label: str
    episode_index: int
    data_path: Path
    length: int
    state_eef: tuple[int, int]
    action_eef: tuple[int, int]
    state_rot: tuple[int, int]
    action_rot: tuple[int, int]
    state_pos: tuple[int, int]
    action_pos: tuple[int, int]
    state_rot_convention: str
    action_rot_convention: str
    action_already_in_state_frame: bool
    fix_left: str | None = None
    fix_right: str | None = None
    fix_transpose: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        help="Specific LeRobot dataset directory. Can be repeated.",
    )
    parser.add_argument(
        "--include-defaults",
        action="store_true",
        help="Include mission2/trimmed and every dataset below mission2/lerobot_v2.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-frames", type=int, default=0, help="Uniform sample per episode; 0 means all frames.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker count. 0 uses all CPU cores.")
    parser.add_argument("--mode", choices=("inspect", "fix"), default="inspect")
    parser.add_argument("--apply", action="store_true", help="Actually write parquet/meta changes in fix mode.")
    parser.add_argument(
        "--allow-mixed-mappings",
        action="store_true",
        help="Allow fix mode even when inspection finds non-expected per-episode mappings.",
    )
    parser.add_argument(
        "--fix-strategy",
        choices=("expected", "best-per-episode"),
        default="expected",
        help="expected applies the current fixed transform; best-per-episode applies each episode's best inspected L/R mapping.",
    )
    parser.add_argument(
        "--max-best-angle-mean",
        type=float,
        default=20.0,
        help="Abort fix if any episode's best mapping mean angle exceeds this threshold.",
    )
    parser.add_argument("--backup-suffix", default=".bak_eef9d_fix")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "meta" / "modality.json").exists()


def discover_default_datasets() -> list[Path]:
    datasets: list[Path] = []
    if is_lerobot_dataset(DEFAULT_TRIMMED):
        datasets.append(DEFAULT_TRIMMED)
    if DEFAULT_LEROBOT_ROOT.exists():
        datasets.extend(sorted(path for path in DEFAULT_LEROBOT_ROOT.glob("*/*") if is_lerobot_dataset(path)))
    return datasets


def slice_from_modality(modality: dict[str, Any], group: str, key: str) -> slice | None:
    group_payload = modality.get(group)
    if not isinstance(group_payload, dict):
        return None
    value = group_payload.get(key)
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return slice(start, end)
    return None


def require_slice(modality: dict[str, Any], group: str, *keys: str) -> slice:
    for key in keys:
        value = slice_from_modality(modality, group, key)
        if value is not None:
            return value
    raise RuntimeError(f"Missing modality slice {group}: {keys}")


def feature_names(info: dict[str, Any], column: str) -> list[str]:
    feature = (info.get("features") or {}).get(column) if isinstance(info.get("features"), dict) else None
    names = feature.get("names") if isinstance(feature, dict) else None
    return [str(name) for name in names] if isinstance(names, list) else []


def rot_suffixes(names: list[str], rot_slice: slice) -> tuple[str, ...] | None:
    if len(names) < rot_slice.stop:
        return None
    return tuple(name.rsplit(".", 1)[-1] for name in names[rot_slice.start : rot_slice.start + 6])


def rot_convention(names: list[str], rot_slice: slice, default: str) -> str:
    suffixes = rot_suffixes(names, rot_slice)
    if suffixes == ROT6D_ROW_MAJOR:
        return "row_major"
    if suffixes == ROT6D_COLUMN_MAJOR:
        return "column_major"
    return default


def action_in_state_frame(info: dict[str, Any]) -> bool:
    teleop = info.get("teleop_stack")
    if not isinstance(teleop, dict):
        return False
    return (
        teleop.get("rot6d_convention") == "row_major_first_two_rows_[r00,r01,r02,r10,r11,r12]"
        and isinstance(teleop.get("action_rot6d_frame_transform"), dict)
        and "in_state_frame" in str(teleop.get("arm_action_semantics") or "")
    )


def episode_data_path(dataset_dir: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    meta = episode.get("teleop_stack_metadata")
    if isinstance(meta, dict) and isinstance(meta.get("data_path"), str):
        return dataset_dir / str(meta["data_path"])
    episode_index = int(episode["episode_index"])
    chunk_size = max(1, int(info.get("chunks_size", 1000)))
    pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    return dataset_dir / pattern.format(episode_chunk=episode_index // chunk_size, episode_index=episode_index)


def load_dataset(dataset_dir: Path) -> DatasetSpec:
    info = read_json(dataset_dir / "meta" / "info.json")
    modality = read_json(dataset_dir / "meta" / "modality.json")
    episodes = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    state_eef = require_slice(modality, "state", "eef_9d")
    action_eef = require_slice(modality, "action", "eef_9d")
    state_pos = require_slice(modality, "state", "arm_eef_pos")
    action_pos = require_slice(modality, "action", "arm_eef_pos_target")
    state_rot = require_slice(modality, "state", "arm_eef_rot6d")
    action_rot = require_slice(modality, "action", "arm_eef_rot6d_target")
    state_names = feature_names(info, "observation.state")
    action_names = feature_names(info, "action")
    already = action_in_state_frame(info)
    return DatasetSpec(
        dataset_dir=dataset_dir,
        info=info,
        modality=modality,
        episodes=episodes,
        state_eef=state_eef,
        action_eef=action_eef,
        state_rot=state_rot,
        action_rot=action_rot,
        state_pos=state_pos,
        action_pos=action_pos,
        state_rot_convention=rot_convention(state_names, state_rot, "row_major"),
        action_rot_convention=rot_convention(action_names, action_rot, "row_major" if already else "column_major"),
        action_already_in_state_frame=already,
    )


def stack_vector_column(series: pd.Series) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise RuntimeError("Inconsistent vector widths")
    return np.vstack(values)


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def row_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    row0 = normalize(values[:, 0:3])
    row1 = values[:, 3:6] - np.sum(row0 * values[:, 3:6], axis=1, keepdims=True) * row0
    row1 = normalize(row1)
    row2 = np.cross(row0, row1)
    matrices = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, :] = row0
    matrices[:, 1, :] = row1
    matrices[:, 2, :] = row2
    return matrices


def col_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    col0 = normalize(values[:, 0:3])
    col1 = values[:, 3:6] - np.sum(col0 * values[:, 3:6], axis=1, keepdims=True) * col0
    col1 = normalize(col1)
    col2 = np.cross(col0, col1)
    matrices = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    matrices[:, :, 0] = col0
    matrices[:, :, 1] = col1
    matrices[:, :, 2] = col2
    return matrices


def matrix_to_row_rot6d(matrices: np.ndarray) -> np.ndarray:
    return np.concatenate([matrices[:, 0, :], matrices[:, 1, :]], axis=1)


def rot6d_to_matrix(rot6d: np.ndarray, convention: str) -> np.ndarray:
    return col_rot6d_to_matrix(rot6d) if convention == "column_major" else row_rot6d_to_matrix(rot6d)


def rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else math.nan


def angle_error_deg(state: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rel = state @ np.swapaxes(pred, 1, 2)
    traces = np.einsum("nii->n", rel)
    cos = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def signed_axis_matrices() -> list[tuple[str, np.ndarray]]:
    axes = "xyz"
    out: list[tuple[str, np.ndarray]] = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for row, col in enumerate(perm):
            base[row, col] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base * np.asarray(signs, dtype=np.float64)[:, None]
            names = []
            for row in range(3):
                col = int(np.argmax(np.abs(matrix[row])))
                names.append(("+" if matrix[row, col] > 0 else "-") + axes[col])
            out.append((",".join(names), matrix))
    return out


SIGNED_AXIS_MATRICES = signed_axis_matrices()
SIGNED_AXIS_MATRIX_BY_NAME = dict(SIGNED_AXIS_MATRICES)


def best_rotation_mapping(state_mats: np.ndarray, action_mats: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_mean = math.inf
    for transpose in (False, True):
        base = np.swapaxes(action_mats, 1, 2) if transpose else action_mats
        for left_name, left in SIGNED_AXIS_MATRICES:
            left_applied = left @ base
            for right_name, right in SIGNED_AXIS_MATRICES:
                pred = left_applied @ right
                angles = angle_error_deg(state_mats, pred)
                mean = float(np.mean(angles))
                if mean < best_mean:
                    best_mean = mean
                    best = {
                        "transpose": transpose,
                        "left": left_name,
                        "right": right_name,
                        "angle_mean": mean,
                        "angle_p50": float(np.percentile(angles, 50)),
                        "angle_p95": float(np.percentile(angles, 95)),
                        "angle_max": float(np.max(angles)),
                        "row_rot6d_rmse": rmse(matrix_to_row_rot6d(state_mats) - matrix_to_row_rot6d(pred)),
                    }
    return best


def mapping_equivalent(left_name: str, right_name: str, expected_left: str, expected_right: str) -> bool:
    matrices = dict(SIGNED_AXIS_MATRICES)
    left = matrices[left_name]
    right = matrices[right_name]
    expected_left_matrix = matrices[expected_left]
    expected_right_matrix = matrices[expected_right]
    return bool(
        (
            np.allclose(left, expected_left_matrix)
            and np.allclose(right, expected_right_matrix)
        )
        or (
            np.allclose(left, -expected_left_matrix)
            and np.allclose(right, -expected_right_matrix)
        )
    )


def rowmajor_identity_mapping(left_name: str, right_name: str) -> bool:
    return mapping_equivalent(left_name, right_name, "+x,+y,+z", "+x,+y,+z")


def corr_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    matrix = np.full((left.shape[1], right.shape[1]), np.nan, dtype=np.float64)
    for i in range(left.shape[1]):
        for j in range(right.shape[1]):
            a = left[:, i]
            b = right[:, j]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3 or np.std(a[mask]) <= 1e-12 or np.std(b[mask]) <= 1e-12:
                continue
            matrix[i, j] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return matrix


def best_position_mapping(action_pos: np.ndarray, state_pos: np.ndarray) -> dict[str, Any]:
    best_value = math.inf
    best: dict[str, Any] = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapped = action_pos[:, perm] * np.asarray(signs, dtype=np.float64)
            offset = np.nanmean(state_pos - mapped, axis=0)
            value = rmse(state_pos - (mapped + offset))
            if value < best_value:
                best_value = value
                best = {
                    "perm": list(perm),
                    "signs": [int(sign) for sign in signs],
                    "offset": [float(x) for x in offset],
                    "rmse": float(value),
                }
    return best


def inspect_episode(task: EpisodeTask) -> dict[str, Any]:
    df = pd.read_parquet(task.data_path)
    state = stack_vector_column(df["observation.state"])
    action = stack_vector_column(df["action"])
    n = min(state.shape[0], action.shape[0])
    state = state[:n]
    action = action[:n]
    if task.length > 0:
        n = min(n, task.length)
        state = state[:n]
        action = action[:n]
    sample_frames = getattr(inspect_episode, "sample_frames", 0)
    if sample_frames > 0 and n > sample_frames:
        idx = np.linspace(0, n - 1, sample_frames).round().astype(int)
        state = state[idx]
        action = action[idx]

    state_rot = state[:, slice(*task.state_rot)]
    action_rot = action[:, slice(*task.action_rot)]
    state_mats = rot6d_to_matrix(state_rot, task.state_rot_convention)
    action_mats = rot6d_to_matrix(action_rot, task.action_rot_convention)
    expected_action_mats = (
        action_mats
        if task.action_already_in_state_frame
        else ACTION_TO_STATE_LEFT @ action_mats @ ACTION_TO_STATE_RIGHT
    )
    direct_angles = angle_error_deg(state_mats, action_mats)
    expected_angles = angle_error_deg(state_mats, expected_action_mats)
    best = best_rotation_mapping(state_mats, action_mats)

    state_row = matrix_to_row_rot6d(state_mats)
    action_row = matrix_to_row_rot6d(action_mats)
    expected_action_row = matrix_to_row_rot6d(expected_action_mats)
    corr_direct = corr_matrix(state_row, action_row)
    corr_expected = corr_matrix(state_row, expected_action_row)

    state_pos = state[:, slice(*task.state_pos)]
    action_pos = action[:, slice(*task.action_pos)]
    pos_best = best_position_mapping(action_pos, state_pos)
    direct_pos_rmse = rmse(state_pos - action_pos)

    expected_match = (not best["transpose"]) and (
        (
            task.action_already_in_state_frame
            and rowmajor_identity_mapping(best["left"], best["right"])
        )
        or (
            (not task.action_already_in_state_frame)
            and mapping_equivalent(best["left"], best["right"], EXPECTED_LEFT_NAME, EXPECTED_RIGHT_NAME)
        )
    )
    return {
        "dataset": task.dataset_label,
        "data_path": str(task.data_path),
        "episode_index": task.episode_index,
        "frames": int(n),
        "state_rot_convention": task.state_rot_convention,
        "action_rot_convention": task.action_rot_convention,
        "action_already_in_state_frame": task.action_already_in_state_frame,
        "direct_angle_mean": float(np.mean(direct_angles)),
        "direct_angle_p95": float(np.percentile(direct_angles, 95)),
        "expected_angle_mean": float(np.mean(expected_angles)),
        "expected_angle_p95": float(np.percentile(expected_angles, 95)),
        "expected_row_rot6d_rmse": rmse(state_row - expected_action_row),
        "best_left": best["left"],
        "best_right": best["right"],
        "best_transpose": bool(best["transpose"]),
        "best_angle_mean": best["angle_mean"],
        "best_angle_p95": best["angle_p95"],
        "best_row_rot6d_rmse": best["row_rot6d_rmse"],
        "best_matches_expected": bool(expected_match),
        "corr_direct": corr_direct.round(6).tolist(),
        "corr_expected": corr_expected.round(6).tolist(),
        "direct_pos_rmse": direct_pos_rmse,
        "best_pos_perm": ",".join(str(x) for x in pos_best["perm"]),
        "best_pos_signs": ",".join(str(x) for x in pos_best["signs"]),
        "best_pos_offset": json.dumps(pos_best["offset"]),
        "best_pos_rmse": pos_best["rmse"],
    }


def init_worker(sample_frames: int) -> None:
    inspect_episode.sample_frames = int(sample_frames)


def dataset_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def make_tasks(specs: list[DatasetSpec]) -> list[EpisodeTask]:
    tasks: list[EpisodeTask] = []
    for spec in specs:
        label = dataset_label(spec.dataset_dir)
        for row in spec.episodes:
            episode_index = int(row["episode_index"])
            tasks.append(
                EpisodeTask(
                    dataset_dir=spec.dataset_dir,
                    dataset_label=label,
                    episode_index=episode_index,
                    data_path=episode_data_path(spec.dataset_dir, spec.info, row),
                    length=int(row.get("length", -1)),
                    state_eef=(spec.state_eef.start, spec.state_eef.stop),
                    action_eef=(spec.action_eef.start, spec.action_eef.stop),
                    state_rot=(spec.state_rot.start, spec.state_rot.stop),
                    action_rot=(spec.action_rot.start, spec.action_rot.stop),
                    state_pos=(spec.state_pos.start, spec.state_pos.stop),
                    action_pos=(spec.action_pos.start, spec.action_pos.stop),
                    state_rot_convention=spec.state_rot_convention,
                    action_rot_convention=spec.action_rot_convention,
                    action_already_in_state_frame=spec.action_already_in_state_frame,
                )
            )
    return tasks


def run_inspect(tasks: list[EpisodeTask], *, jobs: int, sample_frames: int) -> list[dict[str, Any]]:
    if jobs == 1:
        init_worker(sample_frames)
        return [inspect_episode(task) for task in tasks]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=init_worker, initargs=(sample_frames,)) as pool:
        futures = [pool.submit(inspect_episode, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: (row["dataset"], int(row["episode_index"])))


def update_feature_names_to_row_major(names: list[str], rot_slice: slice) -> list[str]:
    if len(names) < rot_slice.stop:
        return names
    out = list(names)
    prefix = out[rot_slice.start].rsplit(".", 1)[0]
    for offset, suffix in enumerate(ROT6D_ROW_MAJOR):
        out[rot_slice.start + offset] = f"{prefix}.{suffix}"
    return out


def fix_dataset_meta(spec: DatasetSpec, *, apply: bool, repair_strategy: str) -> dict[str, Any]:
    info = json.loads(json.dumps(spec.info))
    features = info.setdefault("features", {})
    if isinstance(features, dict):
        for column, rot_slice in (("observation.state", spec.state_rot), ("action", spec.action_rot)):
            feature = features.get(column)
            if isinstance(feature, dict) and isinstance(feature.get("names"), list):
                feature["names"] = update_feature_names_to_row_major([str(x) for x in feature["names"]], rot_slice)
    teleop = info.setdefault("teleop_stack", {})
    if isinstance(teleop, dict):
        if teleop.get("arm_action_semantics") == "absolute_wrist_pose_xyz_rot6d_target":
            teleop["arm_action_semantics"] = "absolute_wrist_pose_xyz_rot6d_target_in_state_frame"
        teleop["rot6d_convention"] = "row_major_first_two_rows_[r00,r01,r02,r10,r11,r12]"
        if repair_strategy == "best-per-episode":
            teleop["action_rot6d_frame_transform"] = {
                "formula": "per_episode_best_matrix_mapping_applied_by_repair",
                "source": "tools/data_chain/check_and_fix_lerobot_eef9d.py",
                "note": "Original action rotations used mixed frame mappings; each episode was repaired with its inspected best L/R mapping.",
            }
        else:
            teleop["action_rot6d_frame_transform"] = {
                "formula": "R_state ~= L @ R_action @ R",
                "left_matrix_rows": ACTION_TO_STATE_LEFT.tolist(),
                "right_matrix_rows": ACTION_TO_STATE_RIGHT.tolist(),
                "rot6d_row_major_mapping": EXPECTED_ACTION_ROT6D_MAPPING,
            }
        teleop["eef9d_repair"] = {
            "script": "tools/data_chain/check_and_fix_lerobot_eef9d.py",
            "strategy": repair_strategy,
            "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    if apply:
        write_json(spec.dataset_dir / "meta" / "info.json", info)
    return {"dataset": dataset_label(spec.dataset_dir), "meta_updated": info != spec.info, "applied": bool(apply)}


def fix_episode(task: EpisodeTask, *, apply: bool, backup_suffix: str) -> dict[str, Any]:
    df = pd.read_parquet(task.data_path)
    state = stack_vector_column(df["observation.state"])
    action = stack_vector_column(df["action"])
    state_rot_slice = slice(*task.state_rot)
    action_rot_slice = slice(*task.action_rot)
    state_mats = rot6d_to_matrix(state[:, state_rot_slice], task.state_rot_convention)
    action_mats = rot6d_to_matrix(action[:, action_rot_slice], task.action_rot_convention)
    if task.fix_left is not None and task.fix_right is not None:
        base = np.swapaxes(action_mats, 1, 2) if task.fix_transpose else action_mats
        action_mats = SIGNED_AXIS_MATRIX_BY_NAME[task.fix_left] @ base @ SIGNED_AXIS_MATRIX_BY_NAME[task.fix_right]
    elif not task.action_already_in_state_frame:
        action_mats = ACTION_TO_STATE_LEFT @ action_mats @ ACTION_TO_STATE_RIGHT
    state[:, state_rot_slice] = matrix_to_row_rot6d(state_mats)
    action[:, action_rot_slice] = matrix_to_row_rot6d(action_mats)
    changed = True
    if apply:
        out_df = df.copy()
        out_df["observation.state"] = [row.astype(np.float32) for row in state]
        out_df["action"] = [row.astype(np.float32) for row in action]
        backup = task.data_path.with_name(task.data_path.name + backup_suffix)
        if not backup.exists():
            shutil.copy2(task.data_path, backup)
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=task.data_path.stem + ".tmp.",
            dir=str(task.data_path.parent),
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        try:
            out_df.to_parquet(tmp_path, index=False)
            tmp_path.replace(task.data_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return {
        "dataset": task.dataset_label,
        "episode_index": task.episode_index,
        "data_path": str(task.data_path),
        "changed": changed,
        "applied": bool(apply),
        "fix_left": task.fix_left,
        "fix_right": task.fix_right,
        "fix_transpose": bool(task.fix_transpose),
    }


def run_fix(
    tasks: list[EpisodeTask],
    specs: list[DatasetSpec],
    *,
    jobs: int,
    apply: bool,
    backup_suffix: str,
    repair_strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if jobs == 1:
        rows = [fix_episode(task, apply=apply, backup_suffix=backup_suffix) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(fix_episode, task, apply=apply, backup_suffix=backup_suffix) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
        rows = sorted(rows, key=lambda row: (row["dataset"], int(row["episode_index"])))
    rows.extend(fix_dataset_meta(spec, apply=apply, repair_strategy=repair_strategy) for spec in specs)
    return rows


def attach_best_episode_mappings(tasks: list[EpisodeTask], inspect_rows: list[dict[str, Any]]) -> list[EpisodeTask]:
    by_key = {
        (str(row["dataset"]), int(row["episode_index"])): row
        for row in inspect_rows
    }
    mapped: list[EpisodeTask] = []
    for task in tasks:
        row = by_key[(task.dataset_label, int(task.episode_index))]
        mapped.append(
            replace(
                task,
                fix_left=str(row["best_left"]),
                fix_right=str(row["best_right"]),
                fix_transpose=bool(row["best_transpose"]),
            )
        )
    return mapped


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    dataset_dirs: list[Path] = []
    if args.include_defaults or not args.dataset_dir:
        dataset_dirs.extend(discover_default_datasets())
    if args.dataset_dir:
        dataset_dirs.extend(path.expanduser().resolve() for path in args.dataset_dir)
    dataset_dirs = sorted({path.resolve() for path in dataset_dirs if is_lerobot_dataset(path)})
    if not dataset_dirs:
        raise RuntimeError("No LeRobot datasets found.")

    specs = [load_dataset(path) for path in dataset_dirs]
    tasks = make_tasks(specs)
    jobs = int(args.jobs) if int(args.jobs) > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, max(1, len(tasks))))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inspect":
        rows = run_inspect(tasks, jobs=jobs, sample_frames=max(0, int(args.sample_frames)))
        episode_df = pd.DataFrame(rows)
        episode_df.to_csv(output_dir / "episode_eef9d_alignment.csv", index=False)
        dataset_df = (
            episode_df.groupby("dataset")
            .agg(
                episodes=("episode_index", "count"),
                expected_match_ratio=("best_matches_expected", "mean"),
                direct_angle_mean=("direct_angle_mean", "mean"),
                expected_angle_mean=("expected_angle_mean", "mean"),
                best_angle_mean=("best_angle_mean", "mean"),
                expected_row_rot6d_rmse=("expected_row_rot6d_rmse", "mean"),
                best_left=("best_left", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
                best_right=("best_right", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
                best_transpose=("best_transpose", lambda x: bool(x.mode().iloc[0]) if len(x.mode()) else False),
                state_rot_convention=("state_rot_convention", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
                action_rot_convention=("action_rot_convention", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
                action_already_in_state_frame=("action_already_in_state_frame", "mean"),
            )
            .reset_index()
        )
        dataset_df.to_csv(output_dir / "dataset_eef9d_alignment.csv", index=False)
        all_expected = bool(episode_df["best_matches_expected"].all()) if len(episode_df) else False
        summary = {
            "mode": "inspect",
            "datasets": len(dataset_dirs),
            "episodes": len(tasks),
            "jobs": jobs,
            "all_episodes_match_expected_mapping": all_expected,
            "expected_mapping": {
                "state_rot6d": list(ROT6D_ROW_MAJOR),
                "action_formula": "R_state ~= L @ R_action @ R",
                "left_rows": EXPECTED_LEFT_NAME,
                "right_rows": EXPECTED_RIGHT_NAME,
                "action_rot6d_row_major_mapping": EXPECTED_ACTION_ROT6D_MAPPING,
            },
            "dataset_summary": dataset_df.to_dict(orient="records"),
            "outputs": {
                "episode_csv": str(output_dir / "episode_eef9d_alignment.csv"),
                "dataset_csv": str(output_dir / "dataset_eef9d_alignment.csv"),
                "summary_json": str(output_dir / "summary.json"),
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        return 0

    inspect_rows = run_inspect(tasks, jobs=jobs, sample_frames=max(0, int(args.sample_frames)))
    inspect_df = pd.DataFrame(inspect_rows)
    inspect_df.to_csv(output_dir / "pre_fix_episode_eef9d_alignment.csv", index=False)
    mixed = bool(len(inspect_df) and not inspect_df["best_matches_expected"].all())
    too_uncertain = inspect_df["best_angle_mean"] > float(args.max_best_angle_mean) if len(inspect_df) else pd.Series(dtype=bool)
    if bool(too_uncertain.any()):
        uncertain = inspect_df.loc[too_uncertain, ["dataset", "episode_index", "best_angle_mean", "best_left", "best_right"]]
        uncertain.to_csv(output_dir / "uncertain_episode_mappings.csv", index=False)
        summary = {
            "mode": "fix",
            "apply": bool(args.apply),
            "aborted": True,
            "reason": "best_mapping_angle_too_high",
            "episodes": len(tasks),
            "uncertain_episodes": int(too_uncertain.sum()),
            "max_best_angle_mean": float(args.max_best_angle_mean),
            "uncertain_csv": str(output_dir / "uncertain_episode_mappings.csv"),
            "pre_fix_csv": str(output_dir / "pre_fix_episode_eef9d_alignment.csv"),
        }
        (output_dir / "fix_summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        return 2

    if mixed and args.fix_strategy == "expected" and not args.allow_mixed_mappings:
        summary = {
            "mode": "fix",
            "apply": bool(args.apply),
            "aborted": True,
            "reason": "non_expected_mappings_found",
            "episodes": len(tasks),
            "non_expected_episodes": int((~inspect_df["best_matches_expected"]).sum()),
            "pre_fix_csv": str(output_dir / "pre_fix_episode_eef9d_alignment.csv"),
            "hint": "Inspect the non-expected rows first, or rerun with --allow-mixed-mappings after explicit confirmation.",
        }
        (output_dir / "fix_summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        return 2

    fix_tasks = attach_best_episode_mappings(tasks, inspect_rows) if args.fix_strategy == "best-per-episode" else tasks
    rows = run_fix(
        fix_tasks,
        specs,
        jobs=jobs,
        apply=bool(args.apply),
        backup_suffix=str(args.backup_suffix),
        repair_strategy=str(args.fix_strategy),
    )
    fix_df = pd.DataFrame(rows)
    fix_df.to_csv(output_dir / "fix_eef9d_results.csv", index=False)
    if "changed" in fix_df:
        changed_rows = sum(value is True or str(value).lower() == "true" for value in fix_df["changed"])
    else:
        changed_rows = 0
    summary = {
        "mode": "fix",
        "apply": bool(args.apply),
        "datasets": len(dataset_dirs),
        "episodes": len(tasks),
        "jobs": jobs,
        "fix_strategy": str(args.fix_strategy),
        "changed_rows": int(changed_rows) if len(fix_df) else 0,
        "output_csv": str(output_dir / "fix_eef9d_results.csv"),
        "pre_fix_csv": str(output_dir / "pre_fix_episode_eef9d_alignment.csv"),
    }
    (output_dir / "fix_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
