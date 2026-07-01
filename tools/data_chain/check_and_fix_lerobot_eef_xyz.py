#!/usr/bin/env python3
"""Inspect and repair LeRobot EEF XYZ action/state axis order."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "missions" / "nero" / "mission2" / "trimmed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "missions" / "nero" / "mission2" / "quality" / "eef_xyz_alignment"

EXPECTED_PERM = (2, 0, 1)
EXPECTED_SIGNS = (1.0, 1.0, 1.0)
IDENTITY_PERM = (0, 1, 2)
IDENTITY_SIGNS = (1.0, 1.0, 1.0)
POSITION_MAPPING_TEXT = "state xyz ~= action raw [z, x, y]; repaired action order is [x, y, z] in the state frame"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_dir: Path
    dataset_label: str
    info: dict[str, Any]
    modality: dict[str, Any]
    episodes: list[dict[str, Any]]
    state_pos: tuple[int, int]
    action_pos: tuple[int, int]


@dataclass(frozen=True)
class EpisodeTask:
    dataset_dir: Path
    dataset_label: str
    episode_index: int
    data_path: Path
    length: int
    state_pos: tuple[int, int]
    action_pos: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("inspect", "fix"), default="inspect")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite parquet/meta files in fix mode.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker count. 0 uses all CPU cores.")
    parser.add_argument("--sample-frames", type=int, default=0, help="Uniform sample per episode; 0 means all frames.")
    parser.add_argument("--backup-suffix", default=".bak_eef_xyz_fix")
    parser.add_argument(
        "--max-best-offset-rmse",
        type=float,
        default=0.04,
        help="Abort fix if any episode's best offset-corrected XYZ RMSE is above this value.",
    )
    parser.add_argument(
        "--allow-non-expected",
        action="store_true",
        help="Allow applying even if the inspected best mapping is not the expected [z,x,y] mapping.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def slice_from_modality(modality: dict[str, Any], group: str, key: str) -> tuple[int, int] | None:
    payload = modality.get(group)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if isinstance(start, int) and isinstance(end, int) and end - start == 3:
        return start, end
    return None


def feature_names(info: dict[str, Any], column: str) -> list[str]:
    feature = (info.get("features") or {}).get(column) if isinstance(info.get("features"), dict) else None
    names = feature.get("names") if isinstance(feature, dict) else None
    return [str(name) for name in names] if isinstance(names, list) else []


def indices_from_names(names: list[str], expected: tuple[str, str, str]) -> tuple[int, int] | None:
    lookup = {name: idx for idx, name in enumerate(names)}
    indices = [lookup.get(name) for name in expected]
    if any(index is None for index in indices):
        return None
    values = [int(index) for index in indices if index is not None]
    if values != list(range(values[0], values[0] + 3)):
        raise RuntimeError(f"Non-contiguous XYZ feature indices for {expected}: {values}")
    return values[0], values[0] + 3


def resolve_pos_slices(info: dict[str, Any], modality: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    state_pos = slice_from_modality(modality, "state", "arm_eef_pos")
    action_pos = slice_from_modality(modality, "action", "arm_eef_pos_target")
    if state_pos is not None and action_pos is not None:
        return state_pos, action_pos

    state_pos = indices_from_names(
        feature_names(info, "observation.state"),
        ("arm_eef_pos.x", "arm_eef_pos.y", "arm_eef_pos.z"),
    )
    action_pos = indices_from_names(
        feature_names(info, "action"),
        ("arm_eef_pos_target.x", "arm_eef_pos_target.y", "arm_eef_pos_target.z"),
    )
    if state_pos is None or action_pos is None:
        raise RuntimeError("Could not resolve EEF XYZ state/action slices from modality.json or feature names.")
    return state_pos, action_pos


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
    state_pos, action_pos = resolve_pos_slices(info, modality)
    return DatasetSpec(
        dataset_dir=dataset_dir,
        dataset_label=dataset_label(dataset_dir),
        info=info,
        modality=modality,
        episodes=read_jsonl(dataset_dir / "meta" / "episodes.jsonl"),
        state_pos=state_pos,
        action_pos=action_pos,
    )


def make_tasks(spec: DatasetSpec) -> list[EpisodeTask]:
    tasks: list[EpisodeTask] = []
    for episode in spec.episodes:
        tasks.append(
            EpisodeTask(
                dataset_dir=spec.dataset_dir,
                dataset_label=spec.dataset_label,
                episode_index=int(episode["episode_index"]),
                data_path=episode_data_path(spec.dataset_dir, spec.info, episode),
                length=int(episode.get("length", -1)),
                state_pos=spec.state_pos,
                action_pos=spec.action_pos,
            )
        )
    return tasks


def stack_vector_column(series: pd.Series) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise RuntimeError("Inconsistent vector widths")
    return np.vstack(values)


def rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else math.nan


def corr_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    matrix = np.full((left.shape[1], right.shape[1]), np.nan, dtype=np.float64)
    for row in range(left.shape[1]):
        for col in range(right.shape[1]):
            a = left[:, row]
            b = right[:, col]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3 or np.std(a[mask]) <= 1e-12 or np.std(b[mask]) <= 1e-12:
                continue
            matrix[row, col] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return matrix


def map_position(action_pos: np.ndarray, perm: tuple[int, int, int], signs: tuple[float, float, float]) -> np.ndarray:
    return action_pos[:, perm] * np.asarray(signs, dtype=np.float64)


def best_position_mapping(state_pos: np.ndarray, action_pos: np.ndarray, *, allow_offset: bool) -> dict[str, Any]:
    best_value = math.inf
    best: dict[str, Any] = {}
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapped = map_position(action_pos, perm, signs)
            offset = np.nanmean(state_pos - mapped, axis=0) if allow_offset else np.zeros(3, dtype=np.float64)
            value = rmse(state_pos - (mapped + offset))
            if value < best_value:
                best_value = value
                best = {
                    "perm": tuple(int(x) for x in perm),
                    "signs": tuple(float(x) for x in signs),
                    "offset": [float(x) for x in offset],
                    "rmse": float(value),
                }
    return best


def parse_sample(state: np.ndarray, action: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(state.shape[0], action.shape[0])
    state = state[:n]
    action = action[:n]
    if length > 0:
        n = min(n, length)
        state = state[:n]
        action = action[:n]
    sample_frames = getattr(inspect_episode, "sample_frames", 0)
    if sample_frames > 0 and n > sample_frames:
        idx = np.linspace(0, n - 1, sample_frames).round().astype(int)
        state = state[idx]
        action = action[idx]
    return state, action


def inspect_episode(task: EpisodeTask) -> dict[str, Any]:
    df = pd.read_parquet(task.data_path, columns=["observation.state", "action"])
    state = stack_vector_column(df["observation.state"])
    action = stack_vector_column(df["action"])
    state, action = parse_sample(state, action, task.length)
    state_pos = state[:, slice(*task.state_pos)]
    action_pos = action[:, slice(*task.action_pos)]

    expected = map_position(action_pos, EXPECTED_PERM, EXPECTED_SIGNS)
    expected_offset = np.nanmean(state_pos - expected, axis=0)
    best_offset = best_position_mapping(state_pos, action_pos, allow_offset=True)
    best_direct = best_position_mapping(state_pos, action_pos, allow_offset=False)
    direct_corr = corr_matrix(state_pos, action_pos)
    expected_corr = corr_matrix(state_pos, expected)
    expected_corr_diag = np.diag(expected_corr)
    direct_corr_diag = np.diag(direct_corr)
    best_perm = tuple(best_offset["perm"])
    best_signs = tuple(best_offset["signs"])
    return {
        "dataset": task.dataset_label,
        "data_path": str(task.data_path),
        "episode_index": task.episode_index,
        "frames": int(state_pos.shape[0]),
        "direct_pos_rmse": rmse(state_pos - action_pos),
        "expected_zxy_pos_rmse": rmse(state_pos - expected),
        "expected_zxy_offset": json.dumps([float(x) for x in expected_offset]),
        "expected_zxy_offset_rmse": rmse(state_pos - (expected + expected_offset)),
        "direct_corr_diag_mean": float(np.nanmean(direct_corr_diag)),
        "expected_zxy_corr_diag_mean": float(np.nanmean(expected_corr_diag)),
        "best_pos_perm": ",".join(str(x) for x in best_perm),
        "best_pos_signs": ",".join(str(int(x)) for x in best_signs),
        "best_pos_offset": json.dumps(best_offset["offset"]),
        "best_pos_rmse": float(best_offset["rmse"]),
        "best_direct_pos_perm": ",".join(str(x) for x in best_direct["perm"]),
        "best_direct_pos_signs": ",".join(str(int(x)) for x in best_direct["signs"]),
        "best_direct_pos_rmse": float(best_direct["rmse"]),
        "best_matches_expected_zxy": bool(best_perm == EXPECTED_PERM and best_signs == EXPECTED_SIGNS),
        "best_matches_identity": bool(best_perm == IDENTITY_PERM and best_signs == IDENTITY_SIGNS),
        "direct_corr": json.dumps(direct_corr.round(6).tolist()),
        "expected_zxy_corr": json.dumps(expected_corr.round(6).tolist()),
    }


def init_worker(sample_frames: int) -> None:
    inspect_episode.sample_frames = int(sample_frames)


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


def fix_episode(task: EpisodeTask, *, apply: bool, backup_suffix: str) -> dict[str, Any]:
    df = pd.read_parquet(task.data_path)
    action = stack_vector_column(df["action"])
    action_pos_slice = slice(*task.action_pos)
    before = action[:, action_pos_slice].copy()
    action[:, action_pos_slice] = map_position(before, EXPECTED_PERM, EXPECTED_SIGNS)
    changed = not np.array_equal(before, action[:, action_pos_slice])
    if apply and changed:
        out_df = df.copy()
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
        "changed": bool(changed),
        "applied": bool(apply and changed),
        "mapping": "[z,x,y]",
    }


def run_fix(tasks: list[EpisodeTask], *, jobs: int, apply: bool, backup_suffix: str) -> list[dict[str, Any]]:
    if jobs == 1:
        return [fix_episode(task, apply=apply, backup_suffix=backup_suffix) for task in tasks]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fix_episode, task, apply=apply, backup_suffix=backup_suffix) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: (row["dataset"], int(row["episode_index"])))


def vector_stats(matrix: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.nanmean(matrix, axis=0).astype(float).tolist(),
        "std": np.nanstd(matrix, axis=0).astype(float).tolist(),
        "min": np.nanmin(matrix, axis=0).astype(float).tolist(),
        "max": np.nanmax(matrix, axis=0).astype(float).tolist(),
        "q01": np.nanquantile(matrix, 0.01, axis=0).astype(float).tolist(),
        "q99": np.nanquantile(matrix, 0.99, axis=0).astype(float).tolist(),
    }


def recompute_action_stats(spec: DatasetSpec) -> bool:
    stats_path = spec.dataset_dir / "meta" / "stats.json"
    if not stats_path.exists():
        return False
    chunks: list[np.ndarray] = []
    for task in make_tasks(spec):
        df = pd.read_parquet(task.data_path, columns=["action"])
        chunks.append(stack_vector_column(df["action"]))
    if not chunks:
        return False
    stats = read_json(stats_path)
    stats["action"] = vector_stats(np.vstack(chunks))
    stats_path.write_text(json.dumps(stats, ensure_ascii=True, indent=4) + "\n", encoding="utf-8")
    return True


def update_dataset_meta(spec: DatasetSpec, *, apply: bool) -> dict[str, Any]:
    info = json.loads(json.dumps(spec.info))
    teleop = info.setdefault("teleop_stack", {})
    if isinstance(teleop, dict):
        if teleop.get("arm_action_semantics") == "absolute_wrist_pose_xyz_rot6d_target":
            teleop["arm_action_semantics"] = "absolute_wrist_pose_xyz_rot6d_target_in_state_frame"
        teleop["action_position_frame_transform"] = {
            "formula": POSITION_MAPPING_TEXT,
            "value_order": ["raw_z", "raw_x", "raw_y"],
            "applies_to": "action arm_eef_pos_target [x,y,z]",
            "offset_applied": False,
            "source": "tools/data_chain/check_and_fix_lerobot_eef_xyz.py",
        }
        teleop["eef_xyz_repair"] = {
            "script": "tools/data_chain/check_and_fix_lerobot_eef_xyz.py",
            "strategy": "fixed_axis_permutation_[z,x,y]_no_offset",
            "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    if apply:
        write_json(spec.dataset_dir / "meta" / "info.json", info)
    return {"dataset": spec.dataset_label, "meta_updated": info != spec.info, "applied": bool(apply)}


def dataset_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    return (
        episode_df.groupby("dataset")
        .agg(
            episodes=("episode_index", "count"),
            best_expected_ratio=("best_matches_expected_zxy", "mean"),
            best_identity_ratio=("best_matches_identity", "mean"),
            direct_pos_rmse_mean=("direct_pos_rmse", "mean"),
            expected_zxy_pos_rmse_mean=("expected_zxy_pos_rmse", "mean"),
            expected_zxy_offset_rmse_mean=("expected_zxy_offset_rmse", "mean"),
            best_pos_rmse_mean=("best_pos_rmse", "mean"),
            best_pos_rmse_max=("best_pos_rmse", "max"),
            expected_zxy_corr_diag_mean=("expected_zxy_corr_diag_mean", "mean"),
            best_pos_perm=("best_pos_perm", lambda values: values.mode().iloc[0] if len(values.mode()) else ""),
            best_pos_signs=("best_pos_signs", lambda values: values.mode().iloc[0] if len(values.mode()) else ""),
        )
        .reset_index()
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not (dataset_dir / "meta" / "info.json").exists():
        raise RuntimeError(f"Not a LeRobot dataset: {dataset_dir}")
    spec = load_dataset(dataset_dir)
    tasks = make_tasks(spec)
    jobs = int(args.jobs) if int(args.jobs) > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, max(1, len(tasks))))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inspect":
        rows = run_inspect(tasks, jobs=jobs, sample_frames=max(0, int(args.sample_frames)))
        episode_df = pd.DataFrame(rows)
        dataset_df = dataset_summary(episode_df)
        episode_df.to_csv(output_dir / "episode_eef_xyz_alignment.csv", index=False)
        dataset_df.to_csv(output_dir / "dataset_eef_xyz_alignment.csv", index=False)
        summary = {
            "mode": "inspect",
            "dataset": spec.dataset_label,
            "episodes": len(tasks),
            "jobs": jobs,
            "expected_mapping": {
                "formula": POSITION_MAPPING_TEXT,
                "perm": list(EXPECTED_PERM),
                "signs": [int(x) for x in EXPECTED_SIGNS],
                "offset_applied_in_fix": False,
            },
            "all_episodes_best_match_expected_zxy": bool(episode_df["best_matches_expected_zxy"].all()) if len(episode_df) else False,
            "dataset_summary": dataset_df.to_dict(orient="records"),
            "outputs": {
                "episode_csv": str(output_dir / "episode_eef_xyz_alignment.csv"),
                "dataset_csv": str(output_dir / "dataset_eef_xyz_alignment.csv"),
                "summary_json": str(output_dir / "summary.json"),
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        return 0

    inspect_rows = run_inspect(tasks, jobs=jobs, sample_frames=max(0, int(args.sample_frames)))
    inspect_df = pd.DataFrame(inspect_rows)
    inspect_df.to_csv(output_dir / "pre_fix_episode_eef_xyz_alignment.csv", index=False)
    not_expected = ~inspect_df["best_matches_expected_zxy"] if len(inspect_df) else pd.Series(dtype=bool)
    too_uncertain = inspect_df["best_pos_rmse"] > float(args.max_best_offset_rmse) if len(inspect_df) else pd.Series(dtype=bool)
    if bool(too_uncertain.any()) or (bool(not_expected.any()) and not args.allow_non_expected):
        if bool(too_uncertain.any()):
            inspect_df.loc[too_uncertain].to_csv(output_dir / "uncertain_episode_xyz_mappings.csv", index=False)
        if bool(not_expected.any()):
            inspect_df.loc[not_expected].to_csv(output_dir / "non_expected_episode_xyz_mappings.csv", index=False)
        summary = {
            "mode": "fix",
            "apply": bool(args.apply),
            "aborted": True,
            "reason": "uncertain_or_non_expected_xyz_mapping",
            "episodes": len(tasks),
            "non_expected_episodes": int(not_expected.sum()) if len(inspect_df) else 0,
            "uncertain_episodes": int(too_uncertain.sum()) if len(inspect_df) else 0,
            "pre_fix_csv": str(output_dir / "pre_fix_episode_eef_xyz_alignment.csv"),
        }
        (output_dir / "fix_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
        return 2

    fix_rows = run_fix(tasks, jobs=jobs, apply=bool(args.apply), backup_suffix=str(args.backup_suffix))
    fix_df = pd.DataFrame(fix_rows)
    fix_df.to_csv(output_dir / "fix_eef_xyz_results.csv", index=False)
    meta_row = update_dataset_meta(spec, apply=bool(args.apply))
    stats_updated = recompute_action_stats(spec) if args.apply else False
    post_summary: dict[str, Any] | None = None
    if args.apply:
        post_rows = run_inspect(tasks, jobs=jobs, sample_frames=max(0, int(args.sample_frames)))
        post_df = pd.DataFrame(post_rows)
        post_df.to_csv(output_dir / "post_fix_episode_eef_xyz_alignment.csv", index=False)
        post_dataset_df = dataset_summary(post_df)
        post_dataset_df.to_csv(output_dir / "post_fix_dataset_eef_xyz_alignment.csv", index=False)
        post_summary = {
            "all_episodes_best_match_identity": bool(post_df["best_matches_identity"].all()) if len(post_df) else False,
            "dataset_summary": post_dataset_df.to_dict(orient="records"),
        }

    changed_rows = int(sum(value is True or str(value).lower() == "true" for value in fix_df.get("changed", [])))
    summary = {
        "mode": "fix",
        "apply": bool(args.apply),
        "dataset": spec.dataset_label,
        "episodes": len(tasks),
        "jobs": jobs,
        "changed_episode_rows": changed_rows,
        "meta": meta_row,
        "action_stats_updated": bool(stats_updated),
        "pre_fix_dataset_summary": dataset_summary(inspect_df).to_dict(orient="records"),
        "post_fix": post_summary,
        "outputs": {
            "pre_fix_csv": str(output_dir / "pre_fix_episode_eef_xyz_alignment.csv"),
            "fix_csv": str(output_dir / "fix_eef_xyz_results.csv"),
            "summary_json": str(output_dir / "fix_summary.json"),
        },
    }
    (output_dir / "fix_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
