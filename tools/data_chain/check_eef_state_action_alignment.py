#!/usr/bin/env python3
"""Batch-check EEF state/action alignment in LeRobot datasets."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "missions" / "nero" / "mission2" / "trimmed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "missions" / "nero" / "mission2" / "quality" / "eef_alignment_check"
COMPACT_DATE_RE = re.compile(r"(20\d{6})")


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    source_date: str
    data_path: Path
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-lag", type=int, default=20, help="Search +/- this many frames for same-axis lag.")
    parser.add_argument(
        "--suspicious-ratio",
        type=float,
        default=0.60,
        help="Flag when best signed-permutation RMSE is below direct RMSE times this ratio.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=0,
        help="Optional uniform frame sample per episode for faster scans. 0 means all frames.",
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


def compact_date_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def source_date(row: dict[str, Any]) -> str:
    meta = row.get("teleop_stack_metadata")
    if not isinstance(meta, dict):
        meta = {}
    for value in (
        meta.get("raw_episode_id"),
        meta.get("raw_episode_dir"),
        meta.get("source_dataset"),
        meta.get("trimmed_at_utc"),
    ):
        if not isinstance(value, str):
            continue
        compact = COMPACT_DATE_RE.search(value)
        if compact:
            return compact_date_to_iso(compact.group(1))
        if re.match(r"\d{4}-\d{2}-\d{2}", value):
            return value[:10]
    return "unknown"


def episode_data_path(dataset_dir: Path, info: dict[str, Any], row: dict[str, Any]) -> Path:
    meta = row.get("teleop_stack_metadata")
    if isinstance(meta, dict) and isinstance(meta.get("data_path"), str):
        return dataset_dir / str(meta["data_path"])
    episode_index = int(row["episode_index"])
    chunk_size = int(info.get("chunks_size", 1000))
    pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    return dataset_dir / pattern.format(
        episode_chunk=episode_index // max(1, chunk_size),
        episode_index=episode_index,
    )


def discover_episodes(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[EpisodeInfo]]:
    info = read_json(dataset_dir / "meta" / "info.json")
    modality = read_json(dataset_dir / "meta" / "modality.json")
    rows = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes = [
        EpisodeInfo(
            episode_index=int(row["episode_index"]),
            source_date=source_date(row),
            data_path=episode_data_path(dataset_dir, info, row),
            length=int(row.get("length", -1)),
        )
        for row in rows
    ]
    return info, modality, episodes


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in series.to_numpy()]
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise RuntimeError(f"{column_name} has inconsistent vector widths")
    return np.vstack(values)


def slice_from_modality(modality: dict[str, Any], section: str, keys: tuple[str, ...]) -> slice:
    section_payload = modality.get(section)
    if not isinstance(section_payload, dict):
        raise RuntimeError(f"modality.json is missing {section!r}")
    for key in keys:
        value = section_payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("start"), int) and isinstance(value.get("end"), int):
            return slice(int(value["start"]), int(value["end"]))
    raise RuntimeError(f"modality.json {section!r} missing any of {keys}")


def rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(finite))))


def lagged_pair(action: np.ndarray, state: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(action.shape[0], state.shape[0])
    action = action[:n]
    state = state[:n]
    if lag > 0:
        return action[:-lag], state[lag:]
    if lag < 0:
        return action[-lag:], state[: n + lag]
    return action, state


def best_lag_rmse(action: np.ndarray, state: np.ndarray, max_lag: int) -> tuple[int, float]:
    best_lag = 0
    best_rmse = math.inf
    for lag in range(-max_lag, max_lag + 1):
        a, s = lagged_pair(action, state, lag)
        if a.shape[0] < 2 or s.shape[0] < 2:
            continue
        value = rmse(s - a)
        if math.isfinite(value) and value < best_rmse:
            best_lag = lag
            best_rmse = value
    return best_lag, best_rmse


def signed_permutation_candidates(dim: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [
        (perm, signs)
        for perm in itertools.permutations(range(dim))
        for signs in itertools.product((-1, 1), repeat=dim)
    ]


def apply_signed_permutation(values: np.ndarray, perm: tuple[int, ...], signs: tuple[int, ...]) -> np.ndarray:
    return values[:, perm] * np.asarray(signs, dtype=np.float64)


def best_signed_permutation_with_offset(
    action: np.ndarray,
    state: np.ndarray,
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict[str, Any]:
    n = min(action.shape[0], state.shape[0])
    action = action[:n]
    state = state[:n]
    best: dict[str, Any] = {
        "rmse": math.nan,
        "perm": (),
        "signs": (),
        "offset": [],
    }
    best_value = math.inf
    for perm, signs in candidates:
        mapped = apply_signed_permutation(action, perm, signs)
        offset = np.nanmean(state - mapped, axis=0)
        value = rmse(state - (mapped + offset))
        if math.isfinite(value) and value < best_value:
            best_value = value
            best = {
                "rmse": value,
                "perm": perm,
                "signs": signs,
                "offset": offset.tolist(),
            }
    return best


def corr_diag(action: np.ndarray, state: np.ndarray) -> list[float]:
    values: list[float] = []
    n = min(action.shape[0], state.shape[0])
    for dim in range(min(action.shape[1], state.shape[1])):
        a = action[:n, dim]
        s = state[:n, dim]
        mask = np.isfinite(a) & np.isfinite(s)
        if mask.sum() < 3:
            values.append(math.nan)
            continue
        if np.std(a[mask]) <= 1e-12 or np.std(s[mask]) <= 1e-12:
            values.append(math.nan)
            continue
        values.append(float(np.corrcoef(a[mask], s[mask])[0, 1]))
    return values


def summarize_episode(
    episode: EpisodeInfo,
    *,
    state_eef_slice: slice,
    action_eef_slice: slice,
    state_pos_slice: slice,
    action_pos_slice: slice,
    max_lag: int,
    candidates_3d: list[tuple[tuple[int, ...], tuple[int, ...]]],
    sample_frames: int,
    suspicious_ratio: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    df = pd.read_parquet(episode.data_path)
    state = stack_vector_column(df["observation.state"], "observation.state")
    action = stack_vector_column(df["action"], "action")
    if sample_frames > 0 and len(df) > sample_frames:
        idx = np.linspace(0, len(df) - 1, sample_frames).astype(int)
        state = state[idx]
        action = action[idx]

    state_eef = state[:, state_eef_slice]
    action_eef = action[:, action_eef_slice]
    state_pos = state[:, state_pos_slice]
    action_pos = action[:, action_pos_slice]
    state_rot = state_eef[:, 3:] if state_eef.shape[1] >= 9 else np.zeros((state_eef.shape[0], 0))
    action_rot = action_eef[:, 3:] if action_eef.shape[1] >= 9 else np.zeros((action_eef.shape[0], 0))

    direct_pos_rmse = rmse(state_pos[: action_pos.shape[0]] - action_pos[: state_pos.shape[0]])
    best_pos_lag, best_pos_lag_rmse = best_lag_rmse(action_pos, state_pos, max_lag)
    direct_eef_rmse = rmse(state_eef[: action_eef.shape[0]] - action_eef[: state_eef.shape[0]])
    best_eef_lag, best_eef_lag_rmse = best_lag_rmse(action_eef, state_eef, max_lag)
    direct_rot_rmse = rmse(state_rot[: action_rot.shape[0]] - action_rot[: state_rot.shape[0]]) if state_rot.size and action_rot.size else math.nan
    best_pos_map = best_signed_permutation_with_offset(action_pos, state_pos, candidates_3d)
    identity_offset = np.nanmean(state_pos - action_pos, axis=0)
    identity_offset_rmse = rmse(state_pos - (action_pos + identity_offset))
    is_identity = tuple(best_pos_map["perm"]) == (0, 1, 2) and tuple(best_pos_map["signs"]) == (1, 1, 1)
    improvement_ratio = float(best_pos_map["rmse"] / direct_pos_rmse) if direct_pos_rmse and math.isfinite(direct_pos_rmse) else math.nan
    offset_improvement_ratio = (
        float(best_pos_map["rmse"] / identity_offset_rmse)
        if identity_offset_rmse and math.isfinite(identity_offset_rmse)
        else math.nan
    )
    suspicious = bool(
        (not is_identity)
        and math.isfinite(improvement_ratio)
        and improvement_ratio <= suspicious_ratio
    )

    row = {
        "episode_index": episode.episode_index,
        "source_date": episode.source_date,
        "frames": int(state.shape[0]),
        "expected_length": episode.length,
        "direct_pos_rmse": direct_pos_rmse,
        "best_pos_lag": int(best_pos_lag),
        "best_pos_lag_rmse": best_pos_lag_rmse,
        "direct_rot6d_rmse": direct_rot_rmse,
        "direct_eef9d_rmse": direct_eef_rmse,
        "best_eef9d_lag": int(best_eef_lag),
        "best_eef9d_lag_rmse": best_eef_lag_rmse,
        "identity_offset_pos_rmse": identity_offset_rmse,
        "best_perm_pos_rmse": best_pos_map["rmse"],
        "best_perm": ",".join(str(v) for v in best_pos_map["perm"]),
        "best_signs": ",".join(str(v) for v in best_pos_map["signs"]),
        "best_offset": json.dumps([float(v) for v in best_pos_map["offset"]]),
        "best_perm_direct_improvement_ratio": improvement_ratio,
        "best_perm_offset_improvement_ratio": offset_improvement_ratio,
        "best_perm_is_identity": is_identity,
        "coord_mismatch_suspected": suspicious,
        "state_pos_min": json.dumps(np.nanmin(state_pos, axis=0).tolist()),
        "state_pos_max": json.dumps(np.nanmax(state_pos, axis=0).tolist()),
        "action_pos_min": json.dumps(np.nanmin(action_pos, axis=0).tolist()),
        "action_pos_max": json.dumps(np.nanmax(action_pos, axis=0).tolist()),
    }
    dim_rows = []
    for prefix, action_values, state_values in (
        ("pos", action_pos, state_pos),
        ("rot6d", action_rot, state_rot),
    ):
        if action_values.size == 0 or state_values.size == 0:
            continue
        correlations = corr_diag(action_values, state_values)
        for dim, corr in enumerate(correlations):
            dim_rows.append(
                {
                    "episode_index": episode.episode_index,
                    "source_date": episode.source_date,
                    "group": prefix,
                    "dim": dim,
                    "direct_rmse": rmse(state_values[:, dim] - action_values[:, dim]),
                    "corr": corr,
                    "state_min": float(np.nanmin(state_values[:, dim])),
                    "state_max": float(np.nanmax(state_values[:, dim])),
                    "action_min": float(np.nanmin(action_values[:, dim])),
                    "action_max": float(np.nanmax(action_values[:, dim])),
                }
            )
    return row, dim_rows, action_pos, state_pos


def global_signed_permutation_report(
    all_action_pos: list[np.ndarray],
    all_state_pos: list[np.ndarray],
    candidates_3d: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict[str, Any]:
    if not all_action_pos or not all_state_pos:
        return {}
    action = np.vstack(all_action_pos)
    state = np.vstack(all_state_pos)
    direct = rmse(state - action)
    identity_offset = np.nanmean(state - action, axis=0)
    identity_offset_value = rmse(state - (action + identity_offset))
    best = best_signed_permutation_with_offset(action, state, candidates_3d)
    return {
        "frames": int(action.shape[0]),
        "direct_pos_rmse": direct,
        "identity_offset_pos_rmse": identity_offset_value,
        "identity_offset": identity_offset.tolist(),
        "best_perm_pos_rmse": best["rmse"],
        "best_perm": list(best["perm"]),
        "best_signs": list(best["signs"]),
        "best_offset": best["offset"],
        "best_perm_direct_improvement_ratio": float(best["rmse"] / direct) if direct else math.nan,
        "best_perm_offset_improvement_ratio": float(best["rmse"] / identity_offset_value) if identity_offset_value else math.nan,
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        return make_json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _info, modality, episodes = discover_episodes(dataset_dir)

    state_eef_slice = slice_from_modality(modality, "state", ("eef_9d",))
    action_eef_slice = slice_from_modality(modality, "action", ("eef_9d",))
    state_pos_slice = slice_from_modality(modality, "state", ("arm_eef_pos",))
    action_pos_slice = slice_from_modality(modality, "action", ("arm_eef_pos_target",))
    candidates_3d = signed_permutation_candidates(3)

    rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []
    all_action_pos: list[np.ndarray] = []
    all_state_pos: list[np.ndarray] = []
    for episode in episodes:
        row, episode_dim_rows, action_pos, state_pos = summarize_episode(
            episode,
            state_eef_slice=state_eef_slice,
            action_eef_slice=action_eef_slice,
            state_pos_slice=state_pos_slice,
            action_pos_slice=action_pos_slice,
            max_lag=max(0, int(args.max_lag)),
            candidates_3d=candidates_3d,
            sample_frames=max(0, int(args.sample_frames)),
            suspicious_ratio=float(args.suspicious_ratio),
        )
        rows.append(row)
        dim_rows.extend(episode_dim_rows)
        all_action_pos.append(action_pos)
        all_state_pos.append(state_pos)

    episode_df = pd.DataFrame(rows)
    dim_df = pd.DataFrame(dim_rows)
    episode_df.to_csv(output_dir / "episode_eef_alignment.csv", index=False)
    dim_df.to_csv(output_dir / "dimension_eef_alignment.csv", index=False)

    by_date = (
        episode_df.groupby("source_date")
        .agg(
            episodes=("episode_index", "count"),
            suspected=("coord_mismatch_suspected", "sum"),
            direct_pos_rmse_mean=("direct_pos_rmse", "mean"),
            best_perm_pos_rmse_mean=("best_perm_pos_rmse", "mean"),
            improvement_ratio_mean=("best_perm_direct_improvement_ratio", "mean"),
        )
        .reset_index()
    )
    by_date.to_csv(output_dir / "date_eef_alignment.csv", index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "episodes": int(len(episode_df)),
        "suspected_coord_mismatch_episodes": int(episode_df["coord_mismatch_suspected"].sum()),
        "suspected_coord_mismatch_ratio": float(episode_df["coord_mismatch_suspected"].mean()) if len(episode_df) else math.nan,
        "direct_pos_rmse_mean": float(episode_df["direct_pos_rmse"].mean()),
        "best_perm_pos_rmse_mean": float(episode_df["best_perm_pos_rmse"].mean()),
        "global_position_mapping": global_signed_permutation_report(all_action_pos, all_state_pos, candidates_3d),
        "date_summary": by_date.to_dict(orient="records"),
        "outputs": {
            "episode_csv": str(output_dir / "episode_eef_alignment.csv"),
            "dimension_csv": str(output_dir / "dimension_eef_alignment.csv"),
            "date_csv": str(output_dir / "date_eef_alignment.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(make_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(make_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
