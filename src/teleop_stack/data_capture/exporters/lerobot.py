from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teleop_stack.data_capture.exporters.base import (
    ExportSummary,
    NormalizedEpisode,
    NormalizedStep,
)

ROKAE_LINKER_L10_SCHEMA = "rokae_xmate3_linker_l10_groot_v1"
ROKAE_LINKER_L10_FULL_ORIENTATION_SCHEMA = "rokae_xmate3_linker_l10_groot_v1_1_full_orientation"
ROKAE_LINKER_L10_SCHEMAS = {ROKAE_LINKER_L10_SCHEMA, ROKAE_LINKER_L10_FULL_ORIENTATION_SCHEMA}
LEGACY_SCHEMA = "legacy"
L10_CANONICAL_JOINT_ORDER: tuple[str, ...] = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _require_optional_dependency(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing optional dependency {module_name!r}. "
            "Please install it in the active environment before exporting."
        ) from exc


def _maybe_float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for item in values:
        if isinstance(item, (int, float)):
            result.append(float(item))
    return result


def _quat_xyzw_to_rot6d(values: object) -> list[float]:
    quat = _maybe_float_list(values)
    if len(quat) != 4:
        return []
    x, y, z, w = quat
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm <= 0.0:
        return []
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r20 = 2.0 * (x * z - y * w)
    r01 = 2.0 * (x * y - z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r21 = 2.0 * (y * z + x * w)
    return [r00, r10, r20, r01, r11, r21]


def _named_joint_values(payload: object, order: tuple[str, ...]) -> list[float]:
    if not isinstance(payload, dict):
        return []
    joint_names = payload.get("joint_names")
    joint_positions = payload.get("joint_positions")
    if not isinstance(joint_names, list) or not isinstance(joint_positions, list):
        return []
    if len(joint_names) != len(joint_positions):
        return []
    by_name: dict[str, float] = {}
    for name, position in zip(joint_names, joint_positions, strict=True):
        if not isinstance(position, (int, float)):
            return []
        by_name[str(name)] = float(position)
    if any(name not in by_name for name in order):
        return []
    return [by_name[name] for name in order]


def _nearest_record(
    records: list[dict[str, object]],
    target_ts_s: float,
    *,
    require_key: str | None = None,
) -> tuple[dict[str, object] | None, float]:
    best: dict[str, object] | None = None
    best_dt = float("inf")
    for record in records:
        if require_key is not None and not isinstance(record.get(require_key), dict):
            continue
        timestamp = record.get("monotonic_ts_s")
        if not isinstance(timestamp, (int, float)):
            continue
        dt = abs(float(timestamp) - target_ts_s)
        if dt < best_dt:
            best = record
            best_dt = dt
    return best, best_dt


def _nearest_future_record(
    records: list[dict[str, object]],
    target_ts_s: float,
    *,
    require_key: str | None = None,
) -> tuple[dict[str, object] | None, float]:
    best: dict[str, object] | None = None
    best_dt = float("inf")
    for record in records:
        if require_key is not None and not isinstance(record.get(require_key), dict):
            continue
        timestamp = record.get("monotonic_ts_s")
        if not isinstance(timestamp, (int, float)):
            continue
        dt = float(timestamp) - target_ts_s
        if dt < -1e-9:
            continue
        if dt < best_dt:
            best = record
            best_dt = dt
    if best is not None:
        return best, best_dt
    return _nearest_record(records, target_ts_s, require_key=require_key)


def _append_segment(
    *,
    vector: list[float],
    names: list[str],
    slices: dict[str, dict[str, int]],
    key: str,
    values: list[float],
    component_names: list[str],
) -> None:
    if not values:
        return
    if len(values) != len(component_names):
        raise RuntimeError(
            f"Vector segment {key!r} produced {len(values)} values for "
            f"{len(component_names)} component names."
        )
    start = len(vector)
    vector.extend(values)
    names.extend(component_names)
    slices[key] = {"start": start, "end": len(vector)}


@dataclass(frozen=True)
class VectorBundle:
    values: list[float]
    names: list[str]
    slices: dict[str, dict[str, int]]


def _state_vector(robot_payload: dict[str, object]) -> VectorBundle:
    vector: list[float] = []
    names: list[str] = []
    slices: dict[str, dict[str, int]] = {}

    arm_joint_positions = _maybe_float_list(robot_payload.get("arm_joint_positions"))
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="arm_joint_positions",
        values=arm_joint_positions,
        component_names=[f"arm_joint_positions.{index}" for index in range(len(arm_joint_positions))],
    )

    hand_joint_positions = _maybe_float_list(robot_payload.get("hand_joint_positions"))
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="hand_joint_positions",
        values=hand_joint_positions,
        component_names=[f"hand_joint_positions.{index}" for index in range(len(hand_joint_positions))],
    )

    arm_ee_pose = robot_payload.get("arm_ee_pose")
    if isinstance(arm_ee_pose, dict):
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="arm_ee_pose.position_xyz",
            values=_maybe_float_list(arm_ee_pose.get("position_xyz")),
            component_names=[
                "arm_ee_pose.position_x",
                "arm_ee_pose.position_y",
                "arm_ee_pose.position_z",
            ],
        )
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="arm_ee_pose.quaternion_xyzw",
            values=_maybe_float_list(arm_ee_pose.get("quaternion_xyzw")),
            component_names=[
                "arm_ee_pose.quaternion_x",
                "arm_ee_pose.quaternion_y",
                "arm_ee_pose.quaternion_z",
                "arm_ee_pose.quaternion_w",
            ],
        )

    return VectorBundle(values=vector, names=names, slices=slices)


def _rokae_linker_l10_state_vector(robot_payload: dict[str, object]) -> VectorBundle:
    return _rokae_linker_l10_state_vector_impl(robot_payload, include_rot6d=False)


def _rokae_linker_l10_full_orientation_state_vector(robot_payload: dict[str, object]) -> VectorBundle:
    return _rokae_linker_l10_state_vector_impl(robot_payload, include_rot6d=True)


def _rokae_linker_l10_state_vector_impl(robot_payload: dict[str, object], *, include_rot6d: bool) -> VectorBundle:
    vector: list[float] = []
    names: list[str] = []
    slices: dict[str, dict[str, int]] = {}

    arm_joint_positions = _maybe_float_list(robot_payload.get("arm_joint_positions"))
    if len(arm_joint_positions) != 7:
        return VectorBundle(values=[], names=[], slices={})
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="arm_joint_pos",
        values=arm_joint_positions,
        component_names=[f"arm_joint_pos.{index}" for index in range(7)],
    )

    arm_ee_pose = robot_payload.get("arm_ee_pose")
    if not isinstance(arm_ee_pose, dict):
        return VectorBundle(values=[], names=[], slices={})
    arm_eef_pos = _maybe_float_list(arm_ee_pose.get("position_xyz"))
    if len(arm_eef_pos) != 3:
        return VectorBundle(values=[], names=[], slices={})
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="arm_eef_pos",
        values=arm_eef_pos,
        component_names=["arm_eef_pos.x", "arm_eef_pos.y", "arm_eef_pos.z"],
    )
    if include_rot6d:
        arm_eef_rot6d = _quat_xyzw_to_rot6d(arm_ee_pose.get("quaternion_xyzw"))
        if len(arm_eef_rot6d) != 6:
            return VectorBundle(values=[], names=[], slices={})
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="arm_eef_rot6d",
            values=arm_eef_rot6d,
            component_names=[
                "arm_eef_rot6d.r00",
                "arm_eef_rot6d.r10",
                "arm_eef_rot6d.r20",
                "arm_eef_rot6d.r01",
                "arm_eef_rot6d.r11",
                "arm_eef_rot6d.r21",
            ],
        )

    raw_snapshot = robot_payload.get("raw_snapshot")
    hand_state = raw_snapshot.get("hand_state") if isinstance(raw_snapshot, dict) else None
    hand_joint_positions = _named_joint_values(
        {
            "joint_names": hand_state.get("joint_names") if isinstance(hand_state, dict) else None,
            "joint_positions": robot_payload.get("hand_joint_positions"),
        },
        L10_CANONICAL_JOINT_ORDER,
    )
    if len(hand_joint_positions) != len(L10_CANONICAL_JOINT_ORDER):
        return VectorBundle(values=[], names=[], slices={})
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="hand_joint_pos",
        values=hand_joint_positions,
        component_names=[f"hand_joint_pos.{name}" for name in L10_CANONICAL_JOINT_ORDER],
    )

    return VectorBundle(values=vector, names=names, slices=slices)


def _action_vector(action_payload: dict[str, object], action_source: str) -> VectorBundle:
    selected = action_payload.get(action_source)
    if not isinstance(selected, dict):
        raise RuntimeError(f"Action payload is missing {action_source!r}.")

    vector: list[float] = []
    names: list[str] = []
    slices: dict[str, dict[str, int]] = {}

    ee_target = selected.get("ee_target")
    if isinstance(ee_target, dict):
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="ee_target.position_xyz",
            values=_maybe_float_list(ee_target.get("position_xyz")),
            component_names=[
                "ee_target.position_x",
                "ee_target.position_y",
                "ee_target.position_z",
            ],
        )
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="ee_target.quaternion_xyzw",
            values=_maybe_float_list(ee_target.get("quaternion_xyzw")),
            component_names=[
                "ee_target.quaternion_x",
                "ee_target.quaternion_y",
                "ee_target.quaternion_z",
                "ee_target.quaternion_w",
            ],
        )

    gripper = selected.get("gripper")
    if isinstance(gripper, dict):
        normalized = gripper.get("normalized_position")
        if isinstance(normalized, (int, float)):
            _append_segment(
                vector=vector,
                names=names,
                slices=slices,
                key="gripper.normalized_position",
                values=[float(normalized)],
                component_names=["gripper.normalized_position"],
            )

    hand_target = selected.get("hand_target")
    if isinstance(hand_target, dict):
        ordered_items = [(key, hand_target[key]) for key in sorted(hand_target.keys())]
        hand_values = [float(value) for key, value in ordered_items if isinstance(value, (int, float))]
        hand_names = [f"hand_target.{key}" for key, value in ordered_items if isinstance(value, (int, float))]
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="hand_target",
            values=hand_values,
            component_names=hand_names,
        )

    return VectorBundle(values=vector, names=names, slices=slices)


def _rokae_linker_l10_action_vector(action_payload: dict[str, object], action_source: str) -> VectorBundle:
    return _rokae_linker_l10_action_vector_impl(action_payload, action_source, include_rot6d=False)


def _rokae_linker_l10_full_orientation_action_vector(action_payload: dict[str, object], action_source: str) -> VectorBundle:
    return _rokae_linker_l10_action_vector_impl(action_payload, action_source, include_rot6d=True)


def _rokae_linker_l10_action_vector_impl(
    action_payload: dict[str, object],
    action_source: str,
    *,
    include_rot6d: bool,
) -> VectorBundle:
    selected = action_payload.get(action_source)
    if not isinstance(selected, dict):
        return VectorBundle(values=[], names=[], slices={})

    vector: list[float] = []
    names: list[str] = []
    slices: dict[str, dict[str, int]] = {}

    ee_target = selected.get("ee_target")
    if not isinstance(ee_target, dict):
        return VectorBundle(values=[], names=[], slices={})
    arm_eef_pos_target = _maybe_float_list(ee_target.get("position_xyz"))
    if len(arm_eef_pos_target) != 3:
        return VectorBundle(values=[], names=[], slices={})
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="arm_eef_pos_target",
        values=arm_eef_pos_target,
        component_names=["arm_eef_pos_target.x", "arm_eef_pos_target.y", "arm_eef_pos_target.z"],
    )
    if include_rot6d:
        arm_eef_rot6d_target = _quat_xyzw_to_rot6d(ee_target.get("quaternion_xyzw"))
        if len(arm_eef_rot6d_target) != 6:
            return VectorBundle(values=[], names=[], slices={})
        _append_segment(
            vector=vector,
            names=names,
            slices=slices,
            key="arm_eef_rot6d_target",
            values=arm_eef_rot6d_target,
            component_names=[
                "arm_eef_rot6d_target.r00",
                "arm_eef_rot6d_target.r10",
                "arm_eef_rot6d_target.r20",
                "arm_eef_rot6d_target.r01",
                "arm_eef_rot6d_target.r11",
                "arm_eef_rot6d_target.r21",
            ],
        )

    hand_joint_target = _named_joint_values(selected.get("hand_target"), L10_CANONICAL_JOINT_ORDER)
    if len(hand_joint_target) != len(L10_CANONICAL_JOINT_ORDER):
        return VectorBundle(values=[], names=[], slices={})
    _append_segment(
        vector=vector,
        names=names,
        slices=slices,
        key="hand_joint_target",
        values=hand_joint_target,
        component_names=[f"hand_joint_target.{name}" for name in L10_CANONICAL_JOINT_ORDER],
    )

    return VectorBundle(values=vector, names=names, slices=slices)


def _select_video(videos: object, camera_name: str) -> dict[str, object] | None:
    if not isinstance(videos, list):
        return None
    for video in videos:
        if not isinstance(video, dict):
            continue
        if str(video.get("camera_name", "")) == camera_name:
            return video
    return None


def _iter_video_payloads(record: dict[str, object]) -> list[dict[str, object]]:
    camera_name = record.get("camera_name")
    if isinstance(camera_name, str):
        return [record]
    videos = record.get("videos")
    if not isinstance(videos, list):
        return []
    return [video for video in videos if isinstance(video, dict)]


def _video_ordinal_maps(
    records: list[dict[str, object]],
    camera_names: tuple[str, ...],
) -> dict[tuple[str, str], dict[int, int]]:
    selected_cameras = set(camera_names)
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    seen: set[tuple[str, str, int]] = set()
    for record_index, record in enumerate(records):
        for video in _iter_video_payloads(record):
            camera_name = video.get("camera_name")
            relative_path = video.get("relative_path")
            frame_index = video.get("frame_index")
            if (
                not isinstance(camera_name, str)
                or camera_name not in selected_cameras
                or not isinstance(relative_path, str)
                or not isinstance(frame_index, int)
            ):
                continue
            seen_key = (camera_name, relative_path, frame_index)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            grouped.setdefault((camera_name, relative_path), []).append((record_index, frame_index))
    return {
        key: {frame_index: ordinal for ordinal, (_, frame_index) in enumerate(sorted(values))}
        for key, values in grouped.items()
    }


def _is_same_schema(
    *,
    expected_names: list[str] | None,
    actual_names: list[str],
    expected_slices: dict[str, dict[str, int]] | None,
    actual_slices: dict[str, dict[str, int]],
) -> bool:
    if expected_names is None or expected_slices is None:
        return True
    return expected_names == actual_names and expected_slices == actual_slices


def _feature_spec(*, dtype: str, shape: list[int], names: list[str] | None = None, info: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"dtype": dtype, "shape": shape, "names": names}
    if info is not None:
        payload["info"] = info
    return payload


def _episode_validity_label(success: bool | None) -> str | None:
    if success is True:
        return "valid"
    if success is False:
        return "invalid"
    return None


@dataclass(frozen=True)
class GrootLeRobotV2ExporterConfig:
    selected_camera: str = "realsense_head"
    selected_cameras: tuple[str, ...] | None = None
    schema: str = ROKAE_LINKER_L10_SCHEMA
    video_alias: str | None = None
    video_aliases: tuple[str, ...] | None = None
    video_feature_key: str | None = None
    video_feature_keys: tuple[str, ...] | None = None
    action_source: str = "safe_action"
    success_only: bool = False
    require_robot_state: bool = True
    fps: int = 10
    max_robot_dt_s: float = 0.10
    max_action_dt_s: float = 0.10
    max_video_dt_s: float = 0.10
    episodes_per_chunk: int = 1000
    dry_run: bool = False


@dataclass(frozen=True)
class NormalizationResult:
    episodes: list[NormalizedEpisode]
    skipped: list[dict[str, object]]


class GrootLeRobotV2Exporter:
    format_name = "groot_lerobot_v2"

    def __init__(self, config: GrootLeRobotV2ExporterConfig | None = None):
        self.config = config or GrootLeRobotV2ExporterConfig()

    def export_capture(self, *, raw_capture_dir: Path, output_dir: Path) -> ExportSummary:
        normalization = self._normalize_capture(raw_capture_dir=raw_capture_dir)
        normalized = normalization.episodes
        output_dir.mkdir(parents=True, exist_ok=True)

        video_cameras = self._resolve_video_cameras()
        video_alias_by_camera = self._resolve_video_aliases(video_cameras)
        video_feature_key_by_camera = self._resolve_video_feature_keys(video_cameras, video_alias_by_camera)
        primary_camera = video_cameras[0]
        video_alias = video_alias_by_camera[primary_camera]
        video_feature_key = video_feature_key_by_camera[primary_camera]
        data_path_template = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        video_path_template = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

        if self.config.dry_run:
            summary = {
                "format_name": self.format_name,
                "episode_count": len(normalized),
                "step_count": sum(len(item.steps) for item in normalized),
                "selected_camera": self.config.selected_camera,
                "selected_cameras": list(video_cameras),
                "schema": self.config.schema,
                "video_alias": video_alias,
                "video_aliases": video_alias_by_camera,
                "video_feature_key": video_feature_key,
                "video_feature_keys": video_feature_key_by_camera,
                "action_source": self.config.action_source,
                "fps": max(1, int(self.config.fps)),
                "episodes_per_chunk": max(1, int(self.config.episodes_per_chunk)),
                "data_path_template": data_path_template,
                "video_path_template": video_path_template,
                "state_names": normalized[0].state_names if normalized else [],
                "action_names": normalized[0].action_names if normalized else [],
                "skipped": normalization.skipped,
            }
            (output_dir / "dry_run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return ExportSummary(
                format_name=self.format_name,
                output_path=output_dir,
                episode_count=len(normalized),
                step_count=sum(len(item.steps) for item in normalized),
            )

        pyarrow = _require_optional_dependency("pyarrow")
        pyarrow_parquet = _require_optional_dependency("pyarrow.parquet")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to build GR00T-compatible LeRobot v2 videos.")

        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        session_metadata = self._load_session_metadata(raw_capture_dir)
        tasks: dict[str, int] = {}
        episodes_jsonl: list[dict[str, object]] = []
        total_steps = 0
        global_index = 0
        total_chunks = 0
        dataset_state_names: list[str] | None = None
        dataset_state_slices: dict[str, dict[str, int]] | None = None
        dataset_action_names: list[str] | None = None
        dataset_action_slices: dict[str, dict[str, int]] | None = None
        dataset_video_shapes: dict[str, tuple[int, int]] = {}
        robot_type = "teleop_stack"

        for episode in normalized:
            if dataset_state_names is None:
                dataset_state_names = episode.state_names
                dataset_state_slices = episode.state_slices
            elif dataset_state_names != episode.state_names or dataset_state_slices != episode.state_slices:
                raise RuntimeError(
                    f"Exported episodes do not share a stable observation.state schema. "
                    f"First schema: {dataset_state_names}, current episode {episode.episode_index}: {episode.state_names}."
                )

            if dataset_action_names is None:
                dataset_action_names = episode.action_names
                dataset_action_slices = episode.action_slices
            elif dataset_action_names != episode.action_names or dataset_action_slices != episode.action_slices:
                raise RuntimeError(
                    f"Exported episodes do not share a stable action schema. "
                    f"First schema: {dataset_action_names}, current episode {episode.episode_index}: {episode.action_names}."
                )

            for camera_name in video_cameras:
                episode_video_shape = (
                    episode.steps[0].video_heights.get(camera_name, episode.steps[0].video_height),
                    episode.steps[0].video_widths.get(camera_name, episode.steps[0].video_width),
                )
                if camera_name not in dataset_video_shapes:
                    dataset_video_shapes[camera_name] = episode_video_shape
                elif dataset_video_shapes[camera_name] != episode_video_shape:
                    raise RuntimeError(
                        f"Exported episodes do not share a stable video shape for camera {camera_name!r}. "
                        f"First shape: {dataset_video_shapes[camera_name]}, "
                        f"current episode {episode.episode_index}: {episode_video_shape}."
                    )

            robot_type = str(
                episode.metadata.get("robot_embodiment")
                or episode.metadata.get("robot_name")
                or session_metadata.get("robot_embodiment")
                or session_metadata.get("robot_name")
                or robot_type
            )

            episode_chunk = episode.episode_index // max(1, int(self.config.episodes_per_chunk))
            total_chunks = max(total_chunks, episode_chunk)
            chunk_name = f"chunk-{episode_chunk:03d}"
            data_dir = output_dir / "data" / chunk_name
            data_dir.mkdir(parents=True, exist_ok=True)
            for feature_key in video_feature_key_by_camera.values():
                (output_dir / "videos" / chunk_name / feature_key).mkdir(parents=True, exist_ok=True)

            task_index = tasks.setdefault(episode.task, len(tasks))
            validity_index: int | None = None
            validity_label = _episode_validity_label(episode.success)
            if validity_label is not None:
                validity_index = tasks.setdefault(validity_label, len(tasks))

            episode_name = f"episode_{episode.episode_index:06d}"
            parquet_path = data_dir / f"{episode_name}.parquet"
            video_paths_by_camera = {
                camera_name: output_dir / "videos" / chunk_name / feature_key / f"{episode_name}.mp4"
                for camera_name, feature_key in video_feature_key_by_camera.items()
            }
            video_path = video_paths_by_camera[primary_camera]
            total_steps += len(episode.steps)

            self._write_episode_parquet(
                pyarrow=pyarrow,
                pyarrow_parquet=pyarrow_parquet,
                episode=episode,
                parquet_path=parquet_path,
                global_index_start=global_index,
                task_index=task_index,
                validity_index=validity_index,
            )
            for camera_name, camera_video_path in video_paths_by_camera.items():
                self._encode_video(
                    ffmpeg=ffmpeg,
                    episode=episode,
                    output_path=camera_video_path,
                    camera_name=camera_name,
                )
            global_index += len(episode.steps)

            episodes_jsonl.append(
                {
                    "episode_index": episode.episode_index,
                    "tasks": [episode.task],
                    "length": len(episode.steps),
                    "teleop_stack_metadata": {
                        "success": episode.success,
                        "raw_episode_dir": str(episode.raw_episode_dir),
                        "raw_episode_id": episode.metadata.get("episode_id"),
                        "trial_mode": episode.metadata.get("trial_mode"),
                        "policy_id": episode.metadata.get("policy_id"),
                        "policy_version": episode.metadata.get("policy_version"),
                        "outcome": episode.metadata.get("outcome"),
                        "outcome_reason": episode.metadata.get("outcome_reason"),
                        "action_source": self.config.action_source,
                        "selected_camera": self.config.selected_camera,
                        "selected_cameras": list(video_cameras),
                        "video_alias": video_alias,
                        "video_aliases": video_alias_by_camera,
                        "video_feature_key": video_feature_key,
                        "video_feature_keys": video_feature_key_by_camera,
                        "data_path": str(parquet_path.relative_to(output_dir)),
                        "video_path": str(video_path.relative_to(output_dir)),
                        "video_paths": {
                            camera_name: str(path.relative_to(output_dir))
                            for camera_name, path in video_paths_by_camera.items()
                        },
                    },
                }
            )

        state_names = dataset_state_names or []
        state_slices = dataset_state_slices or {}
        action_names = dataset_action_names or []
        action_slices = dataset_action_slices or {}
        total_videos = len(normalized) * len(video_cameras)
        video_feature_specs = {}
        for camera_name, feature_key in video_feature_key_by_camera.items():
            video_height, video_width = dataset_video_shapes.get(camera_name, (0, 0))
            video_feature_specs[feature_key] = _feature_spec(
                dtype="video",
                shape=[video_height, video_width, 3],
                names=["height", "width", "channels"],
                info={
                    "video.height": video_height,
                    "video.width": video_width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": max(1, int(self.config.fps)),
                    "video.channels": 3,
                    "has_audio": False,
                },
            )

        info_payload = {
            "codebase_version": "teleop_stack.groot_lerobot_v2.v1",
            "robot_type": robot_type,
            "total_episodes": len(normalized),
            "total_frames": total_steps,
            "total_tasks": len(tasks),
            "chunks_size": max(1, int(self.config.episodes_per_chunk)),
            "fps": max(1, int(self.config.fps)),
            "splits": {"train": f"0:{len(normalized)}"},
            "data_path": data_path_template,
            "video_path": video_path_template,
            "features": {
                "action": _feature_spec(
                    dtype="float32",
                    shape=[len(action_names)],
                    names=action_names,
                ),
                "observation.state": _feature_spec(
                    dtype="float32",
                    shape=[len(state_names)],
                    names=state_names,
                ),
                **video_feature_specs,
                "timestamp": _feature_spec(dtype="float32", shape=[1]),
                "frame_index": _feature_spec(dtype="int64", shape=[1]),
                "episode_index": _feature_spec(dtype="int64", shape=[1]),
                "index": _feature_spec(dtype="int64", shape=[1]),
                "task_index": _feature_spec(dtype="int64", shape=[1]),
                "annotation.human.action.task_description": _feature_spec(dtype="int64", shape=[1]),
                "annotation.human.validity": _feature_spec(dtype="int64", shape=[1]),
                "next.reward": _feature_spec(dtype="float32", shape=[1]),
                "next.done": _feature_spec(dtype="bool", shape=[1]),
            },
            "total_chunks": total_chunks,
            "total_videos": total_videos,
            "teleop_stack": {
                "format_name": self.format_name,
                "raw_capture_dir": str(raw_capture_dir),
                "schema": self.config.schema,
                "selected_camera": self.config.selected_camera,
                "selected_cameras": list(video_cameras),
                "video_alias": video_alias,
                "video_aliases": video_alias_by_camera,
                "video_feature_key": video_feature_key,
                "video_feature_keys": video_feature_key_by_camera,
                "action_source": self.config.action_source,
                "require_robot_state": self.config.require_robot_state,
                "state_dim": len(state_names),
                "action_dim": len(action_names),
                "arm_frame": "rokae_base" if self.config.schema in ROKAE_LINKER_L10_SCHEMAS else None,
                "arm_action_semantics": (
                    "absolute_wrist_position_target" if self.config.schema == ROKAE_LINKER_L10_SCHEMA else None
                )
                or (
                    "absolute_wrist_pose_xyz_rot6d_target"
                    if self.config.schema == ROKAE_LINKER_L10_FULL_ORIENTATION_SCHEMA
                    else None
                ),
                "hand_state_semantics": (
                    "linkerhand_sdk_get_state_reported" if self.config.schema in ROKAE_LINKER_L10_SCHEMAS else None
                ),
                "hand_action_semantics": (
                    "absolute_l10_joint_target" if self.config.schema in ROKAE_LINKER_L10_SCHEMAS else None
                ),
                "hand_joint_order": list(L10_CANONICAL_JOINT_ORDER)
                if self.config.schema in ROKAE_LINKER_L10_SCHEMAS
                else None,
            },
        }
        modality_payload = {
            "state": state_slices,
            "action": action_slices,
            "video": {
                alias: {
                    "original_key": video_feature_key_by_camera[camera_name],
                }
                for camera_name, alias in video_alias_by_camera.items()
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": "annotation.human.action.task_description",
                },
                "human.validity": {
                    "original_key": "annotation.human.validity",
                },
            },
        }
        tasks_payload = [
            {"task_index": index, "task": task}
            for task, index in sorted(tasks.items(), key=lambda item: item[1])
        ]

        (meta_dir / "info.json").write_text(
            json.dumps(info_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (meta_dir / "modality.json").write_text(
            json.dumps(modality_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_jsonl(meta_dir / "tasks.jsonl", tasks_payload)
        self._write_jsonl(meta_dir / "episodes.jsonl", episodes_jsonl)

        return ExportSummary(
            format_name=self.format_name,
            output_path=output_dir,
            episode_count=len(normalized),
            step_count=total_steps,
        )

    def _normalize_capture(self, *, raw_capture_dir: Path) -> NormalizationResult:
        episodes_root = raw_capture_dir / "episodes"
        if not episodes_root.is_dir():
            raise RuntimeError(f"Raw capture directory does not contain episodes/: {raw_capture_dir}")
        normalized: list[NormalizedEpisode] = []
        skipped: list[dict[str, object]] = []
        for episode_index, episode_dir in enumerate(sorted(path for path in episodes_root.iterdir() if path.is_dir())):
            episode, reason = self._normalize_episode(episode_dir=episode_dir, episode_index=episode_index)
            if episode is None:
                skipped.append(
                    {
                        "episode_dir": str(episode_dir),
                        "reason": reason or "no_exportable_steps",
                    }
                )
                continue
            normalized.append(episode)
        return NormalizationResult(episodes=normalized, skipped=skipped)

    def _state_vector_for_schema(self, robot_payload: dict[str, object]) -> VectorBundle:
        if self.config.schema == ROKAE_LINKER_L10_SCHEMA:
            return _rokae_linker_l10_state_vector(robot_payload)
        if self.config.schema == ROKAE_LINKER_L10_FULL_ORIENTATION_SCHEMA:
            return _rokae_linker_l10_full_orientation_state_vector(robot_payload)
        return _state_vector(robot_payload)

    def _action_vector_for_schema(self, action_payload: dict[str, object]) -> VectorBundle:
        if self.config.schema == ROKAE_LINKER_L10_SCHEMA:
            return _rokae_linker_l10_action_vector(action_payload, self.config.action_source)
        if self.config.schema == ROKAE_LINKER_L10_FULL_ORIENTATION_SCHEMA:
            return _rokae_linker_l10_full_orientation_action_vector(action_payload, self.config.action_source)
        return _action_vector(action_payload, self.config.action_source)

    def _normalize_episode(
        self,
        *,
        episode_dir: Path,
        episode_index: int,
    ) -> tuple[NormalizedEpisode | None, str | None]:
        metadata = _load_json(episode_dir / "episode.json")
        video_cameras = self._resolve_video_cameras()
        frames = _load_jsonl(episode_dir / "frames.jsonl")
        video_log_path = episode_dir / "video.jsonl"
        video_log_records = _load_jsonl(video_log_path) if video_log_path.is_file() else []
        video_ordinal_maps = _video_ordinal_maps(video_log_records or frames, video_cameras)
        action_records = [frame for frame in frames if isinstance(frame.get("action"), dict)]
        robot_records = [frame for frame in frames if isinstance(frame.get("robot"), dict)]
        video_records = [
            frame
            for frame in frames
            if all(_select_video(frame.get("videos"), camera_name) is not None for camera_name in video_cameras)
        ]
        success_value = metadata.get("success")
        success = bool(success_value) if isinstance(success_value, bool) else None
        if self.config.success_only and success is not True:
            return None, "filtered_non_success"

        task_instruction = str(metadata.get("task_instruction") or "").strip()
        task_name = str(metadata.get("task_name") or "").strip()
        task = task_instruction or task_name or "teleop"

        step_payloads: list[dict[str, object]] = []
        missing_robot_state = 0
        missing_video = 0
        state_names: list[str] | None = None
        state_slices: dict[str, dict[str, int]] | None = None
        action_names: list[str] | None = None
        action_slices: dict[str, dict[str, int]] | None = None
        video_shape: tuple[int, int] | None = None

        if self.config.schema in ROKAE_LINKER_L10_SCHEMAS:
            action_timestamps = [
                float(frame["monotonic_ts_s"])
                for frame in action_records
                if isinstance(frame.get("monotonic_ts_s"), (int, float))
            ]
            robot_timestamps = [
                float(frame["monotonic_ts_s"])
                for frame in robot_records
                if isinstance(frame.get("monotonic_ts_s"), (int, float))
            ]
            video_timestamps = [
                float(frame["monotonic_ts_s"])
                for frame in video_records
                if isinstance(frame.get("monotonic_ts_s"), (int, float))
            ]
            if not action_timestamps or not robot_timestamps or not video_timestamps:
                return None, "missing_frame_timestamps"
            video_first_ts = min(video_timestamps)
            start_ts = max(min(action_timestamps), min(robot_timestamps), video_first_ts)
            end_ts = min(max(action_timestamps), max(robot_timestamps), max(video_timestamps))
            if end_ts <= start_ts:
                return None, "empty_stream_time_intersection"
            metadata["teleop_stack_export"] = {
                "schema": self.config.schema,
                "resample_start_monotonic_ts_s": start_ts,
                "resample_end_monotonic_ts_s": end_ts,
                "video_start_offset_s": max(0.0, start_ts - video_first_ts),
            }
            step_count = int((end_ts - start_ts) * max(1, int(self.config.fps))) + 1
            candidate_frames: list[dict[str, object]] = []
            for step_index in range(step_count):
                target_ts = start_ts + step_index / max(1, int(self.config.fps))
                action_frame, action_dt = _nearest_future_record(action_records, target_ts, require_key="action")
                robot_frame, robot_dt = _nearest_record(robot_records, target_ts, require_key="robot")
                video_frame, video_dt = _nearest_record(video_records, target_ts)
                if (
                    action_frame is None
                    or robot_frame is None
                    or video_frame is None
                    or action_dt > self.config.max_action_dt_s
                    or robot_dt > self.config.max_robot_dt_s
                    or video_dt > self.config.max_video_dt_s
                ):
                    if robot_frame is None or robot_dt > self.config.max_robot_dt_s:
                        missing_robot_state += 1
                    if video_frame is None or video_dt > self.config.max_video_dt_s:
                        missing_video += 1
                    continue
                candidate_frames.append(
                    {
                        "monotonic_ts_s": target_ts,
                        "source_ts_s": target_ts - start_ts,
                        "teleop_frame_id": step_index,
                        "action": action_frame["action"],
                        "robot": robot_frame["robot"],
                        "videos": video_frame.get("videos"),
                    }
                )
        else:
            candidate_frames = frames

        for frame in candidate_frames:
            if not isinstance(frame, dict):
                continue

            action_payload = frame.get("action")
            if not isinstance(action_payload, dict):
                continue
            action_bundle = self._action_vector_for_schema(action_payload)
            if not action_bundle.values:
                continue
            if not _is_same_schema(
                expected_names=action_names,
                actual_names=action_bundle.names,
                expected_slices=action_slices,
                actual_slices=action_bundle.slices,
            ):
                return None, "inconsistent_action_schema"
            if action_names is None:
                action_names = action_bundle.names
                action_slices = action_bundle.slices

            robot_payload = frame.get("robot")
            if not isinstance(robot_payload, dict):
                if self.config.require_robot_state:
                    missing_robot_state += 1
                    continue
                state_bundle = VectorBundle(values=[], names=[], slices={})
            else:
                state_bundle = self._state_vector_for_schema(robot_payload)

            if self.config.require_robot_state and not state_bundle.values:
                missing_robot_state += 1
                continue
            if not _is_same_schema(
                expected_names=state_names,
                actual_names=state_bundle.names,
                expected_slices=state_slices,
                actual_slices=state_bundle.slices,
            ):
                return None, "inconsistent_state_schema"
            if state_names is None:
                state_names = state_bundle.names
                state_slices = state_bundle.slices

            video_payload: dict[str, object] | None = None
            video_paths: dict[str, Path] = {}
            video_widths: dict[str, int] = {}
            video_heights: dict[str, int] = {}
            video_frame_indices: dict[str, int] = {}
            video_frame_ordinals: dict[str, int] = {}
            primary_video_payload: dict[str, object] | None = None
            for camera_name in video_cameras:
                camera_video_payload = _select_video(frame.get("videos"), camera_name)
                if camera_video_payload is None:
                    missing_video += 1
                    break

                relative_path = camera_video_payload.get("relative_path")
                width = camera_video_payload.get("width")
                height = camera_video_payload.get("height")
                raw_video_frame_index = camera_video_payload.get("frame_index")
                if (
                    not isinstance(relative_path, str)
                    or not isinstance(width, int)
                    or not isinstance(height, int)
                    or not isinstance(raw_video_frame_index, int)
                ):
                    missing_video += 1
                    break

                frame_ordinal = video_ordinal_maps.get((camera_name, relative_path), {}).get(raw_video_frame_index)
                if frame_ordinal is None:
                    missing_video += 1
                    break

                current_video_shape = (height, width)
                if video_shape is None:
                    video_shape = current_video_shape
                elif camera_name == video_cameras[0] and video_shape != current_video_shape:
                    return None, "inconsistent_video_shape"

                video_path = episode_dir / relative_path
                if not video_path.is_file():
                    missing_video += 1
                    break
                video_paths[camera_name] = video_path
                video_widths[camera_name] = width
                video_heights[camera_name] = height
                video_frame_indices[camera_name] = raw_video_frame_index
                video_frame_ordinals[camera_name] = frame_ordinal
                if camera_name == video_cameras[0]:
                    primary_video_payload = camera_video_payload
            else:
                video_payload = primary_video_payload

            if video_payload is None:
                continue

            source_ts = frame.get("source_ts_s")
            monotonic_ts = frame.get("monotonic_ts_s")
            if self.config.schema in ROKAE_LINKER_L10_SCHEMAS and isinstance(source_ts, (int, float)):
                timestamp = float(source_ts)
            elif isinstance(monotonic_ts, (int, float)):
                timestamp = float(monotonic_ts)
            elif isinstance(source_ts, (int, float)):
                timestamp = float(source_ts)
            else:
                timestamp = float(len(step_payloads))

            raw_frame_index = frame.get("teleop_frame_id")
            video_frame_index = video_payload.get("frame_index")
            if isinstance(raw_frame_index, int):
                frame_index = raw_frame_index
            elif isinstance(video_frame_index, int):
                frame_index = video_frame_index
            else:
                frame_index = len(step_payloads)

            step_payloads.append(
                {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "state": state_bundle.values,
                    "action": action_bundle.values,
                    "video_path": video_paths[video_cameras[0]],
                    "video_width": video_widths[video_cameras[0]],
                    "video_height": video_heights[video_cameras[0]],
                    "video_paths": video_paths,
                    "video_widths": video_widths,
                    "video_heights": video_heights,
                    "video_frame_indices": video_frame_indices,
                    "video_frame_ordinals": video_frame_ordinals,
                }
            )

        if not step_payloads:
            if missing_robot_state > 0 and missing_video == 0:
                return None, "missing_robot_state"
            if missing_video > 0 and missing_robot_state == 0:
                return None, "missing_selected_camera_frames"
            if missing_video > 0 or missing_robot_state > 0:
                return None, f"filtered_all_steps missing_robot_state={missing_robot_state} missing_video={missing_video}"
            return None, "no_exportable_steps"

        steps: list[NormalizedStep] = []
        for export_step_index, payload in enumerate(step_payloads):
            steps.append(
                NormalizedStep(
                    episode_index=episode_index,
                    step_index=export_step_index,
                    frame_index=int(payload["frame_index"]),
                    timestamp=float(payload["timestamp"]),
                    task=task,
                    state=list(payload["state"]),
                    action=list(payload["action"]),
                    video_path=Path(payload["video_path"]),
                    video_width=int(payload["video_width"]),
                    video_height=int(payload["video_height"]),
                    done=(export_step_index == len(step_payloads) - 1),
                    reward=0.0,
                    video_paths=dict(payload["video_paths"]),
                    video_widths=dict(payload["video_widths"]),
                    video_heights=dict(payload["video_heights"]),
                    video_frame_indices=dict(payload["video_frame_indices"]),
                    video_frame_ordinals=dict(payload["video_frame_ordinals"]),
                )
            )

        return (
            NormalizedEpisode(
                episode_index=episode_index,
                raw_episode_dir=episode_dir,
                task=task,
                success=success,
                steps=steps,
                state_names=state_names or [],
                state_slices=state_slices or {},
                action_names=action_names or [],
                action_slices=action_slices or {},
                metadata=metadata,
            ),
            None,
        )

    def _write_episode_parquet(
        self,
        *,
        pyarrow: Any,
        pyarrow_parquet: Any,
        episode: NormalizedEpisode,
        parquet_path: Path,
        global_index_start: int,
        task_index: int,
        validity_index: int | None,
    ) -> None:
        table = pyarrow.table(
            {
                "index": [global_index_start + offset for offset, _ in enumerate(episode.steps)],
                "episode_index": [step.episode_index for step in episode.steps],
                "frame_index": [step.frame_index for step in episode.steps],
                "timestamp": [step.timestamp for step in episode.steps],
                "task_index": [task_index for _ in episode.steps],
                "annotation.human.action.task_description": [task_index for _ in episode.steps],
                "annotation.human.validity": [validity_index for _ in episode.steps],
                "observation.state": [step.state for step in episode.steps],
                "action": [step.action for step in episode.steps],
                "next.reward": [step.reward for step in episode.steps],
                "next.done": [step.done for step in episode.steps],
            }
        )
        pyarrow_parquet.write_table(table, parquet_path)

    def _encode_video(
        self,
        *,
        ffmpeg: str,
        episode: NormalizedEpisode,
        output_path: Path,
        camera_name: str | None = None,
    ) -> None:
        unique_video_paths = []
        seen_paths: set[Path] = set()
        for step in episode.steps:
            video_path = step.video_paths.get(camera_name, step.video_path) if camera_name is not None else step.video_path
            if video_path not in seen_paths:
                unique_video_paths.append(video_path)
                seen_paths.add(video_path)

        if len(unique_video_paths) == 1 and unique_video_paths[0].suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            camera_key = camera_name or next(iter(episode.steps[0].video_paths), "")
            frame_ordinals = [step.video_frame_ordinals.get(camera_key) for step in episode.steps]
            if not all(isinstance(frame_ordinal, int) for frame_ordinal in frame_ordinals):
                raise RuntimeError(f"Missing raw video frame ordinals for camera {camera_key!r}.")
            ordinal_values = [int(frame_ordinal) for frame_ordinal in frame_ordinals]
            if ordinal_values != sorted(ordinal_values):
                raise RuntimeError(f"Video frame ordinals must be non-decreasing for camera {camera_key!r}.")
            self._encode_video_by_ordinals(
                ffmpeg=ffmpeg,
                input_video_path=unique_video_paths[0],
                output_path=output_path,
                frame_ordinals=ordinal_values,
                width=episode.steps[0].video_widths.get(camera_key, episode.steps[0].video_width),
                height=episode.steps[0].video_heights.get(camera_key, episode.steps[0].video_height),
            )
            return

        with tempfile.TemporaryDirectory(prefix="groot_lerobot_v2_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            suffix = episode.steps[0].video_path.suffix or ".ppm"
            for index, step in enumerate(episode.steps):
                frame_path = tmp_dir / f"{index:06d}{suffix}"
                try:
                    frame_path.symlink_to(step.video_path)
                except OSError:
                    shutil.copy2(step.video_path, frame_path)
            input_pattern = str(tmp_dir / f"%06d{suffix}")
            command = [
                ffmpeg,
                "-y",
                "-framerate",
                str(max(1, int(self.config.fps))),
                "-i",
                input_pattern,
                "-frames:v",
                str(len(episode.steps)),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _encode_video_by_ordinals(
        self,
        *,
        ffmpeg: str,
        input_video_path: Path,
        output_path: Path,
        frame_ordinals: list[int],
        width: int,
        height: int,
    ) -> None:
        cv2 = _require_optional_dependency("cv2")
        capture = cv2.VideoCapture(str(input_video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open raw video for export: {input_video_path}")

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(max(1, int(self.config.fps))),
            "-i",
            "pipe:0",
            "-an",
            "-frames:v",
            str(len(frame_ordinals)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        with tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
            try:
                if process.stdin is None:
                    raise RuntimeError("Could not open ffmpeg stdin for video export.")
                current_ordinal = -1
                current_frame = None
                for frame_ordinal in frame_ordinals:
                    while current_ordinal < frame_ordinal:
                        ok, current_frame = capture.read()
                        current_ordinal += 1
                        if not ok or current_frame is None:
                            raise RuntimeError(
                                f"Raw video ended before frame ordinal {frame_ordinal}: {input_video_path}"
                            )
                    frame_height, frame_width = current_frame.shape[:2]
                    if frame_width != int(width) or frame_height != int(height):
                        raise RuntimeError(
                            f"Raw video frame shape changed for {input_video_path}: "
                            f"expected {(int(height), int(width))}, got {(frame_height, frame_width)}"
                        )
                    process.stdin.write(current_frame.tobytes())
                process.stdin.close()
                status = process.wait()
            finally:
                capture.release()
                if process.poll() is None:
                    process.kill()
            if status != 0:
                stderr.seek(0)
                error_text = stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed while encoding {output_path}: {error_text}")

    def _load_session_metadata(self, raw_capture_dir: Path) -> dict[str, object]:
        session_path = raw_capture_dir / "session.json"
        if not session_path.is_file():
            return {}
        return _load_json(session_path)

    def _resolve_video_cameras(self) -> tuple[str, ...]:
        configured = self.config.selected_cameras
        if configured is None:
            configured = (self.config.selected_camera,)
        cameras = tuple(camera.strip() for camera in configured if camera.strip())
        if not cameras:
            raise RuntimeError("At least one selected camera is required for GR00T LeRobot export.")
        if len(set(cameras)) != len(cameras):
            raise RuntimeError(f"Selected cameras must be unique: {cameras!r}")
        return cameras

    def _resolve_video_aliases(self, cameras: tuple[str, ...]) -> dict[str, str]:
        configured = self.config.video_aliases
        if configured is not None:
            aliases = tuple(alias.strip() for alias in configured)
            if len(aliases) != len(cameras) or any(not alias for alias in aliases):
                raise RuntimeError("--video-alias count must match --camera count when exporting multiple cameras.")
            if len(set(aliases)) != len(aliases):
                raise RuntimeError(f"Video aliases must be unique: {aliases!r}")
            return dict(zip(cameras, aliases, strict=True))

        aliases_by_camera: dict[str, str] = {}
        for camera in cameras:
            if camera == self.config.selected_camera:
                aliases_by_camera[camera] = self._resolve_video_alias()
            elif camera == "wrist_d405_rgb":
                aliases_by_camera[camera] = "wrist_view"
            else:
                aliases_by_camera[camera] = camera
        if len(set(aliases_by_camera.values())) != len(aliases_by_camera):
            raise RuntimeError(f"Video aliases must be unique: {aliases_by_camera!r}")
        return aliases_by_camera

    def _resolve_video_feature_keys(
        self,
        cameras: tuple[str, ...],
        aliases_by_camera: dict[str, str],
    ) -> dict[str, str]:
        configured = self.config.video_feature_keys
        if configured is not None:
            feature_keys = tuple(feature_key.strip() for feature_key in configured)
            if len(feature_keys) != len(cameras) or any(not feature_key for feature_key in feature_keys):
                raise RuntimeError("--video-feature-key count must match --camera count when exporting multiple cameras.")
            if len(set(feature_keys)) != len(feature_keys):
                raise RuntimeError(f"Video feature keys must be unique: {feature_keys!r}")
            return dict(zip(cameras, feature_keys, strict=True))

        return {
            camera: (
                self._resolve_video_feature_key(alias)
                if camera == self.config.selected_camera
                else f"observation.images.{alias}"
            )
            for camera, alias in aliases_by_camera.items()
        }

    def _resolve_video_alias(self) -> str:
        alias = (self.config.video_alias or "").strip()
        if alias:
            return alias
        return self.config.selected_camera

    def _resolve_video_feature_key(self, video_alias: str) -> str:
        feature_key = (self.config.video_feature_key or "").strip()
        if feature_key:
            return feature_key
        return f"observation.images.{video_alias}"

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
                handle.write("\n")
