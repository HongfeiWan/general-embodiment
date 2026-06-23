#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Frame-accurate L10 multiview LeRobot episode trim viewer.

Run with a lightweight data-cleaning environment, for example:

    streamlit run tools/data_chain/trim_lerobot_episode_viewer.py
    streamlit run tools/data_chain/trim_lerobot_episode_viewer.py -- \
        --dataset-dir missions/nero/mission2/lerobot_v2

Expected packages:

    pip install streamlit opencv-python pandas pyarrow

The tool is designed for the L10 multiview LeRobot-v2 datasets where ego_view
and wrist_view are both present and each video frame maps 1:1 to one parquet
row. It lets you inspect an episode by frame index, choose a start/end frame,
and export the selected range as a new episode in another LeRobot dataset
directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import streamlit as st
except ImportError:
    st = None


def _has_streamlit_context() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def _cache_data(**kwargs):
    if _has_streamlit_context():
        return st.cache_data(**kwargs)

    def decorator(func):
        return func

    return decorator


DEFAULT_DATASET_ROOT = REPO_ROOT / "missions" / "nero" / "mission2" / "lerobot_v2"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "missions"
DEFAULT_OUTPUT_DATASET_NAME = "trimmed"
TRIM_MANIFEST_FILENAME = "trim_manifest.jsonl"
REQUIRED_VIDEO_KEYS = ("ego_view", "wrist_view")


@dataclass(frozen=True)
class EpisodePaths:
    data_path: Path
    video_path: Path
    chunk_index: int
    video_original_key: str


@dataclass(frozen=True)
class DatasetRouting:
    hardware: str
    mission: str
    mission_description: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Source LeRobot v2 root. The app only reads datasets under date folders "
            "inside this directory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for routed trimmed datasets. The app writes to "
            "<output-root>/<hardware>/<mission>/trimmed by default."
        ),
    )
    parser.add_argument(
        "--output-dataset-name",
        default=DEFAULT_OUTPUT_DATASET_NAME,
        help="Leaf dataset directory name under <output-root>/<hardware>/<mission>.",
    )
    parser.add_argument(
        "--codec",
        choices=("mp4v", "avc1", "H264"),
        default="H264",
        help="Default mp4 codec selected in the sidebar.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").exists()


def _discover_date_dirs(lerobot_root: Path) -> list[Path]:
    if not lerobot_root.exists():
        return []
    return sorted(
        child
        for child in lerobot_root.iterdir()
        if child.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", child.name)
    )


def _discover_source_datasets(lerobot_root: Path, selected_dates: set[str]) -> list[Path]:
    datasets: list[Path] = []
    for date_dir in _discover_date_dirs(lerobot_root):
        if date_dir.name not in selected_dates:
            continue
        for child in sorted(date_dir.iterdir()):
            if child.is_dir() and _is_lerobot_dataset(child):
                datasets.append(child)
    return datasets


def _format_dataset_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_path_component(value: str | None, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        text = fallback
    text = text.replace("\\", "/").strip("/")
    text = Path(text).name if "/" in text else text
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or fallback


def _mission_description_from_file(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("- mission:"):
            return stripped.split(":", 1)[1].strip() or None
    return None


def _find_mission_file(source_dataset: Path) -> Path | None:
    for parent in (source_dataset, *source_dataset.parents):
        candidate = parent / "MISSION.md"
        if candidate.exists():
            return candidate
        if parent == REPO_ROOT:
            break
    return None


def _infer_source_routing(source_dataset: Path) -> DatasetRouting:
    mission_file = _find_mission_file(source_dataset)
    mission_description = _mission_description_from_file(mission_file) if mission_file else None

    hardware: str | None = None
    mission: str | None = None
    try:
        rel_parts = source_dataset.resolve().relative_to(DEFAULT_DATASET_ROOT.resolve()).parts
    except ValueError:
        rel_parts = ()
    if rel_parts:
        hardware = rel_parts[0]
        if len(rel_parts) > 1 and rel_parts[1].startswith("mission"):
            mission = rel_parts[1]

    if mission is None:
        for part in source_dataset.parts:
            if part.startswith("mission"):
                mission = part
                break

    if hardware is None:
        lowered = [part.lower() for part in source_dataset.parts]
        if "nero" in lowered:
            hardware = "nero"
        elif "rokae" in lowered:
            hardware = "rokae"
        else:
            try:
                info = _read_json(source_dataset / "meta" / "info.json")
            except Exception:
                info = {}
            teleop_stack = info.get("teleop_stack")
            robot_type = str(
                info.get("robot_type")
                or (teleop_stack.get("robot_type") if isinstance(teleop_stack, dict) else "")
                or ""
            ).lower()
            if "nero" in robot_type:
                hardware = "nero"
            elif "rokae" in robot_type or "xmate" in robot_type:
                hardware = "rokae"

    return DatasetRouting(
        hardware=_safe_path_component(hardware, "unknown_hardware"),
        mission=_safe_path_component(mission, "mission_unspecified"),
        mission_description=mission_description,
    )


def _routed_output_dataset(
    *,
    output_root: Path,
    routing: DatasetRouting,
    output_dataset_name: str,
) -> Path:
    return (
        output_root
        / _safe_path_component(routing.hardware, "unknown_hardware")
        / _safe_path_component(routing.mission, "mission_unspecified")
        / _safe_path_component(output_dataset_name, DEFAULT_OUTPUT_DATASET_NAME)
    )


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_episode_paths(dataset_dir: Path, episode_index: int, video_key: str) -> EpisodePaths:
    info = _read_json(dataset_dir / "meta" / "info.json")
    modality = _read_json(dataset_dir / "meta" / "modality.json")
    chunk_size = int(info.get("chunks_size", 1000))
    chunk_index = episode_index // chunk_size
    video_original_key = modality["video"][video_key].get(
        "original_key", f"observation.images.{video_key}"
    )
    data_path = dataset_dir / info["data_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )
    video_path = dataset_dir / info["video_path"].format(
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key=video_original_key,
    )
    return EpisodePaths(
        data_path=data_path,
        video_path=video_path,
        chunk_index=chunk_index,
        video_original_key=video_original_key,
    )


@_cache_data(show_spinner=False)
def _load_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


@_cache_data(show_spinner=False)
def _video_metadata(video_path: str) -> dict[str, int | float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


@_cache_data(show_spinner=False)
def _load_episode_table(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def _validate_existing_output_meta(source_dataset: Path, output_meta: Path) -> None:
    source_modality = _read_json(source_dataset / "meta" / "modality.json")
    output_modality = _read_json(output_meta / "modality.json")
    updated = False
    for group in ("video", "state", "action"):
        source_group = source_modality.get(group, {})
        output_group = output_modality.get(group, {})
        conflicting_keys = [
            key
            for key, value in output_group.items()
            if key in source_group and source_group[key] != value
        ]
        extra_keys = [key for key in output_group if key not in source_group]
        if conflicting_keys or extra_keys:
            raise RuntimeError(
                f"Output dataset {group} schema does not match source dataset. "
                f"Use a fresh output directory for this dataset layout.\n"
                f"source={source_group}\noutput={output_group}"
            )
        missing_keys = [key for key in source_group if key not in output_group]
        if missing_keys:
            output_modality.setdefault(group, {}).update(
                {key: source_group[key] for key in missing_keys}
            )
            updated = True

    if updated:
        _write_json(output_meta / "modality.json", output_modality)

    source_tasks_path = source_dataset / "meta" / "tasks.jsonl"
    output_tasks_path = output_meta / "tasks.jsonl"
    if source_tasks_path.exists() and output_tasks_path.exists():
        source_tasks = _read_jsonl(source_tasks_path)
        output_tasks = _read_jsonl(output_tasks_path)
        if source_tasks != output_tasks:
            raise RuntimeError(
                "Output dataset tasks.jsonl does not match source dataset tasks.jsonl. "
                "Use a fresh output directory or normalize task_index/task labels before appending."
            )


def _copy_base_meta(source_dataset: Path, output_dataset: Path) -> None:
    output_meta = output_dataset / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    existing_modality_path = output_meta / "modality.json"
    if existing_modality_path.exists():
        _validate_existing_output_meta(source_dataset, output_meta)

    for src in (source_dataset / "meta").iterdir():
        if src.is_file() and src.name not in {"episodes.jsonl", "info.json", "stats.json", "relative_stats.json"}:
            if (output_meta / src.name).exists():
                continue
            shutil.copy2(src, output_meta / src.name)
    if not (output_meta / "episodes.jsonl").exists():
        _write_jsonl(output_meta / "episodes.jsonl", [])
    if not (output_meta / "info.json").exists():
        info = _read_json(source_dataset / "meta" / "info.json")
        info["total_episodes"] = 0
        info["total_frames"] = 0
        info["total_videos"] = 0
        info["total_chunks"] = 0
        info["splits"] = {"train": "0:0"}
        _write_json(output_meta / "info.json", info)


def _next_episode_index(output_dataset: Path) -> int:
    rows = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    if not rows:
        return 0
    return max(int(row["episode_index"]) for row in rows) + 1


def _current_total_frames(output_dataset: Path) -> int:
    rows = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    return sum(int(row["length"]) for row in rows)


def _write_trimmed_video(
    *,
    source_video: Path,
    output_video: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    codec: str,
) -> None:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    transcode_h264 = codec in {"avc1", "H264"}
    writer_codec = "mp4v" if transcode_h264 else codec
    writer_video = (
        output_video.with_name(f"{output_video.stem}.tmp-mp4v{output_video.suffix}")
        if transcode_h264
        else output_video
    )
    if writer_video.exists():
        writer_video.unlink()
    writer = cv2.VideoWriter(
        str(writer_video),
        cv2.VideoWriter_fourcc(*writer_codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open output video writer: {writer_video} codec={writer_codec}"
        )

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {source_video}")
            writer.write(frame_bgr)
    finally:
        writer.release()
        cap.release()

    if transcode_h264:
        h264_tmp = output_video.with_name(f"{output_video.stem}.tmp-h264{output_video.suffix}")
        if h264_tmp.exists():
            h264_tmp.unlink()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(writer_video),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(h264_tmp),
                ],
                check=True,
            )
            h264_tmp.replace(output_video)
        finally:
            if writer_video.exists():
                writer_video.unlink()
            if h264_tmp.exists():
                h264_tmp.unlink()


def _trim_dataframe(
    df: pd.DataFrame,
    *,
    start_frame: int,
    end_frame: int,
    new_episode_index: int,
    global_start_index: int,
    fps: float,
) -> pd.DataFrame:
    out = df.iloc[start_frame : end_frame + 1].copy().reset_index(drop=True)
    length = len(out)
    if "episode_index" in out.columns:
        out["episode_index"] = new_episode_index
    if "frame_index" in out.columns:
        out["frame_index"] = np.arange(length, dtype=np.int64)
    if "index" in out.columns:
        out["index"] = global_start_index + np.arange(length, dtype=np.int64)
    if "timestamp" in out.columns:
        out["timestamp"] = np.arange(length, dtype=np.float32) / float(fps)
    if "next.done" in out.columns:
        out["next.done"] = False
        out.loc[length - 1, "next.done"] = True
    return out


def _update_output_info(output_dataset: Path) -> None:
    info_path = output_dataset / "meta" / "info.json"
    info = _read_json(info_path)
    episodes = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    total_episodes = len(episodes)
    total_frames = sum(int(row["length"]) for row in episodes)
    chunk_size = int(info.get("chunks_size", 1000))
    total_chunks = 0 if total_episodes == 0 else ((total_episodes - 1) // chunk_size + 1)
    num_video_keys = len(_read_json(output_dataset / "meta" / "modality.json").get("video", {}))

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes * num_video_keys
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    _write_json(info_path, info)


def _append_trimmed_episode(
    *,
    source_dataset: Path,
    output_dataset: Path,
    routing: DatasetRouting | None = None,
    source_episode_index: int,
    primary_video_key: str,
    start_frame: int,
    end_frame: int,
    codec: str,
) -> dict[str, Any]:
    if source_dataset.resolve() == output_dataset.resolve():
        raise RuntimeError("Source dataset and fixed trimmed output dataset must be different.")

    _copy_base_meta(source_dataset, output_dataset)

    source_info = _read_json(source_dataset / "meta" / "info.json")
    source_modality = _read_json(source_dataset / "meta" / "modality.json")
    output_info = _read_json(output_dataset / "meta" / "info.json")
    source_episodes = _read_jsonl(source_dataset / "meta" / "episodes.jsonl")
    source_ep = next(
        row for row in source_episodes if int(row["episode_index"]) == int(source_episode_index)
    )

    video_keys = list(source_modality.get("video", {}))
    missing_video_keys = [key for key in REQUIRED_VIDEO_KEYS if key not in video_keys]
    if missing_video_keys:
        raise RuntimeError(
            "Source dataset must contain ego_view and wrist_view in meta/modality.json; "
            f"missing={missing_video_keys}, available={video_keys}"
        )
    if primary_video_key not in video_keys:
        raise RuntimeError(f"Primary video key {primary_video_key!r} not in {video_keys}")

    paths_by_key = {
        key: _resolve_episode_paths(source_dataset, source_episode_index, key)
        for key in video_keys
    }
    primary_paths = paths_by_key[primary_video_key]
    df = _load_episode_table(str(primary_paths.data_path))
    fps = float(source_info.get("fps", 10))
    length = int(end_frame - start_frame + 1)

    new_episode_index = _next_episode_index(output_dataset)
    global_start_index = _current_total_frames(output_dataset)
    chunk_size = int(output_info.get("chunks_size", 1000))
    new_chunk_index = new_episode_index // chunk_size

    out_df = _trim_dataframe(
        df,
        start_frame=start_frame,
        end_frame=end_frame,
        new_episode_index=new_episode_index,
        global_start_index=global_start_index,
        fps=fps,
    )

    output_data_path = output_dataset / output_info["data_path"].format(
        episode_chunk=new_chunk_index,
        episode_index=new_episode_index,
    )
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_data_path, index=False)

    output_video_paths: dict[str, Path] = {}
    source_video_paths: dict[str, Path] = {}
    for key, paths in paths_by_key.items():
        source_video_paths[key] = paths.video_path
        output_video_path = output_dataset / output_info["video_path"].format(
            episode_chunk=new_chunk_index,
            episode_index=new_episode_index,
            video_key=paths.video_original_key,
        )
        _write_trimmed_video(
            source_video=paths.video_path,
            output_video=output_video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            codec=codec,
        )
        output_video_paths[key] = output_video_path

    episode_row = dict(source_ep)
    episode_row["episode_index"] = new_episode_index
    episode_row["length"] = length
    metadata = dict(episode_row.get("teleop_stack_metadata", {}))
    metadata["source_dataset"] = str(source_dataset.resolve())
    metadata["source_episode_index"] = source_episode_index
    metadata["source_start_frame"] = start_frame
    metadata["source_end_frame"] = end_frame
    metadata["source_length"] = int(source_ep["length"])
    metadata["trimmed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if routing is not None:
        metadata["hardware"] = routing.hardware
        metadata["mission"] = routing.mission
        if routing.mission_description:
            metadata["mission_description"] = routing.mission_description
    metadata["data_path"] = str(output_data_path.relative_to(output_dataset))
    metadata["primary_video_key"] = primary_video_key
    metadata["video_path"] = str(output_video_paths[primary_video_key].relative_to(output_dataset))
    metadata["video_paths"] = {
        key: str(path.relative_to(output_dataset))
        for key, path in output_video_paths.items()
    }
    episode_row["teleop_stack_metadata"] = metadata

    episodes = _read_jsonl(output_dataset / "meta" / "episodes.jsonl")
    episodes.append(episode_row)
    _write_jsonl(output_dataset / "meta" / "episodes.jsonl", episodes)
    _update_output_info(output_dataset)

    result = {
        "new_episode_index": new_episode_index,
        "length": length,
        "data_path": str(output_data_path),
        "primary_video_key": primary_video_key,
        "video_paths": {key: str(path) for key, path in output_video_paths.items()},
    }
    manifest_row = {
        "created_at_utc": metadata["trimmed_at_utc"],
        "output_dataset": str(output_dataset.resolve()),
        "output_dataset_name": output_dataset.name,
        "output_hardware": routing.hardware if routing else None,
        "output_mission": routing.mission if routing else None,
        "mission_description": routing.mission_description if routing else None,
        "output_episode_index": new_episode_index,
        "output_episode_length": length,
        "output_data_path": str(output_data_path.relative_to(output_dataset)),
        "output_video_paths": {
            key: str(path.relative_to(output_dataset))
            for key, path in output_video_paths.items()
        },
        "source_dataset": str(source_dataset.resolve()),
        "source_dataset_name": source_dataset.name,
        "source_episode_index": int(source_episode_index),
        "source_episode_length": int(source_ep["length"]),
        "source_data_path": str(primary_paths.data_path.relative_to(source_dataset)),
        "source_video_paths": {
            key: str(path.relative_to(source_dataset))
            for key, path in source_video_paths.items()
        },
        "trim_start_frame": int(start_frame),
        "trim_end_frame": int(end_frame),
        "trim_inclusive": True,
        "fps": fps,
        "codec": codec,
        "primary_video_key": primary_video_key,
        "video_keys": video_keys,
        "source_tasks": source_ep.get("tasks", []),
    }
    _append_jsonl_row(output_dataset / TRIM_MANIFEST_FILENAME, manifest_row)
    _append_jsonl_row(output_dataset / "meta" / TRIM_MANIFEST_FILENAME, manifest_row)
    result["manifest_path"] = str(output_dataset / TRIM_MANIFEST_FILENAME)
    return result


def _format_vector(value: Any, max_items: int = 6) -> str:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return str(value)
    shown = ", ".join(f"{x:.4f}" for x in arr[:max_items])
    suffix = ", ..." if arr.size > max_items else ""
    return f"[{shown}{suffix}] shape={arr.shape}"


def main() -> None:
    args = _parse_args()
    if st is None:
        raise SystemExit(
            "streamlit is not installed. Install it with: pip install streamlit opencv-python pandas pyarrow"
        )
    if not _has_streamlit_context():
        raise SystemExit(
            "This viewer is a Streamlit app. Run it with:\n\n"
            "    streamlit run trim_lerobot_episode_viewer.py\n\n"
            "To pass arguments, use:\n\n"
            "    streamlit run trim_lerobot_episode_viewer.py -- "
            "--dataset-dir <dataset_dir>"
        )

    st.set_page_config(page_title="LeRobot Episode Trim Viewer", layout="wide")
    st.title("LeRobot Episode Trim Viewer")

    with st.sidebar:
        dataset_root = Path(
            st.text_input("LeRobot v2 root", value=str(args.dataset_dir.expanduser()))
        ).expanduser()
        output_root = Path(
            st.text_input("Output root", value=str(args.output_root.expanduser()))
        ).expanduser()
        output_dataset_name = st.text_input(
            "Output dataset name",
            value=_safe_path_component(args.output_dataset_name, DEFAULT_OUTPUT_DATASET_NAME),
            help="The routed dataset path is <output-root>/<hardware>/<mission>/<name>.",
        )
        date_dirs = _discover_date_dirs(dataset_root)
        if not date_dirs:
            st.error(
                "No dated LeRobot v2 folders found. Expected "
                "<lerobot_v2_root>/<YYYY-MM-DD>/<dataset>/meta/info.json."
            )
            st.stop()

        selected_dates: set[str] = set()
        st.caption("Capture dates")
        date_columns = st.columns(min(3, len(date_dirs)))
        for index, date_dir in enumerate(date_dirs):
            with date_columns[index % len(date_columns)]:
                if st.checkbox(date_dir.name, value=True, key=f"date::{date_dir.name}"):
                    selected_dates.add(date_dir.name)
        if not selected_dates:
            st.error("Select at least one capture date.")
            st.stop()

        source_datasets = _discover_source_datasets(dataset_root, selected_dates)
        if not source_datasets:
            st.error("No LeRobot v2 dataset found for the selected dates.")
            st.stop()
        if len(source_datasets) == 1:
            dataset_dir = source_datasets[0]
            st.caption(f"Source dataset: {dataset_dir}")
        else:
            selected_source = st.selectbox(
                "Source capture dataset",
                source_datasets,
                format_func=lambda path: _format_dataset_label(path, dataset_root),
                index=0,
            )
            dataset_dir = Path(selected_source)

        inferred_routing = _infer_source_routing(dataset_dir)
        hardware = st.text_input(
            "Hardware folder",
            value=inferred_routing.hardware,
            key=f"hardware::{dataset_dir}",
            help="Used to route trimmed data under the output root.",
        )
        mission = st.text_input(
            "Mission folder",
            value=inferred_routing.mission,
            key=f"mission::{dataset_dir}",
            help="Used to keep tasks for the same hardware separated.",
        )
        routing = DatasetRouting(
            hardware=_safe_path_component(hardware, inferred_routing.hardware),
            mission=_safe_path_component(mission, inferred_routing.mission),
            mission_description=inferred_routing.mission_description,
        )
        output_dir = _routed_output_dataset(
            output_root=output_root,
            routing=routing,
            output_dataset_name=output_dataset_name,
        )
        st.caption(f"Routed output dataset: {output_dir}")
        if routing.mission_description:
            st.caption(f"Mission: {routing.mission_description}")
        if dataset_dir.resolve() == output_dir.resolve():
            st.error("Source dataset cannot be the routed trimmed output dataset.")
            st.stop()
        st.caption(f"Next output episode index: {_next_episode_index(output_dir)}")
        codecs = ["mp4v", "avc1", "H264"]
        codec = st.selectbox("Output mp4 codec", codecs, index=codecs.index(args.codec))

        info = _read_json(dataset_dir / "meta" / "info.json")
        modality = _read_json(dataset_dir / "meta" / "modality.json")
        episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        episode_ids = [int(row["episode_index"]) for row in episodes]
        video_keys = list(modality.get("video", {}).keys())
        missing_video_keys = [key for key in REQUIRED_VIDEO_KEYS if key not in video_keys]
        if missing_video_keys:
            st.error(
                "Dataset meta/modality.json must contain ego_view and wrist_view.\n"
                f"Missing: {missing_video_keys}\nAvailable: {video_keys}"
            )
            st.stop()

        selected_episode = st.selectbox("Episode", episode_ids, index=0)
        primary_index = video_keys.index("ego_view")
        primary_video_key = st.selectbox(
            "Primary trim view",
            video_keys,
            index=primary_index,
            help="Use this view as the main visual reference. Export still trims every video key.",
        )
        st.caption(f"fps={info.get('fps', 'unknown')} total_episodes={len(episodes)}")

    paths_by_key = {
        key: _resolve_episode_paths(dataset_dir, int(selected_episode), key)
        for key in video_keys
    }
    paths = paths_by_key[primary_video_key]
    if not paths.data_path.exists():
        st.error(f"Parquet not found: {paths.data_path}")
        st.stop()
    missing_video_paths = [
        str(video_paths.video_path)
        for video_paths in paths_by_key.values()
        if not video_paths.video_path.exists()
    ]
    if missing_video_paths:
        st.error("Video not found:\n" + "\n".join(missing_video_paths))
        st.stop()

    df = _load_episode_table(str(paths.data_path))
    video_meta_by_key = {
        key: _video_metadata(str(video_paths.video_path))
        for key, video_paths in paths_by_key.items()
    }
    frame_count = min(
        len(df),
        *(int(video_meta["frame_count"]) for video_meta in video_meta_by_key.values()),
    )
    if frame_count <= 0:
        st.error("No frames found.")
        st.stop()

    if "trim_start" not in st.session_state:
        st.session_state.trim_start = 0
    if "trim_end" not in st.session_state:
        st.session_state.trim_end = frame_count - 1
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0
    st.session_state.frame_index = min(max(0, int(st.session_state.frame_index)), frame_count - 1)
    st.session_state.trim_start = min(max(0, int(st.session_state.trim_start)), frame_count - 1)
    st.session_state.trim_end = min(max(0, int(st.session_state.trim_end)), frame_count - 1)
    if st.session_state.trim_start > st.session_state.trim_end:
        st.session_state.trim_end = st.session_state.trim_start

    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Video Preview")
        preview_cols = st.columns(len(video_keys))
        for col, video_key in zip(preview_cols, video_keys):
            with col:
                st.caption(video_key)
                st.video(str(paths_by_key[video_key].video_path))
        st.caption(
            "Use the exact frame slider below for frame-accurate start/end selection. "
            "The native video players are only for quick visual playback."
        )

    with right:
        st.subheader("Exact Synchronized Frame")
        col_prev, col_slider, col_next = st.columns([0.12, 0.76, 0.12])
        with col_prev:
            if st.button("<", use_container_width=True):
                st.session_state.frame_index = max(0, int(st.session_state.frame_index) - 1)
        with col_next:
            if st.button(">", use_container_width=True):
                st.session_state.frame_index = min(frame_count - 1, int(st.session_state.frame_index) + 1)
        with col_slider:
            st.session_state.frame_index = st.slider(
                "Frame index",
                min_value=0,
                max_value=frame_count - 1,
                value=int(st.session_state.frame_index),
                step=1,
            )

        frame_cols = st.columns(len(video_keys))
        for col, video_key in zip(frame_cols, video_keys):
            with col:
                frame = _load_frame(
                    str(paths_by_key[video_key].video_path),
                    int(st.session_state.frame_index),
                )
                st.caption(video_key)
                st.image(frame, channels="RGB", use_container_width=True)

        row = df.iloc[int(st.session_state.frame_index)]
        st.write(
            {
                "episode": int(selected_episode),
                "frame": int(st.session_state.frame_index),
                "timestamp": float(row.get("timestamp", st.session_state.frame_index / float(info.get("fps", 10)))),
                "parquet_rows": int(len(df)),
                "video_frames": {
                    key: int(video_meta["frame_count"])
                    for key, video_meta in video_meta_by_key.items()
                },
            }
        )
        with st.expander("Current row preview"):
            if "observation.state" in row:
                st.text(f"state:  {_format_vector(row['observation.state'])}")
            if "action" in row:
                st.text(f"action: {_format_vector(row['action'])}")
            st.json(
                {
                    key: row[key].item() if hasattr(row[key], "item") else row[key]
                    for key in row.index
                    if key not in {"observation.state", "action"}
                }
            )

    st.divider()
    st.subheader("Trim Range")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Set Start = Current Frame", use_container_width=True):
            st.session_state.trim_start = int(st.session_state.frame_index)
            if st.session_state.trim_start > st.session_state.trim_end:
                st.session_state.trim_end = st.session_state.trim_start
    with c2:
        if st.button("Set End = Current Frame", use_container_width=True):
            st.session_state.trim_end = int(st.session_state.frame_index)
            if st.session_state.trim_end < st.session_state.trim_start:
                st.session_state.trim_start = st.session_state.trim_end
    with c3:
        st.session_state.trim_start = st.number_input(
            "Start frame", min_value=0, max_value=frame_count - 1, value=int(st.session_state.trim_start)
        )
    with c4:
        st.session_state.trim_end = st.number_input(
            "End frame", min_value=0, max_value=frame_count - 1, value=int(st.session_state.trim_end)
        )

    start_frame = int(st.session_state.trim_start)
    end_frame = int(st.session_state.trim_end)
    if start_frame > end_frame:
        st.error("Start frame must be <= end frame.")
        st.stop()

    selected_len = end_frame - start_frame + 1
    st.info(
        f"Selected frames: {start_frame}..{end_frame} inclusive, "
        f"length={selected_len}, duration={selected_len / float(info.get('fps', 10)):.2f}s"
    )

    export_col, preview_col = st.columns([0.25, 0.75])
    with export_col:
        if st.button("Export Trimmed Episode", type="primary", use_container_width=True):
            with st.spinner("Writing trimmed parquet and mp4..."):
                result = _append_trimmed_episode(
                    source_dataset=dataset_dir,
                    output_dataset=output_dir,
                    routing=routing,
                    source_episode_index=int(selected_episode),
                    primary_video_key=primary_video_key,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    codec=codec,
                )
            st.success(f"Exported episode {result['new_episode_index']} length={result['length']}")
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
    with preview_col:
        st.caption(f"Output dataset: {output_dir}")
        st.caption(f"Manifest: {output_dir / TRIM_MANIFEST_FILENAME}")


if __name__ == "__main__":
    main()
