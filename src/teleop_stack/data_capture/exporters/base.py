from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class NormalizedStep:
    episode_index: int
    step_index: int
    frame_index: int
    timestamp: float
    task: str
    state: list[float]
    action: list[float]
    video_path: Path
    video_width: int
    video_height: int
    done: bool
    reward: float
    video_paths: dict[str, Path] = field(default_factory=dict)
    video_widths: dict[str, int] = field(default_factory=dict)
    video_heights: dict[str, int] = field(default_factory=dict)
    video_frame_indices: dict[str, int] = field(default_factory=dict)
    video_frame_ordinals: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedEpisode:
    episode_index: int
    raw_episode_dir: Path
    task: str
    success: bool | None
    steps: list[NormalizedStep]
    state_names: list[str]
    state_slices: dict[str, dict[str, int]]
    action_names: list[str]
    action_slices: dict[str, dict[str, int]]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ExportSummary:
    format_name: str
    output_path: Path
    episode_count: int
    step_count: int = 0


class DatasetExporter(Protocol):
    format_name: str

    def export_capture(self, *, raw_capture_dir: Path, output_dir: Path) -> ExportSummary:
        ...
