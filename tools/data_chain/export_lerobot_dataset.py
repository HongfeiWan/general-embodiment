from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from teleop_stack.data_capture.exporters import (
    GrootLeRobotV2Exporter,
    GrootLeRobotV2ExporterConfig,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export raw captured trials to a GR00T-compatible LeRobot v2 dataset.",
    )
    parser.add_argument("raw_capture_dir", type=Path, help="Path to raw/<capture_id>.")
    parser.add_argument("output_dir", type=Path, help="Output dataset directory.")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera stream name used as observation.image. May be repeated for multi-view export.",
    )
    parser.add_argument(
        "--schema",
        choices=[
            "legacy",
            "rokae_xmate3_linker_l10_groot_v1",
            "rokae_xmate3_linker_l10_groot_v1_1_full_orientation",
        ],
        default="rokae_xmate3_linker_l10_groot_v1",
        help="Exporter schema used to build observation.state/action vectors.",
    )
    parser.add_argument(
        "--video-alias",
        action="append",
        default=None,
        help="Short alias recorded under meta/modality.json video.<alias>. May be repeated to match --camera.",
    )
    parser.add_argument(
        "--video-feature-key",
        action="append",
        default=None,
        help="Full LeRobot video feature key, for example observation.images.ego_view. May be repeated to match --camera.",
    )
    parser.add_argument(
        "--action-source",
        choices=["safe_action", "commanded_action"],
        default="safe_action",
        help="Which raw action field should become exporter action.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Export only episodes explicitly marked success=true.",
    )
    parser.add_argument(
        "--allow-missing-robot-state",
        action="store_true",
        help="Allow export even when robot observation.state is missing.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Target video FPS for exported mp4 files.",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=1000,
        help="Maximum number of episodes written into each chunk-XXX directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write parquet/mp4; only write a summary file.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    selected_cameras = tuple(args.camera or ["realsense_head"])
    video_aliases = tuple(args.video_alias) if args.video_alias is not None else None
    video_feature_keys = tuple(args.video_feature_key) if args.video_feature_key is not None else None
    exporter = GrootLeRobotV2Exporter(
        GrootLeRobotV2ExporterConfig(
            selected_camera=selected_cameras[0],
            selected_cameras=selected_cameras,
            schema=args.schema,
            video_alias=video_aliases[0] if video_aliases else "ego_view",
            video_aliases=video_aliases,
            video_feature_key=video_feature_keys[0] if video_feature_keys else None,
            video_feature_keys=video_feature_keys,
            action_source=args.action_source,
            success_only=bool(args.success_only),
            require_robot_state=not bool(args.allow_missing_robot_state),
            fps=max(1, int(args.fps)),
            episodes_per_chunk=max(1, int(args.episodes_per_chunk)),
            dry_run=bool(args.dry_run),
        )
    )
    summary = exporter.export_capture(
        raw_capture_dir=args.raw_capture_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(
        "[export-lerobot] "
        f"format={summary.format_name} output={summary.output_path} "
        f"episodes={summary.episode_count} steps={summary.step_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
