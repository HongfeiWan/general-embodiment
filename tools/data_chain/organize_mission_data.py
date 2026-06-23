#!/usr/bin/env python3
"""Organize mission data links and manifests by filesystem creation date."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_SOURCE_MISSION_DIR = Path("/home/whf/Project/Isaac-GR00T/demo_data/l10_hand/nero/mission2")
DEFAULT_TRIMMED_DATASET = Path("/home/whf/Project/Isaac-GR00T/outputs/IsaacLab/nero/mission2/trimmed")
DEFAULT_SMOOTH_DATASET = Path("/home/whf/Project/Isaac-GR00T/outputs/IsaacLab/nero/mission2/smooth")
DEFAULT_PREPARED_SMOOTH_DIR = Path(
    "/home/whf/Project/Isaac-GR00T/outputs/IsaacLab/nero/mission2/prepared_smooth"
)


@dataclass(frozen=True)
class LinkedItem:
    item_type: str
    name: str
    date: str
    source_path: Path
    link_path: Path
    created_at: str
    mode: str


def _creation_timestamp(path: Path) -> float:
    try:
        raw = subprocess.check_output(
            ["stat", "-c", "%W", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        timestamp = int(raw)
        if timestamp > 0:
            return float(timestamp)
    except Exception:
        pass
    stat = path.stat()
    return float(getattr(stat, "st_birthtime", 0.0) or stat.st_ctime or stat.st_mtime)


def _creation_datetime(path: Path) -> datetime:
    return datetime.fromtimestamp(_creation_timestamp(path)).astimezone()


def _is_lerobot_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _date_dir_for(path: Path) -> tuple[str, str]:
    created = _creation_datetime(path)
    return created.strftime("%Y-%m-%d"), created.isoformat()


def _replace_link_or_copy(source: Path, destination: Path, *, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    if mode == "copy":
        shutil.copytree(source, destination, symlinks=True)
    else:
        destination.symlink_to(source.resolve(), target_is_directory=True)


def _discover_raw_sessions(source_mission_dir: Path) -> list[Path]:
    raw_root = source_mission_dir / "data_capture" / "raw"
    if not raw_root.exists():
        return []
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def _discover_lerobot_datasets(source_mission_dir: Path) -> list[Path]:
    datasets: list[Path] = []
    for path in source_mission_dir.iterdir():
        if path.is_dir() and _is_lerobot_dataset(path):
            datasets.append(path)
    return sorted(datasets)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_manifest(mission_dir: Path, items: list[LinkedItem]) -> None:
    new_rows = [
        {
            "item_type": item.item_type,
            "name": item.name,
            "date": item.date,
            "created_at": item.created_at,
            "source_path": str(item.source_path),
            "workspace_path": str(item.link_path),
            "mode": item.mode,
        }
        for item in sorted(items, key=lambda item: (item.item_type, item.date, item.name))
    ]
    replaced_types = {str(row["item_type"]) for row in new_rows}
    old_rows = [
        row
        for row in _read_jsonl(mission_dir / "manifest.jsonl")
        if str(row.get("item_type")) not in replaced_types
    ]
    rows = sorted(
        old_rows + new_rows,
        key=lambda row: (
            str(row.get("item_type", "")),
            str(row.get("date", "")),
            str(row.get("name", "")),
        ),
    )
    _write_jsonl(mission_dir / "manifest.jsonl", rows)

    by_type: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_type.setdefault(str(row["item_type"]), []).append(row)
    for item_type, type_rows in by_type.items():
        if item_type in {"trimmed", "smooth", "prepared_smooth"}:
            continue
        type_dir = mission_dir / item_type
        if type_dir.is_symlink():
            continue
        _write_jsonl(type_dir / "manifest.jsonl", type_rows)


def _link_group(
    *,
    source_paths: list[Path],
    item_type: str,
    destination_root: Path,
    mode: str,
) -> list[LinkedItem]:
    linked: list[LinkedItem] = []
    for source in source_paths:
        date, created_at = _date_dir_for(source)
        destination = destination_root / date / source.name
        _replace_link_or_copy(source, destination, mode=mode)
        linked.append(
            LinkedItem(
                item_type=item_type,
                name=source.name,
                date=date,
                source_path=source.resolve(),
                link_path=destination,
                created_at=created_at,
                mode=mode,
            )
        )
    return linked


def _link_trimmed(trimmed_dataset: Path, mission_dir: Path, *, mode: str) -> list[LinkedItem]:
    if not trimmed_dataset.exists():
        return []
    date, created_at = _date_dir_for(trimmed_dataset)
    destination = mission_dir / "trimmed"
    _replace_link_or_copy(trimmed_dataset, destination, mode=mode)
    date_index = mission_dir / "trimmed_by_date" / date / trimmed_dataset.name
    _replace_link_or_copy(destination, date_index, mode="symlink")
    return [
        LinkedItem(
            item_type="trimmed",
            name=trimmed_dataset.name,
            date=date,
            source_path=trimmed_dataset.resolve(),
            link_path=destination,
            created_at=created_at,
            mode=mode,
        )
    ]


def _link_named_dataset(
    *,
    source: Path,
    mission_dir: Path,
    item_type: str,
    destination_name: str,
    date_index_name: str,
    mode: str,
) -> list[LinkedItem]:
    if not source.exists():
        return []
    date, created_at = _date_dir_for(source)
    destination = mission_dir / destination_name
    _replace_link_or_copy(source, destination, mode=mode)
    date_index = mission_dir / date_index_name / date / source.name
    _replace_link_or_copy(destination, date_index, mode="symlink")
    return [
        LinkedItem(
            item_type=item_type,
            name=source.name,
            date=date,
            source_path=source.resolve(),
            link_path=destination,
            created_at=created_at,
            mode=mode,
        )
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-dir", type=Path, default=DEFAULT_MISSION_DIR)
    parser.add_argument("--source-mission-dir", type=Path, default=DEFAULT_SOURCE_MISSION_DIR)
    parser.add_argument("--trimmed-dataset", type=Path, default=DEFAULT_TRIMMED_DATASET)
    parser.add_argument("--smooth-dataset", type=Path, default=DEFAULT_SMOOTH_DATASET)
    parser.add_argument("--prepared-smooth-dir", type=Path, default=DEFAULT_PREPARED_SMOOTH_DIR)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument(
        "--skip-trimmed",
        action="store_true",
        help="Do not link the existing trimmed LeRobot dataset.",
    )
    parser.add_argument("--skip-smooth", action="store_true", help="Do not link smooth outputs.")
    parser.add_argument("--skip-raw", action="store_true", help="Do not refresh raw session copies.")
    parser.add_argument("--skip-lerobot", action="store_true", help="Do not refresh LeRobot v2 copies.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mission_dir = args.mission_dir.expanduser().resolve()
    source_mission_dir = args.source_mission_dir.expanduser().resolve()
    trimmed_dataset = args.trimmed_dataset.expanduser().resolve()
    smooth_dataset = args.smooth_dataset.expanduser().resolve()
    prepared_smooth_dir = args.prepared_smooth_dir.expanduser().resolve()
    mission_dir.mkdir(parents=True, exist_ok=True)

    items: list[LinkedItem] = []
    if not args.skip_raw:
        items.extend(
            _link_group(
                source_paths=_discover_raw_sessions(source_mission_dir),
                item_type="raw",
                destination_root=mission_dir / "raw",
                mode=args.link_mode,
            )
        )
    if not args.skip_lerobot:
        items.extend(
            _link_group(
                source_paths=_discover_lerobot_datasets(source_mission_dir),
                item_type="lerobot_v2",
                destination_root=mission_dir / "lerobot_v2",
                mode=args.link_mode,
            )
        )
    if not args.skip_trimmed:
        items.extend(_link_trimmed(trimmed_dataset, mission_dir, mode=args.link_mode))
    if not args.skip_smooth:
        items.extend(
            _link_named_dataset(
                source=smooth_dataset,
                mission_dir=mission_dir,
                item_type="smooth",
                destination_name="smooth",
                date_index_name="smooth_by_date",
                mode=args.link_mode,
            )
        )
        items.extend(
            _link_named_dataset(
                source=prepared_smooth_dir,
                mission_dir=mission_dir,
                item_type="prepared_smooth",
                destination_name="prepared_smooth",
                date_index_name="prepared_smooth_by_date",
                mode=args.link_mode,
            )
        )

    _write_manifest(mission_dir, items)
    print(f"[organize] mission={mission_dir} items={len(items)} mode={args.link_mode}")
    print(f"[organize] manifest={mission_dir / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
