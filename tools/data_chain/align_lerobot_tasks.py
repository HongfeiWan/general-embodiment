#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Align LeRobot v2 task labels to a mission-level task description."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _mission_task(mission_dir: Path) -> str:
    mission_path = mission_dir / "MISSION.md"
    for line in mission_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- Mission:"):
            task = stripped.split(":", 1)[1].strip()
            if task:
                return task
    raise RuntimeError(f"Could not find '- Mission:' in {mission_path}")


def _find_dataset_roots(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted(
        path.parent.parent
        for path in root.rglob("meta/info.json")
        if (path.parent / "tasks.jsonl").is_file()
    )


def _align_dataset(dataset: Path, task: str, *, dry_run: bool) -> dict[str, int | str]:
    pyarrow_parquet = __import__("pyarrow.parquet", fromlist=["parquet"])
    tasks_path = dataset / "meta" / "tasks.jsonl"
    episodes_path = dataset / "meta" / "episodes.jsonl"
    tasks = _read_jsonl(tasks_path)
    non_validity_tasks = [row for row in tasks if row.get("task") not in {"valid", "invalid"}]
    target_task_index = int(non_validity_tasks[0].get("task_index", 0)) if non_validity_tasks else 0
    aligned_tasks = [{"task_index": target_task_index, "task": task}]
    next_index = max(target_task_index + 1, 1)
    for label in ("valid", "invalid"):
        existing = next((row for row in tasks if row.get("task") == label), None)
        task_index = int(existing["task_index"]) if existing is not None else next_index
        aligned_tasks.append({"task_index": task_index, "task": label})
        next_index = max(next_index, task_index + 1)

    episodes = _read_jsonl(episodes_path)
    for row in episodes:
        row["tasks"] = [task]

    parquet_updates = 0
    for parquet_path in sorted((dataset / "data").rglob("*.parquet")):
        table = pyarrow_parquet.read_table(parquet_path)
        data = table.to_pydict()
        changed = False
        row_count = table.num_rows
        for column in ("task_index", "annotation.human.action.task_description"):
            if column in data and set(data[column]) != {target_task_index}:
                data[column] = [target_task_index] * row_count
                changed = True
        if changed:
            parquet_updates += 1
            if not dry_run:
                pyarrow = __import__("pyarrow")
                pyarrow_parquet.write_table(pyarrow.table(data), parquet_path)

    if not dry_run:
        _write_jsonl(tasks_path, aligned_tasks)
        _write_jsonl(episodes_path, episodes)

    return {
        "dataset": str(dataset),
        "episodes": len(episodes),
        "parquet_updates": parquet_updates,
        "target_task_index": target_task_index,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mission_dir", type=Path, help="Mission directory containing MISSION.md.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="LeRobot v2 dataset root or parent. Defaults to <mission_dir>/lerobot_v2.",
    )
    parser.add_argument("--task", default=None, help="Override task label instead of reading MISSION.md.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    mission_dir = args.mission_dir.expanduser().resolve()
    dataset_root = (args.dataset_root or (mission_dir / "lerobot_v2")).expanduser().resolve()
    task = str(args.task or "").strip() or _mission_task(mission_dir)
    for dataset in _find_dataset_roots(dataset_root):
        result = _align_dataset(dataset, task, dry_run=bool(args.dry_run))
        print(
            f"[align-tasks] dataset={result['dataset']} episodes={result['episodes']} "
            f"parquet_updates={result['parquet_updates']} task_index={result['target_task_index']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
