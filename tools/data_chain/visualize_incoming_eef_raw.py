#!/usr/bin/env python3
"""Visualize raw incoming EEF state/action 9D curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CAPTURE = REPO_ROOT / "incoming" / "raw_packages" / "extracted" / "nero_l10_20260701T021817Z"
DEFAULT_OUTPUT = REPO_ROOT / "incoming" / "raw_packages" / "nero_l10_20260701T021817Z_eef_9d_alignment.html"

DIM_LABELS = (
    "x: state.x vs action.z",
    "y: state.y vs action.x",
    "z: state.z vs action.y",
    "rot6d.r00: state.r00 vs action.r22",
    "rot6d.r01: state.r01 vs -action.r20",
    "rot6d.r02: state.r02 vs -action.r21",
    "rot6d.r10: state.r10 vs action.r02",
    "rot6d.r11: state.r11 vs -action.r00",
    "rot6d.r12: state.r12 vs -action.r01",
)

ROT6D_ROW_LABELS = ("r00", "r01", "r02", "r10", "r11", "r12")
ACTION_TO_STATE_LEFT_MATRIX = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
ACTION_TO_STATE_RIGHT_MATRIX = (
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
)
ACTION_ROT6D_MAPPING_TEXT = "R_state ~= L @ R_action @ R; L rows=[z,x,y], R rows=[-y,-z,x]; rot6d=[r22,-r20,-r21,r02,-r00,-r01]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-capture", type=Path, default=DEFAULT_RAW_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action-source", choices=("safe_action", "commanded_action", "raw_command"), default="safe_action")
    parser.add_argument("--max-points", type=int, default=420)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def float_list(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    out: list[float] = []
    for item in value:
        parsed = finite_float(item)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def quat_xyzw_to_matrix(quat: list[float]) -> list[list[float]] | None:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return None
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]


def matrix_to_rot6d_rows(matrix: list[list[float]]) -> list[float]:
    return [*matrix[0], *matrix[1]]


def quat_xyzw_to_rot6d_rows(quat: list[float]) -> list[float] | None:
    matrix = quat_xyzw_to_matrix(quat)
    if matrix is None:
        return None
    return matrix_to_rot6d_rows(matrix)


def matmul3(left: tuple[tuple[float, float, float], ...] | list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            left[row][0] * right[0][col] + left[row][1] * right[1][col] + left[row][2] * right[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def action_matrix_to_state_matrix(matrix: list[list[float]]) -> list[list[float]]:
    left_applied = matmul3(ACTION_TO_STATE_LEFT_MATRIX, matrix)
    return matmul3(left_applied, [list(row) for row in ACTION_TO_STATE_RIGHT_MATRIX])


def eef9d_state(robot_record: dict[str, Any]) -> list[float] | None:
    pose = robot_record.get("arm_ee_pose")
    if not isinstance(pose, dict):
        return None
    pos = float_list(pose.get("position_xyz"), 3)
    quat = float_list(pose.get("quaternion_xyzw"), 4)
    if pos is None or quat is None:
        return None
    matrix = quat_xyzw_to_matrix(quat)
    if matrix is None:
        return None
    rot6d = matrix_to_rot6d_rows(matrix)
    return [*pos, *rot6d]


def eef9d_action(action_record: dict[str, Any], action_source: str) -> list[float] | None:
    selected = action_record.get(action_source)
    if not isinstance(selected, dict):
        return None
    ee_target = selected.get("ee_target")
    if not isinstance(ee_target, dict):
        return None
    pos = float_list(ee_target.get("position_xyz"), 3)
    quat = float_list(ee_target.get("quaternion_xyzw"), 4)
    if pos is None or quat is None:
        return None
    matrix = quat_xyzw_to_matrix(quat)
    if matrix is None:
        return None
    mapped_matrix = action_matrix_to_state_matrix(matrix)
    action_zxy = [pos[2], pos[0], pos[1]]
    return [*action_zxy, *matrix_to_rot6d_rows(mapped_matrix)]


def compact_series(values: list[float], max_points: int) -> list[float]:
    if max_points <= 0 or len(values) <= max_points:
        return values
    if max_points == 1:
        return [values[0]]
    result: list[float] = []
    last = len(values) - 1
    for index in range(max_points):
        src = round(index * last / (max_points - 1))
        result.append(values[int(src)])
    return result


def rmse(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n == 0:
        return None
    total = 0.0
    count = 0
    for idx in range(n):
        av = a[idx]
        bv = b[idx]
        if math.isfinite(av) and math.isfinite(bv):
            total += (av - bv) ** 2
            count += 1
    return math.sqrt(total / count) if count else None


def episode_payload(episode_dir: Path, action_source: str, max_points: int) -> dict[str, Any]:
    robots = read_jsonl(episode_dir / "robot.jsonl")
    actions = read_jsonl(episode_dir / "actions.jsonl")
    state_dims: list[list[float]] = [[] for _ in range(9)]
    action_dims: list[list[float]] = [[] for _ in range(9)]
    timestamps: list[float] = []
    skipped = 0
    start_ts: float | None = None
    for robot_record, action_record in zip(robots, actions):
        state = eef9d_state(robot_record)
        action = eef9d_action(action_record, action_source)
        if state is None or action is None:
            skipped += 1
            continue
        timestamp = finite_float(action_record.get("source_ts_s"))
        if timestamp is None:
            timestamp = finite_float(action_record.get("monotonic_ts_s"))
        if timestamp is None:
            timestamp = float(len(timestamps))
        if start_ts is None:
            start_ts = timestamp
        timestamps.append(timestamp - start_ts)
        for dim in range(9):
            state_dims[dim].append(state[dim])
            action_dims[dim].append(action[dim])

    compact_state = [compact_series(values, max_points) for values in state_dims]
    compact_action = [compact_series(values, max_points) for values in action_dims]
    return {
        "episode_id": episode_dir.name,
        "frames": len(timestamps),
        "raw_robot_records": len(robots),
        "raw_action_records": len(actions),
        "skipped": skipped,
        "timestamp": compact_series(timestamps, max_points),
        "state": compact_state,
        "action": compact_action,
        "rmse": [rmse(state_dims[dim], action_dims[dim]) for dim in range(9)],
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Incoming Raw EEF 9D State / Action</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; background: #fbfcfd; }}
    h1 {{ margin-bottom: 6px; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
    select {{ padding: 6px 8px; min-width: 440px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 16px 0; }}
    .pill {{ border: 1px solid #d7dde5; border-radius: 6px; background: #fff; padding: 6px 10px; }}
    .legend {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin: 8px 0 14px; font-size: 13px; }}
    .swatch {{ display: inline-block; width: 28px; height: 0; border-top: 3px solid currentColor; vertical-align: middle; margin-right: 5px; }}
    .swatch.dashed {{ border-top-style: dashed; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 12px; align-items: start; }}
    .card {{ border: 1px solid #d7dde5; border-radius: 6px; padding: 8px; background: #fff; }}
    canvas {{ width: 100%; height: 190px; display: block; }}
    .note {{ color: #667085; max-width: 1100px; line-height: 1.45; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(2, minmax(260px, 1fr)); }} }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} select {{ min-width: 0; width: 100%; }} }}
  </style>
</head>
<body>
  <h1>Incoming Raw EEF 9D State / Action</h1>
  <p class="note">
    Raw capture: <code id="raw-path"></code>. Action position is displayed as <code>[z, x, y]</code>.
    State Rot6D is displayed as GR00T row-major <code>[r00, r01, r02, r10, r11, r12]</code>.
    Action Rot6D is first converted to row-major, then remapped by the incoming raw correlation/RMSE fit as
    <code id="rot6d-map"></code>.
    Each chart uses independent y axes: state on the left, action on the right.
  </p>
  <div class="controls">
    <label for="episode-select">Episode</label>
    <select id="episode-select"></select>
    <span class="pill" id="episode-summary"></span>
  </div>
  <div class="legend">
    <span style="color:#2563eb"><span class="swatch"></span>state left y-axis</span>
    <span style="color:#dc2626"><span class="swatch dashed"></span>action right y-axis</span>
  </div>
  <div class="grid" id="charts"></div>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    document.getElementById('raw-path').textContent = payload.raw_capture || '';
    document.getElementById('rot6d-map').textContent = payload.action_rot6d_mapping || '';
    const labels = payload.dim_labels || [];
    const episodes = payload.episodes || [];
    const select = document.getElementById('episode-select');
    episodes.forEach((episode, index) => {{
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${{episode.episode_id}} | frames=${{episode.frames}} | skipped=${{episode.skipped}}`;
      select.appendChild(option);
    }});
    function finiteValues(arr) {{
      return (arr || []).map(Number).filter(Number.isFinite);
    }}
    function fmt(value) {{
      const num = Number(value);
      if (!Number.isFinite(num)) return '';
      if (Math.abs(num) >= 1000 || (Math.abs(num) > 0 && Math.abs(num) < 0.001)) return num.toExponential(3);
      return num.toFixed(5).replace(/0+$/, '').replace(/\\.$/, '');
    }}
    function drawPanel(canvas, label, t, stateArr, actionArr, rmse) {{
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = '13px system-ui';
      const stateFinite = finiteValues(stateArr);
      const actionFinite = finiteValues(actionArr);
      if (!stateFinite.length && !actionFinite.length) {{
        ctx.fillStyle = '#667085';
        ctx.fillText(`${{label}} | no finite values`, 12, 22);
        return;
      }}
      const stateMin = stateFinite.length ? Math.min(...stateFinite) : 0;
      const stateMax = stateFinite.length ? Math.max(...stateFinite) : 1;
      const actionMin = actionFinite.length ? Math.min(...actionFinite) : 0;
      const actionMax = actionFinite.length ? Math.max(...actionFinite) : 1;
      const stateSpan = Math.max(1e-9, stateMax - stateMin);
      const actionSpan = Math.max(1e-9, actionMax - actionMin);
      const left = 52, right = 52, top = 32, bottom = 28;
      const plotRight = canvas.width - right;
      const plotBottom = canvas.height - bottom;
      ctx.strokeStyle = '#d7dde5'; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, plotBottom);
      ctx.lineTo(plotRight, plotBottom);
      ctx.lineTo(plotRight, top);
      ctx.stroke();
      ctx.fillStyle = '#1f2933';
      ctx.font = '13px system-ui';
      ctx.textAlign = 'left';
      ctx.fillText(`${{label}} | RMSE=${{fmt(rmse)}}`, 12, 20);
      ctx.font = '11px system-ui';
      ctx.fillStyle = '#2563eb';
      ctx.fillText(fmt(stateMax), 6, top + 4);
      ctx.fillText(fmt(stateMin), 6, plotBottom);
      ctx.fillStyle = '#dc2626';
      ctx.textAlign = 'right';
      ctx.fillText(fmt(actionMax), canvas.width - 6, top + 4);
      ctx.fillText(fmt(actionMin), canvas.width - 6, plotBottom);
      ctx.textAlign = 'left';
      function drawLine(arr, color, dash, min, span) {{
        if (!arr || !arr.length) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.7;
        ctx.setLineDash(dash);
        let started = false;
        arr.forEach((raw, i) => {{
          const value = Number(raw);
          if (!Number.isFinite(value)) return;
          const x = left + i * (plotRight - left) / Math.max(1, arr.length - 1);
          const y = plotBottom - ((value - min) / span) * (plotBottom - top);
          if (!started) {{ ctx.moveTo(x, y); started = true; }} else {{ ctx.lineTo(x, y); }}
        }});
        ctx.stroke();
        ctx.setLineDash([]);
      }}
      drawLine(stateArr, '#2563eb', [], stateMin, stateSpan);
      drawLine(actionArr, '#dc2626', [8, 6], actionMin, actionSpan);
    }}
    function showEpisode() {{
      const episode = episodes[Number(select.value || 0)];
      const charts = document.getElementById('charts');
      charts.innerHTML = '';
      if (!episode) return;
      document.getElementById('episode-summary').textContent =
        `records robot=${{episode.raw_robot_records}} action=${{episode.raw_action_records}} plotted=${{episode.frames}}`;
      for (let dim = 0; dim < 9; dim++) {{
        const card = document.createElement('div');
        card.className = 'card';
        const canvas = document.createElement('canvas');
        canvas.width = 560;
        canvas.height = 220;
        card.appendChild(canvas);
        charts.appendChild(card);
        drawPanel(canvas, labels[dim] || `dim ${{dim}}`, episode.timestamp, episode.state[dim], episode.action[dim], episode.rmse[dim]);
      }}
    }}
    select.addEventListener('change', showEpisode);
    if (episodes.length) showEpisode();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_capture = args.raw_capture.expanduser().resolve()
    output = args.output.expanduser().resolve()
    episodes_root = raw_capture / "episodes"
    if not episodes_root.is_dir():
        raise RuntimeError(f"Missing episodes directory: {episodes_root}")
    episodes = [
        episode_payload(path, args.action_source, max(1, int(args.max_points)))
        for path in sorted(episodes_root.iterdir())
        if path.is_dir()
    ]
    payload = {
        "raw_capture": str(raw_capture),
        "action_source": args.action_source,
        "dim_labels": list(DIM_LABELS),
        "rot6d_row_labels": list(ROT6D_ROW_LABELS),
        "action_rot6d_mapping": ACTION_ROT6D_MAPPING_TEXT,
        "episodes": episodes,
    }
    write_html(output, payload)
    print(json.dumps({"episodes": len(episodes), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
