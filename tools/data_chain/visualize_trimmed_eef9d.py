#!/usr/bin/env python3
"""Visualize LeRobot trimmed EEF 9D state/action curves."""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "missions" / "nero" / "mission2" / "trimmed"
DEFAULT_OUTPUT = REPO_ROOT / "missions" / "nero" / "mission2" / "quality" / "trimmed_eef9d_viewer.html"

DIM_LABELS = (
    "x",
    "y",
    "z",
    "rot6d.r00",
    "rot6d.r01",
    "rot6d.r02",
    "rot6d.r10",
    "rot6d.r11",
    "rot6d.r12",
)

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trimmed LeRobot EEF 9D State / Action</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; background: #fbfcfd; }
    h1 { margin-bottom: 6px; font-size: 26px; line-height: 1.2; }
    code { background: #eef2f7; padding: 2px 4px; border-radius: 4px; }
    select { padding: 6px 8px; min-width: 520px; max-width: 100%; }
    .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 16px 0; }
    .pill { border: 1px solid #d7dde5; border-radius: 6px; background: #fff; padding: 6px 10px; }
    .legend { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin: 8px 0 14px; font-size: 13px; }
    .swatch { display: inline-block; width: 28px; height: 0; border-top: 3px solid currentColor; vertical-align: middle; margin-right: 5px; }
    .swatch.dashed { border-top-style: dashed; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 12px; align-items: start; }
    .card { border: 1px solid #d7dde5; border-radius: 6px; padding: 8px; background: #fff; }
    canvas { width: 100%; height: 190px; display: block; }
    .note { color: #667085; max-width: 1200px; line-height: 1.45; }
    @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, minmax(260px, 1fr)); } }
    @media (max-width: 760px) { body { margin: 14px; } .grid { grid-template-columns: 1fr; } select { min-width: 0; width: 100%; } }
  </style>
</head>
<body>
  <h1>Trimmed LeRobot EEF 9D State / Action</h1>
  <p class="note">
    Dataset: <code id="dataset-path"></code>.
    State and action are read directly from the repaired trimmed parquet files.
    Action position transform: <code id="position-transform"></code>.
    Rot6D convention: <code id="rot6d-convention"></code>.
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
  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    document.getElementById('dataset-path').textContent = payload.dataset_dir || '';
    document.getElementById('position-transform').textContent = payload.position_transform || 'as stored';
    document.getElementById('rot6d-convention').textContent = payload.rot6d_convention || 'unknown';
    const labels = payload.dim_labels || [];
    const episodes = payload.episodes || [];
    const select = document.getElementById('episode-select');

    episodes.forEach((episode, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      const rmse = Number(episode.rmse_mean);
      const rmseText = Number.isFinite(rmse) ? ` | meanRMSE=${fmt(rmse)}` : '';
      option.textContent = `${episode.episode_id} | frames=${episode.frames}${rmseText}`;
      select.appendChild(option);
    });

    function finiteValues(arr) {
      return (arr || []).map(Number).filter(Number.isFinite);
    }

    function fmt(value) {
      const num = Number(value);
      if (!Number.isFinite(num)) return '';
      if (Math.abs(num) >= 1000 || (Math.abs(num) > 0 && Math.abs(num) < 0.001)) return num.toExponential(3);
      return num.toFixed(5).replace(/0+$/, '').replace(/\.$/, '');
    }

    function paddedRange(values) {
      if (!values.length) return [0, 1];
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (Math.abs(max - min) < 1e-9) {
        const pad = Math.max(1e-4, Math.abs(max) * 0.05);
        min -= pad;
        max += pad;
      }
      return [min, max];
    }

    function drawPanel(canvas, label, stateArr, actionArr, rmse) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = '13px system-ui';
      const stateFinite = finiteValues(stateArr);
      const actionFinite = finiteValues(actionArr);
      if (!stateFinite.length && !actionFinite.length) {
        ctx.fillStyle = '#667085';
        ctx.fillText(`${label} | no finite values`, 12, 22);
        return;
      }

      const [stateMin, stateMax] = paddedRange(stateFinite);
      const [actionMin, actionMax] = paddedRange(actionFinite);
      const stateSpan = Math.max(1e-12, stateMax - stateMin);
      const actionSpan = Math.max(1e-12, actionMax - actionMin);
      const left = 52, right = 52, top = 32, bottom = 28;
      const plotRight = canvas.width - right;
      const plotBottom = canvas.height - bottom;

      ctx.strokeStyle = '#d7dde5';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, plotBottom);
      ctx.lineTo(plotRight, plotBottom);
      ctx.lineTo(plotRight, top);
      ctx.stroke();

      ctx.fillStyle = '#1f2933';
      ctx.font = '13px system-ui';
      ctx.textAlign = 'left';
      ctx.fillText(`${label} | RMSE=${fmt(rmse)}`, 12, 20);

      ctx.font = '11px system-ui';
      ctx.fillStyle = '#2563eb';
      ctx.fillText(fmt(stateMax), 6, top + 4);
      ctx.fillText(fmt(stateMin), 6, plotBottom);
      ctx.fillStyle = '#dc2626';
      ctx.textAlign = 'right';
      ctx.fillText(fmt(actionMax), canvas.width - 6, top + 4);
      ctx.fillText(fmt(actionMin), canvas.width - 6, plotBottom);
      ctx.textAlign = 'left';

      function drawLine(arr, color, dash, min, span) {
        if (!arr || !arr.length) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.7;
        ctx.setLineDash(dash);
        let started = false;
        arr.forEach((raw, i) => {
          const value = Number(raw);
          if (!Number.isFinite(value)) return;
          const x = left + i * (plotRight - left) / Math.max(1, arr.length - 1);
          const y = plotBottom - ((value - min) / span) * (plotBottom - top);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }

      drawLine(stateArr, '#2563eb', [], stateMin, stateSpan);
      drawLine(actionArr, '#dc2626', [8, 6], actionMin, actionSpan);
    }

    function showEpisode() {
      const episode = episodes[Number(select.value || 0)];
      const charts = document.getElementById('charts');
      charts.innerHTML = '';
      if (!episode) return;
      const task = episode.task ? ` | ${episode.task}` : '';
      document.getElementById('episode-summary').textContent =
        `parquet=${episode.data_path} | plotted=${episode.plotted_frames}/${episode.frames} | meanRMSE=${fmt(episode.rmse_mean)}${task}`;
      for (let dim = 0; dim < 9; dim++) {
        const card = document.createElement('div');
        card.className = 'card';
        const canvas = document.createElement('canvas');
        canvas.width = 560;
        canvas.height = 220;
        card.appendChild(canvas);
        charts.appendChild(card);
        drawPanel(canvas, labels[dim] || `dim ${dim}`, episode.state[dim], episode.action[dim], episode.rmse[dim]);
      }
    }

    select.addEventListener('change', showEpisode);
    if (episodes.length) showEpisode();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-points", type=int, default=800)
    parser.add_argument("--limit", type=int, default=0, help="Only include the first N episodes, for quick checks.")
    parser.add_argument("--serve", action="store_true", help="Serve the generated HTML with a local HTTP server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host used by --serve. Use 0.0.0.0 for LAN access.")
    parser.add_argument("--port", type=int, default=0, help="Port used by --serve. Use 0 for a random free port.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = info.get("features", {}).get(key, {})
    names = feature.get("names")
    return names if isinstance(names, list) else []


def slice_from_modality(modality: dict[str, Any], section: str, key: str) -> tuple[int, int] | None:
    value = modality.get(section, {}).get(key)
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return start, end
    return None


def indices_from_names(names: list[str], expected: tuple[str, ...]) -> list[int] | None:
    lookup = {name: idx for idx, name in enumerate(names)}
    indices: list[int] = []
    for name in expected:
        if name not in lookup:
            return None
        indices.append(lookup[name])
    return indices


def eef_indices(info: dict[str, Any], modality: dict[str, Any]) -> tuple[list[int], list[int]]:
    state_slice = slice_from_modality(modality, "state", "eef_9d")
    action_slice = slice_from_modality(modality, "action", "eef_9d")
    if state_slice is not None and action_slice is not None:
        state_start, state_end = state_slice
        action_start, action_end = action_slice
        if state_end - state_start == 9 and action_end - action_start == 9:
            return list(range(state_start, state_end)), list(range(action_start, action_end))

    state_expected = (
        "arm_eef_pos.x",
        "arm_eef_pos.y",
        "arm_eef_pos.z",
        "arm_eef_rot6d.r00",
        "arm_eef_rot6d.r01",
        "arm_eef_rot6d.r02",
        "arm_eef_rot6d.r10",
        "arm_eef_rot6d.r11",
        "arm_eef_rot6d.r12",
    )
    action_expected = (
        "arm_eef_pos_target.x",
        "arm_eef_pos_target.y",
        "arm_eef_pos_target.z",
        "arm_eef_rot6d_target.r00",
        "arm_eef_rot6d_target.r01",
        "arm_eef_rot6d_target.r02",
        "arm_eef_rot6d_target.r10",
        "arm_eef_rot6d_target.r11",
        "arm_eef_rot6d_target.r12",
    )
    state_indices = indices_from_names(feature_names(info, "observation.state"), state_expected)
    action_indices = indices_from_names(feature_names(info, "action"), action_expected)
    if state_indices is None or action_indices is None:
        raise RuntimeError("Could not resolve EEF 9D indices from modality.json or feature names.")
    return state_indices, action_indices


def data_path_for_episode(dataset_dir: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    metadata = episode.get("teleop_stack_metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("data_path"), str):
        path = dataset_dir / metadata["data_path"]
        if path.exists():
            return path

    episode_index = int(episode["episode_index"])
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    path = dataset_dir / pattern.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)
    if path.exists():
        return path

    matches = sorted(dataset_dir.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing parquet for episode {episode_index}: {path}")


def stack_vector_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise KeyError(f"Missing column {column!r}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in df[column].to_numpy()]
    if not values:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(values)


def sample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= 0:
        return np.empty((0,), dtype=np.int64)
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    if max_points == 1:
        return np.array([0], dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, max_points)).astype(np.int64)


def rmse_series(state: np.ndarray, action: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    for dim in range(9):
        left = state[:, dim]
        right = action[:, dim]
        mask = np.isfinite(left) & np.isfinite(right)
        if not np.any(mask):
            values.append(None)
            continue
        diff = left[mask] - right[mask]
        values.append(float(np.sqrt(np.mean(diff * diff))))
    return values


def compact_matrix(matrix: np.ndarray, indices: np.ndarray) -> list[list[float | None]]:
    result: list[list[float | None]] = []
    compact = matrix[indices, :] if len(indices) else np.empty((0, matrix.shape[1]), dtype=np.float64)
    for dim in range(compact.shape[1]):
        values: list[float | None] = []
        for raw in compact[:, dim]:
            parsed = finite_float(raw)
            values.append(parsed)
        result.append(values)
    return result


def compact_list(values: np.ndarray, indices: np.ndarray) -> list[float | None]:
    result: list[float | None] = []
    for raw in values[indices] if len(indices) else []:
        result.append(finite_float(raw))
    return result


def task_text(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return ""


def episode_payload(
    dataset_dir: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    state_indices: list[int],
    action_indices: list[int],
    max_points: int,
) -> dict[str, Any]:
    data_path = data_path_for_episode(dataset_dir, info, episode)
    df = pd.read_parquet(data_path, columns=["timestamp", "frame_index", "observation.state", "action"])
    states = stack_vector_column(df, "observation.state")[:, state_indices]
    actions = stack_vector_column(df, "action")[:, action_indices]
    if states.shape[1] != 9 or actions.shape[1] != 9:
        raise RuntimeError(f"{data_path} did not resolve to 9D EEF state/action arrays.")

    frames = int(min(len(states), len(actions)))
    states = states[:frames]
    actions = actions[:frames]
    indices = sample_indices(frames, max(1, int(max_points)))

    if "timestamp" in df.columns:
        timestamp = np.asarray(df["timestamp"].to_numpy()[:frames], dtype=np.float64)
    else:
        timestamp = np.arange(frames, dtype=np.float64)
    frame_index = np.asarray(df["frame_index"].to_numpy()[:frames], dtype=np.float64) if "frame_index" in df.columns else np.arange(frames)
    rmses = rmse_series(states, actions)
    rmse_finite = [value for value in rmses if value is not None and math.isfinite(value)]
    episode_index = int(episode["episode_index"])
    return {
        "episode_id": f"episode_{episode_index:06d}",
        "episode_index": episode_index,
        "data_path": str(data_path.relative_to(dataset_dir)),
        "frames": frames,
        "plotted_frames": int(len(indices)),
        "task": task_text(episode),
        "timestamp": compact_list(timestamp, indices),
        "frame_index": compact_list(frame_index, indices),
        "state": compact_matrix(states, indices),
        "action": compact_matrix(actions, indices),
        "rmse": rmses,
        "rmse_mean": float(np.mean(rmse_finite)) if rmse_finite else None,
    }


def build_payload(dataset_dir: Path, max_points: int, limit: int) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    modality_path = dataset_dir / "meta" / "modality.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not info_path.exists() or not modality_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"{dataset_dir} does not look like a LeRobot dataset with meta/info.json, modality.json, episodes.jsonl.")

    info = read_json(info_path)
    modality = read_json(modality_path)
    episodes_meta = read_jsonl(episodes_path)
    episodes_meta.sort(key=lambda row: int(row.get("episode_index", 0)))
    if limit > 0:
        episodes_meta = episodes_meta[:limit]
    state_indices, action_indices = eef_indices(info, modality)

    episodes = [
        episode_payload(dataset_dir, info, episode, state_indices, action_indices, max_points)
        for episode in episodes_meta
    ]
    teleop_stack = info.get("teleop_stack", {})
    return {
        "dataset_dir": str(dataset_dir),
        "dim_labels": list(DIM_LABELS),
        "state_indices": state_indices,
        "action_indices": action_indices,
        "rot6d_convention": teleop_stack.get("rot6d_convention", ""),
        "position_transform": (teleop_stack.get("action_position_frame_transform") or {}).get("formula", ""),
        "repair": teleop_stack.get("eef9d_repair", {}),
        "xyz_repair": teleop_stack.get("eef_xyz_repair", {}),
        "episodes": episodes,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_doc = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def local_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address and not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    try:
        completed = subprocess.run(
            ["hostname", "-I"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        for token in completed.stdout.split():
            if "." in token and ":" not in token and not token.startswith("127."):
                addresses.add(token)
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(addresses)


def serve_html(output: Path, host: str, port: int) -> None:
    directory = output.parent.resolve()
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    actual_port = int(server.server_address[1])
    relative_url = output.name
    urls = [f"http://127.0.0.1:{actual_port}/{relative_url}"]
    if host in {"0.0.0.0", "::"}:
        urls.extend(f"http://{address}:{actual_port}/{relative_url}" for address in local_ipv4_addresses())
    else:
        urls.append(f"http://{host}:{actual_port}/{relative_url}")
    print("Serving trimmed EEF 9D viewer:", flush=True)
    for url in dict.fromkeys(urls):
        print(f"  {url}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = build_payload(dataset_dir, max_points=max(1, int(args.max_points)), limit=max(0, int(args.limit)))
    write_html(output, payload)
    summary = {
        "dataset_dir": str(dataset_dir),
        "episodes": len(payload["episodes"]),
        "output": str(output),
        "max_points": args.max_points,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.serve:
        serve_html(output, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
