#!/usr/bin/env python3
"""Visualize before/after smoothing EEF 9D state/action curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIR = REPO_ROOT / "missions" / "nero" / "mission2"
DEFAULT_BEFORE_DATASET = MISSION_DIR / "trimmed"
DEFAULT_SMOOTH_DATASET = MISSION_DIR / "smooth"
DEFAULT_OUTPUT = MISSION_DIR / "batch_quality" / "mission2_smooth_all" / "batch_quality_report.html"
DEFAULT_QUALITY_SUMMARY = MISSION_DIR / "batch_quality" / "mission2_smooth_all" / "summary.json"
DEFAULT_SMOOTH_SUMMARY = MISSION_DIR / "smooth_action_summary.json"

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
  <title>Mission2 Smooth EEF 9D Quality</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1f2933; background: #fbfcfd; }
    h1 { margin-bottom: 6px; font-size: 26px; line-height: 1.2; }
    h2 { font-size: 16px; margin: 22px 0 10px; }
    code { background: #eef2f7; padding: 2px 4px; border-radius: 4px; }
    select { padding: 6px 8px; min-width: 520px; max-width: 100%; }
    label { font-size: 14px; }
    input[type="checkbox"] { vertical-align: middle; }
    .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin: 16px 0; }
    .pill { border: 1px solid #d7dde5; border-radius: 6px; background: #fff; padding: 6px 10px; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 16px 0; max-width: 1200px; }
    .summary-card { border: 1px solid #d7dde5; border-radius: 6px; padding: 10px 12px; background: #fff; }
    .summary-label { color: #667085; font-size: 12px; text-transform: uppercase; }
    .summary-value { font-size: 20px; margin-top: 4px; }
    .legend { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin: 8px 0 14px; font-size: 13px; }
    .swatch { display: inline-block; width: 28px; height: 0; border-top: 3px solid currentColor; vertical-align: middle; margin-right: 5px; }
    .swatch.thin { border-top-width: 2px; opacity: 0.45; }
    .swatch.dashed { border-top-style: dashed; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 12px; align-items: start; }
    .chart { border: 1px solid #d7dde5; border-radius: 6px; padding: 8px; background: #fff; }
    canvas { width: 100%; height: 230px; display: block; }
    .note { color: #667085; max-width: 1200px; line-height: 1.45; }
    .links { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
    .links code { color: #344054; }
    @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, minmax(260px, 1fr)); } }
    @media (max-width: 760px) { body { margin: 14px; } .grid { grid-template-columns: 1fr; } select { min-width: 0; width: 100%; } }
  </style>
</head>
<body>
  <h1>Mission2 Smooth EEF 9D Quality</h1>
  <p class="note">
    Before dataset: <code id="before-path"></code>.
    Smooth dataset: <code id="smooth-path"></code>.
    State and action are read from LeRobot parquet EEF 9D slices. State curves use the left y-axis, action curves use the right y-axis.
  </p>
  <div class="summary" id="summary"></div>
  <div class="controls">
    <label for="episode-select">Episode</label>
    <select id="episode-select"></select>
    <span class="pill" id="episode-summary"></span>
  </div>
  <div class="controls">
    <label><input id="show-before" type="checkbox" checked> before</label>
    <label><input id="show-smooth" type="checkbox" checked> smooth</label>
    <span class="pill" id="quality-status"></span>
  </div>
  <div class="legend">
    <span style="color:#2563eb"><span class="swatch thin"></span>state before</span>
    <span style="color:#059669"><span class="swatch"></span>state smooth</span>
    <span style="color:#dc2626"><span class="swatch dashed thin"></span>action before</span>
    <span style="color:#f97316"><span class="swatch dashed"></span>action smooth</span>
  </div>
  <div class="grid" id="charts"></div>
  <h2>Files</h2>
  <div class="links">
    <code>summary.json</code>
    <code>episode_checks.csv</code>
    <code>issue_rows.csv</code>
    <code>video_checks.csv</code>
    <code>episode_metrics.png</code>
  </div>
  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    document.getElementById('before-path').textContent = payload.before_dataset_dir || '';
    document.getElementById('smooth-path').textContent = payload.smooth_dataset_dir || '';
    const labels = payload.dim_labels || [];
    const episodes = payload.episodes || [];
    const select = document.getElementById('episode-select');
    const showBefore = document.getElementById('show-before');
    const showSmooth = document.getElementById('show-smooth');

    function fmt(value) {
      const num = Number(value);
      if (!Number.isFinite(num)) return '';
      if (Math.abs(num) >= 1000 || (Math.abs(num) > 0 && Math.abs(num) < 0.001)) return num.toExponential(3);
      return num.toFixed(5).replace(/0+$/, '').replace(/\.$/, '');
    }

    function addSummary(label, value) {
      const card = document.createElement('div');
      card.className = 'summary-card';
      const labelDiv = document.createElement('div');
      labelDiv.className = 'summary-label';
      labelDiv.textContent = label;
      const valueDiv = document.createElement('div');
      valueDiv.className = 'summary-value';
      valueDiv.textContent = value;
      card.appendChild(labelDiv);
      card.appendChild(valueDiv);
      document.getElementById('summary').appendChild(card);
    }

    addSummary('Episodes', String(payload.quality_summary?.episodes ?? episodes.length));
    addSummary('Frames', String(payload.quality_summary?.frames ?? payload.total_frames ?? ''));
    addSummary('Errors', String(payload.quality_summary?.errors ?? 0));
    addSummary('Warnings', String(payload.quality_summary?.warnings ?? 0));
    addSummary('Action mean change', fmt(payload.smooth_summary?.action_mean_abs_change));
    addSummary('State mean change', fmt(payload.smooth_summary?.state_mean_abs_change));
    document.getElementById('quality-status').textContent = `quality=${payload.quality_summary?.status || 'unknown'}`;

    episodes.forEach((episode, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `${episode.episode_id} | frames=${episode.frames} | AΔ=${fmt(episode.action_delta_rmse_mean)} | SΔ=${fmt(episode.state_delta_rmse_mean)}`;
      select.appendChild(option);
    });

    function finiteValues(arr) {
      return (arr || []).map(Number).filter(Number.isFinite);
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

    function drawLine(ctx, arr, color, dash, min, span, layout, width, alpha) {
      if (!arr || !arr.length) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      let started = false;
      arr.forEach((raw, i) => {
        const value = Number(raw);
        if (!Number.isFinite(value)) return;
        const x = layout.left + i * (layout.plotRight - layout.left) / Math.max(1, arr.length - 1);
        const y = layout.plotBottom - ((value - min) / span) * (layout.plotBottom - layout.top);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.restore();
    }

    function drawPanel(canvas, label, episode, dim) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = '13px system-ui';
      const stateBefore = episode.state_before[dim] || [];
      const stateSmooth = episode.state_smooth[dim] || [];
      const actionBefore = episode.action_before[dim] || [];
      const actionSmooth = episode.action_smooth[dim] || [];
      const visibleState = [];
      const visibleAction = [];
      if (showBefore.checked) {
        visibleState.push(...finiteValues(stateBefore));
        visibleAction.push(...finiteValues(actionBefore));
      }
      if (showSmooth.checked) {
        visibleState.push(...finiteValues(stateSmooth));
        visibleAction.push(...finiteValues(actionSmooth));
      }
      if (!visibleState.length && !visibleAction.length) {
        ctx.fillStyle = '#667085';
        ctx.fillText(`${label} | no finite values`, 12, 22);
        return;
      }

      const [stateMin, stateMax] = paddedRange(visibleState);
      const [actionMin, actionMax] = paddedRange(visibleAction);
      const stateSpan = Math.max(1e-12, stateMax - stateMin);
      const actionSpan = Math.max(1e-12, actionMax - actionMin);
      const layout = {left: 52, right: 52, top: 48, bottom: 28};
      layout.plotRight = canvas.width - layout.right;
      layout.plotBottom = canvas.height - layout.bottom;

      ctx.strokeStyle = '#d7dde5';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(layout.left, layout.top);
      ctx.lineTo(layout.left, layout.plotBottom);
      ctx.lineTo(layout.plotRight, layout.plotBottom);
      ctx.lineTo(layout.plotRight, layout.top);
      ctx.stroke();

      ctx.fillStyle = '#1f2933';
      ctx.font = '13px system-ui';
      ctx.textAlign = 'left';
      ctx.fillText(label, 12, 18);
      ctx.font = '11px system-ui';
      const sDelta = episode.state_delta_rmse[dim];
      const aDelta = episode.action_delta_rmse[dim];
      const saBefore = episode.state_action_rmse_before[dim];
      const saSmooth = episode.state_action_rmse_smooth[dim];
      ctx.fillText(`SΔ=${fmt(sDelta)} AΔ=${fmt(aDelta)} SA=${fmt(saBefore)}->${fmt(saSmooth)}`, 12, 34);

      ctx.font = '11px system-ui';
      ctx.fillStyle = '#2563eb';
      ctx.fillText(fmt(stateMax), 6, layout.top + 4);
      ctx.fillText(fmt(stateMin), 6, layout.plotBottom);
      ctx.fillStyle = '#dc2626';
      ctx.textAlign = 'right';
      ctx.fillText(fmt(actionMax), canvas.width - 6, layout.top + 4);
      ctx.fillText(fmt(actionMin), canvas.width - 6, layout.plotBottom);
      ctx.textAlign = 'left';

      if (showBefore.checked) {
        drawLine(ctx, stateBefore, '#2563eb', [], stateMin, stateSpan, layout, 1.5, 0.42);
        drawLine(ctx, actionBefore, '#dc2626', [7, 6], actionMin, actionSpan, layout, 1.5, 0.42);
      }
      if (showSmooth.checked) {
        drawLine(ctx, stateSmooth, '#059669', [], stateMin, stateSpan, layout, 2.0, 1.0);
        drawLine(ctx, actionSmooth, '#f97316', [8, 6], actionMin, actionSpan, layout, 2.0, 1.0);
      }
    }

    function showEpisode() {
      const episode = episodes[Number(select.value || 0)];
      const charts = document.getElementById('charts');
      charts.innerHTML = '';
      if (!episode) return;
      const task = episode.task ? ` | ${episode.task}` : '';
      document.getElementById('episode-summary').textContent =
        `parquet=${episode.data_path} | plotted=${episode.plotted_frames}/${episode.frames} | AΔ=${fmt(episode.action_delta_rmse_mean)} SΔ=${fmt(episode.state_delta_rmse_mean)}${task}`;
      for (let dim = 0; dim < 9; dim++) {
        const chart = document.createElement('div');
        chart.className = 'chart';
        const canvas = document.createElement('canvas');
        canvas.width = 560;
        canvas.height = 260;
        chart.appendChild(canvas);
        charts.appendChild(chart);
        drawPanel(canvas, labels[dim] || `dim ${dim}`, episode, dim);
      }
    }

    select.addEventListener('change', showEpisode);
    showBefore.addEventListener('change', showEpisode);
    showSmooth.addEventListener('change', showEpisode);
    if (episodes.length) showEpisode();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dataset", type=Path, default=DEFAULT_BEFORE_DATASET)
    parser.add_argument("--smooth-dataset", type=Path, default=DEFAULT_SMOOTH_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality-summary", type=Path, default=DEFAULT_QUALITY_SUMMARY)
    parser.add_argument("--smooth-summary", type=Path, default=DEFAULT_SMOOTH_SUMMARY)
    parser.add_argument("--max-points", type=int, default=800)
    parser.add_argument("--limit", type=int, default=0, help="Only include the first N episodes.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def feature_names(info: dict[str, Any], key: str) -> list[str]:
    feature = (info.get("features") or {}).get(key)
    names = feature.get("names") if isinstance(feature, dict) else None
    return [str(name) for name in names] if isinstance(names, list) else []


def slice_from_modality(modality: dict[str, Any], section: str, key: str) -> tuple[int, int] | None:
    value = (modality.get(section) or {}).get(key)
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
        path = dataset_dir / str(metadata["data_path"])
        if path.exists():
            return path

    episode_index = int(episode["episode_index"])
    chunks_size = max(1, int(info.get("chunks_size", 1000)))
    pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
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
    width = values[0].shape[0]
    if any(value.shape[0] != width for value in values):
        raise RuntimeError(f"Inconsistent vector width in {column}")
    return np.vstack(values)


def sample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= 0:
        return np.empty((0,), dtype=np.int64)
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    if max_points == 1:
        return np.array([0], dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, max_points)).astype(np.int64)


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return round(parsed, 6)


def compact_matrix(matrix: np.ndarray, indices: np.ndarray) -> list[list[float | None]]:
    compact = matrix[indices, :] if len(indices) else np.empty((0, matrix.shape[1]), dtype=np.float64)
    result: list[list[float | None]] = []
    for dim in range(compact.shape[1]):
        result.append([finite_float(value) for value in compact[:, dim]])
    return result


def compact_list(values: np.ndarray, indices: np.ndarray) -> list[float | None]:
    return [finite_float(value) for value in (values[indices] if len(indices) else [])]


def rmse_vector(left: np.ndarray, right: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    width = int(min(left.shape[1], right.shape[1]))
    for dim in range(width):
        a = left[:, dim]
        b = right[:, dim]
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            values.append(None)
            continue
        diff = a[mask] - b[mask]
        values.append(float(np.sqrt(np.mean(diff * diff))))
    return values


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def task_text(episode: dict[str, Any]) -> str:
    tasks = episode.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return ""


def load_eef9d(
    dataset_dir: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    state_indices: list[int],
    action_indices: list[int],
) -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    data_path = data_path_for_episode(dataset_dir, info, episode)
    df = pd.read_parquet(data_path, columns=["timestamp", "frame_index", "observation.state", "action"])
    state = stack_vector_column(df, "observation.state")[:, state_indices]
    action = stack_vector_column(df, "action")[:, action_indices]
    if state.shape[1] != 9 or action.shape[1] != 9:
        raise RuntimeError(f"{data_path} did not resolve to 9D EEF state/action arrays.")
    return data_path, df, state, action


def episode_payload(
    before_dataset: Path,
    smooth_dataset: Path,
    before_info: dict[str, Any],
    smooth_info: dict[str, Any],
    episode: dict[str, Any],
    before_indices: tuple[list[int], list[int]],
    smooth_indices: tuple[list[int], list[int]],
    max_points: int,
) -> dict[str, Any]:
    before_path, before_df, before_state, before_action = load_eef9d(
        before_dataset,
        before_info,
        episode,
        before_indices[0],
        before_indices[1],
    )
    smooth_path, smooth_df, smooth_state, smooth_action = load_eef9d(
        smooth_dataset,
        smooth_info,
        episode,
        smooth_indices[0],
        smooth_indices[1],
    )
    frames = int(min(len(before_state), len(before_action), len(smooth_state), len(smooth_action)))
    before_state = before_state[:frames]
    before_action = before_action[:frames]
    smooth_state = smooth_state[:frames]
    smooth_action = smooth_action[:frames]
    indices = sample_indices(frames, max(1, int(max_points)))

    if "timestamp" in smooth_df.columns:
        timestamp = np.asarray(smooth_df["timestamp"].to_numpy()[:frames], dtype=np.float64)
    elif "timestamp" in before_df.columns:
        timestamp = np.asarray(before_df["timestamp"].to_numpy()[:frames], dtype=np.float64)
    else:
        timestamp = np.arange(frames, dtype=np.float64)
    if "frame_index" in smooth_df.columns:
        frame_index = np.asarray(smooth_df["frame_index"].to_numpy()[:frames], dtype=np.float64)
    elif "frame_index" in before_df.columns:
        frame_index = np.asarray(before_df["frame_index"].to_numpy()[:frames], dtype=np.float64)
    else:
        frame_index = np.arange(frames, dtype=np.float64)

    state_delta_rmse = rmse_vector(before_state, smooth_state)
    action_delta_rmse = rmse_vector(before_action, smooth_action)
    state_action_rmse_before = rmse_vector(before_state, before_action)
    state_action_rmse_smooth = rmse_vector(smooth_state, smooth_action)
    episode_index = int(episode["episode_index"])
    return {
        "episode_id": f"episode_{episode_index:06d}",
        "episode_index": episode_index,
        "data_path": str(smooth_path.relative_to(smooth_dataset)),
        "before_data_path": str(before_path.relative_to(before_dataset)),
        "frames": frames,
        "plotted_frames": int(len(indices)),
        "task": task_text(episode),
        "timestamp": compact_list(timestamp, indices),
        "frame_index": compact_list(frame_index, indices),
        "state_before": compact_matrix(before_state, indices),
        "state_smooth": compact_matrix(smooth_state, indices),
        "action_before": compact_matrix(before_action, indices),
        "action_smooth": compact_matrix(smooth_action, indices),
        "state_delta_rmse": state_delta_rmse,
        "action_delta_rmse": action_delta_rmse,
        "state_action_rmse_before": state_action_rmse_before,
        "state_action_rmse_smooth": state_action_rmse_smooth,
        "state_delta_rmse_mean": finite_mean(state_delta_rmse),
        "action_delta_rmse_mean": finite_mean(action_delta_rmse),
        "state_action_rmse_before_mean": finite_mean(state_action_rmse_before),
        "state_action_rmse_smooth_mean": finite_mean(state_action_rmse_smooth),
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    before_dataset = args.before_dataset.expanduser().resolve()
    smooth_dataset = args.smooth_dataset.expanduser().resolve()
    before_info = read_json(before_dataset / "meta" / "info.json")
    before_modality = read_json(before_dataset / "meta" / "modality.json")
    smooth_info = read_json(smooth_dataset / "meta" / "info.json")
    smooth_modality = read_json(smooth_dataset / "meta" / "modality.json")
    episodes = read_jsonl(smooth_dataset / "meta" / "episodes.jsonl")
    episodes.sort(key=lambda row: int(row.get("episode_index", 0)))
    if int(args.limit) > 0:
        episodes = episodes[: int(args.limit)]

    before_indices = eef_indices(before_info, before_modality)
    smooth_indices = eef_indices(smooth_info, smooth_modality)
    payload_episodes = [
        episode_payload(
            before_dataset,
            smooth_dataset,
            before_info,
            smooth_info,
            episode,
            before_indices,
            smooth_indices,
            max_points=max(1, int(args.max_points)),
        )
        for episode in episodes
    ]
    total_frames = sum(int(episode["frames"]) for episode in payload_episodes)
    teleop_stack = smooth_info.get("teleop_stack") if isinstance(smooth_info.get("teleop_stack"), dict) else {}
    return {
        "before_dataset_dir": str(before_dataset),
        "smooth_dataset_dir": str(smooth_dataset),
        "dim_labels": list(DIM_LABELS),
        "state_indices_before": before_indices[0],
        "action_indices_before": before_indices[1],
        "state_indices_smooth": smooth_indices[0],
        "action_indices_smooth": smooth_indices[1],
        "rot6d_convention": teleop_stack.get("rot6d_convention", ""),
        "position_transform": (teleop_stack.get("action_position_frame_transform") or {}).get("formula", ""),
        "quality_summary": read_json(args.quality_summary.expanduser().resolve()),
        "smooth_summary": read_json(args.smooth_summary.expanduser().resolve()),
        "total_frames": total_frames,
        "episodes": payload_episodes,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_doc = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    payload = build_payload(args)
    write_html(output, payload)
    print(
        json.dumps(
            {
                "before_dataset": payload["before_dataset_dir"],
                "smooth_dataset": payload["smooth_dataset_dir"],
                "episodes": len(payload["episodes"]),
                "frames": payload["total_frames"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
