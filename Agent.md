# Data Chain Agent Guide

This guide is for agents or operators who add new robot raw captures and process them into the standard smooth LeRobot dataset used by downstream GR00T/LeRobot workflows.

## Goal

For each new capture batch, keep the chain in this order:

```text
raw capture session -> LeRobot v2 dataset -> trimmed dataset -> smooth dataset -> comparison reports
```

The canonical mission directory is:

```text
missions/nero/mission2/
```

## Environment

Install Python dependencies from the repository root:

```bash
python3 -m pip install -r requirement.txt
```

Install `ffmpeg` separately and make sure it is on `PATH`:

```bash
ffmpeg -version
```

`ffmpeg` is required for LeRobot video export and H264 transcode during trimming.

## Directory Contract

Keep mission2 data under these paths:

- `missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session>/`
  Raw robot capture sessions. New raw data must be added here before export.
- `missions/nero/mission2/lerobot_v2/<YYYY-MM-DD>/<dataset>/`
  Exported LeRobot v2 datasets, grouped by the same date used for raw.
- `missions/nero/mission2/trimmed/`
  One flat LeRobot v2 dataset root containing all accepted trimmed episodes.
- `missions/nero/mission2/smooth/`
  One flat LeRobot v2 dataset root generated from `trimmed/` after action smoothing.
- `missions/nero/mission2/prepared_smooth/`
  Prepared smooth metadata and statistics for downstream use when present.
- `missions/nero/mission2/manifest.jsonl`
  Inventory for raw, LeRobot v2, trimmed, smooth, and prepared smooth data.

Do not physically split `trimmed/` or `smooth/` by date. GR00T/LeRobot readers expect one dataset root with `meta/`, `data/`, and `videos/`. Preserve date provenance through `manifest.jsonl`, `trimmed/meta/trim_manifest.jsonl`, and smooth metadata files.

## Adding New Raw Data

Place each new raw capture session as a complete directory under:

```text
missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session>/
```

Use the capture date for `<YYYY-MM-DD>`. Use the original capture directory name for `<capture_session>` when possible, for example:

```text
missions/nero/mission2/raw/2026-06-22/nero_l10_20260622T091509Z/
```

Each raw capture session should remain self-contained. Do not flatten episode folders or move files out of the original capture session directory. The exporter expects to read the capture session root directly.

## Export Raw To LeRobot V2

Export each raw session into a dated LeRobot v2 dataset directory:

```bash
python3 tools/data_chain/export_lerobot_dataset.py \
  missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session> \
  missions/nero/mission2/lerobot_v2/<YYYY-MM-DD>/<dataset_name> \
  --camera realsense_head \
  --camera realsense_wrist \
  --video-alias ego_view \
  --video-alias wrist_view \
  --video-feature-key observation.images.ego_view \
  --video-feature-key observation.images.wrist_view \
  --schema rokae_xmate3_linker_l10_groot_v1_1_full_orientation
```

Choose `<dataset_name>` so it identifies the hardware, mission, and date, for example:

```text
groot_lerobot_v2_nero_l10_20260622
```

Use `--dry-run` first when validating an unfamiliar raw session:

```bash
python3 tools/data_chain/export_lerobot_dataset.py \
  missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session> \
  missions/nero/mission2/lerobot_v2/<YYYY-MM-DD>/<dataset_name> \
  --camera realsense_head \
  --camera realsense_wrist \
  --video-alias ego_view \
  --video-alias wrist_view \
  --video-feature-key observation.images.ego_view \
  --video-feature-key observation.images.wrist_view \
  --schema rokae_xmate3_linker_l10_groot_v1_1_full_orientation \
  --dry-run
```

## Trim Accepted Episodes

Open the trim viewer against the LeRobot v2 parent directory:

```bash
streamlit run tools/data_chain/trim_lerobot_episode_viewer.py -- \
  --dataset-dir missions/nero/mission2/lerobot_v2 \
  --output-root missions \
  --output-dataset-name trimmed
```

The viewer writes accepted clips into:

```text
missions/nero/mission2/trimmed/
```

Keep appending accepted episodes to this single `trimmed/` dataset. The trim manifest is written to both:

```text
missions/nero/mission2/trimmed/trim_manifest.jsonl
missions/nero/mission2/trimmed/meta/trim_manifest.jsonl
```

## Generate Smooth Dataset

Regenerate the smooth dataset after new trimmed episodes are accepted:

```bash
python3 tools/data_chain/smooth_action_commands.py --overwrite
```

By default this reads:

```text
missions/nero/mission2/trimmed/
```

and writes:

```text
missions/nero/mission2/smooth/
```

Only the raw 19D `action` column is smoothed. Videos, observations, tasks, and metadata are copied from `trimmed/`.

## Refresh Inventory

Refresh `manifest.jsonl` when the local mission directory should reflect the latest raw, LeRobot v2, trimmed, smooth, and prepared smooth paths:

```bash
python3 tools/data_chain/organize_mission_data.py --link-mode copy
```

If only smooth outputs changed and raw/LeRobot/trimmed data should not be recopied, run:

```bash
python3 tools/data_chain/organize_mission_data.py \
  --link-mode copy \
  --skip-raw \
  --skip-lerobot \
  --skip-trimmed
```

## Validate Smooth Output

Generate action smoothing plots and metrics:

```bash
python3 tools/data_chain/plot_trimmed_vs_smooth_action.py
```

Compare config-relative actions:

```bash
python3 tools/data_chain/compare_trimmed_vs_smooth_relative.py
```

Review outputs under:

```text
missions/nero/mission2/action_smooth_comparison/
missions/nero/mission2/relative_smooth_comparison/
```

## Agent Checklist

When new raw data arrives:

1. Put the raw session under `missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session>/` without changing its internal layout.
2. Export it to `missions/nero/mission2/lerobot_v2/<YYYY-MM-DD>/<dataset_name>/` with `export_lerobot_dataset.py`.
3. Use the Streamlit trim viewer to append valid clips into the single flat `missions/nero/mission2/trimmed/` dataset.
4. Run `smooth_action_commands.py --overwrite` to rebuild `missions/nero/mission2/smooth/`.
5. Run the comparison scripts and inspect metrics/plots before treating the smooth dataset as ready.
6. Refresh `manifest.jsonl` with `organize_mission_data.py` if inventory metadata needs to be current.
