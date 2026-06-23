# Mission2 Data Chain

This repository now carries a local test copy of the Nero L10 mission2 data chain.

## Layout

- `missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session>/`
  Raw robot capture sessions grouped by filesystem creation date.
- `missions/nero/mission2/lerobot_v2/<YYYY-MM-DD>/<dataset>/`
  Exported LeRobot v2 datasets grouped by the same date rule.
- `missions/nero/mission2/trimmed/`
  One standard LeRobot v2 dataset root for all trimmed episodes.
- `missions/nero/mission2/smooth/`
  One standard LeRobot v2 dataset root after smoothing the raw 19D action column.
- `missions/nero/mission2/prepared_smooth/`
  Prepared smooth metadata and statistics used by downstream training/evaluation.
- `missions/nero/mission2/trimmed_by_date/<YYYY-MM-DD>/trimmed`
  Date index that points back to the standard `trimmed` dataset root.
- `missions/nero/mission2/smooth_by_date/<YYYY-MM-DD>/smooth`
  Date index that points back to the standard `smooth` dataset root.
- `missions/nero/mission2/manifest.jsonl`
  Unified inventory for raw, LeRobot v2, trimmed, smooth, and prepared smooth data.

## Why Trimmed Stays Flat

Keep `trimmed` as one mixed LeRobot v2 dataset root. GR00T/LeRobot readers expect a single
dataset root with `meta/`, `data/`, and `videos/`. If trimmed data is physically split by
date, each date becomes a separate dataset unless a custom aggregate reader is added.

Date provenance is still preserved in `manifest.jsonl`, `trimmed_by_date/`, and
`trimmed/meta/trim_manifest.jsonl`.

The same rule applies to `smooth`: keep it as a standard flat LeRobot v2 dataset root,
and use `smooth_by_date/` plus `manifest.jsonl` for provenance.

## Common Commands

The smooth and plotting scripts expect the data environment to provide `numpy`, `pandas`,
`scipy`, `matplotlib`, and `pyarrow`.

Refresh the local copy from the Isaac-GR00T repository:

```bash
python3 tools/data_chain/organize_mission_data.py --link-mode copy
```

Refresh only smooth outputs without re-copying raw data:

```bash
python3 tools/data_chain/organize_mission_data.py \
  --link-mode copy \
  --skip-raw \
  --skip-lerobot \
  --skip-trimmed
```

Export a raw capture session to LeRobot v2:

```bash
python3 tools/data_chain/export_lerobot_dataset.py \
  missions/nero/mission2/raw/2026-06-10/nero_l10_20260610T115505Z \
  missions/nero/mission2/lerobot_v2/2026-06-10/my_export \
  --camera realsense_head \
  --camera realsense_wrist \
  --video-alias ego_view \
  --video-alias wrist_view \
  --video-feature-key observation.images.ego_view \
  --video-feature-key observation.images.wrist_view \
  --schema rokae_xmate3_linker_l10_groot_v1_1_full_orientation
```

Open the trim viewer:

```bash
streamlit run tools/data_chain/trim_lerobot_episode_viewer.py -- \
  --dataset-dir missions/nero/mission2/lerobot_v2 \
  --output-root missions \
  --output-dataset-name trimmed
```

Regenerate smoothed actions from local trimmed data:

```bash
python3 tools/data_chain/smooth_action_commands.py --overwrite
```

Plot trimmed-vs-smooth absolute action overlays:

```bash
python3 tools/data_chain/plot_trimmed_vs_smooth_action.py
```

Compare trimmed-vs-smooth config-relative actions:

```bash
python3 tools/data_chain/compare_trimmed_vs_smooth_relative.py
```
