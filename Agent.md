# Data Chain Agent Guide

This guide is for agents or operators who add new robot raw captures and process them into the standard smooth LeRobot dataset used by downstream GR00T/LeRobot workflows.

## Goal

For each new capture batch, keep the chain in this order:

```text
raw capture session -> LeRobot v2 dataset -> trimmed dataset -> smooth dataset -> unified quality report -> comparison reports
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
- `missions/nero/mission2/trimmed_by_date/<YYYY-MM-DD>/trimmed/`
  Date-filtered views of `trimmed/`. Each view keeps only the episodes whose
  source capture date matches `<YYYY-MM-DD>`. Metadata is filtered, while
  parquet and video payloads are file symlinks back to the canonical `trimmed/`
  files.
- `missions/nero/mission2/smooth/`
  One flat LeRobot v2 dataset root generated from `trimmed/` after action smoothing.
- `missions/nero/mission2/smooth_by_date/<YYYY-MM-DD>/smooth/`
  Date-filtered views of `smooth/`, rebuilt from the smooth metadata after
  smoothing is regenerated.
- `missions/nero/mission2/prepared_smooth/`
  Prepared smooth metadata and statistics for downstream use when present.
- `missions/nero/mission2/manifest.jsonl`
  Inventory for raw, LeRobot v2, trimmed, smooth, and prepared smooth data.

Do not physically split the canonical `trimmed/` or `smooth/` datasets by date. GR00T/LeRobot readers expect one canonical dataset root with `meta/`, `data/`, and `videos/`. Use `trimmed_by_date/` and `smooth_by_date/` when a workflow needs only one capture date; these are filtered dataset views, not symlinks to the full canonical dataset.

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
  --camera wrist_d405_rgb \
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

The exporter uses the nearest mission `MISSION.md` `- Mission:` value as the
LeRobot task label by default, so all mission2 batches share the same
`meta/tasks.jsonl` task. Use `--task "..."` only when an explicit override is
needed.

The wrist camera stream is recorded in raw captures as `wrist_d405_rgb`; keep
its LeRobot alias and feature key as `wrist_view` /
`observation.images.wrist_view`.

## EEF State/Action Alignment Contract

Before trusting any newly exported, trimmed, or repaired LeRobot dataset, inspect
EEF state/action alignment. Do this before smoothing so axis mistakes do not get
propagated into `smooth/`.

Use these checks on the target dataset root:

```bash
python3 tools/data_chain/check_and_fix_lerobot_eef_xyz.py \
  --mode inspect \
  --dataset-dir missions/nero/mission2/trimmed \
  --output-dir missions/nero/mission2/quality/eef_xyz_inspect

python3 tools/data_chain/check_and_fix_lerobot_eef9d.py \
  --mode inspect \
  --dataset-dir missions/nero/mission2/trimmed \
  --output-dir missions/nero/mission2/quality/eef9d_alignment
```

For ROKAE xMate3 + Linker L10 data, the raw action EEF position axes are not in
the same order as state EEF position. The required position mapping is:

```text
state xyz ~= action raw [z, x, y]
```

The repaired LeRobot action position must be stored as state-frame `[x, y, z]`,
using raw action values `[raw_z, raw_x, raw_y]`. Do not apply per-episode XYZ
offsets during repair; offsets are useful for diagnosis, but writing them would
change the action target semantics.

For full-orientation datasets, state Rot6D is stored in GR00T row-major order:

```text
[r00, r01, r02, r10, r11, r12]
```

Raw action rotation must be converted into the state frame before being stored.
The expected matrix relation is:

```text
R_state ~= L @ R_action @ R
L rows = [z, x, y]
R rows = [-y, -z, x]
```

In row-major Rot6D component form, this action-to-state mapping is:

```text
[r22, -r20, -r21, r02, -r00, -r01]
```

The current exporter applies both the XYZ `[z, x, y]` position mapping and the
Rot6D frame mapping before writing LeRobot v2 action features. For already
exported historical data, first run the inspect mode and only apply repairs when
the whole dataset or a clearly identified episode group matches the expected
mapping:

```bash
python3 tools/data_chain/check_and_fix_lerobot_eef_xyz.py \
  --mode fix \
  --dataset-dir missions/nero/mission2/trimmed \
  --output-dir missions/nero/mission2/quality/eef_xyz_fix_dryrun

python3 tools/data_chain/check_and_fix_lerobot_eef_xyz.py \
  --mode fix \
  --dataset-dir missions/nero/mission2/trimmed \
  --output-dir missions/nero/mission2/quality/eef_xyz_fix_apply \
  --apply
```

If Rot6D inspection shows a historical dataset is still in raw action frame,
repair it with the rotation checker. If the inspect CSV shows mixed best
mappings, group episodes by their best matrix relation or use
`--fix-strategy best-per-episode` only after confirming the grouping; do not
blindly apply one transform to every episode.

```bash
python3 tools/data_chain/check_and_fix_lerobot_eef9d.py \
  --mode fix \
  --fix-strategy best-per-episode \
  --dataset-dir missions/nero/mission2/trimmed \
  --output-dir missions/nero/mission2/quality/eef9d_fix_dryrun

python3 tools/data_chain/check_and_fix_lerobot_eef9d.py \
  --mode fix \
  --fix-strategy best-per-episode \
  --dataset-dir missions/nero/mission2/trimmed \
  --output-dir missions/nero/mission2/quality/eef9d_fix_apply \
  --apply
```

For a repaired trimmed dataset, the XYZ inspect summary should report
`best_identity_ratio = 1.0`, and the Rot6D inspect summary should report the
action rotation as already in the state frame. Use
`tools/data_chain/visualize_trimmed_eef9d.py` to visually compare the 9 EEF
dimensions after repair.

Use `--dry-run` first when validating an unfamiliar raw session:

```bash
python3 tools/data_chain/export_lerobot_dataset.py \
  missions/nero/mission2/raw/<YYYY-MM-DD>/<capture_session> \
  missions/nero/mission2/lerobot_v2/<YYYY-MM-DD>/<dataset_name> \
  --camera realsense_head \
  --camera wrist_d405_rgb \
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

After each accepted clip, the viewer also refreshes the matching date view under:

```text
missions/nero/mission2/trimmed_by_date/<YYYY-MM-DD>/trimmed/
```

The date view should contain only that date's episode metadata and only file
links for that date's parquet/video payloads. It must not be a directory symlink
to the full `trimmed/` dataset.

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

This also rebuilds `trimmed_by_date/`, `smooth_by_date/`, and
`prepared_smooth_by_date/` as filtered views when the corresponding canonical
datasets exist.

If only smooth outputs changed and raw/LeRobot/trimmed data should not be recopied, run:

```bash
python3 tools/data_chain/organize_mission_data.py \
  --link-mode copy \
  --skip-raw \
  --skip-lerobot \
  --skip-trimmed
```

## Validate Smooth Output

Run the unified quality gate against the canonical `smooth/` dataset. It writes
all quality outputs to `missions/nero/mission2/quality/`, including cached
episode/video/timing metrics, an incremental processed log, plots, and the main
HTML report:

```bash
python3 tools/data_chain/analyze_quality.py \
  --dataset-dir missions/nero/mission2/smooth \
  --output-dir missions/nero/mission2/quality \
  --raw-root missions/nero/mission2/raw
```

The main report is written to:

```text
missions/nero/mission2/quality/index.html
```

The quality pipeline is incremental. It records processed episode fingerprints
in `missions/nero/mission2/quality/cache/processed_episodes.jsonl`; later runs
reuse unchanged episode metrics and only analyze new or changed episodes. Use
`--force` when a full recompute is required. For quick smoke checks while
iterating, use:

```bash
python3 tools/data_chain/analyze_quality.py --skip-video --skip-timing
```

The report includes dataset completeness, source-date length distributions,
trajectory velocity/acceleration/jerk summaries, hand metrics, video metrics,
raw-log timing metrics when raw provenance is available, and optional embedding
plots when an `--embedding-backend module:function` is configured. If raw logs,
embedding models, or object detector backends are unavailable, those sections are
marked skipped or unavailable instead of emitting synthetic metrics.

Review the gate criteria in:

```text
docs/embodied_data_quality_standard.md
```

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
3. Inspect EEF XYZ and Rot6D state/action alignment on the exported dataset; repair only after the inspect output confirms the expected mapping.
4. Use the Streamlit trim viewer to append valid clips into the single flat `missions/nero/mission2/trimmed/` dataset.
5. Inspect EEF XYZ and Rot6D alignment again on `trimmed/`; fixed data should show XYZ identity and action Rot6D already in the state frame.
6. Run `smooth_action_commands.py --overwrite` to rebuild `missions/nero/mission2/smooth/`.
7. Confirm `trimmed_by_date/<YYYY-MM-DD>/trimmed/` contains only that date's accepted clips.
8. Run `analyze_quality.py` and inspect `missions/nero/mission2/quality/index.html` for completeness, timing, trajectory, hand, video, and date-distribution issues.
9. Run the comparison scripts and inspect metrics/plots before treating the smooth dataset as ready.
10. Refresh `manifest.jsonl` with `organize_mission_data.py` if inventory metadata needs to be current.
