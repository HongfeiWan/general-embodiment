# Embodied Data Quality Standard

This repository treats every accepted dataset batch as usable only after it has
passed structural, motion, visual, distribution, and provenance checks. The goal
is not only to find corrupt files, but also to catch demonstrations that are
valid files yet poor training examples.

## When To Run

Run the quality gate after each data-chain update:

```text
raw -> lerobot_v2 -> trimmed -> smooth -> quality reports -> review decision
```

For mission2 smooth data, regenerate the all-date report:

```bash
python3 tools/data_chain/analyze_smooth_quality_all_dates.py \
  --dataset-dir missions/nero/mission2/smooth \
  --output-dir missions/nero/mission2/smooth_quality/all_dates
```

Then run the standard LeRobot batch checker by source date:

```bash
python3 tools/data_chain/check_lerobot_batch_quality.py \
  --dataset-dir missions/nero/mission2/smooth \
  --output-root missions/nero/mission2/batch_quality \
  --all-dates \
  --date-field source \
  --max-video-checks 20
```

Use `--max-video-checks -1` when a full ffprobe frame-count pass is required and
runtime is acceptable. The all-date smooth report already performs a faster full
video pass over every stream.

## Required Outputs

The all-date smooth quality report writes:

- `smooth_quality_all_dates_report.html`: human-facing report.
- `summary.json`: gate summary, date colors, top outliers, table paths.
- `episode_quality_metrics.csv`: one row per episode.
- `date_quality_summary.csv`: date-level aggregates for distribution review.
- `video_quality.csv`: one row per episode/video stream.
- `issue_rows.csv`: blocking errors and warnings.
- `outlier_rankings.csv`: robust outlier scores for human review.
- `metric_distributions_by_source_date.png`: date-colored metric distributions.
- `episode_metric_pca_by_source_date.png`: date-colored embedding of episode metrics.
- `video_quality_by_source_date.png`: date-colored video sharpness/brightness.

Keep these reports under `missions/nero/mission2/smooth_quality/` so every
processed dataset has an auditable quality trail.

## Quality Dimensions

### Structural Integrity

Check that the LeRobot root is internally consistent:

- `meta/info.json`, `meta/episodes.jsonl`, `meta/modality.json`, and `meta/tasks.jsonl` exist.
- `info.json.total_episodes` equals `episodes.jsonl` rows.
- `info.json.total_frames` equals the sum of episode lengths.
- Every episode parquet exists and row count equals metadata length.
- Required columns exist: `action`, `observation.state`, `timestamp`, `frame_index`, `episode_index`, `next.done`.
- `action` is 19D and `observation.state` is 26D for Nero L10 mission2.
- Numeric arrays contain no NaN or Inf.
- `frame_index` is contiguous from `0..length-1`.
- `episode_index` inside parquet matches metadata.
- `next.done` is true only on the final frame.

Blocking rule: any structural error blocks release until fixed or the episode is
removed from the accepted dataset.

### Temporal Alignment

Check that signals and videos agree with the configured fps:

- Timestamps are finite and monotonic.
- Adjacent timestamp deltas match `1 / fps` within tolerance.
- Video duration roughly matches `episode.length / fps`.
- Video fps matches `info.json.fps`.

Warnings require review when isolated; repeated timestamp or duration warnings
within one source date should block release until the source capture/export path
is inspected.

### Motion Plausibility

For action and state trajectories, inspect adjacent-step, velocity,
acceleration, jerk, path length, and pause ratio. Main indicators:

- `action_step_xyz_l2_mm_*`
- `state_step_xyz_l2_mm_*`
- `action_acceleration_xyz_l2_mm_s2_*`
- `action_jerk_xyz_l2_mm_s3_*`
- `hand_action_step_l2_*`
- `eef_action_state_xyz_error_l2_mm_*`
- `hand_action_state_error_l2_*`

Use these metrics to catch discontinuities, controller spikes, unintended pauses,
and demonstrations where the commanded target and observed state diverge.

Default warning thresholds in `analyze_smooth_quality_all_dates.py`:

- EEF action adjacent step max above `30 mm`.
- EEF state adjacent step max above `30 mm`.
- h32 prepared q01/q99 clip ratio above `10%`.

These thresholds are intentionally conservative; compare against the date-colored
distribution plots before rejecting a batch.

### Pose Representation Validity

For 6D rotations, check both representation validity and temporal continuity:

- Rotation columns should stay near unit norm.
- The two 3D columns should stay near orthogonal.
- Adjacent-frame relative rotation angle should not spike unexpectedly.

High `rot6d_col01_abs_dot` or norm error indicates invalid orientation encoding,
usually from smoothing, export, or schema mismatch.

### Config-Relative Action Range

Downstream GR00T training consumes future relative action semantics, so absolute
actions alone are insufficient. The report computes relative EEF, hand, and arm
metrics for horizons `1, 2, 4, 8, 16, 32` using the local config semantics.

When `prepared_smooth/meta/relative_stats.json` exists, q01/q99 clip ratios are
computed against prepared training bounds. Inspect:

- `relative_h32_all_l2_p95`
- `relative_h32_q01q99_clip_ratio`
- `relative_h32_eef_q01q99_clip_ratio`
- `relative_h32_hand_q01q99_clip_ratio`
- `relative_h32_arm_q01q99_clip_ratio`

A date with much higher clip ratios than prior accepted dates should be reviewed
for changed task distribution, calibration drift, or erroneous trim windows.

### Visual Quality

Every video stream should be readable and browser-safe for review tooling:

- Codec is `h264` and pixel format is `yuv420p`.
- Width, height, fps, and duration are plausible.
- Sampled frame sharpness, brightness, and contrast are comparable across dates.
- Sample frame differences are not all near zero, which can indicate frozen video.

Low sharpness or unusual brightness should trigger spot-checking in the viewer,
especially when concentrated in one source date or camera stream.

### Distribution Shift And Coverage

Compare dates, not only individual episodes. The date-colored plots are the main
review surface:

- Episode length distribution.
- EEF motion step/acceleration/jerk distribution.
- Relative action magnitude and q01/q99 clip distribution.
- Action-state tracking error distribution.
- Video sharpness/brightness distribution.
- PCA plot of episode metrics, colored by source date.

Expected date differences are allowed when collection conditions or task variants
changed. Unexpected date clusters or isolated outliers should be reviewed before
the dataset is used for training.

## Review Procedure

1. Open `smooth_quality/all_dates/smooth_quality_all_dates_report.html`.
2. Confirm `summary.json` reports `errors = 0`.
3. Inspect `date_quality_summary.csv` for dates with unusual means or p95 values.
4. Inspect `metric_distributions_by_source_date.png` for shifted distributions.
5. Inspect `episode_metric_pca_by_source_date.png` for isolated episodes or date clusters.
6. Open `outlier_rankings.csv` and manually review the top 10-20 episodes.
7. Check `issue_rows.csv`; warnings require a written decision to accept, trim, or reject.
8. Confirm `batch_quality/*/summary.json` status is `pass` for every source date.

## Release Criteria

A smooth dataset is ready for downstream preparation/training when:

- All structural checks have zero errors.
- All source-date batch reports are `pass`.
- Videos are readable, browser-safe, and visually plausible.
- No date has an unexplained distribution shift in core motion or relative-action metrics.
- Top outlier episodes have been reviewed and either accepted, re-trimmed, or removed.
- The report artifacts are kept with the dataset for auditability.

When in doubt, keep the canonical `trimmed/` and `smooth/` datasets intact and
exclude questionable episodes through a documented follow-up trim/removal step.
