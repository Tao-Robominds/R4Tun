# BO search space — Enhancing stage

## Tunable parameters

- **upsampling_stage1_target_distance (m)** — Target spacing after first upsampling pass. Range: **[0.055, 0.111]** (baseline 0.08).

- **upsampling_stage2_target_distance (m)** — Target spacing after second pass. Range: **[0.028, 0.056]** (baseline 0.04).

- **upsampling_stage3_target_distance (m)** — Target spacing after third pass. Range: **[0.014, 0.028]** (baseline 0.02).

- **inter_radius (m)** — Search radius for outlier pair detection in joint enhancement. Range: **[0.03, 0.08]** (baseline 0.06).

- **n_segment_end** — End index of the high-density scanner window along rings (integer). Range: **[5, 21]** (baseline 5).

## Proven defaults (fixed, not searched)

| Parameter | Value |
|---|---|
| curvature_threshold | 0.005 |
| depth_threshold_low | 0.005 |
| depth_threshold_high | 0.015 |

## Locked parameters (baseline, not searched)

| Parameter | Baseline |
|---|---|
| n_segment_start | 0 |
| duplicate_threshold | 0.02 |
| num_neighbors | 20 |
| num_interpolations | 2 |
| resolution | 0.005 |
| window_size | 9 |

## Constraints

- Maintain approximate **2:1 spacing ratio** across upsampling stages: stage1 ≈ 2× stage2 ≈ 4× stage3 (within the stated box bounds).
- **n_segment_end** should not exceed available ring count when that count is known from upstream artefacts.
