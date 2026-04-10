# BO search space — Unfolding stage

## Tunable parameters

- **diameter (m)** — Nominal inner diameter for polar conversion and downstream mask sizing. Range: **[5.31, 7.6]** (baseline 5.5).

- **vertical_filter_window (m)** — Vertical extent of the polar filter when projecting slice points. Range: **[4.5, 6.9]** (baseline 4.5).

- **delta (m)** — Half-thickness of each slicing slab along the tunnel axis. Range: **[0.005, 0.01]** (baseline 0.005).

- **slice_spacing_factor (m)** — Nominal spacing between slice planes; strongly affects inferred ring count (ring_count ≈ tunnel XY length / slice_spacing_factor). Range: **[1.2, 1.8]** (typical construction spacings).

## Proven defaults (fixed, not searched)

None for this stage.

## Locked parameters (baseline, not searched)

| Parameter | Value |
|---|---|
| ransac_threshold | 1.0 |
| ransac_probability | 0.9 |
| ransac_inlier_ratio | 0.75 |
| ransac_sample_size | 5 |
| ransac_initial_iterations | 999 |
| ransac_inlier_threshold_multiplier | 0.8 |
| polynomial_degree | 3 |
| num_samples_factor | 1210 |
| t_extrapolation_start | -20 |
| t_extrapolation_end | 20 |
| batch_size | 1000000 |
| n_jobs | 12 |

## Constraints

- `vertical_filter_window` should stay consistent with tunnel scale (often correlated with `diameter` in practice).
- `slice_spacing_factor` is a primary lever for inferred ring count from unfolding geometry (≈ tunnel axis extent / spacing); keep within the stated range when co-tuning with downstream ring priors.
