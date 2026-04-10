## Tunnel Families

- **T1/T2 (1-x, 2-x):** 5.5 m inner diameter, 1.2 m rings, 6 segments/ring, staggered joints
- **T3 (3-x):** 5.5 m diameter, continuous joints, multi-station registration
- **T4/T5 (4-x, 5-x):** 7.5 m inner diameter, 1.8 m rings, 7 segments/ring, complex interleaved K-blocks

## Critical Parameters (Unfolding Stage)

Four parameters require tunnel-specific adaptation. All others are locked to baseline values.

### Tunnel-Responsive Parameters

- **diameter (m)** — Physical tunnel diameter from RANSAC measurement. Empirical range: **[5.31, 7.6]**, baseline 5.5. Set to **5.5** for T1/T2/T3 and scale up for T4/T5 based on raw characteristics. Drives polar conversions and downstream mask sizing. Adapted in 27/30 tunnels by all 3 LLMs.

- **vertical_filter_window (m)** — Vertical extent of the polar filter window. Empirical range: **[4.5, 6.9]**, baseline 4.5. Scales with tunnel diameter: T1/T2 keep 4.5; T3 → 5.0–5.5; T4/T5 → **6.5–6.9**. Adapted in **60/90** m_s_k files (100% of T3/T4/T5, 0% of T1/T2).

- **delta (m)** — Half-thickness of point cloud slices. Empirical range: **[0.005, 0.01]**, baseline 0.005. T1/T2/T4/T5 keep 0.005. T3 (multi-station registration) → **0.006–0.01** for thicker slices to handle registration offsets. Adapted in **9/9 T3 files** (100% of T3 by all 3 LLMs).

### Physical Constant (set by family)

- **slice_spacing_factor (m)** — Ring spacing used for slicing the point cloud. **1.2** for T1/T2/T3; **1.8** for T4/T5. Directly determines `ring_count` — wrong value = wrong number of rings. Adapted to 1.8 in **all 51/51 T4/T5 m_s_k files**. **Must match the physical ring spacing.**

### Locked Parameters (do not change from baseline)

| Parameter | Baseline |
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

### Adaptation Rules

- `diameter`: use RANSAC-estimated diameter from raw characteristics. Fallback: 5.5 for T1/T2/T3, 7.5 for T4/T5.
- `vertical_filter_window`: 4.5 for T1/T2; 5.0–5.5 for T3; 6.5–6.9 for T4/T5. Scales approximately with diameter.
- `delta`: 0.005 for all families except T3 → 0.006–0.01 (thicker slices for multi-station registration).
- `slice_spacing_factor`: **1.2** for T1/T2/T3; **1.8** for T4/T5. Construction constant.
