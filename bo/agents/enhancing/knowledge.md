## Tunnel Families

- **T1/T2 (1-x, 2-x):** 5.5 m inner diameter, 1.2 m rings, 6 segments/ring, staggered joints
- **T3 (3-x):** 5.5 m diameter, continuous joints, multi-station registration
- **T4/T5 (4-x, 5-x):** 7.5 m inner diameter, 1.8 m rings, 7 segments/ring, complex interleaved K-blocks

## Critical Parameters (Enhancing Stage)

Five parameters are tunnel-responsive. Three parameters are proven baseline corrections. The rest are locked.

### Tunnel-Responsive Parameters

- **upsampling_stage1_target_distance (m)** — Target spacing after first upsampling pass. Empirical range: **[0.055, 0.111]**, baseline 0.08, CV=0.064. Two clusters: ~0.06 for dense tunnels, ~0.068 for sparse/large tunnels. Adapted in **30/30** tunnels.

- **upsampling_stage2_target_distance (m)** — Target spacing after second pass. Empirical range: **[0.028, 0.056]**, baseline 0.04, CV=0.064. Scales proportionally with stage1. Adapted in **30/30** tunnels.

- **upsampling_stage3_target_distance (m)** — Target spacing after third pass. Empirical range: **[0.014, 0.028]**, baseline 0.02, CV=0.064. Scales proportionally with stage1. Adapted in **30/30** tunnels.

- **inter_radius (m)** — Search radius for outlier pair detection in joint enhancement. Empirical range: **[0.03, 0.08]**, baseline 0.06, CV=0.130. Dense tunnels (1-x, 2-x) → 0.03; sparse large (4-x, 5-x) → 0.038–0.043. Adapted in **30/30** tunnels.

- **n_segment_end** — End of high-density scanner window in ring indices. Empirical range: **[5, 21]**, baseline 5. Scale with ring count: use ~half of ring_count. Adapted in **23/30** tunnels.

### Hard-Coded Proven Defaults (do NOT adapt)

| Parameter | Baseline | Proven value | Always use |
|---|---|---|---|
| curvature_threshold | 0.0005 | **0.005** | 0.005 |
| depth_threshold_low | 0.003 | **0.005** | 0.005 |
| depth_threshold_high | 0.008 | **0.015** | 0.015 |

### Locked Parameters (keep baseline)

| Parameter | Baseline |
|---|---|
| n_segment_start | 0 |
| duplicate_threshold | 0.02 |
| num_neighbors | 20 |
| num_interpolations | 2 |
| resolution | 0.005 |
| window_size | 9 |

### Adaptation Rules

- Upsampling distances should maintain 2:1 ratio between stages (stage1 ≈ 2× stage2 ≈ 4× stage3)
- `inter_radius`: use 0.03 for most tunnels; increase to 0.038–0.043 for T4/T5 with wider gaps
- `n_segment_end`: scale with ring count from characteristics; typical 5–10 for short subsets, up to 21 for long ones
