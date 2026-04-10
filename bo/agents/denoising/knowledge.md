## Tunnel Families

- **T1/T2 (1-x, 2-x):** 5.5 m inner diameter, 1.2 m rings, 6 segments/ring, staggered joints
- **T3 (3-x):** 5.5 m diameter, continuous joints, multi-station registration
- **T4/T5 (4-x, 5-x):** 7.5 m inner diameter, 1.8 m rings, 7 segments/ring, complex interleaved K-blocks

## Critical Parameters (Denoising Stage)

Four parameters are tunnel-responsive (must adapt per tunnel). Four parameters are proven baseline corrections (hard-coded, same for all tunnels).

### Tunnel-Responsive Parameters

- **mask_r_low (m)** — Inner radial gate before histogramming. Empirical range: **[2.09, 3.75]**, baseline 2.7, CV=0.082. Set from `r_percentiles.p10` in unfolded characteristics. Families 1-x/2-x cluster at 2.25–2.38; families 4-x/5-x at 2.62–2.91. Adapted in **30/30** tunnels by all 3 LLMs.

- **mask_r_high (m)** — Outer radial gate. Empirical range: **[2.78, 4.38]**, baseline 2.8, CV=0.147. Set from `r_percentiles.p99`. Sharp binary split: families 1-x/2-x ≈2.79 (barely above baseline); families 4-x/5-x ≈3.85 (+37–40%). If p99 > 3.0, increase mask_r_high — insufficient width causes large white areas in depth maps. Adapted in **30/30** tunnels.

- **default_cutoff_z (m)** — Fallback radius when a θ-bin has no reliable counts. Empirical range: **[2.65, 6.27]**, baseline 2.7, CV=0.142. Keep synchronized with diameter / 2. Adapted in **29/30** tunnels.

- **z_step (m)** — Radial bin size per histogram column. Empirical range: **[0.003, 0.005]**, baseline 0.001, CV=0.181. Families 4-2 to 4-9 (high-res scans) get 0.003; families 1-x/2-x/5-x (standard scans) get 0.005. Adapted in **30/30** tunnels.

### Hard-Coded Proven Defaults (do NOT adapt — same value for all tunnels)

All 3 LLMs independently converge on these corrections regardless of tunnel:

| Parameter | Baseline | Proven value | Always use |
|---|---|---|---|
| smoothing_window_size | 3 | **5** | 5 |
| smoothing_offset | -0.003 | **-0.002** | -0.002 |
| grad_threshold | 0.2 | **0.15** | 0.15 |
| y_step | 0.5 | **0.4** | 0.4 |

### Mask Setting Rules

- `mask_r_low`: at or below **p10** of r distribution from unfolded characteristics
- `mask_r_high`: at least **p99** of r distribution; if p99 > 3.0, must increase
- `default_cutoff_z`: ≈ diameter / 2, synchronized with unfolding `diameter`
- `z_step`: 0.005 for standard scans, 0.003 for high-resolution family 4 scans
