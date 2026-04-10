## Chain of Thought — Denoising (Critical Parameters Only)

Follow this 3-step process. Only 4 tunnel-responsive parameters may be adapted; 4 others must use proven defaults; all remaining stay at baseline.

### 1. ANCHORING
Compare the target tunnel's unfolded characteristics against the sample:
- Radial distribution: r_percentiles (p10, p25, p50, p75, p90, p95, p99)
- Tunnel diameter from unfolding stage
- Point density (nearest neighbor distance)
- Tunnel family (1-x/2-x = small, 3-x = continuous, 4-x/5-x = large)

### 2. PARAMETER ADAPTATION
Adapt exactly 4 tunnel-responsive parameters from DOMAIN KNOWLEDGE:
- `mask_r_low` from p10 of r_percentiles; range [2.09, 3.75]
- `mask_r_high` from p99 of r_percentiles; range [2.78, 4.38]
- `default_cutoff_z` ≈ diameter / 2; range [2.65, 6.27]
- `z_step` based on scan resolution; range [0.003, 0.005]

Hard-code these proven defaults (NOT the baseline values):
- `smoothing_window_size` = 5
- `smoothing_offset` = -0.002
- `grad_threshold` = 0.15
- `y_step` = 0.4

### 3. OUTPUT
Output the full JSON with tunnel-responsive values adapted and proven defaults applied.

### Parameter Guidelines:
- **Always provide EXACT numerical values**
- Derive mask values from r_percentiles evidence
- Output flowing analysis with section headers and final JSON parameter block
