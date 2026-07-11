## Chain of Thought Instructions for Enhancing Parameter Recommendations

Follow this structured analysis process when evaluating denoised tunnel characteristics and making enhancing parameter recommendations. Use **DOMAIN KNOWLEDGE** for all numeric thresholds, regime definitions, proven defaults, and per-parameter adaptation logic.

### 0. CONSERVATIVE DEFAULT PRINCIPLE (read first, applies to every parameter)

When uncertain whether a parameter should deviate from the SAM4Tun reference
default, keep the reference default. Only change a parameter when you have
clear evidence from the tunnel characteristics that the default would cause a
specific problem.

SAM4Tun reference defaults for enhancing:
- upsampling_stage1/2/3_target_distance = 0.08 / 0.04 / 0.02
- curvature_threshold = 0.0005
- depth_threshold_low / depth_threshold_high = 0.003 / 0.008
- inter_radius = 0.06
- duplicate_threshold = 0.02
- n_segment_start = 0, n_segment_end = segment_per_ring − 1 (5 for 6-seg, 6 for 7-seg)
- num_neighbors = 20, num_interpolations = 2
- resolution = 0.005, window_size = 9

For complex tunnels (4-\*, 5-\*) where geometry differs significantly from the
sample tunnel, prefer SAM4Tun reference defaults as the safe starting point.
The "proven robust defaults" listed later in this document were calibrated on
regular/continuous tunnel types and may not generalise to complex tunnels.

### 0b. DEPTH-MAP FUNNEL (read after denoising state)

If denoised state shows **retention < 50%** on T3 (`3-*`) OR **mapped_points < 50,000**:

- **STOP** — denoising mask is still wrong; do not tune upsampling/`window_size`.
- Report MASK_FAILURE and return to denoising with wider `mask_r_high ≥ p99 + 0.02`.

For T1/T2, keep the 20% / 50k threshold.

### 0c. DEPTH_MAP_COVERAGE_GATE (mandatory when denoise retention ≥ 20%)

White pixels in `depth_map.png` are **NaN cells** after projecting `(h, θ, r)` at `resolution=0.005`. High denoise retention alone does not guarantee a filled map.

**Estimate coverage before choosing parameters:**

```
grid_cells ≈ (h_span / resolution) × (theta_span / resolution)
point_density = valid_denoised_points / grid_cells
```

**COVERAGE_FAILURE** if `point_density < 0.08` OR denoised state shows median NN distance > 1.2× `upsampling_stage1_target_distance`.

**Peripheral vs central gaps:** if edge white fraction > central white + 10pp, prioritize `depth_threshold_low`, larger `window_size`, and finer upsampling — not denoise mask widening.

**Tuning order (apply only with evidence; document each change):**

| Lever | Effect | Rule |
|-------|--------|------|
| `window_size` | Interpolates NaN neighborhoods post-projection | 9 → 11 → 13 for scattered gaps; max 15 |
| `upsampling_stage1/2/3` | Midpoint density before projection | stage1 ≈ `0.85 × median_NN`; stage2 = stage1/2; stage3 = stage1/4 |
| `depth_threshold_low` | Outlier gap-fill outside n_segment band | Lower toward 0.003 when peripheral gaps |
| `depth_threshold_high` | Outlier gap-fill inside n_segment band | Lower when central gaps persist |
| `inter_radius` | Pairwise outlier interpolation reach | Increase when gap-fill adds < 500 points |
| `num_interpolations` | Points per outlier pair | 3 when gap-fill < 1000 points |
| `curvature_threshold` | Midpoint acceptance in upsampling | Increase up to 0.008 if stage-1 adds < 10k points |
| `n_segment_end` | Ring window for dual thresholds | **`ring_count − 1`** when `ring_count` known; not only `segment_per_ring − 1` |

For T3 (`3-*`): sam4tun subsets are sparser than legacy full-cloud runs — **do not skip upsampling**; run COVERAGE_GATE even when density looks uniform.

### 1. ANCHORING
Compare the current tunnel's denoised point cloud characteristics against the sample baseline to establish differences that affect enhancing performance.

**Key metrics to compare:**
- Point density after denoising (mean/median nearest neighbor distance)
- Data retention rate and point distribution patterns
- Surface geometry complexity and curvature distribution
- Spatial coverage and gap patterns that need enhancement

**Calculation process:**
- Calculate percentage differences: Δ = (new_value - sample_value) / sample_value × 100%
- Focus on denoised characteristics that impact enhancement effectiveness
- Document density and geometric variations that affect upsampling needs

### 2. CLASSIFICATION
Group the tunnel into enhancing regimes based on the anchoring comparison. Apply **classification criteria and logic** from DOMAIN KNOWLEDGE (SIMILAR, SPARSE, DENSE, LOW-QUALITY, COMPLEX-GEOMETRY, LARGE-SCALE, CRITICAL-SPARSE).

### 3. DIAGNOSTIC INSPECTION
Examine enhancing-specific challenges based on the classification. Follow **diagnostic rules** in DOMAIN KNOWLEDGE for each regime (upsampling distances, curvature, depth thresholds, inter_radius, n_segment ranges, CRITICAL-SPARSE combinations).

### 4. PARAMETER ADAPTATION
Consult DOMAIN KNOWLEDGE for reference vs proven defaults, proven robust defaults, and parameter-specific adaptation logic.

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- For SIMILAR / T1/T2 tunnels: keep **REFERENCE PARAMETERS** unless state shows a named failure.
- For T3 (`3-*`) and any tunnel with COVERAGE_FAILURE: apply **DEPTH_MAP_COVERAGE_GATE** levers in order; do not copy external parameter sets.
- Large curvature differences (>100%) often reflect processing variations — verify significance
- Moderate density differences (<25%) and curvature changes (<150%) should lean towards SIMILAR classification
- curvature_threshold 0.005 is robust across tunnel types — prefer keeping unless extreme validated differences
- Document specific reasoning for each parameter decision with evidence

**Evidence requirements:**
- Each parameter change must be supported by specific evidence from diagnostic inspection
- Quantify the problem being solved (e.g., "upsampling_stage1_target_distance=0.08 too coarse for NN distance=0.044")
- Consider parameter interdependencies and enhancement quality

### 5. VALIDATION
Check that proposed changes resolve identified issues without undermining enhancement effectiveness:

**Consistency checks:**
- Ensure parameter changes align with tunnel regime classification
- Verify that modifications address the specific issues identified in diagnostic inspection
- Confirm that adaptations follow the evidence-based reasoning from anchoring analysis

**Enhancement effectiveness:**
- Check that upsampling parameters produce appropriate density enhancement
- Ensure geometry thresholds maintain surface feature detection
- Verify interpolation parameters preserve surface quality

**Final validation criteria:**
- Parameters should resolve identified challenges while preserving proven robustness
- For SIMILAR tunnels: changes should be minimal or absent
- All modifications should have clear quantitative justification
- Output flowing analysis with section headers and final JSON parameter block

### Parameter Guidelines:
- **Always provide EXACT numerical values** — Never use ranges like "0.03-0.05" or "15-25"
- **Choose the most appropriate single value** from any range DOMAIN KNOWLEDGE suggests
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Keep upsampling_stage1_target_distance at 0.06 (proven default — tunnel characteristics are similar to sample)"
- "Set curvature_threshold to 0.005" (not "0.003-0.005")
- "Use inter_radius of 0.03" (not "0.03-0.06")

Remember: The system requires exact values for implementation — ranges cannot be processed.
