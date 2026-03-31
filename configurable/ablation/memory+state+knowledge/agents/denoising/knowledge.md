## Tunnel Variations and Adaptations (T1-T5)

### Base Tunnels (T1 & T2)
- **Geometry**: 5.5 m inner diameter metro tunnels
- **Ring structure**: 1.2 m ring length with six segments per ring
- **Joint type**: Staggered joints arrangement
- **Scanning**: Leica C10 scanner, single-station acquisitions in Wuxi
- **SAM4Tun approach**: Default parameter set, base case for template and threshold development
- **Test sets**: T1-4, T2-2 among others used for validation

### Tunnel T3 Variations
- **Geometry**: 5.5 m nominal diameter (similar to T1/T2)
- **Key difference**: Continuous joints instead of staggered
- **Data formation**: Multi-station registration creating more uniform density distribution
- **SAM4Tun adaptations**:
  - **No upsampling** applied due to uniform density
  - **Global outlier threshold**: 0.01 m
  - **Detection logic**: Reuses T1/T2 approach with fallback for horizontal-segment detection failures
  - **Template customization**: Adjusted prompts for T3's specific segment dimensions and bolt-hole locations
- **Evaluation scope**: First 50 rings spanning two stations

### Large Tunnels (T4 & T5)
- **Geometry**: 7.5 m inner diameter (larger scale)
- **Ring structure**: 1.8 m ring length with seven segments per ring
- **Joint type**: Complex interleaved K-block arrangement
- **Scanning**: Leica C10 scanner in Fuzhou with offset scanner center from tunnel axis
- **SAM4Tun adaptations**:
  - **Density zone detection**: Identifies high-density zones (seven rings nearest scanner)
  - **Adaptive outlier thresholds**: 10 mm for high-density areas vs 4 mm for low-density areas
  - **Parameter consistency**: Maintains T1/T2 setup for other parameters

### Cross-Tunnel Variation Summary
The five tunnels span:
- **Two diameters**: 5.5 m vs 7.5 m
- **Two ring lengths**: 1.2 m vs 1.8 m  
- **Two segment counts**: 6 vs 7 segments per ring
- **Two joint assemblies**: Staggered vs continuous joints
- **Scanning configurations**: Single-station TLS (T1/T2/T4/T5) vs multi-station registration (T3)

### Parameter Adaptation Strategy
- **Uniform-density tunnels** (T3): Drop upsampling, adjust prompts
- **Larger-diameter tunnels** (T4/T5): Split scenes into density regions with distinct thresholds
- **Core pipeline**: Downstream processing remains largely unchanged across variations
- **Customization focus**: Pre-processing choices adapt to geometric and scanning differences

## Success Criteria
- **Line detection accuracy**: Successful identification of tunnel joint patterns
- **Prompt point generation**: Valid intersection points for SAM guidance
- **Geometric consistency**: Detected patterns match expected tunnel structure
- **Processing robustness**: Graceful handling of missing or weak line features
- **Cross-tunnel adaptability**: Parameters adjust appropriately for T1-T5 variations

## Critical Parameter Distinctions

### segment_per_ring vs ring_count
- **segment_per_ring**: Fixed property of tunnel TYPE (T1/T2 = 6, T4/T5 = 7)
- **ring_count**: Variable number of rings in the dataset (e.g., 2-2 dataset has 10 rings)
- **segment_order**: Must match segment_per_ring length, NOT ring_count

### Tunnel Type Identification
- **T1/T2-type**: 5.5m diameter → segment_per_ring = 6
- **T4/T5-type**: 7.5m diameter → segment_per_ring = 7
- **Dataset naming**: "2-2" means tunnel type 2, dataset 2 (still 6 segments per ring)

## Parameter Reference (Denoising Stage)

Parameters in `configurable/*/parameters_denoising.json` control the cylindrical grid used to identify sparse/noisy points before Algorithm 2 produces `denoised.csv`. Keep the following intent and ranges in mind:

- **mask_r_low / mask_r_high (m)** – radial gate applied before any histogramming. Set them to bracket the expected tunnel radius (≈ diameter / 2). Observed settings span **2.2–3.5** for 5.5 m tunnels and up to **4.0** for 7.5 m tunnels. Tighten the window to suppress stray scaffolding.
- **y_step (m)** – azimuth bin size (along θ). Typical values **0.4–0.7** give 25–40 bins around the circumference; shrink it only if the dataset has very high angular density.
- **z_step (m)** – radial bin size per histogram column. Use **0.001–0.0055** depending on how much vertical noise the scanner introduced; finer steps support sharper cutoffs but require more samples.
- **grad_threshold** – gradient drop (ratio) that marks the point where occupied bins end. Stable values sit between **0.15–0.20**; raise it when the wall fades gradually due to heavy occlusion.
- **smoothing_window_size** – number of bins in the moving average when flattening the cutoff curve. Typical choices are **3** for dense scans and **5** when more smoothing is needed.
- **smoothing_offset (m)** – safety margin added after smoothing so we do not delete the actual wall. Keep it slightly negative (**‑0.003 to ‑0.002**) to bias toward retaining more points.
- **default_cutoff_z (m)** – fallback radius used when a θ-bin has no reliable counts. Always keep it synchronized with the unfolding `diameter` (≈ diameter / 2). Current datasets use **2.7–3.75** depending on tunnel size; updating diameter requires updating this constant as well for the LLM agent.

## Denoising — Classification Criteria (CoT domain knowledge)

- **SIMILAR**: <25% difference in key density metrics AND radial span <200% change → minimal/no changes needed
- **DENSE**: Significantly higher point density (>30% decrease in NN distance) → may need finer grid parameters
- **SPARSE**: Significantly lower point density (>30% increase in NN distance) → may need coarser grid parameters
- **THICK-RING**: Extreme radial span difference (>300% change AND validated as real geometric difference) → adjust radial masking parameters
- **ANGULAR-DENSE**: Significant theta coverage difference (>25% change) → adjust angular grid parameters
- **LARGE-DIAMETER**: Tunnel diameter significantly larger than sample (>30% increase) → scale radial mask parameters appropriately
- **EXTREME-RANGE**: Very wide radial distribution (>60% radial span increase OR >50% diameter increase + irregular distribution) → aggressive radial parameter adaptation

**Classification logic:** Primary — cylindrical density changes; secondary — radial span and angular coverage; multiple regimes possible (e.g., DENSE + THICK-RING).

## Denoising — Diagnostic Rules

**DENSE:** Check y_step reduction for finer sampling; grad_threshold for noise sensitivity.

**SPARSE:** Check y_step increase for points per cell; grad_threshold relaxation for sparse data.

**THICK-RING:** mask_r_low/mask_r_high vs radial spans; z_step for radial density variations.

**LARGE-DIAMETER (>30% diameter increase, i.e. >7.2m):** Scale mask_r_low/mask_r_high to median radius (e.g. ~7.2m: [3.5, 3.8] vs ~5.5m default [2.8, 3.0]). Modest increases (<30%) keep [2.8, 3.0]. Keep **y_step = 0.4**; do NOT increase y_step for larger tunnels. Keep **z_step = 0.005** unless extreme radial variation.

**EXTREME-RANGE:** Expand radial mask (e.g. 2.2–3.9 vs 2.8–3.0); z_step may decrease for finer radial sampling; irregular shape may need non-standard mask positioning.

**General:** Grid compatibility (y_step vs angular resolution); radial masking vs diameter; grad_threshold vs density pattern.

## Denoising — Reference vs Proven Defaults

The REFERENCE PARAMETERS block in the prompt is sam4tun starting-point defaults — **not** proven optimal. **Proven defaults supersede reference** for SIMILAR tunnels:

| Reference (typical) | Proven default |
|---------------------|----------------|
| y_step 0.5 | **0.4** |
| z_step 0.001 | **0.005** |
| grad_threshold 0.2 | **0.15** |
| smoothing_window_size 3 | **5** |
| smoothing_offset -0.003 | **-0.002** |

## Denoising — Proven Robust Defaults

- **y_step** = 0.4
- **z_step** = 0.005
- **grad_threshold** = 0.15
- **smoothing_window_size** = 5
- **smoothing_offset** = -0.002
- **mask_r_low / mask_r_high**: From **r_percentiles** in unfolded characteristics — mask_r_low at or below **p10**; mask_r_high at least **p99** of r. If p99 > 3.0, increase mask_r_high — insufficient width causes large white areas in depth maps.

## Denoising — Parameter-Specific Adaptation Logic

- **mask_r_low/mask_r_high**: p10 / p99 from unfolded r_percentiles; must capture ≥99% of wall points.
- **y_step**: Default 0.4; only 0.5–0.6 for VERY-SPARSE angular bins; never > 0.6.
- **z_step**: Default 0.005; EXTREME-RANGE wide spans → 0.002–0.003.
- **grad_threshold**: Default 0.15; SPARSE + high noise → 0.2–0.25; lower values → cleaner denoising.
- **smoothing_window_size**: Default 5; reduce to 3 only for very short tunnels / few angular bins.
- **smoothing_offset**: Default -0.002; change only for calibration needs.