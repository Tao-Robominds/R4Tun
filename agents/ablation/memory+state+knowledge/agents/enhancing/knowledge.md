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
- **SAM4Tun adaptations (sam4tun/agents subset)**:
  - Subset point counts are **much lower** than legacy full-cloud pipelines — upsampling and gap-fill are usually required
  - Run **DEPTH_MAP_COVERAGE_GATE** after denoising; tune `window_size`, upsampling, and depth thresholds from `median_NN` and `h/θ` spans
  - **Global outlier threshold**: use `depth_threshold_low` / `depth_threshold_high` pair — lower low threshold when peripheral white bands dominate
  - **Detection logic**: Reuses T1/T2 approach with fallback for horizontal-segment detection failures

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
- **Uniform-density tunnels** (T3 on full cloud): may reduce upsampling; **sam4tun subset** still needs coverage tuning
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

## Parameter Reference (Enhancing Stage)

The upsampling / interpolation stage (Algorithm 3) uses `agents/ablation/{condition}/parameters/<tunnel_id>/parameters_enhancing.json`. Each field has a direct effect on point synthesis quality:

- **upsampling_stage{1,2,3}_target_distance (m)** – desired spacing between neighbors after each pass. Start near **0.08 / 0.04 / 0.02** for 5.5 m tunnels and reduce by ~25% for clean data; larger diameters can increase stage 1 to **0.10** to limit runtime.
- **curvature_threshold** – max acceptable curvature difference between neighbors when deciding to interpolate. Tight tunnels use **3e‑4–5e‑4**; rougher scans (T3/T4) can go up to **5e‑3**.
- **depth_threshold_low / depth_threshold_high (m)** – intensity of radial deviation required to mark “meaningful” outliers in low- vs high-density sections. Empirical ranges: **0.003–0.006** (low) and **0.008–0.015** (high).
- **inter_radius (m)** – search radius when picking outlier pairs for joint enhancement. Values between **0.03–0.08** cover all tunnels (shorter for dense stations, longer for sparse large-diameter scans).
- **duplicate_threshold (m)** – minimum spacing between newly generated points (default **0.02**). Increase slightly if you observe overlapping interpolations in T4/T5.
- **n_segment_start / n_segment_end** – high-density window in ring indices. **`n_segment_end = segment_per_ring − 1`** (5 for 6-seg T1–T3, 6 for 7-seg T4/T5). T1/T2: keep sample `0–5`.
- **num_neighbors** – number of neighbors queried in KDTree lookups, typically **20**. Raising it increases smoothing but costs time.
- **num_interpolations** – number of points inserted per qualifying pair (usually **2**).
- **resolution (m)** – target grid resolution when projecting to depth maps; the pipeline assumes **0.005** and downstream SAM processing expects the same.
- **window_size (px)** – sliding window for filling missing pixels during projection. Default **9**. Increase to **11–13** when depth maps show >15% white space after adequate denoise retention; each +2 fills wider NaN neighborhoods via nearest-neighbor interpolation.

## Enhancing — DEPTH_MAP_WHITE_SPACE (CoT domain knowledge)

White pixels = no projected point in that `(h, θ)` bin at `resolution=0.005`, after optional `window_size` fill.

**Diagnose before tuning:**

| Signal | Interpretation |
|--------|----------------|
| Denoise retention ≥ 50% but map still >15% white | COVERAGE_FAILURE — enhancing issue |
| Edge white > center white | Peripheral gap-fill weak — lower `depth_threshold_low`, increase `window_size` |
| Stage-1 upsampling adds < 10k points | `upsampling_stage1` too coarse vs `median_NN` — reduce stage1 |
| "Number of new added points" < 500 | Raise `inter_radius` or lower depth thresholds |
| `ring_count` ≫ `n_segment_end` | Expand `n_segment_end` to `ring_count − 1` |

**Parameter interaction (white-space reduction):**
- Finer upsampling → more `df_enhance_segment` points → more occupied bins
- Lower depth thresholds → more `enhance_outlier_points` joint fill → fills θ/h holes
- Larger `window_size` → fills isolated NaNs without new points (last resort; can blur)

## Enhancing — Classification Criteria (CoT domain knowledge)

- **SIMILAR**: <25% difference in density metrics AND <150% curvature change → minimal/no changes needed
- **SPARSE**: Lower point density after denoising (>30% difference) → may need more aggressive upsampling
- **DENSE**: Higher point density after denoising (>30% difference) → may need less aggressive upsampling
- **LOW-QUALITY**: Poor denoising retention rate (>35% difference) → may need adjusted thresholds
- **COMPLEX-GEOMETRY**: Extreme curvature patterns (>300% change AND validated as significant) → may need sensitivity adjustments
- **LARGE-SCALE**: Tunnel diameter >50% larger than sample (~8.25m+) → scaled enhancement parameters; increase target distances
- **CRITICAL-SPARSE**: Extremely sparse post-denoising (>80% data loss + large scale + high curvature) → specialized parameter combinations

**Classification logic:** Primary — denoised point density; secondary — data quality and geometric complexity; multiple regimes possible (e.g., SPARSE + COMPLEX-GEOMETRY).

## Enhancing — Diagnostic Rules

**SPARSE:** Upsampling target distances for denser enhancement; curvature_threshold relaxation for fewer feature points.

**DENSE:** Upsampling distances to avoid over-densification; duplicate_threshold for higher density.

**LOW-QUALITY:** depth_threshold_low/high for gap filling; inter_radius for interpolation.

**LARGE-SCALE (>50% diameter increase only):** Stage distances e.g. 0.08/0.04/0.02 for >8m; modest (<30%) keep 0.06/0.03/0.015. inter_radius 0.05 only for genuinely large + wide gaps; default 0.03 usually enough. n_segment range vs ring count (e.g. 10–21 for 20 rings, 0–9 for 10 rings).

**CRITICAL-SPARSE:** Larger targets 0.10/0.05/0.025; curvature_threshold e.g. 0.0003 vs 0.005; inter_radius 0.08; extend n_segment (e.g. 0–13 for 16 rings).

**General:** Upsampling vs density; curvature/depth vs surface complexity; neighbors/resolution vs data quality.

## Enhancing — Reference vs Proven Defaults

Reference block is sam4tun starting-point — **proven defaults supersede** for SIMILAR:

| Reference (typical) | Proven default |
|---------------------|----------------|
| target distances 0.08/0.04/0.02 | **0.06 / 0.03 / 0.015** |
| inter_radius 0.06 | **0.03** |
| depth_threshold_high 0.008 | **0.015** |

## Enhancing — Proven Robust Defaults

- **upsampling_stage1/2/3_target_distance** = 0.06 / 0.03 / 0.015
- **curvature_threshold** = 0.005
- **depth_threshold_low / depth_threshold_high** = 0.005 / 0.015
- **inter_radius** = 0.03
- **duplicate_threshold** = 0.02
- **num_neighbors** = 20, **num_interpolations** = 2, **resolution** = 0.005, **window_size** = 9

## Enhancing — Parameter-Specific Adaptation Logic

- **upsampling_stage1/2/3_target_distance**: Default 0.06/0.03/0.015. 0.08/0.04/0.02 for CRITICAL-SPARSE. 0.10/0.05/0.025 for extreme LARGE-SCALE (>50% diameter increase). SIMILAR → always defaults.
- **curvature_threshold**: Default 0.005. 0.0003–0.001 for CRITICAL-SPARSE aggressive capture. 0.006–0.008 only for very simple geometry.
- **depth_threshold_low/high**: Default 0.005/0.015; relax for LOW-QUALITY. Higher high threshold → more outliers detected → more gap fill.
- **inter_radius**: Default 0.03. CRITICAL-SPARSE → 0.05–0.08; larger can blur detail.
- **duplicate_threshold**: Default 0.02; range 0.015–0.03 as needed.
- **num_neighbors, num_interpolations**: Default 20, 2 — generally stable.
- **resolution, window_size**: resolution fixed **0.005**. `window_size` default 9; **11–13** for COVERAGE_FAILURE on T3/multi-station tunnels.