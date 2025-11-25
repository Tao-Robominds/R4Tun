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

## Parameter Reference (Enhancing Stage)

The upsampling / interpolation stage (Algorithm 3) uses `configurable/*/parameters_enhancing.json`. Each field has a direct effect on point synthesis quality:

- **upsampling_stage{1,2,3}_target_distance (m)** – desired spacing between neighbors after each pass. Start near **0.08 / 0.04 / 0.02** for 5.5 m tunnels and reduce by ~25% for clean data; larger diameters can increase stage 1 to **0.10** to limit runtime.
- **curvature_threshold** – max acceptable curvature difference between neighbors when deciding to interpolate. Tight tunnels use **3e‑4–5e‑4**; rougher scans (T3/T4) can go up to **5e‑3**.
- **depth_threshold_low / depth_threshold_high (m)** – intensity of radial deviation required to mark “meaningful” outliers in low- vs high-density sections. Empirical ranges: **0.003–0.006** (low) and **0.008–0.015** (high).
- **inter_radius (m)** – search radius when picking outlier pairs for joint enhancement. Values between **0.03–0.08** cover all tunnels (shorter for dense stations, longer for sparse large-diameter scans).
- **duplicate_threshold (m)** – minimum spacing between newly generated points (default **0.02**). Increase slightly if you observe overlapping interpolations in T4/T5.
- **n_segment_start / n_segment_end** – defines the high-density window (in ring indices) around the scanner to apply stricter thresholds. Use **0–5** when the scanner sits near the first ring, up to **10–21** when the scanner is embedded deeper (T4/T5).
- **num_neighbors** – number of neighbors queried in KDTree lookups, typically **20**. Raising it increases smoothing but costs time.
- **num_interpolations** – number of points inserted per qualifying pair (usually **2**).
- **resolution (m)** – target grid resolution when projecting to depth maps; the pipeline assumes **0.005** and downstream SAM processing expects the same.
- **window_size (px)** – sliding window for filling missing pixels during projection. Choose **5** for dense data and **9** when large gaps exist (e.g., 7.5 m tunnels).