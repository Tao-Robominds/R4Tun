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

## Parameter Reference (Unfolding Stage)

The configurable parameters in `configurable/*/parameters_unfolding.json` drive the geometry recovery in stage 1. Use the following definitions and empirically validated ranges when setting up a new tunnel:

- **delta (m)** – half thickness of each slicing slab used to sample rings. Smaller deltas preserve detail but increase noise sensitivity. Typical range: **0.005–0.010** (5.5 m tunnels stay near 0.005–0.007; 7.5 m cases can tolerate up to 0.010).
- **slice_spacing_factor (m)** – target spacing between consecutive slicing planes measured along the tunnel axis. Drives how many rings are generated. Use **0.8** for dense sampling (often 7.5 m tunnels) and **1.2** when 1.2 m rings are expected.
- **vertical_filter_window (m)** – window applied when filtering ellipse points in the projected plane. Keeps only the top portion of each slice to avoid floor clutter. Values between **4.5–7.5** cover T1–T5 (wider windows for large-diameter scans).
- **ransac_threshold (m)** – max distance from an ellipse fit before a point is treated as an outlier. Tight scans (T3) work with **0.5–1.0**; noisier or offset scans (T4/T5) need **1.0–1.5**.
- **ransac_probability** – probability of finding a valid model during ellipse fitting. Held at **0.9** (tuning is rarely required).
- **ransac_inlier_ratio** – expected inlier fraction for ellipse fits. Defaults to **0.75**; reduce only if slices are severely occluded.
- **ransac_sample_size** – number of points sampled per RANSAC iteration. **5–6** points are sufficient for the observed tunnels (6 for wider rings).
- **polynomial_degree** – degree of the polynomial used to fit the 3‑D centerline. Use **3** when the tunnel has noticeable curvature (T1/T2) and **2** for mostly straight runs (T3–T5).
- **num_samples_factor** – oversampling factor for resampling along the centerline. Current datasets operate well around **1 200 ± 200**; increasing it adds computation without noticeable benefit.
- **diameter (m)** – expected tunnel diameter used both for polar conversions and downstream sanity checks. Set **5.5** for T1–T3 data and **7.5** for T4/T5. Keep denoising `default_cutoff_z ≈ diameter / 2` in sync with this value.