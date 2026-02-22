# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:41:43

---


Analysis:

1. ANCHORING
- cross_section_radius_m = 2.5189050626219616 → radius_min ≈ 2.5189050626219616 − 0.1 = 2.4189050626219616, radius_max ≈ 2.5189050626219616 + 0.1 = 2.6189050626219616 (we widen by 0.1m to recover lost points).
- median_nn_distance_m = 0.019353007770707257 → depth_map_resolution target = 0.2 × 0.019353007770707257 = 0.0038706015541414514.
- target_distances base = [4×nn, 2×nn, 1×nn] = [0.07741203108282903, 0.038706015541414514, 0.019353007770707257].
- density_cv = 0.43814102923872605 → moderate, low variability, so a gentler denoising threshold (max allowed).

2. CLASSIFICATION
- median_nn_distance_m < 0.02 m → HIGH-DENSITY regime (needs fine resolution).
- density_cv < 0.5 → UNIFORM-DENSITY (can tolerate higher gradient_threshold).

3. DIAGNOSTIC INSPECTION
- pre_theta_coverage_pct = 99.994% within [98–102%] → unfolding OK.
- pre_point_retention_pct = 1.42% < 70% → over‐aggressive denoising.
- pre_depth_map_valid_pixels = 3,356 < 8,000 → too sparse.

4. PARAMETER ADAPTATION
- Increase gradient_threshold to 0.4 (max) to preserve more points.
- Increase curvature_neighbors to 30 (more smoothing of density gradient).
- Widen radius bounds to [2.4189050626219616, 2.6189050626219616] to include marginal wall points.
- Use depth_map_resolution = 0.0038706015541414514 for high-density sampling.
- Use target_distances = [0.07741203108282903, 0.038706015541414514, 0.019353007770707257].
- Retain ring_spacing = 1.2, tunnel_diameter = 5.037810125243923, interpolation_window = 9.

5. VALIDATION
- radius_min < radius_max: 2.4189 < 2.6189 ✓
- target_distances descending order ✓
- gradient_threshold = 0.4 matches uniform-density recommendation ✓

Recommended Parameters:
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.037810125243923,
  "radius_min": 2.4189050626219616,
  "radius_max": 2.6189050626219616,
  "gradient_threshold": 0.4,
  "target_distances": [0.07741203108282903, 0.038706015541414514, 0.019353007770707257],
  "curvature_neighbors": 30,
  "depth_map_resolution": 0.0038706015541414514,
  "interpolation_window": 9
}
```