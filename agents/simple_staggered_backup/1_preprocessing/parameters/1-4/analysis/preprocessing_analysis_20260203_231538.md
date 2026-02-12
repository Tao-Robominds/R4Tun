# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 23:15:38

---


ANCHORING:
- cross_section_radius_m = 2.77 → radius_min = 2.72, radius_max = 2.82 (matches current)
- median_nn_distance_m = 0.01935 → depth_map_resolution = 0.1×nn = 0.001935, target_distances = [3×nn, 2×nn, 1×nn] = [0.05806, 0.03871, 0.01935]
- density_cv = 0.438 → gradient_threshold ≈ 0.2, curvature_neighbors = 20

CLASSIFICATION:
- median_nn < 0.02 → HIGH-DENSITY regime (fine resolution)
- density_cv between 0.3 and 0.5 → STANDARD denoising aggressiveness

DIAGNOSTIC INSPECTION:
- pre_theta_coverage_pct = 99.99% (OK)
- pre_point_retention_pct = 72.92% (within 70–98%)
- pre_depth_map_valid_pixels = 15198 (within 8000–35000)
- All intrinsics pass → pre_ready_for_detection = true

PARAMETER ADAPTATION:
- Intrinsics are already in spec; no adjustments needed.

VALIDATION:
- radius_min < radius_max ✓
- target_distances descending ✓
- gradient_threshold matches density profile ✓

FINAL PARAMETERS:
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.54,
  "radius_min": 2.72,
  "radius_max": 2.82,
  "gradient_threshold": 0.2,
  "target_distances": [0.0580590233, 0.0387060155, 0.0193530078],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0019353008,
  "interpolation_window": 9
}