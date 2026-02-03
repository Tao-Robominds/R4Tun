# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:28:49

---


1. ANCHORING
- cross_section_radius_m = 2.518905 → radius_min = 2.518905 − 0.05 = 2.468905, radius_max = 2.518905 + 0.05 = 2.568905
- median_nn_distance_m = 0.019353 → default depth_map_resolution ∼0.1–0.2×nn = 0.0019–0.0039; current was 0.0029
- density_cv = 0.438 → mid-range variability; default gradient_threshold ≈0.2

2. CLASSIFICATION
- median_nn < 0.02 m → HIGH-DENSITY regime (finer resolution, tighter target distances)
- density_cv between 0.3 and 0.5 → STANDARD variability

3. DIAGNOSTIC INSPECTION
- pre_theta_coverage_pct = 99.99% (OK)
- pre_point_retention_pct = 1.42% (<70%) → over-aggressive denoising
- pre_depth_map_valid_pixels = 3,356 (<8,000) → too sparse depth map

4. PARAMETER ADAPTATION
- Increase gradient_threshold to reduce denoising aggressiveness. Set from 0.2 to 0.3 (closer to upper BO bound) to boost retention >70%.
- Increase depth_map_resolution from 0.0029 to 0.005 to coarsen the grid and fill more pixels (>8,000).
- Leave ring_spacing, tunnel_diameter, radius_min/max unchanged (they are already correct).
- Keep target_distances at [3×nn, 2×nn, 1×nn] = [0.058059, 0.038706, 0.019353].
- Keep curvature_neighbors at 20 (sufficient for medium variability).
- Keep interpolation_window = 9 (visualization only).

5. VALIDATION
- radius_min < radius_max ✔
- target_distances in descending order ✔
- gradient_threshold = 0.3 matches moderate variability ✔
- Proposed depth_map_resolution (0.005) within BO range [0.003–0.008] ✔

Recommended Parameters:
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.03781,
  "radius_min": 2.468905,
  "radius_max": 2.568905,
  "gradient_threshold": 0.3,
  "target_distances": [0.058059, 0.038706, 0.019353],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.005,
  "interpolation_window": 9
}
```