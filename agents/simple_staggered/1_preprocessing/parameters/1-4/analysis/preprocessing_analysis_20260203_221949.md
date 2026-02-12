# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:19:49

---


1. ANCHORING:
- cross_section_radius_m = 2.518905 → recommended radius_min = 2.518905 - 0.05 = 2.468905, radius_max = 2.518905 + 0.05 = 2.568905. Current [2.7, 2.8] is too loose relative to measured radius.
- median_nn_distance_m = 0.019353 → depth_map_resolution should be 0.1–0.2×0.019353 = [0.001935, 0.003870]. Selecting mid‐range 0.002900 for balance of detail and compute.
- target_distances based on 2×nn = 0.038706, 3×nn = 0.058059 → use [0.058059, 0.038706, 0.019353]. Current [0.08, 0.04, 0.02] is slightly coarse on the top scale.
- density_cv = 0.438 → standard density regime. gradient_threshold = 0.2 is appropriate.

2. CLASSIFICATION:
- STANDARD regime (median_nn in typical range, density_cv between 0.3 and 0.5).

3. DIAGNOSTIC INSPECTION:
- pre_theta_coverage_pct = 99.99% (within 98–102%).
- pre_point_retention_pct = 73.10% (within 70–98%).
- pre_depth_map_valid_pixels = 14328 (within 8000–35000).
→ No intrinsics failures detected; minimal adjustments needed.

4. PARAMETER ADAPTATION:
- Update physical spec of tunnel_diameter to measured 2×2.518905 = 5.037810.
- Set radius_min = 2.468905, radius_max = 2.568905 (must satisfy radius_min < radius_max).
- Keep gradient_threshold = 0.2, curvature_neighbors = 20 (defaults are appropriate).
- Adjust depth_map_resolution to 0.002900 (within calculated band).
- Adjust target_distances to [0.058059, 0.038706, 0.019353] in descending order.
- Keep interpolation_window = 9 (default, low sensitivity).

5. VALIDATION:
- All constraints satisfied (radius_min < radius_max; target_distances descending; gradient_threshold matches density profile).
- No intrinsics warnings; changes are minimal and quantitatively justified.

Final recommended parameters:
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.037810,
  "radius_min": 2.468905,
  "radius_max": 2.568905,
  "gradient_threshold": 0.2,
  "target_distances": [0.058059, 0.038706, 0.019353],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.002900,
  "interpolation_window": 9
}
```