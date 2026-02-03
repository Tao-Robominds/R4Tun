# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:58:04

---


ANCHORING:
The tunnel has cross_section_radius_m=2.77 m, median_nn_distance_m=0.01935 m, density_cv=0.438.  Derived:
  • radius_min = 2.77 - 0.05 = 2.72 m
  • radius_max = 2.77 + 0.05 = 2.82 m
  • depth_map_resolution ≈ 0.1×0.01935 = 0.0019353008 m
  • target_distances baseline = [4×nn, 2×nn, 1×nn] = [0.07741, 0.03871, 0.01935]

CLASSIFICATION:
median_nn < 0.02 → HIGH-DENSITY regime ⇒ use finer resolution and smaller target distances.

diagnostics:
pre_point_retention_pct = 1.42% (<70%) and pre_depth_map_valid_pixels = 3356 (<8000) → denoising too aggressive or mis-configured.

PARAMETER ADAPTATION:
  • ring_spacing remains 1.2 m (engineering spec)
  • tunnel_diameter = 2×2.77 = 5.54 m (update to match measured radius)
  • radius_min = 2.72 m, radius_max = 2.82 m (tight bounds)
  • gradient_threshold = 0.2  (moderate for density_cv≈0.44, less aggressive than current 0.4)
  • depth_map_resolution = 0.0019353008 m (finer grid for high-density)
  • target_distances = [0.0580590233, 0.0387060155, 0.0193530078]  (reduce coarse scale to 3×nn for high-density)
  • curvature_neighbors = 20  (default for medium density)
  • interpolation_window = 9  (default)

VALIDATION:
  • radius_min < radius_max ✓
  • target_distances in descending order ✓
  • gradient_threshold aligned with density profile ✓

RECOMMENDED PARAMETERS:
```json
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
```