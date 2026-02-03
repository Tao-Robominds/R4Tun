# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 23:27:48

---


Anchoring:
- Measured cross_section_radius_m=2.77 → radius_min=2.72, radius_max=2.82.
- median_nn_distance_m=0.01935 → target_distances=[0.058059, 0.038706, 0.019353], depth_map_resolution kept at 0.0019353.
- density_cv=0.438 → gradient_threshold=0.2, curvature_neighbors=20.

Classification:
- median_nn < 0.02 m → HIGH-DENSITY regime (use finer resolution, smaller target_distances).

Diagnostic Inspection:
- pre_theta_coverage_pct=99.99% is within 98–102% → unfolding OK.
- pre_point_retention_pct=1.54% (<70%) → over-aggressive denoising due to too tight radius bounds.
- pre_depth_map_valid_pixels=3736 (<8000) → depth map too sparse.

Parameter Adaptation:
- Increase tunnel_diameter from 5.04 to 5.54 m to widen radius bounds to [2.72, 2.82], as in prior successful run.
- Keep ring_spacing=1.2, gradient_threshold=0.2, target_distances=[0.0580590233,0.0387060155,0.0193530078], curvature_neighbors=20, depth_map_resolution=0.0019353008, interpolation_window=9.

Validation:
- radius_min (2.72) < radius_max (2.82).
- target_distances descending from 0.058059 to 0.019353.
- Parameters match density and spacing profiles.

Final Parameters:
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