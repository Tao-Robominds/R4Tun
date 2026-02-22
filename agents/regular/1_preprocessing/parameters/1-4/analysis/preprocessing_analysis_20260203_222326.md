# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:23:26

---


Anchoring:
The measured cross_section_radius_m is 2.5189050626219616, implying radius_min = 2.5189050626219616 - 0.05 = 2.4689050626219616, radius_max = 2.5189050626219616 + 0.05 = 2.5689050626219616. The median_nn_distance_m is 0.019353007770707257, density_cv is 0.43814102923872605.

Classification:
Median nearest‐neighbor spacing (0.01935 m) is below 0.02 m → HIGH‐DENSITY regime; density_cv = 0.438 is between 0.3 and 0.5 → STANDARD denoising strength.

Diagnostic Inspection:
All intrinsics are within target: pre_theta_coverage_pct = 99.99% (ideal 98–102%), pre_point_retention_pct = 73.10% (70–98%), pre_depth_map_valid_pixels = 14,328 (8k–35k). No preprocessing issues detected.

Parameter Adaptation:
Since intrinsics are already good, only adjust radius bounds to match the actual cross‐section. Keep aggressive denoising (gradient_threshold = 0.2) and standard enhancing settings. Recalculate diameter from measured radius: tunnel_diameter = 2 × 2.5189050626219616 = 5.037810125243923.

Validation:
radius_min (2.4689050626219616) < radius_max (2.5689050626219616); target_distances descending; gradient_threshold matches density profile.

Final Parameters:
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.037810125243923,
  "radius_min": 2.4689050626219616,
  "radius_max": 2.5689050626219616,
  "gradient_threshold": 0.2,
  "target_distances": [0.07741203108282903, 0.038706015541414514, 0.019353007770707257],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.005,
  "interpolation_window": 9
}
```