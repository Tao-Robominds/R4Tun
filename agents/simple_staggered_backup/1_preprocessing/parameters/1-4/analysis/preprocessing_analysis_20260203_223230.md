# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:32:30

---


1. ANCHORING
– cross_section_radius_m = 2.5189050626219616 → radius_min = 2.4689050626219616, radius_max = 2.5689050626219616
– median_nn_distance_m = 0.019353007770707257 → depth_map_resolution should be 0.1–0.2×nn (≈0.0019–0.0039)
– density_cv = 0.43814102923872605 → gradient_threshold mid‐range ≈0.2

2. CLASSIFICATION
– median_nn < 0.02 m → high‐density regime (requires finer resolution)
– density_cv ≈0.44 → moderate variation (standard denoising aggressiveness)

3. DIAGNOSTIC INSPECTION
– pre_theta_coverage_pct = 99.99% (OK)
– pre_point_retention_pct = 1.67% (<70%) → over‐aggressive denoising
– pre_depth_map_valid_pixels = 3324 (<8000) → depth map too sparse

4. PARAMETER ADAPTATION
– gradient_threshold: lower from 0.40 to 0.20 to retain more points (baseline memory shows 0.20→73% retention)
– depth_map_resolution: refine from 0.0038706015541418513 to 0.0029 (≈0.15×nn) to increase valid pixels
– radius_min / radius_max unchanged (2.4689050626219616 / 2.5689050626219616)
– target_distances unchanged ([0.05805902331212177, 0.038706015541414514, 0.019353007770707257])
– curvature_neighbors unchanged (20)
– interpolation_window unchanged (9)

5. VALIDATION
– radius_min < radius_max (2.4689 < 2.5689)
– target_distances descending order
– gradient_threshold matches density_cv
– depth_map_resolution in allowed range

Final tuned parameters:
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.03781,
  "radius_min": 2.4689050626219616,
  "radius_max": 2.5689050626219616,
  "gradient_threshold": 0.2,
  "target_distances": [0.05805902331212177, 0.038706015541414514, 0.019353007770707257],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0029,
  "interpolation_window": 9
}