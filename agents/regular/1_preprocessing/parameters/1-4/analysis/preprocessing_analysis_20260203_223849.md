# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:38:49

---


1. ANCHORING:
• cross_section_radius_m = 2.5189050626219616 ⇒ radius_min = 2.4689050626219616, radius_max = 2.5689050626219616
• median_nn_distance_m = 0.019353007770707257 ⇒ depth_map_resolution ≈ 0.15×nn = 0.0029029511656060886 ≈ 0.0029
• density_cv = 0.43814102923872605 ⇒ moderate variation ⇒ gradient_threshold ≈ 0.2

2. CLASSIFICATION:
• median_nn = 0.01935 m (0.02 > nn > 0.02?) borderline high-density regime but density_cv moderate ⇒ treat as STANDARD regime

3. DIAGNOSTIC INSPECTION:
• pre_theta_coverage_pct = 99.99% within [98–102%]
• pre_point_retention_pct = 3.9% (<70% ⇒ over-aggressive denoising or incorrect radius bounds)
• pre_depth_map_valid_pixels = 9154 (>8000 OK)

Root cause: radius_min/max (2.7–2.8 m) do not align with actual radius ≈2.52 m, so almost all points were discarded. Fixing radius bounds will restore retention.

4. PARAMETER ADAPTATION:
• ring_spacing: keep 1.2 (engineering spec)
• tunnel_diameter: set to 2×2.5189050626 = 5.037810125243923
• radius_min = 2.4689050626219616, radius_max = 2.5689050626219616 (tight ±0.05 m around actual)
• gradient_threshold = 0.2 (matches moderate density_cv)
• target_distances = [3×nn, 2×nn, 1×nn] = [0.05805902331212177, 0.038706015541414514, 0.019353007770707257]
• curvature_neighbors = 20 (default, sufficient for moderate density)
• depth_map_resolution = 0.0029 (0.15×nn)
• interpolation_window = 9 (default for visual depth map)

5. VALIDATION:
• radius_min < radius_max ✓
• target_distances in descending order ✓
• gradient_threshold matches density profile ✓

Expected outcome: pre_point_retention_pct within [70–98%], pre_depth_map_valid_pixels ≳ 14328, pre_theta_coverage_pct ≈ 100%, pre_ready_for_detection → true

Final parameters:
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.037810125243923,
  "radius_min": 2.4689050626219616,
  "radius_max": 2.5689050626219616,
  "gradient_threshold": 0.2,
  "target_distances": [0.05805902331212177, 0.038706015541414514, 0.019353007770707257],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0029,
  "interpolation_window": 9
}
```