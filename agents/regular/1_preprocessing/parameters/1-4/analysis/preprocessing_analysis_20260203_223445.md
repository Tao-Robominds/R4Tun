# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:34:45

---


### 1. ANCHORING
The tunnel’s median radius is 2.5189 m → current radius bounds [2.4689, 2.5689] are very tight. Median NN distance is 0.01935 m → depth_map_resolution≈0.1–0.2×NN = [0.00194, 0.00387]. Density CV=0.438 → mid‐range density.

### 2. CLASSIFICATION
• median_nn_distance_m=0.01935 m (≥0.02 m borderline) → STANDARD regime
• density_cv=0.438 (0.3<CV<0.5) → UNIFORM‐VARIABLE mix; moderate threshold

### 3. DIAGNOSTIC INSPECTION
- pre_theta_coverage_pct=99.99% (OK)
- pre_point_retention_pct=1.42% (<70%) → extreme over‐cropping/over‐denoising
- pre_depth_map_valid_pixels=3 356 (<8 000) → depth map too sparse

### 4. PARAMETER ADAPTATION
1) **Expand radial bounds**: increase margin from ±0.05 m to ±0.10 m to retain more points
   • radius_min=2.5189050626219616–0.10=2.4189050626219616
   • radius_max=2.5189050626219616+0.10=2.6189050626219614
2) **Reduce denoising aggressiveness**: set highest permissible gradient_threshold
   • gradient_threshold=0.4 (gentlest denoising)
3) **Use coarser upsampling scales** to fill sparse map
   • target_distances=[4×NN, 2×NN, 1×NN] → [0.07741203108282903, 0.038706015541414514, 0.019353007770707257]
4) **Depth map resolution** at upper recommended bound
   • depth_map_resolution=0.2×NN=0.0038706015541418513
5) Keep defaults for rest
   • ring_spacing=1.2, tunnel_diameter=5.03781, curvature_neighbors=20, interpolation_window=9

### 5. VALIDATION
- radius_min (2.4189) < radius_max (2.6189): ✔
- target_distances descending: 0.0774 > 0.0387 > 0.0194: ✔
- gradient_threshold=0.4 matches moderate CV

```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.03781,
  "radius_min": 2.4189050626219616,
  "radius_max": 2.6189050626219614,
  "gradient_threshold": 0.4,
  "target_distances": [0.07741203108282903, 0.038706015541414514, 0.019353007770707257],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0038706015541418513,
  "interpolation_window": 9
}
```