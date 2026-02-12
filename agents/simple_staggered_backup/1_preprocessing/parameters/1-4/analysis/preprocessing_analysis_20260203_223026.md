# Preprocessing Analysis - 1-4
**Generated**: 2026-02-03 22:30:26

---


ANCHORING:
The tunnel median radius is 2.5189 m → radius_min=2.468905 m, radius_max=2.568905 m.  The median NN distance is 0.019353 m → depth_map_resolution≈0.2×NN=0.0038706 m; target_distances=[3×NN,2×NN,1×NN]=[0.0580590,0.0387060,0.0193530].  Density CV=0.438 suggests a standard regime (moderate variability).

CLASSIFICATION:
Density CV between 0.3 and 0.5 → STANDARD.  No need for extreme threshold adjustments based on CV alone.

DIAGNOSTIC INSPECTION:
pre_theta_coverage_pct=99.99% (within 98–102% → OK).  pre_point_retention_pct=1.49% (<70% → denoising too aggressive).  pre_depth_map_valid_pixels=3205 (<8000 → too sparse).  Both failures point to over‐aggressive denoising removing nearly all points.

PARAMETER ADAPTATION:
To retain more points, increase gradient_threshold (higher threshold = gentler denoising).  Set gradient_threshold=0.40 (maximum BO range).  Other parameters recalculated from tunnel specs:
- radius_min=2.4689050626219616 (2.5189050626219616−0.05)
- radius_max=2.5689050626219616 (2.5189050626219616+0.05)
- depth_map_resolution=0.0038706 (0.2×0.019353)
- target_distances=[0.0580590, 0.0387060, 0.0193530]
- curvature_neighbors=20 (standard density)
- ring_spacing=1.2 (spec)
- tunnel_diameter=5.03781 (spec)
- interpolation_window=9 (default)

VALIDATION:
radius_min<radius_max ✓; target_distances in descending order ✓; gradient_threshold aligned with goal of gentler denoising ✓.  This adjustment addresses the extreme point loss and sparse depth map.

RECOMMENDED PARAMETERS:
```json
{
  "ring_spacing": 1.2,
  "tunnel_diameter": 5.03781,
  "radius_min": 2.4689050626219616,
  "radius_max": 2.5689050626219616,
  "gradient_threshold": 0.40,
  "target_distances": [0.05805902331212177, 0.038706015541414514, 0.019353007770707257],
  "curvature_neighbors": 20,
  "depth_map_resolution": 0.0038706015541418513,
  "interpolation_window": 9
}
```