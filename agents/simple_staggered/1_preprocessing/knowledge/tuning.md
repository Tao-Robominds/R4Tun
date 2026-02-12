# Preprocessing Stage - Parameter Tuning Guide

## Overview

Preprocessing is the **lowest impact stage** for mIoU improvement (+0.1% from BO experiments). Detection and SAM stages have far greater impact. However, correct preprocessing is still necessary as a foundation.

## Impact by Sub-Stage

| Sub-Stage | mIoU Impact | Sensitivity | Notes |
|-----------|-------------|-------------|-------|
| Unfolding | +0.0% | LOW | Defaults already optimal. Score variance only 0.17 across all BO runs |
| Denoising | +0.1% | MEDIUM | `gradient_threshold` matters most |
| Enhancing | (combined) | LOW-MEDIUM | Mostly fixed defaults work well |

## Parameter Classification

### CRITICAL Sensitivity (Tunnel Geometry)

These parameters are **MOST CRITICAL** for preprocessing success. The `extract_characteristics.py` script can produce inaccurate measurements, so these require empirical tuning based on intrinsics feedback.

| Parameter | Description | Initial Source | Tuning Range |
|-----------|-------------|----------------|--------------|
| `tunnel_diameter` | Tunnel diameter in meters | Tunnel specs OR `2 × cross_section_radius_m` | ±0.5m from initial |
| `ring_spacing` | Ring width in meters | Tunnel construction drawings | ±0.2m from initial |

**Derived Parameters:**

| Parameter | Formula |
|-----------|---------|
| `radius_min` | `tunnel_diameter / 2 - margin` (typically 0.05-0.1m margin) |
| `radius_max` | `tunnel_diameter / 2 + margin` (typically 0.05-0.1m margin) |

**Critical Constraints:**
- `radius_min < radius_max` must ALWAYS hold
- If `pre_point_retention_pct` drops dramatically (<30%), the radius bounds are likely wrong

#### Tuning Strategy for tunnel_diameter / radius bounds

| Symptom | Cause | Action |
|---------|-------|--------|
| `pre_point_retention_pct` < 30% | radius bounds too tight | **Increase** `tunnel_diameter` by 0.1-0.2m |
| `pre_point_retention_pct` > 98% | radius bounds too loose (noise not filtered) | **Decrease** `tunnel_diameter` by 0.1m |
| Depth map looks distorted | Wrong tunnel geometry | Try opposite direction |

**Example from experience:**
- Characteristics measured `cross_section_radius_m = 2.52m` → calculated `tunnel_diameter = 5.04m`
- Result: `pre_point_retention_pct = 1.4%` (FAILURE - too tight!)
- Fix: Increased to `tunnel_diameter = 5.54m` → `pre_point_retention_pct = 72.9%` (SUCCESS)

#### Tuning Strategy for ring_spacing

| Symptom | Cause | Action |
|---------|-------|--------|
| Wrong number of rings detected | ring_spacing mismatch | Adjust by ±0.1m increments |
| Unfolding artifacts | Geometry mismatch | Try opposite direction |

### HIGH Sensitivity

| Parameter | BO Range | Notes |
|-----------|----------|-------|
| `gradient_threshold` | 0.1 - 0.4 | Lower = more aggressive denoising. BO found lower bound often best |
| `depth_map_resolution` | 0.003 - 0.008 | Affects all downstream stages |

#### MEDIUM Sensitivity

| Parameter | BO Range | Notes |
|-----------|----------|-------|
| `curvature_neighbors` | 15 - 30 | K neighbors for curvature estimation |
| `target_distances` | Multi-scale | Progressive upsampling levels (e.g., [0.08, 0.04, 0.02]) |

#### LOW Sensitivity (Use Defaults)

These parameters showed minimal impact in BO - use fixed defaults:

| Parameter | Default | BO Range | Notes |
|-----------|---------|----------|-------|
| `interpolation_window` | 9 | 3-7 | Gap filling window for main depth_map visualization. BO found 6 optimal for 2-2. Does NOT affect detection (depth_map_outlier always uses window=1) |
| `theta_step` | 0.5 | - | Angular resolution for denoising |
| `radial_step` | 0.001 | - | Radial resolution for denoising |
| `smoothing_window` | 3 | - | Boundary smoothing |
| Unfolding params | Various | - | All showed <0.17 score variance |

**Important:** `interpolation_window` affects only the main `depth_map.png` (visualization), NOT `depth_map_outlier.npy` (detection input). The outlier depth map always uses `window_size=1` (no gap filling) which is critical for detection to work correctly.

## Tuning Strategy

1. **Start with initial geometry estimates** - `ring_spacing`, `tunnel_diameter` from specs or characteristics
2. **Calculate radius bounds** - `radius_min/max` from diameter with appropriate margin
3. **Run preprocessing and check intrinsics** - especially `pre_point_retention_pct`
4. **If retention is bad (<30% or >98%)** - adjust `tunnel_diameter` in opposite direction
5. **Iterate until intrinsics pass** - target 70-90% retention, 8k-35k valid pixels
6. **Then tune other parameters** - `gradient_threshold`, `depth_map_resolution` only if needed

**Key Insight:** The characteristics extraction can be inaccurate due to point cloud offset from true tunnel center. Always validate with intrinsics feedback and be prepared to override calculated values.

## Cross-Stage Dependencies

```
Preprocessing outputs used by Detection:
├── tunnel_diameter → k_height_mm, ab_height_mm calculation
├── depth_map_resolution → resolution for pixel↔mm conversion
└── depth_map_outlier.npy → input for line detection
```
