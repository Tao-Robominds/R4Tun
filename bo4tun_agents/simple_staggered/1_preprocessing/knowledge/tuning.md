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

### Physical Constants (Engineering Specs)

| Parameter | Description | How to Obtain |
|-----------|-------------|---------------|
| `ring_spacing` | Ring width in meters | Tunnel construction drawings |
| `tunnel_diameter` | Tunnel diameter in meters | Tunnel specifications |

| Parameter | Formula |
|-----------|---------|
| `radius_min` | `tunnel_diameter / 2 - margin` (typically 0.05m margin) |
| `radius_max` | `tunnel_diameter / 2 + margin` (typically 0.05m margin) |

**Critical:** `radius_min < radius_max` must ALWAYS hold.


#### HIGH Sensitivity

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

1. **Set physical constants first** - `ring_spacing`, `tunnel_diameter`
2. **Calculate radius bounds** - `radius_min/max` from diameter
3. **Start with aggressive denoising** - `gradient_threshold=0.1` 
4. **Keep defaults for rest** - preprocessing has minimal impact
5. **Only tune if detection fails** - preprocessing rarely the bottleneck

## Cross-Stage Dependencies

```
Preprocessing outputs used by Detection:
├── tunnel_diameter → k_height_mm, ab_height_mm calculation
├── depth_map_resolution → resolution for pixel↔mm conversion
└── depth_map_outlier.npy → input for line detection
```
