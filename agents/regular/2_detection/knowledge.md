# Detection Stage - Parameter Tuning Guide

## Overview

Detection is the **highest impact stage** for mIoU improvement (+6.3% from BO experiments). Getting these parameters right is critical.

## Parameter Sensitivity Classification

### CRITICAL Parameters (must be exact)

These 4 parameters MUST be correct or detection fails completely (score=0):

| Parameter | Sensitivity | BO Range | Notes |
|-----------|-------------|----------|-------|
| `dilation_iterations` | **CRITICAL** | 1-3 | Tunnel-specific. Wrong value drops score 5-10%+ |
| `dilation_kernel_size` | **CRITICAL** | 2-5 | Tunnel-specific. All top runs converge to specific values |
| `hough_vertical_threshold` | **CRITICAL** | 400-700 | Must be high (>500). Low values cause too many vertical line detections → score=0 |
| `angle_positive_max` | **CRITICAL** | 8.0-9.0 | Tight range required (~8.0-8.7°). Controls oblique line filtering |

### MEDIUM Parameters (can vary)

These parameters are less sensitive and can vary more freely:

| Parameter | BO Range | Notes |
|-----------|----------|-------|
| `binary_threshold` | 80-150 | Edge detection threshold. Wide range works |
| `hough_oblique_threshold` | 30-70 | Line detection votes. Wide range works |
| `angle_positive_min` | 4.0-7.0 | Lower bound for oblique angle. Flexible |
| `hough_oblique_min_length` | 40-120 | Minimum line segment length in pixels |
| `hough_oblique_max_gap` | 30-80 | Maximum gap between line segments |

### Angle Parameters

Negative angles mirror positive angles for the opposite slope direction:
- `angle_negative_min` = -`angle_positive_max`
- `angle_negative_max` = -`angle_positive_min`

## Physical Constants

Physical constants are **NOT defined in detection** - they are automatically read from preprocessing stage:

| Constant | Source | Calculation |
|----------|--------|-------------|
| `tunnel_diameter` | preprocessing params | Direct read |
| `ring_spacing` | preprocessing params | Direct read |
| `resolution` | preprocessing params (`depth_map_resolution`) | Direct read |
| `k_height_mm` | calculated | π × tunnel_diameter × 1000 / 16 |
| `ab_height_mm` | calculated | 3 × k_height_mm |

## Preprocessing Parameters Affecting Detection

Detection uses `depth_map_outlier.npy` from preprocessing. Key preprocessing considerations:

| Parameter | In Preprocessing | Effect on Detection |
|-----------|------------------|---------------------|
| `depth_map_resolution` | Tunable | Directly affects pixel-to-mm conversion in detection |
| `interpolation_window` | Tunable (LOW impact) | **Only affects visualization** (`depth_map.png`), NOT detection input |

**Critical:** The `depth_map_outlier.npy` (detection input) is ALWAYS generated with `window_size=1` (no gap interpolation). This ensures:
- Sparse boundary points for clear line detection
- No filled-in gaps that would confuse Hough transform
- Consistent behavior regardless of `interpolation_window` setting

If detection suddenly produces many false K-points (n_detected > 30) and the depth_map looks "filled in", check that preprocessing is generating `depth_map_outlier.npy` correctly with sparse pixels (~15k-20k valid pixels, not ~400k).

## Why Runs Fail (score=0)

Analysis of failed BO runs shows:

1. **Too many K positions detected** (n_detected > 30)
   - Caused by: Low `hough_vertical_threshold` (<500)
   - Caused by: High `dilation_iterations` (3-4)
   
2. **Wrong K positions** (incorrect Y values)
   - Caused by: `angle_positive_max` outside tight range

## Tuning Strategy

1. **Start with CRITICAL parameters** - get these right first
2. **Use high `hough_vertical_threshold`** - better to detect fewer lines than too many
3. **Keep `angle_positive_max` tight** - around 8.0-8.7°
4. **Tune MEDIUM parameters** only after CRITICAL ones are set
5. **Validate n_detected** - should match expected ring count (±2)

## Parameter Interdependencies

```
dilation_iterations + dilation_kernel_size
    → Control edge dilation before Hough transform
    → Too much dilation = merged lines = wrong detections
    
hough_vertical_threshold
    → Controls number of vertical lines detected
    → Directly affects number of ring centers found
    
angle_positive_max
    → Must match actual K-block oblique angle
    → Too wide = false positive lines
    → Too narrow = missed K-blocks
```

## Comparison with Other Stages

| Stage | mIoU Impact | Tuning Priority |
|-------|-------------|-----------------|
| **Detection** | +6.3% | **1st (Highest)** |
| SAM | +7.4% | 2nd |
| Denoising | +0.1% | 3rd |
| Enhancing | (combined) | 4th |
| Unfolding | +0.0% | 5th (Lowest) |

Detection has the highest single-stage impact - prioritize tuning here.
