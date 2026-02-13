# Simplified Detection Pipeline for Bayesian Optimization

This module detects oblique lines, finds intersections, and calculates K-block positions with **only critical parameters** exposed for experimentation.

## Key Finding

**Detection provides +6.3% mIoU improvement - the LARGEST single-stage gain.**

This is the most impactful stage to tune. Poor detection cannot be compensated by downstream stages.

## Critical Parameters (14 total)

### HIGH Sensitivity (tune first)

| Parameter | BO Range | Description |
|-----------|----------|-------------|
| `binary_threshold` | 100-150 | Edge detection threshold |
| `hough_oblique_threshold` | 30-80 | Oblique line detection votes |
| `angle_positive_min` | 4.0-7.0° | Min positive slope angle |
| `angle_positive_max` | 8.0-12.0° | Max positive slope angle |
| `angle_negative_min` | -12.0 to -8.0° | Min negative slope angle |
| `angle_negative_max` | -7.0 to -4.0° | Max negative slope angle |
| `hough_vertical_threshold` | 400-700 | Ring boundary detection |

### MEDIUM Sensitivity (tune if needed)

| Parameter | BO Range | Description |
|-----------|----------|-------------|
| `dilation_kernel_size` | 3-7 | Morphological kernel size |
| `dilation_iterations` | 1-3 | Dilation passes |
| `hough_oblique_min_length` | 60-150 | Min line length (px) |
| `hough_oblique_max_gap` | 20-80 | Max gap between segments |

### Physical Constants (READ from preprocessing stage)

These values are **not defined** in detection parameters - they are automatically read from `1_preprocessing/parameters/{tunnel_id}/parameters_preprocessing.json`:

| Parameter | Source | Formula |
|-----------|--------|---------|
| `tunnel_diameter` | preprocessing | Direct read |
| `depth_map_resolution` | preprocessing | Direct read (as `resolution`) |
| `k_height_mm` | calculated | `π * tunnel_diameter * 1000 / 16` |
| `ab_height_mm` | calculated | `3 * k_height_mm` |

**No duplication!** Physical constants are defined once in preprocessing and inherited by detection.

## Fixed Parameters (Non-Critical)

- `hough_horizontal_*`: Fixed (used for visualization only)
- `merge_distance_threshold`: 3
- `merge_close_threshold`: 6

## BO Optimization Results

| Tunnel | binary_threshold | hough_oblique | angle_min | angle_max | vertical |
|--------|------------------|---------------|-----------|-----------|----------|
| 1-4 | 119 | 73 | 6.0° | 9.0° | 500 |
| 2-2 | 149 | 69 | 5.509° | 8.652° | 700 |

## Usage

```bash
# Run detection on tunnel 1-4
python 2_detection.py 1-4

# Run detection on tunnel 2-2
python 2_detection.py 2-2
```

## Parameters File Structure

```
parameters/
├── sample/
│   └── parameters_detection.json   # Template with documentation
├── 1-4/
│   └── parameters_detection.json   # Tunnel 1-4 (BO optimized)
└── 2-2/
    └── parameters_detection.json   # Tunnel 2-2 (BO optimized)
```

## Example Parameters File

```json
{
    "binary_threshold": 127,
    "hough_oblique_threshold": 50,
    "angle_positive_min": 6.0,
    "angle_positive_max": 9.0,
    "angle_negative_min": -9.0,
    "angle_negative_max": -6.0,
    "hough_vertical_threshold": 500,
    "k_height_mm": 1079.92,
    "ab_height_mm": 3239.77,
    "resolution": 0.005
}
```

## Outputs

| File | Description |
|------|-------------|
| `detected.csv` | K-block positions (Type, X, Y) for SAM |
| `detected_lines.png` | Visualization of detected lines |

## Detection Types

The `Type` column in `detected.csv` indicates how each K-position was determined:

- `midpoint`: Both positive and negative slope lines intersected (most reliable)
- `positive_slope`: Only positive slope line found, offset applied
- `negative_slope`: Only negative slope line found, offset applied
- `assume`: No lines found, position inferred from alternation pattern
- `default`: Fallback to image center

**Goal:** Maximize `midpoint` detections for best accuracy.

## Dependencies

Requires outputs from `1_preprocessing`:
- `depth_map_outlier.npy`: Enhanced depth map
- `ring_count.txt`: Number of rings

## References

- `/reports/P4TUN_OPTIMIZATION_JOURNEY_2-2.md` - Detection BO findings (+6.3% improvement)
- `/reports/P4TUN_OPTIMIZATION_JOURNEY_3-1_1-4.md` - Multi-tunnel optimization
