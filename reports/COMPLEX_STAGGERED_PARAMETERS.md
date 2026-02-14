# Complex Staggered Parameters for BO Optimization

This document lists all parameters extracted for complex_staggered detection and SAM operations for tunnels 4-1 and 5-1.

## Detection Parameters (`parameters_detection.json`)

### Standard Detection Parameters (shared with simple_staggered)
- `preprocessing.*`: Edge detection and morphological operations
- `hough_oblique.*`: Initial oblique line detection
- `hough_horizontal.*`: Horizontal line detection
- `hough_vertical.*`: Vertical line detection
- `line_processing.*`: Line merging and filtering

### Complex Staggered Specific Parameters

#### 1. `complex_staggered.hough_re_detect`
Re-detection parameters for wider angle range detection:
- `threshold`: [20-50] (default: 30) - Lower threshold for more lines
- `min_length`: [30-100] (default: 50) - Minimum line length
- `max_gap`: [50-150] (default: 100) - Maximum gap to connect lines

#### 2. `complex_staggered.angle_range`
Wider angle ranges for complex_staggered patterns:
- `positive_min`: [3-6] (default: 4.0) - Minimum positive slope angle
- `positive_max`: [10-15] (default: 12.0) - Maximum positive slope angle
- `negative_min`: [-15 to -10] (default: -12.0) - Minimum negative slope angle
- `negative_max`: [-6 to -3] (default: -4.0) - Maximum negative slope angle

#### 3. `complex_staggered.line_filtering`
Line span filtering:
- `min_y_span`: [20-50] (default: 30) - Minimum vertical span
- `min_x_span`: [20-50] (default: 30) - Minimum horizontal span

#### 4. `complex_staggered.clustering`
DBSCAN clustering parameters:
- `eps_candidates`: List of [0.02-0.20] (default: [0.03, 0.05, 0.08, 0.10, 0.15])
- `min_clusters`: [3-8] (default: 5 for 4-1, 3 for 5-1)
- `subdivision_threshold`: [1.0-2.5] (default: 1.5) - Ring width multiplier for subdivision
- `max_subdivisions`: [2-5] (default: 4 for 4-1, 3 for 5-1)

#### 5. `complex_staggered.confidence`
Confidence score calculation:
- `subdivision_base`: [0.3-0.7] (default: 0.5)
- `subdivision_factor`: [0.02-0.10] (default: 0.05)
- `cluster_base`: [0.3-0.7] (default: 0.5)
- `cluster_factor`: [0.05-0.15] (default: 0.1)
- `midpoint`: [0.5-0.9] (default: 0.7)
- `final_intersection`: [0.7-1.0] (default: 0.9)
- `final_midpoint`: [0.4-0.8] (default: 0.6)

## SAM Parameters (`parameters_sam.json`)

### Standard SAM Parameters
- `segment_geometry.*`: Segment dimensions and angles
- `prompt_points.template_mask.*`: Template mask dimensions

### Complex Staggered Specific Parameters

#### `complex_staggered.template_sizing`
Template size adjustment factors (for BO optimization):
- `k_block_width_factor`: [0.8-1.2] (default: 1.0)
- `k_block_height_factor`: [0.8-1.2] (default: 1.0)
- `ab_block_width_factor`: [0.8-1.2] (default: 1.0)
- `ab_block_height_factor`: [0.8-1.2] (default: 1.0)

## High-Impact Parameters for BO (Recommended Priority)

### Detection (Top Priority)
1. `complex_staggered.hough_re_detect.threshold` - Controls line detection sensitivity
2. `complex_staggered.angle_range.*` - Defines which oblique lines are captured
3. `complex_staggered.clustering.eps_candidates` - Controls intersection clustering
4. `complex_staggered.clustering.subdivision_threshold` - Controls cluster subdivision

### SAM (Secondary Priority)
1. `complex_staggered.template_sizing.*` - Adjusts template mask sizes for better segmentation

## Current Performance

### Tunnel 4-1
- Detection Score: 0.865
- mIoU: 0.424
- OA: 0.621

### Tunnel 5-1
- Detection Score: 0.700
- mIoU: 0.188
- OA: 0.360

## Notes

- All parameters are stored in JSON format for easy BO integration
- Parameters are tunnel-specific (4-1 vs 5-1 may need different values)
- The `complex_staggered` section is only used by `4-1_detection_complex.py`
- Standard detection parameters still apply for initial line detection
