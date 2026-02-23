# Targeted Detection BO Results (Round 3)

## Summary

Targeted BO runs completed with widened bounds and reduced search space (29D → 21D).

| Tunnel | Round 2 Best | Round 3 Best | Improvement | Status |
|--------|-------------|-------------|-------------|--------|
| **4-1** | 0.5909 | 0.5909 | 0.0000 | No improvement |
| **5-1** | 0.6810 | **0.6902** | +0.0092 | Improved |

## Detailed Results

### 4-1 (complex_staggered)
- **Round 2**: Best wF1=0.5909 at eval 133
- **Round 3**: Best wF1=0.5909 (warm-start still best)
- **New best found**: 0.5322 at eval 479 (TP=5, FP=4, FN=0, mean_dist=76.5px)
- **Status**: No improvement, but found alternative solution with perfect recall
- **Note**: The new solution has lower precision (5 TP vs 4 FP) but perfect recall (FN=0). The warm-start solution remains better due to position bonus.

### 5-1 (complex_staggered)
- **Round 2**: Best wF1=0.6810 at eval 62
- **Round 3**: Best wF1=0.6902 at eval 243 (new best found)
- **Improvement**: +0.0092 (+1.4%)
- **Status**: Modest but meaningful improvement
- **Details**: TP=5, FP=2, FN=1, mean_dist=30.8px (much better than previous 39.5px)

## Key Changes Made

### 1. Widened BO Bounds
- `complex_min_y_span`: (20, 50) → **(5, 50)** - allows shorter lines
- `complex_min_x_span`: (20, 50) → **(5, 50)** - allows shorter lines
- `complex_hough_threshold`: (20, 50) → **(10, 50)** - more sensitive detection
- `complex_hough_min_length`: (30, 100) → **(15, 100)** - shorter lines allowed

### 2. Reduced Search Space (29D → 21D)
**Fixed parameters:**
- `complex_max_subdivisions`: 3 (both tunnels)
- `complex_conf_midpoint`: tunnel-specific
- `complex_conf_intersection`: tunnel-specific
- `complex_subdivision_threshold`: tunnel-specific
- `hough_vertical_threshold`: tunnel-specific
- `merge_distance_threshold`: tunnel-specific
- `dilation_kernel_size`: tunnel-specific
- `hough_horizontal_min_length`: tunnel-specific

**Narrowed bounds** around best-found regions for remaining 21 parameters.

## Analysis

### 4-1: Still Struggling
- The core issue remains: **insufficient oblique lines** (only 32 passing filters vs 181 for 5-1)
- Even with widened bounds (min_y_span down to 5), the algorithm still struggles
- The new solution at eval 479 shows perfect recall but lower precision
- **Recommendation**: May need algorithm-level changes or different approach for 4-1

### 5-1: Steady Improvement
- Consistent improvement across rounds (0.6810 → 0.6902)
- Better position accuracy (mean_dist: 39.5px → 30.8px)
- Still has 1 FN and 2 FP, but overall performance is good
- **Recommendation**: May benefit from more calls or further bound refinement

## Next Steps

1. **4-1**: Consider algorithm-level investigation - why are oblique lines so sparse?
2. **5-1**: Run additional BO calls to see if further improvement is possible
3. **Both**: Consider running detection and evaluating on actual images to see visual improvements

## Files Modified

- `bo/complex_staggered/run_detection_bo.py`: Widened bounds for span filters
- `bo/complex_staggered/configs/detect_4-1.json`: Created with fixed params and narrowed bounds
- `bo/complex_staggered/configs/detect_5-1.json`: Created with fixed params and narrowed bounds
- `agents/complex_staggered/2_detection/parameters/4-1/parameters_detection.json`: Updated (no change)
- `agents/complex_staggered/2_detection/parameters/5-1/parameters_detection.json`: Updated with new best
