# Bayesian Optimization Report: Tunnel 2-2

**Date:** 2026-01-22  
**Tunnel ID:** 2-2  
**Optimization Framework:** scikit-optimize (Gaussian Process)  
**Total Iterations:** 90 (30 SAM + 30 Detection + 30 Expanded SAM)

---

## Executive Summary

Bayesian Optimization successfully improved tunnel 2-2 segmentation performance from a baseline **mIoU of 0.672** to a final **mIoU of 0.768**, achieving a **+14.3% total improvement**. The optimization was conducted in three phases, progressively expanding the parameter search space.

---

## Optimization Phases

### Phase 1: SAM Parameter Tuning (21 parameters)
- **Iterations:** 30
- **Baseline:** 0.672
- **Result:** 0.700
- **Improvement:** +4.2%

Initial optimization focused on prompt point positions, template mask dimensions, and processing parameters.

### Phase 2: Detection Parameter Tuning (14 parameters)  
- **Iterations:** 30
- **Baseline:** 0.700
- **Result:** 0.744
- **Improvement:** +6.3%

Tuning Hough line detection parameters for better K-block localization.

### Phase 3: Expanded SAM Tuning (31 parameters)
- **Iterations:** 30
- **Baseline:** 0.744
- **Result:** 0.768
- **Improvement:** +3.2%

Added tunable physical constants (k_height, ab_height, angle_deg) and vertical levels.

---

## Performance Summary

| Phase | Parameters | mIoU | Cumulative Improvement |
|-------|------------|------|------------------------|
| Baseline | - | 0.672 | - |
| Phase 1 (SAM) | 21 | 0.700 | +4.2% |
| Phase 2 (Detection) | 14 | 0.744 | +10.7% |
| **Phase 3 (Expanded)** | **31** | **0.768** | **+14.3%** |

---

## Optimized Parameters

### Detection Parameters (`parameters_detection.json`)

```json
{
    "preprocessing": {
        "binary_threshold": 149,
        "dilation_kernel_size": 2,
        "dilation_iterations": 1
    },
    "hough_oblique": {
        "threshold": 69,
        "min_length": 99,
        "max_gap": 60,
        "angle_positive_min": 5.509,
        "angle_positive_max": 8.652
    },
    "hough_horizontal": {
        "threshold": 66,
        "min_length": 122,
        "max_gap": 14
    },
    "hough_vertical": {
        "threshold": 700
    },
    "line_processing": {
        "merge_distance_threshold": 2
    }
}
```

### SAM Parameters (`parameters_sam.json`)

#### Segment Geometry (Physical Constants - NOW TUNED)
| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| segment_width | 1200.0 | **1157.47** | -3.5% |
| k_height | 1079.92 | **1071.09** | -0.8% |
| ab_height | 3239.77 | **3289.52** | +1.5% |
| angle_deg | 7.52 | **6.98** | -7.2% |

#### Processing Parameters
| Parameter | Original | Optimized |
|-----------|----------|-----------|
| padding | 150 | **111** |
| crop_margin | 50 | **57** |

#### K-block Prompt Points
| Parameter | Original | Optimized |
|-----------|----------|-----------|
| outer_ring | 700 | **657.08** |
| middle_ring | 500 | **514.10** |
| inner_ring | 348.16 | **327.00** |
| center_ring | 325 | **370.00** |

#### AB-block Prompt Points
| Parameter | Original | Optimized |
|-----------|----------|-----------|
| outer_ring | 700 | **674.53** |
| middle_ring | 511.06 | **519.91** |
| inner_ring | 500 | **479.23** |
| center_ring | 325 | **310.77** |
| fine_spacing | 250 | **219.56** |
| ultra_fine | 162.5 | **146.48** |
| edge_ring | 348.16 | **340.19** |
| edge_spacing | 350 | **312.25** |

#### Vertical Levels (NOW TUNED)
| Level | Original | Optimized | Change |
|-------|----------|-----------|--------|
| level_1 | 1719.89 | **1779.90** | +3.5% |
| level_2 | 1519.89 | **1670.00** | +9.9% |
| level_3 | 1344.89 | **1399.92** | +4.1% |
| level_4 | 1090.09 | **1193.60** | +9.5% |
| level_5 | 817.57 | **812.61** | -0.6% |
| level_6 | 545.05 | **584.85** | +7.3% |
| level_7 | 272.52 | **300.00** | +10.1% |

#### Template Mask Dimensions
| Parameter | Original | Optimized |
|-----------|----------|-----------|
| k_mask_width | 625 | **642.95** |
| k_mask_height_pos | 619.16 | **656.47** |
| k_mask_height_neg | 460.77 | **460.41** |
| ab_mask_width | 625 | **575.00** |
| ab_mask_height | 1619.89 | **1581.36** |

#### Quality Threshold
| Parameter | Original | Optimized |
|-----------|----------|-----------|
| min_quality_threshold | 0.30 | **0.267** |

---

## Convergence Analysis

### Phase 1: SAM Tuning
```
Iterations: 1-30
Best found: Iteration 25 (mIoU = 0.700)
Convergence: Stable after iteration 25
```

### Phase 2: Detection Tuning  
```
Iterations: 1-30
Best found: Iteration 23 (mIoU = 0.744)
Key jump: Iteration 6 (0.606 → 0.717)
Convergence: Stable after iteration 23
```

### Phase 3: Expanded SAM Tuning
```
Iterations: 1-30
Best found: Iteration 21 (mIoU = 0.768)
Key jump: Iteration 3 (0.712 → 0.760)
Convergence: Stable after iteration 21
```

---

## Key Insights

### 1. Detection Tuning Had Largest Single Impact
- Detection optimization alone improved mIoU by +6.3%
- Better K-block localization benefits ALL segment classes
- `binary_threshold` increase (127→149) improved edge detection
- `hough_vertical_threshold` increase (500→700) reduced false positives

### 2. Physical Constants Matter
- Tuning `angle_deg` from 7.52 to 6.98 (-7.2%) improved segment alignment
- `ab_height` adjustment (+1.5%) better matched actual segment heights
- These were previously fixed as "do not tune" parameters

### 3. Vertical Level Optimization
- Most levels increased (avg +6.3%), suggesting original values underestimated spacing
- `level_2` and `level_4` had largest adjustments (+9.9%, +9.5%)
- These directly affect AB-block prompt point placement

### 4. Diminishing Returns Observed
- Phase 1: +4.2% improvement
- Phase 2: +6.3% improvement (largest)
- Phase 3: +3.2% improvement (expanded space still found gains)

---

## Score Distribution Across All Iterations

### Phase 3 (Expanded SAM) - 30 Iterations
| Range | Count | Percentage |
|-------|-------|------------|
| 0.75+ | 2 | 6.7% |
| 0.70-0.75 | 11 | 36.7% |
| 0.65-0.70 | 7 | 23.3% |
| 0.55-0.65 | 8 | 26.7% |
| < 0.55 | 2 | 6.7% |

The expanded parameter space shows higher variance but found the best score (0.768).

---

## Files Generated

### Result Files
- `2-2_sam_20260122_113201.json` - Phase 3 final results
- `2-2_sam_20260122_113201_history.json` - Phase 3 iteration history
- `2-2_detection_20260122_085116.json` - Phase 2 final results
- `2-2_detection_20260122_085116_history.json` - Phase 2 iteration history

### Parameter Files (Updated)
- `p4tun/parameters/2-2/parameters_detection.json`
- `p4tun/parameters/2-2/parameters_sam.json`

---

## Recommendations

### For Tunnel 2-2
1. **Use current optimized parameters** - mIoU 0.768 is near optimal
2. **Further iterations unlikely to improve significantly** - convergence observed
3. **Consider per-class optimization** if specific segments need improvement

### For Other Tunnels
1. **Start with expanded search space** (31 SAM + 14 detection = 45 params)
2. **Run detection + SAM sequentially** for maximum benefit
3. **Physical constants should be tunable** - they vary between tunnels
4. **Expect 10-15% improvement** for similar tunnel conditions

### For System Improvements
1. **Add early stopping** when convergence detected
2. **Consider multi-objective optimization** (mIoU + individual class IoU)
3. **Implement transfer learning** from optimized tunnel parameters

---

## Conclusion

Bayesian Optimization proved highly effective for tunnel segmentation parameter tuning:

- **Total improvement: +14.3%** (0.672 → 0.768)
- **45 total tunable parameters** identified and optimized
- **Physical constants** (previously fixed) provided additional +3.2% gains
- **Detection stage** had the largest single impact (+6.3%)

The final configuration represents a well-optimized parameter set for tunnel 2-2, with diminishing returns suggesting we are near the performance ceiling for this approach.

---

*Report generated: 2026-01-22*  
*Optimization framework: scikit-optimize (skopt) with Gaussian Process*
