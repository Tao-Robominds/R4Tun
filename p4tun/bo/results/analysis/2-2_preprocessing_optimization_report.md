# Tunnel 2-2 Preprocessing Optimization Report

## Executive Summary

Bayesian Optimization was applied to tune **preprocessing parameters** (Denoising + Enhancing stages) for tunnel 2-2. The optimization achieved a **best mIoU of 0.769**, representing a marginal +0.001 improvement over the baseline 0.768.

**Key Finding**: Preprocessing parameters have limited direct impact on final segmentation quality when detection and SAM parameters are already well-tuned. However, preprocessing quality does affect detection geometry, creating interdependencies in the pipeline.

---

## Optimization Configuration

| Setting | Value |
|---------|-------|
| **Tunnel** | 2-2 |
| **Stage** | Preprocessing (Denoising + Enhancing) |
| **Parameters Tuned** | 23 (7 denoising + 16 enhancing) |
| **Iterations** | 30 |
| **Optimizer** | Gaussian Process (gp) |
| **Metric** | mIoU |
| **Baseline** | 0.768 |
| **Best Achieved** | 0.769 (+0.1%) |

---

## Search Space

### Denoising Parameters (7)

| Parameter | Range | Best Value |
|-----------|-------|------------|
| `radius_center` | [2.70, 2.80] | 2.707 |
| `radius_half_width` | [0.03, 0.10] | 0.088 |
| `theta_step` | [0.3, 0.7] | 0.569 |
| `radial_step` | [0.0008, 0.0015] | 0.00146 |
| `gradient_threshold` | [0.1, 0.4] | 0.1 |
| `smoothing_window` | [2, 5] | 4 |
| `smoothing_offset` | [-0.005, -0.001] | -0.00148 |

### Enhancing Parameters (16)

| Parameter | Range | Best Value |
|-----------|-------|------------|
| `curvature_neighbors` | [15, 30] | 16 |
| `target_distance_1` | [0.06, 0.10] | 0.091 |
| `target_distance_2` | [0.03, 0.06] | 0.060 |
| `target_distance_3` | [0.015, 0.03] | 0.022 |
| `curvature_threshold` | [0.0003, 0.0008] | 0.000377 |
| `upsampling_neighbors` | [15, 30] | 25 |
| `distance_tolerance_low` | [0.7, 1.1] | 1.02 |
| `distance_tolerance_high` | [1.5, 2.5] | 2.053 |
| `radius_filter_factor` | [0.10, 0.20] | 0.185 |
| `depth_threshold_low` | [0.002, 0.005] | 0.00283 |
| `depth_threshold_high` | [0.006, 0.012] | 0.00962 |
| `outlier_neighbors` | [15, 30] | 30 |
| `interpolation_radius` | [0.04, 0.08] | 0.0544 |
| `num_interpolations` | [1, 3] | 3 |
| `resolution` | [0.004, 0.006] | 0.00491 |
| `interpolation_window` | [3, 7] | 6 |

---

## Convergence Analysis

```
Iteration  Best mIoU   Note
---------  ---------   ----
    1      0.000       Failed (parameter error)
    5      0.723       First significant improvement
   10      0.723       Stable
   15      0.723       Stable
   20      0.723       Stable
   21      0.750       New improvement
   24      0.750       Stable
   26      0.769       Best achieved
   30      0.769       Final
```

### Score Distribution
- **Failed iterations**: 5 (returned 0.0 due to invalid parameters)
- **Successful iterations**: 25
- **Score range**: 0.045 - 0.769
- **Mean score**: 0.48 (excluding failures)

---

## Key Observations

### 1. Pipeline Interdependencies
The preprocessing stage affects downstream stages through:
- **Denoising** → removes noise points, affects point cloud quality
- **Enhancing** → adds interpolated points, creates depth map
- **Detection** → uses depth map to detect K-block positions
- **SAM** → uses detection geometry for segmentation

This creates a cascading effect where preprocessing changes can improve or degrade detection geometry.

### 2. Parameter Sensitivity
Most impactful parameters:
- `gradient_threshold=0.1` (at lower bound): More aggressive noise detection
- `radius_half_width=0.088`: Wider radius range keeps more points
- `num_interpolations=3` (at upper bound): More surface smoothing
- `outlier_neighbors=30` (at upper bound): Better outlier detection

### 3. Failure Modes
~17% of iterations failed due to:
- Invalid parameter combinations (e.g., radius constraints)
- Edge cases in point cloud processing
- Empty arrays after filtering

---

## Comparison with Previous Optimizations

| Phase | Stage | Parameters | mIoU Improvement |
|-------|-------|------------|------------------|
| 1 | SAM (initial) | 21 | 0.672 → 0.700 (+4.2%) |
| 2 | Detection | 14 | 0.700 → 0.744 (+6.3%) |
| 3 | SAM (expanded) | 31 | 0.744 → 0.768 (+3.2%) |
| 4 | **Preprocessing** | **23** | **0.768 → 0.769 (+0.1%)** |

**Total improvement**: 0.672 → 0.769 (+14.4%)

---

## Recommendations

### 1. Preprocessing Ceiling Reached
The marginal improvement (+0.1%) suggests that:
- Current preprocessing defaults are already near-optimal for 2-2
- Further preprocessing tuning unlikely to yield significant gains
- Focus optimization efforts on other tunnels or combined tuning

### 2. For Other Tunnels
Preprocessing tuning may be more beneficial for tunnels with:
- Higher noise levels in raw data
- Irregular surface geometries
- Different radius/depth characteristics

### 3. Combined Optimization
Consider combined tuning approaches:
- Denoising + Detection (noise affects line detection)
- Enhancing + SAM (depth map resolution affects segmentation)

---

## Files Generated

- `2-2_preprocessing_20260122_135958.json` - Best parameters
- `2-2_preprocessing_20260122_135958_history.json` - Full optimization history
- `parameters_denoising.json` - Updated denoising parameters
- `parameters_enhancing.json` - Updated enhancing parameters

---

## Conclusion

Preprocessing optimization for tunnel 2-2 achieved the target 30 iterations with a final best mIoU of **0.769**. While the improvement is marginal (+0.1%), this confirms that the preprocessing stage is already well-configured for this tunnel. The optimization revealed important pipeline interdependencies and identified the most sensitive preprocessing parameters.

**Final 2-2 Performance Summary:**
- **mIoU**: 0.769 (best achieved during BO)
- **Total optimization improvement**: +14.4% from baseline 0.672
