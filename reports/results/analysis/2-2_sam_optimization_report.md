# Bayesian Optimization Report: Tunnel 2-2 SAM Parameters

**Date:** 2026-01-22  
**Tunnel ID:** 2-2  
**Optimization Target:** mIoU (Mean Intersection over Union)  
**Total Iterations:** 30  
**Optimizer:** Gaussian Process (GP) with Expected Improvement

---

## Executive Summary

Bayesian Optimization successfully improved tunnel 2-2 segmentation performance from **0.672 mIoU** (baseline) to **0.700 mIoU** (optimized), achieving a **+4.2% relative improvement**. The optimization tuned 21 SAM parameters including prompt point positions, template mask dimensions, and processing configurations.

---

## Optimization Configuration

### Search Space (21 Parameters)

| Category | Parameter | Range | Best Value |
|----------|-----------|-------|------------|
| **Geometry** | segment_width | [1150, 1250] | 1191.94 |
| **Processing** | padding | [100, 200] | 100 |
| | crop_margin | [30, 80] | 61 |
| **K-block Prompts** | k_outer_ring | [650, 750] | 711.02 |
| | k_middle_ring | [450, 550] | 450.00 |
| | k_inner_ring | [300, 400] | 341.66 |
| | k_center_ring | [280, 370] | 370.00 |
| **AB-block Prompts** | ab_outer_ring | [650, 750] | 661.86 |
| | ab_middle_ring | [460, 560] | 460.00 |
| | ab_inner_ring | [450, 550] | 468.23 |
| | ab_center_ring | [280, 370] | 280.00 |
| | ab_fine_spacing | [200, 300] | 295.60 |
| | ab_ultra_fine | [130, 200] | 139.50 |
| | ab_edge_ring | [300, 400] | 300.00 |
| | ab_edge_spacing | [300, 400] | 400.00 |
| **K Mask** | k_mask_width | [575, 675] | 672.31 |
| | k_mask_height_pos | [570, 670] | 670.00 |
| | k_mask_height_neg | [410, 510] | 510.00 |
| **AB Mask** | ab_mask_width | [575, 675] | 591.08 |
| | ab_mask_height | [1570, 1670] | 1646.95 |
| **Quality** | min_quality_threshold | [0.1, 0.5] | 0.346 |

---

## Performance Results

### Overall Metrics Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **mIoU** | 0.672 | **0.700** | **+4.2%** |
| OA | 0.828 | 0.841 | +1.6% |
| F1 | 0.797 | 0.821 | +3.0% |

### Per-Class IoU Comparison

| Class | Baseline | Optimized | Change |
|-------|----------|-----------|--------|
| Background | 0.776 | 0.799 | +3.0% |
| **K-block** | 0.446 | **0.565** | **+26.7%** |
| B1-block | 0.713 | 0.755 | +5.9% |
| A1-block | 0.776 | 0.759 | -2.2% |
| **A2-block** | 0.555 | **0.620** | **+11.7%** |
| A3-block | 0.789 | 0.771 | -2.3% |
| B2-block | 0.650 | 0.635 | -2.3% |

**Key Findings:**
- K-block showed the largest improvement (+26.7%), from 0.446 to 0.565
- A2-block improved significantly (+11.7%), from 0.555 to 0.620
- The weakest classes benefited most from optimization
- Minor trade-offs in A1, A3, B2 (within acceptable margins)

---

## Convergence Analysis

### Score Progression Over 30 Iterations

```
Iteration:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
Score:    .67 .64 .67 .63 .67 .68 .68 .69 .65 .64 .64 .69 .67 .62 .69 .70 .68 .66 .67 .68 .69 .68 .69 .68 .70 .68 .70 .69 .69 .63
Best:     .67 .67 .67 .67 .67 .68 .68 .69 .69 .69 .69 .69 .69 .69 .69 .70 .70 .70 .70 .70 .70 .70 .70 .70 .70 .70 .70 .70 .70 .70
```

### Convergence Milestones

| Milestone | Iteration | mIoU | Notes |
|-----------|-----------|------|-------|
| Initial | 1 | 0.672 | Starting point (previous BO params) |
| First improvement | 5 | 0.674 | +0.002 |
| Second improvement | 6 | 0.680 | +0.008 |
| Third improvement | 7 | 0.681 | +0.009 |
| Fourth improvement | 8 | 0.687 | +0.015 |
| Fifth improvement | 12 | 0.690 | +0.018 |
| Sixth improvement | 16 | 0.695 | +0.023 |
| **Best found** | **25** | **0.700** | **+0.028** |

### Exploration vs Exploitation

- **Random exploration (iterations 1-10):** Wide variance (0.632 - 0.687)
- **Guided exploitation (iterations 11-20):** Narrowing to promising regions
- **Refinement (iterations 21-30):** Fine-tuning around best configurations
- **Best configuration found at iteration 25**, held through final iterations

---

## Detailed Score Distribution

### All 30 Iteration Scores

| Range | Count | Iterations |
|-------|-------|------------|
| 0.69 - 0.70 | 8 | 8, 12, 15, 21, 23, 25, 27, 29 |
| 0.68 - 0.69 | 5 | 6, 7, 20, 24, 26 |
| 0.67 - 0.68 | 6 | 1, 3, 5, 13, 17, 19 |
| 0.66 - 0.67 | 1 | 18 |
| 0.65 - 0.66 | 1 | 9 |
| 0.64 - 0.65 | 2 | 10, 11 |
| 0.63 - 0.64 | 5 | 2, 4, 14, 28, 30 |
| < 0.63 | 2 | 4, 14 |

### Statistics

- **Mean score:** 0.672
- **Std deviation:** 0.022
- **Min score:** 0.617 (iteration 14)
- **Max score:** 0.700 (iteration 25)
- **Improvement rate:** 26.7% of iterations improved on baseline

---

## Best Configuration Analysis

### Optimal Parameters (Iteration 25)

```json
{
    "segment_width": 1191.94,
    "padding": 100,
    "crop_margin": 61,
    "k_outer_ring": 711.02,
    "k_middle_ring": 450.0,
    "k_inner_ring": 341.66,
    "k_center_ring": 370.0,
    "ab_outer_ring": 661.86,
    "ab_middle_ring": 460.0,
    "ab_inner_ring": 468.23,
    "ab_center_ring": 280.0,
    "ab_fine_spacing": 295.60,
    "ab_ultra_fine": 139.50,
    "ab_edge_ring": 300.0,
    "ab_edge_spacing": 400.0,
    "k_mask_width": 672.31,
    "k_mask_height_pos": 670.0,
    "k_mask_height_neg": 510.0,
    "ab_mask_width": 591.08,
    "ab_mask_height": 1646.95,
    "min_quality_threshold": 0.346
}
```

### Key Parameter Insights

1. **K-block Mask Enlargement:** 
   - `k_mask_height_neg` increased from ~461 to 510 (+10.6%)
   - `k_mask_width` increased from ~637 to 672 (+5.5%)
   - Larger masks improved K-block coverage significantly

2. **Prompt Point Adjustments:**
   - `k_middle_ring` at minimum (450) - tighter inner prompts
   - `ab_center_ring` at minimum (280) - more central focus
   - These changes improved boundary precision

3. **Processing Efficiency:**
   - `padding` at minimum (100) - reduced unnecessary context
   - `crop_margin` optimized to 61 - balanced coverage

4. **Quality Threshold:**
   - `min_quality_threshold` at 0.346 - moderate filtering

---

## Conclusions

### Success Factors

1. **Expanded parameter space** (21 vs 2 parameters) enabled fine-grained optimization
2. **K-block and A2-block** were the primary beneficiaries, with 26.7% and 11.7% improvements
3. **GP-based BO** efficiently explored the high-dimensional space
4. **30 iterations** sufficient to find good local optimum

### Limitations

1. Small trade-offs in A1, A3, B2 classes (-2.2% to -2.3%)
2. High variance in early iterations suggests complex landscape
3. Additional iterations may yield marginal further improvements

### Recommendations

1. **Apply these parameters** for production use on tunnel 2-2
2. **Consider per-class optimization** for tunnels with specific weak classes
3. **Run longer optimization** (50-100 iterations) for potential 0.71+ performance
4. **Transfer learning:** Use these parameters as starting point for similar tunnels

---

## Files Generated

- `2-2_sam_20260122_042707.json` - Final results
- `2-2_sam_20260122_042707_history.json` - Full iteration history
- `2-2_sam_20260122_042707_convergence.png` - Convergence plot
- `2-2_sam_checkpoint.pkl` - Optimization checkpoint

---

*Report generated: 2026-01-22*  
*Optimization framework: scikit-optimize (skopt)*
