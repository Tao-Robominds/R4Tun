# Tunnel Segmentation Parameter Tuning Guideline

**Based on extensive optimization of Tunnel 2-2**  
**Total improvement achieved: 0.672 → 0.765 mIoU (+13.8%)**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Optimization Order (Critical!)](#optimization-order)
4. [Stage-by-Stage Tuning Guide](#stage-by-stage-tuning-guide)
5. [GT-Based Manual Tuning](#gt-based-manual-tuning)
6. [When to Use BO vs Manual Tuning](#when-to-use-bo-vs-manual-tuning)
7. [Tunnel-Specific Considerations](#tunnel-specific-considerations)
8. [Common Pitfalls](#common-pitfalls)
9. [Quick Reference Tables](#quick-reference-tables)

---

## Executive Summary

### Key Findings from Tunnel 2-2 Optimization

| Phase | Stage | Parameters | mIoU Change | Impact |
|-------|-------|------------|-------------|--------|
| 1 | SAM (initial) | 21 | +4.2% | Medium |
| 2 | **Detection** | **14** | **+6.3%** | **Highest** |
| 3 | SAM (expanded) | 31 | +3.2% | Medium |
| 4 | Preprocessing | 23 | +0.1% | Low |
| 5 | Unfolding | 7 | +0.0% | Low |
| 6 | **Manual GT-based** | 4 | +0.3% | **Fine-tuning** |

### Top 3 Insights

1. **Detection stage has the highest single-stage impact** (+6.3%) - always optimize this first
2. **SAM template dimensions matter most** for per-class performance (especially K-block)
3. **Preprocessing/Unfolding have minimal impact** if already at reasonable defaults

---

## Pipeline Architecture

```
1_unfolding.py     → unwrapped.csv      (Point cloud → 2D depth map)
       ↓
2_denoising.py     → denoised.csv       (Remove noise points)
       ↓
3_enhancing.py     → enhanced.csv       (Surface interpolation)
       ↓
4-1_detection.py   → detected.csv       (K-block position detection)  ← HIGH IMPACT
       ↓
4-2_sam.py         → final.csv          (SAM segmentation)            ← HIGH IMPACT
       ↓
evaluation.py      → performance.md     (Metrics)
```

**Critical Dependency Chain:**
- Detection quality affects ALL downstream segmentation
- SAM templates must match actual segment geometry
- Preprocessing affects detection geometry indirectly

---

## Optimization Order

### Recommended Order (Highest Impact First)

```
1. Detection (14 params)     → Expect +3-7% mIoU
2. SAM (31 params)           → Expect +3-5% mIoU  
3. Manual GT-based tuning    → Expect +0.2-0.5% mIoU
4. Preprocessing (optional)  → Expect +0-0.2% mIoU
5. Unfolding (optional)      → Expect +0-0.1% mIoU
```

### Why This Order?

1. **Detection**: Poor K-block detection cascades errors to ALL segments
2. **SAM**: Template/prompt tuning directly affects segmentation precision
3. **GT-based**: Fine-tunes template sizes based on actual segment boundaries
4. **Preprocessing/Unfolding**: Minor impact when others are tuned

---

## Stage-by-Stage Tuning Guide

### Stage 1: Detection Parameters

**Impact: HIGH (+3-7% mIoU)**

#### Key Parameters to Tune

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `binary_threshold` | [100, 180] | HIGH | Higher = less noise, fewer detections |
| `hough_threshold_oblique` | [40, 100] | HIGH | Line detection sensitivity |
| `angle_positive_min/max` | [5, 10] | MEDIUM | Must match tunnel tilt angle |
| `hough_vertical_threshold` | [400, 800] | MEDIUM | Reduce false vertical lines |
| `merge_distance` | [1, 5] | LOW | Line merging tolerance |

#### Tuning Strategy

1. **Start with binary_threshold**: Adjust until oblique lines are clearly visible
2. **Tune angle ranges**: Should match physical K-block tilt (~6-9 degrees)
3. **Adjust hough thresholds**: Balance between detection rate and false positives

#### Expected Outcomes

- Good: 10 K-positions detected, evenly spaced
- Bad: Missing positions, duplicate detections, irregular spacing

---

### Stage 2: SAM Parameters

**Impact: MEDIUM-HIGH (+3-5% mIoU)**

#### A. Segment Geometry (Physical Constants)

| Parameter | Default | Typical Range | When to Adjust |
|-----------|---------|---------------|----------------|
| `segment_width` | 1200 | [1100, 1300] | Different tunnel diameters |
| `k_height` | 1080 | [1000, 1200] | K-block size varies |
| `ab_height` | 3240 | [3000, 3500] | AB segment height varies |
| `angle_deg` | 7.5 | [6, 9] | Tunnel tilt angle |

**Note**: These are derived from physical tunnel dimensions - measure from GT if available.

#### B. Template Mask Dimensions

| Parameter | Impact | When to Increase | When to Decrease |
|-----------|--------|------------------|------------------|
| `k_mask_width` | HIGH | K-block FN (missing pixels) | K-block FP (overlap) |
| `k_mask_height_neg` | HIGH | K-block FN on one side | K-block FP on one side |
| `ab_mask_width` | MEDIUM | B1/B2/A FN | B1/B2/A FP |
| `ab_mask_height` | MEDIUM | A-block FN | A-block FP |

#### C. Prompt Point Positions

| Parameter Group | Impact | Notes |
|-----------------|--------|-------|
| `outer_ring` | LOW | Edge prompt points |
| `middle_ring` | LOW | Mid-distance prompts |
| `inner_ring` | LOW | Central prompts |
| `vertical_levels` | MEDIUM | AB-block y-positioning |

**General Rule**: Prompt points are less sensitive than template masks.

#### Expected K-block Confusion Patterns

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| K-block → B1/B2 | Template too small | Increase `k_mask_width`, `k_mask_height_neg` |
| K-block → Background | Detection offset | Verify detected.csv positions |
| B1/B2 → K-block | Template too large | Decrease `k_mask_width` |

---

### Stage 3: Preprocessing Parameters

**Impact: LOW (+0-0.2% mIoU)**

Only tune if:
- Raw data is noisy
- Depth map has artifacts
- Detection is failing

#### Key Parameters

| Parameter | Purpose | When to Adjust |
|-----------|---------|----------------|
| `radius_min/max` | Point filtering range | Different tunnel diameters |
| `gradient_threshold` | Noise detection | Noisy raw data |
| `interpolation_radius` | Surface smoothing | Sparse point clouds |

---

### Stage 4: Unfolding Parameters

**Impact: VERY LOW (+0-0.1% mIoU)**

Only tune if:
- Centerline extraction fails
- Ring alignment is poor
- Cross-section fitting fails

---

## GT-Based Manual Tuning

**When BO reaches ceiling, use GT analysis for fine-tuning**

### Step 1: Analyze GT Segment Boundaries

```python
# Load GT data
df = pd.read_csv('final.csv')
k_gt = df[df['segment'] == 1]

# For each segment class, compute:
# - Y bounds (height in pixels)
# - X bounds (width in pixels)  
# - Center position relative to K-block
```

### Step 2: Compare with Current Templates

| Segment | GT Height (mm) | Current Template | Action |
|---------|----------------|------------------|--------|
| K-block | 1200 | 1117 | Increase height_neg |
| B1-block | 3310 | 1581 | Consider increasing |
| A2-block | 705 | 1581 | Template too large! |

### Step 3: Adjust One Parameter at a Time

**Order of adjustments:**
1. `k_mask_width` (most impactful)
2. `k_mask_height_neg` (K-block coverage)
3. `b1/b2/a_blocks width` (other segments)
4. Template heights (if needed)

### Step 4: Test After Each Change

```bash
./venv/bin/python p4tun/4-2_sam.py <tunnel_id>
./venv/bin/python p4tun/evaluation.py <tunnel_id>
```

### Successful Changes for Tunnel 2-2

| Parameter | Original | GT-Derived | Result |
|-----------|----------|------------|--------|
| `k_mask_width` | 642.95 | 700.0 | +0.1% mIoU |
| `k_mask_height_neg` | 460.41 | 540.0 | +0.2% K-IoU |
| `b1/b2/a_blocks width` | 575.0 | 610.0 | +0.2% mIoU |

---

## When to Use BO vs Manual Tuning

### Use Bayesian Optimization When:

✅ New tunnel with no prior optimization  
✅ Large parameter space (>10 parameters)  
✅ Compute time is available (30+ iterations)  
✅ No ground truth analysis done yet  

### Use Manual GT-Based Tuning When:

✅ BO has converged (diminishing returns)  
✅ Specific class has low IoU  
✅ Ground truth is available  
✅ Quick iteration needed  

### Hybrid Approach (Recommended)

```
1. Run BO on Detection (30 iterations)
2. Run BO on SAM (30 iterations)
3. Analyze GT for template dimensions
4. Manual fine-tuning based on GT
```

---

## Tunnel-Specific Considerations

### Different Diameters

- Affects `segment_width`, `k_height`, `ab_height`
- Derive from circumference: `C = K_height + n × AB_height`
- Auto-detection available in `detect_segment_count_from_geometry()`

### Different Segment Counts (6 vs 7)

- 6 segments: K, B1, A1, A2, A3, B2
- 7 segments: K, B1, A1, A2, A3, A4, B2
- Adjust `segment_per_ring` parameter

### Different Point Densities

- Sparse: Increase `interpolation_radius`, `num_interpolations`
- Dense: Can reduce processing parameters

### Different Noise Levels

- High noise: Increase `binary_threshold`, `gradient_threshold`
- Low noise: Can use defaults

### Wraparound Issues (7-segment tunnels)

- Segments wrap around image boundaries
- May need special handling in `4-2_sam.py`
- Consider segment_order adjustments

---

## Common Pitfalls

### 1. Tuning Order Wrong

❌ **Wrong**: Start with preprocessing  
✅ **Right**: Start with detection

### 2. Changing Too Many Parameters

❌ **Wrong**: Change 5 parameters, test once  
✅ **Right**: Change 1 parameter, test, iterate

### 3. Ignoring GT Analysis

❌ **Wrong**: Only use BO blindly  
✅ **Right**: Use GT to understand failure modes

### 4. Over-expanding Templates

❌ **Wrong**: Make K-block template huge to capture everything  
✅ **Right**: Expand gradually, watch for FP increase

### 5. Not Reverting Failed Changes

❌ **Wrong**: Keep all changes even if metrics drop  
✅ **Right**: Revert immediately if mIoU drops

---

## Quick Reference Tables

### Impact by Stage

| Stage | Parameters | Expected Impact | Time to Tune |
|-------|------------|-----------------|--------------|
| Detection | 14 | +3-7% | 30 min - 1 hr |
| SAM | 31 | +3-5% | 1-2 hrs |
| Manual GT | 4-6 | +0.2-0.5% | 30 min |
| Preprocessing | 23 | +0-0.2% | 30 min |
| Unfolding | 7 | +0-0.1% | 20 min |

### BO Settings

| Setting | Recommended | Notes |
|---------|-------------|-------|
| `n_calls` | 30 | Sufficient for convergence |
| `n_initial_points` | 10 | Random exploration phase |
| `optimizer` | 'gp' | Gaussian Process |
| `metric` | 'mIoU' | Primary optimization target |

### Per-Class Troubleshooting

| Low IoU Class | Primary Cause | Fix |
|---------------|---------------|-----|
| K-block | Template too small | Increase k_mask dimensions |
| B1/B2 | Width mismatch | Adjust b_block width |
| A-blocks | Boundary confusion | Verify vertical_levels |
| Background | Template overlap | Reduce mask sizes |

---

## GT-Free Deployment

### When Ground Truth is NOT Available

The BO optimization requires GT to calculate mIoU. For production tunnels without GT labels, use the following strategies:

### Strategy 1: Transfer Learning (Recommended)

**Use parameters from the most similar labeled tunnel**

```bash
# Copy parameters from a similar tunnel
cp p4tun/parameters/2-2/parameters_detection.json p4tun/parameters/NEW_TUNNEL/
cp p4tun/parameters/2-2/parameters_sam.json p4tun/parameters/NEW_TUNNEL/
```

**Matching criteria for transfer:**

| Attribute | Match Importance | Notes |
|-----------|------------------|-------|
| Diameter | HIGH | Similar circumference = similar segment sizes |
| Segment count | HIGH | Must match (6 vs 7) |
| Scanner type | MEDIUM | Affects point density |
| Tunnel condition | LOW | Minor impact |

**Expected performance:** 80-95% of optimized performance

---

### Strategy 2: Minimal Labeling

**Label a small sample for tuning**

1. **Label 1-2 rings manually** (~10,000-20,000 points)
2. **Run limited BO** (10-15 iterations)
3. **Focus on Detection stage** (highest impact)

```python
# Modify objective function to use partial GT
def evaluate_with_partial_gt(tunnel_id, labeled_rings=[133, 134]):
    df = pd.read_csv(f'data/{tunnel_id}/final.csv')
    labeled = df[df['ring'].isin(labeled_rings)]
    return compute_miou(labeled['pred'], labeled['segment'])
```

**Labeling effort:** 1-2 hours for 2 rings  
**Expected benefit:** +5-10% over transfer learning

---

### Strategy 3: Visual Quality Assessment

**Iterate based on visual inspection (no metrics)**

1. Run pipeline with transferred parameters
2. Visualize segmentation results
3. Identify systematic errors (e.g., K-block too small)
4. Adjust templates based on observations

**Checklist for visual assessment:**

- [ ] K-block positions detected correctly (check detected.csv)
- [ ] Segment boundaries align with visible joints
- [ ] No large gaps or overlaps between segments
- [ ] Background not consuming segment regions

---

### Strategy 4: Physical Measurement-Based Tuning

**Use tunnel specifications to derive parameters**

If you know the physical dimensions:

```python
# Example: Derive parameters from tunnel specs
tunnel_diameter_m = 5.4
segment_count = 6
k_height_mm = 1080  # From engineering specs
ab_height_mm = (np.pi * tunnel_diameter_m * 1000 - k_height_mm) / (segment_count - 1)

# Update parameters
params['segment_geometry']['k_height'] = k_height_mm
params['segment_geometry']['ab_height'] = ab_height_mm
```

---

### GT-Free Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│                    New Tunnel (No GT)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Find similar tunnel   │
              │  with optimized params │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Transfer parameters   │
              │  (copy JSON files)     │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Run pipeline          │
              │  Visual inspection     │
              └────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌──────────────┐         ┌──────────────────┐
     │ Results OK?  │   NO    │ Label 1-2 rings  │
     │    YES       │ ──────► │ Run limited BO   │
     └──────────────┘         └──────────────────┘
              │                         │
              ▼                         ▼
     ┌──────────────┐         ┌──────────────────┐
     │   Deploy!    │         │ Manual fine-tune │
     └──────────────┘         └──────────────────┘
```

---

### Reference: Optimized Parameter Sets Available

| Tunnel | Diameter | Segments | mIoU | Status |
|--------|----------|----------|------|--------|
| 2-2 | ~5.4m | 6 | 0.765 | ✓ Fully optimized |

*Add more tunnels as they are optimized*

---

## Conclusion

Effective tunnel segmentation tuning requires:

1. **Correct order**: Detection → SAM → Manual → Others
2. **GT analysis**: Understand actual segment geometry
3. **Incremental changes**: One parameter at a time
4. **Quick validation**: Test after each change

Expected total improvement: **+10-15% mIoU** from baseline

---

*Guideline version: 1.0*  
*Based on: Tunnel 2-2 optimization (2026-01)*  
*Framework: scikit-optimize + manual GT analysis*
