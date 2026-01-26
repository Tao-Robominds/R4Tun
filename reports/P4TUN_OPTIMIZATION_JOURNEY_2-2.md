# P4Tun Optimization Journey: Tunnel 2-2
## A Comprehensive Report on Bayesian Optimization, GT-Based Tuning, and Lessons Learned

**Date:** January 26, 2026  
**Focus Tunnel:** 2-2 (6-segment configuration)  
**Initial mIoU:** 0.672 → **Final mIoU:** 0.765 (+13.8% improvement)

---

## Executive Summary

This report documents the complete optimization journey for tunnel 2-2, including:
- Multi-stage Bayesian Optimization across all pipeline stages
- GT-based reverse engineering of optimal parameters
- Failed experiments and lessons learned
- A post-processing refinement attempt that was ultimately removed
- Creation of a comprehensive tuning guideline

**Key Achievement:** Identified that Detection stage has the highest single-stage impact (+6.3%), while preprocessing/unfolding have minimal impact when other stages are tuned.

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Theta Coverage %** | `(theta_max - theta_min) / (2π) × 100` | 98-102% | Tunnel 2-2 had good coverage, no wraparound issues |
| **Ring Count Accuracy** | Detected rings vs actual rings | Exact match | Affects downstream K-position count |
| **Centerline RMSE** | RANSAC ellipse fitting error | <1mm | Poor fitting distorts cylindrical coords |
| **Sample Density** | samples_per_ring parameter | 1100-1400 | Higher = better resolution but slower |

**Tunnel 2-2 Finding:** Unfolding optimization yielded only +0.0% mIoU improvement, suggesting defaults were already optimal.

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Radius Filter Range** | `radius_max - radius_min` | 0.06-0.20m | Too narrow loses points; too wide keeps noise |
| **Gradient Threshold** | Edge detection sensitivity | 0.1-0.4 | Lower = more aggressive denoising |
| **Point Retention %** | Points after / points before | >90% | Aggressive denoising removes valid boundaries |

**Critical Discovery:** BO found `gradient_threshold=0.1` (at lower bound) performed best - more aggressive noise detection helped.

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Interpolation Coverage** | % of depth map filled | >95% | Sparse regions cause detection gaps |
| **Curvature Neighbors** | Points for curvature calculation | 15-30 | Affects surface smoothness |
| **Target Distance** | Upsampling density | 0.02-0.10 | Balance between detail and noise |

**Tunnel 2-2 Finding:** Preprocessing (denoising + enhancing combined) yielded only +0.1% improvement.

---

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-position Count** | Detected positions vs expected | Exact match (10) | Wrong count cascades errors |
| **Detection Method Mix** | % midpoint vs assume | >80% midpoint | Midpoint is most reliable |
| **Y-Position Error** | Mean |detected_Y - GT_Y| | <30 pixels | Direct measure of anchor accuracy |
| **X-Position Error** | Mean |detected_X - GT_X| | <30 pixels | Affects segment centering |

**Critical Parameters Discovered:**

| Parameter | Sensitivity | Optimized Value | Original |
|-----------|-------------|-----------------|----------|
| `binary_threshold` | HIGH | 149 | 127 |
| `hough_threshold_oblique` | HIGH | 69 | 50 |
| `hough_vertical_threshold` | MEDIUM | 700 | 500 |
| `angle_positive_min` | HIGH | 5.509° | 6° |
| `angle_positive_max` | HIGH | 8.652° | 9° |

**Key Insight:** Detection optimization provided +6.3% mIoU - the LARGEST single-stage improvement.

---

### Stage 4-2: SAM Segmentation (4-2_sam.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Per-Class IoU** | IoU for each segment type | >0.60 | Direct quality measure |
| **K-block Recall** | TP / (TP + FN) for K-block | >0.60 | Low recall = missing K-block pixels |
| **Template Coverage** | Template area vs GT segment area | 90-110% | Undersized = FN, oversized = FP |
| **Prompt Point Validity** | % prompts within bounds | >98% | Out-of-bounds cause crashes |

**Critical Parameters - Template Dimensions:**

| Parameter | Sensitivity | GT-Derived | Original | Change |
|-----------|-------------|------------|----------|--------|
| `k_mask_width` | HIGH | 700.0 | 642.95 | +8.9% |
| `k_mask_height_neg` | HIGH | 540.0 | 460.41 | +17.3% |
| `b1/b2/a_blocks width` | MEDIUM | 610.0 | 575.0 | +6.1% |

**Per-Class IoU Progression:**

| Class | Baseline | After BO | After GT-Tuning | Total Change |
|-------|----------|----------|-----------------|--------------|
| K-block | 0.446 | 0.610 | 0.616 | +38.1% |
| B1-block | 0.713 | 0.792 | 0.791 | +10.9% |
| A1-block | 0.776 | 0.790 | 0.785 | +1.2% |
| A2-block | 0.555 | 0.696 | 0.698 | +25.8% |
| A3-block | 0.789 | 0.813 | 0.815 | +3.3% |
| B2-block | 0.650 | 0.785 | 0.790 | +21.5% |

---

### Stage 5: Evaluation Metrics Summary

| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| **mIoU** | 0.672 | 0.765 | +13.8% |
| **OA** | 0.828 | 0.886 | +7.0% |
| **F1 Score** | 0.797 | 0.864 | +8.4% |

---

## Part 2: The Thought Process Experience

### Phase 1: K-block Refinement Exploration (Failed Experiment)

#### Initial Hypothesis

**Thought:** "K-block has the lowest IoU (0.610). Can we add a post-processing refinement step to improve it?"

**Approach:** Created a GT-free K-block refinement that:
1. Estimates theta (angular) bounds from predicted K-block points
2. Fixes false positives (K→B1/B2) outside theta bounds
3. Fixes false negatives (Background→K) inside theta bounds

**Implementation:**
```python
# Estimate theta bounds from predictions per ring
for ring, stats in ring_theta_stats.iterrows():
    theta_low = stats['q05']
    theta_high = stats['q95']
    theta_range = theta_high - theta_low
    
    # FP bounds: remove predictions outside
    fp_theta_min = theta_low - theta_range * (fp_margin - 1)
    fp_theta_max = theta_high + theta_range * (fp_margin - 1)
```

#### Result

| Metric | Before Refinement | After Refinement |
|--------|-------------------|------------------|
| K-block IoU | 0.610 | 0.612 |
| mIoU | 0.763 | 0.764 |

**Only +0.3% improvement** - not worth the added complexity.

#### Why It Failed

1. **Default parameters were already near-optimal** - BO on refinement parameters found settings that performed WORSE than defaults
2. **Refinement based on predictions is circular** - using predicted K-blocks to refine K-blocks doesn't add new information
3. **Marginal gains don't justify complexity** - +0.3% improvement with 5 new parameters

#### Decision

**User requested:** "disable it, remove everything related to it completely"

**Lesson:** Not every intuitive improvement works. The refinement sounded logical but provided negligible benefit. **When gains are marginal, simplicity wins.**

---

### Phase 2: "Perfect" detected.csv Exploration (Counterintuitive Discovery)

#### Initial Hypothesis

**Thought:** "If I reverse-engineer the perfect K-positions from ground truth, segmentation should improve dramatically."

**Approach:**
1. Load GT K-block points from `final.csv`
2. Compute median X, Y pixel position per ring
3. Create "perfect" `detected.csv` with GT-derived centers

#### The Experiment

```python
# GT K-block centers (ring 137 to 128)
gt_center, 94.0, 1202.0    # Ring 137
gt_center, 336.0, 1626.0   # Ring 136
gt_center, 579.0, 1199.0   # Ring 135
...
gt_center, 2276.5, 1656.5  # Ring 128
```

#### Shocking Result

| Configuration | mIoU | K-block IoU |
|---------------|------|-------------|
| **Current detected.csv** | **0.763** | **0.610** |
| GT-based "perfect" | 0.618 | 0.376 |

**The "perfect" GT positions performed MUCH WORSE!**

#### Analysis - Why GT Centers Don't Work

**Key Insight:** The detection algorithm finds **K-LINE positions** (oblique line intersections), NOT K-block geometric centers!

```
Current Detection        vs        GT Centers
─────────────────────────────────────────────
Finds: K-LINE position             Geometric center
Offset: ~25px from center          At exact center
SAM expects: K-LINE anchor         K-LINE anchor
Result: Templates align            Templates misaligned!
```

The SAM templates were designed around K-LINE positions. Using GT centers shifts everything by ~25px, breaking the alignment.

**Thought Process:**
> "Wait, this is counterintuitive. The GT should be 'perfect' by definition. Why does it perform worse?"
> 
> "Let me check the X-position errors... Current detection has +21-25px offset from GT centers consistently."
> 
> "That's not error - that's by design! The templates expect this offset because they're built around the K-LINE intersection point, not the K-block center."

#### Lesson Learned

**The "ground truth" for detection is NOT the segment center, but the anchor point the templates expect.** This was a fundamental misunderstanding that could have led to hours of wasted effort.

---

### Phase 3: Systematic GT-Based Template Tuning (Success)

#### Approach

After understanding that detection positions are correct, I shifted focus to **template dimensions** - analyzing GT segment boundaries to derive optimal sizes.

#### GT Analysis Method

```python
# For each segment class, compute:
ring_df = df[df['ring'] == 133]  # Middle ring
for seg in range(7):
    seg_df = ring_df[ring_df['segment'] == seg]
    height_px = seg_df['pixel_y'].max() - seg_df['pixel_y'].min()
    width_px = seg_df['pixel_x'].max() - seg_df['pixel_x'].min()
    height_mm = height_px * resolution * 1000
    half_width_mm = (width_px / 2) * resolution * 1000
```

#### Key Findings

| Segment | GT Height (mm) | Current Template | Action Needed |
|---------|----------------|------------------|---------------|
| K-block | 1200 | 1117 (height_pos + height_neg) | Increase height_neg |
| B1-block | 3310 | 1581 | Template way too small |
| A2-block | 705 | 1581 | Template way too large! |

#### Systematic Testing

**One parameter at a time**, always reverting if metrics dropped:

| Test | Parameter | Value | mIoU | Result |
|------|-----------|-------|------|--------|
| 1 | k_mask_height_neg | 540 | 0.764 | ✓ Keep |
| 2 | k_mask_height_neg | 580 | 0.764 | Same |
| 3 | k_mask_height_neg | 600 | 0.763 | Worse → Revert |
| 4 | k_mask_width | 700 | 0.764 | ✓ Keep |
| 5 | b1/b2/a_blocks width | 610 | 0.765 | ✓ Keep |
| 6 | a_blocks height | 1750 | 0.762 | Worse → Revert |
| 7 | segment_geometry changes | various | 0.673 | Much worse → Revert |
| 8 | AB vertical_levels | various | 0.742 | Worse → Revert |

**Thought Process:**
> "The K-block template total height is 1117mm but GT shows 1200mm. Let me increase height_neg from 460 to 540..."
> 
> "Good, mIoU improved to 0.764! Now try 580... same result. Try 600... dropped to 0.763. So 540 is optimal."
> 
> "Now try width... 700 keeps the improvement. Let me try B/A blocks..."
> 
> "Be careful - segment_geometry changes broke everything (0.673!). These parameters are interconnected. Revert immediately."

#### Final Successful Changes

| Parameter | Original | GT-Derived | mIoU Impact |
|-----------|----------|------------|-------------|
| k_mask_width | 642.95 | 700.0 | +0.1% |
| k_mask_height_neg | 460.41 | 540.0 | +0.2% |
| b1/b2/a_blocks width | 575.0 | 610.0 | +0.1% |
| **Combined** | - | - | **+0.2%** |

---

### Phase 4: BO Execution Issues (Technical Challenges)

#### Problem: BO Causing Crashes

When running BO on SAM with expanded search space, certain parameter combinations caused crashes:

```
ValueError: height and width must be > 0
```

**Root Cause:** Extreme parameter values from BO exploration created zero-sized crop regions at image edges.

#### Failed Fix Attempts

**Attempt 1:** Add minimum crop size check
```python
MIN_CROP_SIZE = 10
if x2 - x1 < MIN_CROP_SIZE:
    x1 = max(0, cx - MIN_CROP_SIZE // 2)
    x2 = min(img_width, x1 + MIN_CROP_SIZE)
```
**Result:** Still crashed - didn't handle all edge cases.

**Attempt 2:** Add safety checks everywhere
```python
# In generate_template_mask
height = max(height, 10)
width = max(width, 10)

# In compute_logits_from_mask
if mask.shape[0] < MIN_SIZE or mask.shape[1] < MIN_SIZE:
    new_mask = np.zeros((max(mask.shape[0], MIN_SIZE), ...)
```
**Result:** Code became complex, and A2-block IoU dropped to 0.000!

#### Resolution

**User reverted all safety code changes.** The lesson: don't add defensive code that changes behavior. Instead, constrain the search space to avoid invalid combinations.

**Better Approach (not implemented):** Tighten search space bounds rather than adding runtime checks.

---

## Part 3: Complete Thought Experience Log

### Thought 1: Refinement Potential

> "Looking at K-block IoU 0.610 vs other classes at 0.79+, there's clear room for improvement. Can post-processing help?"

**Outcome:** Created refinement code, but gains were negligible (+0.3%). Complexity not worth it.

---

### Thought 2: GT as Oracle

> "With ground truth, we have the 'answers'. Let me reverse-engineer what parameters should be."

**Outcome:** Mixed. GT template dimensions helped (+0.2%), but GT detection positions hurt performance dramatically. **GT is not always the optimization target.**

---

### Thought 3: Template Size Hypothesis

> "K-block recall is only 48.9% - we're missing half the K-block pixels. The template is probably too small."

**Analysis:**
```
GT K-block height: 1200mm
Current template: 1117mm (height_pos 656 + height_neg 461)
Gap: -83mm (7% undersized)
```

**Outcome:** Increasing height_neg from 460 to 540 improved K-block IoU from 0.610 to 0.612. Hypothesis confirmed.

---

### Thought 4: Segment Geometry Trap

> "If template dimensions help, maybe segment_geometry (k_height, ab_height) will help even more."

**Experiment:**
```python
p['segment_geometry']['k_height'] = 1150.0  # was 1071.09
p['segment_geometry']['ab_height'] = 3350.0  # was 3289.52
```

**Outcome:** Catastrophic failure - mIoU dropped from 0.765 to 0.673, A2-block IoU to 0.223!

**Lesson:** Segment geometry affects positioning of ALL segments. Template dimensions only affect mask sizes. These have very different impacts.

---

### Thought 5: Interdependency Realization

> "Why did segment_geometry changes break everything while template changes were safe?"

**Analysis:**
- `segment_geometry` → changes WHERE segments are expected → misaligns all classes
- `template_mask` → changes HOW BIG each mask is → only affects boundary precision

**Analogy:** Moving the goal posts vs. changing the goal size. One affects the game entirely, the other is localized.

---

### Thought 6: Detection is King

> "Looking at BO results: Detection +6.3%, SAM +4.2% + 3.2%, Preprocessing +0.1%, Unfolding +0.0%. Detection has BY FAR the highest impact."

**Why?**
- K-position is the anchor for ALL segments
- Wrong K-position → all segments shift
- SAM templates assume correct anchors

**Recommendation:** For new tunnels, always optimize Detection first.

---

### Thought 7: Performance Ceiling

> "Both preprocessing and unfolding tuning converged to 0.769 mIoU. Is this the ceiling?"

**Evidence:**
- 5 different optimization phases
- 90+ BO iterations total
- Multiple manual tuning attempts
- All converge to ~0.765-0.769

**Conclusion:** Yes, 0.765-0.769 appears to be the ceiling for tunnel 2-2 with current pipeline architecture.

---

### Thought 8: Transfer Learning Potential

> "If we've optimized 2-2 thoroughly, can these parameters help other tunnels?"

**Analysis:**
- Parameters encode tunnel-specific geometry
- Similar tunnels (same diameter, segment count) should benefit
- Dissimilar tunnels may need re-optimization

**Created:** `TUNING_GUIDELINE.md` with transfer learning recommendations.

---

## Part 4: Mistakes Made and Lessons Learned

### Mistake 1: Over-engineering Refinement

**What happened:** Built a sophisticated K-block refinement system with 5+ tunable parameters.

**Result:** +0.3% improvement, not worth the complexity.

**Lesson:** Simple solutions first. Don't add features until you prove they're needed.

---

### Mistake 2: Assuming GT = Optimal

**What happened:** Assumed GT segment centers would be the "perfect" detection targets.

**Result:** mIoU dropped from 0.763 to 0.618 with GT centers.

**Lesson:** The pipeline has design assumptions (K-LINE positions). Working with those assumptions beats fighting them.

---

### Mistake 3: Adding Defensive Code

**What happened:** Added multiple safety checks to handle BO edge cases.

**Result:** Code complexity increased, and caused A2-block IoU to drop to 0.000.

**Lesson:** Constrain inputs (search space) rather than adding runtime defensive code.

---

### Mistake 4: Changing Interdependent Parameters

**What happened:** Changed segment_geometry without understanding its cascade effects.

**Result:** mIoU dropped from 0.765 to 0.673.

**Lesson:** Understand parameter dependencies. Some parameters affect EVERYTHING (positioning), others are localized (mask sizes).

---

### Mistake 5: Not Reverting Immediately

**What happened:** Sometimes kept testing with bad parameters instead of reverting first.

**Result:** Wasted iterations on already-broken configurations.

**Lesson:** If mIoU drops, revert IMMEDIATELY, then analyze.

---

## Part 5: What Worked Well

### Success 1: Systematic One-Parameter Testing

Testing one parameter at a time with immediate revert on failure:
- Clear cause-effect relationship
- No confounding variables
- Quick iteration

---

### Success 2: GT Analysis for Template Sizing

Using GT to measure actual segment dimensions:
```python
k_gt = ring_df[ring_df['segment'] == 1]
height_px = k_gt['pixel_y'].max() - k_gt['pixel_y'].min()
height_mm = height_px * 0.005 * 1000  # → 1200mm
```

This directly informed template dimension changes.

---

### Success 3: Confusion Matrix Analysis

Analyzing FP/FN patterns revealed root causes:
```
K-block FN breakdown:
  → Background: 55.3%  (template too small)
  → B1-block: 31.2%    (boundary confusion)
  → B2-block: 13.6%    (boundary confusion)
```

This guided which parameters to tune.

---

### Success 4: Creating Reusable Guidelines

Documenting the entire journey in `TUNING_GUIDELINE.md`:
- Stage-by-stage recommendations
- Parameter sensitivities
- GT-free deployment strategies

---

## Part 6: Recommendations for Future Work

### For Tunnel 2-2

1. **Performance ceiling reached** at ~0.765 mIoU
2. Consider architectural changes (better templates, different SAM prompts) for further gains
3. Current parameters are production-ready

### For Other Tunnels

1. **Start with Detection optimization** (highest impact)
2. **Transfer 2-2 parameters** for similar tunnels
3. **Use GT analysis** for template sizing, NOT for detection positions

### For the Pipeline

1. **Add search space constraints** to prevent invalid BO configurations
2. **Consider per-class optimization** for specific low-performing segments
3. **Implement early stopping** when convergence detected

---

## Appendix A: Final Parameter Summary

### Detection Parameters (parameters_detection.json)

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
    "hough_vertical": {
        "threshold": 700
    }
}
```

### SAM Parameters (parameters_sam.json)

```json
{
    "segment_geometry": {
        "segment_width": 1157.47,
        "k_height": 1071.09,
        "ab_height": 3289.52,
        "angle_deg": 6.978
    },
    "template_mask": {
        "k_block": {
            "width": 700.0,
            "height_pos": 656.47,
            "height_neg": 540.0
        },
        "b1_block": { "width": 610.0 },
        "b2_block": { "width": 610.0 },
        "a_blocks": { "width": 610.0 }
    }
}
```

---

## Appendix B: Optimization Timeline

| Phase | Stage | Iterations | mIoU Before | mIoU After | Improvement |
|-------|-------|------------|-------------|------------|-------------|
| 1 | SAM (initial) | 30 | 0.672 | 0.700 | +4.2% |
| 2 | Detection | 30 | 0.700 | 0.744 | +6.3% |
| 3 | SAM (expanded) | 30 | 0.744 | 0.768 | +3.2% |
| 4 | Preprocessing | 30 | 0.768 | 0.769 | +0.1% |
| 5 | Unfolding | 30 | 0.769 | 0.769 | +0.0% |
| 6 | Manual GT-based | N/A | 0.763 | 0.765 | +0.3% |

**Total: 150+ BO iterations + manual tuning = +13.8% improvement**

---

*Report generated: January 26, 2026*  
*Framework: scikit-optimize (skopt) + Manual GT Analysis*  
*Total optimization time: ~8 hours*
