# P4Tun Optimization Journey: Tunnels 3-1 and 1-4
## A Comprehensive Report on Detection & SAM Bayesian Optimization

**Date:** January 26, 2026  
**Focus Tunnels:** 3-1 (6-segment) and 1-4 (6-segment, alternating pattern)  
**Optimization Approach:** Stage-by-stage Bayesian Optimization with GT-based reverse engineering

**Key Achievements:**
- **Tunnel 3-1:** mIoU 0.490 → **0.687** (+40.2% improvement)
- **Tunnel 1-4:** mIoU 0.626 → **0.748** (+19.5% improvement)
- **Critical Discovery:** K-block parameters require protection during BO to prevent performance degradation

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-block Detection Count** | Number of K positions detected | Exactly matches ring_count | Missing K positions cascade to all segments |
| **K-block Position Accuracy** | Euclidean error to GT position | <50 pixels (excellent), <100 pixels (good) | Primary anchor for all other segments |
| **K-block Horizontal Alignment** | Std dev of detected Y positions | <10 pixels | K-blocks should be horizontally aligned |
| **Y-Position Range** | Max Y - Min Y across all K positions | <30 pixels | Measures horizontal consistency |
| **Hough Line Count** | Positive + negative slope lines detected | >20 each | Too few = unreliable intersections |
| **Vertical Line Count** | Number of vertical lines detected | ≥ ring_count | Vertical lines define ring boundaries |
| **Detection Method Distribution** | % using midpoint vs single-slope vs assume | >60% midpoint | Midpoint method is most reliable |
| **Average Position Error** | Mean Euclidean distance to GT | <100 pixels | Direct measure of detection quality |
| **Match Rate** | % of GT positions with detected match <100px | >90% | Coverage metric |

**Critical Finding (Tunnel 3-1):** Initial detection had Y-range of 72 pixels. After BO focused on horizontal alignment, reduced to 21.7 pixels, significantly improving downstream SAM performance.

**Critical Finding (Tunnel 1-4):** Alternating Y-pattern (5 positions at Y~1562, 5 at Y~1130) requires detection to handle both levels correctly. Missing vertical lines in columns 3 and 5 (X=584, X=1065) can be inferred from even spacing pattern (~240px).

---

### Stage 4-2: SAM Segmentation (4-2_sam.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **mIoU (Mean IoU)** | Average IoU across all classes | >0.60 (good), >0.70 (excellent) | Primary optimization target |
| **Per-Class IoU** | IoU for each segment type | >0.50 for all classes | Identifies weak classes |
| **K-block IoU** | IoU specifically for K-block | >0.50 (critical) | K-block anchors all segments |
| **Background Ratio** | % points classified as background | <30% | High background = undersized templates |
| **OA (Overall Accuracy)** | Correct predictions / total | >0.80 | Overall segmentation quality |
| **F1 Score (Macro)** | Harmonic mean of precision/recall | >0.75 | Handles class imbalance |
| **Template Coverage** | Template area vs GT segment area | 90-110% | Undersized = FN, oversized = FP |
| **Prompt Point Validity** | % prompts within image bounds | >98% | Out-of-bounds cause failures |

**Critical Discovery:** During SAM BO, optimizing for overall mIoU can sacrifice K-block performance. K-block IoU dropped from 0.410 to 0.403 while overall mIoU improved. This revealed that **K-block parameters need protection** during optimization.

**Key Insight:** The parameters `k_mask_height` and `angle_deg` are critical for K-block performance and should not be allowed to degrade significantly for marginal overall mIoU gains.

---

## Part 2: The Thought Process Experience

### 2.1 Initial Problem Understanding

**Starting Context:**
```
THOUGHT: "User wants to optimize detection and SAM parameters for tunnels 3-1 and 1-4.
The approach is:
1. Use ground truth to reverse-engineer optimal detection results
2. Run BO to tune detection parameters
3. Run BO to tune SAM parameters using mIoU as objective

This is different from previous work - we're doing stage-by-stage optimization
rather than end-to-end, which should be more focused and effective."
```

**Key Constraint Identified:**
```
THOUGHT: "Ground truth should be used to LEARN optimal parameters, not as direct
input to the solution. The solution must work on tunnels without GT.

For detection: GT K positions tell us WHERE K-blocks should be, which helps
us tune detection parameters to find them accurately.

For SAM: GT segment boundaries tell us WHAT SIZE templates should be, which
helps us tune template dimensions."
```

---

### 2.2 Tunnel 3-1 Optimization Journey

#### Phase 1: Detection Optimization

**Initial State:**
```
OBSERVATION: "Tunnel 3-1 has 6 rings. Need to detect 6 K positions.
Current detection shows Y positions around 1652.5, but user says they're
'lower than supposed centers'."

THOUGHT: "I need to:
1. Check the ground truth K positions
2. Compare with detected positions
3. Understand why there's a discrepancy"
```

**Ground Truth Analysis:**
```
ACTION: Loaded detected_gt.csv - found Y values ~1711-1724
ACTION: User provided reference from data/bo/3-1/detected.csv with Y~1392-1466
ACTION: Visual inspection shows reference values look more correct

DISCOVERY: "The GT file has incorrect Y values! The reference BO results
have visually correct positions. I need to regenerate GT from the reference
data, but scale to current image dimensions."

THOUGHT: "The issue is that different versions of final.csv have different
theta values for K-blocks. The BO reference used a different final.csv with
theta ~8.59, while current has theta ~9.73. This explains the discrepancy."
```

**Detection BO Implementation:**
```
DESIGN DECISION: "I'll create a detection-specific BO that:
1. Uses actual p4tun.4-1_detection.py script (not inline version)
2. Compares detected K positions to GT K positions
3. Scores based on match count and position accuracy
4. Focuses on horizontal alignment (minimize Y variance)"

IMPLEMENTATION: Created p4tun/bo/detection_bo.py with:
- Search space: 14 detection parameters
- Objective: Match GT positions, penalize count errors and distance
- Uses subprocess to call actual detection script for consistency
```

**Breakthrough: Horizontal Alignment Focus**
```
USER FEEDBACK: "Use BO to further improve the K results, because they supposed
to be horizontally aligned."

THOUGHT: "Ah! K-blocks should be at the same Y position across all rings.
The current detection has Y variance. I should modify the objective to
penalize Y variance specifically."

MODIFICATION: Added Y-variance penalty to objective function:
score = match_score + distance_score - count_penalty - y_variance_penalty

RESULT: Y range reduced from 72 pixels to 21.7 pixels!
```

**Key Parameters Discovered (Tunnel 3-1 Detection):**
```
Best parameters from BO:
- binary_threshold: 134
- hough_oblique_threshold: 58
- hough_oblique_min_length: 78
- angle_positive_min: 5.91°
- angle_positive_max: 9.16°
- hough_vertical_threshold: 466
```

---

#### Phase 2: SAM Optimization

**Initial SAM BO:**
```
APPROACH: "Run SAM BO with 40 iterations, optimizing for mIoU.
Search space includes 16 parameters covering segment geometry, prompt points,
template masks, and quality threshold."

RESULT: Achieved mIoU 0.655 (up from baseline 0.490)
```

**Critical Discovery: K-block Performance Degradation**
```
OBSERVATION: "After SAM BO, overall mIoU improved to 0.655, but K-block IoU
dropped from 0.410 to 0.403. User noticed this and asked why."

ANALYSIS: "Let me check what parameters changed for K-block..."

DISCOVERY:
- k_mask_height: 655 → 550 (-16%) ← PROBLEM!
- angle_deg: 6.5 → 5.5 (-15%) ← PROBLEM!

THOUGHT: "The BO optimized for overall mIoU, which is dominated by A/B blocks
(since there are more of them). It sacrificed K-block performance to improve
A/B blocks. But K-block is critical as an anchor!"

INSIGHT: "K-block parameters need PROTECTION during BO. We can't let them
degrade significantly for marginal overall gains."
```

**Solution: Parameter Restoration**
```
ACTION: Restored critical K-block parameters:
- angle_deg: 6.5 (restored from 5.5)
- k_mask_height: 655 (restored from 550)

RESULT: K-block IoU jumped from 0.403 to 0.558 (+38.5%)!
Overall mIoU improved to 0.687 (from 0.655)

LESSON: "Some parameters are more critical than others. K-block parameters
should have tighter bounds or separate optimization objective."
```

**Final Tunnel 3-1 Results:**
```
Baseline → Final:
- mIoU: 0.490 → 0.687 (+40.2%)
- OA: 0.721 → 0.854 (+18.4%)
- F1: 0.638 → 0.794 (+24.5%)
- K-block IoU: 0.410 → 0.558 (+36.1%)
```

---

### 2.3 Tunnel 1-4 Optimization Journey

#### Phase 1: Detection Optimization

**Initial Analysis:**
```
OBSERVATION: "Tunnel 1-4 has 10 rings with alternating Y pattern:
- 5 K positions at Y~1562 (HIGH)
- 5 K positions at Y~1130 (LOW)

This is different from 3-1 which has uniform Y positions."

THOUGHT: "The detection needs to handle this alternating pattern correctly.
The 'assume' logic in detection should account for alternation."
```

**Detection BO Run 1 (30 iterations):**
```
APPROACH: "Run detection BO with 30 iterations, 10 initial points.
Objective: Match GT positions with alternating pattern awareness."

RESULT: Best score 0.8068 at evaluation 13
- 8/10 GT positions matched
- Missing: Column 3 (X=584) and Column 5 (X=1065)

PARAMETERS FOUND:
- binary_threshold: 101
- hough_oblique_threshold: 47
- hough_oblique_min_length: 67
- angle_positive_min: 5.88°
- angle_positive_max: 8.24°
- hough_vertical_threshold: 574
```

**Investigation: Why Columns 3 and 5 Are Missing**
```
USER OBSERVATION: "From the image, columns 3 and 5 K centroid points are
either missing or misplaced. But we have perfect oblique line detection
for column 3, just no vertical lines."

THOUGHT: "Interesting! If we have oblique lines but no vertical lines,
we should be able to infer the vertical line position from the spacing
pattern. All vertical lines should be evenly distributed."

ANALYSIS:
- Column 3 (X=584): 17 oblique lines detected, 0 vertical lines
- Column 5 (X=1065): 22 oblique lines detected, 0 vertical lines
- GT spacing: ~240px between vertical lines (very consistent!)

INSIGHT: "If we detect vertical lines at positions 2 and 4, and spacing
is ~240px, we can interpolate position 3 between them. This is a valid
geometric inference, not hard-coding."
```

**Attempted Solution: Interpolation Logic**
```
IMPLEMENTATION: Added interpolation logic to detect_lines() and compute_ring_centers():
1. If 2+ vertical lines detected with consistent spacing
2. Find gaps >1.5x average spacing
3. Interpolate missing vertical lines in gaps

RESULT: "Didn't trigger because only 1 vertical line detected (at X=2403).
The vertical threshold of 574 is too high for these regions."

USER FEEDBACK: "Do not apply this logic, it is too hard-coded. Keep the
previous settings."

LESSON: "Don't add complex interpolation logic. Keep detection simple.
If vertical lines are weak, that's a parameter tuning problem, not an
algorithm problem."
```

**Detection BO Run 2 (50 iterations):**
```
APPROACH: "Run more BO with:
- Lower vertical threshold range: [150, 500] instead of [200, 600]
- Extra penalty for missing columns 3 and 5 in objective function"

RESULT: Best score 0.4410 (worse than first run!)
- Lower threshold caused too many false positives

LESSON: "Lowering thresholds too much creates noise. Need balanced approach."
```

**Manual Threshold Testing:**
```
APPROACH: "Test vertical thresholds 350-550 around the best value (574)"

RESULTS:
- Threshold 350: 7/10 matched, too many false positives
- Threshold 400: 7/10 matched
- Threshold 450: 8/10 matched, only column 5 missing
- Threshold 500: 8/10 matched
- Threshold 550: 8/10 matched, columns 2 and 4 missing

BEST: Threshold 450 gives 8/10 matched, column 3 now detected!

BUT: User feedback: "Column 1 and 4 are misplaced with threshold 450"

DECISION: "Restore original best parameters (threshold 574) that give
correct columns 1 and 4, even if columns 3 and 5 are missing."
```

**Final Detection Status (Tunnel 1-4):**
```
Using BO reference detected.csv (from data/bo/1-4/):
- 10 K positions detected
- Columns 3 and 5 present (from previous successful BO)
- Columns 1 and 4 correctly positioned
```

---

#### Phase 2: SAM Optimization

**SAM BO Execution:**
```
APPROACH: "Run SAM BO with 40 iterations, protecting K-block parameters:
- angle_deg: [6.5, 8.5] (tighter range)
- k_mask_height: [580, 680] (kept high)"

PROGRESS:
- Evaluation 1: mIoU 0.597
- Evaluation 2: mIoU 0.674
- Evaluation 5: mIoU 0.715
- Evaluation 11: mIoU 0.719
- Evaluation 14: mIoU 0.727
- Evaluation 15: mIoU 0.736
- Evaluation 17: mIoU 0.748 ← BEST!

RESULT: Best mIoU 0.748 achieved at evaluation 17
```

**Critical Issue: Results Not Saved Properly**
```
PROBLEM: "After BO completed, I couldn't find the best parameters!
The inline BO script didn't save results to p4tun/bo/results/ like the
proper BO framework does."

DISCOVERY: "The best parameters were only printed to terminal, not saved
to JSON files. I had to extract them from terminal logs."

SOLUTION: Created extract_best_params.py script to:
1. Parse terminal logs for best parameters
2. Extract from BO history JSON files
3. Convert back to SAM JSON format

LESSON: "Always use the proper BO framework (p4tun.bo.optimize or
p4tun.bo.detection_bo) that saves results automatically. Don't write
inline BO scripts that don't log properly."
```

**Best SAM Parameters (Tunnel 1-4, mIoU 0.748):**
```
- segment_width: 1150.0
- k_height: 1150.0
- ab_height: 3104.77
- angle_deg: 6.5
- padding: 100
- crop_margin: 40
- k_outer_ring: 650.0
- k_middle_ring: 477.0
- k_inner_ring: 300.0
- ab_outer_ring: 711.58
- ab_middle_ring: 489.12
- k_mask_width: 639.0
- k_mask_height: 580.0
- ab_mask_width: 680.0
- ab_mask_height: 1577.29
- min_quality_threshold: 0.4965
```

**Final Tunnel 1-4 Results:**
```
Baseline → Final:
- mIoU: 0.626 → 0.748 (+19.5%)
- OA: 0.846 → 0.885 (+4.6%)
- F1: 0.734 → 0.851 (+15.9%)
- K-block IoU: 0.485 → 0.598 (+23.3%)
- A2-block IoU: 0.087 → 0.574 (+559.8% - wraparound issue partially addressed)
```

---

### 2.4 Mistakes Made and Lessons Learned

#### Mistake 1: Not Logging BO Results Properly

```
WHAT HAPPENED: "Wrote inline BO script that didn't save results to
p4tun/bo/results/ directory. Best parameters were only in terminal output."

WHY IT HAPPENED: "Tried to be quick and write a simple script, but didn't
follow the established pattern of using the BO framework."

IMPACT: "Had to manually extract parameters from terminal logs, which was
error-prone and time-consuming."

FIX: "Created extract_best_params.py script, but should have used proper
BO framework from the start."

LESSON: "Always use established frameworks that have proper logging.
Don't reinvent the wheel - it leads to maintenance issues."
```

#### Mistake 2: Optimizing K-block Parameters Away

```
WHAT HAPPENED: "SAM BO optimized for overall mIoU and reduced k_mask_height
from 655 to 550 and angle_deg from 6.5 to 5.5, causing K-block IoU to drop."

WHY IT HAPPENED: "Overall mIoU is dominated by A/B blocks (more pixels).
Optimizer found that sacrificing K-block slightly improved overall score."

IMPACT: "K-block IoU dropped from 0.410 to 0.403, even though overall
mIoU improved. User correctly identified this as a problem."

FIX: "Restored critical K-block parameters manually. For future: add
constraint or separate objective for K-block performance."

LESSON: "Not all classes are equal. Critical anchor classes (like K-block)
need protection during optimization. Consider multi-objective optimization
or constraints."
```

#### Mistake 3: Adding Hard-Coded Interpolation Logic

```
WHAT HAPPENED: "Added complex interpolation logic to infer missing vertical
lines based on spacing patterns."

WHY IT HAPPENED: "User's observation that 'vertical lines should be evenly
distributed' seemed like a valid geometric inference to implement."

IMPACT: "Logic was too hard-coded and didn't work well. User correctly
rejected it."

FIX: "Removed interpolation logic, kept detection simple."

LESSON: "Keep algorithms simple. If something needs fixing, fix the parameters
or data quality, not the algorithm logic. Hard-coded heuristics don't generalize."
```

#### Mistake 4: Not Verifying BO Results Before Proceeding

```
WHAT HAPPENED: "Assumed BO would save best parameters automatically, didn't
verify until user asked where results were stored."

WHY IT HAPPENED: "Focused on getting BO to run, didn't think about result
persistence."

IMPACT: "Had to reverse-engineer best parameters from terminal logs."

FIX: "Always verify BO results are saved before considering optimization complete."

LESSON: "Always verify outputs at each stage. Don't assume things work correctly."
```

---

### 2.5 What Made Success Possible

#### Success Factor 1: Stage-by-Stage Optimization

```
APPROACH: "Optimize detection first, then SAM separately."

BENEFIT: "Can focus on each stage's specific challenges without interference.
Detection issues don't mask SAM issues, and vice versa."

RESULT: "Clear attribution of improvements to each stage:
- Detection BO: Improved K position accuracy
- SAM BO: Improved segmentation quality"
```

#### Success Factor 2: Ground Truth Reverse Engineering

```
APPROACH: "Use GT to understand what 'good' looks like, then tune parameters
to achieve that."

BENEFIT: "GT provides target for optimization, not a cheat. Parameters learned
from GT are tunnel-specific priors."

RESULT: "Detection BO had clear objective (match GT positions), SAM BO had
clear objective (maximize mIoU)."
```

#### Success Factor 3: Horizontal Alignment Focus

```
INSIGHT: "K-blocks should be horizontally aligned. Adding Y-variance penalty
to detection objective improved results significantly."

RESULT: "Tunnel 3-1 Y-range reduced from 72px to 21.7px, improving downstream
SAM performance."
```

#### Success Factor 4: Parameter Protection Strategy

```
DISCOVERY: "K-block parameters are critical and can't be sacrificed."

SOLUTION: "Restore critical parameters after BO, or use tighter bounds during BO."

RESULT: "K-block IoU improved from 0.403 to 0.558 after parameter restoration."
```

---

### 2.6 Recommendations for Future Optimization

#### For Detection BO

**High Priority:**
```yaml
parameters:
  - name: hough_oblique_threshold
    range: [30, 70]
    sensitivity: HIGH
    
  - name: angle_positive_min
    range: [5.0, 7.0]
    sensitivity: HIGH
    
  - name: angle_positive_max
    range: [8.0, 12.0]
    sensitivity: HIGH
    
  - name: hough_vertical_threshold
    range: [400, 600]
    sensitivity: MEDIUM
    note: "Too low causes false positives, too high misses weak lines"
```

**Objective Function Enhancement:**
```python
# Add horizontal alignment penalty
y_variance = np.std(detected['Y'].values)
y_variance_penalty = y_variance * 2  # Penalize high variance

# Add specific column penalties for problematic regions
if missing_column_3_or_5:
    column_penalty += 200
```

#### For SAM BO

**Critical: Protect K-block Parameters**
```yaml
# Option 1: Tighter bounds for K-block params
k_mask_height: [600, 680]  # Don't allow <600
angle_deg: [6.0, 8.0]      # Don't allow <6.0

# Option 2: Multi-objective optimization
objectives:
  - maximize: overall_mIoU
  - maximize: k_block_IoU (weight: 0.3)
  
# Option 3: Constraint
constraints:
  - k_block_IoU >= 0.50  # Must maintain K-block performance
```

**High Sensitivity Parameters:**
```yaml
# Focus optimization on these
ab_mask_height: [1500, 1750]  # VERY HIGH sensitivity
ab_height: [3000, 3400]        # VERY HIGH sensitivity
k_mask_height: [580, 680]      # HIGH sensitivity (protect!)
angle_deg: [6.5, 8.5]          # HIGH sensitivity (protect!)
```

---

## Part 3: Key Parameters by Stage

### Stage 4-1: Detection Parameters

#### Preprocessing Sub-stage

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `binary_threshold` | int | 140 | 134 | 101 | HIGH | Threshold for binary conversion |
| `dilation_kernel_size` | int | 3 | 3 | 3 | MEDIUM | Morphological kernel size |
| `dilation_iterations` | int | 3 | 1 | 2 | MEDIUM | Number of dilation passes |

#### Hough Oblique Line Detection

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `hough_oblique_threshold` | int | 60 | 58 | 47 | HIGH | Vote threshold for line detection |
| `hough_oblique_min_length` | int | 111 | 78 | 67 | MEDIUM | Minimum line length in pixels |
| `hough_oblique_max_gap` | int | 33 | 34 | 32 | LOW | Maximum gap between line segments |
| `angle_positive_min` | float | 5.92° | 5.91° | 5.88° | HIGH | Minimum angle for positive slope lines |
| `angle_positive_max` | float | 8.67° | 9.16° | 8.24° | HIGH | Maximum angle for positive slope lines |

#### Hough Horizontal Line Detection

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `hough_horizontal_threshold` | int | 44 | 34 | 70 | MEDIUM | Vote threshold for horizontal lines |
| `hough_horizontal_min_length` | int | 113 | 102 | 103 | MEDIUM | Minimum line length |
| `hough_horizontal_max_gap` | int | 6 | 20 | 17 | LOW | Maximum gap between segments |

#### Hough Vertical Line Detection

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `hough_vertical_threshold` | int | 617 | 466 | 574 | MEDIUM | Vote threshold for vertical lines |

**Critical Finding:** Vertical threshold is a delicate balance:
- Too high (600+): Misses weak vertical lines in some columns
- Too low (300-): Creates false positives, detects noise as lines
- Optimal range: 450-600 depending on tunnel characteristics

#### Line Processing

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `merge_distance_threshold` | int | 5 | 4 | 6 | LOW | Distance for merging close lines |
| `merge_close_threshold` | int | 6 | 3 | 5 | LOW | Threshold for merging points |

---

### Stage 4-2: SAM Parameters

#### Segment Geometry (Physical Dimensions)

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `segment_width` | float | 1200.0 | 1100.0 | 1150.0 | MEDIUM | Segment width in mm |
| `k_height` | float | 1079.92 | 1112.41 | 1150.0 | MEDIUM | K-block height in mm |
| `ab_height` | float | 3239.77 | 3400.0 | 3104.77 | VERY HIGH | A/B block height in mm |
| `angle_deg` | float | 7.52 | 6.5 | 6.5 | HIGH | Segment angle in degrees |

**Critical Discovery:** `angle_deg` must be protected:
- Tunnel 3-1: Optimal at 6.5° (not 5.5° from initial BO)
- Tunnel 1-4: Optimal at 6.5° (protected during BO)
- Range should be [6.5, 8.5] to prevent degradation

#### Processing Parameters

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `padding` | int | 150 | 80 | 100 | LOW | Padding around crop region |
| `crop_margin` | int | 50 | 45 | 40 | LOW | Margin for cropping |

#### K-block Prompt Points

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `k_outer_ring` | float | 700.0 | 682.77 | 650.0 | MEDIUM | Outer ring radius for prompts |
| `k_middle_ring` | float | 500.0 | 503.69 | 477.0 | MEDIUM | Middle ring radius |
| `k_inner_ring` | float | 348.16 | 315.71 | 300.0 | MEDIUM | Inner ring radius |

#### AB-block Prompt Points

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `ab_outer_ring` | float | 700.0 | 722.64 | 711.58 | MEDIUM | Outer ring radius |
| `ab_middle_ring` | float | 511.06 | 535.74 | 489.12 | MEDIUM | Middle ring radius |

#### Template Mask Dimensions (K-block)

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `k_mask_width` | float | 625.0 | 637.54 | 639.0 | HIGH | K-block template width |
| `k_mask_height` | float | 655.14 | 655.0 | 580.0 | **CRITICAL** | K-block template height |

**Critical Finding:** `k_mask_height` is the most sensitive K-block parameter:
- Tunnel 3-1: Must be ≥650 to maintain K-block performance
- Tunnel 1-4: Optimal at 580 (lower than 3-1, but still critical)
- If reduced too much (e.g., 550), K-block IoU drops significantly

#### Template Mask Dimensions (AB-blocks)

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `ab_mask_width` | float | 625.0 | 595.44 | 680.0 | HIGH | AB-block template width |
| `ab_mask_height` | float | 1619.89 | 1614.55 | 1577.29 | **VERY HIGH** | AB-block template height |

**Critical Finding:** `ab_mask_height` has the highest sensitivity:
- Small changes (50-100mm) cause large mIoU changes
- Optimal range is narrow: [1500, 1750]mm
- Must be tuned carefully

#### Pattern-Aware Parameters

| Parameter | Type | Default | Optimized (3-1) | Optimized (1-4) | Sensitivity | Description |
|-----------|------|---------|-----------------|-----------------|-------------|-------------|
| `min_quality_threshold` | float | 0.3 | 0.388 | 0.4965 | MEDIUM | Minimum quality for mask acceptance |

---

## Part 4: Complete Thought Process Log

This section captures the actual internal reasoning process during the exploration - including hypotheses formed, dead ends encountered, and the chain of logic that led to discoveries.

---

### 4.1 Initial Hypothesis Formation

**Starting Point Thoughts:**
```
THOUGHT: "User wants to optimize detection and SAM for tunnels 3-1 and 1-4.
The approach is stage-by-stage: optimize detection first, then SAM.

This is different from end-to-end optimization - we can focus on each
stage's specific challenges."

HYPOTHESIS 1: "Detection accuracy will be the bottleneck"
→ Need to verify: Compare detected vs GT positions

HYPOTHESIS 2: "SAM parameters can be optimized independently once detection is good"
→ Need to verify: Run SAM BO after detection is optimized

HYPOTHESIS 3: "Ground truth can guide parameter ranges"
→ Need to verify: Analyze GT to understand optimal parameter values
```

---

### 4.2 Thought Chain: Tunnel 3-1 Detection Optimization

**Initial Observation:**
```
THOUGHT: "User says detected K positions are 'lower than supposed centers'.
Let me check the ground truth positions."

ACTION: Loaded detected_gt.csv - Y values ~1711-1724
ACTION: User provided reference from data/bo/3-1/ with Y~1392-1466

OBSERVATION: "The reference values look more visually correct. There's a
discrepancy between the GT file and what looks correct."

THOUGHT: "Maybe the GT file was generated incorrectly, or there's a version
mismatch. Let me check the final.csv theta values."
```

**Discovering the Theta Discrepancy:**
```
ACTION: Analyzed final.csv theta values for K-blocks
DISCOVERY: "Current final.csv has theta ~9.73, but BO reference has theta ~8.59"

THOUGHT: "Different versions of final.csv! The BO reference used a different
data version. I should use the BO reference's final.csv to regenerate GT,
but scale to current image dimensions."

REALIZATION: "The image dimensions are also different:
- BO reference: 2925 height
- Current: 3305 height

I need to scale the GT positions proportionally."
```

**Detection BO Design:**
```
THOUGHT: "I need to create a detection BO that:
1. Uses the actual detection script (not inline version)
2. Compares detected positions to GT
3. Scores based on match quality"

DESIGN DECISION: "Use subprocess to call p4tun.4-1_detection.py for each
evaluation. This ensures consistency with actual pipeline."

IMPLEMENTATION: Created objective function that:
- Runs detection script
- Reads detected.csv
- Matches to GT using nearest neighbor
- Scores: match_count * 100 + distance_score - count_penalty
```

**Horizontal Alignment Breakthrough:**
```
USER FEEDBACK: "Use BO to further improve the K results, because they supposed
to be horizontally aligned."

THOUGHT: "Ah! K-blocks should all be at the same Y position. The current
detection has Y variance. I should add a Y-variance penalty to the objective."

MODIFICATION:
y_variance = np.std(detected['Y'].values)
y_variance_penalty = y_variance * 5  # Strong penalty for misalignment

RESULT: "Y range reduced from 72px to 21.7px! This is a significant improvement."
```

---

### 4.3 Thought Chain: Tunnel 3-1 SAM Optimization

**Initial SAM BO:**
```
APPROACH: "Run SAM BO with 40 iterations, optimizing for mIoU.
Search space includes 16 parameters."

PROGRESS TRACKING:
- Evaluation 1-10: Random exploration
- Evaluation 11-20: Exploitation of promising regions
- Evaluation 21-30: Fine-tuning
- Evaluation 31-40: Final optimization

RESULT: Achieved mIoU 0.655 (up from 0.490 baseline)
```

**Discovering K-block Degradation:**
```
USER OBSERVATION: "Why K dropped? We got almost perfect detection, tuning sam
should make it improve a lot rather than dropping."

THOUGHT: "Good point! Let me check what happened to K-block parameters during BO."

ANALYSIS:
- Before BO: k_mask_height=655, angle_deg=6.5, K-block IoU=0.410
- After BO: k_mask_height=550, angle_deg=5.5, K-block IoU=0.403

REALIZATION: "The BO optimized for overall mIoU, which is dominated by A/B blocks.
It found that reducing K-block parameters slightly improved overall score, but
hurt K-block specifically."

INSIGHT: "K-block is an anchor - if it's wrong, all segments are affected.
We can't sacrifice K-block for marginal overall gains."
```

**Parameter Restoration:**
```
ACTION: Restored critical K-block parameters:
- angle_deg: 6.5 (from 5.5)
- k_mask_height: 655 (from 550)

RESULT: 
- K-block IoU: 0.403 → 0.558 (+38.5%)
- Overall mIoU: 0.655 → 0.687 (+4.9%)

THOUGHT: "This confirms that K-block parameters are critical and need protection."
```

---

### 4.4 Thought Chain: Tunnel 1-4 Detection Investigation

**User's Key Observation:**
```
USER: "One interesting observation though, the column 3 is the only one with
perfect oblique lines detection but without vertical lines, however, all the
vertical lines are supposed to be evenly distributed, which means it should not
be hard to deduce there will be one vertical lines between the wide area
between line2 and line 3 (which supposed to be line 4)."

THOUGHT: "Excellent observation! The user is right - if we have vertical lines
at positions 2 and 4, and spacing is even, we can interpolate position 3.
This is valid geometric reasoning, not hard-coding."
```

**Investigation Results:**
```
ANALYSIS:
- Column 3 (X=584): 17 oblique lines, 0 vertical lines
- Column 5 (X=1065): 22 oblique lines, 0 vertical lines
- GT spacing: ~240px (very consistent!)

THOUGHT: "The vertical lines exist geometrically, but the Hough transform
threshold is too high to detect them. We could:
1. Lower threshold (but causes false positives)
2. Interpolate based on spacing (geometric inference)
3. Use oblique intersections to infer vertical positions"
```

**Interpolation Attempt:**
```
IMPLEMENTATION: Added interpolation logic to compute_ring_centers():
- If 2+ vertical lines with consistent spacing
- Find gaps >1.5x average spacing
- Interpolate missing vertical lines

RESULT: "Didn't work because only 1 vertical line detected. The logic
needs at least 2 lines to establish a pattern."

USER FEEDBACK: "Do not apply this logic, it is too hard-coded."

LESSON: "User is right - keep algorithms simple. If vertical lines are weak,
that's a parameter problem, not an algorithm problem. Fix the parameters,
don't add complex heuristics."
```

---

### 4.5 Thought Chain: BO Result Recovery

**The Problem:**
```
USER: "I saw during the process, you have reached the best performance more
than 0.72 mIoU, but I cannot see where you actually stored the bo results as
the previous experiment. How can you find out the best parameter set that
give the best performance?"

THOUGHT: "Oh no! The inline BO script I wrote didn't save results to
p4tun/bo/results/ like the proper framework does. The best parameters are
only in the terminal output."

PROBLEM: "How do I recover the best parameters from terminal logs?"
```

**Solution Development:**
```
APPROACH: "I need to:
1. Parse terminal logs for 'Best parameters:' section
2. Extract parameter names and values
3. Convert back to SAM JSON format"

IMPLEMENTATION: Created extract_best_params.py that:
- Reads terminal log files
- Finds "Best parameters:" section
- Parses parameter lines
- Converts to JSON format

RESULT: Successfully extracted best parameters (mIoU 0.748) from evaluation 17
```

**Verification:**
```
ACTION: Applied extracted parameters and ran SAM + evaluation
RESULT: Confirmed mIoU 0.748 matches the best BO result

LESSON: "Always use proper BO framework that saves results automatically.
Don't write inline scripts that don't log properly."
```

---

### 4.6 Dead Ends and What Didn't Work

**Dead End 1: Lowering Vertical Threshold Too Much**
```
ATTEMPT: "Lowered vertical threshold from [200, 600] to [150, 500] to help
detect columns 3 and 5."

RESULT: "Best score dropped from 0.8068 to 0.4410. Too many false positives."

ANALYSIS: "Lower threshold detects noise as vertical lines, creating many
false ring centers. The original threshold (574) was actually well-tuned."

LESSON: "Don't assume lower thresholds are always better. There's a sweet
spot that balances detection rate and false positives."
```

**Dead End 2: Hard-Coded Interpolation Logic**
```
ATTEMPT: "Added interpolation logic to infer missing vertical lines from
spacing patterns."

RESULT: "User rejected as 'too hard-coded'."

ANALYSIS: "The logic was complex and tunnel-specific. It didn't generalize
well and made the code harder to maintain."

LESSON: "Keep algorithms simple. If something needs fixing, fix parameters
or data quality, not algorithm logic."
```

**Dead End 3: Optimizing K-block Parameters Away**
```
ATTEMPT: "Let BO optimize all parameters freely, including K-block ones."

RESULT: "K-block IoU dropped from 0.410 to 0.403 while overall mIoU improved."

ANALYSIS: "BO found that sacrificing K-block slightly improved overall score
(because A/B blocks dominate). But K-block is critical as an anchor."

LESSON: "Not all classes are equal. Critical anchor classes need protection
during optimization."
```

---

### 4.7 Key Reasoning Patterns That Led to Success

**Pattern 1: Stage-by-Stage Optimization**
```
APPROACH: "Optimize detection first, then SAM separately."

BENEFIT: "Clear attribution of improvements. Can focus on each stage's
specific challenges."

RESULT: "Detection BO improved K position accuracy, SAM BO improved
segmentation quality. Both stages contributed clearly."
```

**Pattern 2: Ground Truth as Learning Tool**
```
APPROACH: "Use GT to understand what 'good' looks like, then tune parameters
to achieve that."

BENEFIT: "GT provides target for optimization without being a cheat."

RESULT: "Detection BO had clear objective (match GT), SAM BO had clear
objective (maximize mIoU)."
```

**Pattern 3: Parameter Protection for Critical Classes**
```
DISCOVERY: "K-block parameters are critical and can't be sacrificed."

SOLUTION: "Restore critical parameters after BO, or use tighter bounds."

RESULT: "K-block IoU improved significantly after parameter restoration."
```

**Pattern 4: Always Log BO Results**
```
REQUIREMENT: "BO results must be saved to p4tun/bo/results/ for future reference."

BENEFIT: "Can recover best parameters, analyze optimization history, compare runs."

IMPLEMENTATION: "Use proper BO framework (p4tun.bo.optimize or p4tun.bo.detection_bo)
that saves results automatically."
```

---

### 4.8 Questions That Drove Discovery

**Questions I Asked:**
```
Q: "Why are detected K positions lower than supposed centers?"
→ Led to discovering GT file discrepancy and theta version differences

Q: "Why did K-block IoU drop during SAM BO?"
→ Led to discovering parameter sacrifice issue

Q: "How can we infer missing vertical lines?"
→ Led to spacing pattern analysis (though interpolation was rejected)

Q: "Where are the best BO parameters stored?"
→ Led to creating result extraction script and improving logging

Q: "Why are columns 3 and 5 missing vertical lines?"
→ Led to understanding that weak vertical lines need parameter tuning, not algorithm changes
```

**Questions That Would Help Future Work:**
```
Q: "Can we automatically detect which parameters are critical and need protection?"
Q: "Is there a way to do multi-objective BO that protects K-block while optimizing overall mIoU?"
Q: "Can vertical line detection be improved with adaptive thresholds per region?"
Q: "Should detection and SAM be optimized jointly or separately?"
```

---

### 4.9 Mental Model Evolution

**Initial Mental Model:**
```
"Optimization process:
1. Run BO for detection → Get best parameters
2. Run BO for SAM → Get best parameters
3. Done!"
```

**Corrected Mental Model:**
```
"Optimization process requires:
1. Verify GT data is correct (check versions, dimensions)
2. Run detection BO with proper logging
3. Verify detection results match GT
4. Run SAM BO with parameter protection for critical classes
5. Verify SAM results - check per-class IoU, not just overall
6. If critical class degraded, restore parameters and re-optimize
7. Always save BO results to proper location for future reference

Key insight: Not all parameters are equal. Some need protection.
Not all classes are equal. Some are critical anchors."
```

---

## Part 5: Complete Parameter Reference

### Detection Parameters (Complete List)

#### Preprocessing
```yaml
preprocessing:
  binary_threshold: [80, 200]      # HIGH sensitivity
  dilation_kernel_size: [2, 5]     # MEDIUM sensitivity
  dilation_iterations: [1, 4]       # MEDIUM sensitivity
```

#### Hough Oblique
```yaml
hough_oblique:
  threshold: [20, 100]              # HIGH sensitivity
  min_length: [40, 120]             # MEDIUM sensitivity
  max_gap: [30, 80]                 # LOW sensitivity
  angle_positive_min: [4.0, 7.0]    # HIGH sensitivity
  angle_positive_max: [8.0, 12.0]   # HIGH sensitivity
```

#### Hough Horizontal
```yaml
hough_horizontal:
  threshold: [30, 70]               # MEDIUM sensitivity
  min_length: [60, 130]             # MEDIUM sensitivity
  max_gap: [5, 25]                  # LOW sensitivity
```

#### Hough Vertical
```yaml
hough_vertical:
  threshold: [200, 600]             # MEDIUM sensitivity
  # Critical: Too high misses weak lines, too low creates false positives
```

#### Line Processing
```yaml
line_processing:
  merge_distance_threshold: [2, 8]  # LOW sensitivity
  merge_close_threshold: [4, 12]     # LOW sensitivity
```

---

### SAM Parameters (Complete List)

#### Segment Geometry
```yaml
segment_geometry:
  segment_width: [1100, 1250]       # MEDIUM sensitivity
  k_height: [1000, 1150]            # MEDIUM sensitivity
  ab_height: [3000, 3400]            # VERY HIGH sensitivity
  angle_deg: [6.5, 8.5]              # HIGH sensitivity (PROTECT!)
```

#### Processing
```yaml
processing:
  padding: [100, 200]               # LOW sensitivity
  crop_margin: [40, 80]              # LOW sensitivity
```

#### K-block Prompt Points
```yaml
prompt_points.k_block:
  outer_ring: [650, 750]            # MEDIUM sensitivity
  middle_ring: [450, 550]           # MEDIUM sensitivity
  inner_ring: [300, 400]            # MEDIUM sensitivity
```

#### AB-block Prompt Points
```yaml
prompt_points.ab_blocks:
  outer_ring: [650, 800]            # MEDIUM sensitivity
  middle_ring: [450, 600]           # MEDIUM sensitivity
```

#### Template Masks
```yaml
template_mask:
  k_block:
    width: [580, 680]               # HIGH sensitivity
    height_pos: [580, 680]          # CRITICAL (PROTECT!)
    height_neg: [580, 680]          # CRITICAL (PROTECT!)
  ab_blocks:
    width: [580, 680]               # HIGH sensitivity
    height: [1500, 1750]            # VERY HIGH sensitivity
```

#### Pattern-Aware
```yaml
pattern_aware:
  min_quality_threshold: [0.2, 0.6] # MEDIUM sensitivity
```

---

## Summary: Key Takeaways

### 1. Always Log BO Results Properly
- Use established BO frameworks that save to `p4tun/bo/results/`
- Don't write inline scripts that don't log results
- Verify results are saved before considering optimization complete

### 2. Protect Critical Parameters
- K-block parameters (`k_mask_height`, `angle_deg`) are critical
- Don't allow them to degrade for marginal overall gains
- Use tighter bounds or multi-objective optimization

### 3. Stage-by-Stage Optimization Works
- Optimize detection first, then SAM
- Clear attribution of improvements to each stage
- Can focus on stage-specific challenges

### 4. Ground Truth as Learning Tool
- Use GT to understand what "good" looks like
- Tune parameters to achieve GT-like results
- GT informs parameters, not the solution itself

### 5. Keep Algorithms Simple
- Don't add hard-coded heuristics
- Fix parameters or data quality, not algorithm logic
- Simple algorithms generalize better

### 6. Verify Results at Each Stage
- Check detection accuracy before SAM optimization
- Check per-class IoU, not just overall mIoU
- Identify issues early before they cascade

**Final Achievement:** 
- Tunnel 3-1: +40.2% mIoU improvement (0.490 → 0.687)
- Tunnel 1-4: +19.5% mIoU improvement (0.626 → 0.748)
- Established best practices for detection and SAM optimization

---

## Appendix: BO Result Files Created

### Tunnel 3-1
- `p4tun/bo/results/3-1_detection_20260126_*.json` - Detection BO results
- Detection parameters saved to `p4tun/parameters/3-1/parameters_detection.json`
- SAM parameters saved to `p4tun/parameters/3-1/parameters_sam.json`

### Tunnel 1-4
- `p4tun/bo/results/1-4_detection_20260126_124912.json` - Detection BO (score 0.8068)
- `p4tun/bo/results/1-4_detection_20260126_125324.json` - Detection BO run 2 (score 0.4410)
- `p4tun/bo/results/1-4_sam_20260126_best_extracted.json` - Extracted best SAM params (mIoU 0.748)
- Detection parameters saved to `p4tun/parameters/1-4/parameters_detection.json`
- SAM parameters saved to `p4tun/parameters/1-4/parameters_sam.json`

### Utility Scripts
- `p4tun/bo/extract_best_params.py` - Script to extract best parameters from BO logs/history

---

**Report Generated:** January 26, 2026  
**Author:** AI Assistant (Auto)  
**Session Focus:** Detection and SAM Bayesian Optimization for Tunnels 3-1 and 1-4
