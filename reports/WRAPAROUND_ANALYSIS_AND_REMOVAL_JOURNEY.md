# Wraparound Analysis and Removal Journey
## A Comprehensive Report on Wraparound Understanding, Analysis, and Code Simplification

**Date:** January 23, 2026  
**Focus:** Understanding wraparound issues, subsection selection strategies, and code simplification  
**Key Decision:** Removal of Y-wraparound functionality from SAM segmentation stage

---

## Executive Summary

This report documents a critical exploration journey focused on understanding and addressing wraparound issues in tunnel segmentation. The conversation evolved from initial confusion about segment coverage patterns to deep analysis of wraparound causes, potential mitigation strategies through subsection selection, and ultimately the decision to remove complex wraparound handling code in favor of simpler, more maintainable solutions.

**Key Outcomes:**
- **Understanding**: Comprehensive analysis of wraparound severity across all tunnel types (1-4, 2-2, 3-1, 4-1, 5-1)
- **Discovery**: Subsection selection can mitigate wraparound for 6-segment tunnels but has limited effectiveness for 7-segment tunnels
- **Decision**: Removed Y-wraparound functionality from `4-2_sam.py` to simplify codebase
- **Tools Created**: Subsection selection utility script for future experiments

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Theta Coverage %** | `(theta_max - theta_min) / (2π) × 100` | 98-102% | Coverage <100% loses segments; >100% causes wraparound |
| **Wraparound Segment Count** | Number of segments spanning 0°/360° boundary | 0-2 (6-seg), 5-7 (7-seg) | Direct measure of wraparound severity |
| **K-block Position Variance** | Std dev of K-block angular positions across rings | <15° | Low variance enables better subsection selection |
| **Ring Count Accuracy** | Detected rings vs actual rings | Exact match | Ring mismatch propagates errors downstream |
| **Point Density per Ring** | Points per ring slice | >10,000 | Sparse slices cause ellipse fitting failures |

**Critical Finding:** Tunnel 4-1 and 5-1 have ALL segments spanning boundaries (100% wraparound), making standard mitigation strategies ineffective.

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Surface Point Retention %** | `surface_points_after / surface_points_before` | 70-95% | Too aggressive = data loss; too lenient = noise retained |
| **Noise Removal Rate** | `noise_points / total_points` | 10-30% | Quantifies denoising effectiveness |
| **Edge Preservation Score** | Gradient magnitude at segment boundaries | >0.7 relative | Denoising should preserve edges |

**Note:** Denoising stage is not directly affected by wraparound, but preserving edges is critical for downstream detection.

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Interpolation Coverage** | % of depth map filled | >95% | Sparse regions cause detection gaps |
| **Coverage Uniformity** | `1 / (1 + std(coverage_grid))` | >0.7 | Uniform coverage improves detection |
| **Intensity Contrast** | `(max_intensity - min_intensity) / mean` | >0.3 | Low contrast makes segment detection hard |

**Note:** Enhancing stage helps fill gaps but cannot fix wraparound issues that occur at the unfolding stage.

---

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-block Detection Accuracy** | Euclidean error to GT position | <200 pixels | Primary anchor for all other segments |
| **K-block Position Variance** | Std dev of detected Y positions | <10 pixels (6-seg), <30 pixels (7-seg) | Measures horizontal consistency |
| **Hough Line Count** | Positive + negative slope lines detected | >20 each | Too few = unreliable intersections |
| **Detection Method Distribution** | % using combined vs fallback | >60% combined | Combined method is most reliable |

**Wraparound Impact:** Detection accuracy can be affected if segments span boundaries, but per-ring detection helps mitigate this.

---

### Stage 4-2: SAM Segmentation (4-2_sam.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Per-Class IoU** | Intersection over Union per segment type | >0.15 | Direct segmentation quality measure |
| **Background Ratio** | % points classified as background | <30% | High background = undersized templates |
| **Template Coverage** | % of segment area covered by template | 85-95% | Templates guide SAM's predictions |
| **Prompt Point Validity** | % prompts within image bounds | >98% | Out-of-bounds prompts cause failures |
| **Boundary Segment Completeness** | % of boundary-spanning segments fully segmented | >80% | Measures wraparound handling effectiveness |

**Critical Discovery:** Y-wraparound handling was removed because:
1. Complexity outweighed benefits
2. Standard cropping with clamping works adequately for most cases
3. True wraparound issues should be addressed at unfolding stage, not segmentation stage

---

## Part 2: The Thought Process Experience

### 2.1 Initial Problem Understanding

**Starting Point:**
```
USER INPUT: Segment coverage diagrams showing wraparound patterns for all tunnels
OBSERVATION: Tunnels 4-1 and 5-1 have ALL segments spanning boundaries
QUESTION: "Can we avoid wraparound by selecting subsections strategically?"
```

**Initial Hypothesis:**
```
THOUGHT: "If we could select ring ranges where K-blocks are better aligned,
maybe we could reduce wraparound severity. This could be a practical solution
for experimental datasets."
```

**Key Insight Formation:**
- Wraparound is a **data quality issue** introduced at the unfolding stage
- It's not just a visualization problem - it affects segmentation quality
- The severity varies dramatically between 6-segment and 7-segment tunnels

---

### 2.2 Breakthrough: Subsection Selection Strategy

**The Question That Drove Discovery:**
```
USER: "Because when we cut some sections from the full tunnels to do experiments,
we did not know it might cause wraparound issues. Will it be avoidable if we
cut subsection based on certain criteria?"
```

**The Analysis Process:**
1. **Understanding the Problem:**
   - When creating experimental subsections, ring selection was arbitrary
   - No consideration of K-block alignment or wraparound impact
   - Result: Inadvertently selected ranges with high wraparound

2. **Formulating Criteria:**
   ```
   CRITERION 1: K-block position consistency (low variance)
   CRITERION 2: K-block position away from boundaries (safe zone)
   CRITERION 3: Contiguous ring selection (maintain spatial coherence)
   ```

3. **Realizing Limitations:**
   - **6-segment tunnels**: Subsection selection can be highly effective
   - **7-segment tunnels**: Limited effectiveness due to geometric constraints

**The Solution:**
Created `p4tun/utils/select_optimal_subsection.py` utility that:
- Analyzes full tunnel K-block positions
- Finds optimal contiguous ring ranges
- Recommends subsections with minimal wraparound

---

### 2.3 The Wraparound Visualization Challenge

**User Confusion:**
```
USER: "I do not understand [the wraparound explanation]. Can you mark on the
4-1 depth_map the wraparound issues to make it intuitive?"
```

**The Challenge:**
- Wraparound is a **conceptual issue** - the left and right edges are the same physical location
- Visualizing this on a 2D image is inherently difficult
- Users need to understand the circular nature of the tunnel

**Attempted Solutions:**
1. Created visualization script (`mark_wraparound_simple.py`)
2. Added detailed explanations with analogies (pizza slices, circular tunnel)
3. Created conceptual diagrams showing the 3D→2D transformation

**Key Learning:**
- Some concepts require **multiple explanations** from different angles
- Visual aids help, but conceptual understanding is more important
- The "pizza slice" analogy was effective for explaining the geometric constraint

---

### 2.4 Decision to Remove Wraparound Functionality

**The Code Complexity:**
```
BEFORE: crop_image_and_mask_logits() had:
- enable_y_wraparound parameter
- Complex stitching logic for Y-wraparound
- apply_mask_logits_with_y_wraparound() function
- Special handling for theta-seam wraparound

AFTER: Simplified to:
- Standard image cropping with clamping
- No wraparound-specific logic
- Cleaner, more maintainable code
```

**The Reasoning:**
1. **Complexity vs. Benefit Analysis:**
   - Wraparound handling added ~200 lines of complex code
   - Benefits were marginal for most use cases
   - True wraparound issues should be fixed at unfolding stage

2. **Maintenance Burden:**
   - Complex coordinate transformations
   - Edge cases and bugs
   - Difficult to test and verify

3. **Better Solution:**
   - Fix wraparound at the source (unfolding stage)
   - Use subsection selection for experiments
   - Keep segmentation stage simple and focused

**The Decision:**
```
THOUGHT: "If wraparound is causing problems, we should fix it at the unfolding
stage where it originates, not add complex workarounds in the segmentation stage.
The segmentation code should be simple and focused on its core task."
```

---

### 2.5 Mistakes Made and Lessons Learned

#### Mistake 1: Over-Engineering the Solution

**What Happened:**
```
WRONG APPROACH: Added complex Y-wraparound handling in segmentation stage
REASONING: "We need to handle wraparound where it affects segmentation"
```

**Why It Was Wrong:**
- Wraparound is a data quality issue, not a segmentation algorithm issue
- Fixing it downstream adds complexity without addressing root cause
- Better to fix at the source (unfolding) or avoid through subsection selection

**Lesson:** Fix problems at their source, not with downstream workarounds.

---

#### Mistake 2: Assuming All Tunnels Need Wraparound Handling

**What Happened:**
```
ASSUMPTION: "All tunnels have wraparound issues, so we need universal handling"
REALITY: Only 7-segment tunnels have severe wraparound; 6-segment tunnels are manageable
```

**Why It Was Wrong:**
- Added complexity for cases that don't need it
- Made code harder to understand and maintain
- Created false sense that wraparound was "solved"

**Lesson:** Understand the problem scope before implementing solutions. Not all cases need the same treatment.

---

#### Mistake 3: Not Considering Subsection Selection Earlier

**What Happened:**
```
MISSED OPPORTUNITY: Could have selected better subsections when creating experimental datasets
IMPACT: Inadvertently created datasets with high wraparound severity
```

**Why It Was a Mistake:**
- Subsection selection is a simple, effective mitigation strategy
- Could have avoided wraparound issues from the start
- Would have made experiments cleaner and results more interpretable

**Lesson:** Consider data quality and selection criteria when creating experimental datasets, not just after problems arise.

---

### 2.6 What Made Success Possible

#### Success Factor 1: Systematic Analysis

**Process:**
1. Analyzed segment coverage for all tunnels
2. Identified patterns (6-segment vs 7-segment differences)
3. Quantified wraparound severity
4. Formulated mitigation strategies

**Result:** Clear understanding of the problem scope and effective solutions.

---

#### Success Factor 2: Question-Driven Exploration

**Key Questions:**
- "Can subsection selection avoid wraparound?"
- "Why don't I understand the wraparound explanation?"
- "Is the complex wraparound code worth maintaining?"

**Result:** Each question led to deeper understanding and better solutions.

---

#### Success Factor 3: Willingness to Simplify

**Decision Process:**
```
COMPLEXITY: 200+ lines of wraparound handling code
BENEFIT: Marginal improvement for edge cases
DECISION: Remove it, fix at source instead
```

**Result:** Cleaner codebase, clearer responsibilities, better maintainability.

---

## Part 3: Detailed Thought Process Log

### 3.1 Initial Hypothesis Formation

**Starting Point Thoughts:**
```
THOUGHT: "The user provided segment coverage diagrams showing wraparound patterns.
Tunnels 4-1 and 5-1 have severe wraparound with ALL segments spanning boundaries.
This is a critical finding that needs documentation and analysis."

HYPOTHESIS 1: "Wraparound severity is related to segment count"
→ Need to verify: Compare 6-segment vs 7-segment tunnel patterns

HYPOTHESIS 2: "Subsection selection could mitigate wraparound"
→ Need to analyze: Can we select rings where K-blocks are better aligned?

HYPOTHESIS 3: "Wraparound handling in code is necessary"
→ Need to evaluate: Is the complexity worth the benefit?
```

---

### 3.2 Thought Chain: Discovering Subsection Selection Strategy

**Initial Question:**
```
USER: "Will it be avoidable if we cut subsection based on certain criteria?"

THOUGHT: "This is a great question! If we could select ring ranges where:
1. K-blocks are in similar angular positions (low variance)
2. K-blocks are away from the 0°/360° boundary
3. Rings are contiguous (maintain spatial coherence)

We might be able to reduce wraparound severity significantly."
```

**Analysis Process:**
```
STEP 1: Understand how subsections are created
→ Currently: Arbitrary ring range selection
→ Problem: No consideration of K-block alignment

STEP 2: Formulate selection criteria
→ Criterion 1: K-block position variance < 15°
→ Criterion 2: K-blocks in safe zone (45°-315°)
→ Criterion 3: Contiguous rings (min 10 rings)

STEP 3: Evaluate effectiveness
→ 6-segment tunnels: Highly effective (can reduce to 0-1 boundary segments)
→ 7-segment tunnels: Limited effectiveness (still 5-6 boundary segments)
```

**Implementation Thinking:**
```
DESIGN: Create utility script that:
1. Loads full tunnel data
2. Computes K-block positions per ring
3. Finds optimal contiguous ranges
4. Recommends top N subsections

BENEFIT: Future experiments can use this to select better subsections
TRADE-OFF: Requires full tunnel analysis first
```

---

### 3.3 Thought Chain: Understanding User Confusion

**User Feedback:**
```
USER: "I do not understand [wraparound explanation]"

THOUGHT: "The wraparound concept is inherently difficult because:
1. It requires understanding 3D→2D transformation
2. The left and right edges are the same physical location
3. This is counter-intuitive when looking at a flat image

I need to:
1. Provide visual annotations on the depth map
2. Use analogies (pizza slices, circular tunnel)
3. Explain the geometric constraint clearly"
```

**Visualization Approach:**
```
IDEA 1: Mark left and right edges with same color
IDEA 2: Draw arrows showing wraparound connection
IDEA 3: Add text annotations explaining the concept
IDEA 4: Create conceptual diagrams (3D circle → 2D rectangle)
```

**Key Insight:**
```
REALIZATION: "Visual aids help, but the core issue is conceptual understanding.
The user needs to internalize that the tunnel is circular, not linear.
Once that's understood, wraparound makes sense."
```

---

### 3.4 Thought Chain: Decision to Remove Wraparound Code

**Initial State:**
```
OBSERVATION: Code has complex Y-wraparound handling:
- enable_y_wraparound parameter
- crop_image_and_mask_logits() with stitching logic
- apply_mask_logits_with_y_wraparound() function
- Special coordinate transformations

THOUGHT: "This is a lot of code. Is it actually necessary?"
```

**Analysis:**
```
QUESTION 1: "When does this code actually help?"
→ Answer: Only for edge cases where segments span Y-boundary
→ But: True wraparound is in X-direction (theta), not Y-direction

QUESTION 2: "What's the maintenance cost?"
→ Answer: High - complex coordinate transformations, edge cases, bugs

QUESTION 3: "Where should wraparound be handled?"
→ Answer: At unfolding stage (fix at source) or through subsection selection (avoid)
```

**Decision Process:**
```
OPTION 1: Keep wraparound code
→ Pros: Handles edge cases
→ Cons: Complexity, maintenance burden, unclear benefits

OPTION 2: Remove wraparound code, fix at source
→ Pros: Simpler code, clearer responsibilities, better maintainability
→ Cons: Need to ensure unfolding stage handles wraparound properly

DECISION: Option 2 - Remove wraparound code from segmentation stage
REASONING: "Segmentation should be simple. Wraparound is a data quality issue
that should be fixed at the unfolding stage or avoided through subsection selection."
```

**Implementation:**
```
ACTION: Removed from 4-2_sam.py:
- enable_y_wraparound parameter
- Y-wraparound stitching logic in crop_image_and_mask_logits()
- apply_mask_logits_with_y_wraparound() function
- Special coordinate handling

RESULT: ~200 lines of code removed, simpler and more maintainable
```

---

### 3.5 Dead Ends and What Didn't Work

**Dead End 1: Trying to Visualize Wraparound on Static Image**

**What We Tried:**
- Marking left and right edges
- Drawing arrows showing connection
- Adding text annotations

**Why It Didn't Fully Work:**
- Static image can't show the circular nature
- Requires conceptual leap to understand 3D→2D transformation
- Some users need multiple explanations from different angles

**Lesson:** Some concepts require conceptual understanding that can't be fully conveyed through static visualizations alone.

---

**Dead End 2: Assuming Universal Wraparound Handling Was Needed**

**What We Assumed:**
- All tunnels need wraparound handling
- Complex code is necessary for robustness

**Why It Was Wrong:**
- Only 7-segment tunnels have severe wraparound
- 6-segment tunnels are manageable with simpler approaches
- Better to fix at source than add downstream complexity

**Lesson:** Understand problem scope before implementing universal solutions.

---

**Dead End 3: Trying to Fix Wraparound at Segmentation Stage**

**What We Did:**
- Added Y-wraparound handling in segmentation
- Complex coordinate transformations
- Special case handling

**Why It Was a Dead End:**
- Wraparound is a data quality issue, not a segmentation algorithm issue
- Fixing downstream doesn't address root cause
- Adds complexity without solving the real problem

**Lesson:** Fix problems at their source, not with downstream workarounds.

---

## Part 4: Key Parameters by Stage

### Stage 1: Unfolding Parameters

| Parameter | Type | Default | Range | Sensitivity | Description |
|-----------|------|---------|-------|-------------|-------------|
| `theta_offset` | float | 0.0 | [0, 120] | HIGH | Starting angle for unwrapping; affects wraparound |
| `theta_coverage` | float | 1.0 | [0.99, 1.01] | CRITICAL | Target coverage ratio; must be ~100% |
| `resolution` | float | 0.005 | [0.003, 0.01] | MEDIUM | Meters per pixel; affects detail level |
| `per_ring_alignment` | bool | False | - | MEDIUM | Align K-blocks across rings to consistent position |

**Critical Insight:** `theta_coverage` must be exactly ~100% to avoid wraparound. `theta_offset` can help for 6-segment tunnels but is ineffective for 7-segment tunnels.

---

### Stage 2: Denoising Parameters

| Parameter | Type | Default | Range | Sensitivity | Description |
|-----------|------|---------|-------|-------------|-------------|
| `outlier_threshold` | float | 0.1 | [0.05, 0.2] | MEDIUM | Distance threshold for outlier detection |
| `neighbor_count` | int | 20 | [10, 50] | LOW | Number of neighbors for local analysis |
| `std_multiplier` | float | 2.0 | [1.5, 3.0] | MEDIUM | Standard deviations for outlier cutoff |

**Note:** Denoising parameters are not directly affected by wraparound, but preserving edges is critical.

---

### Stage 3: Enhancing Parameters

| Parameter | Type | Default | Range | Sensitivity | Description |
|-----------|------|---------|-------|-------------|-------------|
| `interpolation_radius` | float | 0.1 | [0.05, 0.2] | MEDIUM | Search radius for interpolation |
| `curvature_k_neighbors` | int | 20 | [15, 30] | LOW | Points for curvature calculation |
| `upsampling_target_distance` | float | 0.05 | [0.02, 0.10] | LOW | Upsampling density |

**Note:** Enhancing helps fill gaps but cannot fix wraparound issues.

---

### Stage 4-1: Detection Parameters

| Parameter | Type | Default | Range | Sensitivity | Description |
|-----------|------|---------|-------|-------------|-------------|
| `binary_threshold` | int | 120 | [50, 200] | HIGH | Threshold for binary conversion |
| `hough_threshold` | int | 40 | [20, 100] | HIGH | Vote threshold for line detection |
| `angle_min` | float | 5.0 | [3, 8] | HIGH | Min angle for oblique lines (degrees) |
| `angle_max` | float | 10.0 | [8, 15] | HIGH | Max angle for oblique lines (degrees) |
| `k_max_width_px` | int | 280 | [200, 350] | HIGH | Max width for K-block candidates (pixels) |

**Note:** Detection works per-ring, so wraparound has limited impact on detection accuracy.

---

### Stage 4-2: SAM Segmentation Parameters

| Parameter | Type | Default | Range | Sensitivity | Description |
|-----------|------|---------|-------|-------------|-------------|
| `K_height` | float | 1079.92 | [1100, 1500] | MEDIUM | K-block height in mm |
| `AB_height` | float | 3239.77 | [3400, 3800] | VERY HIGH | A/B block height in mm |
| `segment_width` | float | 1200.0 | [1000, 1800] | MEDIUM | Segment width in mm |
| `template_coverage_ratio` | float | 0.95 | [0.85, 0.99] | MEDIUM | How far inside template to place points |
| `mask_eps` | float | 0.001 | [0.0001, 0.01] | LOW | Epsilon for mask logits computation |

**Removed Parameters:**
- ~~`enable_y_wraparound`~~ - Removed (was bool, default True)
- ~~Y-wraparound stitching logic~~ - Removed from `crop_image_and_mask_logits()`
- ~~`apply_mask_logits_with_y_wraparound()`~~ - Function removed

**Critical Change:** Y-wraparound handling removed. Standard image cropping with clamping is now used. Wraparound should be addressed at unfolding stage or avoided through subsection selection.

---

### Subsection Selection Parameters (New Utility)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `min_rings` | int | 10 | [5, 50] | Minimum rings in subsection |
| `max_variance` | float | 225.0 | [100, 500] | Maximum acceptable K-block variance (degrees²) |
| `k_safe_zone_min` | float | 45.0 | [0, 90] | Minimum angle for safe K-block position |
| `k_safe_zone_max` | float | 315.0 | [270, 360] | Maximum angle for safe K-block position |
| `prefer_longer` | bool | True | - | Prefer longer ranges with same variance |

**Usage:**
```bash
python p4tun/utils/select_optimal_subsection.py <tunnel_id> \
    --min-rings 10 \
    --max-variance 225.0 \
    --top-n 5
```

---

## Part 5: Complete Thought Experience Log

### 5.1 Initial Understanding Phase

**Thought Process:**
```
THOUGHT: "User provided segment coverage diagrams. This is valuable data showing
wraparound patterns. I need to:
1. Verify the information
2. Create a comprehensive report
3. Analyze implications for pipeline processing

The key observation is that 7-segment tunnels (4-1, 5-1) have ALL segments
spanning boundaries, which is a critical finding."
```

**Action Taken:**
- Created `SEGMENT_COVERAGE_REPORT.md` with detailed analysis
- Verified segment counts and boundary-spanning patterns
- Provided recommendations for each tunnel type

---

### 5.2 Subsection Selection Discovery

**Thought Process:**
```
USER QUESTION: "Will it be avoidable if we cut subsection based on certain criteria?"

THOUGHT: "This is an excellent question! If we could select ring ranges where
K-blocks are better aligned, we might reduce wraparound. Let me think about
the criteria:

1. K-block position consistency (low variance across rings)
2. K-block position away from boundaries (safe zone)
3. Contiguous rings (maintain spatial coherence)

For 6-segment tunnels, this could be highly effective. For 7-segment tunnels,
it might help but won't eliminate wraparound completely due to geometric constraints."
```

**Analysis Steps:**
1. Understood how subsections are currently created (arbitrary ring selection)
2. Formulated selection criteria based on K-block alignment
3. Evaluated effectiveness for different tunnel types
4. Created utility script for future use

**Key Insight:**
```
REALIZATION: "Subsection selection is a simple, effective mitigation strategy
that could have been used from the start. It's much simpler than complex
wraparound handling code, and addresses the problem at the data selection stage."
```

---

### 5.3 Visualization Challenge

**Thought Process:**
```
USER: "I do not understand [wraparound explanation]"

THOUGHT: "The wraparound concept is inherently difficult because it requires
understanding a 3D→2D transformation. The left and right edges of the image
are the same physical location, which is counter-intuitive.

I need to:
1. Create visual annotations on the depth map
2. Use clear analogies (pizza slices, circular tunnel)
3. Explain the geometric constraint step-by-step"
```

**Attempted Solutions:**
1. Created visualization script (`mark_wraparound_simple.py`)
2. Added detailed explanations with multiple analogies
3. Created conceptual diagrams

**Learning:**
```
INSIGHT: "Some concepts require multiple explanations from different angles.
Visual aids help, but conceptual understanding is more important. The user
needs to internalize that the tunnel is circular, not linear."
```

---

### 5.4 Code Simplification Decision

**Thought Process:**
```
OBSERVATION: Code has complex Y-wraparound handling with ~200 lines of code.

THOUGHT: "Is this complexity necessary? Let me analyze:
1. When does this code actually help?
2. What's the maintenance cost?
3. Where should wraparound be handled?

The answers suggest:
- Only helps in edge cases
- High maintenance cost
- Should be handled at unfolding stage or avoided through subsection selection

DECISION: Remove wraparound code from segmentation stage. Keep it simple."
```

**Implementation:**
- Removed `enable_y_wraparound` parameter
- Simplified `crop_image_and_mask_logits()` function
- Removed `apply_mask_logits_with_y_wraparound()` function
- Result: ~200 lines of code removed, simpler and more maintainable

---

## Key Takeaways

1. **Fix Problems at Source**: Wraparound is a data quality issue that should be addressed at the unfolding stage, not with downstream workarounds.

2. **Subsection Selection is Effective**: For 6-segment tunnels, strategic ring selection can significantly reduce wraparound. For 7-segment tunnels, it helps but doesn't eliminate the issue.

3. **Simplicity Over Complexity**: Complex wraparound handling code was removed in favor of simpler solutions. Better to fix at source or avoid through data selection.

4. **Understand Problem Scope**: Not all tunnels need the same treatment. 6-segment tunnels are manageable; 7-segment tunnels require special handling.

5. **Conceptual Understanding Matters**: Wraparound is a difficult concept that requires understanding the 3D→2D transformation. Multiple explanations and analogies help.

6. **Tools for Future**: Created subsection selection utility for future experiments to avoid wraparound issues from the start.

---

## Recommendations for Future Work

### For New Experiments

1. **Always analyze full tunnel first** - Compute K-block positions for all rings
2. **Use subsection selection utility** - Select optimal ring ranges before creating experimental datasets
3. **Document selection criteria** - Record why specific subsections were chosen

### For Existing Datasets

1. **Retrospectively analyze** - Check if better subsections could be selected
2. **Re-evaluate if needed** - Consider re-selecting if wraparound is problematic

### For Code Maintenance

1. **Keep segmentation simple** - Don't add complex wraparound handling
2. **Fix at source** - Address wraparound at unfolding stage
3. **Use subsection selection** - Avoid wraparound through data selection

---

**Report Generated:** January 23, 2026  
**Next Steps:**
- Test subsection selection utility on full tunnels
- Retrospectively analyze existing datasets
- Document best practices for future experiments
