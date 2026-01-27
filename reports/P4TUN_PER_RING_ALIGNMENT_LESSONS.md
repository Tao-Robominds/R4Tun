# P4Tun Per-Ring Alignment Exploration: A Comprehensive Lessons Learned Report
## From Theory to Failure: Understanding Why Per-Ring Alignment Doesn't Work

**Date:** January 23, 2026  
**Focus Tunnel:** 3-1 (6-segment configuration)  
**Objective:** Universal solution for wraparound issues across all tunnels  
**Outcome:** Complete failure - approach fundamentally flawed

---

## Executive Summary

This report documents an exploration into solving the tunnel segmentation wraparound problem through "per-ring alignment" - a theoretically elegant but practically disastrous approach. The key lesson: **spatial coherence across rings is fundamental to the depth map representation, and any approach that breaks this coherence will fail catastrophically.**

**Final Verdict:** Per-ring alignment must be **abandoned**. The correct approach is to use a global theta_offset or handle wraparound at the SAM level through cropping/stitching.

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Theta Coverage %** | `(theta_max - theta_min) / (2π) × 100` | 99-101% | Coverage outside this range causes wraparound or gaps |
| **K-Block Angular Consistency** | Std dev of K-block theta across rings | <30° for aligned, ~60° for alternating | High spread indicates per-ring variation requiring special handling |
| **Ring Count Match** | Detected rings vs expected rings | 100% match | Missing rings indicate ellipse fitting failures |
| **Centerline R² Score** | Polynomial fit quality | >0.99 | Poor fit distorts theta calculation |
| **Point Retention Rate** | Points after filtering / total points | >90% | Low retention indicates data quality issues |

**Critical Finding from This Exploration:**
```
Tunnel 3-1 K-block positions BEFORE alignment:
  Ring 1: 154.6°
  Ring 2: 180.2°
  Ring 3: 183.4°
  Ring 4: 182.9°
  Ring 5: 204.1°
  Ring 6: 156.3°
  
Spread: 49.5° (indicating some per-ring variation but NOT extreme)

WRONG ASSUMPTION: "Aligning all K-blocks to 180° will help"
REALITY: It destroyed spatial coherence completely
```

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Noise Removal Rate** | % points classified as noise | 70-85% | Too low = noise retained; too high = surface damage |
| **Surface Coverage** | % of expected tunnel surface covered | >95% | Gaps in coverage cause detection failures |
| **Radius Distribution** | Std dev of point radii | <0.1m | High variance indicates fit issues or noise |

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Depth Map Fill Rate** | Non-zero pixels / total pixels | >80% | Low fill causes interpolation artifacts |
| **Edge Sharpness** | Gradient magnitude at boundaries | >0.5 relative | Determines segment boundary visibility |
| **Upsampling Point Count** | New points added | 30K-50K | Too few = sparse; too many = over-interpolated |

---

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Oblique Line Count** | Positive + negative slope lines | >10 total | Too few = unreliable K-block detection |
| **K-Block Y Position Consistency** | Std dev of detected Y across rings | Pattern-dependent | Should match expected pattern (aligned vs alternating) |
| **Detection Method Distribution** | % midpoint vs inferred | >50% midpoint | Midpoint is most reliable |
| **Y Position Error** | |detected_Y - GT_Y| mean | <150 pixels | Direct quality measure |

**Key Parameters Discovered:**
```
Parameter                      Sensitivity    Discovered Value
──────────────────────────────────────────────────────────────
binary_threshold               MEDIUM         127
dilation_iterations            LOW            2
hough_oblique_threshold        HIGH           29
hough_oblique_min_length       HIGH           46
angle_positive_min             VERY HIGH      3.4°
angle_positive_max             VERY HIGH      9.8°
```

---

### Stage 4-2: SAM Segmentation (4-2_sam.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Per-Class IoU** | IoU for each segment type | >0.3 | Direct segmentation quality |
| **Background Ratio** | % classified as background | <40% | High = undersized templates |
| **K-Block IoU** | IoU specifically for K segments | >0.2 | K is anchor for all segments |
| **Depth Map Dimensions Consistency** | Width/Height vs expected | Within 5% | Changed dimensions invalidate parameters |

**Critical SAM Parameters:**
```
Parameter                      Sensitivity    Range
──────────────────────────────────────────────────────────────
y_bounds                       CRITICAL       Must match depth map height
segment_width                  HIGH           1100-1250mm
k_height                       MEDIUM         1000-1160mm
ab_height                      VERY HIGH      3100-3400mm
template mask dimensions       HIGH           Must cover full segment
```

---

## Part 2: The Thought Process Experience

### 2.1 The Genesis of Per-Ring Alignment

**Initial Problem Statement:**
```
CONTEXT: Tunnel 3-1 had A2 segment with 65.8% wraparound
- Points appearing at both top AND bottom of depth map
- SAM couldn't segment split segments
- Global theta_offset couldn't fix all segments simultaneously

THOUGHT: "What if we apply a DIFFERENT theta offset to each ring,
so that K-blocks align perfectly to 180° across ALL rings?"

HYPOTHESIS: "If K-blocks are at consistent theta, segments won't wrap around"
```

**Why This Seemed Like a Good Idea:**
```
REASONING:
1. K-blocks have distinctive depth patterns (oblique edges)
2. We can detect K-block center for each ring
3. If we shift each ring's theta so K is at 180°...
4. ...then all K-blocks would align vertically in the depth map!

EXPECTED BENEFIT:
- No wraparound (K is never at 0°/360° boundary)
- Simpler detection (K always at same Y position)
- Universal solution (works for any tunnel)
```

---

### 2.2 Implementation and Initial "Success"

**What I Built:**
```python
def compute_per_ring_k_centers(df, cylindrical_coords, diameter):
    """Compute K-block center theta for each ring."""
    # Used depth values to identify K (distinctive pattern)
    
def compute_per_ring_offsets(k_centers, target_position=180.0):
    """Compute offset for each ring to align K to target."""
    offsets = {}
    for ring, k_center in k_centers.items():
        offsets[ring] = target_position - k_center
    return offsets
    
def apply_per_ring_offsets(df, cylindrical, offsets, diameter):
    """Apply per-ring theta shifts."""
    for ring, offset in offsets.items():
        ring_mask = df['ring'] == ring
        theta_deg = cylindrical[ring_mask, 1] * 360 / circumference
        theta_deg = (theta_deg + offset) % 360  # SHIFT BY OFFSET
        cylindrical[ring_mask, 1] = theta_deg * circumference / 360
```

**The "Promising" Output:**
```
Step 5b: Computing per-ring K block alignment...
  Found K blocks in 6 rings
  K positions before alignment: min=154.6°, max=204.1°, spread=49.5°
  Applied per-ring offsets to align K blocks to 180.0°
  K positions after alignment: min=180.0°, max=180.0°, spread=0.0°

THOUGHT: "Perfect! K-blocks are now perfectly aligned at 180°!"
```

---

### 2.3 The Catastrophic Failure

**First Sign of Trouble:**
```
Ran detection on aligned depth map:
  Detection types: {'aligned_infer': 4, 'positive_slope': 1, 'alternation_infer': 1}
  
OBSERVATION: "Only 1 positive_slope detection? The oblique lines should be clearer now..."

Ran SAM:
  mIoU: 0.102 (down from 0.378 baseline!)
  
THOUGHT: "Something went terribly wrong. How can aligned K-blocks give WORSE results?"
```

**The Devastating Visualization:**
```
USER: "i do not think it is working. the per-ring approach. 
@data/3-1/depth_map.png looks completely distorted"

OPENED THE IMAGES:
- Baseline depth map: Clear tunnel structure, visible segments, proper grid
- Per-ring aligned depth map: COMPLETELY FRAGMENTED, no recognizable structure

REALIZATION: "Oh no. The depth map is destroyed."
```

---

### 2.4 Understanding WHY It Failed

**The Fundamental Flaw:**
```
THE WRONG ASSUMPTION:
"Each ring can be shifted independently"

THE REALITY:
Segments span MULTIPLE rings. A1 in Ring 1 is physically connected 
to A1 in Ring 2, Ring 3, etc. They must maintain spatial coherence.

WHAT HAPPENS WITH PER-RING SHIFTS:
- Ring 1: Shifted by +25°
- Ring 2: Shifted by +0°
- Ring 3: Shifted by -3°
- Ring 4: Shifted by -3°
- Ring 5: Shifted by -24°
- Ring 6: Shifted by +24°

A continuous segment that spans rings 1-6 now has its parts
scattered to DIFFERENT Y positions in the depth map!
```

**Visual Explanation:**
```
BEFORE (correct):
Ring 1: |--A1--|--B1--|--K--|--A2--|--B2--|--A3--|
Ring 2: |--A1--|--B1--|--K--|--A2--|--B2--|--A3--|
Ring 3: |--A1--|--B1--|--K--|--A2--|--B2--|--A3--|
        ↑ Segments are vertically aligned (continuous)

AFTER PER-RING ALIGNMENT (broken):
Ring 1:     |--A1--|--B1--|--K--|--A2--|--B2--|--A3--|  (shifted right)
Ring 2: |--A1--|--B1--|--K--|--A2--|--B2--|--A3--|      (no shift)
Ring 3:  |--A1--|--B1--|--K--|--A2--|--B2--|--A3--|     (shifted left)
         ↑ Segments are NO LONGER ALIGNED (fragmented)
```

**The Depth Map Interpretation:**
```
Depth map Y-axis = theta (angular position)
Depth map X-axis = h (position along tunnel axis)

Each pixel column represents one position along the tunnel.
Each pixel row represents one angular position around the circumference.

When we shift theta differently per ring:
- The SAME physical point now appears at DIFFERENT Y positions
- What was a continuous surface becomes fragmented
- The depth map no longer represents a coherent unwrapped cylinder
```

---

### 2.5 Key Mistakes Made

#### Mistake 1: Ignoring Spatial Coherence
```
WRONG THOUGHT: "We can transform each ring independently"

CORRECT UNDERSTANDING: "The depth map is a 2D projection of a 3D surface.
Segments are continuous surfaces that span multiple rings.
Any transformation must preserve this spatial relationship."

LESSON: Always think about what the data REPRESENTS, not just its numerical values.
```

#### Mistake 2: Testing on Metrics Before Visual Inspection
```
WRONG APPROACH: "K positions are now at 180° ± 0°. Success!"

SHOULD HAVE DONE: "Let me visually inspect the depth map BEFORE running metrics"

LESSON: Numerical metrics can be misleading. The K-blocks WERE at 180°,
but the rest of the image was destroyed. Always visualize intermediate results.
```

#### Mistake 3: Not Considering the Full Pipeline Impact
```
WRONG THOUGHT: "If K-blocks align, detection will be easier"

REALITY: Detection looks for OBLIQUE LINES. Per-ring shifts break the
linear patterns that Hough transform depends on.

LESSON: Changes to early pipeline stages cascade through all downstream stages.
Understand how each stage uses the output of previous stages.
```

#### Mistake 4: Optimistic Interpretation of Theory
```
WRONG: "Mathematically, per-ring alignment should work"

CORRECT: "The math is correct, but the assumptions are wrong.
The depth map isn't just a collection of independent rings - 
it's a coherent representation of a continuous surface."

LESSON: Mathematical correctness doesn't guarantee practical success.
Always validate assumptions against physical reality.
```

---

### 2.6 What Would Have Worked Instead

**Option 1: Global Theta Offset (Already Exists)**
```
Apply the SAME offset to ALL rings:
- Preserves spatial coherence
- Moves the "cut" position uniformly
- Can position K-blocks away from 0°/360° boundary
- Limitation: Can't fix ALL tunnels (some have per-ring K variation)
```

**Option 2: Wraparound Handling at SAM Level**
```
Accept that some segments wrap around, but handle it in SAM:
- Detect wraparound segments
- Process them specially (crop top AND bottom, stitch masks)
- More complex but preserves depth map integrity
```

**Option 3: Ring-by-Ring SAM Processing (But NOT Ring-by-Ring Unfolding)**
```
Keep the depth map coherent, but:
- Run SAM per-ring with individual prompts
- Aggregate masks across rings
- Handle edge cases per-ring while maintaining global structure
```

---

### 2.7 The Recovery Process

**Steps Taken After Realizing the Failure:**
```
1. Disabled per_ring_alignment in parameters
2. Restored baseline data from data/baseline/3-1/
3. Verified depth map structure was correct
4. Re-ran detection → Got expected K positions
5. Re-ran SAM → mIoU back to 0.378 (baseline level)

LESSON: Always keep backups. The ability to restore baseline 
saved hours of re-processing.
```

---

## Part 3: Complete Thought Process Log

### 3.1 Initial Analysis Phase

**Thought 1: Understanding the Wraparound Problem**
```
USER CONTEXT: "3-1 has wraparound issue for A2 segment"

MY ANALYSIS:
- Checked enhanced.csv for segment theta distributions
- Found A2 spans theta 0° to ~65° AND theta ~295° to 360°
- This means A2 appears at BOTH top and bottom of depth map
- SAM sees this as TWO separate regions, not one segment

THOUGHT: "A global theta_offset could shift A2 away from the boundary,
but then another segment might wrap. Is there a universal solution?"
```

**Thought 2: The Per-Ring Alignment Idea**
```
OBSERVATION: K-blocks have variable theta across rings:
- Tunnel 3-1: K varies from 130° to 180°
- Tunnel 4-1: K varies randomly across rings
- Tunnel 1-4: K alternates between ~75° and ~199°

THOUGHT: "If K position varies, that means segments are ROTATED 
differently in each ring. What if we UN-ROTATE them?"

HYPOTHESIS: "Apply per-ring offsets to align K-blocks,
which would also align all other segments."

THIS SEEMED ELEGANT: "One algorithm to fix ALL wraparound issues!"
```

### 3.2 Implementation Phase

**Thought 3: Designing the Algorithm**
```
ALGORITHM DESIGN:
1. For each ring, find K-block center (using depth pattern analysis)
2. Compute offset needed to move K to 180°
3. Apply offset to ALL points in that ring
4. Result: K at 180° in all rings, no wraparound possible

IMPLEMENTATION APPROACH:
- Add per_ring_alignment parameter to parameters_unfolding.json
- Add k_target_position parameter (default 180°)
- Implement compute_per_ring_k_centers() function
- Implement apply_per_ring_offsets() function
- Call after main cylindrical transform

THOUGHT: "This is clean and parameterized. Should be easy to enable/disable."
```

**Thought 4: The "Success" Moment**
```
RAN THE CODE:
"K positions after alignment: min=180.0°, max=180.0°, spread=0.0°"

MY REACTION: "Perfect! The algorithm works exactly as designed.
K-blocks are now perfectly aligned."

CRITICAL MISTAKE HERE: I celebrated the numerical success
without checking the visual result.
```

### 3.3 Failure Discovery Phase

**Thought 5: When Things Went Wrong**
```
RAN DETECTION:
"Detection types: {'aligned_infer': 4, 'positive_slope': 1}"

THOUGHT: "Hmm, mostly inferred positions, only 1 actual detection.
That's worse than before. Strange."

RAN SAM + EVALUATION:
"mIoU: 0.102"

THOUGHT: "0.102?! That's terrible. Baseline was 0.378.
Something is fundamentally broken."
```

**Thought 6: The Devastating Realization**
```
USER SHOWED ME THE DEPTH MAPS:
- Baseline: Beautiful, clear tunnel structure
- Per-ring aligned: Fragmented, unrecognizable mess

MY REACTION: "Oh. Oh no. I broke everything."

IMMEDIATE UNDERSTANDING:
"The depth map isn't just data points - it's a coherent 2D representation
of a 3D surface. By shifting rings independently, I tore the surface apart."
```

### 3.4 Understanding Phase

**Thought 7: Why Spatial Coherence Matters**
```
PHYSICAL REALITY:
- Segment A1 is a continuous piece of concrete
- It exists at the same angular position in rings 1, 2, 3...
- In the depth map, it should appear as a vertical band

PER-RING ALIGNMENT EFFECT:
- Ring 1 shifted by +25°: A1 moves right
- Ring 2 shifted by 0°: A1 stays
- Ring 3 shifted by -3°: A1 moves slightly left

RESULT: A1 is no longer a vertical band - it's a jagged, 
fragmented pattern that the detection and SAM can't interpret.
```

**Thought 8: The Fundamental Assumption Was Wrong**
```
MY ASSUMPTION: "Rings are independent units that can be transformed separately"

REALITY: "Rings are slices of a continuous surface. 
Transformations must preserve continuity between adjacent rings."

THE FIX: Per-ring alignment is mathematically valid but 
SEMANTICALLY WRONG for this representation.
```

### 3.5 Recovery Phase

**Thought 9: Damage Control**
```
IMMEDIATE ACTIONS:
1. Disabled per_ring_alignment: false in parameters
2. Restored baseline files from data/baseline/3-1/
3. Verified depth map was correct visually

VERIFICATION:
- Depth map: 2925x1495 (correct)
- Detection: 6 K positions with proper types
- mIoU: 0.378 (back to baseline)

LESSON: Always keep clean backups before experimental changes.
```

**Thought 10: What I Should Have Done Differently**
```
BEFORE IMPLEMENTATION:
- Should have thought through what the depth map REPRESENTS
- Should have considered how segments span multiple rings
- Should have visualized a simple example first

DURING IMPLEMENTATION:
- Should have added visualization checkpoints
- Should have tested on a tiny subset first

AFTER INITIAL RUN:
- Should have looked at the depth map BEFORE running metrics
- Should have compared against baseline visually
```

### 3.6 Lessons Crystallized

**Thought 11: Key Takeaways**
```
1. DATA REPRESENTATION MATTERS
   The depth map isn't just numbers - it's a coherent unwrapped cylinder.
   Any transformation must respect this semantic meaning.

2. VISUALIZE BEFORE MEASURING
   mIoU of 0.102 told me something was wrong, but visual inspection
   would have revealed the problem immediately.

3. SPATIAL COHERENCE IS FUNDAMENTAL
   Tunnel segments span multiple rings. They must maintain
   continuity in the depth map representation.

4. ELEGANT THEORY ≠ PRACTICAL SUCCESS
   Per-ring alignment is mathematically sound but practically useless
   because it violates the fundamental structure of the data.

5. TEST INCREMENTALLY
   I should have tested on one ring, then two, then the full tunnel.
   The fragmentation would have been obvious with just 2-3 rings.
```

---

## Part 4: Summary and Recommendations

### 4.1 What NOT To Do

```
❌ Per-ring theta alignment - destroys spatial coherence
❌ Any transformation that treats rings independently
❌ Optimizing metrics without visual verification
❌ Implementing complex changes without incremental testing
```

### 4.2 What To Do Instead

```
✅ Use global theta_offset for uniform wraparound control
✅ Handle wraparound at SAM level with masking/stitching
✅ Always visualize intermediate results
✅ Respect the semantic meaning of data representations
✅ Test transformations incrementally (1 ring → 2 rings → full)
```

### 4.3 For Future Wraparound Solutions

```
RECOMMENDED APPROACH:
1. Accept that some segments will wrap in some tunnels
2. Detect which segments wrap (analyze theta ranges)
3. For wrapped segments:
   - Extend depth map or
   - Process top/bottom crops separately and stitch
4. Never break spatial coherence at the unfolding level
```

---

## Appendix: Configuration That Was Tested (And Failed)

```json
// parameters_unfolding.json - THE BROKEN APPROACH (DO NOT USE)
{
    "wraparound": {
        "theta_offset": 0.0,
        "per_ring_alignment": true,  // ❌ CAUSES CATASTROPHIC FAILURE
        "k_target_position": 180.0
    }
}
```

```json
// parameters_unfolding.json - THE CORRECT APPROACH
{
    "wraparound": {
        "theta_offset": 0.0,  // Can adjust globally if needed
        "per_ring_alignment": false  // ✅ MUST BE FALSE
    }
}
```

---

## Part 5: Key Parameters Reference by Stage

This section documents the key parameters for each pipeline stage, their recommended values, sensitivity levels, and tuning guidelines based on exploration of Tunnel 3-1.

### 5.1 Stage 1: Unfolding Parameters

```json
// p4tun/parameters/3-1/parameters_unfolding.json
{
    "physical_constants": {
        "ring_spacing": 1.215,          // meters between rings
        "tunnel_diameter": 5.5          // meters
    },
    "slicing": {
        "slice_half_thickness": 0.007,  // ±7mm slice for ellipse fitting
        "max_distance_from_top": 5.5    // meters from tunnel crown
    },
    "curve_fitting": {
        "polynomial_degree": 2          // 2=quadratic, 3=cubic centerline
    },
    "ransac_ellipse": {
        "inlier_ratio": 0.75,           // 75% points must be inliers
        "confidence": 0.9,              // 90% confidence in result
        "min_samples": 6,               // minimum points for fitting
        "inlier_threshold": 0.4         // max distance to be inlier (meters)
    },
    "arc_length": {
        "samples_per_ring": 1210        // resolution of theta discretization
    }
}
```

| Parameter | Sensitivity | Recommended Range | Notes |
|-----------|-------------|-------------------|-------|
| ring_spacing | HIGH | Measure from data | Must match physical tunnel |
| tunnel_diameter | HIGH | Measure from data | Critical for theta calculation |
| slice_half_thickness | MEDIUM | 0.005-0.010 | Too small = missing points |
| polynomial_degree | LOW | 2-3 | 2 for straight, 3 for curved |
| inlier_threshold | MEDIUM | 0.3-0.8 | Tunnel-dependent noise level |

---

### 5.2 Stage 2: Denoising Parameters

```json
// p4tun/parameters/3-1/parameters_denoising.json
{
    "radius_filtering": {
        "radius_min": 2.8,              // minimum valid radius (meters)
        "radius_max": 3.0               // maximum valid radius (meters)
    },
    "grid_resolution": {
        "theta_step": 0.4,              // angular resolution (degrees)
        "radial_step": 0.0055           // radial resolution (meters)
    },
    "gradient_detection": {
        "gradient_threshold": 0.15,     // surface gradient cutoff
        "gradient_epsilon": 1e-06       // numerical stability
    },
    "cutoff_smoothing": {
        "smoothing_window": 5,          // window size for smoothing
        "smoothing_offset": -0.002      // offset for cutoff boundary
    }
}
```

| Parameter | Sensitivity | Recommended Range | Notes |
|-----------|-------------|-------------------|-------|
| radius_min/max | HIGH | diameter/2 ± 10% | Must capture tunnel surface |
| theta_step | MEDIUM | 0.3-0.7 | Finer = more detail, slower |
| radial_step | MEDIUM | 0.001-0.01 | Finer = better edge detection |
| gradient_threshold | HIGH | 0.1-0.3 | Controls noise vs signal tradeoff |
| smoothing_window | LOW | 3-7 | Larger = smoother boundaries |

---

### 5.3 Stage 3: Enhancing Parameters

```json
// p4tun/parameters/3-1/parameters_enhancing.json
{
    "curvature": {
        "curvature_neighbors": 20       // neighbors for curvature estimation
    },
    "upsampling": {
        "target_distances": [0.06, 0.03, 0.015],  // multi-scale upsampling
        "curvature_threshold": 0.005,   // high-curvature threshold
        "upsampling_neighbors": 20,     // neighbors for interpolation
        "distance_tolerance_low": 0.9,  // min distance factor
        "distance_tolerance_high": 2.0, // max distance factor
        "radius_filter_factor": 0.15    // radius consistency filter
    },
    "outlier_detection": {
        "depth_threshold_low": 0.005,   // min depth variation (meters)
        "depth_threshold_high": 0.015,  // max depth variation (meters)
        "outlier_neighbors": 20         // neighbors for outlier detection
    },
    "outlier_interpolation": {
        "interpolation_radius": 0.03,   // radius for filling holes
        "num_interpolations": 2,        // interpolation iterations
        "max_outlier_points": 5000      // limit for memory efficiency
    },
    "depth_map": {
        "resolution": 0.005,            // 5mm per pixel
        "interpolation_window": 9       // window for gap filling
    }
}
```

| Parameter | Sensitivity | Recommended Range | Notes |
|-----------|-------------|-------------------|-------|
| target_distances | MEDIUM | [0.06, 0.03, 0.015] | Multi-scale hierarchy |
| curvature_threshold | LOW | 0.001-0.01 | Focus upsampling on flat areas |
| depth_threshold_* | MEDIUM | 0.003-0.02 | Segment boundary detection |
| resolution | HIGH | 0.004-0.006 | Determines depth map size |
| interpolation_window | LOW | 5-11 | Odd numbers, larger = smoother |

---

### 5.4 Stage 4-1: Detection Parameters

```json
// p4tun/parameters/3-1/parameters_detection.json
{
    "preprocessing": {
        "binary_threshold": 127,        // image binarization threshold
        "dilation_kernel_size": 2,      // morphological dilation
        "dilation_iterations": 2        // number of dilations
    },
    "hough_oblique": {
        "threshold": 29,                // Hough accumulator threshold
        "min_length": 46,               // minimum line length (pixels)
        "max_gap": 52,                  // max gap in line (pixels)
        "angle_positive_min": 3.4,      // min positive slope angle (deg)
        "angle_positive_max": 9.8,      // max positive slope angle (deg)
        "angle_negative_min": -9.8,     // min negative slope angle (deg)
        "angle_negative_max": -3.4      // max negative slope angle (deg)
    },
    "hough_horizontal": {
        "threshold": 50,                // accumulator threshold
        "min_length": 100,              // minimum line length
        "max_gap": 15,                  // max gap in line
        "angle_tolerance": 1            // degrees from horizontal
    },
    "hough_vertical": {
        "threshold": 500                // accumulator threshold
    },
    "line_processing": {
        "merge_distance_threshold": 5,  // merge lines within distance
        "merge_close_threshold": 6      // merge close points
    },
    "physical_constants": {
        "resolution": 0.005,            // must match enhancing
        "k_height_mm": 1079.92,         // K-block height in mm
        "ab_height_mm": 3239.77         // AB-block height in mm
    }
}
```

| Parameter | Sensitivity | Recommended Range | Notes |
|-----------|-------------|-------------------|-------|
| binary_threshold | MEDIUM | 100-150 | Depends on depth map contrast |
| hough_oblique.threshold | HIGH | 25-60 | Lower = more lines, more noise |
| hough_oblique.min_length | HIGH | 40-120 | Tunnel-dependent |
| angle_positive_min/max | VERY HIGH | 3-10° | Must match tunnel geometry |
| hough_vertical.threshold | MEDIUM | 300-700 | Higher = fewer ring boundaries |

**Critical Insight:** The angle parameters (positive/negative min/max) are CRITICAL. They must match the oblique line angles in the specific tunnel. Wrong angles = no K-block detection.

---

### 5.5 Stage 4-2: SAM Parameters

```json
// p4tun/parameters/3-1/parameters_sam.json
{
    "segment_geometry": {
        "segment_width": 1100.0,        // segment width in mm
        "k_height": 1112.4,             // K-block height in mm
        "ab_height": 3400.0,            // AB-block height in mm
        "angle_deg": 6.5                // oblique angle in degrees
    },
    "processing": {
        "padding": 80,                  // padding around crops (pixels)
        "crop_margin": 45,              // margin for cropping (pixels)
        "mask_eps": 0.001,              // mask threshold
        "y_bounds": [4200, 13100]       // valid Y range for processing
    },
    "prompt_points": {
        "k_block": {
            "outer_ring": 682.77,       // outer prompt distance from center
            "middle_ring": 503.69,      // middle prompt distance
            "inner_ring": 315.71,       // inner prompt distance
            "center_ring": 318.67       // center prompt distance
        },
        "ab_blocks": {
            "outer_ring": 671.26,       // outer prompt distance
            "middle_ring": 647.69,      // middle prompt distance
            "inner_ring": 478.39,       // inner prompt distance
            "center_ring": 280.0,       // center prompt distance
            "fine_spacing": 221.61,     // fine grid spacing
            "ultra_fine": 147.03        // ultra-fine grid spacing
        },
        "template_mask": {
            "k_block": {
                "width": 705.67,        // K template width
                "height_pos": 655.0,    // K template height (positive)
                "height_neg": 655.0     // K template height (negative)
            },
            "a_blocks": {
                "width": 680.0,         // A template width
                "height": 1591.81       // A template height
            },
            "b1_block": {
                "width": 680.0,         // B1 template width
                "height_top": 1591.81   // B1 template height
            }
        }
    },
    "pattern_aware": {
        "use_quality_weighting": true,
        "min_quality_threshold": 0.444  // minimum detection quality
    }
}
```

| Parameter | Sensitivity | Recommended Range | Notes |
|-----------|-------------|-------------------|-------|
| segment_width | HIGH | 1100-1250mm | Physical segment width |
| k_height | MEDIUM | 1000-1160mm | K-block physical height |
| ab_height | VERY HIGH | 3100-3400mm | Critical for template sizing |
| y_bounds | CRITICAL | [0, depth_map_height] | MUST match depth map |
| template_mask.*.width | HIGH | 550-750mm | Too small = missed edges |
| template_mask.*.height | VERY HIGH | 1500-1800mm | Undersizing causes failures |
| min_quality_threshold | MEDIUM | 0.1-0.5 | Filter low-confidence detections |

**Critical Insight:** The `y_bounds` parameter MUST match the depth map dimensions. If the depth map is 2925 pixels tall, y_bounds should be [0, 2925], not [4200, 13100] (which is for a different tunnel configuration).

---

### 5.6 Cross-Stage Parameter Dependencies

```
DEPENDENCY CHAIN:
┌─────────────────────────────────────────────────────────────────┐
│ Unfolding                                                       │
│   tunnel_diameter → affects theta calculation                   │
│   samples_per_ring → determines depth map width                 │
│                      ↓                                          │
│ Denoising                                                       │
│   radius_min/max → must match tunnel_diameter/2 ± tolerance     │
│   theta_step → affects surface detail preservation              │
│                      ↓                                          │
│ Enhancing                                                       │
│   resolution → determines depth_map pixel size (5mm default)    │
│   depth_map dimensions → flows to all downstream stages         │
│                      ↓                                          │
│ Detection                                                       │
│   resolution → MUST match enhancing resolution                  │
│   k_height_mm, ab_height_mm → physical dimensions               │
│   angle ranges → MUST match actual oblique line angles          │
│                      ↓                                          │
│ SAM                                                             │
│   y_bounds → MUST match depth_map height                        │
│   segment_geometry → MUST match physical tunnel                 │
│   template dimensions → derived from segment sizes              │
└─────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Changing parameters in early stages (especially unfolding 
and enhancing resolution) requires re-tuning ALL downstream parameters.
```

---

### 5.7 Tunnel-Specific vs Universal Parameters

**Universal Parameters (same across tunnels):**
```
- resolution: 0.005 (5mm per pixel)
- gradient_epsilon: 1e-06
- mask_eps: 0.001
- polynomial_degree: 2-3
```

**Tunnel-Specific Parameters (must be measured/tuned):**
```
- tunnel_diameter: Physical measurement
- ring_spacing: Physical measurement  
- radius_min/max: diameter/2 ± noise tolerance
- angle_positive/negative_min/max: From tunnel geometry
- y_bounds: From depth_map dimensions
- segment_width, k_height, ab_height: From tunnel design
```

**Tunable via BO (optimize per tunnel):**
```
- hough thresholds: 25-80 range
- binary_threshold: 100-150 range
- template_mask dimensions: ±15% from nominal
- prompt_point positions: ±20% from defaults
- min_quality_threshold: 0.1-0.5
```

---

## Final Reflection

This exploration was a valuable failure. The per-ring alignment approach seemed theoretically sound - align K-blocks to create a universal solution for wraparound. But it violated a fundamental principle: **the depth map is not just data, it's a coherent representation of a physical surface.**

The key insight is that any transformation applied to the tunnel data must preserve the continuity between adjacent rings. Segments are continuous structures that span the entire tunnel length, and treating rings as independent units destroys this essential property.

**The most important lesson:** Before implementing any data transformation, ask "What does this data REPRESENT?" and "Does my transformation preserve that meaning?"

This failure, while initially frustrating, has deepened my understanding of the tunnel segmentation problem and will inform better approaches in the future.

---

**Report Generated:** January 23, 2026  
**Total Time Spent on Failed Approach:** ~2 hours  
**Value Gained:** Deep understanding of why spatial coherence matters
