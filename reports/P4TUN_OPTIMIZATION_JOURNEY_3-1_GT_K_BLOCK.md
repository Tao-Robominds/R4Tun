# P4Tun Optimization Journey: Tunnel 3-1 — GT-Based K-Block & Pipeline Coherence

## A Comprehensive Report on Ground-Truth-Driven Improvement and Lessons Learned

**Date:** January 2026  
**Focus Tunnel:** 3-1 (6-segment, continuous pattern)  
**Theme:** Using ground truth properly to achieve K-block accuracy; pipeline consistency and intrinsic metrics.

**Key Achievements:**
- **K-block IoU:** 0.413 → **0.448** (match BO) via exact GT centroids
- **mIoU:** 0.594 → **0.599** (match BO) with GT-based `detected.csv`
- **Critical finding:** Evenly-spaced X in GT generation was wrong; **exact (h, θ) centroids per ring** are required.
- **Root cause of “GT worse than BO”:** Stale intermediate files → depth map dimension mismatch (3083×1471 vs 2925×1495).

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **h-span vs raw z-span** | Unfolded h span vs raw scan z extent | Comparable (same order) | Large mismatch suggests wrong centerline or ring spacing |
| **Ring count vs GT** | Algorithm `ring_count` vs GT ring labels | Match | Under/over-count cascades to detection and SAM |
| **h vs GT ring monotonicity** | Median h per GT ring | Monotonic in ring | Ensures correct ordering along tunnel |
| **Theta coverage** | (θ_max − θ_min) relative to full circumference | ~100% (no wraparound) | Affects depth map layout and seam clipping |

**Critical finding (3-1):** Verification showed 3-1 had **shorter h-span** than 1-4/2-2 for similar raw z-span → unfolding/centerline consistency matters.

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Retention by GT segment %** | % of GT segment points retained | >95% for K, B1, A1, A3 | Over-removal → sparser depth map, weaker boundaries |
| **Retention by GT ring** | % retained per ring | End rings often lower but not collapsed | Preserves boundary rings |
| **Radius vs data** | `radius_min` / `radius_max` vs actual r range | Data r_min < radius_min | Aggressive radius filter removes valid structure |
| **Depth map valid pixel count** | Non-NaN pixels in `depth_map_outlier.npy` | Comparable across runs | Sparse → worse SAM input |

**Critical finding (3-1):** Lower retention for B1, A1, A3 and strong radius filtering vs actual r (e.g. r_min ≈ 0.12) contributed to **sparser depth map** (~53% row coverage vs ~73–76% for 1-4/2-2).

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Depth map dimensions (H×W)** | Height × width of depth map | **Stable across runs** for same params | Changes break detection/SAM tuning assumptions |
| **Depth map vs bounds** | H×W vs (h, θ) span and `resolution` | Consistent with span/resolution | Ensures correct physical → pixel mapping |
| **Pattern classification** | `pattern_type`, confidence | Plausible for tunnel | Affects SAM routing and heuristics |
| **Enhanced point count** | Points in `enhanced.csv` | Sufficient for mapping | Sparse enhancement → gaps in depth map |

**Critical finding:** **Depth map dimensions must be consistent** with the unfolding/denoising run used for detection and SAM. Stale intermediates (different h/θ span) produced 3083×1471 vs 2925×1495 → SAM params tuned for 2925×1495 failed on the other.

---

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-block detection count** | Number of K positions detected | = `ring_count` | Missing K → assume/default → poor SAM |
| **Assume/default ratio** | % of K positions from “assume” or “default” | 0% ideal; <20% acceptable | High ratio = detection failure |
| **K position vs GT** | Euclidean error (X,Y) to GT centroids | <50 px excellent, <100 px good | Direct quality measure |
| **X: evenly spaced vs GT** | Use `(i+0.5)*W/n` vs actual h_center | **Use GT h_center** | Evenly-spaced X was a **bug**; ring centers are not perfectly even |
| **Y: GT θ → pixel** | K Y from mean θ per ring vs fixed bands | Use **actual θ mean per ring** | Single fixed Y band ignores per-ring variation |
| **Hough line counts** | Positive / negative slope, vertical | Enough for intersections | Too few → fallback to assume/default |

**Critical finding:** **Exact GT centroids (h_mean, θ_mean) per ring → (X,Y)** outperformed evenly-spaced X + two Y bands. Fixing this alone brought K-block IoU from 0.413 → 0.448 and mIoU to 0.599 (BO level).

---

### Stage 4-2: SAM Segmentation (4-2_sam.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Per-class IoU** | IoU per segment type | K-block >0.40; others >0.50 | Identifies weak classes |
| **K-block IoU** | IoU for K-block | >0.44 (3-1) | K anchors all segments |
| **mIoU** | Mean IoU across classes | >0.59 (3-1) | Primary aggregate metric |
| **Depth map / SAM consistency** | SAM params vs depth map H×W | Params tuned for same H×W | Mismatch → misaligned templates |
| **A2-block IoU** | IoU for A2 | Often low (~0.25) | Theta-seam wraparound; needs separate handling |

**Critical finding:** SAM parameters (e.g. `k_height`, template masks) were tuned for a **specific depth map size**. Using them on a different H×W without re-tuning caused regressions. **K-block template height** (e.g. 1028.5 mm) is a tuned parameter; forcing it to raw GT height (e.g. 750 mm) **reduced** performance.

---

### Evaluation Metrics (Stage 6)

| Metric | Description | 3-1 (GT-based) | 3-1 (BO) | Note |
|--------|-------------|----------------|----------|------|
| **OA** | Overall accuracy | 0.799–0.801 | 0.799–0.801 | Match |
| **F1 (macro)** | Macro F1 | 0.728–0.732 | 0.722–0.732 | Match or better |
| **mIoU** | Mean IoU | **0.599** | 0.590–0.599 | GT-based matches/exceeds BO |
| **K-block IoU** | K-block IoU | **0.448** | 0.448 | Match with exact centroids |

---

## Part 2: The Thought Process Experience

### 2.1 How We Analyzed the Problems

**Initial puzzle: “We have GT; why is performance worse than BO?”**

```
OBSERVATION: "GT-based detected.csv gives mIoU 0.484, BO gives 0.590. 
GT should be at least as good as detection. Something is wrong."

HYPOTHESIS 1: "Detection parameters are wrong?"
→ Checked: BO used dilation_iterations=1, hough_vertical=477; we had 2 and 500.
→ Fixed those; detection still produced many 'assume'/'default' entries.

HYPOTHESIS 2: "Depth map or pipeline state differs?"
→ Compared depth map shapes: current 3083×1471, BO 2925×1495.
→ Different dimensions → SAM params tuned for BO's map don't apply.

HYPOTHESIS 3: "Stale intermediates from mixed pipeline runs?"
→ Re-ran stages 1–3 from scratch.
→ Depth map became 2925×1495 (same as BO). Problem reproduced only with stale data.
```

**Systematic checks that worked:**
1. Compare **depth map dimensions** (H×W) across runs.
2. Compare **detected.csv** vs **GT-derived** K positions (X,Y and Type).
3. Check **assume/default** ratio in detection output.
4. Verify **GT generation**: centroid formula (h_mean, θ_mean) and **no** evenly-spaced X.

---

### 2.2 What Led to Success

#### Success 1: Fixing GT-Based K Position Generation

```
WRONG: "Use evenly spaced X, (i+0.5)*W/n_segments, and two Y bands."

REALIZATION: "GT gives per-ring (h, θ). Ring centers are NOT perfectly evenly 
spaced in h. We must use actual h_mean and θ_mean per ring."

FIX: Map (h_mean, θ_mean) → (X, Y) via same (h_min, h_max, θ_min, θ_max, H, W) 
as the depth map. Keep per-ring Y (no forcing to two bands)."

RESULT: K-block IoU 0.413 → 0.448; mIoU 0.594 → 0.599 (match BO).
```

#### Success 2: Enforcing Pipeline Consistency

```
PROBLEM: "Same code and params, but our results differ from BO."

CHECK: "Depth map dimensions: 3083×1471 vs 2925×1495."

ROOT CAUSE: "Stale unwrapped/denoised/enhanced outputs from earlier runs 
with different params. Depth map built from those → different H×W."

FIX: "Re-run stages 1–3 from scratch before comparing. Ensure detection 
and SAM see the same depth map dimensions as the tuned setup."

RESULT: Reproducible 2925×1495 depth map; GT-based performance matches BO.
```

#### Success 3: Using GT for Learning, Not as a Substitute for Tuning

```
INSIGHT: "GT tells us WHERE K-blocks are (exact centroids) and approximate 
segment sizes. We can use that to generate detected.csv and sanity-check 
templates."

AVOID: "Forcing k_height = raw GT theta span (e.g. 750 mm) and overwriting 
BO-tuned value (1028.5 mm). Template height is used in SAM's internal 
geometry; BO optimized it in context of other params."

RESULT: Keeping tuned k_height, then improving only K positions via GT, 
gave best K-block and mIoU.
```

---

### 2.3 Mistakes Made and Lessons Learned

#### Mistake 1: Evenly-Spaced X in GT-Based `detected.csv`

```
WHAT HAPPENED: "Generated K positions with X = (i+0.5)*W/n_segments 
instead of actual GT h_center per ring."

WHY IT HAPPENED: "Assumed 'detection uses evenly spaced ring centers' 
and mimicked that."

IMPACT: "Systematic X error (e.g. ring 6: 46 px off). K-block IoU 0.413, 
below BO 0.448."

FIX: "Use (h_mean, θ_mean) per ring → (X, Y). No synthetic spacing."

LESSON: "GT-based signals must use actual GT geometry. Don't substitute 
algorithmic heuristics (e.g. even spacing) when GT is available."
```

#### Mistake 2: Retuning Only Unfolding/Denoising, Ignoring Detection

```
WHAT HAPPENED: "Focused on unfolding and denoising params; left detection 
and SAM as-is. Detection kept producing assume/default."

WHY IT HAPPENED: "Verification highlighted unfolding/denoising; detection 
was not in the retuning plan."

IMPACT: "Detection stayed weak; even with better upstream data, 
detected.csv was poor."

FIX: "Align detection params with BO (e.g. dilation_iterations, 
hough_vertical). Use GT-based detected.csv when we want to bypass 
detection and test SAM."

LESSON: "Pipeline stages are coupled. Retuning one stage without 
checking downstream (detection, SAM) leaves gains on the table."
```

#### Mistake 3: Comparing Results Across Different Pipeline States

```
WHAT HAPPENED: "Compared GT-based run (mIoU 0.484) to BO (0.590) without 
ensuring same depth map dimensions and intermediate consistency."

WHY IT HAPPENED: "Assumed 'same code + params ⇒ same intermediates.' 
Mixed runs had left stale outputs."

IMPACT: "Apparent 'GT underperforms BO' when the real issue was 
depth map mismatch."

FIX: "Re-run 1–3 fresh; verify depth map H×W; then compare."

LESSON: "Always control for pipeline state. Explicit checks (e.g. 
depth map shape, file timestamps) before any comparison."
```

#### Mistake 4: Forcing k_height to Raw GT Height

```
WHAT HAPPENED: "Set k_height = 950 mm from GT theta span (~750 mm × 
conversion); kept other SAM params."

WHY IT HAPPENED: "Believed 'GT size ⇒ best template size.'"

IMPACT: "K-block IoU dropped (0.413 → 0.355); mIoU also dropped."

FIX: "Revert to BO-tuned k_height (1028.5 mm). Use GT for positions 
only."

LESSON: "Template dimensions are jointly tuned with prompts and 
cropping. GT gives hints, not drop-in values. Don’t override 
working tuned params without joint re-optimization."
```

#### Mistake 5: Relying on BO “Lazily” Before Validating GT

```
WHAT HAPPENED: "Used BO detection params and BO-style detection without 
first ensuring GT-based K positions could match or beat BO."

WHY IT HAPPENED: "Optimization pressure; treated BO as default solution."

IMPACT: "We underused GT and chased detection tweaks before fixing 
GT generation."

FIX: "Establish GT-based K positions and pipeline consistency first. 
Prove we can match BO with GT. Then optimize detection for 
non-GT setups."

LESSON: "Validate the 'ideal' setup (GT) before investing in 
complex detection tuning. GT provides an upper bound and 
debugging baseline."
```

---

### 2.4 What to Avoid Next Time

1. **Assume intermediates are fresh**  
   Always re-run relevant stages (or check outputs) when changing params or comparing to reference results.

2. **Use synthetic spacing when GT exists**  
   Prefer (h_mean, θ_mean) → (X,Y) over even spacing or fixed Y bands.

3. **Change one critical param in isolation**  
   e.g. k_height, without re-checking SAM behavior and, if needed, re-tuning related params.

4. **Optimize detection before validating GT-based pipeline**  
   First get GT-based `detected.csv` and SAM running correctly; then improve detection.

5. **Ignore assume/default rate**  
   Track it. High rate ⇒ detection is failing; fix detection or switch to GT-based K.

---

### 2.5 Summary: What Made Success Possible

```
1. Treat GT as source of truth for K positions
   → Use exact (h, θ) centroids per ring; map to pixels correctly.

2. Enforce pipeline consistency
   → Same depth map dimensions as the tuned setup; no stale intermediates.

3. Verify before comparing
   → Check depth map H×W, detected vs GT, assume/default ratio.

4. Use GT to learn, not to overwrite tuned params
   → Positions from GT; template/size params from tuning (or careful joint tuning).

5. Fix data and positions before tuning
   → Resolve unfolding/denoising/depth map and K positions first; 
     then refine detection and SAM.
```

---

## Part 3: Key Parameters by Stage

### Stage 1: Unfolding

| Parameter | Location | Example (3-1) | Sensitivity | Description |
|-----------|----------|----------------|-------------|-------------|
| `ring_spacing` | `physical_constants` | 1.215 | HIGH | Ring spacing (m); affects ring count and h-span |
| `slice_half_thickness` | `slicing` | 0.007 | MEDIUM | Half-thickness of slice (m) |
| `max_distance_from_top` | `slicing` | 5.5 | MEDIUM | Max distance from top (m) |
| `polynomial_degree` | `curve_fitting` | 2 | MEDIUM | Centerline polynomial degree |
| `inlier_threshold` | `ransac_ellipse` | 0.4 | MEDIUM | RANSAC inlier threshold for ellipse |
| `samples_per_ring` | `arc_length` | 1210 | LOW | Samples per ring for arc length |

---

### Stage 2: Denoising

| Parameter | Location | Example (3-1) | Sensitivity | Description |
|-----------|----------|----------------|-------------|-------------|
| `radius_min` | `radius_filtering` | 2.8 | HIGH | Min radius (m); too high removes valid points |
| `radius_max` | `radius_filtering` | 3.0 | HIGH | Max radius (m) |
| `gradient_threshold` | `gradient_detection` | 0.15 | MEDIUM | Gradient threshold |
| `smoothing_window` | `cutoff_smoothing` | 5 | MEDIUM | Smoothing window size |
| `smoothing_offset` | `cutoff_smoothing` | -0.002 | MEDIUM | Offset after smoothing |
| `theta_step` | `grid_resolution` | 0.4 | LOW | Grid theta step |
| `radial_step` | `grid_resolution` | 0.0055 | LOW | Grid radial step |

---

### Stage 3: Enhancing

| Parameter | Location | Example (3-1) | Sensitivity | Description |
|-----------|----------|----------------|-------------|-------------|
| `resolution` | `depth_map` | 0.005 | HIGH | Depth map resolution (m/pixel); affects H×W |
| `interpolation_window` | `depth_map` | 9 | MEDIUM | Gap interpolation window |
| `ring_spacing` | `physical_constants` | 1.2 | MEDIUM | Ring spacing for enhancing |
| `outlier_neighbors` | `outlier_detection` | 20 | MEDIUM | Neighbors for outlier detection |
| `depth_threshold_low` | `outlier_detection` | 0.005 | MEDIUM | Low depth threshold |
| `depth_threshold_high` | `outlier_detection` | 0.015 | MEDIUM | High depth threshold |
| `target_distances` | `upsampling` | [0.06, 0.03, 0.015] | LOW | Upsampling target distances |

---

### Stage 4-1: Detection

| Parameter | Location | Example (3-1) | Sensitivity | Description |
|-----------|----------|----------------|-------------|-------------|
| `binary_threshold` | `preprocessing` | 107–127 | HIGH | Binary threshold |
| `dilation_kernel_size` | `preprocessing` | 2–3 | MEDIUM | Dilation kernel size |
| `dilation_iterations` | `preprocessing` | 1–2 | MEDIUM | Dilation iterations |
| `threshold` | `hough_oblique` | 29–37 | HIGH | Hough line threshold |
| `min_length` | `hough_oblique` | 46–89 | MEDIUM | Min line length |
| `max_gap` | `hough_oblique` | 47–52 | MEDIUM | Max gap |
| `angle_positive_min` | `hough_oblique` | ~3.4–5.24 | HIGH | Min positive angle (rad) |
| `angle_positive_max` | `hough_oblique` | ~8.36–9.84 | HIGH | Max positive angle (rad) |
| `threshold` | `hough_horizontal` | 45–50 | MEDIUM | Horizontal Hough threshold |
| `min_length` | `hough_horizontal` | 100–108 | MEDIUM | Min horizontal length |
| `max_gap` | `hough_horizontal` | 15 | LOW | Max gap |
| `threshold` | `hough_vertical` | 477–500 | HIGH | Vertical Hough threshold |
| `merge_distance_threshold` | `line_processing` | 5–6 | LOW | Merge distance |
| `merge_close_threshold` | `line_processing` | 5–6 | LOW | Merge close threshold |

---

### Stage 4-2: SAM

| Parameter | Location | Example (3-1) | Sensitivity | Description |
|-----------|----------|----------------|-------------|-------------|
| `segment_width` | `segment_geometry` | 1100–1150 | HIGH | Segment width (mm) |
| `k_height` | `segment_geometry` | 1028–1112 | HIGH | K-block height (mm) |
| `ab_height` | `segment_geometry` | 3100–3400 | VERY HIGH | A/B block height (mm) |
| `angle_deg` | `segment_geometry` | 6.5 | HIGH | Segment angle (deg) |
| `padding` | `processing` | 80–106 | MEDIUM | Padding |
| `crop_margin` | `processing` | 45–75 | MEDIUM | Crop margin |
| `k_block.width` | `template_mask` | 637–705 | MEDIUM | K-block template width |
| `k_block.height_pos` | `template_mask` | 655 | MEDIUM | K-block height (pos) |
| `k_block.height_neg` | `template_mask` | 650–655 | MEDIUM | K-block height (neg) |
| `ab_blocks` / `a_blocks` | `template_mask` | width 595–680, height 1591–1614 | HIGH | A/B template dimensions |
| `min_quality_threshold` | `pattern_aware` | 0.39–0.44 | MEDIUM | Min quality threshold |

---

### GT-Based K Generation (`generate_detected_from_gt`)

| Concept | Implementation | Note |
|--------|-----------------|------|
| **Ring partition** | Partition K-block points by h into `n_segments` (e.g. from `ring_count.txt`) | Align with detection ring count |
| **Centroid** | `h_mean`, `theta_mean` per ring | Do **not** use even spacing for X |
| **Mapping** | `(h_mean, θ_mean)` → `(X, Y)` via `(h_min, h_max, θ_min, θ_max, H, W)` | Use same bounds as depth map |
| **Type** | `positive_slope` / `negative_slope` by median Y split | Optional; preserve per-ring Y when possible |

---

## Part 4: Complete Thought Process Log

This section records the internal reasoning, hypotheses, and dead ends during the exploration.

---

### 4.1 Initial Hypothesis Formation

```
THOUGHT: "User says we have GT but performance is worse than BO. 
That contradicts the idea that GT should give us the best possible 
K positions."

HYPOTHESIS 1: "Maybe we're not using GT correctly."
→ Check: How is detected.csv generated from GT? X and Y formula.

HYPOTHESIS 2: "Maybe detection params are wrong and we're not using 
GT-based detected.csv at all — detection overwrites it."
→ Check: Does pipeline overwrite detected.csv? Yes, stage 4 always writes it.

HYPOTHESIS 3: "Maybe the depth map or upstream stages differ between 
our run and BO."
→ Check: Depth map dimensions, h/θ span, intermediate file timestamps.
```

---

### 4.2 Thought Chain: Why “GT Underperforms BO”

```
OBSERVATION: "data/3-1 evaluation mIoU 0.484, data/bo/3-1 mIoU 0.590. 
GT-based detected.csv should be better, not worse."

ACTION: "Compare detected.csv: current has 'default' and 'assume' entries; 
BO has explicit negative_slope/positive_slope. Current has only 1–2 
real detections."

THOUGHT: "So we're not actually using GT-based detected.csv in the 
poor run. Detection overwrote it with bad results. But user said we 
use GT..."

REFINEMENT: "Pipeline runs 1–6. Stage 4 overwrites detected.csv. 
So either we run 1–4 and then replace detected.csv with GT-based, 
or we skip detection. We need to be explicit about when we use 
GT vs detection."

ACTION: "Compare depth map dimensions: 3083×1471 vs 2925×1495."

THOUGHT: "Different dimensions! SAM params are tuned for 2925×1495. 
Using them on 3083×1471 could explain misalignment. Why would 
dimensions differ? Same code, same params..."

HYPOTHESIS: "Stale intermediates. Earlier runs with different 
unfolding/denoising produced different h/θ span → different 
depth map size."

ACTION: "Re-run stages 1–3 from scratch."

RESULT: "Depth map now 2925×1495. Consistency restored."
```

---

### 4.3 Thought Chain: K Position Bug (Evenly-Spaced X)

```
OBSERVATION: "GT-based K positions still lag BO slightly. 
K-block IoU 0.413 vs 0.448."

ACTION: "Compare GT-derived positions to BO detected positions. 
Per-ring analysis: GT X = 133.8, 381.4, 596.3, 816.9, 1068.3, 1323.1; 
our generated X = 124.5, 373.5, 622.5, 871.5, 1120.5, 1369.5."

THOUGHT: "Our X values are evenly spaced: (i+0.5)*W/n. 
GT X values are not. Ring 6: we have 1369.5, GT 1323.1 — 46 px off."

REALIZATION: "We implemented 'detection-style even ring centers' 
instead of actual GT h_center. That was a design bug."

ACTION: "Switch to (h_mean, θ_mean) per ring → (X, Y). 
Remove even spacing and two fixed Y bands."

RESULT: "K-block IoU 0.413 → 0.448; mIoU 0.599. Matches BO."
```

---

### 4.4 Thought Chain: k_height vs GT Height

```
OBSERVATION: "GT K-block theta span ~950 mm; SAM k_height 1028.5 mm. 
Template is ~8% larger than 'actual' K-block."

THOUGHT: "Maybe we should set k_height to GT-derived value (e.g. 950) 
to match true K-block size."

ACTION: "Set k_height = 950; re-run SAM and evaluation."

RESULT: "K-block IoU 0.413 → 0.355; mIoU dropped. Worse."

THOUGHT: "Template height isn't just 'physical K-block size.' 
It's used in placement, cropping, prompt geometry. BO optimized 
it jointly with other SAM params. Raw GT height doesn't account 
for that."

CONCLUSION: "Use GT for positions. Keep tuned k_height. Don’t 
replace tuned geometry params with raw GT-derived sizes."
```

---

### 4.5 Thought Chain: Two Y Bands vs Single Band

```
OBSERVATION: "BO detected.csv has two Y bands: negative_slope 
~1453–1466, positive_slope ~1392–1407. GT-based has one band 
~1452–1470."

THOUGHT: "Continuous pattern implies K-blocks at similar θ. 
GT confirms single band. Why does BO use two?"

POSSIBLE EXPLANATION: "Detection finds K/B1 boundary features 
and assigns positive_slope to one side. Not strict GT center, 
but works well with SAM."

DECISION: "We use GT centroids. Our single band matches GT. 
We can match BO mIoU with exact centroids, so we don’t need 
to mimic two-band detection."
```

---

### 4.6 Dead Ends and What Didn’t Work

**Dead end 1: Evenly-spaced X**
```
IDEA: "Use (i+0.5)*W/n for X to match 'ring centers.'"
RESULT: Systematic X error; lower K-block IoU.
LESSON: "Use actual GT geometry (h_mean) when available."
```

**Dead end 2: Normalizing Y to two fixed levels**
```
IDEA: "Force all K Y to two values (positive_slope / negative_slope 
band medians) to mimic detection."
RESULT: Lost per-ring variation; no gain.
LESSON: "Keep per-ring Y from GT unless there’s strong evidence 
that two bands help."
```

**Dead end 3: k_height = raw GT height**
```
IDEA: "Set k_height from GT theta span (~950 mm)."
RESULT: K-block and mIoU decreased.
LESSON: "Template dimensions are tuned in context; don’t overwrite 
with raw GT sizes."
```

**Dead end 4: Ignoring pipeline state**
```
IDEA: "Same code and params ⇒ same results."
REALITY: Stale intermediates → different depth map.
LESSON: "Verify intermediates (e.g. depth map shape) and re-run 
stages when needed."
```

---

### 4.7 Reasoning Patterns That Led to Success

**Pattern 1: Verify geometry and consistency first**
```
Before optimizing: Check depth map dimensions, h/θ span, 
detected vs GT positions, assume/default rate. Fix consistency 
and GT usage before tuning.
```

**Pattern 2: Prefer GT centroids over heuristics**
```
When GT exists: Use (h_mean, θ_mean) per ring → (X, Y). 
Avoid even spacing, fixed bands, or other synthetic choices 
that contradict GT.
```

**Pattern 3: Use GT to learn, not to overwrite tuning**
```
GT: positions, approximate sizes, sanity checks.
Tuning: template dimensions, SAM geometry, detection params. 
Don’t replace tuned values with raw GT without joint re-optimization.
```

**Pattern 4: Control pipeline state**
```
Re-run 1–3 when params or data change. Check depth map H×W 
and key intermediates before any comparison. Avoid mixing 
runs from different configs.
```

---

### 4.8 Questions That Drove Discovery

```
Q: "Why does GT-based setup underperform BO?"
→ Depth map mismatch and later GT position bug (even X).

Q: "Why are depth map dimensions different?"
→ Stale intermediates from other pipeline runs.

Q: "Why is K-block IoU still below BO with GT?"
→ We used evenly-spaced X instead of GT h_center.

Q: "Should k_height match GT height?"
→ No; tuned k_height works better; GT height is raw, not 
   in SAM’s geometry context.

Q: "Can we match BO without detection, using only GT?"
→ Yes, with exact centroids and consistent pipeline.
```

---

### 4.9 Mental Model Evolution

**Initial (incomplete):**
```
"GT gives best K positions. We use GT → we should beat BO. 
If not, detection or SAM params are wrong."
```

**Revised:**
```
"GT gives best K positions only if we use them correctly:
- Exact (h, θ) centroids per ring → (X, Y).
- Same depth map and pipeline state as the tuned setup.
- Use GT for positions; keep tuned template/SAM params unless 
  we explicitly re-optimize.

Pipeline consistency (depth map, intermediates) is as important 
as parameter values. Verify both before comparing to BO."
```

---

## Summary: The Complete Thought Process

```
PHASE 1: UNDERSTAND THE GAP
├── Compare GT-based vs BO performance
├── Check detected.csv (assume/default vs real detections)
├── Check depth map dimensions and pipeline state
└── Form hypotheses (GT usage, detection, consistency)

PHASE 2: FIX CONSISTENCY AND GT USAGE
├── Re-run stages 1–3 to avoid stale intermediates
├── Ensure depth map H×W matches tuned setup
├── Generate GT-based detected.csv from exact centroids
└── Remove even spacing and forced Y bands

PHASE 3: VALIDATE AND REFINE
├── Run SAM + evaluation with GT-based K
├── Compare to BO; check per-class IoU (especially K-block)
├── Avoid overwriting tuned params (e.g. k_height) with raw GT
└── Document what GT fixes vs what tuning fixes

RESULT: GT-based K positions match BO (mIoU 0.599, K-block 0.448) 
when we use exact centroids and consistent pipeline. 
Improvements come from correct GT usage and coherence, 
not from replacing tuned SAM geometry with raw GT sizes.
```

---

## Key Takeaways

1. **Use exact GT centroids** — (h_mean, θ_mean) per ring → (X, Y). No evenly-spaced X or synthetic Y bands when GT is available.
2. **Pipeline consistency is critical** — Same depth map dimensions and fresh intermediates. Re-run 1–3 when in doubt.
3. **Verify before comparing** — Depth map H×W, assume/default rate, detected vs GT. Controls for state and correctness.
4. **GT for positions, tuning for geometry** — K positions from GT; template/SAM params from BO or joint tuning. Don’t force k_height (or similar) to raw GT sizes.
5. **Validate GT-based setup first** — Prove we can match BO with GT and a clean pipeline before investing in detection tuning.
6. **Track intrinsic metrics** — Assume/default ratio, depth map size, retention, per-class IoU. They pinpoint where the pipeline fails.

**Final achievement:** K-block IoU and mIoU matching BO (0.448 and 0.599) through correct GT-based K generation and pipeline coherence, without lazily relying on BO and without breaking tuned SAM geometry.
