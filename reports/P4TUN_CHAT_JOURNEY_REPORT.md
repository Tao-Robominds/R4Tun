# P4Tun Chat Window Journey: Comprehensive Report
## Intrinsic Quality Metrics, Thought Process Experience, Key Parameters & Full Thought Log

**Date:** January 2026  
**Scope:** Multi-tunnel pipeline exploration (1-4, 2-2, 3-1, 4-1, 5-1); detection improvements; pattern classification; wraparound; GT-free design.  
**Reference:** Structure follows `P4TUN_OPTIMIZATION_JOURNEY_4-1.md`.

---

## Executive Summary

This report documents the end-to-end exploration from a single chat window: pipeline execution, 3-1 performance investigation, Bayesian Optimization, pattern classification, wraparound experiments, and **generalized detection improvements** across all tunnels. A recurring theme is **using ground truth only to reverse-engineer and tune; the solution itself must never depend on GT**.

**Key outcomes:**
- **Detection:** Fixed vertical-line filter (image-width–based); Canny + binary preprocessing; Hough tuning. Fewer “assume” fallbacks, more line-based K positions.
- **Pattern classification:** Correct Seg2Tunnel mapping (1-4, 2-2 → simple_staggered; 3-1 → continuous; 4-1, 5-1 → complex_staggered).
- **Wraparound:** Selective, geometry-based strategy explored; full wraparound harmed mIoU; geometric boundary checks (no GT) identified as the right direction.
- **Generalization:** Same detection improvements applied to 1-4, 2-2, 3-1, 4-1, 5-1; parameters tuned with GT insight but pipeline remains GT-free.

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Theta Coverage %** | `(theta_max - theta_min) / (2π) × 100` | 98–102% | <100% loses segments; >100% causes wraparound |
| **Ring Count** | Detected vs `ring_count.txt` | Exact match | Mismatch propagates to detection and SAM |
| **Point Density** | Points per pixel in depth map | >0.8 | Sparse regions → detection gaps |
| **Ring Width Consistency** | Std of ring widths (px) | <5% of mean | Inconsistent widths confuse detection |

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Surface Retention %** | `valid_after / valid_before × 100` | 70–95% | Too aggressive → lost boundaries; too lenient → noise |
| **Outlier Ratio** | NaN fraction in `depth_map_outlier.npy` | 10–30% | Balance noise removal vs. data loss |
| **Edge Preservation** | Gradient magnitude at segment boundaries | >0.7 (relative) | Denoising must preserve edges for detection |

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Interpolation Coverage** | % of depth map filled | >95% | Gaps hurt line detection |
| **Pattern Classification** | `pattern_type` vs. expected (Seg2Tunnel) | Correct label | Drives detection/SAM strategy; must match tunnel type |
| **Oblique Line Stats** | Count, angle mean/std, Y mean/std | Consistent with tunnel | Feeds pattern classifier and downstream logic |
| **Intensity Contrast** | `(max − min) / mean` | >0.3 | Low contrast → poor line detection |

**Pattern types:** `continuous` (T3), `simple_staggered` (T1/T2), `complex_staggered` (T4/T5). Classification uses oblique angles and Y-variance only (no GT).

---

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-Position Count** | Detected vs. `ring_count` | Exact match | Wrong count cascades to SAM |
| **Assume %** | % of K positions from “assume” fallback | <30% | Lower → more line-based detections |
| **Line-Based %** | % midpoint / positive_slope / negative_slope | >70% | Direct indicator of detection quality |
| **Vertical Line Count** | Hough vertical lines | ≥ ring_count | Needed for ring separation |
| **Oblique Line Counts** | Positive + negative slope | >2 each (typical) | Too few → weak intersections, more assume |
| **Y-Position Spread** | Range or std of detected Y | Pattern-dependent | Continuous: low; staggered: two bands |

**Critical:** Detection must **not** use GT. Use GT only to reverse-engineer parameter ranges and validate.

---

### Stage 4-2: SAM Segmentation (4-2_sam.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **mIoU** | Mean IoU over classes | >0.55 (good), >0.65 (excellent) | Primary quality signal |
| **Per-Class IoU** | IoU per segment type | >0.15 each | Surfaces weak classes (e.g. K, A2) |
| **Background Ratio** | % background | <30% | High → undersized templates |
| **Template Coverage** | Template vs. segment area | 85–95% | Templates guide SAM |

**Note:** mIoU requires GT. In production, use **proxy metrics** (e.g. template coverage, prompt validity, geometric consistency), not mIoU.

---

### Stage 5: Evaluation

| Metric | Description | Use |
|--------|-------------|-----|
| **OA** | Overall accuracy | Research / validation only |
| **F1 (macro)** | Macro F1 | Research / validation only |
| **mIoU** | Mean IoU | Research / validation only |
| **Per-Class IoU** | Per-class IoU | Diagnose weak classes |

All evaluation metrics are GT-based and **not** for use in the deployed solution.

---

## Part 2: The Thought Process Experience

### 2.1 How We Analyzed Problems

**Systematic flow:**
1. **Reproduce baseline** – Run full pipeline, record metrics (OA, F1, mIoU, per-class IoU).
2. **Compare tunnels** – e.g. 3-1 vs. 1-4, 2-2; check `detected.csv`, `depth_map.png`, `detected_lines.png`.
3. **Trace upstream** – Low SAM performance → check detection (K positions, line counts) → check enhancing → unfolding/denoising.
4. **Use GT only for analysis** – Compute GT K positions, compare to detected, infer parameter issues. Never feed GT into the pipeline.
5. **Hypothesize, change one thing, re-run** – e.g. fix rho→X, adjust Hough thresholds, then re-evaluate.

**Example (3-1 underperformance):**
- **Observation:** 3-1 mIoU much lower than baseline.
- **Check:** `detected.csv` vs. `sam4tun` detection; parameter diffs; detection code (rho vs. X).
- **Findings:** (1) `compute_ring_centers` used `rho` as X; should use `rho * cos(theta)`. (2) Different Hough/preprocessing params.
- **Actions:** Fix rho→X in code; align params. **Result:** mIoU recovery.

---

### 2.2 What Led to Success

**1. Fix upstream before downstream**  
Detection errors propagate. We fixed detection (vertical filter, preprocessing, Hough) before debating SAM or wraparound tweaks.

**2. GT for reverse engineering only**  
- Use GT to derive “good” parameter ranges, validate `detected.csv`, and check pattern types.  
- Never use GT inside the pipeline.  
- Design heuristics (e.g. alternation bands as `0.25*L`, `0.65*L`) to be geometry-based, not GT-based.

**3. Generalize across tunnels**  
After improving 4-1/5-1 detection, we **re-ran detection on 1-4, 2-2, 3-1** to ensure no regression. Same code, tunnel-specific params.

**4. Preprocessing + geometry**  
Combining **Canny** with binary edges, and using **image width** for the vertical-line filter instead of a fixed pixel value, improved line detection across tunnels.

**5. Pattern-aware logic without GT**  
Pattern classifier uses **oblique angles and Y-variance** from the depth map. Strategies (e.g. wraparound, alternation) key off `pattern_type`, not GT.

---

### 2.3 What to Avoid Next Time

**1. Using GT inside the solution**  
- **Wrong:** Feeding GT positions, mIoU, or GT-derived masks into detection/SAM.  
- **Right:** Use GT only for parameter tuning, validation, and ablation studies.

**2. Optimizing downstream before upstream**  
- **Wrong:** Tweaking SAM or wraparound while detection is broken.  
- **Right:** Fix detection (and enhancing if needed) first, then SAM.

**3. Hardcoding tunnel-specific behaviour**  
- **Wrong:** `if block == 'A2'` for wraparound; tunnel-specific magic numbers.  
- **Right:** Geometry-based rules (e.g. crop vs. image bounds, segment layout) that work for any tunnel.

**4. Assuming mIoU is available in production**  
- **Wrong:** Using mIoU to decide when to enable wraparound or other strategies.  
- **Right:** Use only **geometry and image-derived cues** (e.g. boundary proximity, pattern_type).

**5. Trusting intermediates without checks**  
- **Wrong:** Taking `all_segments.csv`, `pattern.csv`, or old `detected` files as ground truth.  
- **Right:** Verify against depth maps, GT (for analysis), and upstream outputs.

---

### 2.4 Mistakes Made

**1. Selective wraparound initially tied to “A2”**  
- **Mistake:** Enabling wraparound only for `block == 'A2'`.  
- **Reality:** The boundary-crossing segment varies by tunnel; it’s not always A2.  
- **Fix:** Use **geometry only** (crop vs. top/bottom boundaries); no block-name logic.

**2. Vertical-line filter too restrictive**  
- **Mistake:** Filtering verticals by `rho <= vert_filter_rings * 1200 / (resolution*1000)` (e.g. 1200 px).  
- **Reality:** For 3-1, image width > 1200 px; valid verticals were dropped.  
- **Fix:** Use `rho <= W` (image width) so all in-image verticals are kept.

**3. Wraparound enabled globally for “continuous”**  
- **Mistake:** Enabling wraparound for all segments when `pattern_type == "continuous"`.  
- **Reality:** A2 improved, but others regressed; mIoU dropped.  
- **Fix:** Reserve wraparound for segments **whose crop crosses boundaries**; same geometry-based rule everywhere.

**4. Pattern classifier bias toward complex_staggered**  
- **Mistake:** 1-4 and 2-2 classified as `complex_staggered` due to strict `angle_std < 3` etc.  
- **Reality:** Per Seg2Tunnel, 1-4 and 2-2 are **simple_staggered** (T1/T2).  
- **Fix:** Broaden criteria (e.g. `y_std < 250` and `angle_std < 8`) so regular-but-staggered tunnels become simple_staggered.

**5. Forgetting to generalize detection changes**  
- **Mistake:** Improving detection for 4-1/5-1 only.  
- **Reality:** User explicitly required **general** detection improvements across 1-4, 2-2, 3-1.  
- **Fix:** Apply same code and tuning philosophy to all tunnels; use GT only to validate.

---

### 2.5 Summary: Optimization Journey

```
Pipeline baseline (1-4, 2-2, 3-1) → 3-1 underperforms
    ↓
Diagnose: detection logic (rho→X) + params vs. sam4tun
    ↓
Fix detection; align params; 3-1 recovers
    ↓
BO (combined detection+SAM) for 3-1 → better mIoU
    ↓
Pattern classification (enhancing); fix 1-4/2-2 → simple_staggered
    ↓
Wraparound: geometric, selective; reject mIoU-based switching
    ↓
Detection improvements (Canny, vertical filter, Hough) for 4-1, 5-1
    ↓
Generalize to 1-4, 2-2, 3-1; ensure solution is GT-free
    ↓
Document tuneable constants (K_height, AB_height, etc.)
```

---

## Part 3: Key Parameters of Each Stage

### Stage 1: Unfolding

| Parameter | Type | Typical | Description |
|-----------|------|---------|-------------|
| `ring_spacing` | float | 1.2–1.3 | Ring spacing (m). |
| `resolution` | float | 0.005 | m/pixel. |
| `theta_coverage` | float | ~1.0 | Target coverage; ~100% to avoid wraparound. |
| `delta`, `slice_spacing_factor` | float | – | Slicing and sampling. |

---

### Stage 2: Denoising

| Parameter | Type | Typical | Description |
|-----------|------|---------|-------------|
| `radius_filter.radius_min/max` | float | – | Valid radius range. |
| `gradient_filter.threshold` | float | 0.1–0.4 | Edge/noise sensitivity. |
| `outlier_detection.k_neighbors` | int | 10–50 | Neighbors for outlier logic. |

---

### Stage 3: Enhancing

| Parameter | Type | Typical | Description |
|-----------|------|---------|-------------|
| `interpolation.radius` | float | – | Interpolation search radius. |
| `interpolation.num_neighbors` | int | – | Neighbors for interpolation. |
| `curvature.k_neighbors` | int | 15–30 | Curvature estimation. |
| `upsampling.target_distance` | float | 0.02–0.10 | Upsampling density. |
| `outlier_neighbors` | int | – | Min neighbors for outlier detection. |

Pattern classification uses **oblique line** and **Y-variance** thresholds (e.g. `y_std < 100` → continuous; `y_std < 250` and `angle_std < 8` → simple_staggered). No GT input.

---

### Stage 4-1: Detection

**Physical constants (tuneable via `physical_constants`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 0.005 | m/pixel. |
| `k_height_mm` | 1079.92 | K-block height (mm). |
| `ab_height_mm` | 3239.77 | A/B block height (mm). |
| `segment_width_mm` | 1200 | Segment width (mm). |

**Preprocessing:**

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `binary_threshold` | 107–140 | Binary conversion. |
| `dilation_kernel_size` | 2–5 | Morphological kernel size. |
| `dilation_iterations` | 1–3 | Dilation passes. |

**Hough:**

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `hough_oblique.threshold` | 29–60 | Oblique line votes. |
| `hough_oblique.min_length` | 46–120 | Min line length (px). |
| `hough_oblique.max_gap` | 47–60 | Max gap (px). |
| `hough_oblique.angle_positive_min/max` | e.g. 3.4–9.8° | Positive slope range. |
| `hough_oblique.angle_negative_min/max` | e.g. −9.8–−3.4° | Negative slope range. |
| `hough_horizontal.threshold` | 30–66 | Horizontal line votes. |
| `hough_horizontal.min_length` | 80–122 | Min length (px). |
| `hough_horizontal.max_gap` | 6–20 | Max gap (px). |
| `hough_vertical.threshold` | 400–700 | Vertical line votes. |

**Line processing:**

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `merge_distance_threshold` | 2–10 | Merge close verticals (px). |
| `merge_close_threshold` | 6 | Merge close Y intersections (px). |

**Vertical-line filter:** Use `rho <= W` (image width). Do **not** use a fixed `filter_rings * 1200 / (resolution*1000)` cap.

---

### Stage 4-2: SAM

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `segment_geometry.segment_width` | 1100–1250 | Segment width (mm). |
| `segment_geometry.k_height` | ~1028–1150 | K height (mm). |
| `segment_geometry.ab_height` | 3100–3400 | A/B height (mm). |
| `segment_geometry.angle_deg` | 6.5–7.52 | Oblique angle (deg). |
| `processing.padding` | 80–150 | Padding (px). |
| `processing.crop_margin` | 45–75 | Crop margin (px). |
| `processing.y_bounds` | [4200, 13100] | Y range (mm). |
| `template_mask.*` | – | K/B1/B2/A template dimensions. |
| `prompt_points.*` | – | Ring radii, spacing, levels. |
| `pattern_aware.min_quality_threshold` | 0.39–0.44 | Quality cutoff. |

---

## Part 4: Complete Thought Experience Log

### 4.1 Initial Setup and Pipeline Runs

```
THOUGHT: "User wants full pipeline on 1-4 with parameters from p4tun/parameters/1-4.
Each script takes tunnel_id. Use venv. Run in order: unfolding → denoising →
enhancing → detection → SAM → evaluation."

ACTION: Ran each stage sequentially for 1-4, then 2-2, then 3-1.

THOUGHT: "3-1 mIoU is much lower than baseline. User points to baseline and
sam4tun parameters. Need to compare detection outputs and parameters, not
assume our pipeline is correct."
```

---

### 4.2 3-1 Underperformance and Root Cause

```
THOUGHT: "Compare enhanced.csv and detected.csv vs. baseline. Focus on
parameter differences and detection logic, not pattern positions."

OBSERVATION: "Detection X/Y positions differ: ~92 px X offset, large Y
differences. SAM with sam4tun-style detection reaches baseline mIoU."

THOUGHT: "So detection is the main culprit. Either we use different
logic (e.g. rho vs. X) or different params."

ACTION: Checked compute_ring_centers and rho→X conversion.

DISCOVERY: "We treat rho as X. Sam4tun uses X = rho * cos(theta).
That explains systematic X (and downstream) errors."

FIX: "Use rho * cos(theta) for X in ring processing. Align Hough
and preprocessing params with sam4tun where relevant."

RESULT: "3-1 mIoU recovers to baseline level. Confirms detection
logic and params were the root cause."
```

---

### 4.3 Bayesian Optimization and Combined Tuning

```
THOUGHT: "User asks if BO can improve 3-1 further. Baseline didn’t use BO.
Combined BO (detection + SAM) optimizes interdependent factors."

ACTION: Ran combined BO for 3-1; ensured only one BO process.

OBSERVATION: "Best mIoU ~0.59, better than manual baseline. Best params
saved to p4tun/parameters/3-1."

THOUGHT: "BO is effective. User then wants pipeline re-run with best
params to save results. Straightforward."
```

---

### 4.4 K-Block and A2-Block Low IoU

```
THOUGHT: "User notes K and A2 have lower IoU. For 3-1: K might be
horizontally aligned (different from alternating); A2 might have
wraparound issues."

ACTION: Checked K Y-positions (horizontal alignment) and A2 vs. theta seam.

FINDINGS: "K is horizontally aligned; issues come from over-segmentation
and confusion with neighbours, not detection. A2 crosses theta seam;
standard SAM crops clamp, so we under-segment."

THOUGHT: "Wraparound-aware cropping could help A2. We have
4-2_sam_wrap_around. But we must not use mIoU to decide when to
apply it—user says we can’t use mIoU in practice."
```

---

### 4.5 Pattern Types and Strategy

```
THOUGHT: "User references Simple Staggered (T1/T2), Continuous (T3),
Complex Staggered (T4/T5). We may need three strategies. Key questions:
(1) Do we need three? (2) Only detection + SAM? (3) Can we detect
pattern earlier (e.g. enhancing)?"

CONCLUSION: "Yes to three strategies; focus on detection and SAM.
Pattern can be inferred from enhancing-stage outputs (depth map,
oblique lines, Y variance) before detection."

THOUGHT: "If we classify at enhancing, does that help detection?
Yes—detection can use pattern_type for strategy (e.g. horizontal
emphasis for continuous, alternation for staggered)."
```

---

### 4.6 Pattern Classifier and 4-2_sam_continuous

```
THOUGHT: "Add pattern classification in 3_enhancing. Then test
enhancing + SAM on 1-4 and 2-2 to ensure no regression. For 3-1,
try 4-2_sam_continuous with wraparound when pattern_type is
continuous."

ACTION: Implemented classify_tunnel_pattern; saved pattern_type.json.
Fixed JSON serialization (numpy → native). Tested 1-4, 2-2 (ok),
then 3-1 with 4-2_sam_continuous.

OBSERVATION: "A2 IoU improves (0.228 → 0.290) but mIoU drops
(0.590 → 0.444). Other segments suffer. Full wraparound helps
A2 but hurts the rest."

THOUGHT: "Wraparound should be selective or better tuned, not
global for continuous."
```

---

### 4.7 Selective Wraparound and “No mIoU in Production”

```
USER: "How can we apply wraparound only to A2? We can’t use mIoU
in practice—it’s ground truth."

THOUGHT: "We need a GT-free rule. We know block names before
segmentation. We can use geometry: enable wraparound only when
the **crop** crosses top/bottom boundaries or when the segment
center is near the boundary."

ACTION: Use crop_y1 < 0, crop_y2 > H, or map_y near 0/H.
Apply per segment; no block-name check.

LATER REFINEMENT: "User says we can’t assume A2 is always the
wraparound segment—it varies by tunnel. So we must **not** use
block == 'A2'. Use only geometric checks for any segment."

PRINCIPLE: "mIoU and GT are for validation only. In production,
only geometry and image-derived cues."
```

---

### 4.8 Wraparound Hurts mIoU; Pattern Classification Still Useful

```
USER: "If we identify pattern but wraparound makes things worse,
what’s the point?"

THOUGHT: "Wraparound *as implemented* degrades mIoU. So we
disable it for now. But pattern classification is still useful:
detection strategy, SAM template strategy, parameter choice.
Use pattern for those; don’t use it to turn on wraparound
until we have a better wraparound strategy."
```

---

### 4.9 Simple vs. Complex Staggered Fix

```
USER: "1-4 and 2-2 should not be complex_staggered; they’re
simple_staggered (T1/T2) per Seg2Tunnel."

THOUGHT: "Classifier was too strict. Old rule: simple_staggered
only if is_alternating or (angle_std < 3 and y_std > 150).
1-4, 2-2 have angle_std ~7.6 → pushed to complex_staggered."

FIX: "Add rule: if y_std < 250 and angle_std < 8 → simple_staggered.
Relax angle condition so regular staggered tunnels aren’t
over-classified as complex."

RESULT: "1-4, 2-2 → simple_staggered. 3-1 → continuous. 4-1,
5-1 → complex_staggered. Matches Seg2Tunnel."
```

---

### 4.10 4-1, 5-1 Verification and Wraparound on 4-1

```
USER: "Check 4-1 and 5-1 pattern classification. Run wraparound
on 4-1; it’s 7-segment, complex. Read reports on why complex
wraparound is hard."

THOUGHT: "4-1, 5-1 are T4/T5: 7 segments, complex_staggered.
Reports say 7-segment tunnels have all segments crossing
0°/360°; theta_offset can’t fix it; mitigation is very hard."

ACTION: Ran 4-2_sam_wrap_around on 4-1. Results: mIoU ~0.10,
very low. Confirms wraparound for 7-segment complex is
extremely difficult."
```

---

### 4.11 Better Detection for 4-1 (and 5-1)

```
USER: "Depth map shows clear vertical, horizontal, oblique lines;
detected_lines doesn’t. Improve detection first; use GT to
reverse-engineer, but **solution cannot use GT**."

THOUGHT: "0 lines detected originally. Likely: (1) vertical
filter too strict, (2) preprocessing weak, (3) Hough params
too strict."

ACTIONS:
- Fix vertical filter: rho <= W instead of fixed 1200.
- Add Canny on normalized depth; OR with binary.
- Lower Hough thresholds; relax min_length, max_gap; widen
  angle ranges.
- Increase dilation (kernel, iterations)."

RESULT: "4-1: many more line-based K positions. Apply same
logic to 5-1."
```

---

### 4.12 Generalize Detection to 1-4, 2-2, 3-1

```
USER: "Apply same detection improvements to 1-4, 2-2, 3-1.
Ensure they improve too. **Use GT only to reverse-engineer;
solution must not use GT.**"

THOUGHT: "Code changes (Canny, vertical filter) already
global. Per-tunnel params may differ. 3-1 had 0 verticals
with old filter; 1-4 had 50% assume."

ACTIONS:
- Keep vertical filter as rho <= W for all.
- Tune 3-1 vertical threshold and dilation from GT insight;
  keep pipeline itself GT-free.
- Re-run detection on 1-4, 2-2, 3-1; check assume % and
  line-based %."

RESULT: "1-4 assume 50%→33%; 2-2 stable; 3-1 back to 6
detections, 1 assume. Same code, no GT in solution."
```

---

### 4.13 Tuneable Constants

```
USER: "Are K_height, AB_height, segment_width, resolution
tuneable?"

THOUGHT: "Check load_parameters and get_param. These are
read from physical_constants with DEFAULT_* fallbacks."

VERIFICATION: "Yes. resolution, k_height_mm, ab_height_mm
are loaded via get_param(..., 'physical_constants', ...).
segment_width_mm appears in some params JSONs but isn’t
loaded in main. The main three are tuneable."
```

---

### 4.14 Dead Ends and What Didn’t Work

```
DEAD END 1: "Use mIoU to decide when to apply wraparound."
→ User constraint: no mIoU in production. Use geometry only.

DEAD END 2: "Enable wraparound only for block == 'A2'."
→ Wraparound segment varies by tunnel. Use geometry, not names.

DEAD END 3: "Enable full wraparound for continuous pattern."
→ A2 gains, rest lose; mIoU drops. Need selective or different design.

DEAD END 4: "Fix 4-1/5-1 detection only."
→ User required generalization to 1-4, 2-2, 3-1. Must re-run and check all.

DEAD END 5: "Keep vertical filter as rho <= 1200."
→ Breaks 3-1 and other tunnels with W > 1200. Use rho <= W.
```

---

### 4.15 Reasoning Patterns That Worked

```
PATTERN 1: "Verify with GT, implement without GT."
Use GT to compare detected vs. expected, tune params, validate.
Never feed GT into detection, SAM, or pattern logic.

PATTERN 2: "Upstream first."
Fix unfolding/denoising if needed, then enhancing, then detection,
then SAM. Avoid optimizing SAM while detection is broken.

PATTERN 3: "Geometry over semantics."
Prefer crop vs. bounds, line counts, Y spread over segment names
or mIoU. Design so it works for any tunnel.

PATTERN 4: "Generalize after local wins."
After improving 4-1/5-1, explicitly run 1-4, 2-2, 3-1 and
check assume % and line-based %.
```

---

## Appendix: Quick Reference

### Detection Improvements (Code)

- **Vertical filter:** `rho <= W` (image width). No fixed `filter_rings * 1200 / (resolution*1000)`.
- **Preprocessing:** Binary + Canny on normalized depth; OR; then dilate.
- **Hough:** Lower thresholds; relax min_length/max_gap; widen oblique angle ranges as needed.

### Pattern Classification

- **continuous:** `y_std < 100`.
- **simple_staggered:** `is_alternating` or `(y_std < 250 and angle_std < 8)`.
- **complex_staggered:** otherwise.

### Wraparound (Current)

- Disabled by default. Pattern type does **not** auto-enable it.
- When revisiting: use **geometry only** (crop vs. boundaries); no block names, no mIoU.

### GT Usage

- **Allowed:** Parameter tuning, validation, ablation, reverse engineering.
- **Not allowed:** Any use inside the running pipeline (detection, SAM, pattern, wraparound).

---

**End of Report.**
