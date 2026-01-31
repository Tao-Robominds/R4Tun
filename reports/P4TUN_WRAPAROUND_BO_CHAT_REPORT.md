# P4Tun Wraparound & BO Chat Report
## Intrinsic Metrics, Thought Process, Key Parameters & Full Thought Log

**Date:** January 27, 2026  
**Scope:** SAM wraparound baseline on 4-1; diagnosis of low mIoU (detection vs wraparound); GT-based segment positions; Bayesian Optimization of SAM wraparound with 30 runs.  
**Reference:** Structure follows `P4TUN_OPTIMIZATION_JOURNEY_4-1.md`.

---

## Executive Summary

This report documents a single chat window that (1) set a baseline for SAM wraparound on tunnel 4-1 before BO, (2) diagnosed why mIoU was very low, (3) isolated the cause as **detection** rather than wraparound by using ground-truth segment positions, and (4) ran Bayesian Optimization on the SAM wraparound script with GT positions to see if mIoU could be improved further.

**Key outcomes:**
- **Baseline (detected positions):** mIoU 0.074, OA 0.252, F1 0.119.
- **With GT segment positions:** mIoU 0.357, OA 0.541, F1 0.523 (**~4.8× mIoU**). Conclusion: detection is the main bottleneck, not wraparound logic.
- **After SAM wraparound BO (30 runs, early stop at 15):** Best mIoU **0.428** (+19.9% over GT baseline), OA 0.621, F1 0.587.
- **Deliverables:** `generate_all_segments_from_gt.py`, parameterizable `4-2_sam_wraparound.py`, `p4tun/bo/sam_wraparound_bo.py`, and best SAM parameters for 4-1 wraparound.

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

These metrics are recommended for monitoring and debugging each stage during R4Tun/P4Tun exploration, including wraparound and BO work.

---

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Theta Coverage %** | `(theta_max - theta_min) / (2π) × 100` | 98–102% | <100% loses segments; >100% causes wraparound |
| **Ring Count** | Detected rings vs `ring_count.txt` | Exact match | Mismatch propagates to detection and SAM |
| **Point Density** | Points per pixel in depth map | >0.8 | Sparse regions cause detection gaps |
| **Ring Width Consistency** | Std of ring widths (px) | <5% of mean | Inconsistent widths confuse detection |
| **Axis Alignment Error** | Deviation from fitted tunnel axis | <2 mm | Poor alignment distorts θ |

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Surface Retention %** | `valid_after / valid_before × 100` | 70–95% | Too aggressive → lost boundaries; too lenient → noise |
| **Outlier Ratio** | NaN fraction in `depth_map_outlier.npy` | 10–30% | Balance noise removal vs data loss |
| **Edge Preservation** | Gradient magnitude at segment boundaries | >0.7 (relative) | Denoising must preserve edges for detection |

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Interpolation Coverage** | % of depth map filled | >95% | Gaps hurt line detection |
| **Intensity Contrast** | `(max − min) / mean` | >0.3 | Low contrast → poor detection |
| **Segment Boundary Sharpness** | 2nd derivative at edges | >threshold | Blurry boundaries reduce detection accuracy |

---

### Stage 4-1: Detection (4-1_detection.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-Position Count** | Detected vs `ring_count` | Exact match | Wrong count cascades to SAM |
| **Assume %** | % of K positions from “assume” fallback | <30% | Lower → more line-based detections |
| **Line-Based %** | % midpoint / positive_slope / negative_slope | >70% | Direct indicator of detection quality |
| **Y-Position Error vs GT** | Mean \|detected_Y − GT_Y\| across rings | <150 px | Direct measure when GT available for analysis |
| **X-Position Error vs GT** | Mean \|detected_X − GT_X\| | <50 px | Affects segment centering in SAM |
| **Hough Line Counts** | Positive + negative slope lines | >5 each | Too few → unreliable intersections |

**Critical for this chat:** Low mIoU after SAM was traced to **detection** (wrong segment positions). Using GT-derived segment positions (for diagnosis only) showed that wraparound SAM itself is capable of much higher mIoU when anchors are correct.

---

### Stage 4-2: SAM Segmentation (4-2_sam.py / 4-2_sam_wraparound.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **mIoU** | Mean IoU over classes | >0.35 (GT baseline), >0.40 (after BO) | Primary quality signal |
| **Per-Class IoU** | IoU per segment type | >0.15 each | Surfaces weak classes (e.g. K, B1, A1) |
| **Background Ratio** | % background | <30% | High → undersized templates or bad anchors |
| **Template Coverage** | Template vs segment area | 85–95% | Templates guide SAM |
| **Prompt Point Validity** | % prompts within crop bounds | >95% | Out-of-bounds cause failures |
| **Segments Processed** | Successfully processed / total in `all_segments.csv` | 100% | Crop errors (“height and width must be > 0”) indicate bad positions or geometry |

**Wraparound-specific:** For tunnels with wrap-around (e.g. complex_staggered), segment positions must come from pattern discovery or GT-derived `all_segments.csv`; row-based SAM assumes linear layout and fails. Intrinsic check: no flood of “Empty crop” or “height and width must be > 0” for valid rings/blocks.

---

### Stage 5: Evaluation

| Metric | Description | Use |
|--------|-------------|-----|
| **OA** | Overall accuracy | Research / validation only |
| **F1 (macro)** | Macro F1 | Research / validation only |
| **mIoU** | Mean IoU | Research / validation only |
| **Per-Class IoU** | Per-class IoU | Diagnose weak classes |

All evaluation metrics are GT-based and not for use in the deployed solution.

---

## Part 2: The Thought Process Experience

### 2.1 How We Analyzed the Problem

**Initial state:**
- Goal: Set baseline for SAM wraparound on 4-1 before BO (for future 5-1 wraparound focus).
- Action: Run `4-2_sam_wraparound.py` on 4-1.
- Result: Very low mIoU (0.074), OA 0.252, F1 0.119; several A1 blocks failed with “height and width must be > 0”.

**Key question:** Is the low mIoU caused by **wraparound handling** or by **earlier stages** (e.g. detection)?

**Analysis strategy:**
1. **Ablate with GT** – Replace detection-derived segment positions with GT-derived positions (`all_segments.csv` from `final.csv` segment labels). If mIoU jumps, the bottleneck is detection/position quality, not wraparound logic.
2. **Keep pipeline otherwise unchanged** – Same SAM wraparound script, same evaluation; only the source of (X, Y) per segment changes.
3. **Quantify** – Compare mIoU, OA, F1, and per-class IoU before and after the swap.

**Outcome:** With GT positions, mIoU went from 0.074 to 0.357 (~4.8×). So **detection (segment positions) was the main cause** of low mIoU; wraparound processing itself was not the primary failure.

---

### 2.2 What Led to Success

**1. Isolating the variable (GT ablation)**  
- Using GT only for **segment positions** (not for anything else) gave a clean answer: “If positions were correct, would wraparound SAM do well?” Answer: yes (mIoU 0.357).
- This avoided endless tuning of wraparound logic when the real issue was upstream.

**2. Making wraparound script parameterizable**  
- `4-2_sam_wraparound.py` initially had hardcoded segment geometry and template sizes.
- Loading `SAM_PARAMS` from `parameters/<tunnel_id>/parameters_sam.json` allowed BO to tune segment_width, K_height, AB_height, angle_deg, and template dimensions (K and A/B mask widths/heights).
- Without this, BO could not have improved over the default 0.357.

**3. Dedicated BO for wraparound**  
- Using a dedicated objective (`sam_wraparound_bo.py`) that (a) runs only SAM wraparound (no detection), (b) uses GT-based `all_segments.csv`, and (c) optimizes mIoU, kept the search focused and interpretable.
- Early stopping (e.g. DeltaYStopper) avoided wasting runs once the optimum was found (converged by run 15).

**4. Reusing existing evaluation pipeline**  
- Reusing `evaluation.py` and `performance.md` kept metrics consistent and made it easy to compare baseline vs GT positions vs BO.

---

### 2.3 Mistakes Made and What to Avoid

**Mistake 1: Assuming the failure was in the stage we were looking at**  
- **Wrong:** “mIoU is low with wraparound SAM → something is wrong with wraparound.”
- **Right:** First check whether **inputs** to that stage are correct (e.g. segment positions). Use GT ablation to test “if inputs were perfect, would this stage perform well?”
- **Avoid next time:** For any low metric, explicitly ask: “Is this stage’s logic bad, or are its inputs bad?” and test with ideal inputs when possible.

**Mistake 2: Running BO before confirming the stage is the bottleneck**  
- If we had run BO on wraparound while still using poor detection, we would have been optimizing SAM to compensate for wrong positions, with limited gain and confusing results.
- **Right order:** Establish that segment positions are good (or substitute GT for the experiment), then run BO on SAM.
- **Avoid next time:** Before BO on stage N, verify that stages 1..N-1 are “good enough” or fixed for the experiment (e.g. by using GT for upstream outputs).

**Mistake 3: Hardcoded parameters in the script under optimization**  
- Initially, wraparound script did not read from JSON; BO would have had no effect.
- **Fix:** Load all tunable quantities (segment geometry, template mask dimensions) from a parameters file that BO updates.
- **Avoid next time:** Before adding BO for a script, ensure every tunable parameter is read from config/JSON, not hardcoded.

**Mistake 4: Ignoring partial failures in the log**  
- Some segments failed with “height and width must be > 0” (e.g. A1 for several rings). Counting “Successfully processed 54/63” (or 55/57) is an intrinsic metric; ignoring it can hide geometry or position issues.
- **Avoid next time:** Always check “processed / total” and log which (ring, block) fail; fix position or crop logic before claiming success.

---

### 2.4 What to Do Next Time

1. **For any “low metric”:** Form a clear hypothesis (e.g. “detection” vs “wraparound”). Use GT or ideal inputs to test that hypothesis before deep tuning.
2. **Before BO:** Ensure (a) the right stage is the bottleneck, (b) the script reads all tuned parameters from config, and (c) the objective is cheap and stable (e.g. same data, same GT, only params change).
3. **Document baselines:** Always record “default params” and “GT-input baseline” so that BO gain (e.g. 0.357 → 0.428) is clear.
4. **Keep GT out of the pipeline:** Use GT only for segment positions in this experiment; do not feed GT into detection or other production code.

---

## Part 3: Key Parameters of Each Stage

### Stage 1: Unfolding

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `slice_half_thickness` | float | 0.004–0.007 | Slice sampling |
| `max_distance_from_top` | float | 4.2–4.8 | Ring extent |
| `polynomial_degree` | int | 2–4 | Centerline fit |
| `inlier_ratio` (RANSAC) | float | 0.70–0.80 | Ellipse robustness |
| `samples_per_ring` | int | 1100–1400 | Arc length resolution |

---

### Stage 2: Denoising

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `radius_min` / `radius_max` | float | Derived from center ± half_width | Radius filter |
| `theta_step` | float | 0.3–0.7 | Grid resolution |
| `gradient_threshold` | float | 0.1–0.4 | Edge/noise sensitivity |
| `smoothing_window` | int | 2–5 | Cutoff smoothing |

---

### Stage 3: Enhancing

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `curvature_neighbors` | int | 15–30 | Curvature estimation |
| `upsampling_neighbors` | int | 15–30 | Upsampling density |
| `resolution` | float | 0.004–0.006 | Depth map resolution |

---

### Stage 4-1: Detection

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `binary_threshold` | int | 100–150 | Preprocessing |
| `hough_oblique_threshold` | int | 30–80 | Oblique line detection |
| `hough_oblique_min_length` | int | 80–150 | Line length |
| `angle_positive_min` / `angle_positive_max` | float | 5–7, 8–10 | Oblique angle band |
| `hough_vertical_threshold` | int | 400–700 | Vertical lines |
| `merge_distance_threshold` | int | 2–5 | Line merging |

---

### Stage 4-2: SAM (Standard & Wraparound)

**Shared / standard SAM:**

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `segment_width` | float | 1150–1350 mm | Segment width |
| `k_height` | float | 950–1200 mm | K-block height |
| `ab_height` | float | 3100–3500 mm | A/B block height |
| `angle_deg` | float | 6–9° | Segment angle |
| `padding` | int | 100–200 | Crop padding |
| `crop_margin` | int | 30–80 | Crop margin |

**Template mask (K-block):**

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `k_mask_width` | float | 550–750 mm | K template half-width ×2 |
| `k_mask_height_pos` | float | 550–750 mm | K positive side height |
| `k_mask_height_neg` | float | 400–650 mm | K negative side height |

**Template mask (A/B blocks):**

| Parameter | Type | Typical Range | Notes |
|-----------|------|----------------|-------|
| `ab_mask_width` | float | 550–700 mm | A/B template width |
| `ab_mask_height` | float | 1500–1750 mm | A/B template height |

**Best values from this chat (4-1, SAM wraparound BO):**

```yaml
# Best after 15 BO evaluations (GT segment positions)
segment_width: 1153.19
k_height: 1007.72
ab_height: 3196.41
angle_deg: 8.05
k_mask_width: 672.00
k_mask_height_pos: 716.64
k_mask_height_neg: 443.34
ab_mask_width: 608.66
ab_mask_height: 1545.56
```

---

## Part 4: Full “Thought” Experience Log

This section records the reasoning steps, hypotheses, and decisions during the chat.

---

### 4.1 Initial Task and Baseline

**Thought:** User wants to focus on wraparound for 5-1 and use 4-2_sam_wraparound.py. First step is to run SAM wraparound on 4-1 and evaluate to set a baseline before BO.

**Action:** Run `4-2_sam_wraparound.py 4-1` (using venv Python). Check for `all_segments.csv` and required inputs.

**Observation:** Script ran; 54/63 segments processed; 9 A1 blocks failed with “height and width must be > 0”. Evaluation: mIoU 0.074, OA 0.252, F1 0.119.

**Thought:** Baseline is very low. Need to understand whether this is due to wraparound logic or to earlier stages (e.g. detection). User has `detected_gt.csv`; they ask: if we use ground truth, will mIoU improve?

---

### 4.2 Hypothesis: Detection vs Wraparound

**Thought:** If we feed SAM wraparound with **correct** segment positions (from GT) and mIoU improves a lot, then the main problem is **detection/position quality**, not the wraparound implementation. If mIoU stays low even with GT positions, then the problem is in wraparound (crops, templates, etc.).

**Action:** Need a GT-based `all_segments.csv`. The script expects columns Ring, Block, X, Y (and optional quality). `detected_gt.csv` has Type, X, Y (ring-level K positions). So we need to build segment-level positions from GT. GT segment labels are in `final.csv` (column `segment`). So: for each ring and each segment label, compute centroid in (h, theta), then map to pixel (X, Y).

**Action:** Implement `generate_all_segments_from_gt.py`: load `final.csv`, partition by h into rings, for each (ring, segment_label) compute centroid (h_mean, theta_mean), map to (X, Y) via existing bounds and depth map shape, output rows (Ring, Block, X, Y, quality).

**Observation:** Generated `all_segments_gt.csv` with 57 segments (some ring/block combinations may have no points). Replaced `all_segments.csv` with this GT-based file and re-ran SAM wraparound.

**Observation:** With GT positions, evaluation gave mIoU 0.357, OA 0.541, F1 0.523. Per-class IoU: no longer 0 for B1, A1, A2, B2; all classes non-trivial.

**Thought:** mIoU went from 0.074 to 0.357 (~4.8×). So **detection (segment positions) is the main bottleneck**. Wraparound processing itself is capable of much better results when given good positions. Conclusion: recommend improving detection for 4-1 (and later 5-1); use GT only for this diagnostic and for BO experiments, not in production.

---

### 4.3 Decision to Run BO on SAM Wraparound with GT

**Thought:** User wants to see if we can improve further by tuning SAM parameters, using the GT result as the baseline. So: keep using GT-based `all_segments.csv`, and run Bayesian Optimization on the SAM wraparound stage only (no detection run). Success = mIoU above 0.357.

**Action:** Check existing BO: `objective.py` runs detection + SAM (or SAM only for stage `sam`) but uses `4-2_sam.py`, not wraparound. So we need either to point the existing BO at the wraparound script, or to add a small BO dedicated to wraparound. Dedicated script is clearer: `sam_wraparound_bo.py` that (1) takes tunnel_id, n_calls, n_initial, (2) defines a search space for segment geometry and template mask dimensions, (3) each evaluation writes parameters to `parameters/<tunnel_id>/parameters_sam.json`, runs `4-2_sam_wraparound.py`, runs evaluation, parses mIoU, returns negative mIoU for minimization.

**Observation:** The wraparound script does not read from JSON; it uses hardcoded segment_width, K_height, AB_height, angle, and template sizes. So BO would have no effect unless the script is made parameterizable.

**Action:** In `4-2_sam_wraparound.py`: (1) at startup, load `parameters/<tunnel_id>/parameters_sam.json` if present; (2) define a default dict `SAM_PARAMS` (segment_width, K_height, AB_height, angle, k_mask_width, k_mask_height_pos/neg, ab_mask_width, ab_mask_height); (3) in `generate_template_mask` and `sam_segment_all_segments` (and wherever segment geometry is used), use `SAM_PARAMS` instead of literals; (4) pass segment_width, K_height, AB_height into `generate_prompt_points` where needed.

**Observation:** After changes, ran wraparound again; it reported “Loaded SAM parameters from …” and processed 57 segments. So the script is now parameterizable.

---

### 4.4 Running BO and Interpreting Results

**Action:** Implement `sam_wraparound_bo.py` with a reduced search space (segment geometry + K and A/B template dimensions), `SAMWraparoundObjective` that updates JSON and runs wraparound + evaluation, and `gp_minimize` with n_calls=30, n_initial=8, callbacks (DeltaYStopper, CheckpointSaver). Run for tunnel 4-1.

**Observation:** BO ran; best mIoU 0.428 appeared at evaluation 5; convergence flat afterward; stopped at 15 evaluations (early stopping). Best parameters: segment_width 1153.19, k_height 1007.72, ab_height 3196.41, angle_deg 8.05, k_mask_width 672, k_mask_height_pos 716.64, k_mask_height_neg 443.34, ab_mask_width 608.66, ab_mask_height 1545.56.

**Thought:** BO improved mIoU from 0.357 (GT baseline) to 0.428 (+19.9%). So even with correct positions, SAM template and geometry parameters matter. Good to record these as recommended ranges for 4-1-like tunnels and for 5-1 wraparound. Results saved in `p4tun/bo/results/4-1_sam_wraparound_*.json` and history; convergence plot saved.

---

### 4.5 Summary Thought Chain

**Phase 1 – Baseline and problem framing**  
- Run wraparound on 4-1 → very low mIoU (0.074).  
- Question: wraparound bug or upstream (detection)?  
- Plan: Ablate with GT segment positions.

**Phase 2 – Isolating cause**  
- Build GT-based `all_segments.csv` from `final.csv`.  
- Re-run wraparound with GT positions → mIoU 0.357.  
- Conclusion: Detection is the main bottleneck; wraparound logic is not.

**Phase 3 – BO setup**  
- Goal: Improve over 0.357 by tuning SAM only.  
- Blocker: Wraparound script had hardcoded parameters.  
- Fix: Load SAM params from JSON; use them in geometry and templates.  
- Implement dedicated BO script and run 30 (effective 15) evaluations.

**Phase 4 – Result**  
- Best mIoU 0.428 with documented best parameters.  
- Takeaway: For wraparound, (1) fix detection first, (2) then tune SAM geometry and templates via BO when using GT positions for experiments.

---

## Appendix A: File and Script Reference

| Item | Path / Purpose |
|------|----------------|
| GT-based segment list generator | `p4tun/generate_all_segments_from_gt.py` – builds `all_segments.csv` from `final.csv` segment labels |
| Parameterizable wraparound SAM | `p4tun/4-2_sam_wraparound.py` – reads `parameters/<tunnel_id>/parameters_sam.json` |
| SAM wraparound BO | `p4tun/bo/sam_wraparound_bo.py` – BO over segment geometry and template dimensions |
| BO results (4-1) | `p4tun/bo/results/4-1_sam_wraparound_<timestamp>.json` and `_history.json`, `_convergence.png` |
| Evaluation | `p4tun/evaluation.py`; outputs in `data/<tunnel_id>/evaluation/performance.md` |

---

## Appendix B: Metric Summary Table

| Scenario | mIoU | OA | F1 |
|----------|------|-----|-----|
| Baseline (detected positions) | 0.074 | 0.252 | 0.119 |
| GT segment positions | 0.357 | 0.541 | 0.523 |
| After SAM wraparound BO | **0.428** | **0.621** | **0.587** |

---

## Key Takeaways

1. **Diagnose with GT ablation** – When a stage’s output is poor, substitute ideal inputs (e.g. GT positions) to see if the stage itself is capable of good results; that tells you whether to fix upstream or tune the stage.
2. **Detection dominates** – For 4-1, fixing segment positions (GT) gave a much larger mIoU gain than BO on SAM; so prioritize detection quality before heavy SAM tuning.
3. **Make the optimized stage configurable** – BO only helps if every tuned parameter is read from config/JSON, not hardcoded.
4. **Use a dedicated BO when changing the pipeline** – For “wraparound + GT positions only,” a dedicated BO script kept the objective clear and made early stopping and logging straightforward.
5. **Record baselines** – “Default,” “GT-input,” and “after BO” numbers make the value of each step clear and reproducible.

**Final achievement:** From 0.074 (detected) → 0.357 (GT positions) → 0.428 (GT + SAM wraparound BO), with clear attribution to detection vs SAM and a reusable BO setup for wraparound.
