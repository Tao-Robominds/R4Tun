# P4Tun Optimization Journey: Tunnel 5-1 Complex Staggered
## Comprehensive Report on GT Alignment, Detection BO, SAM, and A3 Analysis

**Date:** January 27, 2026  
**Focus Tunnel:** 5-1 (complex_staggered, 7 blocks per ring, 6 K positions)  
**Initial mIoU:** 0.431 → **Final mIoU:** 0.509 (+18%); **A3 IoU:** 0.175 → 0.358 (+105%)

**Reference:** Structure follows `P4TUN_OPTIMIZATION_JOURNEY_4-1.md`.

---

## Executive Summary

This report documents a single chat-window exploration focused on: (1) correct ground-truth generation for tunnel 5-1 using the point cloud and current depth map dimensions; (2) comparison of SAM scripts and detection-stage improvements via Bayesian Optimization; (3) root-cause analysis of the lowest-performing class (A3-block); and (4) the full “thought” process—what worked, what failed, and what to avoid next time.

**Key outcomes:**
- **GT generation:** Fixed `generate_all_segments_from_gt.py` to use the **actual `ring` column** from the point cloud instead of partitioning by `h`; this removed large segment-position errors (up to ~2000 px) and doubled A3 IoU.
- **Detection:** Regenerated K-point GT from enhanced point cloud (6 K positions; ring 110 has no K block). Ran detection BO; mean K-position error improved from 204 px to 110 px.
- **SAM:** Ran complex_staggered SAM with GT segment positions; BO did not improve over baseline (best mIoU 0.431). Restored best SAM params; final mIoU 0.509 after GT fix.
- **A3 analysis:** A3 was lowest because Ring 110 (no K block) had 0% A3 accuracy (96.5% predicted as Background). Crop-window and template-size experiments (larger crop, larger A-block mask) hurt overall mIoU; the main fix was correct ring-aligned GT.

---

## Part 1: Recommended Intrinsic Quality Metrics by Stage

### Stage 0: Ground Truth Generation (`generate_all_segments_from_gt.py`)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Ring source** | Use actual `ring` column vs h-partition | Prefer `ring` when present | h-partition creates wrong ring ordering → segment positions off by hundreds–thousands of pixels |
| **Pixel mapping** | (h, θ) → (X, Y) using same bounds as depth map | Same as `depth_map_outlier.npy` / final.csv | Inconsistent bounds cause misalignment with detection and SAM |
| **Segment count per ring** | Blocks per ring (K, B1, A1–A4, B2) | Match point cloud labels | Missing blocks (e.g. ring 110 no K) must be reflected in GT |
| **Centroid vs intersection** | K in GT = centroid of K-block points | Centroid | Intersection-based “K” does not match segment centroids used by SAM |

**Critical finding:** For 5-1, the script was partitioning by `h` and assigning ring IDs 0..n_rings-1. The point cloud uses rings 107–113; ring 110 has no K block. Using the actual `ring` column and computing centroids per (ring, segment) gave correct segment positions and fixed A3 and overall mIoU.

---

### Stage 1: Unfolding (1_unfolding.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Theta coverage %** | `(theta_max - theta_min) / (2π) × 100` | 98–102% | &lt;100% loses circumference; &gt;100% causes wraparound |
| **Point density** | Points per pixel in depth map | &gt;0.8 | Sparse regions cause detection and SAM gaps |
| **Ring width consistency** | Std of ring widths (px) | &lt;5% of mean | Inconsistent widths confuse complex_staggered detection |

*(Unfolding was not re-tuned in this chat; metrics carried from prior R4Tun exploration.)*

---

### Stage 2: Denoising (2_denoising.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Surface retention %** | Valid points after / before | 70–95% | Too aggressive removes segment boundaries |
| **Outlier ratio** | NaN fraction in depth map | 10–30% | Balances noise removal vs data loss |
| **Edge preservation** | Gradient at segment boundaries | &gt;0.7 (relative) | Needed for line detection and SAM boundaries |

*(Denoising was not re-tuned in this chat.)*

---

### Stage 3: Enhancing (3_enhancing.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **Interpolation coverage** | % of depth map filled | &gt;95% | Gaps hurt line detection |
| **Pattern type** | complex_staggered for 5-1 | Correct label | Drives detection and SAM strategy |
| **Intensity contrast** | `(max − min) / mean` | &gt;0.3 | Low contrast hurts detection and SAM |

*(Enhancing was not re-tuned in this chat.)*

---

### Stage 4-1: Detection (4-1_detection_complex.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **K-position count** | Detected K count vs GT K count | Match (6 for 5-1) | Wrong count cascades to SAM segment layout |
| **Mean K-position error** | Mean Euclidean distance to nearest GT K | &lt;150 px (good), &lt;200 px (acceptable) | Primary detection quality signal |
| **Max K-position error** | Worst-ring error | &lt;300 px | Surfaces rings needing better line/intersection logic |
| **Per-ring error** | Error per ring | Identify outlier rings | Non-uniform ring spacing (e.g. 5-1 ring 6 gap) increases difficulty |
| **Line detection** | Horizontal + oblique line counts | Enough for intersections | Sparse lines → more cluster/midpoint fallbacks, higher error |

**Critical finding:** 5-1 has **non-uniform ring spacing** (e.g. one large X-gap). Detection BO improved mean error from 204 px to 110 px; one ring (e.g. Ring 108 near top edge) can still have large residual error (e.g. 473 px). GT must be K-block **centroids** from the point cloud, not line-intersection “K” positions.

---

### Stage 4-2: SAM Segmentation (4-2_sam_complex.py)

| Metric | Description | Good Range | Why It Matters |
|--------|-------------|------------|----------------|
| **mIoU** | Mean IoU over classes | &gt;0.45 (good), &gt;0.50 (excellent) | Primary segmentation quality |
| **Per-class IoU** | IoU per segment type | &gt;0.15 each; identify lowest | Surfaces weak classes (e.g. A3) and guides analysis |
| **Background ratio** | % points predicted as background | &lt;30% | High → undersegmentation or misaligned GT |
| **Ring-wise accuracy (per class)** | % correct per ring for a class | Identify rings with 0% or very low | Structural issues (e.g. ring without K block) show up here |
| **Segment processing count** | Segments processed vs expected (e.g. 45 for 5-1) | No skips | Skipped segments (e.g. due to format bugs) cause whole rings to fail |

**Critical finding:** For 5-1, A3 was lowest mainly because **Ring 110 (no K block) had 0% A3 accuracy** (almost all A3 GT predicted as Background). SAM mask for that ring’s A3 was very small; enlarging crop or A-block template hurt overall mIoU due to overlap. Correct GT alignment was the main fix.

---

### Stage 5: Evaluation (evaluation.py)

| Metric | Description | Use |
|--------|-------------|-----|
| **OA** | Overall accuracy | Research/validation only |
| **F1 (macro)** | Macro F1 | Research/validation only |
| **mIoU** | Mean IoU | Primary summary metric |
| **Per-class IoU** | IoU per class | Diagnose weak classes (e.g. A3) |
| **Confusion (GT→pred)** | For a GT class, % predicted as each class | Find systematic misclassification (e.g. A3→Background) |

All evaluation metrics are GT-based and not for use in the deployed solution.

---

## Part 2: The Thought Process Experience

### 2.1 Problem Analysis Journey

#### What we started with

- User request: (1) Ensure 5-1 K-point GT is correct from `data/5-1.txt` and current depth map dimensions; (2) Compare `sam.py` vs `sam_complex.py`; (3) Run detection BO to improve detection; (4) Run SAM and check final mIoU and whether further BO helps; (5) Investigate why A3 IoU is so low.
- Pipeline: complex_staggered detection and SAM on 5-1; previous SAM BO had given mIoU ~0.431.
- Observation: Detection “getting obviously worse”; expectation of more horizontal and oblique lines.

#### First step: GT and detection

- **GT:** Regenerated K-point GT from enhanced point cloud: use cylindrical (h, θ), bounds from `depth_map_outlier.npy` (or final.csv), map to pixel (X, Y), then centroid per ring for segment label K. Result: 6 K positions (rings 107–113; ring 110 has no K block in labels). This became the correct `detected_gt.csv` for detection BO.
- **Detection BO:** Ran detection BO against this GT. Best score improved (e.g. ~0.855); mean K error went from 204 px to ~110 px. Detection still outputs 7 K positions for 6 GT K; one spurious detection and one bad ring (e.g. 473 px error) remain.
- **SAM:** Ran SAM complex with `all_segments_gt.csv` (generated from same pipeline). Initially `all_segments_gt.csv` was missing; we regenerated it. mIoU was 0.431; SAM BO over 30 iterations did not improve (best remained 0.431). Restored best BO params; after fixing GT generation (see below), mIoU reached 0.509.

#### Why A3 was lowest

- **Per-class IoU:** A3 was 0.175 (then 0.358 after GT fix). We inspected confusion: most A3 GT points were predicted as Background; then by ring: Ring 110 had 0% A3 accuracy (96.5% as Background).
- **Segment positions:** Checked `all_segments_gt.csv` vs point cloud. Discovered that **segment positions were wrong**: the script partitioned points by `h` and assigned ring IDs 0..n_rings-1, ignoring the actual `ring` column. That caused large position errors (e.g. 776–2053 px) and wrong ring–segment mapping.
- **Fix:** Use the actual `ring` column when available; compute centroids per (ring, segment); map (h, θ) to pixels with the same bounds as the depth map. Regenerated `all_segments_gt.csv` (45 segments for 5-1). Re-ran SAM and evaluation: mIoU 0.431 → 0.509; A3 0.175 → 0.358.
- **Ring 110:** Even with correct GT, Ring 110 (no K, no B1) still had ~0% A3 accuracy. Analysis: crop window (233×620 px) smaller than A3 span (361×698 px); SAM mask for that ring’s A3 was very small (~3197 px). Trying larger crop or larger A-block template reduced mIoU (overlap/confusion). Conclusion: Ring 110 A3 is a structural limitation of the current SAM setup (no K anchor), not fixed by simple parameter scaling.

#### Bug introduced during debugging

- Added debug print for Ring 3 A3 using `score:.4f`. `score` was a numpy array; formatting raised `unsupported format string passed to numpy.ndarray.__format__`, so **Ring 3 (PC 110) A3 was skipped** and only 44 segments were processed. Fix: remove the faulty format or use `score` in a way that handles arrays. After fix, 45 segments processed and A3 stayed at 0.358.

---

### 2.2 What Made Success Possible

1. **Verify GT and alignment first.**  
   When a class (A3) or overall metric (mIoU) is off, check that GT segment positions and ring mapping match the point cloud and depth map. Position errors of hundreds/thousands of pixels often come from GT generation, not from SAM or detection alone.

2. **Per-ring and per-class breakdown.**  
   Aggregating only mIoU/per-class IoU can hide one bad ring (e.g. 0% A3 on Ring 110). Breaking accuracy by ring and by “what GT class was predicted as what” quickly pointed to Ring 110 and A3→Background.

3. **Compare GT to source data.**  
   We compared segment positions from `all_segments_gt.csv` to centroids computed from the point cloud (per ring, per segment). That immediately showed the h-partition ring mapping was wrong and that fixing it was necessary before further tuning.

4. **Incremental experiments.**  
   After fixing GT, we tried larger crop and larger A-block mask one at a time. Both made mIoU worse. Reverting and documenting “larger is not better here” avoided stacking bad changes.

---

### 2.3 Mistakes and What to Avoid Next Time

1. **Trusting ring IDs from h-partition.**  
   Using `np.linspace` over sorted `h` to assign ring IDs ignored the true `ring` column and produced wrong segment positions. **Always prefer actual ring labels when available** and validate positions against the point cloud.

2. **Debug code that can change behavior.**  
   A print with `score:.4f` caused an exception and skipped one segment. **Avoid formatting variables that might be arrays/scalars** in hot path; or catch and log safely so one segment cannot be silently skipped.

3. **Scaling up crops/templates without checking overlap.**  
   Larger crop or larger A-block mask increased overlap between segments and reduced mIoU. **Test crop/template size in a controlled way** and monitor per-class and per-ring impact, not only mIoU.

4. **Optimizing downstream before fixing GT.**  
   Running SAM BO and tuning SAM while GT segment positions were wrong wasted compute. **Confirm GT (and detection GT) alignment with the data before heavy BO on SAM.**

5. **Assuming all rings are equivalent.**  
   Ring 110 has no K (and no B1). Detection and SAM behavior can differ for such rings. **Explicitly check rings with missing block types** and consider ring-specific logic or accept limited accuracy there.

---

### 2.4 Recommendations for Future Work

- **GT generation:** For any new tunnel or pipeline change, add a sanity check: compare generated segment centroids to point-cloud centroids (same ring, same segment) and flag large discrepancies.
- **Detection:** Continue to use K-block centroids (from point cloud) as detection GT, not line-intersection points. Consider ring-specific or pattern-specific detection parameters for non-uniform ring spacing.
- **SAM:** For rings without K block, consider dedicated strategies (e.g. different prompt strategy or post-processing) rather than only enlarging templates/crops.
- **BO:** Ensure evaluation reads metrics correctly (e.g. parsing `performance.md` table format for mIoU/OA/F1) so BO history and best params are saved correctly.

---

## Part 3: Key Parameters of Each Stage

### 3.1 Ground Truth Generation

| Parameter / design choice | Description | Recommended (5-1) |
|---------------------------|-------------|-------------------|
| **Ring source** | Use `ring` from point cloud when present | Use actual `ring`; fallback to h-partition only if no `ring` |
| **Bounds for (h, θ)→(X, Y)** | h_min, h_max, theta_min, theta_max | From `final.csv` (or same as depth map); shape H, W from `depth_map_outlier.npy` |
| **Mapping** | Linear map (h, θ) to [0, W-1], [0, H-1] | `x = (h - h_min) / (h_max - h_min) * (W - 1)`; same for θ → y |
| **Centroids** | Per (ring, segment) | Mean of (h, θ) over points with that ring and segment label; then map to pixel |

---

### 3.2 Detection (5-1, complex_staggered)

File: `p4tun/parameters/5-1/parameters_detection.json`. Key groups:

| Group | Key parameters | Typical role |
|-------|----------------|--------------|
| **preprocessing** | `binary_threshold`, `dilation_kernel_size`, `dilation_iterations`, `use_morphological_closing`, `use_depth_gradients` | Edge/binary image for Hough and lines |
| **hough_oblique** | `threshold`, `min_length`, `max_gap`, `angle_positive_min/max`, `angle_negative_min/max` | Oblique line detection |
| **hough_horizontal** | `threshold`, `min_length`, `max_gap`, `angle_tolerance` | Horizontal lines |
| **hough_vertical** | `threshold` | Vertical lines |
| **line_processing** | `merge_distance_threshold`, `merge_close_threshold`, `oblique_min_y_span`, `oblique_min_x_span`, clustering flags | Merging and filtering lines |
| **complex_staggered** | **hough_re_detect:** `threshold`, `min_length`, `max_gap` | Re-detection for complex pattern |
| | **angle_range:** `positive_min/max`, `negative_min/max` | Angle filter for oblique lines |
| | **line_filtering:** `min_y_span`, `min_x_span` | Minimum line extent |
| | **clustering:** `eps_candidates`, `min_clusters`, `subdivision_threshold`, `max_subdivisions` | DBSCAN-style clustering for K positions |
| | **confidence:** `subdivision_base/factor`, `cluster_base/factor`, `midpoint`, `final_intersection`, `final_midpoint` | Confidence scoring |

**Example values (after BO):**  
`binary_threshold` 82, `dilation_kernel_size` 3, `dilation_iterations` 4; oblique `threshold` 49, `min_length` 59, `max_gap` 64; complex_staggered `hough_re_detect.threshold` 43, `min_length` 100, `max_gap` 52; `eps_candidates` [0.05, 0.11, 0.15, 0.17, 0.22]; confidence weights ~0.69–0.90.

---

### 3.3 SAM (5-1, complex_staggered)

File: `p4tun/parameters/5-1/parameters_sam.json`. Key groups:

| Group | Key parameters | Typical role |
|-------|----------------|--------------|
| **segment_geometry** | `segment_width`, `k_height`, `ab_height`, `angle_deg` | Physical dimensions; used for crop size and layout |
| **image** | `resolution` | mm per pixel (e.g. 0.005) |
| **pattern_aware** | `use_quality_weighting`, `min_quality_threshold` | Quality-based weighting |
| **prompt_points.template_mask** | **k_block:** `width`, `height_pos`, `height_neg` | K-block template polygon |
| | **b1_block / b2_block:** `width`, heights (top/bottom, pos/neg) | B-block template polygons |
| | **a_blocks:** `width`, `height` | A-block template polygon (shared) |
| **complex_staggered** | **template_sizing:** `k_block_width_factor`, `k_block_height_factor`, `ab_block_width_factor`, `ab_block_height_factor` | Scale factors for templates |
| | **prompt_density:** `k_block_points`, `ab_block_points` | e.g. `"sparse"` / `"standard"` / `"dense"` (if implemented) |

**Example values (BO / restored):**  
`segment_width` 1166.5, `k_height` 950, `ab_height` 3100, `angle_deg` ~7.29; k_block template width ~563, height_pos ~707, height_neg ~583; a_blocks width 550, height 1567.2; template_sizing factors ~0.97–1.2; prompt_density sparse for k and ab.

---

### 3.4 Evaluation

No tunable parameters; reads `final.csv` vs GT and writes `performance.md` and per-class IoU. Metrics: OA, F1 (macro), mIoU, per-class IoU.

---

## Part 4: Full “Thought” Experience Log

This section records the reasoning chains, hypotheses, dead ends, and questions that drove the exploration.

---

### 4.1 Initial Setup and GT for Detection

**Thought:**  
“User wants 5-1 K-point GT correct from data/5-1.txt and current depth map dimensions.”

**Action:**  
Checked enhanced.csv (has `segment`, `ring`, `h`, `theta`). Confirmed depth map size (e.g. 2545×3469). Filtered points with segment==1 (K) and pred!=8 (original). Mapped (h, θ) to pixel using bounds from the same source as the depth map.

**Observation:**  
Only 6 rings had K-block points (107–113; 110 has no K in labels). Saved 6 K centroids to `detected_gt.csv`.

**Thought:**  
“Detection evaluation and BO need this GT. Count 6 matches the data; ring 110 is special.”

---

### 4.2 Detection BO and SAM Run

**Thought:**  
“Run detection BO to improve detection; then run SAM and see final mIoU and whether more BO helps.”

**Action:**  
Ran detection BO (e.g. 50 calls, 10 initial). Best score ~0.855; mean K error 204→110 px. Ran SAM complex with `all_segments_gt.csv` (generated once); mIoU 0.431. Ran SAM BO (e.g. 30 calls); best mIoU stayed 0.431.

**Observation:**  
SAM BO did not beat the initial SAM params. Restored best SAM params from BO history. Noted that BO result file stored `best_score` and `history` with key `score` (not `miou`); values were correct.

**Thought:**  
“For this tunnel and this setup, SAM BO did not find a better point. Main lever later turned out to be GT alignment, not more SAM tuning.”

---

### 4.3 Why Is A3 So Low?

**Thought:**  
“A3 IoU is the lowest. Need to understand: confusion (what A3 is predicted as), and whether some rings are much worse.”

**Action:**  
Computed for A3 GT: distribution of predicted labels. Then split by ring and by Y band (top vs middle).

**Observation:**  
Most A3 GT predicted as Background; Ring 110 had 0% A3 accuracy (almost all A3→Background). Other rings had 67–86% A3 accuracy.

**Thought:**  
“So the problem is concentrated in one ring (110). Next: Is it position (edge), or is it that ring 110 has no K block and something in the pipeline treats it differently?”

---

### 4.4 Discovering Wrong Segment Positions (GT Bug)

**Thought:**  
“Maybe segment positions in `all_segments_gt.csv` are wrong for some rings. If SAM is given wrong (X, Y), it will segment the wrong place.”

**Action:**  
Compared segment positions in `all_segments_gt.csv` to centroids computed from the point cloud (same ring, same segment). Used same (h, θ)→(X, Y) mapping as the depth map.

**Observation:**  
For A3, positions from the file did not match point-cloud centroids; errors were huge (e.g. 776–2053 px). Checked how `generate_all_segments_from_gt.py` assigns rings: it partitioned by sorted `h` and assigned ring_id 0..n_rings-1, **ignoring the actual `ring` column**.

**Realization:**  
“Ring IDs in the script were not the same as the point cloud’s rings. So segment positions were wrong for multiple rings. Fix: use actual `ring` when available.”

**Action:**  
Updated script to use `ring` from the CSV when present; compute centroids per (ring, segment); keep same pixel mapping. Regenerated `all_segments_gt.csv` (45 segments). Re-ran SAM and evaluation.

**Result:**  
mIoU 0.431 → 0.509; A3 0.175 → 0.358. Confirmed: fixing GT alignment was the main fix for A3 and overall mIoU.

---

### 4.5 Ring 110 and Crop/Template Size

**Thought:**  
“Ring 110 still has 0% A3. Is the crop too small so SAM doesn’t see the full A3?”

**Action:**  
Computed A3 span in pixels (X and Y) for ring 110 and compared to crop dimensions (from segment_width and ab_height). Observed crop smaller than A3 span (e.g. 233 vs 361 px width).

**Experiment 1:**  
Increased `segment_width` and `ab_height` to get larger crop. Re-ran SAM and evaluation.

**Result:**  
mIoU dropped (e.g. 0.509 → 0.325). More overlap between segments; worse overall.

**Thought:**  
“Larger crop hurts. Revert and try only template size.”

**Experiment 2:**  
Reverted crop; increased A-block template width/height (e.g. 700, 1800). Re-ran SAM.

**Result:**  
mIoU and A3 both slightly worse (e.g. A3 0.358 → 0.344). Reverted.

**Conclusion:**  
“For this setup, blindly enlarging crop or A-block template is harmful. Ring 110 A3 remains a structural limitation (no K anchor); we document it rather than force a simple scale-up.”

---

### 4.6 Debug Code Bug (Skipped Segment)

**Thought:**  
“Add a quick debug print for Ring 3 A3 to see score and mask size.”

**Action:**  
Inserted `print(f"..., score={score:.4f}, ...")` after SAM prediction for block A3, ring_id 3.

**Observation:**  
Run reported “Successfully processed 44 segments” instead of 45; Ring 3 A3 was not in results. Error: `unsupported format string passed to numpy.ndarray.__format__` — `score` was an array.

**Realization:**  
“The exception in the loop caused that segment to be skipped. So Ring 3 (PC 110) A3 was never applied; its pixels stayed background. That’s why we saw 0% A3 for ring 110.”

**Action:**  
Removed the faulty format (or made it safe for arrays). Re-ran; 45 segments processed; A3 back to 0.358.

**Lesson:**  
“Debug code in the hot path must not assume variable types (e.g. scalar vs array). Prefer safe logging or separate checks so one segment cannot be silently skipped.”

---

### 4.7 Questions That Drove Discovery

- “Why is A3 the lowest?” → Confusion and per-ring breakdown → Ring 110 0% A3.
- “Do segment positions in all_segments_gt match the point cloud?” → Compare centroids → No; ring mapping was wrong.
- “Can we fix A3 by larger crop or template?” → Two experiments → No; mIoU and A3 got worse.
- “Why only 44 segments processed?” → Check logs and code → Format bug in debug print caused one segment to be skipped.

---

### 4.8 Dead Ends and What Didn’t Work

- **Larger global crop (segment_width, ab_height):** Improved coverage of A3 in ring 110 but increased overlap; mIoU dropped a lot. Reverted.
- **Larger A-block template only:** Slight degradation in mIoU and A3. Reverted.
- **Relying on h-partition for ring IDs:** Produced wrong segment positions and wrong ring–segment mapping. Replaced by using actual `ring` column.

---

### 4.9 Summary of Thought Flow

```
PHASE 1: ESTABLISH CORRECT GT
├── Regenerate K-point GT from point cloud (centroids, correct bounds)
├── Use actual ring column in generate_all_segments_from_gt when available
└── Validate segment positions against point-cloud centroids

PHASE 2: IMPROVE DETECTION AND SAM
├── Run detection BO against K-point GT → lower mean K error
├── Run SAM with GT segment list; run SAM BO → no gain over baseline
└── Restore best SAM params; confirm mIoU with current GT

PHASE 3: DIAGNOSE WEAK CLASS (A3)
├── Confusion: A3 mostly → Background
├── Per-ring: Ring 110 has 0% A3 accuracy
├── Check segment positions → discover wrong ring mapping in GT script
└── Fix GT generation → big gain in mIoU and A3

PHASE 4: UNDERSTAND RING 110 AND LIMITS
├── Crop/template size experiments → worse mIoU
├── Fix debug-print bug that skipped Ring 3 A3
└── Conclude: Ring 110 (no K) is structural; document and avoid harmful scaling

RESULT: mIoU 0.431 → 0.509; A3 0.175 → 0.358; robust GT generation and clear limits of simple parameter scaling.
```

---

## Appendix: Quick Reference

### Tunnel 5-1 facts (from this exploration)

- **Rings (point cloud):** 107–113 (7 rings). **K blocks:** 6 (ring 110 has no K in labels).
- **Depth map:** e.g. 2545×3469 px; bounds from final.csv / depth_map_outlier.npy.
- **GT segments:** 45 (some rings missing K or B1).
- **Detection:** 6 GT K positions; after BO, mean error ~110 px; one ring can still have large error.
- **SAM:** Best mIoU 0.509 with correct GT; A3 0.358; Ring 110 A3 remains ~0% with current approach.

### Files touched

- `p4tun/generate_all_segments_from_gt.py`: use actual `ring` when available; same (h, θ)→pixel mapping as depth map.
- `p4tun/parameters/5-1/parameters_detection.json`: detection BO result for 5-1.
- `p4tun/parameters/5-1/parameters_sam.json`: SAM params (BO best restored).
- `data/5-1/detected_gt.csv`: 6 K centroids from point cloud.
- `data/5-1/all_segments_gt.csv`: 45 segment positions (ring-aligned).

### Takeaways

1. **GT alignment is foundational.** Wrong ring mapping or wrong pixel mapping can cost hundreds of pixels and dominate over algorithm tuning.
2. **Per-ring and per-class analysis** quickly finds the worst ring (e.g. 110) and the dominant confusion (e.g. A3→Background).
3. **Debug code in the hot path** must not assume types; one skipped segment can zero out a whole ring’s class.
4. **Larger crop/template** can reduce mIoU; test and revert if needed.
5. **Rings with missing block types** (e.g. no K) may need different handling; document and consider targeted strategies.

---

*End of report.*
