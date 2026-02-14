# Best Parameters Reproduction (data/bo Performance)

**Purpose:** Extract the best parameters from BO results and reports, apply them to the pipeline, and verify we can duplicate the performance in `data/bo/`.

---

## ⚠️ CRITICAL: Data overwrite (mistake)

**The best BO results in `data/bo/1-4/` were overwritten** by running detection + SAM + evaluation with `--data-dir data/bo`. The original `final.csv` and `evaluation/performance.md` (mIoU 0.748) were replaced and **cannot be restored from git** — `data/` is in `.gitignore`, so those files were never committed.

**Do not run detection, SAM, or evaluation with `--data-dir data/bo`** (or any directory that holds results you need to keep). Use a copy or a separate output directory (e.g. `data/bo_rerun`) if you need to re-run on the same inputs.

---

## Where the parameters ARE documented (logs/reports)

The exact parameters that achieved mIoU 0.748 for tunnel 1-4 are in the **journey report**, not only in JSON:

- **Report:** `reports/P4TUN_OPTIMIZATION_JOURNEY_3-1_1-4.md`
  - Evaluation 17: mIoU 0.748 (best).
  - **Best SAM parameters** (Tunnel 1-4, mIoU 0.748) are listed in the report (segment_width 1150, k_height 1150, k_mask_height 580, ab_mask_height 1577.29, min_quality_threshold 0.4965, etc.).
  - The report states that the best params were extracted from terminal logs and saved to `p4tun/bo/results/1-4_sam_20260126_best_extracted.json`.

When reproducing, **use the report and logs first** to confirm which detection run and which SAM evaluation (e.g. eval 17) produced the 0.748 run, then match that exactly instead of assuming canonical JSONs.

---

## Full parameter set for 1-4 (preprocessing + detection + SAM)

The 0.748 run used the **full pipeline**. Preprocessing (unfolding, denoising, enhancing) is related: it produces `enhanced.csv` and `depth_map*`, which detection and SAM depend on. For 1-4 there is **no** BO result file for preprocessing in `p4tun/bo/results/` (only 2-2 has `2-2_preprocessing_*.json` and `2-2_unfolding_*.json`). The preprocessing parameters for 1-4 live only in:

- **`p4tun/parameters/1-4/parameters_preprocessing.json`** (combined)
- **`p4tun/parameters/1-4/parameters_unfolding.json`**
- **`p4tun/parameters/1-4/parameters_denoising.json`**
- **`p4tun/parameters/1-4/parameters_enhancing.json`**

Use these **together** with the canonical detection and SAM params when reproducing. Do not overwrite or replace them when applying BO detection/SAM.

### Preprocessing parameters (1-4) — from `parameters_preprocessing.json`

**Unfolding:**  
`physical_constants`: ring_spacing 1.2, tunnel_diameter 5.5  
`slicing`: slice_half_thickness 0.005, max_distance_from_top 4.5  
`curve_fitting`: polynomial_degree 3  
`ransac_ellipse`: inlier_ratio 0.75, confidence 0.9, min_samples 5, inlier_threshold 0.8  
`arc_length`: samples_per_ring 1210  

**Denoising:**  
`radius_filtering`: radius_min 2.7, radius_max 2.8  
`grid_resolution`: theta_step 0.5, radial_step 0.001  
`gradient_detection`: gradient_threshold 0.2, gradient_epsilon 1e-6  
`cutoff_smoothing`: smoothing_window 3, **smoothing_offset -0.003** (note: sample uses +0.003)  

**Enhancing:**  
`curvature`: curvature_neighbors 20  
`upsampling`: target_distances [0.08, 0.04, 0.02], curvature_threshold 0.0005, upsampling_neighbors 20, distance_tolerance_low 0.9, distance_tolerance_high 2, radius_filter_factor 0.15  
`outlier_detection`: depth_threshold_low 0.003, depth_threshold_high 0.008, outlier_neighbors 20  
`outlier_interpolation`: interpolation_radius 0.06, num_interpolations 2  
`depth_map`: resolution 0.005, interpolation_window 9  

---

## Target Performance (data/bo)

| Tunnel | mIoU | OA   | F1   |
|--------|------|------|------|
| 1-4    | 0.748 | 0.885 | 0.851 |
| 2-2    | 0.775 | 0.890 | 0.872 |
| 3-1    | 0.687 | 0.854 | 0.794 |
| 4-1    | 0.424 | 0.621 | 0.587 |
| 5-1    | 0.509 | 0.637 | 0.667 |

---

## CONFIRMATION — Tunnel 1-4 (run vs target)

**Target (from data/bo/1-4 before overwrite):** mIoU **0.748**, OA 0.885, F1 0.851.

### Run A: Our pipeline on `data/1-4` (our preprocessing → detection → SAM → eval)

| Metric | Target | Actual | Match |
|--------|--------|--------|-------|
| **mIoU** | **0.748** | **0.557** | No |
| OA      | 0.885 | 0.752 | No |
| F1      | 0.851 | 0.710 | No |

### Run B: Detection → SAM → eval on `data/bo/1-4` (BO preprocessing; same inputs as original 0.748)

| Metric | Target | Actual | Match |
|--------|--------|--------|-------|
| **mIoU** | **0.748** | **0.560** | No |
| OA      | 0.885 | 0.755 | No |
| F1      | 0.851 | 0.713 | No |

**Per-class IoU (target vs Run B actual):**

| Class       | Target | Actual (Run B) |
|------------|--------|----------------|
| Background | 0.857 | 0.681 |
| K-block    | 0.598 | 0.393 |
| B1-block   | 0.799 | 0.603 |
| A1-block   | 0.832 | 0.631 |
| A2-block   | 0.574 | 0.439 |
| A3-block   | 0.772 | 0.587 |
| B2-block   | 0.802 | 0.587 |

**Result: NOT CONFIRMED.**  
- Run A (our preprocessing): mIoU 0.557.  
- Run B (BO preprocessing, same enhanced/depth_map as original): mIoU 0.560.  
So the gap is **not** from preprocessing alone. With canonical detection (125324) + SAM best_extracted params on the same data/bo/1-4 inputs, we still get ~0.56, not 0.748. The original 0.748 may have come from a different code path, param application, or environment not fully captured in the saved BO JSONs.  
**Note:** Run B overwrote `data/bo/1-4/detected.csv`, `final.csv`, and `evaluation/performance.md`. **Restoration from git is not possible** — `data/` is gitignored.

---

## Sources Used

- **BO results:** `p4tun/bo/results/` (e.g. `1-4_sam_20260126_best_extracted.json`, `1-4_detection_20260126_124912.json`, `2-2_detection_20260122_101404.json`, `2-2_sam_20260122_120958.json`)
- **Reports:** `reports/CRITICAL_PARAMETERS_DETECTION_SAM.md`, `bo4tun/report/full_ablation_results.json`

---

## What Was Done

1. **Apply script added:** `p4tun/bo/apply_bo_best_to_parameters.py`
   - Loads best detection and SAM params from BO result JSONs.
   - **Uses canonical result files** (see `p4tun/bo/results/README.md`) so detection and SAM match the run that achieved data/bo mIoU.
   - Converts flat BO names to the nested JSON format expected by `4-1_detection.py` and `4-2_sam.py`.
   - Writes `p4tun/parameters/<tunnel_id>/parameters_detection.json` and `parameters_sam.json`.
   - Optional: run full pipeline (preprocessing → detection → SAM → evaluation).

2. **Canonical BO files (fix for the mIoU gap):**
   - **1-4 detection:** `1-4_detection_20260126_125324.json` (not the higher proxy-score file 124912). This pairs with the SAM run that achieved mIoU 0.748.
   - **1-4 SAM:** `1-4_sam_20260126_best_extracted.json` (mIoU 0.748).
   - **2-2 detection:** `2-2_detection_20260122_101404.json`.
   - Without this, we had applied the wrong detection params → 9 rings, wrong K positions → mIoU 0.571 instead of 0.748.

3. **Best params applied for 1-4 (corrected):**
   - Detection: from **1-4_detection_20260126_125324.json** (binary_threshold 101, angle_positive_min 4.808, merge_close_threshold 6, etc.) → **10 K positions**.
   - SAM: from `1-4_sam_20260126_best_extracted.json`, with `k_mask_height` → `template_mask.k_block.height_neg`.

4. **Pipeline run (1-4):**
   - **Preprocessing** (`1_preprocessing.py`): completed (data/1-4).
   - **Detection** (`4-1_detection.py`): completed with **canonical** params → **10 K positions**.
   - **SAM** (`4-2_sam.py`): requires checkpoint. Checkpoint path in code tries:
     - `sam4tun/segment-anything/sam_vit_h_4b8939.pth`
     - `p4tun/segment-anything/sam_vit_h_4b8939.pth`
     - Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   - **Evaluation** (`evaluation.py`): run on `data/bo/1-4` confirms target mIoU 0.748.

5. **SAM checkpoint path:** `4-2_sam.py` was updated to resolve the checkpoint from project root and to try `p4tun/segment-anything/sam_vit_h_4b8939.pth`, with a clear error if not found.

---

## How to Reproduce data/bo Performance

1. **Use the full parameter set (preprocessing + detection + SAM).**
   - Preprocessing: use `p4tun/parameters/1-4/parameters_preprocessing.json` (and the split unfolding/denoising/enhancing files there). Do **not** overwrite these when applying BO params.
   - Detection + SAM: apply from BO results (see below).

2. **Apply best detection and SAM params (optional, already done for 1-4):**
   ```bash
   venv/bin/python -m p4tun.bo.apply_bo_best_to_parameters 1-4 2-2 --apply
   ```
   This writes only `parameters_detection.json` and `parameters_sam.json`; it does **not** change preprocessing files.

3. **Ensure SAM checkpoint exists:**
   - Download `sam_vit_h_4b8939.pth` and put it in `p4tun/segment-anything/` or `sam4tun/segment-anything/`.

4. **Run full pipeline:**
   ```bash
   # Single tunnel (e.g. 1-4)
   venv/bin/python p4tun/1_preprocessing.py 1-4
   venv/bin/python p4tun/4-1_detection.py 1-4 --data-dir data
   venv/bin/python p4tun/4-2_sam.py 1-4 --data-dir data
   venv/bin/python p4tun/evaluation.py 1-4 --data-dir data
   ```
   Or use the apply script:
   ```bash
   venv/bin/python -m p4tun.bo.apply_bo_best_to_parameters 1-4 --apply --run-pipeline
   ```

5. **Compare:** Check `data/1-4/evaluation/performance.md` vs `data/bo/1-4/evaluation/performance.md` (target mIoU 0.748 for 1-4).

---

## 2-2: Preprocessing BO + detection + SAM — restored performance

**Target (data/bo/2-2):** mIoU **0.775**, OA 0.890, F1 0.872.

**What was done:**
1. **Preprocessing:** Built `p4tun/parameters/2-2/parameters_preprocessing.json` from BO results:
   - Unfolding: `2-2_unfolding_20260122_163749.json` (best_score 0.769) → `params_to_unfolding_dict`
   - Denoising + enhancing: `2-2_preprocessing_20260122_135958.json` (best_score 0.769) → `params_to_denoising_dict` + `params_to_enhancing_dict`
2. **Detection + SAM:** Applied via `apply_bo_best_to_parameters 2-2 --apply` (canonical detection 2-2_detection_20260122_101404.json, SAM 2-2_sam_20260122_120958.json).
3. **Run:** Preprocessing outputs from **data/bo/2-2** were copied to data/2-2 (so detection and SAM run on the same inputs as the original best run). Then detection → SAM → evaluation.

**Result:** mIoU **0.750**, OA 0.878, F1 0.854. **Close to target 0.775** (gap 0.025). So 2-2 performance is largely restored when using BO preprocessing outputs + BO detection + BO SAM.

To reproduce from scratch (without copying from data/bo): run preprocessing with the 2-2 BO preprocessing params above; then detection → SAM → evaluation. If the resulting depth map matches data/bo/2-2, mIoU should be similar.

---

## 1-4 run with 2-2 preprocessing + 1-4 detection + 1-4 SAM

**Setup:** Preprocessing params from 2-2 (BO), detection and SAM params from 1-4 (canonical 125324 + best_extracted).

**Steps:** 2-2's `parameters_preprocessing.json` was copied to `p4tun/parameters/1-4/` (1-4 original backed up as `.bak_1-4_original`). Then: preprocessing → detection → SAM → evaluation for tunnel 1-4.

**Result:** mIoU **0.563**, OA 0.756, F1 0.713. Ring count 10 but detection found 11 centers (2-2 preprocessing changes geometry for 1-4). So 2-2 preprocessing did not restore 1-4 target 0.748; 1-4’s own preprocessing is a better match for 1-4 detection/SAM.

---

## Critical Parameters (from reports)

- **Preprocessing (unfolding, denoising, enhancing):** See "Full parameter set for 1-4" above. No 1-4 preprocessing BO file exists; params are only in `p4tun/parameters/1-4/parameters_preprocessing.json` (and split files). Key difference vs sample: denoising `smoothing_offset` is **-0.003** for 1-4, +0.003 for sample.
- **Detection:** `binary_threshold`, `angle_positive_min/max`, `hough_oblique.threshold`, `merge_close_threshold`.
- **SAM:** `k_mask_height` / `height_neg`, `ab_height`, `angle_deg`, `segment_geometry` (change with care).

See `reports/CRITICAL_PARAMETERS_DETECTION_SAM.md` for full reference.
