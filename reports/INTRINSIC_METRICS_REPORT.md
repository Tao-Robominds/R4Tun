# Intrinsic Metrics: Design, Rationale, and Implementation

**Date:** February 2026  
**Scope:** Detection and SAM stages (bo4tun)  
**Purpose:** Document why these intrinsic metrics exist, how they were chosen, and how they support mIoU prediction without ground truth.

---

## Executive Summary

Intrinsic metrics are pipeline-output features computed **without ground truth**. They are used to train a surrogate model `mIoU = f(intrinsic_metrics)` so that at runtime we can predict mIoU from observable outputs. This enables Bayesian optimization and early stopping without running full evaluation. The 13 metrics were selected through **domain analysis** from optimization journey reports, **adaptation** to no-GT constraints, and **engineering judgment**—not formal statistical validation.

---

## 1. Purpose

| Goal | Description |
|------|-------------|
| **Surrogate model** | Train a regressor to predict mIoU from intrinsic metrics |
| **Runtime use** | At BO time: run detection + SAM, compute intrinsic metrics, predict mIoU |
| **No GT required** | All metrics derived from pipeline outputs (CSVs, depth maps) only |

**Training data:** `bo4tun/training/intrinsic_training_data.csv` (35 records as of Feb 2026)  
**Implementation:** `bo4tun/intrinsic_metrics.py`

---

## 2. Metric Inventory

### 2.1 Detection Stage Metrics (`det_*`)

| Metric | Description | Data Source | Rationale |
|--------|-------------|-------------|-----------|
| `det_k_count` | Number of detected K-block/ring positions | `detected.csv` row count | Wrong count → cascade failures. Expected: 6–10 per tunnel. |
| `det_k_count_match` | 1 if count = expected, else 0 | `detected.csv` + `ring_count.txt` or `TUNNEL_EXPECTED_RINGS` | Strong signal for detection quality. |
| `det_assume_default_ratio` | Fraction of detections from `assume` or `default` fallback | `Type` column in `detected.csv` | High ratio = detection failure; 0% ideal, &lt;20% acceptable (from reports). |
| `det_midpoint_ratio` | Fraction from `midpoint` detection type | `Type` column | Midpoint method most reliable; distinguishes detection strategies. |
| `det_real_detection_ratio` | Fraction from `midpoint`, `positive_slope`, `negative_slope` | `Type` column | Real vs fallback ratio; high = stronger detection. |
| `det_y_range` | Vertical span of Y positions (px) | `Y` column | Rings should span reasonable range; too narrow = clustered/spurious. |
| `det_y_std` | Std dev of Y positions | `Y` column | Proxy for K-block horizontal alignment; low = consistent. |
| `det_x_spacing_cv` | Coefficient of variation of X spacing | `X` column (sorted, diff) | Low CV = even spacing; high = irregular, missed/spurious rings. |

### 2.2 SAM Stage Metrics (`sam_*`)

| Metric | Description | Data Source | Rationale |
|--------|-------------|-------------|-----------|
| `sam_prompt_count` | Number of prompts (detected rings) fed to SAM | `detected.csv` row count | Should match `det_k_count`; too few → under-segmentation. |
| `sam_segment_count` | Unique non-background segments in output | `final.csv` `pred` column | Expected 6–7 per tunnel; too few = under-segmentation. |
| `sam_segment_count_match` | 1 if segment count ≥ expected−1, else 0 | `final.csv` + tunnel-specific expected | Simple quality proxy. |
| `sam_mask_fill_rate` | Fraction of pixels with non-background prediction | `final.csv` `pred` column | Low = under-segmentation or failed masks. |
| `sam_template_coverage` | Estimated fraction of depth map covered by SAM templates | `depth_map_outlier.npy` + prompt count | Low = large uncovered regions. |

---

## 3. How These Metrics Were Chosen

### 3.1 Domain Analysis from Optimization Reports

The following reports document recommended intrinsic metrics based on BO runs and root-cause analysis:

| Report | Key Metrics Documented |
|--------|------------------------|
| **P4TUN_OPTIMIZATION_JOURNEY_3-1_GT_K_BLOCK.md** | K-block count, assume/default ratio (0% ideal, &lt;20% acceptable), Hough line counts, K position vs GT |
| **P4TUN_OPTIMIZATION_JOURNEY_3-1_1-4.md** | K-block detection count, K-block horizontal alignment (Y std), detection method distribution (midpoint vs assume) |
| **P4TUN_CHAT_JOURNEY_REPORT.md** | Assume % (&lt;30% target), oblique line counts |
| **P4TUN_WRAPAROUND_BO_CHAT_REPORT.md** | Assume %, success counts (e.g. 54/63 processed) |
| **WRAPAROUND_ANALYSIS_AND_REMOVAL_JOURNEY.md** | K-block position variance, depth map consistency |

**Quote (GT_K_BLOCK):** *"Assume/default ratio: % of K positions from 'assume' or 'default' — 0% ideal; &lt;20% acceptable. High ratio = detection failure."*

### 3.2 Adaptation to No-GT Constraint

Reports also recommend GT-dependent metrics that **cannot** be used at runtime:

| Recommended (report) | Requires GT? | In `intrinsic_metrics.py`? |
|----------------------|--------------|----------------------------|
| Assume/default ratio | No | ✅ `det_assume_default_ratio` |
| K-block count | No | ✅ `det_k_count` |
| K position vs GT | Yes | ❌ |
| Per-class IoU | Yes | ❌ |
| X evenly spaced vs GT | Implicit | ✅ Proxy: `det_x_spacing_cv` |
| Y variance / alignment | No | ✅ `det_y_std`, `det_y_range` |

For SAM, reports list per-class IoU, K-block IoU, mIoU—all GT-dependent. The implementation uses proxies: `sam_mask_fill_rate`, `sam_segment_count`, `sam_template_coverage`.

### 3.3 Data Availability

Metrics are derived from **existing pipeline outputs**:

- **`detected.csv`** — Produced by detection stage. Contains `X`, `Y`, `Type`. `Type` values: `assume`, `default`, `midpoint`, `positive_slope`, `negative_slope`, `inferred`, etc.
- **`final.csv`** — Produced by SAM. Contains `pred` (segment ID per point).
- **`depth_map_outlier.npy`** — Used for `sam_template_coverage` (approximate).

No new instrumentation was added; metrics use what the pipeline already writes.

### 3.4 What Was NOT Done

- **No correlation analysis** — No script computes Pearson/Spearman between each metric and mIoU.
- **No feature selection** — No RFE, LASSO, or importance ranking.
- **No validation** — No ablation study or holdout validation of metric predictive power.

Selection is heuristic and report-driven, not statistically validated.

---

## 4. Expected Ring Count (No GT?)

`det_k_count_match` compares detected count to an "expected" value. That expected value comes from:

1. **`ring_count.txt`** — `data/<tunnel_id>/ring_count.txt` if present.
2. **`TUNNEL_EXPECTED_RINGS`** — Hardcoded fallback: `1-4`→10, `2-2`→10, `3-1`→6, `4-1`→10, `5-1`→7.

This is **structural metadata** (physical ring count per tunnel), not per-run ground truth. It comes from design specs or prior inspection and does not vary with each run.

---

## 5. Current Training Data Summary

| Statistic | Value |
|-----------|-------|
| Records | 35 |
| Intrinsic metrics | 13 |
| Target | mIoU |
| mIoU mean | 0.561 |
| mIoU std | 0.19 |
| mIoU range | [0.099, 0.768] |
| Stages | detection, sam, combined, complex_sam, sam_wraparound |

---

## 6. Recommendations

1. **Correlation analysis** — Compute correlation of each intrinsic metric with mIoU; drop or deprioritize weak predictors.
2. **Feature importance** — After training the predictor, report SHAP or permutation importance.
3. **Ablation** — Retrain with subsets of metrics to quantify contribution.
4. **Expand metrics** — Consider: Hough line counts, "Successfully processed X/Y" (from wraparound report), depth map row coverage.
5. **Document thresholds** — Explicit "good" ranges per metric (e.g. assume_default &lt; 0.2) for guardrails.

---

## 7. Related Files

| File | Role |
|------|------|
| `bo4tun/intrinsic_metrics.py` | Metric computation |
| `bo4tun/config_runner.py` | Runs configs, collects metrics |
| `bo4tun/build_training_data.py` | Builds `intrinsic_training_data.csv` |
| `p4tun/bo/predictor.py` | Trains and uses mIoU predictor |
| `bo4tun/training/intrinsic_training_metadata.json` | Column list, stats |
| `reports/P4TUN_OPTIMIZATION_JOURNEY_3-1_GT_K_BLOCK.md` | Primary source of recommended metrics |
