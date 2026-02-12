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

## 2. Metric Inventory (No Ground Truth)

All metrics are computable from pipeline outputs only; no labeled masks, per-pixel annotations, or expected counts required. **Excluded:** `det_k_count`, `det_k_count_match`, `sam_prompt_count`, `sam_segment_count`, `sam_segment_count_match` — counts cannot be verified without ground truth (we do not know if the number is correct).

### 2.1 Detection Stage Metrics (`det_*`)

| Metric | Description | Rationale |
|--------|-------------|-----------|
| `det_assume_default_ratio` | Fraction of detections from `assume` or `default` fallback | High ratio = detection failure; 0% ideal, &lt;20% acceptable (from reports). |
| `det_midpoint_ratio` | Fraction from `midpoint` detection type | Midpoint method most reliable; distinguishes detection strategies. |
| `det_real_detection_ratio` | Fraction from `midpoint`, `positive_slope`, `negative_slope` | Real vs fallback ratio; high = stronger detection. |
| `det_y_range` | Vertical span of Y positions (px) | Rings should span reasonable range; too narrow = clustered/spurious. |
| `det_y_std` | Std dev of Y positions | Proxy for K-block horizontal alignment; low = consistent. |
| `det_x_spacing_cv` | Coefficient of variation of X spacing | Low CV = even spacing; high = irregular, missed/spurious rings. |

### 2.2 SAM Stage Metrics (`sam_*`)

| Metric | Description | Rationale |
|--------|-------------|-----------|
| `sam_mask_fill_rate` | Fraction of pixels with non-background prediction | Low = under-segmentation or failed masks. r=−0.82 (simple). |
| `sam_template_coverage` | Estimated fraction of depth map covered by SAM templates | Low = large uncovered regions. |
| `complex_sam_ring_completeness` | Fraction of rings with >50% coverage | Cross-ring coverage consistency. r≈0.80 (complex). |
| `complex_sam_segment_height_cv` | CV of segment heights across segments | Height uniformity. r≈0.80 (complex). |
| `complex_sam_segment_width_cv` | CV of segment widths across segments | Width inconsistency. r≈−0.40 (complex). |

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

### 3.4 Initial Selection vs Later Validation

Initial metric selection was heuristic and report-driven. Subsequent ablation and correlation experiments (see Section 6) validated some metrics and identified gaps.

### 3.5 Methodology: From BO to Predictor

| Step | Action | When critical metrics are decided |
|------|--------|-----------------------------------|
| 1 | Use Bayesian optimization with mIoU ground truth to find optimal parameters | — |
| 2 | Identify critical **parameters** from optimization logs | — |
| 3 | Define intrinsic candidates from pipeline outputs, domain reports, and **correlation** with mIoU | Candidates identified |
| 4 | Run leave-one-out ablation to measure each metric’s contribution | — |
| **5** | **Decide on critical metrics** — select final set from correlation and ablation (keep HIGH/MEDIUM, drop LOW/redundant) | **Critical metrics chosen** |
| 6 | Train predictor (Ridge) to obtain weighted formula | — |
| 7 | Evaluate prediction accuracy (MAE, Spearman, LOO CV) | — |

---

## 4. Metrics Excluded (Not No-GT)

**Count-based metrics** (`det_k_count`, `det_k_count_match`, `sam_prompt_count`, `sam_segment_count`, `sam_segment_count_match`): We cannot verify whether a count is correct without ground truth. `det_k_count_match` and `sam_segment_count_match` also require knowing the expected value. The implementation uses `ring_count.txt` or hardcoded `TUNNEL_EXPECTED_RINGS` per tunnel — these are not general no-GT metrics.

---

## 5. Current Training Data Summary

| Statistic | Value |
|-----------|-------|
| Records | 35 |
| Intrinsic metrics (no-GT) | 8 |
| Intrinsic metrics (incl. count_match) | 13 |
| Target | mIoU |
| mIoU mean | 0.561 |
| mIoU std | 0.19 |
| mIoU range | [0.099, 0.768] |
| Stages | detection, sam, combined, complex_sam, sam_wraparound |

---

## 6. Ablation and Feature Importance Experiments

Findings from `bo4tun/report/NO_GT_BO_COMPREHENSIVE_REPORT.md`, `p4tun/bo/results/analysis/PREDICTOR_EVALUATION_REPORT.md`, and `bo4tun/report/MIOU_PREDICTOR_TRAINING.md`.

### 6.1 Correlation and Ablation (Combined Table)

**Correlation:** Spearman with mIoU (PREDICTOR_EVALUATION_REPORT, n=20). **Ablation:** Full LOO ablation (run_full_ablation.py, n=8 simple patterns) — ΔMAE and ΔR² when feature removed. Higher ΔMAE or more negative ΔR² → more important.

| Rank | Metric | Spearman | p-value | Correlation | ΔMAE | ΔR² | Ablation |
|------|--------|----------|---------|-------------|------|-----|----------|
| 1 | `det_midpoint_ratio` | +0.87 | 0.000 | Strong positive; single best predictor | +0.0031 | −0.22 | **Critical** |
| 2 | `sam_mask_fill_rate` | −0.82 | 0.001 | Strong negative; SAM quality signal | +0.0000 | −0.002 | Low (correlation strong; keep) |
| 3 | `det_real_detection_ratio` | +0.69 | 0.001 | Moderate positive | +0.0008 | −0.04 | Important |
| 4 | `det_x_spacing_cv` | +0.50 | 0.025 | Moderate positive | +0.0002 | −0.01 | KEEP |
| 5 | `det_y_std` | −0.45 | 0.044 | Moderate negative | −0.0012 | −0.05 | KEEP |

**Correlation conclusion:** All five main metrics correlate significantly with mIoU. `det_midpoint_ratio` and `sam_mask_fill_rate` are strongest. For complex patterns, SAM-stage metrics (`complex_sam_*`) dominate.

**Ablation conclusion:** `det_midpoint_ratio` is critical; `det_real_detection_ratio` is important; `sam_mask_fill_rate` adds MEDIUM value. Drop `det_assume_default_ratio` (redundant); consider dropping `det_k_count_match`. Recommended set: `det_midpoint_ratio`, `det_real_detection_ratio`, `det_x_spacing_cv`, `sam_mask_fill_rate`. Combined model achieves R²=0.72, Spearman=0.84.

### 6.2 Correlation Process (Method)

Spearman correlation measures how strongly each metric co-varies with mIoU (monotonic relationship, no model assumed).

1. **Compute:** For each intrinsic metric, compute Spearman rank correlation coefficient (r) and p-value with ground-truth mIoU across samples.
2. **Interpret |r|:** Higher |r| (closer to ±1) → stronger monotonic relationship; r > 0 → metric increases with mIoU; r < 0 → metric decreases with mIoU.
3. **Significance:** p < 0.05 → correlation is statistically significant; keep the metric. p ≥ 0.05 → weak or noisy; deprioritize.
4. **Rank:** Order metrics by |r|; top-ranked metrics are the best standalone predictors.

### 6.3 Ablation Process (Method)

Leave-one-out feature ablation measures how much each metric contributes to prediction given the others.

1. **Baseline:** Train a Ridge model with all candidate features; record MAE (or R²).
2. **Ablate:** For each feature, train a model with that feature removed; record the new MAE (or R²).
3. **ΔMAE / ΔR²:** Compute the change when the feature is removed. Higher ΔMAE (or more negative ΔR²) when a feature is removed → that feature is more important.
4. **Interpret:** Features with large positive ΔMAE are critical; those with near-zero or negative ΔMAE may be redundant or harmful.

### 6.4 Ridge Predictor Process (Method)

Train a Ridge model to obtain a weighted formula for predicting mIoU from intrinsic metrics.

1. **Define inputs and target:** Use intrinsic metrics (e.g. `det_midpoint_ratio`, `det_real_detection_ratio`, `det_x_spacing_cv`, `sam_mask_fill_rate`, `det_y_std`) as features; ground-truth mIoU as target.
2. **Fit Ridge:** Fit a linear model mIoU = b₀ + b₁·x₁ + b₂·x₂ + … with L2 regularization; minimize squared error plus penalty on large coefficients.
3. **Extract coefficients:** Each learned coefficient bᵢ is the weight for that feature. Positive → higher feature value predicts higher mIoU; negative → higher feature value predicts lower mIoU.
4. **Write formula:** The fitted model yields a weighted formula, e.g.:

   mIoU = 0.0299·det_midpoint_ratio + 0.0090·det_real_detection_ratio − 0.0078·det_x_spacing_cv + 0.0041·sam_mask_fill_rate + 0.0006·det_y_std + 0.4341

5. **Validate (MAE, R², Spearman):** Evaluate how well the formula predicts mIoU — in-sample and LOO CV. Lower MAE and higher R²/Spearman → formula works better.

**Simple patterns (1-4, 2-2, 3-1) — validation result:**

| Metric | In-sample | LOO CV |
|--------|-----------|--------|
| MAE | 0.041 (≈4.1% mIoU) | 0.067 (≈6.7% mIoU) |
| R² | 0.57 | −0.16 |
| Spearman | 0.50 | 0.12 |

In-sample MAE ≈ 4.1% mIoU indicates the formula fits the training data reasonably; LOO CV MAE ≈ 6.7% shows generalization error with small samples.

### 6.5 Model Comparison (Completed)

| Model | Samples | Spearman | Notes |
|-------|---------|----------|-------|
| Detection only | 10 | 0.873 | Best for simple patterns |
| Combined (det + sam) | 8 | 0.548 | Loses samples due to SAM nulls |
| SAM only | 8 | 0.524 | SAM metrics not predictive alone |
| det_* + sam_* combined | 20 | 0.84 | Best when both available |

**Complex patterns:** Detection intrinsic metrics are NOT predictive (Spearman≈0.59). Complex patterns use **SAM-stage intrinsic metrics** from `compute_complex_sam_metrics()` — `complex_sam_ring_completeness` (r≈0.80), `complex_sam_segment_height_cv` (r≈0.80), `complex_sam_segment_width_cv` (−0.40), `complex_sam_segment_area_cv`, `complex_sam_ring_coverage_cv`, `complex_sam_aspect_ratio_mean` (train_miou_predictors.py, COMPLEX_FEATURES). Alternatively, SAM **geometry parameters** (segment_width, k_height, ab_height) correlate strongly (r=−0.79 for segment_width) and are used by the complex predictor.

### 6.6 Ridge Predictor Validation Error Rates (LOO CV)

**Source:** `bo4tun/models/predictor_training_summary.json`

#### Simple Patterns (1-4, 2-2, 3-1)

| Metric | In-sample | LOO CV |
|--------|-----------|--------|
| MAE | 0.041 (≈4.1% mIoU) | 0.067 (≈6.7% mIoU) |
| R² | 0.57 | −0.16 |
| Spearman | 0.50 | 0.12 |
| Samples | 8 | 8 |
| Features | 5 | 5 |

#### Complex Patterns (4-1, 5-1)

| Metric | In-sample | LOO CV |
|--------|-----------|--------|
| MAE | 0.096 (≈9.6% mIoU) | 0.175 (≈17.5% mIoU) |
| R² | 0.55 | −0.59 |
| Spearman | 0.80 | −0.80 |
| Samples | 4 | 4 |
| Features | 6 | 6 |

**Interpretation:** LOO CV MAE and Spearman are worse than in-sample, indicating overfitting with small sample sizes. Negative LOO R² means the model generalizes worse than predicting the mean.

### 6.7 What Was NOT Done

- **SHAP / permutation importance** — Not implemented (Gap 2, NO_GT_BO_COMPREHENSIVE_REPORT).
- **Full systematic ablation** — Early runs compared detection-only vs SAM-only vs combined; leave-one-out per metric was added later. Need re-run with updated no-GT metric set (excluding count-based metrics).
- **Comprehensive ablation script** — `bo4tun/comprehensive_ablation.py` and `bo4tun/run_ablation_study.py` exist; `bo4tun/train_miou_predictors.py` references ablation. Results in `bo4tun/training/ablation_results.json` and `comprehensive_ablation_results.json`.

### 6.8 Documented Guardrail Thresholds

From NO_GT_BO_COMPREHENSIVE_REPORT:

```python
# Simple patterns
GUARDRAIL_THRESHOLDS = {
    'det_midpoint_ratio': {'min': 0.4},
    'det_real_detection_ratio': {'min': 0.5},
    'det_x_spacing_cv': {'max': 0.15},
    'sam_mask_fill_rate': {'max': 0.95},
}
# PREDICTOR_EVALUATION: det_midpoint_ratio < 0.5 fails; sam_mask_fill_rate > 0.9 fails
```

---

## 7. Recommendations

1. **Re-run ablation** with no-GT metric set (exclude count-based metrics); update `comprehensive_ablation.py` / `run_ablation_study.py` feature lists.
2. **Feature importance** — Add SHAP or permutation importance to ablation pipeline (Gap 2).
3. **Expand metrics** — Consider: Hough line counts, "Successfully processed X/Y" (from wraparound report), depth map row coverage.
4. **Document thresholds** — Integrate guardrail rules into predictor/BO pipeline; align with no-GT metrics only.

---

## 8. Related Files

| File | Role |
|------|------|
| `bo4tun/intrinsic_metrics.py` | Metric computation |
| `bo4tun/config_runner.py` | Runs configs, collects metrics |
| `bo4tun/build_training_data.py` | Builds `intrinsic_training_data.csv` |
| `bo4tun/run_ablation_study.py` | Leave-one-out ablation |
| `bo4tun/comprehensive_ablation.py` | Full ablation + correlations |
| `p4tun/bo/predictor.py` | Trains and uses mIoU predictor |
| `p4tun/bo/evaluate_predictor.py` | Ablation experiment |
| `bo4tun/training/ablation_results.json` | Ablation output |
| `bo4tun/report/NO_GT_BO_COMPREHENSIVE_REPORT.md` | Ablation results, gaps |
| `bo4tun/report/MIOU_PREDICTOR_TRAINING.md` | Feature selection, training |
| `p4tun/bo/results/analysis/PREDICTOR_EVALUATION_REPORT.md` | Correlation + ablation |
| `reports/P4TUN_OPTIMIZATION_JOURNEY_3-1_GT_K_BLOCK.md` | Primary source of recommended metrics |
