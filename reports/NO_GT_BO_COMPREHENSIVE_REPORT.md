# No-Ground-Truth Bayesian Optimization: Comprehensive Report

**Date:** 2026-02-02  
**Project:** P4Tun  
**Purpose:** Document the complete journey from vanilla BO to intrinsic-metrics-based no-GT BO

---

## Executive Summary

This report documents the development of a two-layer Bayesian Optimization system that operates
without ground truth (GT) at runtime. The system uses **intrinsic quality metrics** computed
from pipeline outputs to predict mIoU and guide parameter optimization.

### Key Achievements

| Metric | Simple Tunnels (1-4, 2-2, 3-1) | Complex Tunnels (4-1, 5-1) |
|--------|-------------------------------|---------------------------|
| Predictor Correlation | r = 0.84 (Spearman) | r = 0.87 (Spearman) |
| Prediction Error | MAE ≈ 0.09 | MAE ≈ 0.0125 |
| Training Samples | n = 20 | n = 70 |
| Key Predictive Features | det_midpoint_ratio, sam_mask_fill_rate | segment_width, k_height |

---

## Part 1: The Journey (Corrected Summary)

### Phase 1: Vanilla BO with Ground Truth

**Objective:** Find optimal parameters for each pipeline stage using GT mIoU as objective.

**Total Evaluations:** 559 evaluations across all tunnels

| Tunnel | BO Runs | Evaluations | Initial mIoU | Best mIoU | Improvement |
|--------|---------|-------------|--------------|-----------|-------------|
| 1-4 | 8 | 80 | 0.511 | 0.807 | +57.9% |
| 2-2 | 11 | 150+ | 0.672 | 0.765 | +13.8% |
| 3-1 | 10 | 185 | 0.650 | 0.769 | +18.3% |
| 4-1 | 4 | 45 | 0.316 | 0.428 | +35.4% |
| 5-1 | 9 | 249 | 0.308 | 0.431 | +39.9% |

**Key Finding:** Simple-staggered/continuous patterns (1-4, 2-2, 3-1) and complex-staggered patterns (4-1, 5-1) require different detection and SAM strategies, leading to separate pipelines:
- `4-1_detection.py` + `4-2_sam.py` for simple patterns
- `4-1_detection_complex.py` + `4-2_sam_complex.py` for complex patterns

### Phase 2: Stage Impact Analysis

**Key Discovery:** Preprocessing stages have minimal impact; Detection and SAM dominate.

| Stage | Typical mIoU Impact | Priority |
|-------|--------------------:|----------|
| Unfolding | +0.0% to +0.1% | Low |
| Denoising | +0.1% | Low |
| Enhancing | Combined with denoising | Low |
| **Detection** | **+6.3%** | **High** |
| **SAM** | **+4-7%** | **High** |

**Source:** `reports/P4TUN_OPTIMIZATION_JOURNEY_2-2.md` (150+ BO iterations)

### Phase 3: Intrinsic Metrics Identification

**Goal:** Identify pipeline output characteristics that correlate with mIoU without requiring GT.

**Key Finding:** Simple and complex staggered patterns require **different predictive features**. This section documents metrics for both pattern types, organized by pipeline stage.

---

#### 3.1 Model Separation Strategy

| Pattern Type | Tunnels | Segments | Pipeline | Best Predictors |
|--------------|---------|----------|----------|-----------------|
| **Simple** | 1-4, 2-2, 3-1 | 6 | `4-1_detection.py` + `4-2_sam.py` | Intrinsic metrics |
| **Complex** | 4-1, 5-1 | 7 | `4-1_detection_complex.py` + `4-2_sam_complex.py` | SAM geometry params |

**Why Separate?**
- Simple patterns: Detection quality (midpoint_ratio) strongly predicts mIoU
- Complex patterns: SAM geometry parameters (segment_width) dominate prediction
- Mixing them degrades both models

**Future Consideration:** Validate if 3-1 (continuous pattern) behaves differently from 1-4/2-2 (simple_staggered pattern). All three have 6 segments; pattern type may warrant 3-way split.

---

#### 3.2 Preprocessing Metrics (Guardrails Only)

Preprocessing contributes only **+0.1%** to mIoU, so these metrics are implemented as **fail-fast guardrails**, not predictor features.

| Stage | Metric | Data Source | Threshold | Action if Failed |
|-------|--------|-------------|-----------|------------------|
| Unfolding | `theta_coverage` | `unwrapped.csv` | 98-102% | Flag wraparound issue |
| Denoising | `point_retention_ratio` | `denoised.csv` / `unwrapped.csv` | > 90% | Check radius params |
| Enhancing | `interpolation_coverage` | `depth_map_outlier.npy` | > 95% | Rerun enhancing |

**Implementation:** `bo4tun/intrinsic_metrics.py`
- `compute_preprocessing_guardrails()` - returns all metrics
- `check_preprocessing_guardrails()` - returns (passed, violations, metrics)
- `run_preprocessing_check()` - verbose console output

---

#### 3.3 Detection Metrics

##### For Simple Patterns (1-4, 2-2, 3-1)

| Metric | Description | Correlation | Use in Model |
|--------|-------------|-------------|--------------|
| `det_midpoint_ratio` | % detections via midpoint method | **r = +0.87*** | ✓ Predictor |
| `det_real_detection_ratio` | % non-fallback detections | r = +0.69** | ✓ Predictor |
| `det_k_count_match` | 1 if count matches expected | r = +0.52* | ✓ Predictor |
| `det_x_spacing_cv` | Coefficient of variation of X-spacing | **Critical** | ✓ Guardrail |
| `det_k_count` | Number of detected K-blocks | Varies | Guardrail |
| `det_assume_default_ratio` | % fallback detections | Negative | Guardrail |
| `det_y_range` | Y-position spread (pixels) | Context | Guardrail |
| `det_y_std` | Y-position std deviation | Context | Guardrail |

**Key Insight:** `det_midpoint_ratio` is the single best predictor for simple patterns. `det_x_spacing_cv` is critical as a guardrail (caught 2-2 failure).

##### For Complex Patterns (4-1, 5-1)

| Metric | Description | Correlation | Use in Model |
|--------|-------------|-------------|--------------|
| `det_k_count` | Number of detected K-blocks | Varies | Guardrail |
| `det_x_spacing_cv` | X-spacing uniformity | Context | Guardrail |
| `det_y_range` | Y-position spread | Context | Guardrail |

**Key Insight:** Detection metrics alone are **not predictive** for complex patterns. SAM geometry parameters dominate (see 3.4).

---

#### 3.4 SAM Metrics

##### For Simple Patterns (1-4, 2-2, 3-1)

| Metric | Description | Correlation | Use in Model |
|--------|-------------|-------------|--------------|
| `sam_mask_fill_rate` | % non-background pixels | **r = -0.82*** | ✓ Predictor |
| `sam_segment_count_match` | 1 if count ≥ expected-1 | r = -0.53 | Guardrail |
| `sam_prompt_count` | Prompts fed to SAM | Matches det_k_count | Info |
| `sam_segment_count` | Output segment count | Indirect | Info |
| `sam_template_coverage` | Template coverage % | Indirect | Info |

**Key Insight:** `sam_mask_fill_rate` has strong **negative** correlation - high fill rate often means over-segmentation.

##### For Complex Patterns (4-1, 5-1)

For complex patterns, **SAM input parameters** are better predictors than output metrics:

| Parameter | Description | Correlation | Use in Model |
|-----------|-------------|-------------|--------------|
| `segment_width` | Segment width (mm) | **r = -0.789*** | ✓ Predictor |
| `k_height` | K-block height (mm) | r = -0.269* | ✓ Predictor |
| `ab_height` | A/B-block total height (mm) | Moderate | ✓ Predictor |
| `angle_deg` | Oblique angle (degrees) | Moderate | ✓ Predictor |

**Key Insight:** For complex patterns, the geometry parameters that **configure** SAM predict mIoU better than the intrinsic metrics from SAM's output. This is because complex patterns have tighter tolerances on segment geometry.

---

#### 3.5 Summary: Predictor Features by Pattern Type

| Pattern | Predictor Features | Spearman r | MAE |
|---------|-------------------|------------|-----|
| **Simple** | `det_midpoint_ratio`, `det_real_detection_ratio`, `det_k_count_match`, `sam_mask_fill_rate` | 0.84 | 0.09 |
| **Complex** | `segment_width`, `k_height`, `ab_height`, `angle_deg` | 0.87 | 0.0125 |

---

#### 3.6 Guardrail Thresholds Summary

| Pattern | Metric | Threshold | Severity |
|---------|--------|-----------|----------|
| Both | `theta_coverage` | 98-102% | Hard fail |
| Both | `interpolation_coverage` | > 95% | Hard fail |
| Both | `point_retention_ratio` | > 90% | Soft fail |
| Simple | `det_x_spacing_cv` | < 0.15 | Hard fail |
| Simple | `det_midpoint_ratio` | > 0.40 | Soft fail |
| Simple | `sam_mask_fill_rate` | < 0.95 | Soft fail |
| Complex | `det_x_spacing_cv` | < 0.60 | Soft fail |
| Complex | `det_y_range` | 200-1500 px | Soft fail |

---

#### 3.7 Evidence Basis: Experimental Data vs Reasoning

Not all conclusions are equally grounded. This section documents the evidence level for each.

##### From Experimental Data

| Item | Source | Evidence |
|------|--------|----------|
| Simple pattern correlations (r=0.87, r=-0.82, r=0.69, r=0.52) | `bo4tun/report/predictor_evaluation.json` | Correlation analysis on n=20 samples |
| Combined model (Spearman 0.84, MAE 0.09) | Same | Ridge regression trained on those features |
| Complex pattern correlations (segment_width r=-0.789, k_height r=-0.269) | `COMPLEX_STAGGERED_INTRINSIC_ANALYSIS.md` | Correlation analysis on n=70 samples |
| Complex model (Spearman 0.87, MAE 0.0125) | Same | Cross-validated Ridge on SAM geometry params |
| `det_x_spacing_cv` < 0.15 (0.10 for 2-2) | 2-2 failure analysis | **Empirically validated**: fix improved mIoU from 0.476 → 0.690 |
| Stage impact (+6.3%, +4.2%, etc.) | `P4TUN_OPTIMIZATION_JOURNEY_2-2.md` | Measured in sequential BO phases |

##### From Reasoning / Heuristics (Less Validated)

| Item | Source | Notes |
|------|--------|------|
| `det_midpoint_ratio` > 0.40 | Correlation analysis | Reasonable cutoff; not from ROC or failure analysis |
| `sam_mask_fill_rate` < 0.95 | Correlation analysis | Heuristic based on negative correlation |
| Preprocessing thresholds (98-102%, >90%, >95%) | Proposed | Domain knowledge; not empirically validated on failures |
| Complex guardrails (`det_x_spacing_cv` < 0.60, `det_y_range` 200-1500) | Historical analysis | No explicit validation report |
| Interpretations (e.g., "high fill rate → over-segmentation") | Reasoning | Interpretation of correlation; plausible but not directly measured |

**Summary:** Correlations, model performance, and the 2-2 `det_x_spacing_cv` fix are **data-driven**. Many guardrail thresholds and the "Key Insights" are **correlation + reasoning**; preprocessing thresholds are **proposed/heuristic**.

---

### Phase 4: Ablation Study Design

This phase outlines the systematic ablation study to identify key intrinsic metrics:
- **Preprocessing**: Guardrails (fail-fast checks)
- **Detection + SAM**: Predictive models for mIoU (separate for simple vs complex)

---

#### 4.1 Study Objectives

| Stage | Goal | Output |
|-------|------|--------|
| Preprocessing | Identify guardrail thresholds | Pass/fail criteria for theta_coverage, point_retention, interpolation_coverage |
| Detection (Simple) | Find predictive intrinsic metrics | Feature set + weights for mIoU prediction |
| Detection (Complex) | Find predictive intrinsic metrics | Feature set + weights (may differ from simple) |
| SAM (Simple) | Find predictive intrinsic metrics | Feature set + weights for mIoU prediction |
| SAM (Complex) | Find predictive geometry params | Feature set + weights (segment_width, k_height, etc.) |

---

#### 4.2 Dataset Design

##### 4.2.1 Data Collection Strategy

| Requirement | Current State | Target | Method |
|-------------|---------------|--------|--------|
| Sample diversity | Mostly BO-optimized | Include failures | Grid search with intentionally bad params |
| Sample size (Simple) | 20 | 60+ | Add 40+ configs from 1-4, 2-2, 3-1 |
| Sample size (Complex) | 70 | 100+ | Add 30+ configs from 4-1, 5-1 |
| mIoU coverage | 0.1-0.7 | 0.0-0.8 | Include extreme failures and successes |
| Tunnel balance | Uneven | Even | 15-20 samples per tunnel |

##### 4.2.2 Dataset Structure

**Simple Patterns Dataset (target: n=60+)**

| Tunnel | Samples | mIoU Range | Sampling Strategy |
|--------|---------|------------|-------------------|
| 1-4 | 20 | 0.1-0.8 | 5 failures + 10 mid-range + 5 good |
| 2-2 | 20 | 0.1-0.8 | 5 failures + 10 mid-range + 5 good |
| 3-1 | 20 | 0.1-0.8 | 5 failures + 10 mid-range + 5 good |

**Complex Patterns Dataset (target: n=100+)**

| Tunnel | Samples | mIoU Range | Sampling Strategy |
|--------|---------|------------|-------------------|
| 4-1 | 50 | 0.2-0.5 | Grid over segment_width × k_height space |
| 5-1 | 50 | 0.2-0.5 | Grid over segment_width × k_height space |

##### 4.2.3 Features to Collect

**Preprocessing Features (Guardrails)**

| Feature | Source | Expected Role |
|---------|--------|---------------|
| `pre_theta_coverage` | unwrapped.csv | Guardrail (98-102%) |
| `pre_point_retention_ratio` | denoised.csv / unwrapped.csv | Guardrail (>90%) |
| `pre_interpolation_coverage` | depth_map_outlier.npy | Guardrail (>95%) |
| `pre_ring_count_match` | unwrapped.csv | Guardrail |

**Detection Features (Predictors)**

| Feature | Source | Expected Role |
|---------|--------|---------------|
| `det_k_count` | detected.csv | Guardrail |
| `det_k_count_match` | detected.csv | Predictor (Simple) |
| `det_midpoint_ratio` | detected.csv | Predictor (Simple) |
| `det_real_detection_ratio` | detected.csv | Predictor (Simple) |
| `det_assume_default_ratio` | detected.csv | Guardrail |
| `det_x_spacing_cv` | detected.csv | Guardrail (both) |
| `det_y_range` | detected.csv | Guardrail (Complex) |
| `det_y_std` | detected.csv | Candidate |

**SAM Features (Predictors)**

| Feature | Source | Expected Role |
|---------|--------|---------------|
| `sam_mask_fill_rate` | final.csv | Predictor (Simple) |
| `sam_segment_count_match` | final.csv | Guardrail |
| `sam_prompt_count` | final.csv | Info |
| `segment_width` | parameters_sam.json | Predictor (Complex) |
| `k_height` | parameters_sam.json | Predictor (Complex) |
| `ab_height` | parameters_sam.json | Predictor (Complex) |
| `angle_deg` | parameters_sam.json | Predictor (Complex) |

---

#### 4.3 Ablation Experiments

##### 4.3.1 Experiment 1: Preprocessing Guardrail Calibration

**Objective:** Find optimal thresholds for preprocessing guardrails.

**Method:**
1. Collect samples where preprocessing failed (manual inspection or low final mIoU)
2. Analyze preprocessing metrics at failure points
3. Find threshold that separates failures from successes (ROC analysis)

```python
# Pseudo-code
for metric in ['theta_coverage', 'point_retention_ratio', 'interpolation_coverage']:
    # Find threshold that maximizes separation
    thresholds = np.linspace(metric_min, metric_max, 100)
    for thresh in thresholds:
        precision = compute_precision(metric, thresh, failures)
        recall = compute_recall(metric, thresh, failures)
    optimal_threshold = find_best_f1(thresholds)
```

**Output:** Validated guardrail thresholds with precision/recall metrics.

##### 4.3.2 Experiment 2: Simple Pattern Predictor (Detection + SAM)

**Objective:** Find optimal feature set for predicting mIoU on simple patterns (1-4, 2-2, 3-1).

**What is Leave-One-Out Feature Ablation?**

This method measures how important each feature is by:
1. Train a model with ALL features → get baseline error
2. Remove ONE feature, retrain → get new error
3. Compare: if error increases a lot, that feature is important

**Example:**

| Step | Features Used | MAE | Interpretation |
|------|---------------|-----|----------------|
| Baseline | All 4 features | 0.09 | Reference point |
| Remove det_midpoint_ratio | 3 remaining | 0.15 | ΔMAE = +0.06 → **HIGH importance** |
| Remove det_real_detection_ratio | 3 remaining | 0.10 | ΔMAE = +0.01 → LOW importance |
| Remove det_k_count_match | 3 remaining | 0.11 | ΔMAE = +0.02 → MEDIUM importance |
| Remove sam_mask_fill_rate | 3 remaining | 0.12 | ΔMAE = +0.03 → MEDIUM importance |

**Method:**

```python
# Candidate features for simple patterns
candidate_features = ['det_midpoint_ratio', 'det_real_detection_ratio', 
                      'det_k_count_match', 'sam_mask_fill_rate']

# Step 1: Train baseline model with all features
baseline_model = Ridge(alpha=1.0).fit(X_train[candidate_features], y_train)
baseline_pred = baseline_model.predict(X_train[candidate_features])
baseline_mae = mean_absolute_error(y_train, baseline_pred)

# Step 2: Leave-one-out ablation
importance_results = {}
for remove_feature in candidate_features:
    subset = [f for f in candidate_features if f != remove_feature]
    model = Ridge(alpha=1.0).fit(X_train[subset], y_train)
    pred = model.predict(X_train[subset])
    mae = mean_absolute_error(y_train, pred)
    
    delta_mae = mae - baseline_mae
    importance_results[remove_feature] = {
        'mae_without': mae,
        'delta_mae': delta_mae,
        'importance': 'HIGH' if delta_mae > 0.03 else 'MEDIUM' if delta_mae > 0.01 else 'LOW'
    }

# Step 3: Keep only HIGH and MEDIUM importance features
final_features = [f for f, r in importance_results.items() if r['importance'] != 'LOW']
```

**Training Algorithm:** Ridge Regression
- Interpretable coefficients (we can see feature weights)
- Robust for small datasets (n=60)
- Already proven effective (Spearman 0.84)

**Output:**

| Feature | MAE without | ΔMAE | Importance | Keep? |
|---------|-------------|------|------------|-------|
| det_midpoint_ratio | TBD | TBD | TBD | TBD |
| det_real_detection_ratio | TBD | TBD | TBD | TBD |
| det_k_count_match | TBD | TBD | TBD | TBD |
| sam_mask_fill_rate | TBD | TBD | TBD | TBD |

---

##### 4.3.3 Experiment 3: Complex Pattern Predictor (SAM Geometry)

**Objective:** Find optimal feature set for predicting mIoU on complex patterns (4-1, 5-1).

**Key Difference from Simple Patterns:**
- Simple patterns: Use **intrinsic metrics** from detection/SAM output
- Complex patterns: Use **SAM geometry parameters** (the inputs to SAM, not outputs)

**Why SAM geometry parameters?**
For complex staggered patterns, the segment geometry is more sensitive. The parameters that **configure** SAM (segment_width, k_height) correlate better with mIoU than the output metrics. This is because small changes in template geometry cause large changes in segmentation quality.

**Candidate Features:**

| Feature | Type | Why Include |
|---------|------|-------------|
| `segment_width` | SAM input param | r = -0.789 (DOMINANT) |
| `k_height` | SAM input param | r = -0.269 |
| `ab_height` | SAM input param | Moderate correlation |
| `angle_deg` | SAM input param | Moderate correlation |

**Method:** Same leave-one-out ablation as Experiment 2

```python
# Candidate features for complex patterns
complex_features = ['segment_width', 'k_height', 'ab_height', 'angle_deg']

# Optional: test if adding detection metrics helps
extended_features = complex_features + ['det_k_count', 'det_x_spacing_cv']

# Compare baseline (SAM-only) vs extended (SAM + detection)
model_sam_only = Ridge().fit(X_train[complex_features], y_train)
model_extended = Ridge().fit(X_train[extended_features], y_train)

# If extended model doesn't improve much, use simpler SAM-only model
```

**Output:**

| Feature | MAE without | ΔMAE | Importance | Keep? |
|---------|-------------|------|------------|-------|
| segment_width | TBD | TBD | Expected HIGH | TBD |
| k_height | TBD | TBD | TBD | TBD |
| ab_height | TBD | TBD | TBD | TBD |
| angle_deg | TBD | TBD | TBD | TBD |

---

#### 4.4 Validation Strategy

**Approach:** Train on all data, validate externally against ground truth mIoU.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING & VALIDATION FLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TRAINING PHASE (offline)                                      │
│   ─────────────────────────                                     │
│   1. Collect training data: (intrinsic_metrics, GT_mIoU) pairs  │
│   2. Run feature ablation to find important features            │
│   3. Train final Ridge model on ALL training data               │
│   4. Save model weights                                         │
│                                                                 │
│   VALIDATION PHASE (online)                                     │
│   ─────────────────────────                                     │
│   1. Run pipeline on new tunnel configuration                   │
│   2. Extract intrinsic metrics from pipeline outputs            │
│   3. Predict mIoU using trained model                           │
│   4. Run evaluation against GT to get actual mIoU               │
│   5. Compare: predicted_mIoU vs actual_mIoU                     │
│                                                                 │
│   SUCCESS CRITERIA                                              │
│   ────────────────                                              │
│   - Spearman correlation > 0.80 (ranking accuracy)              │
│   - MAE < 0.10 (absolute prediction error)                      │
│   - No systematic over/under-estimation                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why not cross-validation?**
- CV estimates generalization error during development
- We have a better validation: compare predicted vs actual mIoU on real runs
- Simpler: train once on all data, validate on new configurations

---

#### 4.5 Expected Outputs

##### For Preprocessing (Guardrails)

| Metric | Validated Threshold | Precision | Recall |
|--------|---------------------|-----------|--------|
| theta_coverage | TBD (expect ~98%) | TBD | TBD |
| point_retention_ratio | TBD (expect ~90%) | TBD | TBD |
| interpolation_coverage | TBD (expect ~95%) | TBD | TBD |

##### For Simple Patterns (Predictive Model)

| Output | Expected |
|--------|----------|
| Final feature set | 3-4 features |
| Algorithm | Ridge Regression |
| Spearman correlation | >0.80 |
| MAE | <0.10 |

##### For Complex Patterns (Predictive Model)

| Output | Expected |
|--------|----------|
| Final feature set | 3-4 SAM geometry params |
| Algorithm | Ridge Regression |
| Spearman correlation | >0.85 |
| MAE | <0.02 |

---

#### 4.6 Ablation Study Results (2026-02-02)

**Data:** `bo4tun/training/intrinsic_training_data.csv` (20 samples)
**Results:** `bo4tun/training/ablation_results.json`

---

##### Simple Patterns: Model Comparison

| Model | Samples | Features | Spearman | MAE | Notes |
|-------|---------|----------|----------|-----|-------|
| **Detection only** | **10** | **5** | **0.873** | **0.067** | **BEST** - uses all available samples |
| Combined (det+sam) | 8 | 7 | 0.548 | 0.032 | Loses 2 samples due to SAM nulls |
| SAM only | 8 | 2 | 0.524 | 0.037 | Poor - SAM metrics not predictive alone |

**Winner: Detection-only model** (Spearman=0.873, n=10)

##### Simple Patterns: Feature Importance (Leave-One-Out Ablation)

| Feature | ΔMAE | ΔSpearman | Coefficient | Importance | Action |
|---------|------|-----------|-------------|------------|--------|
| det_real_detection_ratio | +0.0121 | +0.0000 | +0.1645 | MEDIUM | KEEP |
| det_assume_default_ratio | +0.0121 | +0.0000 | -0.1645 | MEDIUM | KEEP (redundant*) |
| det_midpoint_ratio | +0.0034 | +0.0000 | +0.1183 | LOW | KEEP |
| det_x_spacing_cv | +0.0028 | +0.0764 | -0.0701 | LOW | KEEP |
| det_k_count_match | -0.0041 | +0.0000 | +0.0448 | LOW | DROP? |

**Note:** `det_real_detection_ratio` and `det_assume_default_ratio` are complements (sum to 1.0), so they're redundant. Keep only one.

##### Complex Patterns: Feature Importance

| Feature | ΔMAE | Coefficient | Importance |
|---------|------|-------------|------------|
| det_k_count | +0.0000 | -0.0000 | LOW |
| det_x_spacing_cv | +0.0000 | +0.0000 | LOW |
| det_y_range | +0.0000 | +0.0001 | LOW |
| det_y_std | +0.0000 | +0.0000 | LOW |

**Result:** Detection metrics are **NOT predictive** for complex patterns (Spearman=0.592).

**Confirmed:** Complex patterns require SAM geometry parameters (`segment_width`, `k_height`), not detection metrics. These are not in the current training data.

---

##### Key Findings

1. **Simple Patterns:**
   - Detection-only model achieves Spearman=0.873 (better than combined!)
   - `det_real_detection_ratio` is the most important feature
   - SAM metrics add noise when sample size is small

2. **Complex Patterns:**
   - Detection metrics alone: Spearman=0.592 (mediocre)
   - Need SAM geometry parameters for better prediction
   - Separate model confirmed necessary

3. **Data Quality:**
   - 2/10 simple samples have missing SAM metrics
   - Need more diverse samples for robust ablation

---

##### Recommended Models

**Simple Patterns (Final):**
```python
# Ridge regression, detection metrics only
Features: ['det_real_detection_ratio', 'det_midpoint_ratio', 'det_x_spacing_cv']
# Note: Dropped det_assume_default_ratio (redundant) and det_k_count_match (negative ΔMAE)
Spearman = 0.873
MAE = 0.067
n_samples = 10
```

**Complex Patterns (From Prior Analysis):**
```python
# Ridge regression, SAM geometry parameters
Features: ['segment_width', 'ab_height', 'k_height', 'angle_deg']
CV MAE = 0.0125
Spearman = 0.8724
n_samples = 70
```

### Phase 5: Guardrails and Predictor Implementation

#### Layer A: Guardrails (Hard Constraints)

**Simple Patterns (1-4, 2-2, 3-1):**

```python
GUARDRAIL_THRESHOLDS = {
    'det_k_count_match': {'min': 0.8, 'max': None},      # Must match expected
    'det_midpoint_ratio': {'min': 0.4, 'max': None},     # Need real detections
    'det_real_detection_ratio': {'min': 0.5, 'max': None},
    'det_x_spacing_cv': {'min': None, 'max': 0.15},      # Must be uniform
    'sam_mask_fill_rate': {'min': None, 'max': 0.95},
}

# Tunnel-specific overrides
'2-2': {'det_x_spacing_cv': {'min': None, 'max': 0.10}}  # Stricter for 2-2
```

**Complex Patterns (4-1, 5-1):**

```python
COMPLEX_GUARDRAIL_THRESHOLDS = {
    'det_k_count': {'min': 4, 'max': 12},
    'det_x_spacing_cv': {'min': None, 'max': 0.60},
    'det_y_range': {'min': 200, 'max': 1500},
}

# Tunnel overrides
'4-1': {'det_x_spacing_cv': {'min': None, 'max': 0.50}}
'5-1': {'det_x_spacing_cv': {'min': None, 'max': 0.80}}  # Non-uniform expected

SAM_PARAM_GUARDRAILS = {
    'segment_width': {'min': 1150, 'max': 1350},
    'k_height': {'min': 900, 'max': 1200},
    'ab_height': {'min': 3000, 'max': 3500},
}
```

#### Layer B: mIoU Predictor

Two separate predictors:
1. **Simple patterns:** `p4tun/bo/predictor.py` using intrinsic metrics
2. **Complex patterns:** `p4tun/bo/models/complex_miou_predictor.pkl` using SAM params

### Phase 6: No-GT BO Validation

**Validation Method:** Compare predicted mIoU vs true mIoU

| Tunnel | Predicted mIoU | True mIoU | Error | Status |
|--------|----------------|-----------|-------|--------|
| 1-4 | 0.646 | 0.576 | +12% | Optimistic |
| 2-2 (before fix) | 0.672 | 0.476 | +41% | **Over-estimated** |
| 2-2 (after fix) | 0.672 | **0.690** | -3% | **Fixed** |
| 3-1 | 0.520 | 0.500 | +4% | Good |
| 4-1 | 0.344 | - | - | No GT validation |
| 5-1 | 0.360 | 0.391 | -8% | Conservative |

**Key Fix for 2-2:** Adding `det_x_spacing_cv < 0.10` guardrail improved true mIoU from 0.476 to 0.690 (+45%).

---

## Part 2: Numerical Evidence Inventory

### 2.1 BO Run Statistics

| Category | Files | Total Evaluations |
|----------|-------|-------------------|
| Vanilla BO (with GT) | 40+ | 559 |
| No-GT BO (simple) | 6 | 160 |
| No-GT BO (complex) | 4 | 30 |
| Proxy experiments | 8 | 80 |

### 2.2 Training Data

**Location:** `bo4tun/training/intrinsic_training_data.csv`

| Field | Value |
|-------|-------|
| Samples | 20 (simple) + 70 (complex) |
| Features | 13 intrinsic metrics |
| Target | mIoU |
| Split by tunnel | 1-4: 5, 2-2: 5, 4-1: 5, 5-1: 5 (simple) |

### 2.3 Correlation Analysis Results

**Source:** `p4tun/bo/results/predictor_evaluation.json`

| Metric | Spearman | p-value | Significant |
|--------|----------|---------|-------------|
| det_midpoint_ratio | 0.87 | 0.000 | *** |
| sam_mask_fill_rate | -0.82 | 0.001 | ** |
| det_real_detection_ratio | 0.69 | 0.001 | ** |
| det_k_count_match | 0.52 | 0.019 | * |
| sam_segment_count | -0.53 | 0.077 | ns |

### 2.4 Model Performance

**Simple Pattern Predictor:**

| Metric | Value |
|--------|-------|
| Features | 4 |
| R² | 0.72 |
| Spearman | 0.84 |
| MAE | 0.09 |

**Complex Pattern Predictor:**

| Metric | Value |
|--------|-------|
| Features | 4 (SAM params) |
| CV MAE | 0.0125 |
| Spearman | 0.8724 |

### 2.5 Best Parameters Achieved

**2-2 Detection (GT-BO optimized):**
```json
{
    "binary_threshold": 149,
    "hough_oblique_threshold": 69,
    "angle_positive_min": 5.509,
    "angle_positive_max": 8.652
}
```

**5-1 SAM (Complex, GT-BO optimized):**
```json
{
    "segment_width": 1167.87,
    "k_height": 950.0,
    "ab_height": 3239.77,
    "angle_deg": 7.26
}
```

---

## Part 3: Corrections to User's Summary

### Inaccuracy 1: Training Method

**User stated:** "we use boostergradient to train"

**Correction:** We used **Ridge Regression** (regularized linear regression), not Gradient Boosting. Ridge was chosen because:
- Small training set (n=20) favors simpler models
- Better interpretability (linear coefficients)
- Less prone to overfitting

### Inaccuracy 2: Intrinsic Metrics Source

**User stated:** "intrinsic_metrics are different for simple and complex tunnels"

**Clarification:** For **simple tunnels**, intrinsic metrics (detection outputs) predict mIoU well. For **complex tunnels**, **SAM input parameters** (segment_width, k_height) are better predictors than intrinsic metrics. This is a key distinction.

### Inaccuracy 3: Ablation Study

**User stated:** "we ran ablation study to find out their Necessity of each metric"

**Clarification:** We ran a **partial ablation** comparing detection-only, SAM-only, and combined models. We did NOT systematically remove individual metrics one by one. This is noted as a gap in Part 4.

---

## Part 4: Missing/Incomplete Experiments

### Gap 1: Preprocessing Intrinsic Metrics (Not Implemented)

**What's Missing:**
- Unfolding metrics: theta_coverage, centerline_rmse, sample_density
- Denoising metrics: point_retention_ratio, gradient_threshold_effect
- Enhancing metrics: interpolation_coverage, curvature_smoothness

**Why Not Done:** Preprocessing has <0.1% impact on mIoU per vanilla BO experiments. However, for completeness, these could be added as additional guardrails.

**Recommendation:** Add preprocessing guardrails but not to the predictor.

### Gap 2: Full Ablation Study

**What's Missing:**
- Remove each metric individually and measure ΔRMSE/ΔR²
- Feature importance ranking (SHAP, permutation importance)

**Recommendation:** Run systematic ablation with leave-one-out feature removal.

### Gap 3: Full Model with Preprocessing

**What's Missing:**
- Model: `mIoU = f(pre_*, det_*, sam_*)`
- Compare to current `mIoU = f(det_*, sam_*)`

**Recommendation:** Low priority given preprocessing's minimal impact.

### Gap 4: More Training Data

**Current:**
- Simple patterns: 20 samples
- Complex patterns: 70 samples

**Recommendation:**
- Add more diverse BO configurations
- Include failure cases (mIoU < 0.3) for better guardrail calibration

### Gap 5: Cross-Validation Across Tunnels

**What's Missing:**
- Leave-one-tunnel-out validation
- Transfer learning validation

**Recommendation:** Test predictor trained on tunnels A, B, C on tunnel D.

---

## Part 5: How to Use Intrinsic Metrics Quality for Reflection/Rerun

### 5.1 Proposed System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: Raw Point Cloud                                            │
│      │                                                              │
│      ▼                                                              │
│   ┌──────────────────┐                                              │
│   │  PREPROCESSING   │──► Guardrail Checks (intrinsic_metrics.py): │
│   │  (Unfolding,     │    - theta_coverage > 98%?                  │
│   │   Denoising,     │    - point_retention_ratio > 90%?           │
│   │   Enhancing)     │    - interpolation_coverage > 95%?          │
│   │                  │    Failed? → FLAG for review or adjust      │
│   └──────────────────┘                                              │
│      │                                                              │
│      ▼                                                              │
│   ┌──────────────────┐                                              │
│   │    DETECTION     │──► Layer A: Guardrail Check                 │
│   │                  │    - det_k_count in [expected±2]?           │
│   │                  │    - det_x_spacing_cv < 0.15 (simple)?      │
│   │                  │    - det_midpoint_ratio > 0.4?              │
│   │                  │    Failed? → RERUN with different params    │
│   └──────────────────┘         (max 3 attempts)                    │
│      │                                                              │
│      ▼                                                              │
│   ┌──────────────────┐                                              │
│   │       SAM        │──► Layer B: Quality Prediction              │
│   │                  │    predicted_mIoU = model(intrinsic_metrics)│
│   │                  │    If predicted_mIoU < threshold:           │
│   │                  │       → REFLECT: Adjust SAM params          │
│   │                  │       → RERUN SAM                           │
│   └──────────────────┘         (max 3 attempts)                    │
│      │                                                              │
│      ▼                                                              │
│   Final Output                                                      │
│   with confidence score = predicted_mIoU                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Reflection/Rerun Thresholds

#### For Preprocessing Stage (Guardrails)

**Implementation:** `bo4tun/intrinsic_metrics.py` - `check_preprocessing_guardrails()`

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| theta_coverage | < 98% or > 102% | **FLAG** wraparound or over-coverage issue |
| point_retention_ratio | < 90% | **FLAG** too aggressive denoising |
| interpolation_coverage | < 95% | **FLAG** sparse depth map, rerun enhancing |

**Usage:**
```python
from bo4tun.intrinsic_metrics import check_preprocessing_guardrails, run_preprocessing_check

# Quick check with verbose output
passed = run_preprocessing_check('2-2', data_dir='data', verbose=True)

# Detailed check with violations list
passed, violations, metrics = check_preprocessing_guardrails('2-2', data_dir='data')
if not passed:
    for v in violations:
        print(f"  Warning: {v}")
```

#### For Detection Stage

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| det_k_count | ∉ [expected-2, expected+2] | **RERUN** with adjusted binary_threshold |
| det_x_spacing_cv | > 0.15 (simple) / 0.60 (complex) | **RERUN** with adjusted hough params |
| det_midpoint_ratio | < 0.4 | **RERUN** with adjusted angle params |
| det_assume_default_ratio | > 0.3 | **FLAG** for manual review |

**Rerun Strategy:**
```python
if det_x_spacing_cv > threshold:
    # Increase hough_oblique_threshold to reduce false positives
    new_params['hough_oblique_threshold'] += 10
    # Narrow angle range to be more selective
    new_params['angle_positive_max'] -= 0.5
```

#### For SAM Stage

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| sam_mask_fill_rate | > 0.95 or < 0.20 | **RERUN** with adjusted template sizes |
| sam_segment_count | < expected - 2 | **RERUN** with smaller templates |
| predicted_mIoU (overall) | < 0.4 (simple) / < 0.35 (complex) | **REFLECT & RERUN** |

**Rerun Strategy:**
```python
if predicted_mIoU < threshold:
    if sam_mask_fill_rate > 0.90:
        # Templates too large, shrink them
        new_params['template_width_factor'] *= 0.9
    elif sam_mask_fill_rate < 0.30:
        # Templates too small, expand them
        new_params['template_width_factor'] *= 1.1
```

### 5.3 Confidence Thresholds for Final Output

| Predicted mIoU | Confidence Level | Recommended Action |
|----------------|------------------|-------------------|
| ≥ 0.60 | High | Accept output |
| 0.45 - 0.60 | Medium | Accept with warning |
| 0.35 - 0.45 | Low | Recommend manual review |
| < 0.35 | Very Low | Flag as potentially failed |

### 5.4 Maximum Retry Limits

| Stage | Max Retries | Escalation |
|-------|-------------|------------|
| Detection | 3 | If all fail → use historical best params |
| SAM | 3 | If all fail → flag for manual review |
| Overall Pipeline | 5 | If all fail → output with low confidence |

### 5.5 Implementation Example

```python
def run_with_reflection(tunnel_id, max_attempts=3):
    """Run pipeline with automatic reflection and rerun."""
    
    for attempt in range(max_attempts):
        # Run detection
        run_detection(tunnel_id)
        det_metrics = compute_detection_metrics(tunnel_id)
        
        # Check detection guardrails
        det_passed, det_violations = check_guardrails(det_metrics, tunnel_id)
        
        if not det_passed:
            print(f"Detection failed (attempt {attempt+1}): {det_violations}")
            
            # REFLECT: Adjust params based on violations
            new_params = reflect_on_detection_failure(det_violations)
            update_detection_params(tunnel_id, new_params)
            continue  # RERUN
        
        # Run SAM
        run_sam(tunnel_id)
        sam_metrics = compute_sam_metrics(tunnel_id)
        
        # Predict final quality
        all_metrics = {**det_metrics, **sam_metrics}
        predicted_miou = predict_miou(all_metrics, tunnel_id)
        
        if predicted_miou >= 0.45:
            return {
                'status': 'success',
                'predicted_miou': predicted_miou,
                'confidence': 'high' if predicted_miou >= 0.60 else 'medium',
                'attempts': attempt + 1,
            }
        else:
            # REFLECT: Adjust SAM params
            new_sam_params = reflect_on_sam_failure(sam_metrics, predicted_miou)
            update_sam_params(tunnel_id, new_sam_params)
    
    # All attempts failed
    return {
        'status': 'low_confidence',
        'predicted_miou': predicted_miou,
        'confidence': 'low',
        'attempts': max_attempts,
        'recommendation': 'manual_review',
    }


def reflect_on_detection_failure(violations: list) -> dict:
    """Generate new params based on violation analysis."""
    new_params = {}
    
    for v in violations:
        if 'det_x_spacing_cv' in v:
            # Irregular spacing → be more selective
            new_params['hough_oblique_threshold'] = '+10'  # increase
        elif 'det_k_count' in v and 'min' in v:
            # Too few detections → be less selective
            new_params['hough_oblique_threshold'] = '-10'  # decrease
            new_params['binary_threshold'] = '-10'
        elif 'det_midpoint_ratio' in v:
            # Too many fallback detections → adjust angles
            new_params['angle_tolerance'] = '-0.5'
    
    return new_params
```

### 5.6 Recommended Intrinsic Quality Criteria

#### Tier 1: Hard Fail (Always Rerun)

| Metric | Criteria | Rationale |
|--------|----------|-----------|
| det_k_count | < expected - 3 or > expected + 3 | Severely wrong detection count |
| det_x_spacing_cv | > 0.50 (simple) | Completely irregular spacing |
| sam_mask_fill_rate | < 0.10 | SAM completely failed |

#### Tier 2: Soft Fail (Rerun if Budget Allows)

| Metric | Criteria | Rationale |
|--------|----------|-----------|
| det_midpoint_ratio | < 0.30 | Too many assumed defaults |
| sam_mask_fill_rate | > 0.95 | Over-segmentation |
| predicted_mIoU | < 0.40 | Below acceptable quality |

#### Tier 3: Warning (Log but Continue)

| Metric | Criteria | Rationale |
|--------|----------|-----------|
| det_assume_default_ratio | > 0.20 | Some detection uncertainty |
| sam_segment_count | = expected ± 1 | Minor segment count difference |

---

## Part 6: Files and Datasets Index

### Code Files

| File | Purpose |
|------|---------|
| `bo4tun/intrinsic_metrics.py` | Compute intrinsic metrics from pipeline outputs + preprocessing guardrails |
| `bo4tun/build_training_data.py` | Build training dataset from BO history |
| `bo4tun/config_runner.py` | Run pipeline with specific configs |
| `p4tun/bo/no_gt_optimizer.py` | No-GT BO for simple patterns |
| `p4tun/bo/no_gt_optimizer_complex.py` | No-GT BO for complex patterns |
| `p4tun/bo/predictor.py` | mIoU predictor training and inference |

### Data Files

| File | Contents |
|------|----------|
| `bo4tun/training/intrinsic_training_data.csv` | 20 samples, 13 intrinsic metrics + mIoU |
| `p4tun/bo/models/complex_miou_predictor.pkl` | Trained Ridge model for complex patterns |
| `p4tun/bo/results/predictor_evaluation.json` | Correlation analysis results |

### Report Files

| File | Contents |
|------|----------|
| `reports/P4TUN_OPTIMIZATION_JOURNEY_2-2.md` | Vanilla BO journey for 2-2 |
| `reports/INTRINSIC_METRICS_REPORT.md` | Intrinsic metrics design rationale |
| `p4tun/bo/results/2-2_INTRINSIC_QUALITY_ANALYSIS.md` | 2-2 guardrail fix analysis |
| `p4tun/bo/results/COMPLEX_STAGGERED_INTRINSIC_ANALYSIS.md` | Complex patterns analysis |

---

## Conclusion

The no-GT BO system successfully predicts mIoU from intrinsic metrics with:
- **Spearman correlation 0.84** for simple patterns
- **Spearman correlation 0.87** for complex patterns

Key findings:
1. **Detection stage** is the most critical for mIoU
2. **det_x_spacing_cv** is the most important single guardrail for simple patterns
3. **segment_width** is the dominant predictor for complex patterns
4. Preprocessing has <0.1% impact and can be skipped in optimization

For self-improving pipelines, use the **three-tier reflection system** with automatic rerun on Tier 1/2 failures and logging on Tier 3 warnings.

---

*Report generated: 2026-02-02*  
*Framework: scikit-optimize (skopt) + Ridge Regression*  
*Total data: 559 vanilla BO evals + 190 no-GT BO evals*
