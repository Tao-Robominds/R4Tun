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

#### Detection Intrinsic Metrics (8 metrics)

| Metric | Description | Good Indicator? | Correlation with mIoU |
|--------|-------------|-----------------|----------------------|
| `det_k_count` | Number of detected K-blocks | ✓ | Varies by tunnel |
| `det_k_count_match` | Count matches expected | ✓ | r = +0.52* |
| `det_assume_default_ratio` | Fallback detection % | ✓ (lower is better) | Negative |
| `det_midpoint_ratio` | Midpoint detection % | ✓✓✓ | **r = +0.87*** |
| `det_real_detection_ratio` | Non-fallback % | ✓✓ | r = +0.69** |
| `det_y_range` | Y-position spread | ✓ | Context-dependent |
| `det_y_std` | Y-position std dev | ✓ | Context-dependent |
| `det_x_spacing_cv` | X-spacing uniformity | ✓✓✓ | **Critical for 2-2** |

#### SAM Intrinsic Metrics (5 metrics)

| Metric | Description | Good Indicator? | Correlation with mIoU |
|--------|-------------|-----------------|----------------------|
| `sam_prompt_count` | Prompts fed to SAM | ✓ | Matches det_k_count |
| `sam_segment_count` | Output segment count | ✓ | Indirect |
| `sam_segment_count_match` | Count ≥ expected-1 | ✓ | r = -0.53 |
| `sam_mask_fill_rate` | Non-background % | ✓✓ | **r = -0.82*** |
| `sam_template_coverage` | Template coverage % | ✓ | Indirect |

**Note:** For complex staggered patterns, SAM **geometry parameters** (segment_width, k_height) showed much stronger correlation than intrinsic metrics:
- `segment_width`: r = -0.789*** with mIoU (lower is better)
- `k_height`: r = -0.269* with mIoU

### Phase 4: Ablation and Model Training

#### Training Data

| Dataset | Samples | Tunnels | Stages | mIoU Range |
|---------|---------|---------|--------|------------|
| Simple patterns | 20 | 1-4, 2-2, 4-1, 5-1 | sam, combined | 0.106 - 0.690 |
| Complex patterns | 70 | 4-1, 5-1 | complex_sam, sam_wraparound | 0.308 - 0.431 |

#### Ablation Results (Simple Patterns)

| Model | Features | R² | Spearman | Best Feature |
|-------|----------|-----|----------|--------------|
| Detection only | 3 | 0.71 | 0.80 | det_midpoint_ratio |
| SAM only | 1 | -0.12 | -0.18 | (not predictive alone) |
| **Combined** | **4** | **0.72** | **0.84** | det_midpoint_ratio + sam_mask_fill_rate |

**Conclusion:** Combined model performs best. Detection metrics are essential; SAM metrics alone are insufficient.

#### Final Model (Simple Patterns)

```python
# Ridge regression, trained on n=20
Features: ['det_midpoint_ratio', 'det_real_detection_ratio', 
           'det_k_count_match', 'sam_mask_fill_rate']
R² = 0.72
Spearman = 0.84
MAE = 0.09
```

#### Final Model (Complex Patterns)

```python
# Ridge regression, trained on n=70
Features: ['segment_width', 'ab_height', 'k_height', 'angle_deg']
CV MAE = 0.0125
Spearman = 0.8724
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
│   │  PREPROCESSING   │──► Intrinsic Check: theta_coverage > 98%    │
│   │  (Unfolding,     │    Failed? → Adjust unfolding params        │
│   │   Denoising,     │              or FLAG for manual review      │
│   │   Enhancing)     │                                              │
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
| `bo4tun/intrinsic_metrics.py` | Compute intrinsic metrics from pipeline outputs |
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
