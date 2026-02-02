# Numerical Evidence Summary

All quantitative results from the no-GT BO development process.

---

## 1. Vanilla BO Performance (Ground Truth)

### Total Evaluations by Tunnel

| Tunnel | Pattern | BO Runs | Evaluations | Initial mIoU | Best mIoU | Δ mIoU |
|--------|---------|---------|-------------|--------------|-----------|--------|
| 1-4 | simple_staggered | 8 | 80 | 0.511 | 0.807 | +0.296 |
| 2-2 | continuous | 11 | 150+ | 0.672 | 0.765 | +0.093 |
| 3-1 | simple_staggered | 10 | 185 | 0.650 | 0.769 | +0.119 |
| 4-1 | complex_staggered | 4 | 45 | 0.316 | 0.428 | +0.112 |
| 5-1 | complex_staggered | 9 | 249 | 0.308 | 0.431 | +0.123 |
| **Total** | - | **42** | **559+** | - | - | - |

### Stage-by-Stage Impact (Tunnel 2-2)

| Phase | Stage | Iterations | mIoU Before | mIoU After | Δ |
|-------|-------|------------|-------------|------------|---|
| 1 | SAM (initial) | 30 | 0.672 | 0.700 | +4.2% |
| 2 | Detection | 30 | 0.700 | 0.744 | **+6.3%** |
| 3 | SAM (expanded) | 30 | 0.744 | 0.768 | +3.2% |
| 4 | Preprocessing | 30 | 0.768 | 0.769 | +0.1% |
| 5 | Unfolding | 30 | 0.769 | 0.769 | +0.0% |

**Source:** `reports/P4TUN_OPTIMIZATION_JOURNEY_2-2.md`

---

## 2. Training Data Statistics

### Simple Patterns Dataset

| Field | Value |
|-------|-------|
| **File** | `bo4tun/training/intrinsic_training_data.csv` |
| **Samples** | 20 |
| **Tunnels** | 1-4, 2-2, 4-1, 5-1 (5 each) |
| **Stages** | combined, sam, sam_wraparound, complex_sam |
| **Features** | 13 intrinsic metrics |
| **Target** | mIoU |

**mIoU Distribution:**
| Statistic | Value |
|-----------|-------|
| Min | 0.106 |
| Max | 0.690 |
| Mean | 0.402 |
| Std | 0.222 |

### Complex Patterns Dataset

| Field | Value |
|-------|-------|
| **Samples** | 70 |
| **Tunnels** | 4-1 (15), 5-1 (55) |
| **Features** | SAM geometry params (segment_width, k_height, ab_height, angle_deg) |
| **Target** | mIoU |

**mIoU Distribution:**
| Statistic | Value |
|-----------|-------|
| Min | 0.308 |
| Max | 0.431 |
| Mean | 0.376 |
| Std | 0.032 |

---

## 3. Correlation Analysis Results

### Simple Patterns (n=20)

| Metric | Spearman r | p-value | Significant |
|--------|-----------|---------|-------------|
| det_midpoint_ratio | **+0.87** | 0.000 | *** |
| sam_mask_fill_rate | **-0.82** | 0.001 | ** |
| det_real_detection_ratio | +0.69 | 0.001 | ** |
| det_k_count_match | +0.52 | 0.019 | * |
| sam_segment_count | -0.53 | 0.077 | ns |

**Source:** `p4tun/bo/results/predictor_evaluation.json`

### Complex Patterns (n=70)

| SAM Parameter | Spearman r | p-value | Interpretation |
|---------------|-----------|---------|----------------|
| segment_width | **-0.789** | 0.000 | Lower = better (DOMINANT) |
| ab_height | -0.341 | 0.002 | Lower = better |
| k_height | -0.269 | 0.024 | Lower = better |
| angle_deg | -0.152 | 0.210 | Not significant |

**Source:** `p4tun/bo/results/COMPLEX_STAGGERED_INTRINSIC_ANALYSIS.md`

---

## 4. Predictor Model Performance

### Simple Pattern Predictor

| Configuration | R² | Spearman | MAE | Features |
|---------------|-----|----------|-----|----------|
| Detection only | 0.71 | 0.80 | - | 3 |
| SAM only | -0.12 | -0.18 | - | 1 |
| **Combined** | **0.72** | **0.84** | **0.09** | **4** |

**Final Features:**
1. det_midpoint_ratio
2. det_real_detection_ratio
3. det_k_count_match
4. sam_mask_fill_rate

### Complex Pattern Predictor

| Metric | Value |
|--------|-------|
| Model | Ridge Regression |
| Features | 4 (SAM params) |
| CV MAE | **0.0125** |
| Spearman | **0.8724** |
| R² (approx) | ~0.76 |

**Model Coefficients (scaled):**
| Feature | Coefficient |
|---------|-------------|
| segment_width | -0.0352 |
| ab_height | -0.0067 |
| k_height | -0.0012 |
| angle_deg | -0.0008 |
| intercept | 0.3763 |

---

## 5. Guardrail Thresholds

### Simple Patterns (Base)

| Metric | Min | Max | Source |
|--------|-----|-----|--------|
| det_k_count_match | 0.8 | - | Correlation analysis |
| det_midpoint_ratio | 0.4 | - | Correlation analysis |
| det_real_detection_ratio | 0.5 | - | Correlation analysis |
| det_x_spacing_cv | - | 0.15 | 2-2 failure analysis |
| sam_mask_fill_rate | - | 0.95 | Correlation analysis |

### Simple Patterns (Tunnel-Specific)

| Tunnel | Metric | Override |
|--------|--------|----------|
| 2-2 | det_x_spacing_cv | max: 0.10 (stricter) |

### Complex Patterns (Base)

| Metric | Min | Max | Source |
|--------|-----|-----|--------|
| det_k_count | 4 | 12 | Historical analysis |
| det_x_spacing_cv | - | 0.60 | Historical analysis |
| det_y_range | 200 | 1500 | Historical analysis |

### Complex Patterns (Tunnel-Specific)

| Tunnel | det_k_count | det_x_spacing_cv | det_y_range |
|--------|-------------|------------------|-------------|
| 4-1 | 7-12 | max: 0.50 | 200-2000 |
| 5-1 | 5-10 | max: 0.80 | 200-3500 |

### SAM Parameter Guardrails (Complex)

| Parameter | Min | Max | Optimal |
|-----------|-----|-----|---------|
| segment_width | 1150 | 1350 | ~1168 (lower better) |
| k_height | 900 | 1200 | ~950 (lower better) |
| ab_height | 3000 | 3500 | ~3240 (lower better) |
| angle_deg | 6.0 | 9.0 | ~7.3 |

---

## 6. No-GT BO Validation Results

### Detection Stage Validation

| Tunnel | Predicted mIoU | True mIoU | Error | % Error |
|--------|----------------|-----------|-------|---------|
| 1-4 | 0.646 | 0.576 | +0.070 | +12% |
| 2-2 (before fix) | 0.672 | 0.476 | +0.196 | **+41%** |
| 2-2 (after fix) | 0.672 | 0.690 | -0.018 | -3% |
| 3-1 | 0.520 | 0.500 | +0.020 | +4% |
| 4-1 | 0.344 | - | - | - |
| 5-1 | 0.360 | 0.391 | -0.031 | -8% |

### 2-2 Fix Impact

| Metric | Before Fix | After Fix | Change |
|--------|------------|-----------|--------|
| True mIoU | 0.476 | 0.690 | **+45%** |
| det_x_spacing_cv | 0.484 | ~0.00 | Fixed |

**Fix Applied:** `det_x_spacing_cv < 0.10` guardrail + tunnel-specific predictor with -0.50 coefficient

---

## 7. File Inventory

### BO Result Files

| Category | Count | Location |
|----------|-------|----------|
| Vanilla BO history | 40+ | `p4tun/bo/results/*_history.json` |
| No-GT BO results | 6 | `p4tun/bo/results/no_gt_bo_*.json` |
| Complex no-GT results | 4 | `p4tun/bo/results/no_gt_complex_*.json` |
| Proxy experiments | 8 | `p4tun/bo/results/proxy_bo_*.json` |

### Training/Model Files

| File | Size | Purpose |
|------|------|---------|
| `intrinsic_training_data.csv` | 22 lines | Simple pattern training |
| `complex_miou_predictor.pkl` | ~5KB | Complex pattern model |
| `predictor_evaluation.json` | 76 lines | Correlation results |

### Report Files

| File | Lines | Purpose |
|------|-------|---------|
| `P4TUN_OPTIMIZATION_JOURNEY_2-2.md` | 950 | Vanilla BO journey |
| `INTRINSIC_METRICS_REPORT.md` | 151 | Metrics rationale |
| `2-2_INTRINSIC_QUALITY_ANALYSIS.md` | 153 | 2-2 fix analysis |
| `COMPLEX_STAGGERED_INTRINSIC_ANALYSIS.md` | 163 | Complex analysis |

---

## 8. Key Numerical Thresholds for Self-Improvement

### Reflection Triggers

| Level | Metric | Threshold | Action |
|-------|--------|-----------|--------|
| **Tier 1** | det_k_count | < expected-3 OR > expected+3 | **MUST RERUN** |
| **Tier 1** | det_x_spacing_cv | > 0.50 (simple) | **MUST RERUN** |
| **Tier 1** | sam_mask_fill_rate | < 0.10 | **MUST RERUN** |
| **Tier 2** | predicted_mIoU | < 0.40 | Rerun if budget |
| **Tier 2** | det_midpoint_ratio | < 0.30 | Rerun if budget |
| **Tier 3** | det_assume_default_ratio | > 0.20 | Log warning |

### Confidence Mapping

| Predicted mIoU | Confidence | Recommendation |
|----------------|------------|----------------|
| ≥ 0.60 | High | Accept |
| 0.45 - 0.60 | Medium | Accept with warning |
| 0.35 - 0.45 | Low | Manual review |
| < 0.35 | Very Low | Likely failed |

### Retry Limits

| Stage | Max Retries | On All Fail |
|-------|-------------|-------------|
| Detection | 3 | Use historical best |
| SAM | 3 | Flag for review |
| Overall | 5 | Output with low confidence |

---

*Generated: 2026-02-02*
