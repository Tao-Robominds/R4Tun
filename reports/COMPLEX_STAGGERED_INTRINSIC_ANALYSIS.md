# Complex Staggered (4-1, 5-1) Intrinsic Quality Analysis

**Date:** 2026-02-02  
**Updated:** 2026-02-02

## Problem Statement

Create a no-GT BO process for complex staggered patterns (tunnels 4-1 and 5-1) that can
predict mIoU from intrinsic metrics without ground truth.

## Historical mIoU Performance

| Tunnel | mIoU Range | Best mIoU | Pattern | Samples |
|--------|------------|-----------|---------|---------|
| 4-1 | 0.316 - 0.428 | 0.428 | complex_staggered | 15 |
| 5-1 | 0.308 - 0.431 | 0.431 | complex_staggered | 55 |
| **Combined** | 0.308 - 0.431 | 0.431 | - | **70** |

---

## TRAINED MODEL: SAM Parameter-Based Predictor

### Key Finding: SAM Geometry Parameters Dominate mIoU

For n=70 evaluations, correlation with mIoU:

| Parameter | Correlation | p-value | Interpretation |
|-----------|-------------|---------|----------------|
| **segment_width** | **-0.789*** | 0.000 | Smaller = better! (DOMINANT) |
| **ab_height** | **-0.341*** | 0.002 | Lower = better |
| **k_height** | **-0.269*** | 0.024 | Lower = better |
| angle_deg | -0.152 | 0.210 | Weak effect |

### Trained Ridge Regression Model

```python
# Model: mIoU = f(segment_width, ab_height, k_height, angle_deg)
# Training: n=70 samples, CV MAE=0.0125, Spearman=0.8724

Model coefficients (scaled features):
  segment_width: -0.0352  (dominant!)
  ab_height:     -0.0067
  k_height:      -0.0012
  angle_deg:     -0.0008
  intercept:      0.3763  (mean mIoU)
```

### Model Validation

| Metric | Value |
|--------|-------|
| Cross-validation MAE | **0.0125** |
| Spearman correlation | **0.8724** |
| R² (approx) | ~0.76 |

### Test Results (5-1 SAM Stage)

| Metric | Value |
|--------|-------|
| True mIoU | 0.3911 |
| Predicted mIoU | 0.3597 |
| **Prediction Error** | **0.0314 (~8%)** |

The model correctly identifies: **lower segment_width → higher mIoU**

---

## Detection Guardrails for Complex Staggered

Since SAM params predict mIoU but detection can still fail badly, use guardrails:

### Base Guardrails

```python
COMPLEX_GUARDRAIL_THRESHOLDS = {
    'det_k_count': {'min': 4, 'max': 12},       # Reasonable K-block count
    'det_x_spacing_cv': {'min': None, 'max': 0.60},  # X-spacing uniformity
    'det_y_range': {'min': 200, 'max': 1500},   # Y position spread
}
```

### Tunnel-Specific Guardrails

| Tunnel | det_k_count | det_x_spacing_cv | det_y_range |
|--------|-------------|------------------|-------------|
| **4-1** | 7-12 | < 0.50 | 200-2000 |
| **5-1** | 5-10 | < 0.80 | 200-3500 |

**5-1 Special:** Has non-uniform ring spacing (large gap), so higher x_spacing_cv is expected.

### SAM Parameter Guardrails

```python
SAM_PARAM_GUARDRAILS = {
    'segment_width': {'min': 1150, 'max': 1350},  # Optimal range (lower=better)
    'k_height': {'min': 900, 'max': 1200},        # Lower is better
    'ab_height': {'min': 3000, 'max': 3500},      # Lower is better
    'angle_deg': {'min': 6.0, 'max': 9.0},        # Reasonable range
}
```

---

## Files Created/Updated

1. **`p4tun/bo/no_gt_optimizer_complex.py`** - No-GT BO optimizer for complex staggered
2. **`p4tun/bo/models/complex_miou_predictor.pkl`** - Trained Ridge regression model
3. **`bo4tun/intrinsic_metrics.py`** - Handle complex detection types
4. **`bo4tun/build_complex_training_data.py`** - Build training data (optional)

---

## Usage

### Run No-GT BO for Complex Staggered

```bash
# Detection only
python -m p4tun.bo.no_gt_optimizer_complex --tunnel 5-1 --stage detection --n-calls 20

# SAM optimization (uses trained predictor)
python -m p4tun.bo.no_gt_optimizer_complex --tunnel 5-1 --stage sam --n-calls 20

# Combined (expensive but most accurate)
python -m p4tun.bo.no_gt_optimizer_complex --tunnel 5-1 --stage combined --n-calls 20
```

---

## Recommendations

### For Production Use

1. **Use SAM stage** with trained predictor (r=0.87 correlation)
2. **Target segment_width ~1150-1180** (lower is better)
3. **Target k_height ~950** (lower is better)
4. Detection guardrails filter clearly bad configs

### Best Known Parameters (5-1)

```json
{
    "segment_width": 1167.87,
    "k_height": 950.0,
    "ab_height": 3239.77,
    "angle_deg": 7.26
}
```
Expected mIoU: ~0.43

---

## Summary

| Component | Approach | Performance |
|-----------|----------|-------------|
| **Predictor** | Ridge regression on SAM params | CV MAE=0.0125, r=0.87 |
| **Guardrails** | Detection metrics + SAM param bounds | Filter bad configs |
| **Key Feature** | segment_width | r=-0.789 with mIoU |

For complex staggered patterns, **SAM geometry parameters are the primary predictors**,
not detection intrinsic metrics. The trained model achieves ~8% prediction error.
