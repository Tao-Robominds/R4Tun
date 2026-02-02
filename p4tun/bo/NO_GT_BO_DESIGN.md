# No-Ground-Truth Bayesian Optimization Design

## Overview

This design enables parameter tuning **without ground truth labels** at runtime by using intrinsic metrics to predict mIoU.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BO LOOP                                   │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │  Sample  │────▶│   Run    │────▶│ Compute  │                │
│  │  Params  │     │ Pipeline │     │ Metrics  │                │
│  └──────────┘     └──────────┘     └──────────┘                │
│                                           │                      │
│                                           ▼                      │
│                        ┌─────────────────────────────────┐      │
│                        │      LAYER A: Guardrails        │      │
│                        │  ┌─────────────────────────┐    │      │
│                        │  │ det_k_count_match > 0.8 │    │      │
│                        │  │ det_midpoint_ratio > 0.4│    │      │
│                        │  │ sam_mask_fill < 0.95    │    │      │
│                        │  └─────────────────────────┘    │      │
│                        │         PASS?                   │      │
│                        └──────────┬──────────────────────┘      │
│                                   │                              │
│                         ┌────────┴────────┐                     │
│                         │                  │                     │
│                    NO (penalty)       YES (proceed)             │
│                         │                  │                     │
│                         └────────┬─────────┘                     │
│                                  ▼                               │
│                        ┌─────────────────────────────────┐      │
│                        │    LAYER B: mIoU Predictor      │      │
│                        │                                 │      │
│                        │  predicted_mIoU = f(metrics)    │      │
│                        │                                 │      │
│                        │  Features:                      │      │
│                        │  • det_midpoint_ratio (+0.87)   │      │
│                        │  • sam_mask_fill_rate (-0.82)   │      │
│                        │  • det_real_detection (+0.69)   │      │
│                        │  • det_k_count_match (+0.52)    │      │
│                        └──────────┬──────────────────────┘      │
│                                   │                              │
│                                   ▼                              │
│  ┌──────────┐     ┌──────────────────────────┐                  │
│  │   Next   │◀────│  Feed predicted_mIoU     │                  │
│  │  Params  │     │  to BO as objective      │                  │
│  └──────────┘     └──────────────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Layer A: Guardrails (Hard Constraints)

Filter out configurations that violate known quality requirements:

| Metric | Constraint | Rationale |
|--------|------------|-----------|
| `det_k_count_match` | > 0.8 | K-block count must match expected |
| `det_midpoint_ratio` | > 0.4 | Segments at expected positions |
| `det_real_detection_ratio` | > 0.5 | Not too many assumed defaults |
| `sam_mask_fill_rate` | < 0.95 | Avoid over-segmentation |

**Behavior:** Configs failing guardrails receive a penalty but are not completely rejected (allows BO to learn from failures).

## Layer B: Learned mIoU Predictor

Predict mIoU from intrinsic metrics using a Ridge regression model:

```
predicted_mIoU = 0.45 
               + 0.35 * det_midpoint_ratio
               + 0.15 * det_real_detection_ratio
               + 0.10 * det_k_count_match
               - 0.25 * sam_mask_fill_rate
```

**Performance (from evaluation):**
- R² = 0.72 (explains 72% of variance)
- Spearman = 0.84 (strong ranking correlation)
- MAE = 0.09 (~9 mIoU points)

## Usage

```bash
# Run no-GT BO on tunnel 1-4
python -m p4tun.bo.no_gt_optimizer --tunnel 1-4 --n-calls 20

# Quick test (5 evaluations)
python -m p4tun.bo.no_gt_optimizer --tunnel 1-4 --n-calls 5 --n-initial 2

# Optimize SAM stage only
python -m p4tun.bo.no_gt_optimizer --tunnel 2-2 --stage sam --n-calls 15
```

## Workflow

1. **Initial phase** (`n_initial` random samples):
   - Explore parameter space randomly
   - Build initial GP model

2. **Optimization phase** (remaining calls):
   - GP suggests promising parameters
   - Pipeline runs with new params
   - Intrinsic metrics computed
   - Guardrails checked (Layer A)
   - mIoU predicted (Layer B)
   - BO updates model with predicted score

3. **Result**:
   - Best parameters based on predicted mIoU
   - Full history for analysis

## Why This Works

1. **Intrinsic metrics correlate with mIoU** (proven in evaluation)
   - `det_midpoint_ratio`: r = +0.87
   - `sam_mask_fill_rate`: r = -0.82

2. **Combined predictor achieves R² = 0.72**
   - Good enough for ranking configurations
   - BO only needs relative ordering, not exact values

3. **Guardrails prevent catastrophic failures**
   - Reject configs that would definitely be bad
   - Based on documented failure modes

## Comparison: GT-BO vs No-GT-BO

| Aspect | GT-BO | No-GT-BO |
|--------|-------|----------|
| Needs labels | Yes | No |
| Runtime | Slower (GT eval) | Faster |
| Accuracy | Exact mIoU | Predicted mIoU |
| When to use | Offline tuning | Online/production |
| Risk | None | ~10% mIoU error |

## Limitations

1. **Predictor is approximate** (R² = 0.72, not 1.0)
2. **Per-tunnel variation**: Some tunnels may have worse prediction
3. **Training data dependency**: Model needs retraining if pipeline changes
4. **Extrapolation risk**: Untested param regions may have poor predictions

## Recommended Workflow

1. **Offline (with GT)**: Run standard BO to find good baseline params
2. **Online (no GT)**: Use no-GT BO for fine-tuning new tunnels
3. **Validation**: Periodically validate with GT on sample tunnels
