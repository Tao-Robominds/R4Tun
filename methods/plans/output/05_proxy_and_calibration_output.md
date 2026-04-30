# 05 Proxy and Calibration Output

## Objective
Report the consolidated proxy model and calibrated probability settings.

## Artifacts
- `proxy.md`
- `proxy_eval.json`
- `calibration.json`
- `confidence_bank.json`

## Required evidence
- Regression validation metrics
- Calibration parameters (`tau`, `a`, `c`)
- Brier score (and optional ECE)
- Acceptance rule and trust policy

## Summary table
| Model | MAE | RMSE | Brier | Acceptance rule |
|---|---:|---:|---:|---|
| Ridge + Platt | TODO | TODO | TODO | `y_hat >= tau AND p_good >= p_min` |
