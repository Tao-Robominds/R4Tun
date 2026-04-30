# 05 Proxy and Calibration

## Goal
Train a single proxy model and calibrate pass probability for reflection/acceptance.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/05_proxy_and_calibration/`

## Inputs
- Feature bank with labels from step 04 (`x = [x_P; x_O]`, `y = mIoU`)
- Train/validation/calibration split definition

## Actions
1. Train regression proxy (default ridge): `y_hat = f(x)`.
2. Validate on holdout (MAE, RMSE, correlation).
3. Calibrate with Platt on margin `s = y_hat - tau` to obtain `p_good = sigma(a*s + c)`.
4. Define acceptance rule: `y_hat >= tau AND p_good >= p_min`.
5. Record uncertainty/trust policy.

## Outputs
- `proxy.md`
- `proxy_eval.json`
- `calibration.json`
- `confidence_bank.json`

## Verify Prompt
`Are regression and calibration both documented with split sizes, metrics, and a usable acceptance rule?`
