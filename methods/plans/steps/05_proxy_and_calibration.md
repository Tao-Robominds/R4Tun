# 05 Spearman and Threshold Selection

## Goal
Validate the fixed combined proxies with a lightweight check and choose reflection thresholds. Do not fit Ridge, Platt, or logistic calibration for the main workflow.

## Runtime Path
Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/05_proxy_and_calibration/`

## Inputs
- Stage proxy scores from step 04.
- Final mIoU labels from step 03.
- Threshold-selection split definition.
- Target failure threshold `tau` for `G_bad = 1[mIoU < tau]`.

## Actions
1. Compute Spearman correlation for:
   - `S_depth` vs final mIoU
   - `S_boundary` vs final mIoU
2. Interpret correlation simply:
   - positive and reasonably strong -> useful proxy
   - weak but separates bad failures -> usable guardrail
   - near zero -> not reliable alone
   - negative -> likely wrong definition or regime-dependent metric
3. Choose `T_depth` and `T_boundary` from labelled validation data to catch low-mIoU cases.
4. Record threshold trade-offs: bad-case recall, trigger precision, false-negative rate, accepted-case mIoU.
5. Do not correlate every sub-metric unless the combined proxy fails.

## Outputs
- `proxy_thresholds.md`
- `proxy_eval.json`
- `thresholds.json`

## Verify Prompt
`Are Spearman checks documented, thresholds selected on labelled validation data, and Ridge/Platt/ablation excluded from the main workflow?`
