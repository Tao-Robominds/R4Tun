# 05 Empirical Proxy Confidence

## Goal

Measure when the selected proxy should be trusted. Confidence is empirical: it must predict actual selection success, not merely produce a high score.

## Runtime Path

Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/05_proxy_confidence/`

## Inputs

- Selected proxy and proxy-family results from step 04.
- Candidate-level dataset from step 03.
- Validation split from step 01.
- BO uncertainty and condition-distance fields where available.

## Actions

1. Compute prediction metrics:
   - MAE/RMSE if predicting mIoU;
   - Spearman/Kendall rank correlation;
   - top-1 candidate selection success.
2. Compute confidence signals:
   - `proxy_margin = score_best - mean(score_rank2_to_rank5)`;
   - BO/surrogate uncertainty if available;
   - ensemble or bootstrap spread if used;
   - condition distance from BO training experience;
   - disagreement between feature groups or proxy variants.
3. Split validation rings into high- and low-confidence groups, for example top 30% by proxy margin versus the remainder.
4. Define success using GT audit only:
   - `improved = selected_GT_mIoU > deterministic_baseline_GT_mIoU`.
5. Report empirical confidence:
   - high-confidence success rate;
   - low-confidence success rate;
   - calibration/reliability curve;
   - confidently wrong cases;
   - abstention threshold if needed.
6. For order-switch candidates, report whether higher proxy score predicts higher GT mIoU within the same ring.
7. Define the minimum-shot stopping rule:
   - start with the one-shot proxy;
   - add a few-shot ring only if high-confidence success, calibration, or hard-negative analysis fails;
   - stop at the smallest shot count that satisfies the frozen confidence criteria.
8. Freeze the confidence definition before held-out testing.

## Expected Pattern

A useful confidence signal should show:

- higher selected-candidate success in high-confidence rings;
- lower success or more abstentions in low-confidence rings;
- degraded confidence on out-of-distribution condition clusters;
- fewer confidently wrong cases after diversity expansion in step 06.

## Outputs

- `confidence_metrics.csv`
- `confidence_groups.json`
- `reliability_curve.csv`
- `hard_negative_candidates.csv`
- `order_switch_confidence.csv`
- `minimum_shot_curve.csv`
- `minimum_shot_decision.md`
- `confidence_definition.md`

## Verify Prompt

`Does the chosen confidence signal separate cases where proxy-selected candidates actually improve mIoU from cases where the proxy is fragile or confidently wrong?`
