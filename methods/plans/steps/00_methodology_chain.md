# Methodology Chain: BO Experience -> Proxy Confidence -> Generalisation

This is the single overview of the paper workflow in `methods/plans/steps/01` to `07`.
Design-time reverse-engineering history lives in `methods/plans/preparation/`.

## Ordered chain

1. **Ring-condition coverage** (step 01) - describe the ring population, define condition descriptors, and create few-shot / validation / held-out splits by condition-space coverage.
2. **BO experience collection** (step 02) - use BO with the SAM4Tun static prior to generate labelled candidate experience and uncertainty-aware search traces.
3. **Proxy dataset construction** (step 03) - convert BO and sampled runtime candidates into a candidate-level dataset with intrinsic features, selected-candidate labels, and GT mIoU audit columns.
4. **Proxy family and feature ablation** (step 04) - compare simple intrinsic proxy variants using 1, 3, 4, and 12 feature groups without making interpretability claims.
5. **Empirical confidence measurement** (step 05) - measure prediction quality, ranking margin, uncertainty/calibration, distance to training experience, and high- vs low-confidence success rates.
6. **Diversity expansion loop** (step 06) - identify hard negatives and out-of-distribution conditions, add additional BO shots only where confidence collapses, and test whether confidence improves.
7. **Held-out runtime evaluation** (step 07) - freeze the proxy and evaluate candidate sampling, confidence groups, and success distributions on unseen rings/tunnels.

## Definitions

| Symbol | Meaning |
|--------|---------|
| `x_i` | Intrinsic, GT-free feature vector for candidate `i`. |
| `y_i` | GT mIoU audit label for candidate `i`; never used at deployment. |
| `p_i` | Proxy score or predicted mIoU for candidate `i`. |
| `c_i` | Empirical confidence signal for candidate or ring `i`. |
| `deterministic_baseline` | SAM4Tun static parameters + ring-bottom alignment + fixed rule-based adaptation. |
| `selected` | Candidate with the highest proxy score under the frozen selector. |
| `improved` | `1[selected_GT_mIoU > deterministic_baseline_GT_mIoU]`, computed only for evaluation. |
| `proxy_margin` | `p_best - mean(p_rank2_to_rank5)`, one simple confidence signal. |
| `condition_distance` | Distance from a test ring/candidate to BO training experience in descriptor space. |

## Main scientific claim

The paper studies whether an observable intrinsic proxy can act as a reliable reward for ring-level self-improvement. The contribution is the empirical development and validation of proxy confidence:

- when the proxy is accurate;
- when it is confidently right;
- when it is confidently wrong;
- which ring conditions are covered or out-of-distribution;
- whether adding diverse BO experience expands the reliable region;
- whether more runtime candidates expose a stable success distribution.

## BO position

BO is not claimed as the only possible experience generator. It is used because it is efficient for few-shot labelled experience and naturally supports uncertainty/confidence measurement through the surrogate model and acquisition trace. The method should be written as compatible with other valid experience generators.

## Baseline Boundary

The baseline should include simple deterministic improvements that are now considered part of the fixed preprocessing/controller setup:

- SAM4Tun static parameters;
- ring-bottom alignment as preprocessing;
- rule-based adaptation where the rule is fixed before proxy learning.

Order switching is not part of this baseline. It remains a proxy-calibration and candidate-selection test: generate the plausible order alternatives, score them with the intrinsic proxy, and verify by GT audit that the higher proxy score tends to correspond to higher mIoU.

## Removed from main claim

- No deployment-time LLM reflection loop.
- No claim of human interpretability without user-study evidence.
- No rule-based expansion of parameters beyond BO-supported sensitivity and bounded candidate generation.
- GT mIoU is used only for design-time training/audit and held-out evaluation, not runtime selection.
