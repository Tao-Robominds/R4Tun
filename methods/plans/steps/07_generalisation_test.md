# 07 Held-Out Runtime Candidate Evaluation

## Goal

Evaluate the frozen proxy and confidence signal on unseen rings/tunnels. The final claim is not that the system reflects intelligently; it is that an intrinsic proxy can select improving candidates and quantify when it should be trusted.

## Runtime Path

Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/07_heldout_candidate_evaluation/`

## Inputs

- Frozen proxy from step 04 or step 06.
- Frozen confidence definition from step 05.
- Held-out rings/tunnels from step 01.
- Candidate generator bounded by BO-supported parameter ranges.
- GT labels for audit only after candidate selection.

## Isolation constraints

Held-out set must not be used for:

- BO optimisation;
- proxy feature selection;
- confidence definition;
- hard-negative expansion;
- candidate-generator tuning;
- any decision before final audit.

## Candidate Sampling Experiment

For each held-out ring:

1. include the deterministic baseline: SAM4Tun static + ring-bottom alignment + fixed rule-based adaptation;
2. generate the current 18 runtime candidates;
3. when compute allows, generate 100 bounded candidates;
4. include order-switch alternatives as candidate-selection cases where order ambiguity exists;
5. score candidates with the frozen proxy;
6. select the highest proxy-score candidate, or abstain if the frozen confidence rule says low confidence;
7. audit selected and deterministic-baseline outputs with GT mIoU.

The 18 to 100 candidate comparison is the Monte Carlo/dropout-style test. More samples do not prove the proxy is correct by themselves; they reveal the distribution of candidate quality and whether success is stable or lucky.

## Metrics

- selected candidate GT mIoU;
- deterministic-baseline GT mIoU;
- `delta_miou = selected - deterministic_baseline`;
- improvement rate;
- high-confidence success rate;
- low-confidence success rate;
- abstention rate;
- proxy margin distribution;
- condition-distance distribution;
- success distribution for 18 vs 100 candidates;
- oracle best candidate gap;
- order-switch selection success;
- runtime cost.

## Final Evidence

Report results by:

- all held-out rings;
- covered versus uncovered condition regions;
- tunnel/ring regime;
- high- versus low-confidence groups;
- candidate budget: 18 versus 100.

The strongest result is:

> high-confidence proxy selections improve more often than low-confidence selections, confidence drops on out-of-distribution regions, and adding diverse BO experience expands the region where high-confidence selections are reliable.

## Outputs

- `heldout_candidate_results.csv`
- `candidate_budget_comparison.csv`
- `confidence_success_summary.md`
- `oracle_gap_report.md`
- `final_proxy_confidence_evaluation.md`

## Verify Prompt

`Is the frozen proxy evaluated on untouched held-out rings, with deterministic-baseline comparison, confidence-group success rates, 18-vs-100 candidate distributions, and GT used only for final audit?`
