# 03 Proxy Dataset Construction

## Goal

Convert BO trials and runtime candidate samples into a candidate-level dataset for proxy learning, confidence analysis, and held-out evaluation.

## Runtime Path

Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/03_proxy_dataset/`

## Inputs

- BO experience from step 02.
- Deterministic baseline registry, where the deterministic baseline is SAM4Tun static + ring-bottom alignment + fixed rule-based adaptation.
- Runtime candidate samples where available.
- GT mIoU audit labels for candidates.
- Ring-condition descriptors from step 01.

## Actions

1. Build one row per `(ring, candidate)` rather than one row per tunnel.
2. Join intrinsic feature vectors to GT audit metrics.
3. Add relative labels:
   - `delta_miou_vs_deterministic_baseline`;
   - `improved = selected_GT_mIoU > deterministic_baseline_GT_mIoU`;
   - candidate rank by GT mIoU;
   - candidate rank by proxy score after a proxy is fitted.
4. Add parameter and feature deltas against the deterministic baseline.
5. Preserve split labels: BO train, validation, hard-negative, held-out.
6. Add condition-distance fields so later steps can distinguish covered from uncovered cases.
7. Store candidate artifacts by reference path only; do not copy large outputs into `methods/plans`.
8. Mark order-switch candidates separately from baseline candidates so their effect is evaluated as proxy selection, not hidden inside the deterministic baseline.

## One-Shot Outputs Needed Before Few-Shot Expansion

The one-shot BO dataset must be enough to answer:

- which parameters are sensitive near the deterministic baseline;
- whether deterministic-baseline parameters and BO critical parameters interact when calibrated together;
- whether the v5 proxy feature watchlist changes monotonically or consistently with mIoU;
- whether the one-shot proxy has a usable high-confidence region;
- which hard negatives remain before adding more shots.

## Candidate Sampling Schema

For confidence experiments, each evaluation ring should support repeated candidate sampling:

- current baseline: 18 candidates;
- expanded sampling target: 100 candidates when compute allows;
- candidate generation must stay within BO-supported parameter bounds;
- the deterministic baseline is always included and marked.
- order-switch alternatives may be included as bounded candidates, but are not part of the deterministic baseline.

This sampling is the dropout-style/Monte Carlo analogy: many bounded perturbations expose whether improvement is common, rare, or dependent on a lucky candidate.

## Outputs

- `candidate_dataset.csv`
- `candidate_features.json`
- `parameter_sensitivity_one_shot.csv`
- `feature_delta_one_shot.csv`
- `split_manifest.json`
- `candidate_sampling_manifest.json`
- `dataset_construction_report.md`

## Verify Prompt

`Does the proxy dataset contain candidate-level intrinsic features, GT audit labels, split membership, deterministic-baseline references, and enough runtime samples to measure confidence distributions?`
