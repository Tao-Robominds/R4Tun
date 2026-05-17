# 02 BO Experience Collection

## Runtime Path

Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/02_bo_experience_collection/`

## Goal

Use one-shot BO as a controlled way to generate valid labelled experience for proxy development. The first purpose is not only to find a good parameter setting; it is to learn which parameters are critical and which observable proxy features move with mIoU.

## Inputs

- Seed and few-shot BO rings from step 01.
- SAM4Tun static parameter configuration plus fixed baseline preprocessing/controller components.
- BO search spaces and hard bounds.
- Read-only baseline artifacts copied into the sandbox when needed.
- GT mIoU or stage-specific GT rewards for design-time labels only.

## BO Setup

- Use the SAM4Tun static parameters as the BO prior/seed, not a blank search.
- Define the `deterministic_baseline` before BO as SAM4Tun static parameters + ring-bottom alignment + fixed rule-based adaptation.
- Treat ring-bottom alignment as preprocessing, not as a learned proxy contribution.
- During calibration, tune deterministic-baseline parameters and BO-identified critical parameters together. Do not freeze all deterministic settings before BO, because interactions between baseline geometry, cleanup, and K/order parameters may explain proxy-feature changes.
- Tune around deterministic-baseline values rather than treating physical/GT values as automatically optimal for point-cloud processing. For example, a GT tunnel diameter may be 5.5, but a nearby processing value such as 5.6 may produce cleaner retained points or cleaner depth maps.
- Use the current Scaled Matern 5/2 kernel with ARD lengthscales unless an experiment explicitly compares kernels.
- Keep the prior neither too strong nor too weak:
  - strong enough to start from a physically meaningful tunnel-processing configuration;
  - weak enough that BO can observe real variation and failure cases.
- Log both trial outcomes and surrogate uncertainty/confidence quantities.

## Critical Parameter Watchlist

Record sensitivity for both deterministic-baseline parameters and parameters already suggested by v5 experience. BO may add or remove parameters from the final critical set, but these must be watched from the one-shot run:

- preprocessing geometry: `tunnel_diameter`, `radius_min`, `radius_max`;
- preprocessing sampling/enhancement: `target_distances`, `interpolation_window`;
- outlier/depth cleanup: `outlier_neighbors`, `outlier_depth_threshold_low`, `outlier_depth_threshold_high`, `outlier_high_density_ring_start`, `outlier_high_density_ring_end`, `outlier_interpolation_radius`;
- fixed preprocessing/controller context: `gravity_anchor`, `depth_height_mode`, `n_segment_start`, `n_segment_end`;
- K/order candidate controls: `regular_k_prior_low_frac`, `regular_k_prior_high_frac`, `regular_k_prior_low_ring_parity`, `anchor_frac`, `branch`, `rotation_shift`.

Parameter sensitivity should be reported as design evidence: which deterministic-baseline parameters must be tuned, which can remain fixed, and which only affect proxy features without improving mIoU.

## Actions
1. Run BO on the one-shot seed ring and record every trial.
2. For each trial, perturb the joint calibration space: deterministic-baseline parameters plus BO critical-parameter candidates.
3. Log both parameter values and observable feature deltas versus the deterministic baseline.
4. Add few-shot BO rings selected for condition diversity only after the one-shot confidence check fails or exposes uncovered conditions.
5. For every trial, save parameters, intrinsic diagnostics, surrogate mean/variance, acquisition value, selected artifacts, and GT audit metrics.
6. Keep the deterministic baseline fixed for each ring: SAM4Tun static + ring-bottom alignment + fixed rule-based adaptation.
7. Compute parameter sensitivity and identify which parameters have enough evidence to be varied at runtime.
8. Do not use BO outputs from held-out confidence-test rings to update the proxy.

## Required trial schema

Each row/object must include:

- `trial_id`, `tunnel_id`, `ring_id`, `condition_cluster`
- `params` and bounded search-space metadata
- `calibration_space`: deterministic baseline, BO critical, or joint interaction
- `param_delta_vs_deterministic_baseline`
- `candidate_role`: seed, BO trial, BO best, or runtime sample
- `artifacts` needed to recompute intrinsic features
- `intrinsic_metrics`
- `observable_feature_delta_vs_deterministic_baseline`
- `bo_surrogate_mean`, `bo_surrogate_std`, and acquisition metadata when available
- `gt_miou` and optional stage GT rewards for audit
- `deterministic_baseline_gt_miou` for improvement labels
- `deterministic_baseline_components`: SAM4Tun static, ring-bottom alignment, fixed rule-based adaptation

## Outputs

- `bo_trials.csv`
- `bo_surrogate_trace.csv`
- `bo_experience_summary.md`
- `parameter_sensitivity.md`
- `deterministic_baseline_registry.json`

## Verify Prompt

`Does every BO trial have enough metadata to reconstruct the candidate, compute intrinsic proxy features, audit GT mIoU, and recover BO uncertainty without touching protected data paths?`
