# 04 Proxy Family and Feature Ablation

## Goal

Define and compare intrinsic proxy candidates. The goal is to identify observable rewards that predict or select mIoU-improving candidates, not to claim interpretability without user-study evidence.

## Runtime Path

Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/04_proxy_family/`

## Inputs

- Candidate-level dataset from step 03.
- Intrinsic feature groups from preprocessing, detection, layout, and candidate-distribution diagnostics.
- Existing proxy variants with 1, 3, 4, and 12 intrinsic features.
- BO surrogate uncertainty features where available.

## V5 Observable Feature Watchlist

Keep these v5-derived features visible from the one-shot BO stage onward:

- 3-feature observable proxy: `struct_missing_ids_before_n`, `depth_row_nonempty_ratio_audit`, `geom_boundary_gap_cv`;
- 4-feature balance proxy: the 3-feature set plus `balance_norm`;
- boundary geometry: `geom_boundary_min_gap_frac`, `geom_boundary_max_gap_frac`, `geom_boundary_expected_resid_frac`;
- K/order observables: `k_anchor_dist_frac`, `k_y_frac`, `horizontal_line_count`, `positive_line_count`, `negative_line_count`;
- predicted-class distribution: `feat_present_ratio`, `feat_entropy`, `feat_cv`, `feat_max_share`, `feat_nonzero_classes`;
- candidate descriptors for audit, not standalone proxy claims: `branch_is_minus`, `rotation_shift_num`, `anchor_frac`, `low_parity`.

The plan should track both absolute feature values and deltas versus the deterministic baseline. Features that do not vary within a ring cannot explain candidate selection for that ring, even if they correlate across rings.

## Proxy Families

Compare simple, defensible proxy families before choosing a main paper model:

- single-feature proxy: one strongest intrinsic signal;
- 3-feature proxy: compact reward from the strongest independent feature groups;
- 4-feature proxy: compact proxy with one additional stability/coverage signal;
- 12-feature proxy: fuller observable proxy;
- ridge / Huber-style linear proxy where useful for stable candidate ranking;
- BO surrogate confidence as auxiliary evidence, not necessarily the final proxy.

Each feature must be GT-free at runtime. GT mIoU is joined only for training and audit.

## Order-Switching Boundary

Order switching remains a proxy-calibration candidate operation, not a baseline preprocessing step. For each ring where order ambiguity exists:

- generate the deterministic baseline;
- generate the plausible switched-order candidate(s);
- compute GT-free proxy features for each candidate;
- select by higher proxy score;
- audit whether the higher proxy score corresponds to higher GT mIoU.

This directly tests whether the proxy can identify a better segmentation after switching order, rather than hiding the order correction inside the baseline.

## Actions

1. Freeze the candidate feature list and normalization rules.
2. Train or fit each proxy only on the allowed BO/few-shot training split.
3. Evaluate both regression quality and selection quality:
   - correlation with candidate GT mIoU;
   - top-1 selected candidate mIoU;
   - improvement over the deterministic baseline;
   - rank agreement within each ring;
   - robustness across condition clusters.
4. Report feature-delta behavior from the one-shot seed before adding more shots.
5. Evaluate order-switch pairs as a specific proxy-selection subtest.
6. Record model complexity and feature count as part of the ablation.
7. Select a main proxy based on validation performance and confidence calibration, not just average accuracy.
8. Keep interpretability claims limited to observable feature definitions and ablation results.

## Outputs

- `proxy_feature_bank.json`
- `v5_feature_watchlist.csv`
- `one_shot_feature_delta_report.md`
- `proxy_family_results.csv`
- `order_switch_proxy_eval.csv`
- `proxy_ablation_report.md`
- `selected_proxy.json`

## Verify Prompt

`Are proxy variants compared on the same split, with GT-free runtime features, candidate-selection metrics, and no unsupported interpretability claims?`
