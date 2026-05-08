# 04 Fixed Intrinsic Proxies

## Goal
Build fixed, GT-free combined proxies from pipeline outputs. The proxies are reflection triggers, not replacements for GT evaluation and not learned mIoU predictors.

## Runtime Path
Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/04_intrinsics_and_ontology/`

## Inputs
- Stage BO artifacts and logs (step 02).
- Trial dataset with final mIoU joins (step 03).
- Preprocessing artifacts (`depth_map.npy`, masks, context files).
- Detection/boundary artifacts (`all_segments.csv`, `final.csv`, boundary lines, K-line diagnostics).

## Fixed proxies
- `S_depth = S_coverage * S_empty`
  - `S_coverage`: normalized valid foreground/depth coverage.
  - `S_empty`: normalized penalty inverse for dominant empty components, near-empty valid ratio, and empty row/column bands.
- `S_boundary = S_continuity * S_K * S_spacing * S_layout`
  - `S_continuity`: boundary continuity and line support.
  - `S_K`: key-segment/key-boundary support.
  - `S_spacing`: spacing regularity.
  - `S_layout`: segment/layout plausibility.

Each component must be normalized to `[0, 1]`, where higher is better.

## Actions
1. Define the exact formula, normalization, clipping, and missing-artifact handling for each component.
2. Compute `S_depth` for preprocessing trial outputs.
3. Compute `S_boundary` for detection/boundary trial outputs.
4. Join combined proxy scores to final mIoU for validation samples.
5. Log component scores for diagnostics, but validate the combined proxies first.
6. Only analyze sub-metric correlations if a combined proxy fails.

## Outputs
- `fixed_proxy_definitions.md`
- `metric_bank.json`
- `stage_proxy_scores.csv`

## Verify Prompt
`Are S_depth and S_boundary fixed, normalized to [0, 1], computed without GT, and joined to final mIoU only for validation?`
