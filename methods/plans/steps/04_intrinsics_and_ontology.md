# 04 Intrinsics and Ontology Features

## Goal
Build two GT-free feature blocks and guardrails from pipeline outputs.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/04_intrinsics_and_ontology/`

## Inputs
- BO artifacts and logs (step 02)
- Segmentation outputs (`all_segments.csv`, `final.csv`, detection artifacts)

## Feature blocks
- `x_P` (pipeline intrinsics): fill ratio, NaN block stats, groove alignment, K-count match, spacing CV, segment coverage, segment-count match, etc.
- `x_O` (ontology/structure): ring completeness, segment-count consistency, K-block plausibility, A/B height consistency, joint continuity, cyclic order plausibility, boundary angle plausibility, coverage plausibility.

## Actions
1. Compute `x_P` and `x_O` for each trial sample.
2. Define guardrails with pass/warn/fail thresholds.
3. Build a feature bank dataset with joins to GT labels.

## Outputs
- `intrinsics_ontology.md`
- `metric_bank.json`
- `feature_bank.csv`

## Verify Prompt
`Are both feature blocks present, guardrails defined, and feature rows aligned with BO/GT samples?`
