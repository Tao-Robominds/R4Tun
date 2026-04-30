# 02 BO Calibration

## Goal
Run ring-wise BO on representative rings and produce complete trial logs for downstream memory and proxy learning.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/02_bo_calibration/`

## Inputs
- Regime sampling panel (step 01)
- BO search space by stage
- Seg2Tunnel GT labels for evaluation

## Actions
1. Run BO per ring/regime representative.
2. For each trial, log: ring condition, tuned params, stage artifact paths, GT metrics.
3. Compute sensitivity summaries and failure attribution by stage.

## Required trial schema
Each row/object must include:
- `trial_id`, `ring_id`, `regime`
- `params` (full dict)
- `artifacts` (depth map, lines, segmentation outputs)
- `metrics`: mIoU, OA, boundary accuracy, K-block accuracy
- optional feature blocks `x_P`, `x_O` if computed inline

## Outputs
- `bo_trials.csv`
- `bo_summary.md`
- `sensitivity_report.md`
- `failure_casebook.md`

## Verify Prompt
`Does each BO trial have enough metadata and artifact references to reproduce quality and feature extraction?`
