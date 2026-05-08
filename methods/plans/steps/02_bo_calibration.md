# 02 Stage-wise BO Calibration

## Goal
Run preprocessing BO and detection/boundary BO separately on representative rings. Produce complete trial logs for tuning memory, fixed proxy scoring, threshold selection, and final mIoU validation.

## Runtime Path
Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/02_bo_calibration/`

## Inputs
- Stage-specific panels from step 01.
- BO search space by stage.
- Read-only baseline artifacts copied into the sandbox when needed.
- GT labels for labelled tuning and final validation only.

## Actions
1. Run preprocessing BO separately and maximize the single GT-derived preprocessing reward `foreground_mask_iou`.
2. Run detection/boundary BO separately using detection-stage outputs and final segmentation/reprojection metrics.
3. For each trial, log ring condition, tuned parameters, stage artifact paths, intrinsic diagnostics, and GT metrics.
4. For selected or important trial outputs, run the downstream steps needed to compute final mIoU after reprojection.
5. Compute sensitivity summaries and failure attribution by stage.

## Required trial schema
Each row/object must include:
- `trial_id`, `stage`, `tunnel_id`, `ring_id`, `regime`
- `params` (full dict)
- `artifacts` (depth map, boundary lines, segmentation outputs as applicable)
- `stage_metrics` (stage reward plus intrinsic diagnostics)
- `final_metrics`: final mIoU after reprojection when available
- `proxy_inputs_ready`: whether the artifacts are sufficient for step 04

## Outputs
- `stage_bo_trials.csv`
- `stage_bo_summary.md`
- `sensitivity_report.md`
- `failure_casebook.md`

## Verify Prompt
`Does each stage BO trial have enough metadata and artifact references to reproduce the stage reward, fixed proxy score, and final mIoU validation?`
