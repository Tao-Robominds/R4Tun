# Pilot experience — preprocessing IoU BO on 1-1/r25

## Objective

Run a first-pilot BO loop using only:

`foreground_mask_iou = TP / (TP + FP + FN)`

with foreground support derived from preprocessing ring data (`denoised.csv` segment labels + `pixel_to_point.pkl`), without detection-stage `gt_ceiling/labelmap.npy`.

## Pilot setup

- runner: `bo/run_preprocessing_iou_bo.py`
- metric helper: `bo/preprocessing_iou_metrics.py`
- ring root: `data/ablation/baseline/1-1/r25`
- run id: `r25_iou_pilot_v1`
- trials: 8
- logs: `logs/preprocessing_bo/r25_iou_pilot_v1/1-1/r25/`

## Outcome

- baseline foreground_mask_iou: **0.003931**
- best foreground_mask_iou: **0.014383**
- delta: **+0.010452**
- result: **improved** in pilot

## Experience notes

- A subset of sampled parameter combinations can break denoising (`zero-size array`); the BO runner now treats those trials as invalid and assigns a poor objective instead of aborting.
- The best trial was more selective (lower valid_ratio and gt_foreground_ratio footprint), but produced better overlap IoU under this foreground mask definition.
- The runner restores original parameter JSON by default after pilot (`--apply-best` is optional), so this pilot is non-destructive to baseline parameters.

## Before full 8-ring BO

- Keep this exact objective (single reward only).
- Reuse the same runner over the remaining representative rings listed in `data/represents/preprocessing/manifest.json`.
- Review per-ring trial diagnostics to ensure improvements are not caused by degenerate tiny masks; keep IoU as objective and use diagnostics only for sanity checks.
