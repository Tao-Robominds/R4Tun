# Preprocessing QA and Retuning Guide

This note defines a deterministic QA loop for per-ring preprocessing outputs.
Use it to decide whether a ring is acceptable as-is or should be retuned with
explicit parameter changes.

## Inputs

- Preprocessing artifacts under one ring folder (`depth_map.npy`,
  `pixel_to_point.pkl`, optional stage metadata files).
- Parameter JSON used for the run.
- Local QA metrics from preprocessing scripts.

## Acceptance Gate

Treat a ring as accepted when all checks pass:

- `finite_ratio >= 0.60`
- `row_nonempty_ratio >= 0.90`
- `largest_empty_vertical_gap_frac <= 0.08`

If any gate fails, mark the ring for retuning and log the failed metric and
value.

## Stage Attribution

Use failure signatures to pick the first stage to retune:

- **Unfolding first** when support collapses in specific angle bands or the
  geometry frame is unstable.
- **Denoising first** when support is removed too aggressively across many rows.
- **Enhancing first** when interpolation/smoothing creates large artificial
  blank bands or distorts local continuity.

Retune one stage at a time unless there is hard evidence that two stages are
coupled for this ring.

## Retuning Rules

- Keep parameter edits small and bounded.
- Record each trial with before/after QA metrics.
- Prefer settings that improve support while preserving structure.
- Do not use downstream detection or segmentation scores as preprocessing
  objectives.

## Data Safety

- Read-only sources remain immutable.
- Write new outputs only to the active experiment sandbox.
- Promote artifacts manually after QA passes.
