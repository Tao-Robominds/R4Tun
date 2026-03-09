# 06 BO Runs

## Goal
Produce `bo_run.md`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/06_bo_runs/bo_run.md`

## Inputs
- warm start (step 05)
- critical set (step 04)
- objective definition

## Known Structural Limitation — `group_offsets`

`group_offsets` (12D) is BO-tunable but **structurally capped**. The group assumption — that all rings within a stagger group share the same offsets — is wrong for irregular tunnels where A2/A3 can sit on opposite sides of K depending on the ring.

**Evidence:** group_offsets give 0.388 mIoU vs GT centres 0.594 (same templates); the 0.206 gap is entirely from wrong A2/A3 centres. BO can find the best compromise offsets across all rings in a group, but it cannot overcome the fact that ring 0's A2 might need offset +1200 px while ring 3's A2 needs -1200 px. A single shared value will always be wrong for at least some rings.

**Realistic BO ceiling:** incremental improvement within the 0.501 neighbourhood (maybe 0.52–0.55). BO **cannot** close the gap to 0.720 (GT centres).

**To close the gap:** per-ring offsets are needed, but that explodes dimensionality to `ring_count × 6` (e.g. 7 × 6 = 42D), too many for standard BO. Possible mitigations: hierarchical BO (group then per-ring), learned offset predictor, or groove-based per-ring refinement.

## Actions
1. Record BO space.
2. Record logging contract.
3. Record stop rule and output paths.
4. Record structured artifact schema for each eval.

## Outputs
- `bo_run.md`
- `run.json` template
- `stage_manifest.json` template
- `reflection_log.json` template

## Support Templates
- `plans/templates/bo_run.md.template`
- `plans/templates/run.json.template`
- `plans/templates/stage_manifest.json.template`
- `plans/templates/reflection_log.json.template`

## Verify Prompt
`Does the BO artifact define metadata, params, GT outcomes, stage artefacts, feature bank, reflection logs, and stop rule?`

## Verify Script
```bash
python methods/plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 06
```
