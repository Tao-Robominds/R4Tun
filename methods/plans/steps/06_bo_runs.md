# 06 BO Runs

## Goal
Produce `bo_run.md`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/06_bo_runs/bo_run.md`

## Inputs
- warm start (step 05)
- critical set (step 04)
- objective definition

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
