# 04 Critical Parameters for BO

## Goal

Produce (1) a **full list of all available parameters** in the pipeline (inventory with stage, type, source, value), and (2) the **critical parameter set ready for BO tuning** (e.g. `critical_params.yaml`): selected params with bounds, excluded params with reasons, and safe fixed values. This step merges parameter inventory, data-flow, and critical selection into a single workflow.

## Runtime Path

`data/{tunnel_id}/workflow/{run_id}/04_critical_params_for_bo/`

## Inputs

- Upgraded solution code (`agents/irregular/` pipeline)
- Challenge map (`output/02_challenge_map_output.md` or runtime equivalent)

## Actions

1. **Parameter inventory**
   - Extract all tunable parameters from preprocessing, detection, and segmentation (stage, name, type, source file, current value).
   - Extract all fixed parameters; record stage, type, source, and value.

2. **Data-flow**
   - List pipeline nodes (stages) and directed edges (outputs → inputs).
   - Identify the critical path (stages that affect final segmentation / mIoU).

3. **Critical param set**
   - Apply selection rule: challenged assumptions + stage criticality + data-flow (params on critical path and linked to broken assumptions get priority).
   - Record **selected** params (name, stage, bounds, default).
   - Record **excluded** params and reason.
   - Record **safe fixed** params (no tuning).

## Outputs

- **All available parameters** (required): full inventory of every parameter in the pipeline (e.g. `parameters.csv` or `all_parameters.yaml`). For each parameter record: stage, name, type, source file, current value, and whether it is tunable or fixed.
- **Critical parameters for BO** (required): `critical_params.yaml` (or equivalent) listing only the selected subset ready for BO tuning (bounds, default), plus excluded params with reason and safe fixed params.
- Optional: `graph.md` (data-flow summary) in the same folder.

## Verify Prompt

```
1. Is there a full inventory listing all available parameters (stage, name, type, source, value, tunable/fixed)?
2. Does the critical artifact list selected parameters with bounds and stage?
3. Are excluded parameters listed with a short reason?
4. Are safe fixed params stated?
5. Is the selection rule (challenge + critical path) documented?
```

## Verify Script

```bash
python methods/plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 04
```
