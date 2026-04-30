# 04 Critical Parameters for BO

## Methodology context

Design-time critical selection is documented below. Evidence-based updates after BO are optional; see [00_methodology_chain.md](00_methodology_chain.md).

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

## Post-BO refinement (optional)

After [step 06](06_bo_runs.md) (ring-by-ring or pooled BO), use **BO logs** to refine the critical set:

1. **Sensitivity / correlation:** rank parameters by correlation with objective (mIoU) or by marginal variance of mIoU when that dimension is perturbed.
2. **Ablation:** drop or fix parameters whose BO trajectory shows flat objective across bounds.
3. **Shrink or reorder** `critical_params.yaml`: demote insensitive params to fixed; promote params that explain variance.

Document changes in a short addendum (e.g. `critical_params_post_bo.md` or a section in the same folder) and version the YAML. This does not replace design-time step 3–4; it **narrows** the tunable set using empirical evidence.

## Outputs

- **All available parameters** (required): full inventory of every parameter in the pipeline (e.g. `parameters.csv` or `all_parameters.yaml`). For each parameter record: stage, name, type, source file, current value, and whether it is tunable or fixed.
- **Critical parameters for BO** (required): `critical_params.yaml` (or equivalent) listing only the selected subset ready for BO tuning (bounds, default), plus excluded params with reason and safe fixed params.
- Optional: `graph.md` (data-flow summary) in the same folder.
- Optional (after BO): `critical_params_post_bo.md` or updated `critical_params.yaml` with post-BO refinement notes.

## Verify Prompt

```
1. Is there a full inventory listing all available parameters (stage, name, type, source, value, tunable/fixed)?
2. Does the critical artifact list selected parameters with bounds and stage?
3. Are excluded parameters listed with a short reason?
4. Are safe fixed params stated?
5. Is the selection rule (challenge + critical path) documented?
6. If BO has run: is post-BO refinement documented or explicitly deferred?
```

## Verify Script

```bash
python methods/plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 04
```
