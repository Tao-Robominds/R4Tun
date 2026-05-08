# 03 Trial Dataset and Tuning Memory

## Goal
Convert stage-wise BO logs into a compact validation dataset and memory bank of informative cases. This step defines which trials are used for fixed proxy validation; it does not train a full mIoU predictor.

## Runtime Path
Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/03_tuning_memory/`

## Inputs
- Stage BO outputs from step 02.
- Final mIoU after downstream reprojection for selected or trial outputs.

## Actions
1. Build `stage_trial_dataset.csv` with one row per selected/trial output:
   - `stage`, `tunnel_id`, `ring_id`, `trial_id`
   - artifact paths
   - final mIoU after reprojection
   - raw intrinsic diagnostics needed for `S_depth` and `S_boundary`
2. Rank runs by informativeness and retain:
   - best-performing cases
   - typical failure cases
   - successful correction cases
   - regime-specific patterns
3. Store each selected memory case as:
   `ring_condition -> parameter_choice -> outcome -> short_explanation`
4. Remove near-duplicate entries that do not add new information.
5. Keep threshold-selection samples and held-out samples explicitly separated.

## Outputs
- `stage_trial_dataset.csv`
- `tuning_memory.json`
- `memory_selection_report.md`

## Verify Prompt
`Does the trial dataset contain final mIoU and intrinsic diagnostics for proxy validation, while memory preserves representative success/failure/correction patterns without dumping all BO trials?`
