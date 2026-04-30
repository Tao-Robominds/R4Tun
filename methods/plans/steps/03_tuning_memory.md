# 03 Tuning Memory

## Goal
Convert BO logs into a compact memory bank of informative cases.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/03_tuning_memory/`

## Inputs
- BO outputs from step 02

## Actions
1. Rank runs by informativeness and retain:
   - best-performing cases
   - typical failure cases
   - successful correction cases
   - regime-specific patterns
2. Store each selected case as:
   `ring_condition -> parameter_choice -> outcome -> short_explanation`
3. Remove near-duplicate entries that do not add new information.

## Outputs
- `tuning_memory.json`
- `memory_selection_report.md`

## Verify Prompt
`Does memory preserve representative success/failure/correction patterns across regimes without dumping all BO trials?`
