# 04 Parameter Inventory

## Goal
Produce `parameters.csv`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/04_parameter_inventory/parameters.csv`

## Inputs
- upgraded solution code

## Actions
1. Extract all tunable params.
2. Extract all fixed params.
3. Record stage, type, source, and current value.

## Outputs
- `parameters.csv`

## Verify Prompt
`Does the inventory cover all parameters with stage, type, source, and current value?`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 04
```
