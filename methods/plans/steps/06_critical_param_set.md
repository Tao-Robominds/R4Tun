# 06 Critical Param Set

## Goal
Produce `critical_params.yaml`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/06_critical_param_set/critical_params.yaml`

## Inputs
- challenge map
- data-flow graph
- parameter inventory

## Actions
1. Apply selection rule from challenged assumptions + stage criticality + data-flow.
2. Record selected params.
3. Record excluded params and reason.
4. Record safe fixed params.

## Outputs
- `critical_params.yaml`

## Verify Prompt
`Does the critical set state the rule, selected params, excluded params, safe fixed params, and rationale?`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 06
```
