# 07 GT Warm Start

## Goal
Produce `warm_start.yaml`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/07_gt_warm_start/warm_start.yaml`

## Inputs
- critical param set
- ground truth analysis

## Actions
1. Reverse-engineer GT optima.
2. Split fixed values vs BO warm start.
3. Set search bounds and priors.
4. Record search-space design.

## Outputs
- `warm_start.yaml`

## Verify Prompt
`Does warm_start.yaml contain warm start, fixed params, search bounds, priors, and anchor/search-space design?`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 07
```
