# 07 Generalisation Test

## Goal
Measure held-out transfer of memory, features, proxy, and fixed-rule reflection.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/07_generalisation_test/`

## Inputs
- Trained artifacts from steps 03 to 06
- Held-out rings/tunnels not used in BO/proxy/threshold fitting

## Isolation constraints
Held-out set must not be used for:
- BO optimisation
- proxy fitting or calibration
- guardrail threshold selection
- memory construction

## Metrics
- mIoU
- OA
- boundary accuracy
- K-block accuracy
- segment-count accuracy
- proxy prediction quality
- reflection success rate
- reflection count
- runtime

## Outputs
- `generalisation_results.csv`
- `final_comparison.md`

## Verify Prompt
`Is split isolation respected and are all baseline/memory/trigger variants evaluated on the same held-out set?`
