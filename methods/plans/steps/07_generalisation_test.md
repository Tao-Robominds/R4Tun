# 07 Generalisation Test

## Goal
Measure held-out transfer of stage-wise BO experience, fixed intrinsic proxies, thresholds, and reflection triggers.

## Runtime Path
Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/07_generalisation_test/`

## Inputs
- Artifacts from steps 03 to 06.
- Held-out rings/tunnels not used in BO, threshold fitting, or proxy validation decisions.

## Isolation constraints
Held-out set must not be used for:
- BO optimisation
- fixed proxy definition changes after thresholds are selected
- threshold selection
- memory construction
- false-negative-driven proxy repairs

## Metrics
- mIoU
- OA
- boundary accuracy
- K-block accuracy
- segment-count accuracy
- `S_depth`, `S_boundary`
- trigger rate
- bad-case recall
- trigger precision
- false-negative rate
- accepted-case mIoU
- reflection count
- runtime

## Outputs
- `generalisation_results.csv`
- `final_comparison.md`

## Verify Prompt
`Is split isolation respected, and do fixed low-proxy triggers catch held-out low-mIoU cases without using GT at deployment time?`
