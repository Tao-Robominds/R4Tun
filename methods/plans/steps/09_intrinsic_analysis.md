# 09 Intrinsic Analysis

## Goal
Produce `intrinsics.md`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/09_intrinsic_analysis/intrinsics.md`

## Inputs
- BO logs
- pipeline outputs

## Actions
1. Extract intrinsic metrics.
2. Build metric bank.
3. Set ranges.
4. Set guardrails and failure signatures.
5. Write back reusable knowledge.

## Outputs
- `intrinsics.md`
- `metric_bank.json`

## Verify Prompt
`Does the intrinsic artifact define metric bank, selected metrics, ranges, guardrails, failure signatures, and knowledge write-back?`

## Support Templates
- `plans/templates/intrinsics.md.template`
- `plans/templates/metric_bank.json.template`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 09
```
