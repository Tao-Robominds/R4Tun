# 06 Reflection Trigger Validation

## Goal
Validate the fixed intrinsic proxies as deployment-time reflection triggers. The goal is to catch bad final outputs when GT is unavailable, not to prove a full mIoU predictor.

## Runtime Path
Sandbox path: `logs/{run_id}/{tunnel_id}/r{ring_id}/06_reflection_ablation/`

## Inputs
- Trial dataset and tuning memory (step 03).
- Fixed proxy scores (step 04).
- Thresholds from step 05.
- Final mIoU labels for validation.

## Fixed reflection rules
- `if S_depth < T_depth`: trigger preprocessing reflection.
- `if S_boundary < T_boundary`: trigger detection/boundary reflection.
- Optional component-level reason tags explain the trigger, but routing remains fixed.

## Evaluation quantities
- Bad-case recall: among `mIoU < tau` cases, how many did the proxy catch?
- Trigger precision: among triggered cases, how many were actually bad?
- False-negative rate: bad cases missed by the proxy.
- Accepted-case mIoU: average final mIoU when the proxy says accept.

Bad-case recall is the primary deployment metric. False negatives are the main risk.

## Actions
1. Apply fixed thresholds to validation samples.
2. Evaluate `S_depth` and `S_boundary` triggers separately and jointly.
3. Report the four quantities above for each trigger.
4. Inspect false negatives and update the proxy definition only if failures are systematic.
5. Keep leave-one-out ablation, Ridge, and logistic calibration as optional appendix analyses only.

## Outputs
- `trigger_validation.csv`
- `reflection_traces.json`
- `trigger_validation_report.md`

## Verify Prompt
`Do low S_depth or S_boundary values catch most low-mIoU cases under fixed rules, with false negatives explicitly reported?`
