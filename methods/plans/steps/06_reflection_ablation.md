# 06 Reflection Ablation (Fixed Rules)

## Goal
Evaluate fixed-rule reflection strategies and feature-block choices under equal budget.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/06_reflection_ablation/`

## Inputs
- Tuning memory (step 03)
- Guardrails + feature bank (step 04)
- Proxy + calibration (step 05)

## Fixed reflection rules
- poor ring boundary quality -> rerun boundary detection
- poor oblique line quality -> adjust K-line detection
- invalid segment count -> adjust geometry segmentation
- high spacing irregularity -> rerun ring boundary detection

## Ablation grid
- Feature blocks: `P only`, `O only`, `P union O`
- Trigger policy: `none`, `guardrails only`, `p_good only`, `guardrails + p_good`

All cells use the same reflection budget (e.g. one correction pass per ring).

## Outputs
- `ablation_runs.csv`
- `reflection_traces.json`
- `ablation_table.md`

## Verify Prompt
`Do all 12 cells use fixed routing rules, identical budget, and no LLM/RL/adaptive routing?`
