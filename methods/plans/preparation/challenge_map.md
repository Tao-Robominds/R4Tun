# 02 Challenge Map

## Goal
Produce `challenge_map.md`.

## Step output path
`data/{tunnel_id}/workflow/{run_id}/02_challenge_map/`

## Outputs
- `challenge_map.md` — assumption challenge map with evidence, bottleneck class, failure mode, and response.

Canonical challenge output (template/reference):
- `methods/plans/output/02_challenge_map_output.md`

## Inputs
- Assumptions artifact
- Target tunnel observations
- Ground-truth evidence

## Actions
1. Compare the assumptions artifact against ground-truth evidence and target-tunnel observations.
2. Mark each assumption as stable or broken.
3. Record the concrete challenge behind each broken assumption.
4. Classify each broken case as `bug|parameter|structural`.
5. State failure mode and required response.

## Verify Prompt
`Does the challenge map mark assumptions as stable or broken, and for each broken assumption record evidence, bottleneck class, failure mode, and response?`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 02
```
