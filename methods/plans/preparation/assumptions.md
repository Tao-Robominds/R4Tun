# 01 Assumptions

## Goal
Produce baseline (common solution scope + default assumptions) and a single assumptions artifact.

## Common solution path
`methods/plans/` — common solution docs and code references (e.g. `sam4tun/` pipeline).

## Step output path
`data/{tunnel_id}/workflow/{run_id}/01_assumptions/`

## Outputs
- `baseline.md` — solution family, scope, default tunnel assumptions, observed gaps.
- `assumptions.md` — explicit and implicit assumptions with code evidence (or reference to canonical output).

Canonical assumptions output (template/reference):
- `methods/plans/output/01_assumptions_output.md`

## Inputs
- Common solution docs and code (e.g. `sam4tun/*.py`)
- Baseline outputs and known scenario list (when run per tunnel)

## Actions
1. Record solution family and scope.
2. Record default tunnel assumptions.
3. Record observed gaps on the target tunnel.
4. List explicit and implicit assumptions; attach code evidence per assumption.

## Verify Prompt
`Does the step artifact state scope, assumptions, and gaps with evidence? Does each assumption have a rule, evidence, and impact tag where applicable?`

## Verify Script
```bash
python plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 01
```
