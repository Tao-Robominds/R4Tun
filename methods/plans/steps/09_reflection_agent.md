# 09 Reflection Agent

## Goal
Produce `policy_prompt.md`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/09_reflection_agent/policy_prompt.md`

## Inputs
- guardrails
- proxy
- prior cases
- BO history
- assumption checklist

## Actions
1. Write prompt-first reflection policy.
2. List inputs.
3. Define decision rules for:
   - guardrails fail hard
   - proxy low and guardrails pass
   - proxy high and uncertainty high
   - proxy and guardrails disagree
4. List actions and fallback conditions.

## Outputs
- `policy_prompt.md`

## Verify Prompt
`Does the reflection policy define inputs, case rules, actions, fallback conditions, and uncertainty-aware decisions?`

## Verify Script
```bash
python methods/plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 09
```
