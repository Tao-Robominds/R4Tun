# 08 Proxy Training

## Goal
Produce `proxy.md`.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/08_proxy_training/proxy.md`

## Inputs
- intrinsic dataset
- held-out split

## Actions
1. Train proxy.
2. Validate on holdout.
3. Build confidence bank.
4. Record uncertainty and trust rule.

## Outputs
- `proxy.md`
- `confidence_bank.json`
- `proxy_eval.json`

## Verify Prompt
`Does the proxy artifact include validation, calibration, uncertainty, confidence bank, and trust rule?`

## Support Templates
- `plans/templates/proxy.md.template`
- `plans/templates/confidence_bank.json.template`
- `plans/templates/proxy_eval.json.template`

## Verify Script
```bash
python methods/plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 08
```
