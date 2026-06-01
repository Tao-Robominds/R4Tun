# Stage A proxy scoring — panel ablation

**Sandbox:** `logs/stage_a_score_v1/`  
**Date:** 2026-06-01  
**Status:** G1 + panel **complete**

## Commands

```bash
# Single-instance gate (required before panel)
./venv/bin/python bo/run_held_out_score.py \
  --only-ring 1-1/r19 --gate \
  --candidates-root logs/stage_a_candidates_v1 \
  --score-root logs/stage_a_score_v1

# Stage A panel (25 rings)
./venv/bin/python bo/run_held_out_score.py \
  --split stage_a_proxy_select \
  --candidates-root logs/stage_a_candidates_v1 \
  --score-root logs/stage_a_score_v1
```

## Scoring formula

`final_score = proxy_score + anchor_plausibility_bonus - failure_penalty - baseline_regression_risk`

- **proxy_score:** frozen Ridge **p11** (primary) or **A3-slim** (ablation)
- **guardrails:** v3 failure memory, rho_K/AB plausibility, abstain to C0 (δ=0.02)

## Gate G1 — `1-1/r19`

| Criterion | Result |
|-----------|--------|
| 18 candidates evaluated | pass |
| p11 features finite | pass |
| Composite selector | pass |
| Abstention exercised | pass |

**Evidence:** `logs/stage_a_score_v1/single_instance_gate.json`

## Panel summary (25 rings)

| Metric | p11 | A3-slim |
|--------|----:|--------:|
| Mean selected mIoU | 0.311 | 0.311 |
| Mean C0 mIoU | 0.311 | 0.311 |
| Mean oracle mIoU | 0.357 | 0.357 |
| Abstain rate | 96% | 96% |
| Regression rate vs C0 | 0% | 0% |

Both variants selected **identical** candidates on all 25 rings (conservative abstention dominates). Zero accepted regressions vs SAM4Tun baseline.

**Per-ring table:** `logs/stage_a_score_v1/stage_a_score_summary.csv`  
**Panel JSON:** `logs/stage_a_score_v1/stage_a_score_panel.json`

## Code map

| Module | Role |
|--------|------|
| `bo/lib/candidate_eval.py` | evaluate_trial + feature extraction |
| `bo/lib/stage_a_score.py` | composite score + abstention |
| `bo/lib/proxy_a3_v5.py` | frozen p11 / A3-slim predict |
| `bo/lib/v5_proxy_features.py` | v5/seg observables |
| `bo/run_held_out_score.py` | CLI runner |

## Frozen proxies

| Variant | Manifest |
|---------|----------|
| p11 | `logs/bo_v5_proxy_v1/PROXY_P11_MANIFEST.json` |
| A3-slim | `logs/bo_feature_enrichment_v1/PROXY_A3_SLIM_MANIFEST.json` |
