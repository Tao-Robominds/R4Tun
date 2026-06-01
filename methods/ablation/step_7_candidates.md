# Step 7 — Stage A candidate generation (18 per ring)

**Sandbox:** `logs/stage_a_candidates_v1/`  
**Date:** 2026-06-01  
**Status:** G1 + G2 **passed**

## Commands

```bash
# Phase 0 — descriptors + split
./venv/bin/python bo/build_held_out_descriptors.py --out-dir logs/stage_a_candidates_v1
./venv/bin/python bo/build_stage_split.py --out-dir logs/stage_a_candidates_v1

# Single-instance gate (required before panel)
./venv/bin/python bo/run_held_out_candidates.py \
  --only-ring 1-1/r19 --gate \
  --run-root logs/stage_a_candidates_v1

# Stage A panel (25 rings)
./venv/bin/python bo/run_held_out_candidates.py \
  --split stage_a_proxy_select \
  --run-root logs/stage_a_candidates_v1
```

## Experience retrieval (cross-ring)

Held-out rings map to nearest calib diversity slot (6 rings), then top-k retrieval per pool:

- **v4:** SAM4Tun correction deltas (`proposal_templates_sam4tun.csv`)
- **v5:** good-form P10–P90 bands only (`proposal_good_form_gt_derived.csv`) — no GT positions
- **v3:** failure reject/penalise (`failure_memory_random.csv` + rules JSON)

Frozen z-score stats: `logs/stage_a_candidates_v1/retrieval_norm_stats.json`

## 18-candidate allocation

| Type | Count |
|------|------:|
| sam4tun_baseline | 1 |
| sam_plus_delta | 4 |
| line_derived | 4 |
| hybrid_sam_line | 4 |
| gt_form_template | 3 |
| diversity_explore | 2 |

## Gate G1 — single instance (`1-1/r19`)

| Criterion | Result |
|-----------|--------|
| Pool size 18 | pass |
| C0 present | pass |
| Type mix | pass |
| Failure filter active | pass (4 rejections logged) |
| Structural bounds | pass |
| No GT injection | pass |

**Evidence:** `logs/stage_a_candidates_v1/single_instance_gate.json`

## Gate G2 — Stage A panel (25 rings)

| Metric | Value |
|--------|------:|
| Rings processed | 25 |
| All pools size 18 | 25/25 |
| Panel summary | `logs/stage_a_candidates_v1/stage_a_candidate_pools_summary.csv` |

**Next:** proxy argmax selection (frozen p11) — out of scope for this step.

## Code map

| Module | Role |
|--------|------|
| `bo/lib/held_out_descriptors.py` | Ring descriptors for coarse match |
| `bo/lib/stage_split.py` | 25/25 stratified split |
| `bo/lib/experience_retrieval.py` | Query + two-stage retrieval |
| `bo/lib/line_reliability.py` | rho_K, rho_AB |
| `bo/lib/line_anchor.py` | K_line / AB_line |
| `bo/lib/candidate_bounds.py` | Structural validation + failure filter |
| `bo/lib/candidate_generator.py` | 18-type pool assembly |
| `bo/run_held_out_candidates.py` | CLI runner + gates |
