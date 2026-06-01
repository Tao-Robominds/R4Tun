# Step 7-prep — Held-out evaluation infrastructure

**Sandbox:** `logs/held_out_eval_v1/`  
**Date:** 2026-05-31  
**Status:** prep complete; single-instance gate **passed**

## Summary

Built the held-out 50-ring evaluation panel infrastructure: ring descriptors, v7 stratified 25/25 split, calib-bucket anchor index, and **p24_experience** candidate sampler (6-block). Validated on representative Stage A ring **1-1/r19** before panel-scale Stage A runs.

**Design rationale:** Reuse v7 stratified 25/25 (hard seg×dia quotas + soft tier balance) rather than random 25/25. **p24_experience** replaces p18's naive LHS-only explore with Step 3 forced low/mid perturbations to test whether frozen **A3-slim** mis-picks plausible-but-bad layouts.

## 1. Ring descriptors

**Command:** `./venv/bin/python bo/build_held_out_descriptors.py --out-dir logs/held_out_eval_v1`

**Input:** `data/held-out/_manifests/data_v6_50ring_calibration_panel.csv` (read-only `data/held-out/`)

**Output:** `logs/held_out_eval_v1/ring_descriptors.csv`, `ring_descriptors.meta.json`

Paper marginals match targets (dense 13 / medium 18 / low 19; k_span narrow 3 / normal 22 / wide 25; pattern canonical 26 / reversed 24; coverage full 48 / partial 2).

## 2. Stratified 25/25 split

**Command:** `./venv/bin/python bo/build_stage_split.py --out-dir logs/held_out_eval_v1`

**Seed:** `20260529` (v7 convention)

**Hard quotas (per side):**

| (seg, dia) | Stage A | Stage B |
|------------|--------:|--------:|
| (6, 5.5)   | 10      | 10      |
| (6, 5.8)   | 5       | 5       |
| (7, 7.5)   | 10      | 10      |

**Outputs:**

- `stage_split_manifest.json` — `stage_a_proxy_select` (25), `stage_b_refinement_verify` (25)
- `split_balance_report.json` — **gate passed=true**
- `deploy_ring_manifest.csv`

Soft-balance checks (density/k_span/pattern/coverage tiers, proportion drift ≤8pp) all pass.

## 3. Calib anchor index

**Command:** `./venv/bin/python bo/build_calib_anchor_index.py --out-dir logs/held_out_eval_v1`

**Sources:** `logs/bo_experience_v1/bo_trials.csv` + `data/bo_calibration/` (read-only)

**Output:** `calib_anchor_index.json` — 3 BO-best buckets:

| Bucket | audit_case_id | gt_miou |
|--------|---------------|--------:|
| 6_5.5  | 1-4/r206      | 0.917   |
| 7_7.5  | 4-1/r116      | 0.903   |
| 7_7.4  | 1-5/r271      | 0.877   |

Fallback: nearest same-segment bucket → K-small geometric prior.

## 4. Candidate pool profiles

Implemented in `bo/lib/held_out_sampler.py`:

| Profile | Layouts | Candidates | Use |
|---------|--------:|-----------:|-----|
| p18 | 9 | 18 | Legacy |
| **p24_experience** | 12 | 24 | **6-block Stage A (default)** |
| p36 | 18 | 36 | 7-block rings |

**p24_experience layout mix (12 layouts × 2 order branches):**

- 1× static + 5× dense_bo_informed (σ=0.08 around calib anchor)
- 2× geometric priors (equal-width + K-small)
- 2× forced perturb: `perturb_wrong_k`, `perturb_guardrail_smoke`
- 1× mid perturb: `perturb_misaligned`
- 1× LHS explore

Auto profile: `p24_experience` if 6-block, else `p36`.

## 5. Single-instance gate (required before panel)

**Ceiling contract (2026-05-31 fix):** `run_ring` now calls `ensure_ceiling_reference()` so every candidate uses GT-derived `r_surface_min` (same as calib BO). Without this, all gt mIoU was computed at `r_surface_min=0` and looked artificially ~0.10–0.23.

**Command:** `./venv/bin/python bo/run_held_out_single_gate.py --run-root logs/held_out_eval_v1`

| Field | Value |
|-------|-------|
| Case | **1-1/r19** (first Stage A ring; 6-block, dia 5.5) |
| Profile | p24_experience |
| `r_surface_min_fixed` | **2.732** (was 0.0 pre-fix) |
| ceiling mIoU | **0.894** |
| Pool oracle (in-pool max) | **0.287** |
| Pool std(gt_miou) | **0.056** (≥ 0.03) |
| **Gate** | **passed** |

**A3-slim selection smoke (2 six-block rings, post-fix):**

| Ring | ceiling | pool max | proxy pick | proxy rank | top-3 hit |
|------|--------:|---------:|-----------:|-----------:|:---------:|
| 1-1/r19 | 0.894 | 0.287 | 0.121 (static) | 20/24 | no |
| 2-1/r64 | 0.944 | 0.262 | 0.125 (static) | 18/24 | no |

Interpretation: p24 spans bad–medium–good **within the pool** (std ≥ 0.03), but pool “good” (~0.26–0.29) is still far below ring ceiling (~0.89–0.94) because layouts are anchor-perturbed, not GT-near. On these two rings A3-slim argmax picks the static prior (worst tier), not pool-best — Stage A panel (25 rings) needed for pooled proxy generality stats.

**Evidence:** `logs/held_out_eval_v1/single_instance_gate.json`, `gate_1-1_r19.csv`, `gate_2-1_r64.csv`

## 6. Next steps (Step 7b)

Panel-scale Stage A (after gate):

```bash
./venv/bin/python bo/run_held_out_candidate_pool.py \
  --split stage_a_proxy_select --profile auto \
  --run-root logs/held_out_eval_v1

./venv/bin/python bo/score_held_out_pool.py \
  --split stage_a_proxy_select \
  --run-root logs/held_out_eval_v1
```

Stage B (A6 refinement, 25 disjoint rings) deferred to Step 7c.

## Artifacts map

| Artifact | Path |
|----------|------|
| Descriptors | `logs/held_out_eval_v1/ring_descriptors.csv` |
| Split manifest | `logs/held_out_eval_v1/stage_split_manifest.json` |
| Balance gate | `logs/held_out_eval_v1/split_balance_report.json` |
| Anchor index | `logs/held_out_eval_v1/calib_anchor_index.json` |
| Single-instance gate | `logs/held_out_eval_v1/single_instance_gate.json` |
| Locked proxy | `logs/bo_feature_enrichment_v1/PROXY_A3_SLIM_MANIFEST.json` |

## Code added

- `bo/lib/held_out_common.py`, `held_out_descriptors.py`, `stage_split.py`, `calib_anchor.py`, `held_out_sampler.py`, `held_out_pool.py`, `held_out_score.py`
- CLIs: `bo/build_held_out_descriptors.py`, `build_stage_split.py`, `build_calib_anchor_index.py`, `run_held_out_candidate_pool.py`, `score_held_out_pool.py`, `run_held_out_single_gate.py`
- `bo/ablation.md` — Step 7-prep / 7a / 7b / 7c + Section C alignment
