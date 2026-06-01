# Step 3 — BO candidate generation (fixed budget, honest v3)

BO is a **design-time experience generator** (not the deployment optimiser). Target pool: **480 candidates** — 240 hard sparse + 240 representative.

**Corpus:** `data/bo_calibration/` · **Search space:** Step 2 · **Sandbox:** `logs/bo_experience_v3/`

**Policy:** GT labels mIoU only. No `gt_layout` oracle trials. Every trial runs `direction_select` (plus/minus). Warm-start = geometric priors + random only.

---

## Status — complete (v3 honest rerun)

All 6 rings under honest experience mode (20% warm / 60% GP-BO / 20% forced perturbation).

| Slot | Ring | Target | Collected | Best mIoU | Regret | Oracle trials | Gate |
|------|------|-------:|----------:|----------:|-------:|:-------------:|:----:|
| Dense 6-block | 1-5/r271 | 60 | 60 | 0.377 | 0.518 | 0 | pass |
| Medium 6-block | 1-1/r20 | 60 | 60 | 0.533 | 0.349 | 0 | pass |
| Sparse 6-block | 1-4/r206 | 120 | 120 | 0.379 | 0.556 | 0 | pass |
| Medium 7-block | 5-5/r258 | 60 | 60 | 0.307 | 0.567 | 0 | pass |
| Sparse 7-block | 4-6/r283 | 120 | 120 | 0.299 | 0.569 | 0 | pass |
| Partial / irregular | 4-1/r116 | 60 | 60 | 0.380 | 0.525 | 0 | pass |

| | Target | Collected |
|---|-------:|----------:|
| Hard sparse (2 × 120) | 240 | **240** |
| Representative (4 × 60) | 240 | **240** |
| **Total** | **480** | **480** |

**6 / 6 experience gates passed · 0 oracle layout trials**

| Stat | v3 (honest) |
|------|------------:|
| Mean best mIoU | 0.379 |
| Mean regret vs ceiling | 0.514 |
| Oracle trial count | **0** |

**Evidence:**
- Panel merge: `logs/bo_experience_v3/bo_trials.csv`
- Panel summary: `logs/bo_experience_v3/panel_summary.csv`, `experience_summary.json`
- Single-instance gate: `logs/bo_experience_v3/single_instance_gate.json` (1-4/r206, 120 evals)
- Honesty gate: `logs/bo_experience_v3/honesty_gate.json`

**Deprecated (deleted):** `logs/bo_experience_v1/`, `logs/bo_experience_v2/` — contained GT-layout oracle trials unsuitable for proxy training.

---

## Per-ring mix

| Phase | Share | n @ 60 | n @ 120 |
|-------|------:|-------:|--------:|
| Warm-start (`geometric_*`, `intrinsic_r_otsu`, `random`) | 20% | 12 | 24 |
| GP-BO | 60% | 36 | 72 |
| Forced perturbation (anchored on best warm, never GT) | 20% | 12 | 24 |

Warm-start kinds: `geometric_0`, `geometric_1`, `intrinsic_r_otsu`, `random`.

---

## Run commands (record)

```bash
# Single-instance gate
./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo_calibration/MANIFEST.json \
  --source-dir data/bo_calibration \
  --run-root logs/bo_experience_v3 \
  --only-ring 1-4/r206

# Full panel (or --skip 1-4/r206 after gate)
./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo_calibration/MANIFEST.json \
  --source-dir data/bo_calibration \
  --run-root logs/bo_experience_v3

# Honesty validation
./venv/bin/python bo/check_experience_honesty_gate.py \
  --run-root logs/bo_experience_v3
```

Per-ring budgets from manifest `diversity_slot` (sparse → 120, else → 60).

---

## Outputs

Per ring: `bo_trials.csv`, `search_space.json`, `best_bo_trial.json`, `ceiling.json`, `experience_gate.json`  
Panel merge: `logs/bo_experience_v3/bo_trials.csv`

Step 4 calibration records should use **v3 only**; do not merge v1/v2.

---

## v4 — SAM4Tun static prior warm-start (complete)

**Sandbox:** `logs/bo_experience_v4_sam4tun_prior/` · **Prior build:** `logs/sam4tun_prior_v1/`  
**Policy:** Replace `geometric_0/1` with resolution-aligned **`sam4tun_static`** (line K + geometric tiling + `_default_irregular` layout tail + `r_lo` radial filter). Keep `intrinsic_r_otsu` + random fill. Still **0 oracle** trials.

| Stat | v3 | v4 | lift |
|------|---:|---:|-----:|
| Mean best BO mIoU | 0.379 | 0.438 | +0.059 |
| Mean warm-start mIoU | 0.125 | 0.273 | +0.148 |
| Mean regret vs ceiling | 0.514 | 0.455 | −0.059 |
| Oracle trials | 0 | 0 | — |

**6-seg stratum:** warm-start flat (−0.005 mean); best-BO mixed (r20 +0.22, r271/r206 down).  
**7-seg stratum:** warm-start +0.301 mean (r283 SAM4Tun static **0.806** mIoU); best-BO +0.016 mean.

**Evidence:** `logs/bo_experience_v4_sam4tun_prior/vs_v3_summary.md`, `sam4tun_prior_gate.json`, `single_instance_gate.json` (1-4/r206).

```bash
# Phase A — build priors
./venv/bin/python bo/build_sam4tun_prior.py \
  --source-dir data/bo_calibration \
  --run-root logs/sam4tun_prior_v1 --smoke-eval

# Phase C — experience panel
./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo_calibration/MANIFEST.json \
  --source-dir data/bo_calibration \
  --run-root logs/bo_experience_v4_sam4tun_prior \
  --prior-root logs/sam4tun_prior_v1

# Compare vs v3
./venv/bin/python bo/compare_experience_runs.py
```

Warm-start kinds (v4): `sam4tun_static`, `intrinsic_r_otsu`, `random`.

---

## v5 — GT-derived anchor (complete)

**Sandbox:** `logs/bo_experience_v5_gt_derived/`  
**Policy:** GT `k_y` + A/B offsets from corpus `gt_layout.json` as warm-start anchor (`gt_layout_ceiling_r`, `gt_layout_otsu_r`); perturbations anchored on GT ceiling-r layout; 20/60/20 phase mix unchanged. **Not** honest v3 (oracle warm trials allowed). Separate gate: `bo/check_gt_experience_gate.py`.

| Stat | v3 | v4 | v5 |
|------|---:|---:|---:|
| Mean best BO mIoU | 0.379 | 0.438 | 0.833 |
| Mean warm-start mIoU (anchor kind) | 0.125 | 0.273 | 0.783 |
| Mean regret vs ceiling | 0.514 | 0.455 | 0.060 |
| gt_layout oracle trials | 0 | 0 | 12 |

**6 / 6 rings collected · 5 / 6 GT warm mIoU gate pass** (`1-5/r271` GT warm mIoU 0.370 — verify also fails vs ceiling 0.895; pipeline/corpus mismatch under frozen agents).

**Evidence:**
- Panel merge: `logs/bo_experience_v5_gt_derived/bo_trials.csv` (480 trials)
- Single-instance gate: `logs/bo_experience_v5_gt_derived/single_instance_gate.json` (1-4/r206, passed)
- GT gate: `logs/bo_experience_v5_gt_derived/gt_experience_gate.json` (5/6 rings)
- 3-way compare: `logs/bo_experience_v5_gt_derived/vs_v3_v4_v5_summary.md`

```bash
INTRINSIC_PARAMS_BASE_DIR_ONLY=1 ./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo_calibration/MANIFEST.json \
  --source-dir data/bo_calibration \
  --run-root logs/bo_experience_v5_gt_derived \
  --warm-anchor gt_derived

./venv/bin/python bo/check_gt_experience_gate.py --run-root logs/bo_experience_v5_gt_derived

./venv/bin/python bo/compare_experience_runs.py --gt-derived logs/bo_experience_v5_gt_derived
```

Warm-start kinds (v5): `gt_layout_ceiling_r`, `gt_layout_otsu_r`, `intrinsic_r_otsu`, `random`.
