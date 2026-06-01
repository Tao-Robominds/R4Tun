# Journal 2026-05-29: Detection layout BO on `data/bo/` (6 calibration rings)

Plan: `.cursor/plans/calib_detection_bo_479ad1fa.plan.md`

---

## Summary

- **384 trials** across 6 BO calibration rings (`data/bo/`), **64 GP-BO + EI evals** per ring.
- **Co-search:** `k_y_positions` + `per_ring_offsets` (A/B arc widths) + `r_surface_min`.
- **Goal:** Step 02 **experience collection** for proxy training — not per-ring reachability proof (capability already on `1-1/r18`, regret 0.059).
- **6/6** rings passed the **pipeline experience gate** (64 trials logged, mIoU spread > 0.02, BO best > random-phase median).
- **Preprocessing frozen** from read-only `data/bo/`; all writes under `logs/calib_detection_bo_v1/`.
- **Next step:** proxy training (Step 03) on `bo_trials.csv` — no additional BO evals required unless selection validation fails.

---

## 1. Detection-stage critical parameters

### Must-have (fixed per ring, not searched)

| Parameter | Stage | Role |
|---|---|---|
| `segment_count` | Detection | 7-block ontology (from GT in `enhanced.csv`) |
| `enabled_blocks` | Detection | Canonical block order K, B1, A1–A4, B2 |
| `k_anchor_semantics` | Detection | `"boundary_start"` (layout injection mode) |
| Frozen preprocessing | Preprocessing | Depth maps from `data/bo/` (3b ceiling ≥ 0.85) |

### Tunable (BO search space)

| Parameter | Stage | Encoding | Role |
|---|---|---|---|
| `k_y_positions[0]` | Detection | `x[0] ∈ [0,1]` → `k_y = x[0] × H` | K-anchor Y on circular axis |
| `per_ring_offsets["0"][block]` | Detection | `x[1:8]` normalized arc widths → cumulative offsets | A/B block placement |
| `r_surface_min` | Segmentation | `x[8] ∈ [0,1]` → `[P1(r), P60(r)]` from `enhanced.csv` | Groove/BG radial rejection |

**Search dimension:** 9 (7-block rings). Block cyclic order **fixed canonical**; only K position, arc widths, and radial cutoff searched.

**Sandbox rule:** `INTRINSIC_PARAMS_BASE_DIR_ONLY=1`; layout injected via sandbox `parameters_detection.json` + `parameters_segmentation.json` (bypasses Hough K-detection).

### BO engine (paper-aligned)

| Component | Spec |
|---|---|
| Surrogate | `GaussianProcessRegressor`, Matérn ν=2.5, ARD lengthscales |
| Acquisition | Expected Improvement on 4096 random candidates in `[0,1]^9` |
| Warm-start | 2 geometric priors + 16 random trials, then GP acquisitions to 64 total |
| Design-time label | GT mIoU from `final.csv` (audit only; not a pass/fail gate) |

### Parameter sensitivity (GP ARD)

All three encoded groups hit the ARD lower bound on pooled 384 trials → **joint sensitivity** to `k_y_frac`, `r_surface_min_frac`, and `arc_width_entropy`. Per-ring ARD consistent across all 6 rings.

Evidence: `logs/calib_detection_bo_v1/parameter_sensitivity.md`

### Per-ring BO-best parameters (audit)

| Ring | Ceiling (ref) | Best BO mIoU | Regret | Best `k_y` | Best `r_surface_min` |
|---|---:|---:|---:|---:|---:|
| 4-9/r365 | 0.913 | 0.275 | 0.638 | 1522 | 3.28 |
| 4-1/r116 | 0.905 | 0.268 | 0.637 | 3701 | 3.13 |
| 4-8/r336 | 0.878 | 0.249 | 0.629 | 2871 | 2.20 |
| 5-5/r253 | 0.886 | 0.280 | 0.606 | 3565 | 3.53 |
| 4-7/r309 | 0.878 | **0.741** | 0.137 | 1509 | 3.72 |
| 5-7/r323 | 0.884 | **0.737** | 0.147 | 3396 | 3.73 |

**mIoU range (all trials):** 0.031 – 0.741. Rings 4-7/r309 and 5-7/r323 approach ceiling under co-search; four others remain hard from geometric priors alone.

---

## 2. Valid intrinsic quality metrics and guardrails

### Layer 3a — detection/layout QA (no GT at runtime)

**Agent-native metrics** (`agents/2_detection/scripts/extract_intrinsics.py`):

| Metric | Documented guardrail | BO panel behaviour |
|---|---|---|
| `det_k_count_match` | == `ring_count` | Constant (always true under `k_y_positions` override) |
| `det_block_count_per_ring` | == 7 | Constant |
| `det_y_coverage_pct` | 85–115% | Constant at 100% (injected full ring coverage) |
| `det_ready_for_segmentation` | composite pass | Mostly constant |

**Layout-derived observables (vary across trials, GT-free):**

| Metric | Role |
|---|---|
| `k_y_frac` | Normalized K position |
| `arc_width_entropy` | Arc-width balance |
| `r_surface_min_frac` | Normalized radial cutoff in search band |
| `n_reclassified_by_r_filter` | Points moved to BG by radial filter (from seg log) |

### Spearman ρ vs GT mIoU (384 pooled trials, design-time only)

| Metric | ρ | p-value | |ρ| ≥ 0.3? |
|---|---:|---:|---|
| `r_surface_min_frac` | −0.21 | 4.2×10⁻⁵ | No |
| `n_reclassified_by_r_filter` | +0.20 | 8.2×10⁻⁵ | No |
| `arc_width_entropy` | +0.18 | 5.7×10⁻⁴ | No |
| `det_min_y_gap_px` | +0.10 | 0.050 | No |
| `det_y_coverage_pct` | −0.05 | 0.29 | No |
| `k_y_frac` | +0.04 | 0.45 | No |

**Interpretation:** Top metrics are **statistically significant** but **weak univariate predictors** (ρ² ≈ 4%). Expected under layout injection + joint parameter effects. Proxy training should use **multivariate** models on `bo_trials.csv`, not single-feature ρ gates.

### Promoted guardrails (runtime QC envelope)

From top-quartile mIoU trials per ring; panel medians in `detection_guardrails.json`:

| Metric | Panel min | Panel max | ρ |
|---|---:|---:|---:|
| `r_surface_min_frac` | 0.10 | 0.68 | −0.21 |
| `n_reclassified_by_r_filter` | 0 | 454 | +0.20 |
| `arc_width_entropy` | 1.69 | 1.90 | +0.18 |
| `det_min_y_gap_px` | 47 | 382 | +0.10 |
| `det_y_coverage_pct` | 100 | 100 | −0.05 |

Per-ring thresholds retained in JSON for ring-adaptive QC where needed.

**Not a detection BO objective:** GT mIoU at deployment. Ceiling mIoU used only as **design-time audit** (`ceiling.json` per ring).

### Layer 3b — capability context (already proven, not re-gated)

| Check | Result | Path |
|---|---|---|
| Agents capability (K + offsets) | regret 0.059 on 1-1/r18 | `logs/aboffset_capability_v1/single_instance_gate.json` |
| Preprocessing ceiling on calib panel | 6/6 ≥ 0.85 | `data/bo/MANIFEST.json` |

---

## 3. BO experience dataset

**Canonical root:**

```
logs/calib_detection_bo_v1/
├── bo_trials.csv              ← 384 rows (proxy training input)
├── panel_summary.csv
├── experience_summary.json
├── parameter_sensitivity.md
├── intrinsic_correlation.csv
├── detection_guardrails.json
├── bo_experience_summary.md
├── sandbox/<tunnel>/r<ring>/  ← per-trial det+seg artifacts
└── <tunnel>/r<ring>/
    ├── bo_trials.csv          ← per-ring copy
    ├── convergence.csv
    ├── best_bo_trial.json
    ├── experience_gate.json
    ├── ceiling.json           ← GT-layout ceiling reference
    └── gt_layout.json
```

### Trial schema (`bo_trials.csv`)

Each row includes:

- `trial_id`, `tunnel_id`, `ring_id`, `case_id`, `kind` (prior | random | bo)
- `k_y`, `per_ring_offsets` (JSON), `r_surface_min`, `search_x` (JSON)
- `gt_miou`, `best_so_far`, `regret_vs_ceiling`
- Layout observables: `k_y_frac`, `arc_width_entropy`, `r_surface_min_frac`, `n_reclassified_by_r_filter`
- Detection intrinsics: `det_*` columns where applicable
- GP metadata (BO trials): `bo_surrogate_mean`, `bo_surrogate_std`, `ei_value`

### Experience gate (per ring)

| Criterion | Threshold |
|---|---|
| Trials completed | 64/64 |
| Trial schema | params + gt_miou + intrinsics populated |
| mIoU spread | std(gt_miou) > 0.02 |
| BO vs random | best_BO > median(prior + random mIoU) |

**6/6 passed.** Regret ≤ 0.10 and best_BO ≥ 0.85 were **not** required (experience collection, not reachability).

---

## Runners

```bash
# Single ring (pipeline proof)
INTRINSIC_PARAMS_BASE_DIR_ONLY=1 ./venv/bin/python bo/validation/run_detection_layout_bo_v1.py \
  --source-dir data/bo --tunnel-id 4-9 --ring-id 365 \
  --run-root logs/calib_detection_bo_v1 --n-evals 64 --seed 7

# Full panel (skip already-run rings)
./venv/bin/python bo/validation/run_detection_layout_bo_panel_v1.py \
  --source-dir data/bo --run-root logs/calib_detection_bo_v1 --n-evals 64 --skip 4-9/r365

# Post-process: sensitivity + guardrails
./venv/bin/python bo/validation/analyze_detection_bo_experience_v1.py \
  --run-root logs/calib_detection_bo_v1
```

Code: `bo/validation/layout_bo_core.py`, `ceiling_gate_core.py`, `run_detection_layout_bo_v1.py`, `run_detection_layout_bo_panel_v1.py`, `analyze_detection_bo_experience_v1.py`.

Knowledge doc: `bo/2_detection/knowledge.md`.

---

## Evidence paths

| Artifact | Path |
|---|---|
| Merged trial log (proxy input) | `logs/calib_detection_bo_v1/bo_trials.csv` |
| Panel summary | `logs/calib_detection_bo_v1/panel_summary.csv` |
| Guardrails | `logs/calib_detection_bo_v1/detection_guardrails.json` |
| Intrinsic correlation | `logs/calib_detection_bo_v1/intrinsic_correlation.csv` |
| Parameter sensitivity | `logs/calib_detection_bo_v1/parameter_sensitivity.md` |
| Experience summary | `logs/calib_detection_bo_v1/bo_experience_summary.md` |
| Aggregate stats | `logs/calib_detection_bo_v1/experience_summary.json` |
| 4-9/r365 gate | `logs/calib_detection_bo_v1/4-9/r365/experience_gate.json` |

---

## Bottom line

1. **Critical parameters confirmed:** `k_y_positions`, `per_ring_offsets`, `r_surface_min` (co-searched).
2. **384 labelled trials** ready for proxy training — sufficient without additional BO evals.
3. **Guardrails promoted** from top-ranked intrinsics; weak univariate ρ expected; validate proxy via **selection lift**, not |ρ| ≥ 0.3 alone.
4. **Next:** Step 03 proxy training on `logs/calib_detection_bo_v1/bo_trials.csv` using multivariate features (`r_surface_min_frac`, `n_reclassified_by_r_filter`, `arc_width_entropy`, `det_min_y_gap_px`, + param encodings).

---

## Proxy training handoff (Step 03)

| Input | Path |
|---|---|
| Trial features + GT mIoU | `logs/calib_detection_bo_v1/bo_trials.csv` |
| Guardrail thresholds | `logs/calib_detection_bo_v1/detection_guardrails.json` |
| Parameter bounds / sensitivity | `logs/calib_detection_bo_v1/parameter_sensitivity.md` |
| Method reference | `methods/plans/steps/03_tuning_memory.md`, `05_proxy_and_calibration.md` |

**Do not** use `data/held-out/` rings for proxy training. Held-out is for Step 07 generalisation only.

**Suggested proxy features (initial):** `k_y_frac`, `r_surface_min_frac`, `arc_width_entropy`, `n_reclassified_by_r_filter`, `det_min_y_gap_px`; optionally normalized arc-width vector or GP trial metadata.

**Validation metric:** top-1 candidate mIoU vs random within each ring's trial pool; rank correlation within ring — not pooled Spearman alone.
