# Daily Journal — 1 May 2026

## Objective

Document ring-native preprocessing, the **B+C+D** context fix (observed-theta crop + tunnel-global coordinates + neighbor-ring context), **guarded Bayesian optimization (BO)** on preprocessing parameters, and outcomes for representative rings. Consolidate methodology so future work does not confuse tunnel-wide vs ring-wise pipelines or IoU-only rewards.

---

## Context & Starting Point

| Topic | Summary |
|--------|---------|
| Ring-native path | Per-ring TXT → local unfolding (PCA) → denoise → enhance → depth map (`agents/1_preprocessing/1_preprocessing.py`). |
| Tunnel-wide reference (`r4tun`) | Whole-tunnel centerline fit → stable `(h, θ, r)`; depth maps often **cropped** to observed θ; visually cleaner than a full 360° canvas. |
| Problem | Ring-only maps showed large empty bands (full-circumference canvas + unstable local frame + no cross-ring support); naive **foreground_mask_iou** BO could reward degenerate sparse maps. |
| Fixed baseline (immutable reference) | **`logs/context_preprocessing_v1/<tunnel>/r<ring>/`** — B+C+D outputs (moved from `data/`); **do not overwrite**. |
| Promoted BO archive | User snapshot: **`data/bo/preprocessing/`** (includes `summary_guarded_bo_v1.json` and selected trial artifacts). **Do not modify.** |

**Intrinsic QA corpus (299 rings):** See `data/preprocessing_qa/report.md` and `corpus_metrics.json` — PASS/WARN/FAIL from intrinsic depth-map metrics only (no GT).

---

## How Ring-Based Preprocessing Works

1. **Input:** Per-ring point cloud (`data/rings/<tunnel>_ring<k>.txt`): columns include `x,y,z,intensity,segment,ring`.
2. **Unfolding (classic ring-native):** PCA-based ring plane → cylindrical-ish `(h, θ, r)` per ring.
3. **Denoising:** Radius masks, gradient/smoothing along θ and `r`; produces `pred` support vs background.
4. **Enhancing:** Curvature-guided upsampling on support; optional outlier interpolation; builds dense `(h,θ,r)` for projection.
5. **Depth map:** Rasterize to grid; ring-native path historically used a **canonical height** (full θ span) so empty θ bins appear as gaps.

**Outputs used downstream:** `depth_map.npy`, `depth_map.png`, `pixel_to_point.pkl`, `denoised.csv` (for GT alignment in BO metrics).

---

## Why Tunnel-Based Context Still Helps (B+C+D)

We keep **ring-level evaluation** (one target ring per run) but inject tunnel-global structure where it matters:

| Variant | Role |
|---------|------|
| **B — observed-theta crop** | Depth map grid matches **actual θ coverage** of points; avoids a hollow “donut” from forcing 360° when data only covers part of the tunnel wall. |
| **C — global `(h,θ,r)`** | Prefer **`r4tun/data/ablation_rules/<tunnel>/unwrapped.csv`** (tunnel centerline); fallback to per-ring `data/<tunnel>/r*/unwrapped.csv`. Stabilizes unfolding vs per-ring PCA. |
| **D — neighbor context** | Concatenate rings **`r−1 … r+1`** for denoise/enhance; **assign** synthetic points to nearest neighbor ring in `(h,θ)` so ownership stays consistent; **metrics and target depth map** use **target ring only**. |

Implementation: `agents/1_preprocessing/context_preprocessing.py`; CLI: `agents/1_preprocessing/scripts/run_context_ring_trial.py`.

---

## Why These BO Metrics (Guarded Reward)

**Design-time use of GT:** Segment labels in `denoised.csv` define a **GT foreground mask** on the depth grid via `pixel_to_point.pkl`. This is for **parameter tuning / BO** on labeled tunnels, not for deployment on new tunnels without labels.

| Quantity | Role |
|----------|------|
| **`target_foreground_recall`** | \( \mathrm{TP} / (\mathrm{TP}+\mathrm{FN}) \) — foreground support in GT should project to valid depth pixels. Primary signal. |
| **`foreground_mask_iou`** | Diagnostic only; optimizing IoU alone encouraged **empty maps** (fewer FP, artificially high IoU). |
| **Coverage guard** | `valid_ratio` must stay ≥ `min_coverage_ratio × baseline_valid_ratio` from fixed `context_preprocessing_v1` reference (default `min_coverage_ratio=0.7`). Soft factor `coverage_factor` ∈ [0,1]. |
| **Empty-band guard** | Largest contiguous empty **row** band vs map height must stay ≤ `max_empty_row_band_ratio` (default `0.45`). Soft factor `empty_factor` ∈ [0,1]. |

**Optimized scalar:**

```text
guarded_score = target_foreground_recall × coverage_factor × empty_factor
```

Code: `bo/preprocessing_iou_metrics.py::compute_target_guarded_metrics`.

---

## BO Setup

| Item | Value |
|------|--------|
| Runner | `bo/run_preprocessing_iou_bo.py` |
| Fixed baseline (read-only) | `--baseline-dir logs/context_preprocessing_v1` |
| Trial outputs | `data/bo/preprocessing/<tunnel>/r<ring>/{baseline,trial_NNN,best}/...` (isolated dirs; **never** write into `logs/context_preprocessing_v1`). If strict immutability for `data/bo/**` is enforced in future, run in `logs/...` and manually promote selected outputs. |
| Logs | `logs/preprocessing_context_bo/<run_id>/<tunnel>/r<ring>/` — `summary.json`, `summary.md`; BO run metadata lives **under `logs/`** next to the baseline tree |
| Parameters | Start from `agents/1_preprocessing/parameters/<tunnel>/r<ring>/parameters_preprocessing.json`; BO perturbs denoise/enhance-related knobs (radii, gradients, target distances, outlier settings, etc.). |
| Representative run id | `context_bcd_representatives_guarded_v1` |
| Pilot run id (`1-1/r25`) | `context_bcd_r25_guarded_v1` |

**Selection rule:** After BO, if **best `guarded_score` > baseline run `guarded_score`**, select **`bo_best`** (`.../best/...`); else keep **`fixed_baseline`** (`logs/context_preprocessing_v1/...`).

---

## BO Results (Representative Rings)

Canonical aggregated summary (also on disk): **`data/bo/preprocessing/summary_guarded_bo_v1.json`**.

| Tunnel / ring | Δ guarded_score | Selected source | Notes |
|-----------------|-----------------|-----------------|-------|
| 5-7 / r315 | +0.003910 | bo_best | Improved under guarded objective. |
| 4-6 / r283 | 0 | fixed_baseline | BO did not beat fixed baseline; sparse coverage / metrics stayed at 0. |
| 4-4 / r215 | +0.042940 | bo_best | Largest relative gain in this set. |
| 1-1 / r25 | 0 | fixed_baseline | Pilot: no improvement; keep fixed B+C+D reference. |
| 5-1 / r114 | 0 | fixed_baseline | Tie on guarded score; baseline kept. |
| 5-6 / r285 | 0 | fixed_baseline | Tie; baseline kept. |
| 5-1 / r116 | +0.126440 | bo_best | Largest absolute Δ in guarded score. |
| 5-1 / r113 | +0.002534 | bo_best | Small but strict improvement. |

**Aggregate:** `bo_best`: **4** rings; `fixed_baseline`: **4** rings.

**Overall interpretation:** Guardrails stopped collapse into empty maps; rings where BO helped did so with recall + structure constraints. Rings with **0** guarded score on both sides indicate GT/projection or extreme sparsity — BO cannot invent signal; context + parameter tuning only helps when the pipeline can still produce supported pixels.

---

## Relation to Intrinsic QA (`data/preprocessing_qa`)

The **299-ring** pass used **intrinsic** metrics (`valid_ratio`, empty bands, etc.) — no GT — to flag FAIL rings for documentation. That is complementary to **GT-guarded BO**: intrinsic QA triages corpus health; BO on representative rings uses GT-derived recall with guardrails for tuning.

---

## Key Insights

1. **Ring-native vs tunnel-wide:** Visual gap was explained by canvas geometry (360° vs observed θ), coordinate stability (PCA vs global centerline), and missing neighbor context — not only “wrong hyperparameters.”
2. **IoU-only BO is unsafe:** Can reward deleting valid foreground on the grid; **recall + coverage + empty-band** guards align optimization with usable maps.
3. **B+C+D is the deployment-shaped baseline** for ring-wise outputs that should resemble tunnel-quality maps without running the full tunnel in production for every ring.
4. **Representative BO is selective:** Half of the eight rings kept the fixed baseline — expected when the objective is conservative or the ring is already near a local optimum.

---

## Files & Artifacts

| Path | Role |
|------|------|
| `agents/1_preprocessing/context_preprocessing.py` | Official B+C+D module. |
| `agents/1_preprocessing/scripts/run_context_ring_trial.py` | CLI wrapper. |
| `bo/run_preprocessing_iou_bo.py` | Guarded preprocessing BO. |
| `bo/preprocessing_iou_metrics.py` | Guarded metrics + IoU diagnostic. |
| `logs/context_preprocessing_v1/**` | Fixed baseline reference (read-only; canonical location after moving out of `data/`). |
| `data/bo/preprocessing/**` | User-promoted archive — **immutable**. |
| `data/preprocessing_qa/report.md`, `corpus_metrics.json` | 299-ring intrinsic QA snapshot. |

---

## Commands for Reproduction

```bash
# B+C+D ring output (example)
./venv/bin/python agents/1_preprocessing/scripts/run_context_ring_trial.py \
  --tunnel-id 1-1 --ring-id 25 \
  --output-root data/bo/preprocessing \
  --reference-base-dir data/ablation/baseline

# Guarded BO (example — writes under data/bo/preprocessing + logs)
./venv/bin/python bo/run_preprocessing_iou_bo.py \
  --tunnel-id 1-1 --ring-id 25 \
  --baseline-dir logs/context_preprocessing_v1 \
  --base-dir data/bo/preprocessing \
  --run-id context_bcd_r25_guarded_v1 \
  --n-calls 6 --n-initial-points 3
```

---

## Repository Maintenance (this journal entry)

- Per-trial JSON logs under `logs/preprocessing_context_bo/` and `bo/logs/` were treated as redundant after consolidation; **summary** artifacts and **`data/bo/preprocessing/`** were preserved.
- **`data/bo/**`** reinforced as untouchable in Cursor rules (includes promoted preprocessing under `data/bo/preprocessing/`).
- **Layout update:** fixed B+C+D baseline artifacts and comparison reports now live under **`logs/context_preprocessing_v1/`**; guarded BO summaries remain under **`logs/preprocessing_context_bo/<run_id>/...`** (both under `logs/`).
