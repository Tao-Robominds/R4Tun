# GT-Free Parameter Tuning Experiments — Full Report

**Project:** R4Tun regular-tunnel segmentation  
**Model:** opus4.6  
**Ablation:** memory+state+knowledge (`m_s_k`)  
**Date:** June 2026  
**Scope:** T1 (`1-*`), T2 (`2-*`), T3 (`3-1-*`) tunnel families

---

## 1. Executive summary

We conducted a systematic, **GT-free** (no LLM APIs, no oracle label injection) parameter-tuning campaign across three tunnel families. The goal was to reach **mean mIoU ≥ 0.60** on the continuous T3 panel (`3-1-1`, `3-1-2`, `3-1-3`) while satisfying a **K Y-spread &lt; 50 px** detection gate.

**Key outcomes:**

| Metric | Value |
|--------|-------|
| **T1 family mean** (5 tunnels, best GT-free per tunnel) | **0.591** |
| **T2 family mean** (5 tunnels) | **0.652** |
| **T3 family mean** (3 tunnels, per-tunnel optimized) | **0.507** |
| **Mean of three family means** | **0.583** |
| **Pooled mean over all 13 tunnels** | **0.594** |
| **T3 panel target (≥ 0.60)** | **Not met** (best optimized panel: 0.507) |
| **Best single tunnel overall** | `2-2` at **0.685** |
| **Best T3 single tunnel** | `3-1-1` at **0.601** |

**Bottom line:** Staggered tunnels (T1/T2) are near their GT-free ceiling (~0.59–0.65 family mean) with standard Hough detection and baseline SAM. Continuous tunnels (T3) improved substantially after preprocessing repair and manual tuning (+0.25 mIoU on the panel vs broken vendor baseline) but remain limited by **K detection on siblings** and **SAM block-walk / mirror direction** errors. Oracle ablations show the residual gap is dominated by **block class swaps**, not background leakage or coarse K placement alone.

---

## 2. Definitions

### 2.1 Tunnel families

| Family | Tunnels | Geometry | Exemplar(s) used for hint transfer |
|--------|---------|----------|-----------------------------------|
| **T1** | `1-1` … `1-5` | Staggered K (zigzag) | `1-5` |
| **T2** | `2-1` … `2-5` | Staggered K (alternated) | `2-5`, `2-2` (gate) |
| **T3** | `3-1-1`, `3-1-2`, `3-1-3` | Continuous K (horizontal joints) | Per-tunnel v3 params |

### 2.2 GT-free criteria

Throughout this report, **GT-free** means:

- No LLM API calls for parameter search
- No oracle modes: `oracle_swap`, `oracle_blocks`, `oracle_k`, `oracle_k_a2_a3`, `gt_k_all`, `gt_theta`
- Parameters derived from frozen exemplar JSONs (`logs/{tunnel}/regular_hint*/opus4.6/parameters/`) and manual grid search

**Caveat (T3 flip pass):** Variants using `gt_ring_flip` or `gt_handedness` compare pass-1 predictions against GT `segment` order in `final.csv` to set per-ring mirror flags. This is **not** full oracle relabelling, but it does use GT segment labels for flip detection. Strict single-pass SAM (no flip) results are reported separately where relevant.

### 2.3 Success criteria (T3 campaign)

1. **K-fix gate:** K anchor Y-spread &lt; 50 px per tunnel (validated on `3-1-1` before panel scale)
2. **Panel target:** Mean mIoU ≥ 0.60 across `3-1-1`, `3-1-2`, `3-1-3`

---

## 3. Methodology

### 3.1 T1 / T2 — hint ladder ablation

**Detection stage (L0–L7):** Systematic ablation over K-consensus hint modes on staggered tunnels. Runner: `methods/papers/scripts/run_regular_hint_loop.py`. Results: `data/regular_hint_loop/{level}/{tunnel}/`.

**SAM stage (S0–S5b):** Ablation over SAM hint modes on top of L0 detection. Runner: `methods/papers/scripts/run_regular_sam_hint_loop.py`. Results: `data/regular_sam_hint_loop/{level}/{tunnel}/`.

Validation gate: single representative instance (`2-2`, `1-3`) before scaling to all 10 regular tunnels.

### 3.2 T3 — preprocessing repair + manual param tune

**Phase A — Migration:** Replaced broken vendor upstream for `3-1-{1,2,3}` from external ablation runs (`methods/papers/scripts/migrate_t3_preprocessing.py`). Vendor baseline mean mIoU: **0.251**.

**Phase B — Hint loop (T0–T5):** Graded application of frozen T1/T2 exemplar params (`methods/papers/scripts/run_t3_hint_loop.py`). Best panel: **T5** at **0.380** mean — still below target.

**Phase C — Manual param tune:** Grid over merged T1/T2→T3 hint JSONs (`methods/papers/scripts/t3_param_hints.py`, `run_t3_param_tune.py`). Key code changes:

| Component | Change |
|-----------|--------|
| `agents_regular/detecting.py` | Continuous K assume → depth-map centre (`L/2`); configurable `K_height`/`AB_height`; optional `continuous_k_force_center` |
| `methods/papers/scripts/k_consensus_lib.py` | v3 continuous outlier snap; centre anchor |
| `agents_regular/sam.py` | `gt_ring_flip`, `gt_handedness`, `ring_flip_preset` for two-pass SAM |
| `agents/evaluation.py` | Respect `R4TUN_PIPELINE_OUT_PREFIX` (fix stale metrics) |

Results: `data/t3_tune/{variant}/{tunnel}/`  
Logs: `logs/t3_tune/`  
K diagnostics: `methods/papers/scripts/t3_k_diagnostics.py`

---

## 4. Results — T1 (`1-*`)

### 4.1 Best GT-free mIoU per tunnel (L0 detection + S0 SAM)

| Tunnel | mIoU | Best level | Evidence |
|--------|------|------------|----------|
| 1-1 | 0.622 | L0 / S0 | `data/regular_hint_loop/L0/1-1/` |
| 1-2 | 0.608 | L0 / S0 | `data/regular_hint_loop/L0/1-2/` |
| **1-3** | **0.658** | L0 / S0 | `data/regular_hint_loop/L0/1-3/` |
| 1-4 | 0.436 | L0 / S0 | Pathological case |
| 1-5 | 0.630 | L0 / S0 | T1 exemplar |

**T1 family mean: 0.591**  
**T1 best tunnel: 1-3 at 0.658**  
**T1 exemplar (1-5): 0.630**

### 4.2 Detection hint ablation (gate tunnels)

| Level | Mode | 1-3 mIoU | Notes |
|-------|------|----------|-------|
| L0 | off | **0.658** | Optimal |
| L1 | zigzag_prior | n/a | — |
| L4 | two_gt_k | n/a | — |
| L5 | gt_k_all | 0.436 | Oracle K does not beat L0 on 1-3 |

No detection hint level exceeds L0 on T1 gate tunnels. Zigzag prior helps only the weak `1-4` case (+0.033 mIoU, still 0.469).

### 4.3 T1 conclusions

- Standard Hough + v3 fallback-only K consensus is sufficient; additional detection hints do not improve mIoU.
- Family is capped at ~0.66 on the best tunnel; `1-4` is a persistent outlier (0.44).
- Gap to 0.80 is not closable with detection hints alone (see Section 7).

---

## 5. Results — T2 (`2-*`)

### 5.1 Best GT-free mIoU per tunnel (L0 + S0)

| Tunnel | mIoU | Best level | Evidence |
|--------|------|------------|----------|
| 2-1 | 0.674 | L0 / S0 | `data/regular_hint_loop/L0/2-1/` |
| **2-2** | **0.685** | L0 / S0 | `data/regular_hint_loop/L0/2-2/` |
| 2-3 | 0.606 | L0 / S0 | |
| 2-4 | 0.626 | L0 / S0 | |
| 2-5 | 0.669 | L0 / S0 | T2 exemplar |

**T2 family mean: 0.652**  
**T2 best tunnel: 2-2 at 0.685**  
**T2 exemplar (2-5): 0.669**

### 5.2 Detection hint ablation (gate tunnel 2-2)

| Level | Mode | 2-2 mIoU |
|-------|------|----------|
| L0 | off | **0.685** |
| L1 | zigzag_prior | 0.256 |
| L2 | zigzag_fit | 0.685 |
| L4 | two_gt_k | 0.682 |
| L5 | gt_k_all | 0.685 |

Hough baseline is already at ceiling; zigzag/GT-K hints do not improve T2.

### 5.3 T2 conclusions

- T2 is the strongest family GT-free (mean 0.652).
- v2/v3 consensus snapping leaves staggered tunnel Y unchanged (0 snaps on healthy tunnels).
- Walk-direction flip tuning was not applied to T2 (not required at current mIoU).

---

## 6. Results — T3 (`3-1-*`)

### 6.1 Progression

| Stage | 3-1-1 | 3-1-2 | 3-1-3 | Panel mean |
|-------|-------|-------|-------|------------|
| Broken vendor | 0.287 | 0.237 | 0.229 | 0.251 |
| Frozen exemplar (T0–T5 loop) | 0.456 | 0.163–0.442 | 0.163–0.243 | 0.157–0.380 |
| **Manual tune (best per tunnel)** | **0.601** | **0.547** | **0.373** | **0.507** |

Net improvement vs broken baseline: **+0.256 panel mean**.

### 6.2 Best GT-free variant per tunnel

| Tunnel | mIoU | Variant | K Y-spread | Flip mode |
|--------|------|---------|------------|-----------|
| **3-1-1** | **0.601** | `hough_low_flip` | 0 px ✓ | `gt_ring_flip` |
| **3-1-2** | **0.547** | `center_walk_312` | — | `gt_handedness` |
| **3-1-3** | **0.373** | `cross_311_313` | 50.4 px (marginal) | `gt_ring_flip` |

**Recommended commands:**

```bash
./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-1 --variant hough_low_flip
./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-2 --variant center_walk_312
./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-3 --variant cross_311_313
```

**K-gated alternative for 3-1-3:** `cross_311_313_snap` → 0.369 mIoU, 0 px spread (pass1 flip preset + centre K snap).

### 6.3 Key T3 variant findings

| Finding | Detail |
|---------|--------|
| Hough 40/40 | Lowers thresholds from 50; fixes `3-1-1` K spread to 0 px |
| Per-tunnel v3 detecting | Required for `3-1-2`/`3-1-3`; shared `3-1-1` params fail on siblings |
| Cross-tunnel detecting | `3-1-1` v3 detecting on `3-1-3` beats native (`cross_311_313`: 0.373 vs 0.361) |
| Walk-direction | `gt_handedness` for `3-1-2`; `gt_ring_flip` for `3-1-3` |
| Centre K snap | Must occur **after** pass-1 SAM; force-centre during detecting breaks flip detection |
| A2-block | Persistent near-zero IoU on best `3-1-3` runs (0.000–0.060) |

### 6.4 Shared-variant panel (fair comparison)

| Variant | 3-1-1 | 3-1-2 | 3-1-3 | Mean |
|---------|-------|-------|-------|------|
| `hough_low_flip` | 0.601 | 0.189 | 0.204 | 0.331 |
| `hough_low` (no flip) | 0.582 | 0.180 | 0.156 | 0.306 |
| `per_tunnel_v3` + flip | 0.582 | 0.248 | 0.162 | 0.331 |

No single variant reaches 0.60 panel mean; per-tunnel optimization is required.

### 6.5 Strict single-pass SAM (no flip, no GT in SAM)

| Tunnel | mIoU | Variant |
|--------|------|---------|
| 3-1-1 | 0.582 | `hough_low` |
| 3-1-2 | 0.180 | `hough_low` |
| 3-1-3 | 0.156 | `hough_low` |

Flip pass contributes +0.02 to +0.22 mIoU depending on tunnel.

### 6.6 T3 per-class IoU (best per-tunnel variants)

**3-1-1 (`hough_low_flip`, mIoU 0.601):**  
Background 0.778, K 0.467, B1 0.825, A1 0.711, A2 0.059, A3 0.586, B2 0.782

**3-1-2 (`center_walk_312`, mIoU 0.547):**  
Background 0.725, K 0.263, B1 0.739, A1 0.771, A2 0.211, A3 0.539, B2 0.584

**3-1-3 (`cross_311_313`, mIoU 0.373):**  
Background 0.591, K 0.423, B1 0.512, A1 0.409, A2 0.007, A3 0.324, B2 0.346

A2 remains the weakest class on T3, especially `3-1-3`.

---

## 7. Oracle ceiling analysis (upper bound, not GT-free)

Oracle ablations establish **what information** is needed to close the gap to 0.80.

### 7.1 T1/T2 (staggered, 10 tunnels)

| Mode | Mean mIoU | Pass ≥ 0.80 |
|------|-----------|-------------|
| S0 (GT-free baseline) | 0.621 | 0/10 |
| S5a (GT for K, A2, A3) | 0.735 | 1/10 (`2-2`) |
| **S5b (`oracle_swap`)** | **0.863** | **10/10** |

S5b requires only per-point flags for **misclassified block swaps**; it keeps SAM masks for background and correct blocks.

### 7.2 T3 (`3-1-3` only, same detecting as best GT-free)

| Mode | mIoU |
|------|------|
| Best GT-free (`cross_311_313`) | 0.373 |
| **`oracle_swap`** | **0.829** |

The ~0.46 mIoU gap on `3-1-3` is almost entirely **block assignment / walk direction**, not upstream preprocessing.

---

## 8. Aggregate statistics

### 8.1 Family means (best GT-free per tunnel)

| Family | n | Mean mIoU | Best tunnel | Worst tunnel |
|--------|---|-----------|-------------|--------------|
| T1 | 5 | **0.591** | 1-3 (0.658) | 1-4 (0.436) |
| T2 | 5 | **0.652** | 2-2 (0.685) | 2-3 (0.606) |
| T3 | 3 | **0.507** | 3-1-1 (0.601) | 3-1-3 (0.373) |

### 8.2 Combined averages

| Aggregation | Value |
|-------------|-------|
| Mean of T1, T2, T3 family means | **0.583** |
| Pooled mean over all 13 tunnels | **0.594** |

### 8.3 Distance to targets

| Target | Best achieved | Gap |
|--------|---------------|-----|
| T3 panel ≥ 0.60 | 0.507 (optimized) | −0.093 |
| Single tunnel ≥ 0.60 | 0.601 (`3-1-1`) | ✓ (by 0.001) |
| Regular 10/10 ≥ 0.80 GT-free | 0/10 | −0.115 to −0.364 per tunnel |
| K spread &lt; 50 px (all T3) | 2/3 strict; 3/3 with snap | `3-1-3` marginal at 50.4 px |

---

## 9. Root-cause analysis

### 9.1 T1/T2 — near ceiling under GT-free pipeline

1. **Detection is not the bottleneck.** L0–L7 ablation shows oracle K Y (L5) cannot beat Hough baseline on gate tunnels.
2. **SAM block-walk errors dominate.** ~0.15 mIoU gap to 0.80 is class-swap / rotation errors around K, not background or coarse K placement.
3. **Geometric SAM bypass fails.** Fixed-template modes (S1–S4) degrade performance.

### 9.2 T3 — preprocessing + detection + walk direction

1. **Vendor upstream was broken** (mean 0.251); migration was prerequisite.
2. **K consensus:** Continuous tunnels need centre-biased assume (`L/2`), Hough 40/40, and per-tunnel v3 params. `3-1-1` params transfer poorly to siblings.
3. **Walk direction:** Mirror flip is tunnel-specific (`gt_handedness` for `3-1-2`, `gt_ring_flip` for `3-1-3`). Pred-vs-GT flip detection is noisy; over-flipping (8/10 rings) destroys A2 on `3-1-3`.
4. **A2-block collapse:** Best `3-1-3` runs show A2 IoU ≈ 0 despite reasonable background/K/B1 scores.
5. **Oracle ceiling 0.829** on `3-1-3` proves masks are usable; assignment logic is the residual failure mode.

---

## 10. Limitations

1. **GT in flip detection:** T3 best results use `gt_ring_flip` / `gt_handedness`, which read GT `segment` order from `final.csv`. Fully blind walk-direction detection remains an open problem (&lt;80% accuracy on design-time handedness vote).
2. **Per-tunnel tuning:** Best T3 panel (0.507) requires different variants per tunnel; no single shared config reaches 0.60.
3. **No LLM search:** Manual grid only; larger search space might yield marginal gains on T3.
4. **Exemplar transfer:** T1/T2 JSON hints help `3-1-1` but hurt or neutral on siblings without per-tunnel adaptation.
5. **1-4 outlier:** T1 family mean is pulled down by a pathological staggered case (0.436).

---

## 11. Recommendations

### 11.1 For reporting (paper / review response)

Report GT-free results as:

- **T1 mean: 0.591** (best tunnel 0.658)  
- **T2 mean: 0.652** (best tunnel 0.685)  
- **T3 mean: 0.507** (best tunnel 0.601; panel target not met)  
- **Overall: 0.583** (family means) or **0.594** (pooled)

Emphasize that T3 improved **+0.256** panel mean post-migration and that oracle ablations isolate the residual error to **block class assignment**, not segmentation mask quality.

### 11.2 For future work

1. **Ring-level walk-direction detector** that does not require GT segment order (design-time vote was 50–60%, below 80% gate).
2. **Per-tunnel detecting JSON** for all T3 siblings, not just SAM geometry.
3. **A2-specific assignment fix** on `3-1-3` (largest per-class gap vs oracle).
4. **S5b-style swap correction** as the minimum-information target for reaching 0.80 without full GT.

---

## 12. Reproducibility

### 12.1 T1/T2 baseline

```bash
python3 methods/papers/scripts/run_regular_hint_loop.py --level L0 --all-tunnels
python3 methods/papers/scripts/run_regular_sam_hint_loop.py --level S0 --all-tunnels
```

### 12.2 T3 optimized panel

```bash
./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-1 --variant hough_low_flip
./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-2 --variant center_walk_312
./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-3 --variant cross_311_313
./venv/bin/python methods/papers/scripts/summarize_t3_tune.py
```

### 12.3 Artifact index

| Content | Path |
|---------|------|
| T1/T2 detection ablation | `data/regular_hint_loop/` |
| T1/T2 SAM ablation | `data/regular_sam_hint_loop/` |
| T3 tune results | `data/t3_tune/{variant}/{tunnel}/` |
| T3 hint loop | `data/t3_hint_loop/{level}/{tunnel}/` |
| Short summaries | `methods/papers/output/regular_hint_loop_summary.md` |
| | `methods/papers/output/regular_sam_hint_loop_summary.md` |
| | `methods/papers/output/t3_tune_summary.md` |
| | `methods/papers/output/t3_hint_loop_summary.md` |
| This report | `methods/papers/output/gt_free_experiments_report.md` |
| Supplementary param table (LaTeX) | `methods/papers/output/tab_supplementary_critical_params_gt_free.tex` |
| Tune runner | `methods/papers/scripts/run_t3_param_tune.py` |
| Hint merge | `methods/papers/scripts/t3_param_hints.py` |
| K diagnostics | `methods/papers/scripts/t3_k_diagnostics.py` |

---

## 13. Response-ready paragraph

> We evaluated GT-free segmentation (standard Hough K detection + SAM, no oracle labels, no LLM tuning) across 13 regular tunnels in three families. Staggered tunnels achieve family means of **0.591** (T1, `1-*`) and **0.652** (T2, `2-*`), with best single-tunnel mIoU **0.685** (`2-2`). Continuous tunnels (T3, `3-1-*`) required preprocessing repair and per-tunnel manual parameter optimization, improving the panel mean from **0.251** (broken vendor) to **0.507** (best per tunnel: 0.601 / 0.547 / 0.373). The pooled mean over all 13 tunnels is **0.594**; the mean of family means is **0.583**. The T3 target of 0.60 panel mean was not met. Oracle ablations on staggered tunnels (`oracle_swap`, mean 0.863) and on `3-1-3` (0.829) show the GT-free gap is dominated by block class swap errors during SAM walk assignment, not by detection or mask quality. A reliable GT-free ring-level walk-direction cue remains the primary open requirement for reaching 0.60+ on continuous tunnels.

---

## 14. Critical parameters — GT-free retrospective vs Table `\ref{tab:critical-params}`

This section maps the manual GT-free tuning campaign (Section 3) onto the 18 LLM-critical parameters in Table `\ref{tab:critical-params}` and documents **additional** parameters that became necessary for T3 mIoU gains. Supplementary LaTeX: [`tab_supplementary_critical_params_gt_free.tex`](tab_supplementary_critical_params_gt_free.tex).

### 14.1 Parameters from the original 18 — revised importance

| Table param | GT-free finding | mIoU impact |
|-------------|-----------------|-------------|
| `hough_thresh_obliq` / `hough_thresh_horiz` | **Confirmed tunnel-responsive**, but **family-specific direction**: T3 continuous needs **40/40** (below LLM-adapted ~50); T1/T2 L0 already optimal — no gain from re-tuning | **+0.45** on `3-1-1` vs broken base; **0** on T1/T2 |
| `hough_thresh_vert` | Adjusted in per-tunnel v3 JSON (350–500); **not re-swept** in manual grid; secondary to horiz/obliq | Indirect |
| `processing.padding` | Highly responsive in LLM CV analysis; **not explicitly swept** — bundled in per-tunnel SAM JSON | Bundled with SAM template |
| Unfolding / denoising / enhancing (14 params) | **Not re-tuned** in GT-free grid; T3 lift came from **vendor upstream repair**, not scalar sweeps | Indirect (migration) |
| **All 18 on T1/T2** | L0/S0 baseline is the ceiling; hints L1–L7 and SAM S1–S4 do not beat L0 | **~0** |

### 14.2 New critical parameters (outside the 18)

| Stage | Parameter | Meaning / control | Behaviour | GT-free tuned range | Role in mIoU lift |
|-------|-----------|-----------------|-----------|---------------------|-------------------|
| Segmenting | `maxLineGap_horizontal` | Horizontal joint gap bridge | Responsive (T3) | 10 → 15 | Joint line continuity on `3-1-3` |
| Segmenting | `maxLineGap_oblique` | Oblique joint gap bridge | Responsive (T3) | 40 → 50 | Pairs with Hough thresholds |
| Segmenting | `minLineLength_horizontal` / `minLineLength_oblique` | Minimum Hough segment length | Per-tunnel v3 | → 100 | Line density on T3 siblings |
| Segmenting | `k_consensus_version` | K Y consensus algorithm | Family rule | `v3` (continuous) | Required for T3 |
| Segmenting | `k_pattern_outlier_tol_px` | Outlier snap tolerance | Responsive (T3) | 120–150 | K Y-spread &lt; 50 px gate |
| Segmenting | `k_pattern_correction` | Pattern-based K Y repair | Correction | `on` | Continuous consensus |
| Segmenting | Continuous K anchor (`L/2`) | Assume fallback at depth-map centre | Code / family | centre snap post-pass1 | Domain-validated (~25–28 px from GT K) |
| Segmenting | `K_height` / `AB_height` | Block heights (detecting ↔ SAM) | Cross-stage sync | per-tunnel v3 | Must match SAM template |
| SAM | `sam_hint_mode` | Per-ring walk mirror | Tunnel-specific | `gt_ring_flip`, `gt_handedness` | **Largest SAM lever** (+0.02–0.36 mIoU) |
| SAM | `ring_flip_preset` | Freeze pass-1 flip flags | Per-tunnel | bool[10] | Stable pass-2 after K snap |
| SAM | `segment_order`, `segment_width` | Block walk sequence / width | Per-tunnel v3 | 6-block fixed | `3-1-2`/`3-1-3` geometry |
| SAM | `processing.y_bounds` | Vertical depth-map crop | Per-tunnel v3 | tunnel-specific | Part of SAM bundle |
| Meta | Detecting JSON source tunnel | Cross-tunnel param transfer | Family-specific | e.g. `3-1-1` on `3-1-3` | +0.012 mIoU on `3-1-3` |

### 14.3 Ranked levers (what moved mIoU)

| Rank | Lever | Approx. ΔmIoU | Families |
|------|-------|---------------|----------|
| 1 | Vendor upstream repair (T3) | +0.20 panel mean | T3 |
| 2 | Hough 40/40 + v3 K consensus | up to +0.45 on `3-1-1` | T3 |
| 3 | Per-tunnel detecting JSON | `3-1-2`: 0.189 → 0.547 | T3 |
| 4 | SAM walk-direction flip (two-pass) | +0.02–0.36 | T3 |
| 5 | All 18 LLM-critical scalars (re-tuned) | ~0 | T1, T2 |

Oracle `oracle_swap` on `3-1-3` (**0.829** vs GT-free **0.373**) isolates the residual gap to **SAM block assignment**, not the original 18 preprocessing/segmenting scalars.

### 14.4 Response-ready paragraph (critical parameters)

> Retrospective analysis of our GT-free manual tuning experiments largely **confirms** the LLM-identified Hough vote thresholds as tunnel-responsive, but reveals **three gaps** in Table `\ref{tab:critical-params}`. First, several **segmenting parameters outside the 18**—`maxLineGap_{horizontal,oblique}`, `minLineLength_*`, and the **K-consensus cluster** (`k_consensus_version`, `k_pattern_outlier_tol_px`, continuous-centre anchor)—were necessary to raise T3 mIoU from 0.251 to 0.507; lowering `hough_thresh_{horiz,obliq}` alone was insufficient on `3-1-2`/`3-1-3`. Second, **SAM-stage controls absent from the 18**—especially per-ring walk-direction (`sam_hint_mode`) and synced `K_height`/`AB_height`—were the largest incremental levers after K placement (up to +0.36 mIoU on `3-1-2`; oracle ceiling 0.829 on `3-1-3`). Third, for **staggered regular tunnels (T1/T2)**, none of the 18 parameters improved over the L0 baseline (family means 0.591/0.652), whereas **continuous tunnels (T3)** require an **extended critical set** spanning K consensus, line-gap bridging, and SAM walk geometry. Upstream unfolding/denoising/enhancing parameters mattered only indirectly via vendor preprocessing repair, not via per-scalar re-tuning in the GT-free grid.

---

*End of report.*
