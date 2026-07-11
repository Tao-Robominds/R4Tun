# Class-Swap Oracle Ceiling Analysis

**Purpose:** Technical report supporting reviewer responses on SAM4Tun segmentation limits, the role of class swaps versus K-detection error, and what can (and cannot) be fixed programmatically without ground truth.

**Date:** 2026-06-25  
**Scope:** SAM4Tun reference tunnel (`sample`) and continuous T3 tunnel `3-1-3`, with cross-reference to the regular-tunnel SAM hint ladder (10 tunnels, 1-* / 2-*).

---

## 1. Executive summary

We ran a controlled decomposition of SAM4Tun segmentation error to separate **mask quality** (block extent / background) from **positional labelling** (class assignment around the K-block). The central finding is:

1. **SAM masks are largely correct; class names are not.** On the SAM4Tun reference tunnel (`sample`), overall accuracy (OA) is **0.829** while mean IoU (mIoU) is only **0.645**. The gap is almost entirely **class swaps** (correct block region, wrong block label), not background leakage or missing blocks.

2. **Oracle swap correction defines a hard ceiling.** Post-hoc relabelling of swapped block points (`oracle_swap`) raises mIoU to **0.874** on `sample` and **0.829** on continuous tunnel `3-1-3` (with adapted detection, no flip pass). This is the practical upper bound of any method that keeps SAM masks but does not change the walk/labelling rule.

3. **K-detection is not the bottleneck on `sample`.** Geometric template hints that assume a fixed walk from detected K perform **worse** than baseline SAM (mIoU 0.34–0.42 vs 0.645). Mirror-flip correction is also irrelevant on `sample` (0/10 mirror rings).

4. **Blind programmatic swap correction is not yet solved.** Existing flip detectors require ground-truth angular order or per-point swap flags. Geometric and partial-oracle hints fail to reach mIoU ≥ 0.8 on regular tunnels. The dominant error mode on `sample` is **cyclical rotation** (8/10 rings), not mirror flip.

5. **Implication for R4Tun:** Parameter adaptation improves detection and mask recovery but cannot remove the positional-labelling ceiling without changing the SAM walk rule or adding a ring-level relabelling stage. The ~0.15 mIoU gap between baseline SAM (~0.62 mean on regular tunnels) and the 0.8 target is dominated by swaps, consistent with the error-composition analysis in the paper (Table: error composition; Constraint C4).

---

## 2. Motivation

Reviewers and internal analysis asked whether poor mIoU on some tunnels is caused by:

- (A) **K-block detection error** (wrong anchor Y / Hough failure), or  
- (B) **SAM walk / class-assignment error** (correct masks, wrong block names due to fixed template rotation or handedness flip).

We test this with **oracle ceilings**: upper bounds that apply increasing amounts of ground-truth information to isolate each failure mode. The minimum informative oracle, `oracle_swap`, keeps SAM background and mask shapes but corrects any block point whose predicted class differs from ground truth.

---

## 3. Methods

### 3.1 SAM4Tun baseline pipeline

Full open-source SAM4Tun pipeline on `data/sample.txt` (tunnel id `sample`, 10 rings, 1.1M GT-labelled points):

1. Unfolding → 2. Denoising → 3. Enhancing → 4. Detection → 5. SAM (`4-2_sam.py`, 6-class) → 6. Evaluation

Outputs: `data/sample/` (artefacts), `data/sample/evaluation/performance.md` (metrics).

Evaluation filters synthetic upsampled points (`pred > 6`) and reports metrics on original GT-labelled points only.

### 3.2 Oracle and hint modes

Implemented in `agents_regular/sam.py` and `methods/papers/scripts/regular_sam_hint_lib.py`.

| Mode | Information used | Mechanism |
|------|------------------|-----------|
| **Baseline (S0)** | None | SAM walk from detected K; fixed block order |
| **geometric (S1)** | Detected K Y per ring | Fixed template tiling; no SAM masks |
| **geometric_gt_k (S2)** | GT-calibrated K Y | As S1 with oracle K position |
| **geometric_gt_k_flip (S3)** | GT K Y + GT flip flags | As S2 with per-ring mirror |
| **gt_theta (S4)** | GT median θ per block | Sector assignment from GT geometry |
| **oracle_k_a2_a3 (S5a)** | GT labels for classes K, A2, A3 | Partial post-hoc relabelling |
| **oracle_swap (S5b)** | GT swap flags (`segment ≠ pred` on blocks) | Minimum ceiling for swap-dominated error |
| **oracle_blocks (S5)** | GT labels for all block points | Full block oracle |
| **gt_ring_flip / gt_handedness** | GT angular block order | Mirror block names per ring during SAM logit merge |

`oracle_swap` logic (post-hoc, no re-inference):

```python
swap = (segment > 0) & (pred > 0) & (segment != pred)
pred[swap] = segment[swap]
```

### 3.3 T3 continuous-tunnel tuning (`3-1-3`)

Manual parameter tuning via `methods/papers/scripts/run_t3_param_tune.py` with variants in `t3_param_hints.py`. Key variant:

- **`oracle_313_solo`:** adapted detection (3-1-3 v3 hints), **no SAM flip pass**, single SAM run + `oracle_swap` post-processing.

### 3.4 Error taxonomy (per ring)

Following `methods/scripts/diag_order_vs_offset.py`, each ring's predicted block angular order (from median θ) is compared to GT:

- **Rotation:** predicted order is a cyclic shift of GT (K-offset error).  
- **Flip:** predicted order is a cyclic shift of **reversed** GT (handedness / walk-direction error, Constraint C4).  
- **Other:** neither (mixed or fragmented detection).

---

## 4. Results

### 4.1 SAM4Tun reference tunnel (`sample`)

**Baseline SAM4Tun** (2026-06-25 rerun on `data/sample.txt`):

| Metric | Value |
|--------|------:|
| mIoU | **0.645** |
| OA | **0.829** |
| F1 | 0.773 |

Per-class IoU (baseline): Background 0.797, K 0.452, B1 0.698, A1 0.718, **A2 0.375**, A3 0.757, B2 0.720.

**Oracle ceilings** (post-hoc on same SAM masks, `data/sample/oracle_hint_check/summary.json`):

| Mode | mIoU | OA | Δ mIoU vs baseline |
|------|-----:|---:|-------------------:|
| Baseline SAM | 0.645 | 0.829 | — |
| **oracle_swap** | **0.874** | 0.931 | **+0.229** |
| oracle_blocks | 0.932 | 0.957 | +0.287 |

**Gap decomposition:**

- **+0.229 mIoU** recoverable by fixing class swaps only (masks and background unchanged).  
- **+0.057 mIoU** remaining to full block oracle (mask boundary / missing-class errors).

**Swap prevalence:** 142,091 / 761,755 block points mislabelled (**18.7%**). Block-level accuracy 0.813 vs overall OA 0.829.

**Per-class swap rate (GT block → wrong pred):**

| Class | Swap rate |
|-------|----------:|
| A2 | 57.1% |
| K | 34.8% |
| B1 | 16.8% |
| A1 | 15.1% |
| B2 | 13.6% |
| A3 | 12.5% |

A2 and K dominate swap errors, consistent with rotation around a misplaced walk origin.

**Ring-level ordering (GT vs pred):**

| Type | Rings (of 10) |
|------|-------------:|
| Rotation | **8** |
| Mirror flip | **0** |
| Other | 2 |

**Blind hint modes (no per-point GT; geometric assignment from detected/GT K):**

| Mode | mIoU | vs baseline |
|------|-----:|------------:|
| geometric | 0.344 | −0.301 |
| geometric_gt_k | 0.273 | −0.372 |
| geometric_gt_k_flip | 0.143 | −0.502 |
| gt_theta | 0.418 | −0.227 |

Geometric hints **degrade** performance on `sample`. Detection on this run used fallback/default K Y on several rings, but the result is still decisive: **template walk from detected K is not a viable substitute for SAM masks.**

### 4.2 Continuous tunnel `3-1-3`

Selected T3 tuning results (`logs/t3_tune/`):

| Variant | Description | mIoU |
|---------|-------------|-----:|
| regular_hint_v3 baseline | Adapted pipeline, SAM only | 0.229 |
| hough_low_flip | Detection tune + SAM flip pass | 0.204 |
| geo_313 | Geometric template (no SAM) | 0.224 |
| geo_313_flip | Geometric + GT flip | 0.361 |
| center_walk_313_nosnap | GT ring-flip pass, no center snap | 0.361 |
| **oracle_313_solo** | **No flip pass; oracle_swap only** | **0.829** |

On `3-1-3`, **`oracle_swap` alone reaches mIoU 0.829** without a mirror-flip SAM pass. Flip-pass variants without oracle remain near 0.22–0.36 mIoU. This supports the claim that **swap correction, not K-detection tuning or flip heuristics, defines the achievable ceiling** on this tunnel.

### 4.3 Regular tunnels (1-*, 2-*) — SAM hint ladder

From `methods/papers/output/regular_sam_hint_loop_summary.md` (10 tunnels, Opus-4.6 adapted pipeline):

| Level | Mode | Mean mIoU | ≥ 0.8 tunnels |
|-------|------|----------:|:-------------:|
| S0 | Baseline SAM | ~0.62 | 0/10 |
| S4 | gt_theta | < 0.57 | 0/10 |
| S5a | oracle_k_a2_a3 (3-class GT) | ~0.74 | 1/10 |
| **S5b** | **oracle_swap** | **0.863** | **10/10** |
| S5 | oracle_blocks | ~0.90 | 10/10 |

Detection hints alone (L0–L7) never exceed **0.685** mIoU (best: 2-2). **K-position oracle does not close the gap.**

### 4.4 Paper-scale error composition (30 tunnels)

From `methods/scripts/analyze_constraints.py` (Opus-4.6 m+s+k vs SAM4Tun baseline):

| Category | Method | Correct | FN | FP | **Swap** |
|----------|--------|--------:|---:|---:|---------:|
| Regular | SAM4Tun | 51% | 21% | 3% | **26%** |
| Regular | m+s+k | 71% | 2% | 5% | **21%** |
| Complex | SAM4Tun | 34% | 66% | 0% | 0% |
| Complex | m+s+k | 43% | 17% | 6% | **34%** |

Adaptation removes false negatives; **residual error is dominated by class swaps** on both regular and (post-recovery) complex tunnels.

---

## 5. Interpretation

### 5.1 OA and mIoU measure different things here

High OA with low mIoU occurs when most points are background or easy classes and a minority of block points are systematically mislabelled. **mIoU is the right metric for block segmentation quality; OA overstates success** when swaps are concentrated on block classes (especially A2, K).

### 5.2 What `oracle_swap` proves

`oracle_swap` is a **diagnostic ceiling**, not a deployable method. It proves:

- SAM **segmentation masks** (block extent) are largely acceptable.  
- The **positional labelling rule** (fixed walk from K) assigns the wrong block name to those masks.  
- Fixing labels alone recovers most of the mIoU gap (**+0.229** on `sample`; **10/10** regular tunnels ≥ 0.8 in S5b).

It does **not** prove that ground truth is required at deployment — only that **any method targeting mIoU ≥ 0.8 must address swap errors**, not just K detection.

### 5.3 Why K-detection tuning is insufficient

| Intervention | Mechanism | Outcome |
|--------------|-----------|---------|
| Detection hints (L0–L7) | Better Hough K / consensus | ≤ 0.685 mIoU on regular |
| geometric_gt_k (S2) | Oracle K Y + fixed template | Below baseline SAM |
| center_snap / hough_low | K Y refinement on T3 | mIoU stays ~0.22–0.37 without oracle |
| gt_ring_flip pass | Mirror per ring (needs GT order) | Helps mirror rings only (~17/130 regular); no effect on `sample` (0/10) |

### 5.4 Error modes require different fixes

| Error mode | Frequency (`sample`) | Programmatic fix status |
|------------|---------------------|-------------------------|
| Cyclical rotation | 8/10 rings | **Not implemented.** Promising direction: blind per-ring rotation search over existing SAM masks (maximise logit / template score). |
| Mirror flip (C4) | 0/10 rings | `gt_ring_flip` works with GT order; blind detector attempted (S3), failed on regular ladder. |
| Mask boundary / missing class | +0.057 mIoU to full oracle | Requires better SAM prompts or mask merge, not walk fix. |

### 5.5 Relation to structural constraints (paper Table C1–C4)

- **C2 (moving K-anchor):** Dominant on complex tunnels; parameter adaptation helps but does not remove swap ceiling.  
- **C4 (hard-coded walk direction):** Mirror flips on ~13% of regular rings; **rotation errors are more common on `sample`** and are not addressed by flip correction.  
- **R4Tun bounded adaptation** improves detection and reduces FN, but **cannot infer correct block names** without changing the labelling substrate or adding a post-SAM relabelling stage.

---

## 6. What can be improved without ground truth?

| Approach | GT required? | Expected impact | Status |
|----------|:------------:|-----------------|--------|
| Parameter tuning (R4Tun) | No | FN reduction, modest swap reduction | **Deployed** |
| K-detection hints | No | Does not reach 0.8 mIoU | **Tested; insufficient** |
| Geometric template walk | No | Hurts vs SAM | **Tested; harmful on `sample`** |
| GT ring-flip / handedness | Ring-level GT order | Mirror rings only | **Implemented; GT-dependent** |
| Partial class oracle (S5a) | 3 block classes | 1/10 tunnels ≥ 0.8 | **Tested; insufficient** |
| **oracle_swap (S5b)** | **Per-point swap flags** | **Closes gap entirely** | **Ceiling diagnostic only** |
| Blind rotation search | No | Unknown; targets dominant `sample` error | **Proposed, not implemented** |
| Cross-ring handedness vote | No | Unknown | **Proposed, not implemented** |

**Bottom line:** Matching the oracle ceiling without GT is **not achieved** in the current codebase. The minimum supervised information for 0.8 mIoU on regular tunnels is **swap identification** (S5b), strictly weaker than full segmentation GT but stronger than detection-only hints.

---

## 7. Recommended response text (draft)

> We decomposed SAM4Tun error into mask quality versus class-assignment (swap) error using oracle ceilings. On the public reference tunnel (`sample`, 10 rings), baseline SAM4Tun achieves OA 0.829 but mIoU only 0.645. Post-hoc correction of class swaps alone (`oracle_swap`, keeping SAM masks fixed) raises mIoU to 0.874 (+0.229), demonstrating that masks are largely correct and the gap is dominated by the fixed positional labelling rule, not K-detection failure on this tunnel. The same pattern holds on regular benchmarks: `oracle_swap` is the minimum hint level that reaches mIoU ≥ 0.8 on 10/10 tunnels (mean 0.863), while detection-only and geometric-template hints do not. Geometric walk hints without SAM masks perform worse than baseline on `sample` (mIoU 0.34–0.42). Mirror-flip correction is irrelevant on `sample` (0/10 mirror rings); 8/10 rings exhibit pure cyclical rotation error. These results support our error-composition analysis (Table X): adaptation removes false negatives, but residual error is class swap. Bounded parameter adaptation cannot reach the swap oracle ceiling without a post-SAM relabelling stage; blind rotation search over existing masks is identified as future work.

---

## 8. Limitations

1. **`sample` detection quality:** The rerun used fallback K Y on several rings; geometric hints may improve with better detection, but prior regular-tunnel ladder results already show geometric modes below baseline SAM.  
2. **`oracle_swap` uses evaluation-time GT** — it is an upper bound, not a production method.  
3. **T3 `3-1-3` baseline** under adapted pipeline (mIoU ~0.23) is harder than `sample`; oracle 0.829 is on tuned detection + SAM, not raw SAM4Tun defaults.  
4. **SAM rerun on `sample`** used CPU inference (GPU contended); masks should be numerically equivalent to GPU.  
5. **`agents_regular/sam.py`** does not activate geometric hint modes for `tunnel_id='sample'` (only 1-*/2-* prefixes); oracle analysis on `sample` was run via direct library calls.

---

## 9. Reproducibility

| Artefact | Path |
|----------|------|
| `sample` baseline metrics | `data/sample/evaluation/performance.md` |
| Oracle / geometric ablation | `data/sample/oracle_hint_check/summary.json` |
| SAM4Tun rerun logs | `logs/sam4tun_sample_*.log`, `logs/sam4tun_sample_sam_cpu_*.log` |
| T3 oracle variant | `logs/t3_tune/oracle_313_solo/3-1-3/evaluation/performance.md` |
| Regular SAM hint ladder | `methods/papers/output/regular_sam_hint_loop_summary.md` |
| Code | `sam4tun/4-2_sam.py`, `agents_regular/sam.py`, `methods/papers/scripts/regular_sam_hint_lib.py` |

**Reproduce `sample` baseline:**

```bash
cd /path/to/R4Tun
export PYTHONPATH="sam4tun/segment-anything${PYTHONPATH:+:$PYTHONPATH}"
venv/bin/python3 sam4tun/1_upfolding.py sample
venv/bin/python3 sam4tun/2_denoising.py sample
venv/bin/python3 sam4tun/3_enhancing.py sample
venv/bin/python3 sam4tun/4-1_detection.py sample
venv/bin/python3 sam4tun/4-2_sam.py sample
venv/bin/python3 sam4tun/evaluation.py sample
```

**Reproduce regular-tunnel S5b ladder:**

```bash
python3 methods/papers/scripts/run_regular_sam_hint_loop.py --level S5b --all-tunnels
```

---

## 10. Summary table (key numbers)

| Tunnel / setting | Baseline mIoU | oracle_swap mIoU | Δ | Dominant error |
|------------------|-------------:|-----------------:|--:|----------------|
| `sample` (SAM4Tun) | 0.645 | 0.874 | +0.229 | Rotation (8/10 rings) |
| `3-1-3` (adapted + oracle only) | ~0.23* | 0.829 | — | Swap ceiling under tuned detection |
| Regular 10-tunnel mean (S0→S5b) | ~0.62 | 0.863 | +0.24 | Swap (21–26% of GT points) |

\*Adapted-pipeline baseline before oracle; not directly comparable to `sample` SAM4Tun defaults.

---

*End of report.*
