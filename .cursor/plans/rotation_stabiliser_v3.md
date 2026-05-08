# Plan: Deterministic rotation stabiliser before intrinsic reflection (v3)

Sandbox path: `logs/v3/rotation_stabiliser/<tunnel>/r<ring>/`
(Plus diagnostic / aggregate outputs under `logs/v3/rotation_stabiliser/`. No
writes ever go under `data/v3/panels/`, `data/<tunnel>/r*/`, `r4tun/`,
`logs/context_preprocessing_v1/`, `methods/plans/output/`, `data/baseline/`,
`data/bo/`, `data/ablation/`, or any other read-only prefix in
`.cursor/rules/intrinsic-project.mdc`.)

## 1. Goal

Replace the ambiguous "gravity bottom anchoring lifts mIoU" claim with the
**actual** deterministic stabilisation that v3 needs:

> Before the LLM intrinsic-reflection loop runs, the pipeline must put every
> ring into a single canonical orientation **frame** so that the K-anchored
> per-ring offsets in `parameters_detection.json` are valid, and `fixed_class
> mIoU` becomes a meaningful, monotone signal for reflection to optimise.

Concretely we need to:

1. Diagnose which rotational degree(s) of freedom are still unstabilised in v3
   on the 40-ring held-out panel (theta-origin vs. direction-of-rotation).
2. Design and implement an **intrinsic, label-free, deterministic** resolver
   for whichever DOF dominates the held-out variance.
3. Integrate it as a single pre-reflection step (preprocessing or early
   detection), with intrinsic confidence and a clean fail-safe.
4. Re-run Arms A/B on the held-out panel under the stabilised pipeline,
   document the lift, and rewrite the paper's "0.109 → 0.256" claim to match
   what the deterministic step actually buys.

## 2. Relationship to SAM4Tun (read first)

SAM4Tun (`r4tun/sample/SAM4Tun.pdf`, notebook
`r4tun/sample/SAM4Tun.ipynb`) handles K-block localisation and label
ordering, but **does not have an automatic direction resolver**:

- **Theta-origin** is hardcoded to gravity-bottom (Algorithm 1, "sliced
  from directly below (0°)"), same as v3 with `gravity_anchor.enabled`.
- **Unwrap direction** is left to the user. Notebook Cell 5 says verbatim:
  *"important: Ensure that the vector direction is consistent with the
  forward direction of the shield machine"*. SAM4Tun trusts the operator
  to feed the centre-curve vector along the shield's advance direction;
  there is no algorithmic flip detector.
- **K detection** uses Hough oblique lines at `±6°…±9°` and a signed-y
  offset (`±0.5·K_height`) — already present in v3's
  `expand_k_with_per_ring_offsets`.
- **Block ordering after K** is delegated to SAM with fixed-design-
  dimension template prompts and a "clockwise from K" label rule
  (paper §5: *"label order follows the lining forward direction, starting
  clockwise with the K-block as label 1"*). SAM is robust to local
  deformations, so the absence of an offset table is fine.
- **Fallback** when a single ring's joint detection fails: inherit K-y
  from a neighbouring ring of the same tunnel (Algorithm 4 / §5.2.1).

v3 inherits Algorithm 1 and the Hough-sign K detection but **replaces SAM
with a deterministic per-ring offsets table**
(`bo/v3/r4tun_seed.py::_physical_offsets_7block` packs all offsets in the
+y direction from K). This is what breaks on `reversed_canonical` rings:
K is still detected, but the signed offsets land every non-K block on
the wrong side. SAM4Tun side-stepped this by not having an offsets
table at all.

So the rotation ambiguity we are fixing is **not solved by SAM4Tun**.
We do reuse two SAM4Tun design choices that already work:

1. Hough-sign K detection (Cell 57–58) as the K oracle.
2. Neighbour inheritance at the *tunnel* level as the fallback when the
   single-ring direction signal is weak.

What is missing in v3 is the explicit **block-sequence-direction
resolver around K** that SAM4Tun didn't need because SAM did the boundary
finding for it.

## 3. Hypothesis (to be tested in §5)

The held-out fixed-class mIoU scoreboard from `heldout_ablation_cursor_mvp`
is bimodal (~0.6–0.8 for canonical-direction rings, ~0.07–0.18 for
reversed-direction rings) and gravity anchoring is irrelevant to the split.
The dominant residual rotation ambiguity in v3 is therefore **block-
sequence direction around K** (the +theta vs −theta sign for the offsets
table), not theta-origin.

Falsifier: if the panel's `pattern_type ∈ {canonical, reversed_canonical}`
does *not* correlate with the bimodal mIoU, the dominant DOF is something
else and §6/§7 must be redirected before §8 implements anything.

## 4. Out of scope

- **Re-running BO** (calibration is frozen).
- **Changing the LLM reflection harness** — Arm C still consumes the same
  intrinsics + ontology bundle.
- **Editing GT files** — `data/v3/panels/heldout/rings/*.txt` and the underlying
  `data/<tunnel>/r*/` corpora are read-only. The stabiliser is a
  *pipeline-side* change only.
- **Re-introducing SAM into the deterministic path.** SAM4Tun avoided the
  offsets-table problem by letting SAM find boundaries; v3 deliberately
  chose deterministic post-K offsets for reproducibility and intrinsics.
  We honour that choice and instead add the missing direction signal.
- **Gravity anchoring redesign**. We keep the existing flag; we are not
  promising it lifts mIoU. The paper claim moves off "gravity bottom anchoring"
  onto whatever this step actually delivers.

## 5. Phase 1 — Diagnostic (read-only, ~1 day)

All outputs go under `logs/v3/rotation_stabiliser/diagnose/`.

### 5.1 Confirm direction is the dominant DOF

- Script: `bo/v3/diagnose_rotation_ambiguity.py`
- For each of the 40 held-out rings, join existing
  `logs/v3/heldout/scoreboard.csv` with the panel's `pattern_type`.
- Stratify `miou_a`, `miou_b`, `miou_c_final` by
  `pattern_type ∈ {canonical, reversed_canonical}`.
- Output: `direction_split.csv` and `direction_split.md` with means, std,
  histograms, two-sample test (Mann-Whitney U; we don't need normality).
- Acceptance: a clear gap (≥ 0.1 mean mIoU) between the two groups
  *or* an explicit "no, the split is something else" verdict.

### 5.2 Per-ring confusion-matrix audit (ground-truth-aided, design-time only)

- Script: `bo/v3/audit_label_misplacement.py`
- For each ring, load `final.csv` + GT and compute the `7×7` confusion matrix
  of `pred_class_id` vs `gt_segment_id` after the same y-rank canonicalisation
  used in `score_unrotated.py`.
- Detect "off-by-mirror" pattern: K (id=1) on diagonal, B1↔B2 swapped, A1↔A4
  swapped, A2↔A3 swapped.
- Output: `direction_audit.csv` with one row per ring tagging
  `direction_match ∈ {canonical, reversed, mixed, undetermined}`.
- This is the ground-truth-aided oracle the resolver in §5/§6 must approximate
  using only intrinsic signals.

### 5.3 Theta-origin residual

- For the rings flagged `direction_match=canonical` in 5.2 (where direction is
  already correct), check whether residual mIoU is bounded below by
  theta-origin error or by other failure modes (fragmented K detection, missing
  blocks, false A4, ...).
- If theta-origin error explains a meaningful share, escalate
  to a second resolver in §6; otherwise note it as a known-residual.

**Decision gate**: if §5.2 confirms ≥ 60% of held-out rings are
`reversed`/`mixed`, we proceed with a direction resolver. Otherwise, replan.

## 6. Phase 2 — Resolver design (paper-side reasoning, ~0.5 day)

Goal: a deterministic, intrinsic, label-free **block-sequence-direction
resolver around K** — i.e. given the existing K detection, decide whether
the canonical block sequence runs in the `+theta` or `-theta` direction
from K. Three candidates, ordered by simplicity. Evaluate on the
diagnostic oracle from §5.2. Pick the simplest one that meets §6.4.

### 6.1 Candidate A — geometric width template around K (preferred)

This is a deterministic generalisation of SAM4Tun's Hough-sign K
detection: the same idea applied beyond K to the next 1–2 blocks.

K's neighbours have fixed canonical widths (rough order
`{B1≈1.7m, A1≈2.3m, A2≈2.3m, A3≈2.3m, A4≈2.3m, B2≈1.7m}` for family-4
rings, with a family-5 variant). Walk outward from detected K in both
`+theta` and `-theta`; correlate the empirical inter-boundary spacings
with the canonical width template and the reversed template; pick the
direction with higher score.

- Pure deterministic, requires only `boundaries_per_ring.json` and the K
  detection that the pipeline already produces.
- Confidence intrinsic: `S_direction = max(score_canon, score_rev) /
  (|score_canon - score_rev| + eps)`.
- Risk: rings with poor boundary detection (sparse, partial coverage)
  give a weak signal. Surface it as a low-confidence flag rather than
  forcing a flip; let §6.3 (per-tunnel inheritance) take over.

### 6.2 Candidate B — depth-signature cross-correlation (backup)

Reuse `methods/plans/scripts/landmark_k_anchor.py`: per tunnel, take a
BO-best calibration ring's depth signature `S_K_calib(t)` in both
directions, cross-correlate against the held-out depth map (gravity-
aligned), choose the direction with the higher peak.

- More code to revive and validate.
- Risk: requires gravity anchoring on, so stacks two flags.

### 6.3 Per-tunnel direction prior (always on, paired with A or B)

Borrow SAM4Tun §5.2.1's neighbour-inheritance idea but lift it to the
tunnel level: aggregate per-ring direction calls across all rings of one
tunnel; if ≥ 80% agree with confidence > τ, **lock** that direction for
the whole tunnel and override any low-confidence per-ring call. This is
cheap, deterministic, and what SAM4Tun proves is enough for K-y when a
single ring's joint detection is noisy.

### 6.4 Selection criterion

- Resolver direction-call accuracy ≥ 90% on the diagnostic oracle (§5.2)
  using **Candidate A + per-tunnel prior** as the primary path.
- Per-ring confidence (`S_direction`) correlates with mIoU lift after
  the flip (Spearman ≥ 0.4).
- ≤ 5 new fields in `parameters_detection.json` (we don't blow up the
  action space the LLM later sees).

## 7. Phase 3 — Implementation (~1.5 days)

### 7.1 Where the resolver lives

We want the stabiliser to be a single, named, deterministic step. Two
integration points are possible:

- **Option I — flip the unfolded image in preprocessing.** Re-emit the
  unfolded `depth_map_outlier.npy`, `final_visualization.png`, and downstream
  artefacts in canonical direction. Detection then runs unchanged.
- **Option II — flip the offset table at detection time.** Keep the unfolded
  image as is; choose between two pre-stored offset tables
  (`offsets_canonical` vs `offsets_reversed`) based on the resolver's call.

Option II is strictly less invasive (no resampling of the depth map, no risk
of reflowing intrinsics that are already calibrated against the unfolded
frame), so it is the default. Option I is held in reserve only if a downstream
intrinsic provably breaks under flipped offsets.

### 7.2 Code changes (Option II default)

- `bo/v3/rotation_stabiliser.py` (new): pure function that takes
  `boundaries_per_ring.json`, K detection, optional depth map, returns
  `direction ∈ {canonical, reversed}` and `confidence ∈ [0, 1]`. Also
  implements the §6.3 per-tunnel inheritance step (takes a list of
  per-ring calls, returns the locked tunnel direction).
- `agents/2_detection/2_detection.py`:
  - extend `parameters_detection.json` schema with
    `per_ring_offsets_reversed` (mirror of `per_ring_offsets`).
  - add a CLI flag / parameter `direction_resolver.mode ∈
    {auto, force_canonical, force_reversed}`; in `auto` it calls the new
    stabiliser and writes `rotation_stabiliser_meta.json` next to
    `boundaries_per_ring.json`.
- `bo/v3/freeze_for_llm.py`: surface
  `S_direction_confidence` as a deterministic stabiliser intrinsic (not a
  reflection knob).

### 7.3 Failure modes

- Resolver low-confidence (< τ) on a single ring: defer to per-tunnel
  prior (§6.3); if the prior is also unset, default to canonical and flag
  `direction=undetermined` in the intrinsic snapshot. This becomes a
  legitimate signal for the LLM reflection action space later.
- Resolver crash / missing inputs: behave like preprocessing failure — log,
  fall back to canonical, do not block the pipeline.

### 7.4 Tests

- Unit: synthetic boundaries that are exact mirrors must yield the same
  resolver score with opposite sign.
- Smoke: rerun the 3-ring Arm C MVP subset (4-3/r177, 4-2/r144,
  4-9/r363) — direction calls must agree with the §5.2 oracle.

## 8. Phase 4 — Re-evaluation (~0.5 day)

### 8.1 Re-run held-out under the stabilised pipeline

- New runner mode in `bo/v3/heldout_runner.py`:
  - `arm = a_unanchored` (pre-existing, unchanged baseline)
  - `arm = b_stabilised` (new: stabilised + canonical pred labels, no
    reflection — this is the "deterministic step alone")
  - `arm = c_stabilised_reflected` (stabilised + reflection; reuse existing
    Arm C code paths, only the inputs change)
- All outputs go to `logs/v3/rotation_stabiliser/heldout/<arm>/`.
- Artefacts: `scoreboard_<arm>.csv`, per-ring `evaluation.json`,
  `rotation_stabiliser_meta.json`.

### 8.2 Comparison report

- `bo/v3/render_rotation_tables.py`: extends the existing
  `render_heldout_tables.py` with the new arm and emits:
  - `papers/rotation_tables.tex`: per-arm summary, deterministic-step lift,
    per-ring trace for the resolved-direction subset.
  - `data/v3/heldout/rotation_report.md`: a self-contained narrative with
    sensitivity, limitations, confidence, exactly as required by the AIC
    methodology rule, plus an explicit comparison line to SAM4Tun (what
    we share, what we add).

### 8.3 Acceptance bar

To be set after Phase 1 diagnostic numbers come in (per the user's
choice on the §11 decision questions). A reasonable starting bar:

- Mean fixed-class mIoU on `arm = b_stabilised` strictly above
  `arm = a_unanchored`'s 0.255 by ≥ 0.05 absolute (i.e. ≥ 0.305).
- Bimodal mIoU split disappears in `arm = b_stabilised` (KS test against
  the canonical-only subset of `arm = a`).
- Resolver direction-call accuracy ≥ 90% on held-out, with the rings the
  resolver flagged low-confidence accounting for the residual.

## 9. Phase 5 — Paper rewrite (~0.5 day)

Only after §8 lands.

- `papers/main.tex` abstract: drop "bottom anchoring reaches 0.256". Replace
  with the actual lift number and step name from §8 (e.g., "deterministic
  K-anchored direction resolver reaches X.XXX, and assess-and-refine
  further lifts performance to Y.YYY").
- `papers/main.tex` §3.x: insert a methods subsection naming the stabiliser
  (e.g., "K-anchored direction resolver"), distinguishing it from gravity
  anchoring **and** explicitly contrasting with SAM4Tun (we replace SAM
  with a deterministic offsets table and therefore add an explicit
  direction signal that SAM absorbed implicitly).
- `papers/main.tex` §4.x and held-out tables: regenerate from §8.2 outputs.
- Add a limitations note: rings with low `S_direction_confidence` remain
  in the reflection action space; results are conditional on the
  canonical width prior holding for K-bearing segmental linings of
  family-4 / family-5 diameters.

## 10. Decision-point status

User has resolved the three pre-coding questions:

1. **Resolver candidate**: Candidate A (geometric width template) primary,
   B as fallback. **Locked.**
2. **Integration point**: decide after §6 once resolver confidence is
   profiled. **Deferred.**
3. **Acceptance bar (§8.3)**: set after Phase 1 diagnostic numbers come
   in. **Deferred.**

Two more decisions, post-SAM4Tun review:

a. **Per-tunnel direction prior (§6.3)**: **Hold until §8.** Implement
   only Candidate A in §7; turn on the per-tunnel inheritance
   *only if* §8.3's acceptance bar is missed by single-ring §6.1 alone.
   §6.3 stays in the design doc as a documented option, not as code.
b. **K-detection refresh**: **Audit first.** Keep v3's current Hough-sign
   K detection in §7. The §5.2 audit will measure the share of the
   residual that's caused by K-detection failure (vs direction failure).
   Port SAM4Tun's horizontal-fallback + neighbour-inheritance only if
   that share is meaningful (≥ 15% of held-out rings).

Implementation order is §5 → §6 → §7 → §8 → §9 with a hard stop after §5
if the diagnostic falsifier triggers.

## 11. Deliverables checklist

- `logs/v3/rotation_stabiliser/diagnose/direction_split.{csv,md}`
- `logs/v3/rotation_stabiliser/diagnose/direction_audit.csv`
- `bo/v3/rotation_stabiliser.py`
- Updated `agents/2_detection/2_detection.py` with `direction_resolver` CLI
- Updated `bo/v3/heldout_runner.py` with `b_stabilised` and
  `c_stabilised_reflected` arms
- `logs/v3/rotation_stabiliser/heldout/scoreboard_*.csv`
- `papers/rotation_tables.tex`
- `data/v3/heldout/rotation_report.md`
- Updated abstract and §3/§4 in `papers/main.tex`
