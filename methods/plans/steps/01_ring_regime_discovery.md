# 01 Ring Dataset Selection

## Goal

Separate the ring corpus into three scientifically defensible dataset roles:

1. **BO calibration / reference set** — small representative panels for expensive stage-wise BO and tuning experience.
2. **Proxy / trigger validation set** — medium labelled validation set for choosing `T_depth` and `T_boundary`.
3. **Held-out reflection test set** — larger isolated paired-test set for proving reflection improves final mIoU/OA.

The current small representative set under `data/represents/` is treated as a **read-only pilot BO/reference set**. Do not use it to claim reflection effectiveness, and do not delete or reorganize it in place. New selection artifacts must be generated under the sandbox path below.

## Reference pool

Use the **canonical ring corpus**, not tunnel-wide subset clouds:

- Raw per-ring point clouds: `data/rings/<tunnel_underscores>_ring<id>.txt` (299 rings across 30 active tunnels aligned with `r4tun/references/data/`).
- Baseline preprocessing outputs (r4tun tunnel-best warm params): `data/{tunnel_id}/r{ring_id}/` (`unwrapped.csv`, `denoised.csv`, `enhanced.csv`, `depth_map.npy`, `depth_map_outlier.npy`, etc.).
- Intrinsic QA for **ring selection / triage only** (not BO objective): `data/preprocessing_qa/report.md`, `data/preprocessing_qa/corpus_metrics.json`.

Do **not** use `data/subsets/*.txt` as the primary reference pool for this step. Do not pull rings from `r4tun/data/` for discovery beyond reading tunnel-best reference metadata when needed.

## Runtime path

Sandbox path: `logs/{run_id}/01_ring_regime_discovery/`

Aggregate panel summaries may be written under `logs/{run_id}/01_ring_regime_discovery/`.

## Inputs

- `data/rings/*.txt` — per-ring point clouds (include `segment`, `ring` for GT support construction only).
- `data/{tunnel}/r{ring}/` — baseline preprocessing artifacts after `agents/1_preprocessing/scripts/run_all_rings.py`.
- `data/preprocessing_qa/report.md` / `corpus_metrics.json` — baseline-poor ring list and intrinsic QA reasons.
- Optional: `data/rings/preprocessing_log.csv` — per-run status and `nan_ratio` for context.

## Dataset roles

| Dataset role | Target size | Purpose | Used for statistical proof? |
|--------------|------------:|---------|-----------------------------|
| BO calibration / reference set | 12–18 rings preferred; 8 acceptable for pilot | Run expensive stage-wise BO, obtain reference parameters and tuning memory | No |
| Proxy / trigger validation set | 30–50 rings | Check `S_depth` / `S_boundary` and choose trigger thresholds | No |
| Held-out reflection test set | 50–80 rings preferred; 30 minimum | Paired proof that reflection improves final mIoU/OA | Yes |
| Qualitative case studies | 4–6 rings | Visual explanation of improvements and failures | No |

Core distinction:

- **40 validation rings = design / threshold selection set.**
- **60+ held-out reflection rings = final proof set.**

These roles must never collapse into one set. Validation rings may influence the method; held-out reflection rings must not influence anything after the method is frozen.

If compute is limited, use the fallback split:

- 8 BO rings.
- 30 threshold-validation rings.
- 30 held-out reflection-test rings.

For a stronger paper target:

- 12–18 BO rings.
- 40 threshold-validation rings.
- 60+ held-out reflection-test rings.

## Ring descriptor vector

Compute a descriptor row for every candidate ring before selecting any split:

- Point density (per-ring point count).
- Point-count percentile within the corpus.
- Angular coverage and largest angular gaps (from ring geometry).
- Valid ratio / visible coverage.
- Empty-band ratio.
- Distance-to-scanner proxy when derivable from geometry.
- Diameter proxy / segment number.
- Assembly type: staggered or continuous.
- Boundary clarity.
- K-boundary clarity.
- Boundary spacing regularity.
- Noise / non-structural interference proxy.
- Baseline QA failure reason from intrinsic QA (`dominant_empty_component`, `near_empty_valid_ratio`, `many_empty_row_bands`, etc.).

Normalize numeric descriptors before max-min sampling. Keep categorical fields as stratification labels.

## Regime stratification

Split the ring universe into the main observed tunnel regimes:

- Small-diameter 6-segment staggered.
- Small-diameter 6-segment continuous.
- Large-diameter 7-segment staggered / complex.

Within each regime, preserve coverage of:

- Low-density / sparse rings.
- Medium-density / normal rings.
- High-density / cluttered rings.
- Strong empty-band / partial-coverage rings.
- Weak-boundary / difficult K-line rings.

## Shared BO calibration panel

Preprocessing and detection/boundary BO use the **same 8–12 BO calibration rings**. The two BO stages still optimize different parameters and rewards, but they run on the same representative ring panel so stage comparisons and tuning experience are easier to interpret.

| BO set | Target size | Selection pressure |
|--------|------------:|--------------------|
| Shared BO calibration rings | 8–12 | Joint coverage of density, visible coverage, empty-band extremes, projection stability, K-boundary clarity, line continuity, spacing regularity, and boundary contrast |

Use max-min selection within strata so chosen rings are far apart in descriptor space. This supports the paper statement:

> BO rings were selected by stratified max-min sampling over ring descriptors, rather than by random selection, to cover the main observed regimes of density, visibility, boundary clarity, assembly type, and tunnel geometry.

Preprocessing BO must not optimize detection-stage behavior: no K localization, segment order, line quality, or segmentation metrics in the preprocessing reward. Detection/boundary BO must not rewrite preprocessing artifacts outside its sandbox. The **rings are shared**, but the **objectives and stage outputs are separate**.

## Failure-driven selection (baseline-poor, not missing outputs)

The ~23 rings marked **FAIL** in `data/preprocessing_qa/report.md` **do have** preprocessing outputs; FAIL means **poor quality under intrinsic QA** with r4tun-best tunnel parameters, not a missing pipeline result.

**Policy:**

1. Group FAIL rings by **distinct QA `reason`** (recoverable failure pattern).
2. Select **one representative ring per distinct reason** to capture recovery experience in BO (do **not** include all FAIL rings).
3. For each such representative, add **control** ring(s): same tunnel where possible, similar preprocessing geometry (density/coverage/radius spread), but **PASS** or acceptable baseline under QA — so BO contrasts failure vs success in context.

Also include **normal** representatives stratified across preprocessing geometry (density, coverage, radius spread, map size, raw sparsity), independent of the failure set.

## Proxy / trigger validation set

Select a medium labelled set after BO candidates are excluded. The validation set is allowed to influence the method because it is used **after BO but before final testing**.

- Target 30–50 rings.
- Use 40 rings for the recommended design.
- Stratify by density level, empty-band severity, boundary clarity, K-boundary clarity, segment pattern, tunnel diameter, 6- vs 7-segment layout, and easy/medium/hard difficulty.
- Include enough bad or borderline cases to choose useful `T_depth` and `T_boundary`.
- Use this set to compute Spearman checks, threshold-trigger metrics, and practical deployment rules.
- Do not use this set for BO optimization or final reflection proof.

Validation may choose:

- `T_depth`: threshold for triggering preprocessing reflection.
- `T_boundary`: threshold for triggering boundary reflection.
- Reflection budget: maximum corrective passes.
- Routing rule: which agent reflects when each proxy fails.
- Prompt or few-shot refinements used by the frozen reflection method.

Threshold validation outputs:

- `S_depth`, `S_boundary`.
- `S_depth = S_coverage * S_empty`.
- `S_boundary = S_continuity * S_K * S_spacing * S_layout_coverage`.
- Final mIoU after reprojection.
- Final OA.
- Whether reflection would have been triggered.
- Whether the ring was actually bad.
- Bad-case label `G_bad = 1[mIoU < tau]`.
- Bad-case recall, trigger precision, false-negative rate, accepted-case mIoU.

Validation questions:

- If `S_depth` is low, does final mIoU tend to be poor?
- If `S_boundary` is low, does final mIoU tend to be poor?
- What thresholds catch most bad cases without triggering reflection everywhere?

Validation experiment:

| Step | Action |
|------|--------|
| 1 | Run fixed or BO-informed pipeline without using GT during inference. |
| 2 | Compute `S_depth` and `S_boundary`. |
| 3 | Compare combined proxy scores with final mIoU. |
| 4 | Choose `T_depth` and `T_boundary`. |
| 5 | Freeze all thresholds, rules, prompts, examples, and reflection budget. |

Use `tau` as the bad-case mIoU threshold, for example `tau = 0.60`, with `G_bad = 1[mIoU < tau]`.

Main validation metrics:

- `Spearman(S_depth, final_mIoU)`.
- `Spearman(S_boundary, final_mIoU)`.
- Bad-case recall.
- Trigger precision.
- False-negative rate.
- Accepted-case mIoU.

## Held-out reflection test set

Choose this set after BO and trigger thresholds are frozen. The held-out reflection test set must not influence the method; it only answers whether the frozen reflection system improves final segmentation on unseen rings.

- Minimum 30 rings.
- Preferred 60+ rings; 50–80 rings across multiple tunnels or scanning stations is the target range.
- Enrich for cases where reflection has a reason to activate, while retaining easy and medium controls.

Never change the following after entering held-out testing:

- BO reference parameters.
- Few-shot examples or tuning memory.
- `S_depth` formula.
- `S_boundary` formula.
- `T_depth`.
- `T_boundary`.
- Reflection prompts.
- Routing rules.
- Reflection budget.

Recommended composition:

| Group | Suggested count |
|-------|----------------:|
| Easy / near-reference rings | 10–15 |
| Medium difficulty rings | 15–25 |
| Hard / sparse / off-reference rings | 20–40 |
| Continuous assembly cases | at least 10 if available |
| 7-segment large-diameter cases | at least 10–20 |

For each held-out ring, run paired variants on the same ring:

- `A0`: no reflection baseline.
- `A1`: proposed reflection with frozen proxy trigger.
- `A2`: always reflect with the same budget.
- `A3`: random trigger or wrong-stage trigger.
- `A4`: oracle BO-best upper bound, not deployable.

The random/wrong-trigger control checks whether routing is meaningful. Without it, reviewers can argue that any gain comes from simply spending another corrective pass rather than from the intrinsic trigger.

Report paired statistics:

- Mean and median `delta_mIoU`.
- 95% confidence interval.
- Paired t-test p-value.
- Wilcoxon signed-rank p-value.
- Improved / unchanged / worsened ring counts.
- Reflection trigger rate.
- Runtime / corrective-pass cost.
- Cluster bootstrap by tunnel/station, or mixed-effects model with tunnel/station as random effect.

Use paired deltas:

- `delta_i = mIoU_reflection_i - mIoU_no_reflection_i`.
- Also report paired OA changes where available.

This is the only dataset that supports the statistical claim: "reflection improves unseen cases."

## Leakage rule

If `T_depth`, `T_boundary`, prompts, few-shot examples, routing, or reflection budget are chosen using the same rings later used to claim improvement, the method has been tuned to the test set. Keep the split strict:

- BO calibration rings: tune stage parameters and collect experience.
- Validation rings: choose and freeze thresholds, routing, prompts, and reflection budget.
- Held-out reflection rings: run frozen paired comparisons only.
- Qualitative cases: explain behavior, not prove performance.

One-sentence paper framing:

> The validation rings are used to choose and freeze the intrinsic-proxy thresholds and reflection rules; the held-out reflection rings are used only after freezing the method, to test by paired comparison whether reflection improves final mIoU/OA on unseen rings.

## Preprocessing BO objective (GT-derived, preprocessing-only)

**BO must optimize preprocessing quality using one simple GT-derived reward only.** Use point-cloud labels (`segment`) as foreground/background support for the depth-map validity mask.

**Only BO reward**:

- `foreground_mask_iou = TP / (TP + FP + FN)`, where:
  - `valid_mask = isfinite(depth_map) & (depth_map > 0)`
  - `gt_fg_mask` is foreground support derived from point-cloud labels (segment > 0) on the same depth-map pixel grid
  - `TP = valid_mask & gt_fg_mask`, `FP = valid_mask & ~gt_fg_mask`, `FN = ~valid_mask & gt_fg_mask`

**Explicit exclusions from preprocessing BO:** K localization, segment ordering, segment mIoU, line detection quality, any downstream detection or segmentation metric.

`foreground_support_ratio`, `largest_fg_hole_ratio`, `overfill_ratio`, `valid_ratio`, and intrinsic empty-space metrics (`largest_empty_ratio`, empty row/column bands) are **diagnostics and triage only** — **not** BO rewards.

**Using `segment`:** treat as **foreground/background support** for depth-map evaluation only; do not use it to reason about K position, segment sequence, block balance, or detection correctness in this step.

## Actions

1. Enumerate all canonical `(tunnel_id, ring_id)` pairs from `data/rings/*.txt`.
2. Compute the descriptor vector for every candidate ring.
3. Assign regime and difficulty labels.
4. Build one shared 8–12 ring BO calibration panel with stratified max-min sampling.
5. Use the same BO calibration ring IDs for preprocessing BO and detection/boundary BO, while keeping each stage objective and artifact directory separate.
6. Exclude BO rings from threshold and held-out pools.
7. Select the proxy / trigger validation set with stratified sampling.
8. Freeze BO and threshold design, then select the held-out reflection test set.
9. Select 4–6 qualitative case studies without replacing statistical test samples.
10. Ensure every selected ring has artifacts needed for its role, and document blockers.

## Run command

Use `./venv/bin/python` and write all generated artifacts under `logs/{run_id}/01_ring_regime_discovery/`. Do not write to `data/represents/`, `methods/plans/output/`, or any other protected corpus/archive path.

**Historical reference** (subset-based, **not** the primary path for preprocessing BO):

```bash
./venv/bin/python methods/ablation/scripts/build_ring_regimes.py \
  --subsets-dir data/subsets \
  --run regime_v1 \
  --families 4 5 \
  --regular-families 1 2 3 \
  --regular-ratio 0.20 \
  --panel-size 30 \
  --holdout-per-regime 1
```

**Preprocessing BO execution** (per ring, after panel is fixed) must use `./venv/bin/python`, copy any needed read-only baseline files into the sandbox, and maximize `foreground_mask_iou` directly (single-reward BO), with other metrics logged as diagnostics.

## Operational pointers (current fixed baseline + BO workflow)

- Fixed B+C+D baseline reference is maintained under `logs/context_preprocessing_v1/<tunnel>/r<ring>/` (read-only snapshot for comparisons).
- Official B+C+D runner: `agents/1_preprocessing/scripts/run_context_ring_trial.py`.
- Guarded BO runner: `bo/run_preprocessing_iou_bo.py`.
- Guarded BO trial outputs must be placed under `logs/<run_id>/<tunnel>/r<ring>/...`.
- For rationale, reward design, and representative outcomes, see `methods/journals/journal_2026-05-01_ring_context_preprocessing_bo.md`.

## Outputs (preprocessing-only artifacts)

Under `logs/{run_id}/01_ring_regime_discovery/`:

- `ring_descriptors.csv`
- `selection_manifest.json`
- `bo_unified_panel.json`
- `bo_preprocessing_panel.json`
- `bo_boundary_panel.json`
- `bo_overlap_panel.json`
- `proxy_threshold_validation_set.json`
- `heldout_reflection_test_set.json`
- `qualitative_case_studies.json`
- `selection_summary.md`
- `preprocessing_failure_cases.csv` (distinct `reason` → chosen representative, optional alternates)

## Verify prompt

Are all canonical rings described, dataset roles separated, preprocessing and boundary BO panels partially overlapping but stage-specific, threshold-validation and held-out test rings isolated from BO, paired reflection-test variants defined, and all generated artifacts targeted to `logs/{run_id}/01_ring_regime_discovery/`?
