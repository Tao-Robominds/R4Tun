# 01 Ring-Regime Discovery (preprocessing BO only)

## Goal

Select **preprocessing-only** representative rings for Bayesian optimization and held-out generalisation. **Detection** uses a **separate** representative-ring set in a later step; this document does not define a shared panel, and preprocessing BO must **not** optimize detection-stage behavior (no K localization, segment order, line quality, or segmentation metrics).

## Reference pool

Use the **canonical ring corpus**, not tunnel-wide subset clouds:

- Raw per-ring point clouds: `data/rings/<tunnel_underscores>_ring<id>.txt` (299 rings across 30 active tunnels aligned with `r4tun/references/data/`).
- Baseline preprocessing outputs (r4tun tunnel-best warm params): `data/{tunnel_id}/r{ring_id}/` (`unwrapped.csv`, `denoised.csv`, `enhanced.csv`, `depth_map.npy`, `depth_map_outlier.npy`, etc.).
- Intrinsic QA for **ring selection / triage only** (not BO objective): `data/preprocessing_qa/report.md`, `data/preprocessing_qa/corpus_metrics.json`.

Do **not** use `data/subsets/*.txt` as the primary reference pool for this step. Do not pull rings from `r4tun/data/` for discovery beyond reading tunnel-best reference metadata when needed.

## Runtime path

`methods/plans/output/{run_id}/01_ring_regime_discovery/`

(Aligns with project convention: plan artifacts under `methods/plans/output/`.)

## Inputs

- `data/rings/*.txt` — per-ring point clouds (include `segment`, `ring` for GT support construction only).
- `data/{tunnel}/r{ring}/` — baseline preprocessing artifacts after `agents/1_preprocessing/scripts/run_all_rings.py`.
- `data/preprocessing_qa/report.md` / `corpus_metrics.json` — baseline-poor ring list and intrinsic QA reasons.
- Optional: `data/rings/preprocessing_log.csv` — per-run status and `nan_ratio` for context.

## Descriptor coverage (preprocessing-relevant only)

Compute or aggregate descriptors that affect **unfolding, denoising, interpolation, and rasterization**:

- Point density (per-ring point count).
- Angular coverage and largest angular gaps (from ring geometry).
- Radial coverage / radius spread (e.g. distribution of `r` after unwrapping when available).
- Ring width / height or depth-map shape (`depth_map.npy` dimensions).
- Valid foreground support and raw sparsity proxies (e.g. finite depth ratio, batch `nan_ratio`) — **diagnostics**, not the BO score.
- Baseline **QA failure reason** from intrinsic QA (`dominant_empty_component`, `near_empty_valid_ratio`, `many_empty_row_bands`, …) for **sampling** baseline-poor representatives.

**Do not** use detection-oriented regime criteria for preprocessing BO ring selection: **no** K position/span, **no** walking order, **no** segment balance, **no** inferred block layout or segment-sequence rules.

## Failure-driven selection (baseline-poor, not missing outputs)

The ~23 rings marked **FAIL** in `data/preprocessing_qa/report.md` **do have** preprocessing outputs; FAIL means **poor quality under intrinsic QA** with r4tun-best tunnel parameters, not a missing pipeline result.

**Policy:**

1. Group FAIL rings by **distinct QA `reason`** (recoverable failure pattern).
2. Select **one representative ring per distinct reason** to capture recovery experience in BO (do **not** include all FAIL rings).
3. For each such representative, add **control** ring(s): same tunnel where possible, similar preprocessing geometry (density/coverage/radius spread), but **PASS** or acceptable baseline under QA — so BO contrasts failure vs success in context.

Also include **normal** representatives stratified across preprocessing geometry (density, coverage, radius spread, map size, raw sparsity), independent of the failure set.

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
2. Compute **preprocessing-relevant** descriptors (see above); **omit** K span, walking order, segment balance, and detection pattern labels.
3. Merge **intrinsic QA** outcomes: tag baseline-poor rings and `reason`; pick **one ring per distinct `reason`** plus **controls**.
4. Stratify **normal** representatives across geometry regimes; reserve a **holdout** set for generalisation (no overlap with BO panel).
5. Ensure each BO candidate ring has the artifacts needed to compute `foreground_mask_iou` (depth map outputs plus point-level foreground labels and pixel mapping), and document any blockers.

## Run command (legacy detection-oriented script)

The existing regime builder is **detection-oriented** (K span, walking order, segment balance). For **preprocessing-only** panels, either extend tooling or run a dedicated script that reads `data/rings` and writes artifacts under `methods/plans/output/{run_id}/01_ring_regime_discovery/`. Until a dedicated CLI exists, implement the actions above in a small script or notebook and record `run_id` in `preprocessing_ring_selection_summary.md`.

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

**Preprocessing BO execution** (per ring, after panel is fixed) should use project venv, ring roots under the current preprocessing dataset, and maximize `foreground_mask_iou` directly (single-reward BO), with other metrics logged as diagnostics.

## Operational pointers (current fixed baseline + BO workflow)

- Fixed B+C+D baseline reference is maintained under `logs/context_preprocessing_v1/<tunnel>/r<ring>/` (read-only snapshot for comparisons).
- Official B+C+D runner: `agents/1_preprocessing/scripts/run_context_ring_trial.py`.
- Guarded BO runner: `bo/run_preprocessing_iou_bo.py`.
- Current guarded BO trial outputs are placed under `data/bo/preprocessing/<tunnel>/r<ring>/...`; run summaries remain under `logs/preprocessing_context_bo/<run_id>/<tunnel>/r<ring>/`.
- For rationale, reward design, and representative outcomes, see `methods/journals/journal_2026-05-01_ring_context_preprocessing_bo.md`.

## Outputs (preprocessing-only artifacts)

Under `methods/plans/output/{run_id}/01_ring_regime_discovery/`:

- `preprocessing_ring_descriptors.csv`
- `preprocessing_failure_cases.csv` (distinct `reason` → chosen representative, optional alternates)
- `preprocessing_gt_metrics_baseline.csv` (per candidate: GT-derived metrics at r4tun-best params)
- `preprocessing_bo_panel.json`
- `preprocessing_holdout.json`
- `preprocessing_ring_selection_summary.md`

## Verify prompt

Are all **299** canonical rings described with **preprocessing-relevant** descriptors, baseline-poor cases sampled as **one representative per distinct QA failure reason** (with controls), preprocessing BO and holdout panels **stage-specific** (preprocessing only), and BO maximization based on the single GT-derived reward `foreground_mask_iou` (with other metrics diagnostic-only)?
