# Daily Journal — 30 April 2026 (Pre-BO checkpoint: `pre-b0`)

## Objective

Freeze the pipeline state **before** stage-wise Bayesian optimization: canonical depth raster aligned to GT ceiling labelmaps, detection emitting comparable `labelmap` artifacts, smoke evaluation on the regime panel, and an optional **LLM warm start** of preprocessing + detection parameters per `regime_label`. This journal records what is implemented, where outputs live locally, and what comes next.

---

## Canonical raster and detection outputs

- **Preprocessing** (`agents/1_preprocessing/1_preprocessing.py`): fixed vertical resolution and bounds so the depth map matches the grid implied by `gt_ceiling/labelmap.npy`; writes `depth_map.npy` and enforces canonical `depth_map_resolution` when callers override it.
- **Detection** (`agents/2_detection/2_detection.py`): rasterizes polylines into **`detection/labelmap.npy`**, **`labelmap.png`**, and **`labelmap_meta.json`** for apples-to-apples comparison with GT ceiling labelmaps.

Supporting scripts live under `methods/ablation/scripts/` (e.g. `eval_detection_vs_gt.py`, `run_detection_smoke.py`, `_labelmap_viz.py`).

---

## Evaluation convention (GT-dependent, design-time)

- **Metric**: mIoU between detection `labelmap` and `gt_ceiling/labelmap.npy` on the six-ring BO calibration panel (from regime workflow JSON).
- **Baseline (static JSON)**: median mIoU ≈ **0.112** on that panel (full table in `data/ablation/detection_baseline_report.md` when generated locally).
- **GT ceiling upper bound** (raw unwrap + dominant label per cell, separate experiment): median mIoU ≈ **0.99** on the same rings — documented in `journal_2026-04-30_ring_pipeline_and_gt_ceiling.md`.

---

## LLM warm start (per regime)

- **Runner**: `methods/ablation/scripts/run_warm_start_llm.py` — reads regime aggregates from **`ring_regimes.csv`** (not `ring_descriptors.csv`; the former carries `regime_label` / `k_span_tier`), calls the configured provider with **temperature 0**, validates and clamps against `methods/ablation/scripts/_warm_start_schema.py`, and forces canonical-raster constraints.
- **Secrets**: `.env` (gitignored); default provider **Anthropic**, model **`claude-sonnet-4-6`** (older model IDs 404’d with current API).
- **Written artifacts**: `agents/1_preprocessing/parameters/_warm_start/{regime_label}/parameters_preprocessing.json`, `agents/2_detection/parameters/_warm_start/{regime_label}/parameters_detection.json`, plus provenance under `methods/ablation/output/warm_start_v1/`.
- **Load path**: both agent scripts accept `--regime-label` and resolve `_warm_start/{regime_label}` after the default tunnel JSON.
- **Result vs baseline**: end-to-end smoke with `--params-set warm_start` did **not** materially move panel mIoU vs static defaults (see `data/ablation/baseline_vs_warm_start_report.md` when generated locally). Warm start is still useful as a **structured prior** and audit trail for BO.

**Dependencies** (pinned in `requirements.txt`): `anthropic`, `google-generativeai` (multi-provider hook).

---

## How to reproduce (local)

All experiment JSON/MD under `data/ablation/` is **gitignored**; regenerate after clone:

```bash
./venv/bin/python methods/ablation/scripts/run_detection_smoke.py --params-set default
./venv/bin/python methods/ablation/scripts/run_warm_start_llm.py   # optional; needs .env
./venv/bin/python methods/ablation/scripts/run_detection_smoke.py --params-set warm_start
```

---

## Next step

**Step 02 — BO calibration**: run per-stage BO (or joint where defined) using regime panels, optionally initializing search from `_warm_start/{regime_label}/`, and log trials under `logs/{tunnel_id}/` without overwriting prior `data/baseline` or `data/bo` trees.

---

## Commit marker

Repository state at this milestone is tagged in git as: **`init: pre-b0`**.
