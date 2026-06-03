# Detection layout BO (`bo/`)

Design-time GP-BO over **layout-recovery variables** with GT mIoU labels.
All writes go to `logs/<run_id>/`; corpora under `data/` are read-only.

Prior proxy / transfer / held-out code archived under `stages/v9/bo/`.

## Layout

```
bo/
  run_layout_bo.py                  # single entry point (all BO trial modes)
  run_direction_select.py           # deploy: dual-run plus/minus branch selection
  check_experience_honesty_gate.py  # validate no oracle layout trials in pool
  lib/
    search_space.py                 # Step 2 layout-recovery bounds + encode/decode
    ceiling_gate.py                 # GT ceiling reference (design-time only)
    layout_bo.py                    # encode/decode, GP-BO, honest experience mode
    direction_select.py             # plus/minus branch scoring (every trial)
    manifest.py                     # manifest loading, panel summaries
    verify.py                       # GT round-trip smoke test
    perturbations.py                # forced perturbation candidates
```

## Honest experience policy (v3)

**GT labels trials; GT never fills in layout coordinates.**

| Allowed | Forbidden |
|---------|-----------|
| GT mIoU as trial objective | `gt_layout` / `gt_layout_*` as trial kinds |
| `ceiling.json` for regret reference | Perturbation anchor from GT layout |
| `gt_layout.json` in corpus (read-only doc) | Oracle layout injected as warm-start |

**Warm-start kinds:** `geometric_0`, `geometric_1`, `intrinsic_r_otsu`, `random`  
**Every trial:** detection → `direction_select` (plus/minus) → segmentation  
**Held-out / Stage-A:** `run_held_out_score.py` → `candidate_eval.evaluate_candidate` (same path; per-ring `direction_select_gate.json`)  
**Canonical pool:** `logs/bo_experience_v3/bo_trials.csv` (480 trials, 0 oracle)

## Search space (Step 2)

`x = [k_y_frac, off_frac[K], off_frac[B1], …, layout_param_frac…, r_surface_min_frac]`

| Variable | Parameter | Stage |
|----------|-----------|-------|
| K position | `k_y_positions` | detection |
| A/B offsets | `per_ring_offsets` | detection |
| Oblique-line Hough threshold | `hough_threshold` | detection |
| Horizontal-line Hough threshold | `hough_horizontal_threshold` | detection |
| Line merge distance | `merge_distance_threshold` | detection |
| Line snapping tolerance | `single_ring_visual_slot_snap_px` | detection |
| Segmentation padding / crop | `slot_inset_y` | segmentation |
| Radial surface cutoff | `r_surface_min` | segmentation |

Bounds and defaults: `bo/lib/search_space.py`. Written per run to `search_space.json`.

## Commands

**Single-instance gate (run before full panel):**

```bash
./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo_calibration/MANIFEST.json \
  --source-dir data/bo_calibration \
  --run-root logs/bo_experience_v3 \
  --only-ring 1-4/r206
```

**Full 480-trial panel:**

```bash
./venv/bin/python bo/run_layout_bo.py experience \
  --manifest data/bo_calibration/MANIFEST.json \
  --source-dir data/bo_calibration \
  --run-root logs/bo_experience_v3
```

**Honesty gate (post-run):**

```bash
./venv/bin/python bo/check_experience_honesty_gate.py \
  --run-root logs/bo_experience_v3
```

## Corpora

| Corpus | Path | Typical mode |
|---|---|---|
| BO calibration (Step 1 panel) | `data/bo_calibration/MANIFEST.json` | `experience` |
| Minimum (hard cases) | `data/minimum/MANIFEST.json` | `ceiling-push` |

## Outputs per ring

`logs/<run_id>/<tunnel>/r<N>/`: `search_space.json`, `bo_trials.csv`, `ceiling.json`, `gt_layout.json` (reference only), `best_bo_trial.json`, `experience_gate.json`, `direction_selection.json` (per trial in sandbox).

## Ring site params (required before BO)

Every ring must have pre-defined `segment_count` (6 or 7) and `tunnel_diameter` in `data/ring_site_params.json`.

```bash
./venv/bin/python bo/build_ring_site_params_registry.py
./venv/bin/python bo/confirm_ring_site_params.py --ring-keys 1-4/r206 4-1/r116
```
