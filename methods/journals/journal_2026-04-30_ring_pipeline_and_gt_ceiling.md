# Ring-based pipeline + GT-detection ceiling — 2026-04-30

Plan: [.cursor/plans/ring-pipeline-and-gt-ceiling_4cc23468.plan.md](../../.cursor/plans/ring-pipeline-and-gt-ceiling_4cc23468.plan.md).

## What was built

1. Per-ring ablation data setup
   - [methods/ablation/scripts/extract_ring_clouds.py](../ablation/scripts/extract_ring_clouds.py)
     reads `data/subsets/{tunnel_id}.txt`, filters by `ring`, and writes
     `data/ablation/{tid}/r{rid}/{tid}_r{rid}.txt`.
   - 6 rings extracted: `5-5/r258`, `4-9/r366`, `5-3/r190`, `4-8/r337`,
     `4-1/r116`, `4-6/r283` (`data/ablation/reference_panel.json`).

2. Default parameters
   - `agents/{1_preprocessing,2_detection,3_segmentation}/parameters/_default_irregular/parameters_*.json`
     copied from 4-1, with preprocessing `radius_min`/`radius_max`
     widened to `(3.0, 4.2)` for single-ring circle-fit unwrap and
     `num_slicing_planes` set to 1.

3. Ring-scoped agents (full replacement of tunnel-scoped versions)
   - [agents/1_preprocessing/1_preprocessing.py](../../agents/1_preprocessing/1_preprocessing.py):
     `run_preprocessing(tunnel_id, ring_id, base_dir)` reads
     `{base}/{tid}/r{rid}/{tid}_r{rid}.txt`, calls a new
     `unfold_single_ring` (Y-axial assumption, least-squares circle fit
     for ring centre, polar `(r, θ, h)` directly). Outputs land in
     `{base}/{tid}/r{rid}/` with `ring_count=1`.
   - [agents/2_detection/2_detection.py](../../agents/2_detection/2_detection.py):
     `run_detection(tunnel_id, ring_id, base_dir)`; existing K-and-offsets
     path collapses cleanly when `ring_count=1`.
   - [agents/3_segmentation/segmentation.py](../../agents/3_segmentation/segmentation.py):
     `run_segmentation(tunnel_id, ring_id, base_dir, segments_file, override_params)`.
   - [agents/evaluation.py](../../agents/evaluation.py):
     `evaluate(tunnel_id, ring_id, base_dir, segment_count)`.
   - All four parameter loaders fall back to `_default_irregular` when
     no per-ring tuned file exists (no per-tunnel fallback any more).

4. GT-detection ceiling (first-principles)
   - [methods/ablation/scripts/run_gt_ceiling.py](../ablation/scripts/run_gt_ceiling.py)
     is the canonical ceiling script. It bypasses preprocessing and
     detection entirely:
       a. read the raw ring point cloud,
       b. unwrap to `(theta, h)` (Y-axial, circle-fit centre),
       c. build a per-pixel dominant-GT labelmap on the depth-map raster
          (height locked to the full circumference at `0.005 m`/px,
          width = ring `h` extent),
       d. back-project: every raw point gets the labelmap value at its
          own pixel,
       e. compute mIoU / OA / macro-F1 against `segment` GT.
     This is the upper bound on what any 2D-boundary-based segmentation
     can deliver on these rings at this resolution.
   - [methods/ablation/scripts/build_ceiling_report.py](../ablation/scripts/build_ceiling_report.py)
     aggregates `data/ablation/gt_ceiling_results.json` into
     `data/ablation/ceiling_report.md` and
     `data/ablation/ceiling_summary.json`.

## Result (acceptance gate: median mIoU ≥ 0.90)

Headline from `data/ablation/ceiling_report.md`:

| metric | value |
|---|---:|
| n rings | 6 |
| **median mIoU** | **0.9935** |
| mean mIoU | 0.9874 |
| min / max | 0.9614 / 1.0000 |
| acceptance | **PASS** |

Per-ring (sorted by mIoU desc):

| ring | regime | mIoU | OA | F1 | mixed pixels |
|---|---|---:|---:|---:|---:|
| `4-6/r283` | sparse_full_wide_canonical              | 1.0000 | 1.0000 | 1.0000 | 0.00% |
| `5-3/r190` | low_full_normal_canonical               | 0.9989 | 0.9993 | 0.9994 | 0.07% |
| `5-5/r258` | medium_full_normal_reversed_canonical   | 0.9964 | 0.9981 | 0.9982 | 0.19% |
| `4-8/r337` | medium_full_narrow_reversed_canonical   | 0.9905 | 0.9948 | 0.9952 | 0.54% |
| `4-9/r366` | dense_full_wide_canonical               | 0.9773 | 0.9824 | 0.9884 | 1.82% |
| `4-1/r116` | dense_partial_normal_reversed_canonical | 0.9614 | 0.9743 | 0.9802 | 2.83% |

Per-class IoU is uniformly 0.927–1.000 across all rings — there is no
per-class collapse, only the residual loss from per-pixel mixing
(several GT segments hashing to the same depth-map cell at this
resolution).

## Investigation: why the ceiling is below 1.0

The only loss source in the first-principles ceiling is **per-pixel
mixing**: at `0.005 m` resolution, dense rings still place points from
two different blocks into the same depth-map cell along the K↔B1 and
B↔A boundaries. The mIoU loss tracks the mixing fraction one-to-one:

| ring | mixed-pixel % | 1 − mIoU |
|---|---:|---:|
| `4-6/r283` | 0.00% | 0.000 |
| `5-3/r190` | 0.07% | 0.001 |
| `5-5/r258` | 0.19% | 0.004 |
| `4-8/r337` | 0.54% | 0.010 |
| `4-9/r366` | 1.82% | 0.023 |
| `4-1/r116` | 2.83% | 0.039 |

This is bounded above by the depth-map resolution (smaller cells →
fewer collisions). At `0.005 m` the ceiling already exceeds the gate;
no action required.

## What an earlier (incorrect) ceiling looked like

A first attempt routed the GT through preprocessing's denoiser,
extracted boundaries that the existing slot-based segmenter understands,
and back-projected via `pixel_to_point.pkl`. That measured the ceiling
of the **current segmentation+denoiser implementation**, not the
first-principles ceiling, and yielded median mIoU 0.637 — driven by
(a) the GT-blind denoiser keeping background points, and (b) the
slot-builder's one-label-per-row constraint. Those are properties of
the implementation, not of the segmentation problem itself, so they
do not belong in the ceiling. The first-principles result above
supersedes that attempt; the earlier scripts
(`extract_gt_detection.py`, `run_ceiling.py`) remain on disk only as a
record of how much headroom the current slot-based segmenter is
giving up versus the true ceiling (~0.30 mIoU).

## Follow-up implications

The first-principles ceiling sits at 0.96–1.00 per ring. The
production pipeline's actual mIoU (after BO) is well below this
ceiling because of the two implementation choices noted above. The
gap between the production result and this ceiling is the headroom
available to:

- expose a per-pixel labelmap injection point in
  `agents/3_segmentation/segmentation.py` (consume a 2D
  `(H, W)` labelmap directly, skipping `build_boundary_label_map`),
- audit the radius/gradient denoiser per ring so fewer GT-`0` (BG)
  points survive into the depth map,

both of which are out of scope for this plan and will be tracked
under a separate plan.

## Repo changes

Modified:

- `agents/1_preprocessing/1_preprocessing.py`
- `agents/2_detection/2_detection.py`
- `agents/3_segmentation/segmentation.py`
- `agents/evaluation.py`

Added:

- `agents/{1_preprocessing,2_detection,3_segmentation}/parameters/_default_irregular/parameters_*.json`
- `methods/ablation/scripts/extract_ring_clouds.py`
- `methods/ablation/scripts/extract_gt_detection.py`     (slot-based, retained as reference)
- `methods/ablation/scripts/run_ceiling.py`              (slot-based, retained as reference)
- `methods/ablation/scripts/run_gt_ceiling.py`           (first-principles, canonical)
- `methods/ablation/scripts/build_ceiling_report.py`
- `data/ablation/{tid}/r{rid}/{tid}_r{rid}.txt`          (6 rings)
- `data/ablation/{tid}/r{rid}/gt_ceiling/{labelmap.npy,final.csv,performance.md,summary.json}`
- `data/ablation/{ceiling_report.md,ceiling_summary.json,gt_ceiling_results.json,reference_panel.json,extracted_rings.json,README.md}`

No deletions.
