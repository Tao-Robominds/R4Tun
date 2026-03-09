# 05 GT Warm Start

## Goal

Reverse-engineer **all** parameter values from `data/5-1.txt` ground truth columns (`segment`, `ring`), covering tunnel-physical, BO-critical, and safe-fixed parameters. Run the full pipeline with GT-derived values to establish a **GT-informed baseline** mIoU before BO begins. The output is a complete set of parameter JSONs plus a baseline evaluation report.

## Runtime Path

`data/irregular/5-1/gt_warm_start/`

## Inputs

- Full parameter inventory (step 04 output)
- Raw point cloud with GT labels: `data/5-1.txt` (columns: `segment`, `ring`)
- Existing preprocessing outputs: `data/irregular/5-1/` (enhanced.csv, depth_map, pixel_to_point.pkl)
- Current parameter JSONs in `agents/irregular/*/parameters/5-1/`

## Actions

### Action 1: Reverse-engineer Preprocessing parameters

Load `data/5-1.txt`, filter to GT-labelled points (segment not NaN).

| Parameter | Method | Source column(s) |
|-----------|--------|-----------------|
| `ring_spacing` | Compute mean h-distance between consecutive ring centres | `ring`, `h` |
| `tunnel_diameter` | `2 × median(r)` across all GT points | `r` |
| `num_slicing_planes` | `max(ring) - min(ring) + 1` | `ring` |
| `radius_min` | `min(r)` across GT-labelled surface points, with small margin | `r`, `segment` |
| `radius_max` | `max(r)` across GT-labelled surface points, with small margin | `r`, `segment` |
| `gradient_threshold` | Find threshold that retains ≥99% of GT surface points without including noise; sweep [0.1, 50] and pick the value that maximises retention F2 | `r`, `theta`, `pred` |
| `double_zero_cutoff` | Test both true/false, measure GT-surface retention | derived |
| `smoothing_offset` | Sweep [-0.01, 0.01], pick value maximising GT surface retention | derived |
| `target_distances` | Compute mean nearest-neighbour distance per ring in GT surface; set [2×, 1×, 0.5×] rounded to 3 dp | `theta`, `h`, `r` |
| `curvature_neighbors` | Compute curvature at GT block boundaries; pick k that maximises boundary curvature contrast | `r`, `theta`, `segment` |
| `interpolation_window` | Count max consecutive NaN columns in depth map per ring; set window = max_gap + 1 | depth_map.png |
| `samples_per_ring` | Count GT points per ring, take median | `ring` |
| `depth_map_resolution` | Keep 0.005 (computational, not GT-derivable) | — |
| Outlier params | Compute GT depth statistics per ring; derive thresholds from depth distribution percentiles | `r`, `ring` |

### Action 2: Reverse-engineer Detection parameters

Requires pixel-space GT: load `pixel_to_point.pkl`, map GT (segment, ring) to pixel (X, Y).

| Parameter | Method | Source |
|-----------|--------|--------|
| `ring_offset` | Mean X of K-block centroids minus ring_spacing_px/2 for ring 0 | GT K pixels |
| `ring_spacing_px` | Mean X-distance between consecutive ring K centroids | GT K pixels |
| `reverse_ring_order` | Check if ring index increases with X or decreases | GT ring→X mapping |
| `k_expected_height_px` | Median Y-extent of GT K blocks across rings | GT K pixels |
| `stagger_groups` | Cluster rings by their non-K block Y-pattern similarity (hierarchical, distance = max offset difference) | GT all-block centroids |
| `group_offsets` | Per group: median (block_Y - K_Y) for each block type, wrapped on image height | GT centroids per (ring, block) |
| `angle_pos_min/max`, `angle_neg_min/max` | Measure actual groove angles from GT block boundaries; compute min/max angle with 1° margin | GT boundary pixels |
| `binary_threshold` | Sweep [50, 250], pick value that maximises groove pixel recall against GT boundaries | depth_map + GT boundaries |
| `hough_threshold/min_length/max_gap` | Grid search small space, pick combo that maximises GT groove line recall | depth_map + GT boundaries |
| `eps` | Compute normalised K Y-spacing from GT K centroids; set eps = 0.8 × min_spacing | GT K pixels |
| `k_gap_tolerance_px` | Max Y-distance between GT K top and bottom edge across rings | GT K pixels |
| `groove_snap_px` | Max distance from GT block edge to nearest detected groove line | GT boundaries + lines |
| `k_candidates_per_ring` | Count distinct K-sized clusters at GT K X-positions; add 2 margin | GT K pixels |

### Action 3: Reverse-engineer Segmentation parameters

From pixel-space GT block extents per (segment, ring):

| Parameter | Method | Source |
|-----------|--------|--------|
| `K_half_width` | Median half-width of GT K blocks across all rings | GT K pixel bboxes |
| `K_half_height_pos/neg` | Median half-height (pos/neg side) of GT K blocks | GT K pixel bboxes |
| `K_centre_offset` | Median (GT_K_centre_Y - detected_K_Y) | GT vs detected K |
| `B1/B2_half_width` | Same method as K, for B1/B2 blocks | GT B1/B2 bboxes |
| `B1/B2_half_height_*` | Median half-heights per edge (trapezoid asymmetry) | GT B1/B2 bboxes |
| `B1/B2_centre_offset` | Median (GT_centre_Y - detected_Y) per block type | GT vs detected |
| `segment_half_width` | Median half-width of GT A-blocks | GT A bboxes |
| `A1–A4_half_height` | Per-type median half-height from GT | GT A bboxes |
| `A1–A4_centre_offset` | Per-type median (GT_centre_Y - detected_Y) | GT vs detected |
| `shrink_x`, `shrink_y` | Set to 0 initially (no shrink when templates are GT-derived) | — |

### Action 4: Validate safe-fixed parameters

For each `FIXED_*` constant in preprocessing, verify it doesn't harm GT retention:
- Run denoising with current FIXED values vs GT-derived alternatives
- If any FIXED value causes >0.5% GT point loss, flag it for promotion to BO-critical
- Record validation result per FIXED parameter

### Action 5: Write GT-derived parameter JSONs

Save reverse-engineered values to:
- `data/irregular/5-1/gt_warm_start/parameters_preprocessing.json`
- `data/irregular/5-1/gt_warm_start/parameters_detection.json`
- `data/irregular/5-1/gt_warm_start/parameters_segmentation.json`

Also save a comparison table: `data/irregular/5-1/gt_warm_start/parameter_comparison.md` showing (parameter, old value, GT-derived value, delta, method).

### Action 6: Run full pipeline with GT-derived parameters

1. Copy GT-derived JSONs to `agents/irregular/*/parameters/5-1/`
2. Run: preprocessing → detection → segmentation → evaluation
3. Save outputs to `data/irregular/5-1/gt_warm_start/`

### Action 7: Evaluate and report baseline

Run `evaluation.py` on the GT-derived pipeline output. Record:
- Overall mIoU, OA, F1
- Per-class IoU breakdown
- Comparison vs previous baseline (current parameter values)
- Gap analysis: which parameters improved most, which had no effect

Save to `data/irregular/5-1/gt_warm_start/baseline_report.md`.

## Outputs

- `data/irregular/5-1/gt_warm_start/parameters_preprocessing.json`
- `data/irregular/5-1/gt_warm_start/parameters_detection.json`
- `data/irregular/5-1/gt_warm_start/parameters_segmentation.json`
- `data/irregular/5-1/gt_warm_start/parameter_comparison.md`
- `data/irregular/5-1/gt_warm_start/baseline_report.md`
- `data/irregular/5-1/gt_warm_start/final.csv` (pipeline output)

## Verify Prompt

```
1. Are all 3 parameter JSONs present with GT-derived values?
2. Does parameter_comparison.md show old vs GT-derived for every parameter?
3. Was the full pipeline run with GT-derived params (final.csv exists)?
4. Does baseline_report.md show mIoU, OA, per-class IoU?
5. Were safe-fixed params validated against GT retention?
6. Is the baseline mIoU higher than the previous baseline?
```

## Verify Script

```bash
python methods/plans/scripts/verify_step.py --root data/irregular/5-1/gt_warm_start --step 05
```
