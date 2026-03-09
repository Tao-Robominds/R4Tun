# 07 Intrinsic Analysis

## Goal
Produce `intrinsics.md` — define computable-without-GT metrics that predict pipeline quality.

## Runtime Path
`data/{tunnel_id}/workflow/{run_id}/07_intrinsic_analysis/intrinsics.md`

## Inputs
- BO logs
- pipeline outputs (depth maps, detected segments, final.csv)

## Known Intrinsic Metric Candidates

### Preprocessing — Depth Map Quality

From Step 05 GT warm start: GT-derived `radius_min/max` ([3.7, 3.848]) destroyed the depth map by excluding background surface points, producing large white (NaN) regions. The proven wider band ([3.526, 4.051]) produces a dense, well-filled depth map. This directly indicates depth map quality as a preprocessing intrinsic.

**Candidates:**
- `pre_depth_map_fill_ratio`: fraction of non-NaN pixels in depth map. Low fill = aggressive denoising excluded valid surface. **Guardrail:** fill_ratio < 0.5 → fail.
- `pre_depth_map_max_nan_block`: largest contiguous NaN region (pixels). Large blocks = holes that break line detection. **Guardrail:** max_block > 500 px → warning.
- `pre_depth_map_row_fill_ratio`: min fill ratio across rows. Rows with very low fill = ring-level coverage gaps.
- `pre_depth_map_col_fill_ratio`: min fill ratio across columns. Low columns = angular coverage gaps.
- `pre_point_retention_pct`: fraction of raw points surviving denoising. Too low = over-filtering; too high = noise retained.
- `pre_theta_coverage_pct`: fraction of theta bins with at least 1 valid point. Gaps = missing angular sectors.
- `pre_theta_gap_max_deg`: largest angular gap (degrees). Large gaps = structural holes.

### Detection — Line and K Quality

- `det_groove_alignment_pct`: percentage of expanded block positions that align with detected grooves. High = good detection. **Already computed** (89.2% for GT warm start vs 23.7% for broken run).
- `det_k_detection_type_ratio`: fraction of K positions found by dbscan/groove_pair vs fallback. Fallback = poor detection.
- `det_k_count_match`: does detected K count match ring_count? Mismatch = detection failure.
- `det_x_spacing_cv`: coefficient of variation of K X-positions spacing. High CV = uneven ring detection.

### Segmentation — Coverage Quality

- `seg_mask_fill_rate`: fraction of depth map pixels assigned to a block label.
- `seg_template_coverage`: fraction of all_segments entries that overlap with non-NaN depth map pixels.
- `seg_segment_count_match`: do we get expected segments per ring?

## Actions
1. Extract intrinsic metrics from pipeline outputs (depth_map, detected.csv, all_segments.csv, final.csv).
2. Build metric bank with value ranges from BO trials.
3. Set guardrail thresholds (hard fail, warning, pass).
4. Identify failure signatures (metric patterns that predict low mIoU).
5. Write back reusable knowledge to `agents/irregular/*/knowledge/`.

## Outputs
- `intrinsics.md`
- `metric_bank.json`

## Verify Prompt
```
1. Does the intrinsic artifact define the metric bank with preprocessing depth map quality metrics?
2. Are depth_map_fill_ratio, max_nan_block, and groove_alignment_pct included?
3. Are guardrail thresholds set with clear pass/warn/fail levels?
4. Are failure signatures documented (e.g. fill_ratio < 0.5 → detection will fail)?
5. Is knowledge written back to stage-specific knowledge dirs?
```

## Support Templates
- `plans/templates/intrinsics.md.template`
- `plans/templates/metric_bank.json.template`

## Verify Script
```bash
python methods/plans/scripts/verify_step.py --root data/{tunnel_id}/workflow/{run_id} --step 07
```
