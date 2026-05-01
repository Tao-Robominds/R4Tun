# Preprocessing BO selection outcomes

- run_id: `preprocessing_bo_selection_v1`
- selected_for_bo: **14**
- holdout: **3**
- failure_reasons_covered: **dominant_empty_component, many_empty_row_bands, near_empty_valid_ratio**

## Selected rings (BO panel)
- failure_mode_representative: 1-3/r128, 2-3/r221, 1-5/r272
- control: 1-3/r127, 2-3/r222, 1-5/r273
- characteristic_representative: 5-7/r315, 4-6/r283, 4-4/r215, 1-1/r25, 5-1/r114, 5-6/r285, 5-1/r116, 5-1/r113

## Holdout rings
- 2-2/r135, 5-7/r319, 4-10/r394

## GT-derived BO objective
- maximize `foreground_mask_iou = TP / (TP + FP + FN)`
- `TP`: valid depth pixels overlapping GT foreground
- `FP`: valid depth pixels outside GT foreground
- `FN`: missing valid depth pixels inside GT foreground
- `foreground_support_ratio`, `largest_fg_hole_ratio`, `overfill_ratio`, `valid_ratio` are diagnostics only

## GT artifact availability
- reward source for preprocessing BO is point-level foreground support from ring data (`segment > 0`) aligned to depth-map pixels
- no dependency on detection-stage `gt_ceiling/labelmap.npy`
