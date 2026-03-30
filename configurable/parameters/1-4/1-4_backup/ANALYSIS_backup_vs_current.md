# 1-4_backup vs current parameters

## Preprocessing (unfolding, denoising, enhancing)

**Result: IDENTICAL.**  
Backup unfolding/denoising/enhancing match current `parameters_preprocessing.json` and split files (same values).

---

## Detection (`parameters_detection.json`)

**Result: DIFFERENT.**

| Parameter | Backup | Current |
|-----------|--------|---------|
| binary_threshold | 140 | 101 |
| dilation_iterations | 3 | 1 |
| hough_oblique.threshold | 60 | 41 |
| hough_oblique.min_length | 111 | 79 |
| hough_oblique.max_gap | 33 | 65 |
| angle_positive_min | 5.918 | 4.808 |
| angle_positive_max | 8.667 | 8.976 |
| hough_horizontal.threshold | 44 | 37 |
| hough_horizontal.min_length | 113 | 75 |
| hough_horizontal.max_gap | 6 | 16 |
| hough_vertical.threshold | 617 | 291 |
| merge_distance_threshold | 5 | 2 |
| merge_close_threshold | (absent) | 6 |

---

## SAM (`parameters_sam.json`)

**Result: DIFFERENT.**

| Parameter | Backup | Current |
|-----------|--------|---------|
| segment_width | 1200 | 1150 |
| k_height | 1079.92 | 1150 |
| ab_height | 3239.77 | 3104.77 |
| angle_deg | 7.52 | 6.5 |
| padding | 150 | 100 |
| crop_margin | 50 | 40 |
| template_mask.k_block.height_neg | 460.77 | 580 |
| template_mask.k_block.width | 625 | 639 |
| ab_blocks / template widths, heights | defaults | BO-tuned |
| min_quality_threshold | 0.3 | 0.496 |

Backup looks like pre-BO defaults; current is BO best_extracted.

---

## Conclusion

Preprocessing: no change. Detection and SAM: backup differs from current. Applied backup detection + SAM and re-ran pipeline.

---

## Run with backup params (after apply)

- **Detection:** 9 K positions (ring_count 10); types: 3 midpoint, 3 assume, 2 negative_slope, 1 positive_slope.
- **Evaluation:** mIoU **0.544**, OA 0.749, F1 0.696.

Current (BO) params had given mIoU ~0.557 with 10 K positions. Backup params give slightly lower mIoU (0.544) and 9 positions. To restore BO params run: `apply_bo_best_to_parameters 1-4 --apply`.
