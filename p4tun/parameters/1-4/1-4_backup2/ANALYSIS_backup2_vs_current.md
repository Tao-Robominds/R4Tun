# 1-4_backup2 vs current parameters

## Preprocessing

**Result: IDENTICAL.**  
Backup2 unfolding/denoising/enhancing match current `parameters_preprocessing.json` (same values as backup and current).

---

## Detection (`parameters_detection.json`)

**Result: SAME as 1-4_backup.**  
Backup2 detection equals backup (binary_threshold 140, hough_vertical.threshold 617, etc.).  
Current (before applying backup2) was already backup detection from the previous run.

---

## SAM (`parameters_sam.json`)

**Result: DIFFERENT from current (BO).**

| Parameter | Backup2 | Current (BO) |
|-----------|---------|----------------|
| padding | **300** | 100 |
| crop_margin | 50 | 40 |
| ab_blocks.vertical_levels | (absent) | present (level_1..center) |
| description | "Optimized" | "6-segment" |

Other SAM values (segment_width 1200, k_height/ab_height, angle_deg 7.52, template_mask, prompt_points) match or are very close. Main applied difference: **padding 300** (backup2) vs 100 (BO).

---

## Run with backup2 applied

- **Preprocessing:** unchanged (current 1-4).
- **Detection:** backup2 (same as backup).
- **SAM:** backup2 (padding 300, no vertical_levels in file; defaults merged at runtime).

**Results (data/1-4):**

| Metric | Value |
|--------|--------|
| OA | 0.754 |
| F1 | 0.676 |
| **mIoU** | **0.532** |
| K positions | 9 |

**Comparison:**

- 1-4_backup (previous run): mIoU **0.544**, 9 K positions.
- 1-4_backup2 (this run): mIoU **0.532**, 9 K positions.

Backup2 is slightly worse than backup (−0.012 mIoU). BO best (when data/bo was intact) was ~0.748; neither backup nor backup2 restores that on data/1-4.
