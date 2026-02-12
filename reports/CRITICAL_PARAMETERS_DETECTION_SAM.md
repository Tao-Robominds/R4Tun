# Critical Parameters: Detection and SAM Stages

**Date:** February 2026  
**Source:** P4TUN_OPTIMIZATION_JOURNEY_2-2, P4TUN_OPTIMIZATION_JOURNEY_3-1_1-4, R4TUN_EXPLORATION_JOURNEY, P4TUN_PER_RING_ALIGNMENT_LESSONS  
**Purpose:** Quick reference for parameter tuning priority and sensitivity.

---

## Detection Stage – Critical Parameters

### HIGH Sensitivity (tune first)

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `binary_threshold` | 100–180 | Edge detection sensitivity; large mIoU impact (+2–3%) |
| `hough_oblique_threshold` / `threshold` | 40–100 | Line detection confidence; +1–2% mIoU |
| `angle_positive_min` | 4–8° | Min positive slope; must match tunnel K-line angles |
| `angle_positive_max` | 7–12° | Max positive slope |
| `angle_negative_min` / `angle_negative_max` | −12° to −4° | Negative slope range |

### MEDIUM Sensitivity

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `hough_vertical.threshold` | 400–800 | Too high = miss rings, too low = false positives |
| `hough_oblique.min_length` | 50–150 px | Filters short noise lines |
| `hough_oblique.max_gap` | 20–80 px | Connects broken lines |

### Reports Consensus

- **P4TUN_OPTIMIZATION_JOURNEY_2-2:** `binary_threshold`, `angle_positive_min/max`, `hough_oblique.threshold` are the most important.
- **P4TUN_PER_RING_ALIGNMENT_LESSONS:** Angle params are CRITICAL; wrong angles → no K-block detection.
- **R4TUN_EXPLORATION_JOURNEY:** `binary_threshold` 100–180, `angle_positive_min/max` 4–12°.

---

## SAM Stage – Critical Parameters

### CRITICAL Sensitivity (tune with care)

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `ab_height` | 3000–3500 mm | A/B block height; large impact |
| `ab_mask_height` / `a_blocks.height` | 1500–1750 mm | A/B template height |
| `k_block.height_neg` | 400–600 mm | K-block template height; K-block recall |
| `k_mask_height` | 580–680 mm | Most sensitive K-block param; ≥650 for 3-1, ~580 for 1-4 |
| `angle_deg` | 6.5–8.5° | Segment tilt; K-block IoU drops if reduced |
| `k_block.height_pos` / `height_neg` | 580–680 mm | K-block anchoring; protect in BO |
| `segment_geometry` | — | **Warning:** Controls where segments are placed; changes can cause catastrophic regressions (e.g. 0.765 → 0.673) |

### MEDIUM Sensitivity

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `k_block.width` | 550–750 mm | K-block template width |
| `k_height` | 1000–1200 mm | K-block height in segment geometry |
| `segment_width` | 1100–1250 mm | Segment width in mm |
| `ab_blocks.width` / `a_blocks.width` | 550–700 mm | A/B block template width |

### Reports Consensus

- **P4TUN_OPTIMIZATION_JOURNEY_3-1_1-4:** K-block params (`k_mask_height`, `angle_deg`) must be **protected** in BO; overall mIoU can improve while K-block IoU drops.
- **P4TUN_OPTIMIZATION_JOURNEY_2-2:** Only `template_mask` is relatively safe to tune; `segment_geometry` is dangerous.
- **R4TUN_EXPLORATION_JOURNEY:** `ab_height`, `k_block.height_neg`, `template_mask.k_block.width` dominate importance; AB params ~5× more impact than K for some tunnels.

---

## Quick Reference by Priority

### Detection (tune in this order)

1. `binary_threshold` — edge sensitivity
2. `angle_positive_min` / `angle_positive_max` — K-line angle range
3. `angle_negative_min` / `angle_negative_max`
4. `hough_oblique.threshold`
5. `hough_vertical.threshold`

### SAM (tune with care)

**Template masks (safer):** `k_block.height_neg`, `k_block.width`, `ab_blocks.height`

**Segment geometry (risky):** `ab_height`, `k_height`, `angle_deg` — only adjust when necessary and verify mIoU and K-block IoU.
