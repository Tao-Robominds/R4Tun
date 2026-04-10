# BO search space — SAM segmenting stage

## Tunable parameters — Structure

- **segment_per_ring (int)** — Number of segment labels per full ring. Range: **[6, 7]** (baseline 6).

- **segment_order (list of str)** — Ordered segment names; **length must equal `segment_per_ring`**. Valid templates:
  - Six segments: `["K","B1","A1","A2","A3","B2"]`
  - Seven segments: `["K","B1","A1","A2","A3","A4","B2"]`

## Tunable parameters — Geometry (pixels)

- **segment_width (px)** — Horizontal crop width per segment column. Range: **[1100, 2600]** (baseline 1200).

- **K_height (px)** — Vertical extent for K-block crops. Range: **[1080, 2290]** (baseline 1080).

- **AB_height (px)** — Vertical extent for A/B block crops. Range: **[3240, 6868]** (baseline 3240).

- **angle (deg)** — Scanner / unwrap skew. Range: **[7.5, 14.0]** (baseline 7.52).

- **processing.padding (px)** — Horizontal padding around crops. Range: **[160, 419]** (baseline 150).

- **processing.y_bounds (px)** — `[y_min, y_max]` allowable vertical span in depth-map coordinates. Search each bound in **[3500, 5500]** and **[11000, 15000]** respectively (baseline `[4200, 13100]`); keep `y_min < y_max`.

- **processing.crop_margin (px)** — Margin around segment crops. Range: **[50, 80]** (baseline 50).

## Derived geometry (`prompt_points`, `template_mask`)

All nested numeric fields should remain **proportional** to the drivers above relative to the baseline reference JSON: scale horizontal offsets with `segment_width`, vertical extents with `K_height` / `AB_height`, and preserve relative ratios between sibling keys.

## Locked parameters (baseline, not searched)

| Parameter | Baseline |
|---|---|
| use_original_label_distributions | true |
| processing.resolution | 0.005 |
| processing.mask_eps | 0.001 |

## Constraints

- `len(segment_order) == segment_per_ring`.
- After changing drivers, re-scale `prompt_points` and `template_mask` so internal consistency is preserved.
