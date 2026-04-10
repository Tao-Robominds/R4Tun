## Tunnel Families

- **T1/T2 (1-x, 2-x):** 5.5 m inner diameter, 1.2 m rings, 6 segments/ring, staggered joints
- **T3 (3-x):** 5.5 m diameter, continuous joints, multi-station registration
- **T4/T5 (4-x, 5-x):** 7.5 m inner diameter, 1.8 m rings, 7 segments/ring, complex interleaved K-blocks

## Critical Parameters (SAM Segmenting Stage)

Six key driving parameters control all downstream geometry. All prompt_points and template_mask values are derived from these drivers and must scale proportionally.

### Key Driving Parameters

- **segment_per_ring** — 6 for T1/T2/T3; 7 for T4/T5. Adapted in 7/30 tunnels (only the 7-segment family).

- **segment_order** — `["K","B1","A1","A2","A3","B2"]` for 6-segment; `["K","B1","A1","A2","A3","A4","B2"]` for 7-segment. Must match segment_per_ring length.

- **segment_width (px)** — Horizontal crop width. Empirical range: **[1100, 2600]**, baseline 1200. Scale linearly with ring length in pixels: 1200 for 1.2 m rings, ~1800 for 1.8 m rings at 0.005 m/px.

- **K_height / AB_height (px)** — Vertical crop heights. K_height range: **[1080, 2290]**, AB_height range: **[3240, 6868]**, baseline 1080/3240. Scale with diameter / 2 in pixels. T4/T5 ≈ +40% vs T1/T2.

- **angle (deg)** — Scanner axis skew. Empirical range: **[7.5, 14.0]**, baseline 7.52. Most T1/T2 stay near 7.5; T4/T5 may need 8–14 depending on scanner tilt.

- **processing.padding (px)** — Horizontal padding around crops. Empirical range: **[160, 419]**, baseline 150, CV=0.265. Scale with segment_width. Adapted in **29/30** tunnels.

- **processing.y_bounds (px)** — Allowable Y-range in depth map. Baseline [4200, 13100]. Adjust proportionally to image height for T4/T5 (taller unwrapped maps).

### Derived Geometry (prompt_points, template_mask)

All values under `prompt_points` and `template_mask` scale proportionally with the key drivers:
- Horizontal values (outer_ring, middle_ring, inner_ring, center_ring, fine_spacing, edge_ring, edge_spacing, widths) scale with `segment_width / baseline_segment_width`
- Vertical values (K_height, AB_height, vertical_spacing, vertical_levels, special_levels, heights) scale with diameter ratio
- Maintain the proportional relationships from the baseline reference JSON

**Scale factor:** For T4/T5, multiply baseline pixel values by approximately `7.5 / 5.5 ≈ 1.364` for diameter-related values, and by `1.8 / 1.2 = 1.5` for ring-length-related values.

### Additional Tunnel-Responsive Parameter

- **processing.crop_margin (px)** — Margin around segment crops. Empirical range: **[50, 80]**, baseline 50. T1/T2 keep 50. T4/T5 → **69–80** (larger segments need wider margins to avoid edge clipping). T3 → 50–54. Adapted in **52/90** m_s_k files (100% T4, 90% T5).

### Locked Parameters

| Parameter | Baseline |
|---|---|
| use_original_label_distributions | true |
| processing.resolution | 0.005 |
| processing.mask_eps | 0.001 |

### Family Configuration Summary

| Family | segment_per_ring | segment_width | K_height | AB_height | ring_spacing |
|---|---|---|---|---|---|
| Regular (1-x, 2-x) | 6 | ~1200 | ~1080 | ~3240 | 1.2 |
| Continuous (3-x) | 6 | ~1200 | ~1080 | ~3240 | 1.2 |
| Complex (4-x, 5-x) | 7 | ~1800 | ~1470 | ~4420 | 1.8 |
