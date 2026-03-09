# Assumptions — sam4tun on 4-1 and 5-1

**Dataset:** `data/4-1.txt` (583k pts, 6 rings 120–125), `data/5-1.txt` (1.5M pts, 7 rings 107–113).
**Code:** `sam4tun/1_upfolding.py`, `2_denoising.py`, `3_enhancing.py`, `4-1_detection.py`, `4-2_sam.py`.

---

## A. Geometry assumptions

| ID | Assumption | Code evidence | Value in sam4tun |
|----|-----------|---------------|------------------|
| A1 | Tunnel diameter | `1_upfolding.py` hardcodes `diameter = 5.5` | 5.5 m |
| A2 | Ring spacing | `n = round(l/1.2)` in slicing | ~1.2 m |
| A3 | Cross-section is elliptical | RANSAC ellipse fit on slice points within 4.5 of max-y | — |
| A4 | Tunnel axis = MBR short edge | Minimum bounding rectangle of XOY projection | — |
| A5 | Centre line is degree-3 polynomial | RANSAC polynomial through ellipse centres | — |
| A6 | Ring count = number of slices | `ring_count = len(slicing_cloud)` drives all later stages | — |
| A7 | Physical ring count = slicing grid count | `ring_count` used both for curve fitting and for downstream grids | — |
| A8 | Slice half-thickness δ = 0.005 | Hardcoded `delta = 0.005` | 0.005 |

## B. Denoising assumptions

| ID | Assumption | Code evidence | Value in sam4tun |
|----|-----------|---------------|------------------|
| B1 | Surface band r ∈ [2.7, 2.8] | `mask_r = (r < 2.7) \| (r > 2.8)` → pred=0 | 2.7–2.8 |
| B2 | Gradient threshold = 0.2 | `grad_threshold = 0.2` | 0.2 |
| B3 | Smoothing offset = −0.003 | `cutoff_z_values_smoothed ... - 0.003` | −0.003 |
| B4 | Grid: y_step=0.5, z_step=0.001 | Hardcoded bin sizes | 0.5, 0.001 |
| B5 | x_step derived from ring_count | `x_step = (max_x - min_x) / ring_count` | — |

## C. Enhancing assumptions

| ID | Assumption | Code evidence | Value in sam4tun |
|----|-----------|---------------|------------------|
| C1 | Support = pred ≠ 0 | Filters `df_point_cloud[pred != 0]` | — |
| C2 | Curvature from k=20 neighbours | `compute_curvature(df, k=20)` | 20 |
| C3 | Target distance cascade | 0.08 → 0.04 → 0.02 | 0.08, 0.04, 0.02 |
| C4 | Curvature threshold = 0.0005 | `curvature_threshold=0.0005` | 0.0005 |
| C5 | n_segment defines high-density band | `n_segment=[10,21]` (half station) | [10, 21] |
| C6 | Depth map resolution = 0.005 | `resolution=0.005` throughout | 0.005 |
| C7 | pixel_to_point only for pred ≠ 8 | Interpolated points (pred=8) excluded from mapping | — |

## D. Detection assumptions

| ID | Assumption | Code evidence | Value in sam4tun |
|----|-----------|---------------|------------------|
| D1 | Vertical lines = ring boundaries | Hough or fallback `W/ring_count` | — |
| D2 | Oblique lines at ±6–9° | `6 <= angle <= 9` and `-9 <= angle <= -6` | ±6–9° |
| D3 | Horizontal lines at ±1° | `-1 <= angle <= 1` | ±1° |
| D4 | K height = 1079.92 mm | `K_height_pixel = 1079.92 / (1000*resolution)` | 1079.92 mm |
| D5 | A/B height = 3239.77 mm | `AB_height_pixel = 3239.77 / (1000*resolution)` | 3239.77 mm |
| D6 | Ring spacing = 1.2 m or W/ring_count | Used for vertical line extrapolation | 1.2 m |
| D7 | Fallback y offset = 431.87 | When no lines detected, assumed_y shifts by 431.87 | 431.87 |
| D8 | K positions are evenly spaced | Vertical lines placed at equal intervals | — |

## E. SAM / segmentation assumptions

| ID | Assumption | Code evidence | Value in sam4tun |
|----|-----------|---------------|------------------|
| E1 | segment_per_ring = 6 | `segment_per_ring=6` → K, B1, A1, A2, A3, B2 | 6 |
| E2 | Segment width = 1200 mm | `segment_width=1200` | 1200 mm |
| E3 | Joint angle = 7.52° | `angle=7.52` | 7.52° |
| E4 | Walk order: K→B1→A1→A2→A3→B2 then reverse | `process_row` serial walk | fixed |
| E5 | One global walk order for all rings | Same `block_labels` list applied to every ring | — |
| E6 | Fixed template vertices per block type | Hardcoded K/B1/B2/A vertex arrays in mm | — |
| E7 | One fixed template size per block family | Same K template for all rings; same A template for all rings | — |
| E8 | One group offset for all rings | K + fixed offset → A/B centres | — |
| E9 | SAM ViT-H can segment depth maps | `sam_model_registry["vit_h"]` | — |
| E10 | Template mask as mask_input | `mask_input=template_mask_logit` | — |
| E11 | Only pred=7 points updated | `valid_update_mask = (pred[...] == 7)` | — |

## F. Dataset-specific facts (4-1, 5-1)

| Fact | 4-1 | 5-1 |
|------|-----|-----|
| Physical rings | 6 (120–125) | 7 (107–113) |
| Points total | 583,744 | 1,504,524 |
| Segment IDs | 0–7 (K, B1, A1–A4, B2) | 0–7 |
| Segments per ring | 7 block types + BG | 7 block types + BG |
| Tunnel diameter | 7.5 m | 7.5 m |
| Ring spacing | 1.816 m | 1.816 m |
| Points per ring | 29k–231k | 34k–681k |
| K centroid x | 1.07, 1.25, −5.87, −1.09, −6.93, −2.01 | −15.41 … −4.72 |
| K pts per ring | 579–13,438 | 1,010–38,926 |
| B2 pts per ring | 5,059–33,446 | 1,026–93,906 |
| GT r range | 2.438–4.448 (q01=3.454, q99=4.185) | 3.542–3.937 (q01=3.572, q99=3.904) |
| Old band [2.7,2.8] retention | 163/441,447 (0.04%) | 0/1,130,609 (0%) |
| Tuned band [3.526,4.051] retention | 424,652/441,447 (96.2%) | 1,130,609/1,130,609 (100%) |
