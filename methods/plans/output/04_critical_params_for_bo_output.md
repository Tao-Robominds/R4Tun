# Step 04 Output — Critical Parameters for BO

## 1. Full Parameter Inventory

### 1.1 Preprocessing (1_preprocessing.py) — 25 parameters

| # | Parameter | Category | Type | Default | 5-1 value | 4-1 value | BO? | Sensitivity |
|---|-----------|----------|------|---------|-----------|-----------|-----|-------------|
| 1 | `ring_spacing` | Tunnel-physical | float | 1.2 | 1.816 | 1.816 | No | — |
| 2 | `tunnel_diameter` | Tunnel-physical | float | 5.5 | 7.5 | 7.5 | No | — |
| 3 | `depth_map_resolution` | Tunnel-physical | float | 0.005 | 0.005 | 0.005 | No | — |
| 4 | `radius_min` | BO-critical | float | 2.7 | 3.526 | 3.526 | **Yes** | HIGH |
| 5 | `radius_max` | BO-critical | float | 2.8 | 4.051 | 4.051 | **Yes** | HIGH |
| 6 | `gradient_threshold` | BO-critical | float | 0.2 | 10.0 | 10.0 | **Yes** | HIGH |
| 7 | `double_zero_cutoff` | BO-critical | bool | true | false | false | **Yes** | MEDIUM |
| 8 | `smoothing_offset` | BO-critical | float | -0.003 | 0.0 | 0.0 | **Yes** | MEDIUM |
| 9 | `target_distances` | BO-critical | list[3] | [0.08,0.04,0.02] | [0.081,0.040,0.02] | [0.081,0.040,0.02] | **Yes** | HIGH |
| 10 | `curvature_neighbors` | BO-critical | int | 20 | 9 | 6 | **Yes** | HIGH |
| 11 | `interpolation_window` | BO-critical | int | 9 | 5 | 5 | **Yes** | MEDIUM |
| 12 | `num_slicing_planes` | BO-critical | int/null | null | 9 | 7 | **Yes** | MEDIUM |
| 13 | `samples_per_ring` | BO-critical | int | 1210 | 1210 | 1210 | **Yes** | LOW |
| 14 | `outlier_depth_threshold_low` | BO-critical | float | 0.003 | 0.003 | 0.003 | **Yes** | MEDIUM |
| 15 | `outlier_depth_threshold_high` | BO-critical | float | 0.008 | 0.008 | 0.008 | **Yes** | MEDIUM |
| 16 | `outlier_high_density_ring_start` | BO-critical | int | 0 | 0 | 0 | **Yes** | MEDIUM |
| 17 | `outlier_high_density_ring_end` | BO-critical | int | 5 | 5 | 5 | **Yes** | MEDIUM |
| 18 | `outlier_neighbors` | BO-critical | int | 20 | 20 | 20 | **Yes** | MEDIUM |
| 19 | `max_outlier_points` | BO-critical | int | 5000 | 5000 | 5000 | **Yes** | MEDIUM |
| 20 | `outlier_interpolation_radius` | BO-critical | float | 0.06 | 0.06 | 0.06 | **Yes** | MEDIUM |
| 21 | `outlier_num_interpolations` | BO-critical | int | 2 | 2 | 2 | **Yes** | LOW |
| 22 | `outlier_duplicate_threshold` | BO-critical | float | 0.02 | 0.02 | 0.02 | **Yes** | LOW |
| 23 | `outlier_bidirectional` | BO-critical | bool | false | false | false | **Yes** | LOW |
| 24 | `outlier_depth_map_window` | BO-critical | int | 1 | 1 | 1 | **Yes** | LOW |
| 25 | `SURFACE_PRED` | Fixed constant | int | 7 | 7 | 7 | No | — |

**Safe-fixed (not in JSON, compile-time only):** `FIXED_SLICE_HALF_THICKNESS` (0.005), `FIXED_MAX_DISTANCE_FROM_TOP` (4.5), `FIXED_POLYNOMIAL_DEGREE` (3), `FIXED_RANSAC_INLIER_RATIO` (0.75), `FIXED_RANSAC_CONFIDENCE` (0.9), `FIXED_RANSAC_MIN_SAMPLES` (5), `FIXED_RANSAC_INLIER_THRESHOLD` (0.8), `FIXED_BATCH_SIZE` (1M), `FIXED_NUM_JOBS` (12), `FIXED_THETA_STEP` (0.5), `FIXED_RADIAL_STEP` (0.001), `FIXED_GRADIENT_EPSILON` (1e-6), `FIXED_SMOOTHING_WINDOW` (3), `FIXED_CURVATURE_THRESHOLD` (0.0005), `FIXED_UPSAMPLING_NEIGHBORS` (20), `FIXED_DISTANCE_TOLERANCE_LOW` (0.9), `FIXED_DISTANCE_TOLERANCE_HIGH` (2.0), `FIXED_RADIUS_FILTER_FACTOR` (0.15), `FIXED_MIN_NEW_POINT_DISTANCE_FACTOR` (0.2).

---

### 1.2 Detection (2_detection.py) — 27+ parameters

| # | Parameter | Category | Type | Default | 5-1 value | 4-1 value | BO? | Sensitivity |
|---|-----------|----------|------|---------|-----------|-----------|-----|-------------|
| 1 | `binary_threshold` | BO-critical | int | 127 | 178 | 139 | **Yes** | HIGH |
| 2 | `hough_threshold` | BO-critical | int | 50 | 50 | 37 | **Yes** | HIGH |
| 3 | `hough_min_length` | BO-critical | int | 100 | 85 | 31 | **Yes** | HIGH |
| 4 | `hough_max_gap` | BO-critical | int | 40 | 85 | 133 | **Yes** | HIGH |
| 5 | `angle_pos_min` | BO-critical | float | 6.0 | 3.33 | 4.84 | **Yes** | HIGH |
| 6 | `angle_pos_max` | BO-critical | float | 9.0 | 12.06 | 13.55 | **Yes** | HIGH |
| 7 | `angle_neg_min` | BO-critical | float | -9.0 | -12.52 | -14.67 | **Yes** | HIGH |
| 8 | `angle_neg_max` | BO-critical | float | -6.0 | -5.86 | -5.82 | **Yes** | HIGH |
| 9 | `eps` | BO-critical | float | 0.07 | 0.057 | 0.07 | **Yes** | HIGH |
| 10 | `k_expected_height_px` | BO-critical | float | 300 | 300 | 300 | **Yes** | HIGH |
| 11 | `k_gap_tolerance_px` | BO-critical | float | 150.0 | 150 | 150 | **Yes** | MEDIUM |
| 12 | `k_candidates_per_ring` | BO-critical | int | 8 | 8 | 8 | **Yes** | MEDIUM |
| 13 | `groove_snap_px` | BO-critical | float | 60.0 | 60 | 60 | **Yes** | MEDIUM |
| 14 | `ring_offset` | BO-critical | float | auto | 157.2 | 193.3 | **Yes** | HIGH |
| 15 | `ring_spacing_px` | BO-critical | float | auto | 363.5 | -360.0 | **Yes** | HIGH |
| 16 | `reverse_ring_order` | BO-critical | bool | false | true | true | **Yes** | HIGH |
| 17 | `stagger_groups` | BO-critical | dict | {} | {A:[4,5,6],B:[0..3]} | {A:[0..4],B:[5]} | **Yes** | HIGHEST |
| 18 | `group_offsets` | BO-critical | dict(12D) | {} | 12 values | 12 values | **Yes** | HIGHEST |

**Safe-fixed (compile-time only):** `DEFAULT_DILATION_KERNEL_SIZE` (3), `DEFAULT_DILATION_ITERATIONS` (1), `DEFAULT_CANNY_LOW` (50), `DEFAULT_CANNY_HIGH` (150), `DEFAULT_HOUGH_HORIZONTAL_THRESHOLD` (50), `DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH` (100), `DEFAULT_HOUGH_HORIZONTAL_MAX_GAP` (10), `DEFAULT_HORIZONTAL_ANGLE_TOLERANCE` (1.0), `DEFAULT_HOUGH_VERTICAL_THRESHOLD` (500), `DEFAULT_MERGE_DISTANCE_THRESHOLD` (3.0), `FIXED_MERGE_CLOSE_THRESHOLD` (6.0).

*Note:* `tunnel_diameter` and `depth_map_resolution` are inherited from preprocessing JSON (tunnel-physical, not BO-tuned in detection).

---

### 1.3 Segmentation (segmentation.py) — 22 parameters

| # | Parameter | Category | Type | Default | 5-1 value | 4-1 value | BO? | Sensitivity |
|---|-----------|----------|------|---------|-----------|-----------|-----|-------------|
| 1 | `K_half_width` | BO-critical | int | 125 | 181 | 189.27 | **Yes** | HIGH |
| 2 | `K_half_height_pos` | BO-critical | int | 124 | 147 | 143 | **Yes** | HIGH |
| 3 | `K_half_height_neg` | BO-critical | int | 92 | 147 | 143 | **Yes** | HIGH |
| 4 | `K_centre_offset` | BO-critical | float | 0 | 0 | -39.59 | **Yes** | HIGH |
| 5 | `B1_half_width` | BO-critical | int | 125 | 181 | 189.27 | **Yes** | HIGH |
| 6 | `B1_half_height_top` | BO-critical | int | 324 | 441 | 383 | **Yes** | HIGH |
| 7 | `B1_half_height_bottom_pos` | BO-critical | int | 308 | 441 | 383 | **Yes** | MEDIUM |
| 8 | `B1_half_height_bottom_neg` | BO-critical | int | 340 | 441 | 383 | **Yes** | MEDIUM |
| 9 | `B1_centre_offset` | BO-critical | float | 0 | 0 | 5.78 | **Yes** | HIGH |
| 10 | `B2_half_width` | BO-critical | int | 125 | 181 | 189.27 | **Yes** | HIGH |
| 11 | `B2_half_height_top_pos` | BO-critical | int | 308 | 441 | 372 | **Yes** | MEDIUM |
| 12 | `B2_half_height_top_neg` | BO-critical | int | 340 | 441 | 372 | **Yes** | MEDIUM |
| 13 | `B2_half_height_bottom` | BO-critical | int | 324 | 441 | 372 | **Yes** | HIGH |
| 14 | `B2_centre_offset` | BO-critical | float | 0 | 0 | -32.72 | **Yes** | HIGH |
| 15 | `segment_half_width` | BO-critical | int | 125 | 181 | 189.27 | **Yes** | HIGH |
| 16 | `A1_half_height` | BO-critical | int | 324 | 441 | 372 | **Yes** | HIGH |
| 17 | `A2_half_height` | BO-critical | int | 324 | 441 | 376 | **Yes** | HIGH |
| 18 | `A3_half_height` | BO-critical | int | 324 | 441 | 361 | **Yes** | HIGH |
| 19 | `A4_half_height` | BO-critical | int | 324 | 441 | 371 | **Yes** | HIGH |
| 20 | `A1_centre_offset` | BO-critical | float | 0 | 0 | -33.28 | **Yes** | HIGH |
| 21 | `A2_centre_offset` | BO-critical | float | 0 | 0 | 19.33 | **Yes** | HIGH |
| 22 | `A3_centre_offset` | BO-critical | float | 0 | 0 | 13.41 | **Yes** | HIGH |
| 23 | `A4_centre_offset` | BO-critical | float | 0 | 0 | -48.2 | **Yes** | HIGH |
| 24 | `shrink_x` | BO-critical | float | 0 | 0 | 4.57 | **Yes** | MEDIUM |
| 25 | `shrink_y` | BO-critical | float | 0 | 0 | 1.41 | **Yes** | MEDIUM |

**Safe-fixed:** None. All segmentation parameters are BO-critical (the stage is purely geometric with no algorithmic constants to fix).

---

## 2. Critical Parameters for BO

### Selection rule

A parameter is **critical** if:
1. It addresses a challenge from `02_challenge_map_output.md` (irregular geometry, varying block sizes, non-uniform K spacing), AND
2. It lies on the data-flow critical path (preprocessing → depth map → detection → segments → segmentation → mIoU), AND
3. Prior BO experiments or GT reverse-engineering show >1% mIoU sensitivity.

### 2.1 BO search space summary

| Stage | Dimension | Critical params |
|-------|-----------|-----------------|
| Preprocessing | 8 | radius_min, radius_max, gradient_threshold, double_zero_cutoff, smoothing_offset, target_distances[0..2], curvature_neighbors |
| Detection | 20+ | binary_threshold, hough_threshold, hough_min_length, hough_max_gap, angle_pos_min/max, angle_neg_min/max, eps, k_expected_height_px, k_gap_tolerance_px, groove_snap_px, ring_offset, ring_spacing_px, group_offsets (12D) |
| Segmentation | 22 | All 22 template shape params |
| **Total** | **~50** | |

### 2.2 Critical params by priority

**Tier 1 — HIGHEST impact (address main bottleneck: wrong block positions)**

| Parameter | Stage | Reason |
|-----------|-------|--------|
| `group_offsets` (12D) | Detection | A2/A3 offset error (~800 px) is the single largest mIoU bottleneck (0.501→0.720 gap). Per-group offsets must match actual block walking order. |
| `stagger_groups` | Detection | Ring-to-group assignment directly controls which offsets apply. Wrong assignment cascades to all non-K blocks. |
| `ring_offset` | Detection | First K position anchor; all segment positions derive from this. |
| `ring_spacing_px` | Detection | Ring column spacing; errors multiply across all rings. |

**Tier 2 — HIGH impact (template coverage and line detection quality)**

| Parameter | Stage | Reason |
|-----------|-------|--------|
| `radius_min`, `radius_max` | Preprocessing | Determine which points survive denoising. On 4-1/5-1, default [2.7,2.8] would exclude 100% of valid surface points (actual range ~[3.5,4.1]). |
| `gradient_threshold` | Preprocessing | Separates surface from noise. Optimal value varies 0.2–10.0 across tunnels. |
| `binary_threshold` | Detection | Controls edge map quality; value ranges 127–178 across tunnels. |
| `angle_pos/neg_min/max` (4D) | Detection | Groove angle band; irregular tunnels have wider ranges than regular. |
| `hough_threshold/min_length/max_gap` (3D) | Detection | Line detection sensitivity; values differ 2–3× between 4-1 and 5-1. |
| `eps` | Detection | DBSCAN clustering radius; directly affects K detection accuracy (~28 px error at optimal). |
| All `*_half_width/height` (18D) | Segmentation | Template sizes control point coverage. K template was 2.7× too small in initial analysis. |
| All `*_centre_offset` (7D) | Segmentation | Fine-tune template position relative to detected centre. |

**Tier 3 — MEDIUM impact (second-order effects)**

| Parameter | Stage | Reason |
|-----------|-------|--------|
| `target_distances` (3D) | Preprocessing | Upsampling density; affects depth map detail for detection. |
| `curvature_neighbors` | Preprocessing | Surface smoothness; affects edge quality. Values 6–20 across tunnels. |
| `double_zero_cutoff` | Preprocessing | Consecutive-empty-bin cutoff; false on 4-1/5-1, true on regular tunnels. |
| `num_slicing_planes` | Preprocessing | Controls depth map grid resolution; needs to match ring density. |
| `k_gap_tolerance_px` | Detection | Gap filtering; too tight = miss K, too loose = false positives. |
| `groove_snap_px` | Detection | Groove alignment reward radius. |
| `shrink_x`, `shrink_y` | Segmentation | Global template shrink; prevents overlap bleeding. |
| Outlier params (11D) | Preprocessing | Outlier enhancement; moderate impact on depth map quality. |

### 2.3 Excluded from BO (with reasons)

| Parameter | Stage | Reason |
|-----------|-------|--------|
| `ring_spacing` | Preprocessing | Physical measurement, not a tuning knob. |
| `tunnel_diameter` | Preprocessing | Physical measurement. |
| `depth_map_resolution` | Preprocessing | Computational choice, fixed at 0.005 m/px for all tunnels. |
| `SURFACE_PRED` | All | Label convention (always 7). |
| `reverse_ring_order` | Detection | Binary choice determined by visual inspection once. |
| All `FIXED_*` constants | Preprocessing | BO experiments showed <0.1% improvement when tuned. |
| All `DEFAULT_*` safe-fixed | Detection | Horizontal/vertical Hough, dilation, canny — auxiliary line detection with negligible impact. |

---

## 3. Data-Flow Critical Path

```
Point cloud (.txt)
  │
  ├─[Preprocessing]─ ring_spacing, tunnel_diameter ─→ Unfolding ─→ unwrapped.csv
  │                   radius_min, radius_max,
  │                   gradient_threshold ──────────→ Denoising ─→ denoised.csv
  │                   target_distances,
  │                   curvature_neighbors ─────────→ Enhancing ─→ enhanced.csv
  │                   depth_map_resolution ────────→            ─→ depth_map.png
  │                   outlier params ──────────────→            ─→ depth_map_outlier.npy
  │
  ├─[Detection]───── binary_threshold, angle_*,
  │                   hough_* ─────────────────────→ Lines ─→ line_data
  │                   eps, k_expected_height_px,
  │                   ring_offset, ring_spacing_px ─→ K positions ─→ detected.csv
  │                   stagger_groups, group_offsets ─→ Expansion ─→ all_segments.csv
  │
  └─[Segmentation]── all 22 template params ──────→ Label map ─→ final.csv ─→ mIoU
```

**Critical path:** `radius_min/max → depth_map → binary_threshold + angles → K positions → group_offsets → template shapes → mIoU`.

Bottleneck: `group_offsets` (12D) has the highest single-parameter impact. Every 100 px of offset error costs ~0.03 mIoU.

---

## 4. Parameter JSON Schema

All parameters stored in `agents/irregular/{stage}/parameters/{tunnel_id}/`.

| Stage | JSON file | Key count |
|-------|-----------|-----------|
| Preprocessing | `parameters_preprocessing.json` | 25 |
| Detection | `parameters_detection.json` | 18 |
| Segmentation | `parameters_geometric_template.json` | 22 |

Sample templates in `parameters/sample/` for new tunnels.

---

## 5. Code Organisation

Each `.py` file now has three clearly labelled sections:

```
# A. TUNNEL-PHYSICAL — measured once per tunnel, not BO-tuned
# B. BO-CRITICAL — tunable per tunnel via JSON, candidates for BO
# C. SAFE-FIXED — proven defaults, negligible BO improvement
```

All BO-critical parameters have `DEFAULT_*` constants that serve as fallbacks when no JSON is provided. All safe-fixed parameters have `FIXED_*` constants that are never read from JSON.
