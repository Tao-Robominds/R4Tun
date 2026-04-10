# BO search space — Detecting stage

## Tunable parameters — Hough

- **hough_threshold_oblique** — Hough accumulator vote threshold for oblique lines. Range: **[20, 83]** (baseline 50).

- **hough_threshold_horizontal** — Same for horizontal-oriented lines. Range: **[20, 83]** (baseline 50).

- **hough_threshold_vertical** — Same for vertical lines. Range: **[320, 980]** (baseline 500).

- **maxLineGap_oblique (px)** — Max gap when linking oblique segments. Range: **[30, 100]** (baseline 40).

- **maxLineGap_horizontal (px)** — Max gap for horizontal segments. Range: **[12, 70]** (baseline 10).

- **minLineLength_oblique (px)** — Minimum oblique segment length. Range: **[60, 240]** (baseline 100).

- **minLineLength_horizontal (px)** — Minimum horizontal segment length. Range: **[60, 220]** (baseline 100).

## Tunable parameters — Image and morphology

- **binary_threshold** — Depth-map binarization threshold. Range: **[115, 127]** (baseline 127).

- **merge_distance (px)** — Max distance to merge nearby line detections. Range: **[3, 8]** (baseline 3).

- **angle_range_oblique_positive (deg)** — Stored as a two-element list `[low, high]` for positive-slope oblique acceptance. Each endpoint searchable within **[4, 12]**; must satisfy low ≤ high. Baseline `[6, 9]`.

- **angle_range_oblique_negative (deg)** — Same for negative slopes; each endpoint within **[-12, -4]**; must satisfy low ≤ high (both negative). Baseline `[-9, -6]`.

## Tunable parameters — Geometry priors

- **ring_spacing_constant (m)** — Expected spacing between rings along the tunnel axis. Range: **[1.2, 1.8]**.

- **dilation_iterations** — Morphological dilation iterations (integer). Range: **[1, 3]** (baseline 1).

- **morphological_kernel_size** — Kernel shape for dilation; use odd sizes, typically **3×3** or **5×5** (list of two equal integers in `{3, 5}`). Baseline `[3, 3]`.

## Locked parameters (baseline, not searched)

| Parameter | Baseline |
|---|---|
| resolution | 0.005 |

## Constraints

- **angle_range_oblique_positive**: valid ordered pair with both values in [4, 12].
- **angle_range_oblique_negative**: valid ordered pair with both values in [-12, -4].
- **ring_spacing_constant** should be consistent with upstream **slice_spacing_factor** when both are optimized.
