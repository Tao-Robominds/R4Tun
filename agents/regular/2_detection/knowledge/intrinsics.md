# Detection Output Intrinsic Metrics

Critical metrics to determine if detection output is ready for SAM segmentation.
Each metric has a threshold for pass/fail decision.

## Critical Metrics (Currently Extracted)

| Metric | Good Range | What It Detects |
|--------|------------|-----------------|
| `det_k_count_match` | == ring_count | Missing or phantom rings |
| `det_x_spacing_cv` | <= 0.15 | Uneven horizontal spacing |
| `det_midpoint_ratio` | >= 0.50 | Detection confidence/quality |
| `det_y_pattern_consistency` | <= 3.0% | Y position pattern regularity |

### 1. det_k_count_match (Ring Count)

- **Source:** `detected.csv` row count vs `ring_count.txt`
- **Formula:** `len(detected) == ring_count` (boolean)
- **Threshold:** Must be exact match
- **Failure mode:**
  - `False`: Missing rings (detected < ring_count) or phantom rings (detected > ring_count). Caused by poor vertical line detection (`hough_vertical_threshold` too high/low) or excessive dilation merging lines.

### 2. det_x_spacing_cv (Horizontal Spacing Regularity)

- **Source:** `detected.csv` X column
- **Formula:** Coefficient of variation (std/mean) of consecutive X gaps
- **Threshold:** <= 0.15 (15% variation)
- **Failure mode:**
  - >0.15: Uneven ring spacing indicates missing rings (creates 2x gaps), merged rings, or split rings. The CV metric correctly flags these issues even when count matches.

### 3. det_midpoint_ratio (Detection Confidence)

- **Source:** `detected.csv` Type column
- **Formula:** Fraction of K positions detected via "midpoint" method (both positive and negative slope lines found)
- **Threshold:** >= 0.50 (at least half detected with both slopes)
- **Failure mode:**
  - <0.50: Low confidence detection. Many K positions relied on single-slope fallback (`positive_slope`, `negative_slope`) or assumptions (`assume`, `default`). Indicates poor oblique line detection quality. Retune `angle_*`, `hough_oblique_threshold`, `binary_threshold`.

### 4. det_y_pattern_consistency (Y Position Pattern)

- **Source:** `detected.csv` Y column + `depth_map_outlier.npy` (for image height)
- **Formula:** Split Y into even-index and odd-index groups. Compute average intra-group std, expressed as % of image height. Works for both continuous and staggered patterns without needing to know the pattern type.
- **Threshold:** <= 3.0% of image height
- **Failure mode:**
  - >3.0%: Inconsistent Y positions. For continuous tunnels, K-blocks should be horizontally aligned (both groups have same mean, low std). For staggered tunnels, K-blocks alternate (two distinct means, but each group internally consistent). High score indicates wrong K Y-positions, broken alternation, or inconsistent vertical alignment.

## Guardrail Summary

| Metric | Guardrail | Action if Failed |
|--------|-----------|-----------------|
| `det_k_count_match` | == ring_count | Retune `hough_vertical_threshold`, `dilation_kernel_size`, `dilation_iterations` |
| `det_x_spacing_cv` | <= 0.15 | Check for missing/extra rings, retune vertical detection parameters |
| `det_midpoint_ratio` | >= 0.50 | Retune `angle_positive_min/max`, `angle_negative_min/max`, `hough_oblique_threshold`, `binary_threshold` |
| `det_y_pattern_consistency` | <= 3.0% | Retune `angle_*` parameters, check preprocessing quality (depth map) |

## Output Format

```json
{
  "det_k_count_match": true,
  "det_x_spacing_cv": 0.0000,
  "det_midpoint_ratio": 0.80,
  "det_y_pattern_consistency": 0.04,
  "det_ready_for_sam": true,
  "det_guardrail_violations": []
}
```

## Known-Good Values

| Tunnel | k_count_match | x_spacing_cv | midpoint_ratio | y_pattern_consistency | Status |
|--------|---------------|--------------|----------------|----------------------|--------|
| 1-4 | False (9 vs 10) | 0.295 | 0.56 | - | ⚠ Missing ring |
| 2-2 | True (10) | 0.000 | 0.80 | 0.04% | ✅ All pass |
| 3-1 | True (6) | 0.000 | 0.83 | 0.30% | ✅ All pass |
