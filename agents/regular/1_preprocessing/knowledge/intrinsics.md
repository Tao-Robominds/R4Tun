# Preprocessing Output Intrinsic Metrics

Critical metrics to determine if preprocessing output is ready for detection.
Each metric has a threshold for pass/fail decision.

## Critical Metrics (Currently Extracted)

| Metric | Good Range | Strict Range | What It Detects |
|--------|------------|--------------|-----------------|
| `pre_theta_coverage_pct` | 98–108% | 99.5–100.5% | Incomplete unfolding, wraparound |
| `pre_depth_map_valid_pixels` | 8k–35k | - | Over-interpolation or too sparse |
| `pre_point_retention_pct` | 65–98% | - | Over/under-denoising |
| `pre_depth_map_max_empty_row_run` | <=100 rows | - | Big white horizontal bands |

### 1. pre_theta_coverage_pct (Unfolding)
- **Source:** `unwrapped.csv` theta column
- **Formula:** (θ_max − θ_min) / 360° × 100
- **Threshold:** [98%, 108%] general, [99.5%, 100.5%] strict
- **Failure mode:** 
  - <98%: Incomplete coverage, loses segments
  - >108%: Wraparound, duplicate segments, or misconfigured tunnel_diameter

### 2. pre_depth_map_valid_pixels (Enhancing → Detection)
- **Source:** `depth_map_outlier.npy`
- **Formula:** Count of non-NaN pixels
- **Threshold:** [8,000 - 35,000]
- **Failure mode:** 
  - >35k (especially >100k): Over-interpolation → false K-points
  - <8k: Too sparse → missed lines

### 3. pre_point_retention_pct (Denoising)
- **Source:** `denoised.csv` vs `unwrapped.csv`
- **Formula:** (valid_denoised / unwrapped) × 100
- **Threshold:** [65%, 98%]
- **Failure mode:**
  - <65%: Aggressive denoising removed valid points
  - >98%: Denoising ineffective

### 4. pre_depth_map_max_empty_row_run (Depth Map Quality)
- **Source:** `depth_map.png`
- **Formula:** Maximum number of consecutive rows where white-pixel fraction > 80%
- **Threshold:** <= 100 rows (approximately 4% of typical depth map height)
- **Failure mode:**
  - >100 rows: Large horizontal white bands indicate missing rings or sparse data. Detection (Hough transform) will miss K-block intersections in those regions, and SAM will have no image data to segment.

## Output Format

```json
{
  "pre_theta_coverage_pct": 105.6,
  "pre_point_retention_pct": 73.1,
  "pre_depth_map_valid_pixels": 14328,
  "pre_depth_map_max_empty_row_run": 48,
  "pre_ready_for_detection": true,
  "pre_guardrail_violations": []
}
```
