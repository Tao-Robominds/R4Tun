# Preprocessing Output Intrinsic Metrics

Critical metrics to determine if preprocessing output is ready for detection.
Each metric has a threshold for pass/fail decision.

## Critical Metrics (Currently Extracted)

| Metric | Good Range | Strict Range | What It Detects |
|--------|------------|--------------|-----------------|
| `pre_theta_coverage_pct` | 98–102% | 99.5–100.5% | Incomplete unfolding, wraparound |
| `pre_depth_map_valid_pixels` | 8k–35k | - | Over-interpolation or too sparse |
| `pre_point_retention_pct` | 70–98% | - | Over/under-denoising |

### 1. pre_theta_coverage_pct (Unfolding)
- **Source:** `unwrapped.csv` theta column
- **Formula:** (θ_max − θ_min) / 360° × 100
- **Threshold:** [98%, 102%] general, [99.5%, 100.5%] strict
- **Failure mode:** 
  - <98%: Incomplete coverage, loses segments
  - >102%: Wraparound, duplicate segments

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
- **Threshold:** [70%, 98%]
- **Failure mode:**
  - <70%: Aggressive denoising removed valid points
  - >98%: Denoising ineffective

## Output Format

```json
{
  "pre_theta_coverage_pct": 99.9,
  "pre_point_retention_pct": 73.1,
  "pre_depth_map_valid_pixels": 14328,
  "pre_ready_for_detection": true,
  "pre_guardrail_violations": []
}
```
