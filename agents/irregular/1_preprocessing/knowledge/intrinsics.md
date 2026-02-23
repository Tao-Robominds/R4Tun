# Preprocessing Output Intrinsic Metrics (Irregular / Complex Staggered)

Critical metrics to determine if preprocessing output is ready for detection.
Adapted from regular pipeline; thresholds adjusted for tunnel_diameter=7.5m and larger depth maps.

## Critical Metrics

| Metric | Good Range | What It Detects |
|--------|------------|-----------------|
| `pre_theta_coverage_pct` | 50-108% | Incomplete unfolding, wraparound (note: 4-1 is 73% partial scan) |
| `pre_depth_map_valid_pixels` | 8k-100k | Over-interpolation or too sparse |
| `pre_point_retention_pct` | 55-98% | Over/under-denoising |
| `pre_depth_map_max_empty_row_run` | <=350 rows | Big white horizontal bands |

### 1. pre_theta_coverage_pct (Unfolding)
- **Source:** `unwrapped.csv` theta column
- **Formula:** (theta_max - theta_min) / 360 x 100
- **Threshold:** [50%, 108%] (lower bound permissive: 4-1 is a 73% partial scan)
- **Failure mode:**
  - <50%: Severely incomplete coverage, loses multiple rings
  - >108%: Wraparound, duplicate segments, or misconfigured tunnel_diameter

### 2. pre_depth_map_valid_pixels (Enhancing -> Detection)
- **Source:** `depth_map_outlier.npy`
- **Formula:** Count of non-NaN pixels
- **Threshold:** [8,000 - 100,000] (larger depth maps: 12M total pixels)
- **Failure mode:**
  - >100k: Over-interpolation -> false K-points
  - <8k: Too sparse -> missed lines

### 3. pre_point_retention_pct (Denoising)
- **Source:** `denoised.csv` vs `unwrapped.csv`
- **Formula:** (valid_denoised / unwrapped) x 100
- **Threshold:** [55%, 98%] (lower bound relaxed: GT-optimal denoising with double_zero_cutoff=false keeps more BG)
- **Failure mode:**
  - <55%: Aggressive denoising removed valid points
  - >98%: Denoising ineffective

### 4. pre_depth_map_max_empty_row_run (Depth Map Quality)
- **Source:** `depth_map.png`
- **Formula:** Maximum number of consecutive rows where white-pixel fraction > 80%
- **Threshold:** <= 350 rows (4-1 has 301 from partial scan edge effects; 5-1 has 113)
- **Failure mode:**
  - >350 rows: Large horizontal white bands indicate missing rings or sparse data

## Differences from Regular Pipeline

| Aspect | Regular | Irregular |
|--------|---------|-----------|
| tunnel_diameter | 5.5-5.89m | 7.5m |
| Depth map size | ~2400x1600 | 4712x2549 (5-1), 3459x3261 (4-1) |
| theta_coverage lower | 98% | 50% (partial scans) |
| point_retention lower | 65% | 55% |
| valid_pixels upper | 35k | 100k |
| max_empty_row_run | <=100 | <=350 |
| Denoising params | Default (aggressive) | GT-optimal (double_zero_cutoff=false, smoothing_offset=0.0) |
