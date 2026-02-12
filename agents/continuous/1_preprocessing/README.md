# Simplified Preprocessing Pipeline for Bayesian Optimization

This module consolidates the three preprocessing stages (Unfolding, Denoising, Enhancing) into a single script with **only critical parameters** exposed for experimentation.

## Scope

**Simple Staggered Tunnels:** 1-4, 2-2, sample

> Note: Tunnel 3-1 is in `bo4tun_agents/continuous/` as it may require different agentic strategies.

## Background

Based on extensive P4TUN optimization experiments (documented in `/reports/`):

| Stage | BO Improvement | Finding |
|-------|----------------|---------|
| **Unfolding** | +0.0% mIoU | Defaults already optimal |
| **Denoising** | +0.1% (combined) | `gradient_threshold` most sensitive |
| **Enhancing** | +0.1% (combined) | Limited value beyond baseline |
| **Detection** | +6.3% mIoU | **Highest impact** |
| **SAM** | +4.2% - 7.4% mIoU | Second highest impact |

**Key insight:** Preprocessing contributes minimally to final mIoU. Detection and SAM stages have the highest impact.

## Critical Parameters (8 total)

| Parameter | Stage | Sensitivity | BO Range | Description |
|-----------|-------|-------------|----------|-------------|
| `ring_spacing` | Unfolding | MEDIUM | Tunnel-specific | Ring width in meters |
| `tunnel_diameter` | Unfolding | LOW | Tunnel-specific | Tunnel diameter for θ calculation |
| `radius_min` | Denoising | **VERY HIGH** | ~(diameter/2 - 0.05) | Must match actual tunnel radius |
| `radius_max` | Denoising | **VERY HIGH** | ~(diameter/2 + 0.05) | Must match actual tunnel radius |
| `gradient_threshold` | Denoising | **HIGH** | 0.1 - 0.4 | Lower = more aggressive denoising |
| `target_distances` | Enhancing | HIGH | [0.06-0.1, 0.03-0.05, 0.015-0.025] | Upsampling density levels |
| `curvature_neighbors` | Enhancing | MEDIUM | 15 - 30 | Surface smoothness |
| `depth_map_resolution` | Enhancing | **HIGH** | 0.003 - 0.008 | Affects all downstream stages |

## Fixed Parameters (Non-Critical)

All other parameters use proven defaults based on BO experiments showing negligible improvement:

### Unfolding (fixed)
- `slice_half_thickness`: 0.005
- `max_distance_from_top`: 4.5
- `polynomial_degree`: 3
- RANSAC parameters (inlier_ratio, confidence, min_samples, inlier_threshold)
- `samples_per_ring`: 1210
- Performance parameters (batch_size, num_jobs)

### Denoising (fixed)
- `theta_step`: 0.5
- `radial_step`: 0.001
- `gradient_epsilon`: 1e-6
- `smoothing_window`: 3
- `smoothing_offset`: 0.003

### Enhancing (fixed)
- Upsampling parameters (curvature_threshold, distance_tolerance_*, etc.)
- Outlier detection parameters
- Outlier interpolation parameters
- `interpolation_window`: 9

## Usage

```bash
# Run preprocessing on tunnel 1-4
python 1_preprocessing.py 1-4

# Run preprocessing on tunnel 2-2
python 1_preprocessing.py 2-2
```

## Parameters File Structure

```
parameters/
├── sample/
│   └── parameters_preprocessing.json   # Template with all documentation
├── 1-4/
│   └── parameters_preprocessing.json   # Tunnel 1-4 specific
└── 2-2/
    └── parameters_preprocessing.json   # Tunnel 2-2 specific (gradient_threshold=0.1)
```

## Example Parameters File

```json
{
    "ring_spacing": 1.2,
    "tunnel_diameter": 5.5,
    "radius_min": 2.7,
    "radius_max": 2.8,
    "gradient_threshold": 0.2,
    "target_distances": [0.08, 0.04, 0.02],
    "curvature_neighbors": 20,
    "depth_map_resolution": 0.005
}
```

## Outputs

The preprocessing pipeline generates:

| File | Description |
|------|-------------|
| `unwrapped.csv` | Point cloud in cylindrical coordinates (r, θ, h) |
| `ring_count.txt` | Number of detected rings |
| `denoised.csv` | Point cloud with noise removed (pred=0 for noise) |
| `enhanced.csv` | Upsampled point cloud with boundary enhancement |
| `depth_map.png` | 2D depth map image for detection |
| `depth_map_outlier.npy` | Depth map with outlier enhancement |
| `pixel_to_point.pkl` | Pixel-to-point index mapping |
| `pattern_type.json` | Auto-detected tunnel pattern classification |

## Key Findings from Reports

1. **Detection is King** - Detection optimization provided +6.3% mIoU, the largest single-stage improvement
2. **Preprocessing at Ceiling** - Combined preprocessing yielded only +0.1% improvement
3. **radius_min/max are Critical** - Must match actual tunnel radius or denoising fails catastrophically
4. **gradient_threshold Matters** - BO found 0.1 (aggressive) best for tunnel 2-2

## Comparison with Original Pipeline

| Aspect | Original (p4tun/) | Simplified (bo4tun_agents/) |
|--------|-------------------|---------------------------|
| Files | 3 separate scripts | 1 consolidated script |
| Parameters | 40+ across 3 JSON files | 8 critical in 1 JSON file |
| Tunable params | All parameters | Only critical parameters |
| Non-critical | Tunable | Fixed to proven defaults |
| Purpose | Full flexibility | BO experimentation |

## References

- `/reports/P4TUN_OPTIMIZATION_JOURNEY_2-2.md` - Detailed BO optimization findings
- `/reports/P4TUN_PARAMETERIZATION_JOURNEY.md` - Complete parameter documentation
- `/reports/PREPROCESSING_VERIFICATION_RESULTS.md` - Pipeline verification
