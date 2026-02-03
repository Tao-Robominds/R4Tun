# Raw Point Cloud Characteristics

Critical characteristics that differentiate tunnels and inform preprocessing parameter choices.

## Characteristics → Parameter Mapping

| Characteristic | What It Measures | → Preprocessing Parameter |
|----------------|------------------|---------------------------|
| `cross_section_radius_m` | Median tunnel radius | `radius_min`, `radius_max` |
| `median_nn_distance_m` | Point spacing | `depth_map_resolution`, `target_distances` |
| `density_cv` | Density variation (CV) | `gradient_threshold`, `curvature_neighbors` |

### 1. cross_section_radius_m
- **Extraction:** PCA → principal axis → median distance from axis
- **Reasoning:** 
  - `radius_min` ≈ radius - 0.05m (tight bound)
  - `radius_max` ≈ radius + 0.05m (tight bound)
- **Example:** radius=2.52m → radius_min=2.47, radius_max=2.57

### 2. median_nn_distance_m
- **Extraction:** Median of k-NN distances (k=5)
- **Reasoning:**
  - `depth_map_resolution` ≈ 0.1–0.2× median_nn (finer than spacing)
  - `target_distances` start at 2–3× median_nn
- **Example:** nn=0.039m → resolution≈0.005m, targets=[0.08, 0.04, 0.02]

### 3. density_cv
- **Extraction:** Coefficient of variation of local point density
- **Reasoning:**
  - High CV (>0.5): variable density → lower `gradient_threshold` (e.g., 0.1)
  - Low CV (<0.3): uniform density → higher `gradient_threshold` (e.g., 0.3)
- **Example:** CV=0.43 → gradient_threshold≈0.2

## Output

```json
{
  "cross_section_radius_m": 2.52,
  "median_nn_distance_m": 0.039,
  "density_cv": 0.43
}
```

## Usage

```bash
python bo4tun_agents/simple_staggered/1_preprocessing/states/extract_raw_characteristics.py 1-4 [--output path]
```
