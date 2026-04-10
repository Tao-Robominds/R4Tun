# Memory-ablation LLM context — tunnel `5-6`

This document is the **same user message** the memory-ablation stage analyst builds (raw characteristics only). Use it for copy-paste into any chat or API.

Regenerate after updating raw characteristics or the tunnel archive under `agents/ablation/memory/parameters/<tunnel_id>/` (else falls back to `agents/parameters/sample/`):

```bash
./venv/bin/python skills/scripts/export_llm_parameter_context.py 5-6
```

---

# ROLE
You are a tuning expert for a point cloud density-difference-based denoising pipeline. Your goal is to adapt the algorithm based on tunnel-specific characteristics provided.


# SAMPLE TUNNEL — RAW CHARACTERISTICS (reference)
```json
{
  "tunnel_id": "sample",
  "input_file": "/home/boringtao/Projects/R4Tun/data/sample.txt",
  "filtered_note": "Contains only characteristics for x, y, z, intensity columns. Ground truth data (segment_type, ring_number) excluded.",
  "point_cloud_analysis": {
    "basic_statistics": {
      "total_points": 1109768,
      "data_structure": {
        "columns": 4,
        "description": "x, y, z, intensity"
      },
      "coordinate_ranges": {
        "x_range": [
          -4.72192383,
          2.286865
        ],
        "y_range": [
          -15.82104492,
          -3.17114305
        ],
        "z_range": [
          -1.40405297,
          3.67260695
        ],
        "intensity_range": [
          -1727.0,
          1899.0
        ]
      }
    },
    "tunnel_geometry": {
      "dimensions": {
        "length_x_axis": 12.155931503734362,
        "width_y_axis": 5.604292068996665,
        "height_z_axis": 5.07665992,
        "units": "meters"
      },
      "estimated_diameter": 5.604292068996665,
      "diameter_estimation": {
        "inner_diameter": 5.604292068996665,
        "outer_diameter": 5.604292068996665,
        "average_diameter": 5.604292068996665,
        "median_diameter": 5.604292068996665,
        "ring_thickness": null,
        "ring_thickness_note": "Not estimated from minimum bounding rectangle; use unfolded/denoised characterisers for r-based ring thickness.",
        "description": "Estimated tunnel diameter based on minimum bounding rectangle width (2D XOY projection). May include surrounding infrastructure.",
        "method": "minimum_bounding_rectangle",
        "note": "This is a 2D projection-based estimate. For more accurate diameter estimation, use cylindrical coordinate analysis (r values) from unfolded point cloud."
      },
      "diameter_discrepancy_note": "Estimated diameter may include surrounding infrastructure"
    },
    "point_density": {
      "mean_nearest_neighbor_distance": 0.008184481631340645,
      "median_nearest_neighbor_distance": 0.006514481254712708,
      "min_nearest_neighbor_distance": 0.0004879300000000253,
      "max_nearest_neighbor_distance": 0.2442797068280462,
      "units": "meters"
    }
  }
}
```

# TARGET TUNNEL — RAW CHARACTERISTICS (tunnel_id=5-6)
```json
{
  "tunnel_id": "5-6",
  "input_file": "/home/boringtao/Projects/R4Tun/data/subsets/5-6.txt",
  "filtered_note": "Contains only characteristics for x, y, z, intensity columns. Ground truth data (segment_type, ring_number) excluded.",
  "point_cloud_analysis": {
    "basic_statistics": {
      "total_points": 1968165,
      "data_structure": {
        "columns": 4,
        "description": "x, y, z, intensity"
      },
      "coordinate_ranges": {
        "x_range": [
          -10.75610447,
          8.60229492
        ],
        "y_range": [
          -8.38793945,
          7.02416992
        ],
        "z_range": [
          -1.37036097,
          6.26928711
        ],
        "intensity_range": [
          -1727.0,
          1887.0
        ]
      }
    },
    "tunnel_geometry": {
      "dimensions": {
        "length_x_axis": 18.044991813800063,
        "width_y_axis": 7.761256359430422,
        "height_z_axis": 7.63964808,
        "units": "meters"
      },
      "estimated_diameter": 7.761256359430422,
      "diameter_estimation": {
        "inner_diameter": 7.761256359430422,
        "outer_diameter": 7.761256359430422,
        "average_diameter": 7.761256359430422,
        "median_diameter": 7.761256359430422,
        "ring_thickness": null,
        "ring_thickness_note": "Not estimated from minimum bounding rectangle; use unfolded/denoised characterisers for r-based ring thickness.",
        "description": "Estimated tunnel diameter based on minimum bounding rectangle width (2D XOY projection). May include surrounding infrastructure.",
        "method": "minimum_bounding_rectangle",
        "note": "This is a 2D projection-based estimate. For more accurate diameter estimation, use cylindrical coordinate analysis (r values) from unfolded point cloud."
      },
      "diameter_discrepancy_note": "Estimated diameter may include surrounding infrastructure"
    },
    "point_density": {
      "mean_nearest_neighbor_distance": 0.00803307057932207,
      "median_nearest_neighbor_distance": 0.006514364339288732,
      "min_nearest_neighbor_distance": 0.0004878000000001492,
      "max_nearest_neighbor_distance": 0.29244103213469663,
      "units": "meters"
    }
  }
}
```

# REFERENCE DENOISING PARAMETERS
Archived tunnel parameters (same file you will save as `agents/ablation/memory/parameters/5-6/parameters_denoising.json`).

```json
{
  "mask_r_low": 2.7,
  "mask_r_high": 2.8,
  "y_step": 0.5,
  "z_step": 0.001,
  "grad_threshold": 0.2,
  "smoothing_window_size": 3,
  "smoothing_offset": -0.003,
  "default_cutoff_z": 2.7
}
```

# PIPELINE CODE (reference)
```python
# Algorithm 2 - Local Point Cloud Density-Difference-Based Denoising extracted from notebook

# # Algorithm 2: Local point cloud density-difference-based denoising

# Cell 1
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
from scipy.ndimage import uniform_filter1d
from numba import njit, prange
import os
import sys

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python 2_denoising.py <tunnel_id>")
    print("Example: python 2_denoising.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]
base_dir = f"data/{tunnel_id}/"
unwrapped_file = os.path.join(base_dir, "unwrapped.csv")
df_point_cloud = pd.read_csv(unwrapped_file)
ring_count = int(open(f'data/{tunnel_id}/ring_count.txt', 'r').read())

print(f"Processing tunnel: {tunnel_id}")

# Add a 'pred' column and initialize to 7
df_point_cloud['pred'] = 7

# Initial filter based on 'r' column
mask_r = (df_point_cloud['r'] < 2.7)|(df_point_cloud['r'] > 2.8) # diameter is 5.5
df_point_cloud.loc[mask_r, 'pred'] = 0

# Remaining point cloud data
filtered_df = df_point_cloud[~mask_r].copy()

# Define bins for X, Y, and Z directions
x_points = filtered_df['h'].values
y_points = filtered_df['theta'].values
z_points = filtered_df['r'].values

min_x, max_x = np.min(x_points), np.max(x_points)
min_y, max_y = np.min(y_points), np.max(y_points)
min_z, max_z = np.min(z_points), np.max(z_points)

# Set grid sizes
x_step = (max_x - min_x) / ring_count
y_step = 0.5
z_step = 0.001

x_bins = np.arange(min_x, max_x + x_step, x_step)
y_bins = np.arange(min_y, max_y + y_step, y_step)
z_bins = np.arange(min_z, max_z + z_step, z_step)

# Pre-compute useful variables
grad_threshold = 0.2
epsilon = 1e-6

@njit(parallel=True)
def calculate_counts_matrix(y_points_sub, z_points_sub, y_bins, z_bins):
    counts_matrix = np.zeros((len(y_bins) - 1, len(z_bins) - 1))
    for i in prange(len(y_bins) - 1):
        y_min, y_max = y_bins[i], y_bins[i + 1]
        for j in range(len(z_bins) - 1):
            z_min, z_max = z_bins[j], z_bins[j + 1]
            mask = (y_points_sub >= y_min) & (y_points_sub < y_max) & (z_points_sub >= z_min) & (z_points_sub < z_max)
            counts_matrix[i, j] = np.sum(mask)
    return counts_matrix

@njit(parallel=True)
def calculate_cutoff_z_values(counts_matrix, z_bins, grad_threshold, epsilon):
    cutoff_z_values = np.full(counts_matrix.shape[0], 2.7)
    max_z_temp_values = np.zeros(counts_matrix.shape[0])
    
    for i in prange(counts_matrix.shape[0]):
        counts = counts_matrix[i, :]
        
        if np.all(counts == 0):
            continue
        
        max_count_idx = np.argmax(counts)
        grad_counts = np.diff(counts) / (counts[:-1] + epsilon)
        
        max_z_temp_values[i] = z_bins[max_count_idx]
        
        last_non_zero_idx = max_count_idx
        for j in range(max_count_idx, 0, -1):
            if counts[j] != 0:
                last_non_zero_idx = j
                
            if grad_counts[j - 1] < -grad_threshold or (counts[j] == 0 and counts[j - 1] == 0):
                cutoff_z_values[i] = z_bins[last_non_zero_idx]
                break
                
    return cutoff_z_values, max_z_temp_values

# Initialize list to store filtered points and count matrices
filtered_points_list = []
count_matrices = []

# Iterate over X bins
for x_min in x_bins[:-1]:
    x_max = x_min + x_step
    mask_x = (x_points >= x_min) & (x_points < x_max)
    y_points_sub = y_points[mask_x]
    z_points_sub = z_points[mask_x]

    # Calculate count matrix using numba
    counts_matrix = calculate_counts_matrix(y_points_sub, z_points_sub, y_bins, z_bins)
    count_matrices.append(counts_matrix)

    # Calculate cutoff values using numba
    cutoff_z_values, max_z_temp_values = calculate_cutoff_z_values(counts_matrix, z_bins, grad_threshold, epsilon)

    # Handle NaNs and smoothing
    nan_indices = np.isnan(cutoff_z_values)
    not_nan_indices = ~nan_indices

    if np.any(nan_indices):
        interp_func = interp1d(
            np.where(not_nan_indices)[0],
            cutoff_z_values[not_nan_indices],
            kind='linear',
            fill_value='extrapolate'
        )
        cutoff_z_values[nan_indices] = interp_func(np.where(nan_indices)[0])

    cutoff_z_values_smoothed = uniform_filter1d(cutoff_z_values, size=3, mode='nearest') - 0.003

    # Vectorized filtering based on cutoff values
    y_indices = np.digitize(y_points_sub, y_bins) - 1
    filtered_mask = (z_points_sub >= cutoff_z_values_smoothed[y_indices])
    
    filtered_points_sub = {
        'h': x_points[mask_x][filtered_mask],
        'theta': y_points[mask_x][filtered_mask],
        'r': z_points[mask_x][filtered_mask]
    }
    filtered_points_list.append(filtered_points_sub)

    # Update filtered out points 'pred' to 0
    filtered_out_indices = filtered_df.index[mask_x][~filtered_mask]
    df_point_cloud.loc[filtered_out_indices, 'pred'] = 0

# Save results
denoised_file = os.path.join(base_dir, "denoised.csv")
os.makedirs(base_dir, exist_ok=True)
df_point_cloud.to_csv(denoised_file, index=False)
```

## Input scope
Use **only** the two raw characteristic JSON blobs, the **REFERENCE … PARAMETERS** JSON block above, and the pipeline code. Do not assume unfolded / denoised / enhanced / detected summaries.

## Required final output (must match `parameters_denoising.json`)
Your reply must end with **exactly one** markdown code fence labelled `json`, containing **one** JSON object and nothing else inside the fence.

That object must:
1. Parse with `json.loads` with **no** trailing commas or comments.
2. Have the **same tree of keys** as the **REFERENCE … PARAMETERS** JSON block above at every level — **no added keys, no removed keys, no renamed keys**.
3. Match **types** at every leaf path listed below (object vs array vs number vs integer vs boolean vs string). Preserve **array lengths** exactly.
4. Change **only** values where raw evidence justifies it; otherwise keep the reference numerics / booleans / strings unchanged.
5. For **string** leaves (e.g. segment codes in `segment_order`), keep the same literals unless a change is explicitly justified; **never** invent new keys under `processing`, `prompt_points`, or `template_mask`.

### Leaf paths and types (from reference JSON above)
| JSON path (must exist with this type) | Type |
| --- | --- |
| `default_cutoff_z` | number |
| `grad_threshold` | number |
| `mask_r_high` | number |
| `mask_r_low` | number |
| `smoothing_offset` | number |
| `smoothing_window_size` | integer |
| `y_step` | number |
| `z_step` | number |

### Before the code fence
At most a **short** prose note (optional); **no** CoT section headers. The fence must contain the full parameters object so it can be copied into `parameters_denoising.json`.
