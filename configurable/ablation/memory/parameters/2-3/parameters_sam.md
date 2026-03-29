# Memory-ablation LLM context — tunnel `2-3`

This document is the **same user message** the memory-ablation stage analyst builds (raw characteristics only). Use it for copy-paste into any chat or API.

Regenerate after updating raw characteristics or the tunnel archive under `configurable/ablation/memory/parameters/<tunnel_id>/` (else falls back to `configurable/sample/`):

```bash
./venv/bin/python skills/scripts/export_llm_parameter_context.py 2-3
```

---

# ROLE
You are a tuning expert for a Segment Anything Model (SAM) based segmentation pipeline. Your goal is to adapt the algorithm based on tunnel-specific characteristics provided.


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

# TARGET TUNNEL — RAW CHARACTERISTICS (tunnel_id=2-3)
```json
{
  "tunnel_id": "2-3",
  "input_file": "/home/boringtao/Projects/R4Tun/data/subsets/2-3.txt",
  "filtered_note": "Contains only characteristics for x, y, z, intensity columns. Ground truth data (segment_type, ring_number) excluded.",
  "point_cloud_analysis": {
    "basic_statistics": {
      "total_points": 2049003,
      "data_structure": {
        "columns": 4,
        "description": "x, y, z, intensity"
      },
      "coordinate_ranges": {
        "x_range": [
          -14.46166992,
          13.53979492
        ],
        "y_range": [
          -7.10864305,
          6.6340332
        ],
        "z_range": [
          -2.53198195,
          3.51098609
        ],
        "intensity_range": [
          -1729.0,
          1989.0
        ]
      }
    },
    "tunnel_geometry": {
      "dimensions": {
        "length_x_axis": 27.741097592932118,
        "width_y_axis": 5.64122858327027,
        "height_z_axis": 6.04296804,
        "units": "meters"
      },
      "estimated_diameter": 5.64122858327027,
      "diameter_estimation": {
        "inner_diameter": 5.64122858327027,
        "outer_diameter": 5.64122858327027,
        "average_diameter": 5.64122858327027,
        "median_diameter": 5.64122858327027,
        "ring_thickness": 0.0,
        "description": "Estimated tunnel diameter based on minimum bounding rectangle width (2D XOY projection). May include surrounding infrastructure.",
        "method": "minimum_bounding_rectangle",
        "note": "This is a 2D projection-based estimate. For more accurate diameter estimation, use cylindrical coordinate analysis (r values) from unfolded point cloud."
      },
      "diameter_discrepancy_note": "Estimated diameter may include surrounding infrastructure"
    },
    "point_density": {
      "mean_nearest_neighbor_distance": 0.007421570466489929,
      "median_nearest_neighbor_distance": 0.005415145612492822,
      "min_nearest_neighbor_distance": 0.00048779999999970514,
      "max_nearest_neighbor_distance": 1.2163606099645212,
      "units": "meters"
    }
  }
}
```

# REFERENCE SAM PARAMETERS
Archived tunnel parameters (same file you will save as `configurable/ablation/memory/parameters/2-3/parameters_sam.json`).

```json
{
  "segment_per_ring": 6,
  "segment_order": [
    "K",
    "B1",
    "A1",
    "A2",
    "A3",
    "B2"
  ],
  "segment_width": 1200,
  "K_height": 1079.92,
  "AB_height": 3239.77,
  "angle": 7.52,
  "use_original_label_distributions": true,
  "processing": {
    "resolution": 0.005,
    "padding": 300,
    "crop_margin": 50,
    "mask_eps": 0.001,
    "y_bounds": [
      4200,
      13100
    ]
  },
  "prompt_points": {
    "k_block": {
      "outer_ring": 700,
      "middle_ring": 500,
      "inner_ring": 348.16,
      "center_ring": 325,
      "spacing_factors": {
        "k_block_spacing": 310.91,
        "vertical_spacing": [
          732.35,
          505.96,
          310.91,
          219.01,
          373.96
        ]
      }
    },
    "ab_blocks": {
      "outer_ring": 700,
      "middle_ring": 511.06,
      "inner_ring": 500,
      "center_ring": 325,
      "fine_spacing": 250,
      "ultra_fine": 162.5,
      "edge_ring": 348.16,
      "edge_spacing": 350,
      "vertical_levels": {
        "level_1": 1719.89,
        "level_2": 1519.89,
        "level_3": 1344.89,
        "level_4": 1090.09,
        "level_5": 817.57,
        "level_6": 545.05,
        "level_7": 272.52,
        "center": 0,
        "special_levels": [
          1298.93,
          1390.84,
          1427.43,
          1612.28,
          1627.49,
          1652.43,
          1673.69,
          1766.08,
          1787.34,
          1812.28,
          1345.01,
          1452.43,
          1473.69,
          1566.08,
          1587.34
        ]
      }
    },
    "template_mask": {
      "k_block": {
        "width": 625,
        "height_pos": 619.16,
        "height_neg": 460.77
      },
      "b1_block": {
        "width": 625,
        "height_top": 1619.89,
        "height_bottom_pos": 1540.69,
        "height_bottom_neg": 1699.08
      },
      "b2_block": {
        "width": 625,
        "height_top_pos": 1540.69,
        "height_top_neg": 1699.08,
        "height_bottom": 1619.89
      },
      "a_blocks": {
        "width": 625,
        "height": 1619.89
      }
    }
  }
}
```

# PIPELINE CODE (reference)
```python
import os
import numpy as np
import pandas as pd
import torch
import cv2
import math
from tqdm import tqdm
import pickle
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path
import sys

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python 4-2_sam.py <tunnel_id>")
    print("Example: python 4-2_sam.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]
base_dir = f"data/{tunnel_id}/"
initial_prompt_points = pd.read_csv(os.path.join(base_dir, "detected.csv"))
pixel_to_point = pickle.load(open(os.path.join(base_dir, "pixel_to_point.pkl"), "rb"))
df_point_cloud = pd.read_csv(os.path.join(base_dir, "enhanced.csv"))
ring_count = int(open(f'data/{tunnel_id}/ring_count.txt', 'r').read())

print(f"Processing tunnel: {tunnel_id}")

sam_checkpoint = "sam4tun/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.imread(f'{base_dir}/depth_map.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def fill_polygon(mask, vertices):
    path = Path(vertices)
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)
    mask[mask_inside] = 1

def generate_template_mask(height, width, prompt_centre, block, resolution=0.005):
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    
    if block == 'K':
        vertices_real = np.array([[x-625,y-619.16],[x-625,y+619.16],[x+625,y+460.77],[x+625,y-460.77]])
    elif block == 'B1':
        vertices_real = np.array([[x-625,y-1619.89],[x-625,y+1540.69],[x+625,y+1699.08],[x+625,y-1619.89]])
    elif block == 'B2':
        vertices_real = np.array([[x-625,y-1540.69],[x-625,y+1619.89],[x+625,y+1619.89],[x+625,y-1699.08]])
    else:
        vertices_real = np.array([[x-625,y-1619.89],[x-625,y+1619.89],[x+625,y+1619.89],[x+625,y-1619.89]])
        
    vertices = vertices_real / (resolution*1000)
    fill_polygon(mask, vertices)
    return mask

def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution=0.005,
                           segment_width=1200, K_height=1079.92, AB_height=3239.77):
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    map_y = map_y * (resolution*1000)
    
    if block == 'K':
        points_real = np.array([
            [x-700,y-732.35],[x-700,y-505.96],[x-700,y-310.91],[x-700,y],[x-700,y+310.91],[x-700,y+505.96],[x-700,y+732.35],
            [x-500,y-705.96],[x-500,y+705.96],
            [x-348.16,y-685.91],[x-348.16,y-310.91],[x-325,y],[x-348.16,y+310.91],[x-348.16,y+685.91],
            [x,y-639.96],[x,y],[x,y+639.96],
            [x+348.16,y-594.01],[x+348.16,y-219.01],[x+325,y],[x+348.16,y+219.01],[x+348.16,y+594.01],
            [x+500,y-573.96],[x+500,y+573.96],
            [x+700,y-547.57],[x+700,y-373.96],[x+700,y-219.01],[x+700,y],[x+700,y+219.01],[x+700,y+373.96],[x+700,y+547.57],
            [x-500,y-505.96],[x-511.06,y-310.91],[x-500,y],[x-511.06,y+310.91],[x-500,y+505.96],
            [x-348.16,y-485.91],[x-348.16,y+485.91],
            [x,y-439.96],[x,y+439.96],
            [x+348.16,y-394.01],[x+348.16,y+394.01],
            [x+500,y-373.96],[x+511.06,y-219.01],[x+500,y],[x+511.06,y+219.01],[x+500,y+373.96]
        ])
        labels = np.repeat([0, 1], [31, 16])
    elif block == 'B1':
        points_real = np.array([
            [x-700,y-1719.89],[x-511.06,y-1719.89],[x-348.16,y-1719.89],[x,y-1719.89],[x+348.16,y-1719.89],[x+511.06,y-1719.89],[x+700,y-1719.89],
            [x-700,y-1519.89],[x+700,y-1519.89],
            [x-700,y-1344.89],[x-348.16,y-1344.89],[x+348.16,y-1344.89],[x+700,y-1344.89],
            [x-700,y-1090.09],[x-325,y-1090.09],[x+325,y-1090.09],[x+700,y-1090.09],
            [x-700,y-817.57],[x+700,y-817.57],
            [x-700,y-545.05],[x+700,y-545.05],
            [x-700,y-272.52],[x+700,y-272.52],
            [x-700,y],[x-325,y],[x,y],[x+325,y],[x+700,y],
            [x-700,y+272.52],[x+700,y+272.52],
            [x-700,y+545.05],[x+700,y+545.05],
            [x-700,y+817.57],[x+700,y+817.57],
            [x-700,y+1090.09],[x-325,y+1090.09],[x+325,y+1090.09],[x+700,y+1090.09],
            [x-700,y+1298.93],[x-350,y+1298.93],[x+350,y+1390.84],[x+700,y+1390.84],
            [x-700,y+1427.43],[x+700,y+1612.28],
            [x-700,y+1627.49],[x-511.06,y+1652.43],[x-350,y+1673.69],[x,y+1719.89],[x+350,y+1766.08],[x+511.06,y+1787.34],[x+700,y+1812.28],
            [x-511.06,y-1519.89],[x-348.16,y-1519.89],[x,y-1519.89],[x+348.16,y-1519.89],[x+511.06,y-1519.89],
            [x-511.06,y-1344.89],[x,y-1344.89],[x+511.06,y-1344.89],
            [x-500,y-1090.09],[x,y-1090.09],[x+500,y-1090.09],
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y-817.57],
            [x-500,y-545.05],[x-250,y-545.05],[x,y-545.05],[x+250,y-545.05],[x+500,y-545.05],
            [x-500,y-272.52],[x-250,y-272.52],[x,y-272.52],[x+250,y-272.52],[x+500,y-272.52],
            [x-500,y],[x-162.5,y],[x+162.5,y],[x+500,y],
            [x-500,y+272.52],[x-250,y+272.52],[x,y+272.52],[x+250,y+272.52],[x+500,y+272.52],
            [x-500,y+545.05],[x-250,y+545.05],[x,y+545.05],[x+250,y+545.05],[x+500,y+545.05],
            [x-500,y+817.57],[x-250,y+817.57],[x,y+817.57],[x+250,y+817.57],[x+500,y+817.57],
            [x-500,y+1090.09],[x,y+1090.09],[x+500,y+1090.09],
            [x-511.06,y+1298.93],[x,y+1345.01],[x+511.06,y+1390.84],
            [x-511.06,y+1452.43],[x-350,y+1473.69],[x,y+1519.89],[x+350,y+1566.08],[x+511.06,y+1587.34]      
        ])
        labels = np.repeat([0,1],[51,56])
    elif block == 'B2':
        points_real = np.array([
            [x-700,y-1627.49],[x-511.06,y-1652.43],[x-350,y-1673.69],[x,y-1719.89],[x+350,y-1766.08],[x+511.06,y-1787.34],[x+700,y-1812.28],
            [x-700,y-1427.43],[x+700,y-1612.28],
            [x-700,y-1298.93],[x-350,y-1298.93],[x+350,y-1390.84],[x+700,y-1390.84],            
            [x-700,y-1090.09],[x-325,y-1090.09],[x+325,y-1090.09],[x+700,y-1090.09],
            [x-700,y-817.57],[x+700,y-817.57],
            [x-700,y-545.05],[x+700,y-545.05],
            [x-700,y-272.52],[x+700,y-272.52],
            [x-700,y],[x-325,y],[x,y],[x+325,y],[x+700,y],
            [x-700,y+272.52],[x+700,y+272.52],
            [x-700,y+545.05],[x+700,y+545.05],
            [x-700,y+817.57],[x+700,y+817.57],
            [x-700,y+1090.09],[x-325,y+1090.09],[x+325,y+1090.09],[x+700,y+1090.09],
            [x-700,y+1344.89],[x-348.16,y+1344.89],[x+348.16,y+1344.89],[x+700,y+1344.89],
            [x-700,y+1519.89],[x+700,y+1519.89],
            [x-700,y+1719.89],[x-511.06,y+1719.89],[x-348.16,y+1719.89],[x,y+1719.89],[x+348.16,y+1719.89],[x+511.06,y+1719.89],[x+700,y+1719.89],
            [x-511.06,y-1452.43],[x-350,y-1473.69],[x,y-1519.89],[x+350,y-1566.08],[x+511.06,y-1587.34],     
            [x-511.06,y-1298.93],[x,y-1345.01],[x+511.06,y-1390.84],
            [x-500,y-1090.09],[x,y-1090.09],[x+500,y-1090.09],
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y+817.57],
            [x-500,y-545.05],[x-250,y-545.05],[x,y-545.05],[x+250,y-545.05],[x+500,y-545.05],
            [x-500,y-272.52],[x-250,y-272.52],[x,y-272.52],[x+250,y-272.52],[x+500,y-272.52],
            [x-500,y],[x-162.5,y],[x+162.5,y],[x+500,y],
            [x-500,y+272.52],[x-250,y+272.52],[x,y+272.52],[x+250,y+272.52],[x+500,y+272.52],
            [x-500,y+545.05],[x-250,y+545.05],[x,y+545.05],[x+250,y+545.05],[x+500,y+545.05],
            [x-500,y+817.57],[x-250,y+817.57],[x,y+817.57],[x+250,y+817.57],[x+500,y+817.57],
            [x-500,y+1090.09],[x,y+1090.09],[x+500,y+1090.09],
            [x-511.06,y+1344.89],[x,y+1344.89],[x+511.06,y+1344.89],
            [x-511.06,y+1519.89],[x-348.16,y+1519.89],[x,y+1519.89],[x+348.16,y+1519.89],[x+511.06,y+1519.89],
        ])
        labels = np.repeat([0,1],[51,56])
    else:
        points_real = np.array([
            [x-700,y-1719.89],[x-511.06,y-1719.89],[x-348.16,y-1719.89],[x,y-1719.89],[x+348.16,y-1719.89],[x+511.06,y-1719.89],[x+700,y-1719.89],
            [x-700,y-1519.89],[x+700,y-1519.89],
            [x-700,y-1344.89],[x-348.16,y-1344.89],[x+348.16,y-1344.89],[x+700,y-1344.89],
            [x-700,y-1090.09],[x-325,y-1090.09],[x+325,y-1090.09],[x+700,y-1090.09],
            [x-700,y-817.57],[x+700,y-817.57],
            [x-700,y-545.05],[x+700,y-545.05],
            [x-700,y-272.52],[x+700,y-272.52],
            [x-700,y],[x-325,y],[x,y],[x+325,y],[x+700,y],
            [x-700,y+272.52],[x+700,y+272.52],
            [x-700,y+545.05],[x+700,y+545.05],
            [x-700,y+817.57],[x+700,y+817.57],
            [x-700,y+1090.09],[x-325,y+1090.09],[x+325,y+1090.09],[x+700,y+1090.09],
            [x-700,y+1344.89],[x-348.16,y+1344.89],[x+348.16,y+1344.89],[x+700,y+1344.89],
            [x-700,y+1519.89],[x+700,y+1519.89],
            [x-700,y+1719.89],[x-511.06,y+1719.89],[x-348.16,y+1719.89],[x,y+1719.89],[x+348.16,y+1719.89],[x+511.06,y+1719.89],[x+700,y+1719.89],
            [x-511.06,y-1519.89],[x-348.16,y-1519.89],[x,y-1519.89],[x+348.16,y-1519.89],[x+511.06,y-1519.89],
            [x-511.06,y-1344.89],[x,y-1344.89],[x+511.06,y-1344.89],
            [x-500,y-1090.09],[x,y-1090.09],[x+500,y-1090.09],
            [x-500,y-817.57],[x-250,y-817.57],[x,y-817.57],[x+250,y-817.57],[x+500,y-817.57],
            [x-500,y-545.05],[x-250,y-545.05],[x,y-545.05],[x+250,y-545.05],[x+500,y-545.05],
            [x-500,y-272.52],[x-250,y-272.52],[x,y-272.52],[x+250,y-272.52],[x+500,y-272.52],
            [x-500,y],[x-162.5,y],[x+162.5,y],[x+500,y],
            [x-500,y+272.52],[x-250,y+272.52],[x,y+272.52],[x+250,y+272.52],[x+500,y+272.52],
            [x-500,y+545.05],[x-250,y+545.05],[x,y+545.05],[x+250,y+545.05],[x+500,y+545.05],
            [x-500,y+817.57],[x-250,y+817.57],[x,y+817.57],[x+250,y+817.57],[x+500,y+817.57],
            [x-500,y+1090.09],[x,y+1090.09],[x+500,y+1090.09],
            [x-511.06,y+1344.89],[x,y+1344.89],[x+511.06,y+1344.89],
            [x-511.06,y+1519.89],[x-348.16,y+1519.89],[x,y+1519.89],[x+348.16,y+1519.89],[x+511.06,y+1519.89],
        ])
        labels = np.repeat([0,1],[51,56])

    keep_mask = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 0:
            y_cond = points_real[i, 1] + map_y < 4200 or points_real[i, 1] + map_y > 13100
            x_cond = abs(points_real[i, 0] - x) <= segment_width * 0.5
            y_limit = K_height if block == 'K' else AB_height
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
            
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    
    points = points_real / (resolution*1000)

    within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (segment_width*0.5+150)/(resolution*1000)) <= image.shape[1])
    points = points[within_bounds]
    labels = labels[within_bounds]
        
    return points, labels

def convert_to_pixel_coords(real_dist, resolution=0.005):
    return int(real_dist / (resolution*1000))

def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution):
    img_height, img_width, _ = image.shape
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)
    y2 = min(cy + crop_height // 2, img_height)

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    prompt_centre_x = cx - x1
    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x,prompt_centre_y)
    
    cropped_template_mask = generate_template_mask(cropped_image.shape[0],cropped_image.shape[1],prompt_centre,block,resolution)
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    return cropped_image, template_mask_logits, prompt_centre

def compute_logits_from_mask(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    assert logits.ndim == 2
    expected_shape = (256, 256)

    if logits.shape == expected_shape:
        pass
    elif logits.shape[0] == logits.shape[1]:
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])
    else:
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        pad_width = ((0, padh), (0, padw))
        logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

    logits = logits[None]
    assert logits.shape == (1, 256, 256)
    return logits

def restore_sam_logits(logits, original_shape):
    orig_h, orig_w = original_shape
    trafo = ResizeLongestSide(max(orig_h, orig_w))
    resized_logits = trafo.apply_image(logits[..., None])
    resized_logits = resized_logits.squeeze()
    resized_logits = resized_logits[:orig_h, :orig_w]
    return resized_logits

def compute_block_label(segment_per_ring):
    block_labels = ['K','B1']
    num_a_labels = segment_per_ring - 3
    block_labels += [f'A{i+1}' for i in range(num_a_labels)]
    block_labels += ['B2']
    return block_labels

def sam_prediction(cropped_image, points, labels, template_mask_logit):
    predictor.set_image(cropped_image)
    mask, score, logit = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=template_mask_logit,
            multimask_output=False,
    )
    return mask, score, logit[0]

def process_row(df_row, image, resolution=0.005, segment_per_ring=6, segment_width=1200, 
                K_height=1079.92, angle=7.52, AB_height=3239.77):
    initial_x, initial_y = df_row['X'], df_row['Y']
    block_labels = compute_block_label(segment_per_ring)

    delta_x = convert_to_pixel_coords(0.5*segment_width + 150, resolution)
    delta_y = 0

    reverse = False
    stop = False
    map_y = 0
    block_label_index = 0

    results = []
    for i in range(segment_per_ring):
        if reverse == False:
            block = block_labels[block_label_index]
            if block_label_index == 0:
                delta_y = convert_to_pixel_coords(0.5*K_height + math.tan(math.radians(angle))*700+100 + 50, resolution)
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + 50, resolution)
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
                else:
                    map_y = map_y - convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y,2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)
        
            if np.any(points[:, 1] < 0):
                within_bounds = (points[:, 1] >= 0)
                points = points[within_bounds]
                labels = labels[within_bounds]
                reverse = True
                
            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)
        
            results.append({
                'left_top': (initial_x-prompt_centre[0], map_y-prompt_centre[1]),
                'block': block,
                'cropped_image': cropped_image,
                'mask': mask,
                'points':points,
                'labels':labels,
                'score': score,
                'logit': logit
            })
            
            if reverse:
                block_label_index = -1
                continue

            block_label_index = block_label_index + 1
            
        if reverse:
            block = block_labels[block_label_index]
            if block_label_index == -1:
                map_y = initial_y + convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
            else:
                map_y = map_y + convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(image, initial_x, map_y, 
                                                                                            2 * delta_x, 2 * delta_y, block, resolution)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)

            if np.any((points[:, 1]+map_y-delta_y) > image.shape[0]):
                within_bounds = ((points[:, 1]+map_y-delta_y) <= image.shape[0])
                points = points[within_bounds]
                labels = labels[within_bounds]
                stop = True

            mask, score, logit = sam_prediction(cropped_image, points, labels, template_mask_logit)

            results.append({
                'left_top': (initial_x-prompt_centre[0], map_y-prompt_centre[1]),
                'block': block,
                'cropped_image': cropped_image,
                'mask': mask,
                'points':points,
                'labels':labels,
                'score': score,
                'logit': logit
            })

            if stop:
                break

            block_label_index = block_label_index - 1
             
    return results

def sam_segment(df, image, resolution=0.005, segment_per_ring=6):
    all_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_row(row, image, resolution, segment_per_ring)
        all_results.append(result)
    return all_results

results = sam_segment(initial_prompt_points, image)

block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}

logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
label_map = np.zeros(image.shape[:2], dtype=int)
ring_map = np.zeros(image.shape[:2], dtype=int)

for ring_index, ring in enumerate(results, start=0):
    for item in ring:
        mask = item['mask'][0]
        logits = item['logit']
        block = item['block']
        start_x, start_y = map(int, item['left_top'])

        end_y, end_x = start_y + mask.shape[0], start_x + mask.shape[1]
        start_y, start_x = max(0, start_y), max(0, start_x)
        end_y, end_x = min(image.shape[0], end_y), min(image.shape[1], end_x)
        
        valid_slice_y = slice(start_y, end_y)
        valid_slice_x = slice(start_x, end_x)

        new_logits = restore_sam_logits(logits, mask.shape)
        current_logits = logits_map[valid_slice_y, valid_slice_x]

        if mask.shape != current_logits.shape or new_logits.shape != current_logits.shape:
            raise ValueError(f"Shape mismatch after resizing: mask {mask.shape}, new_logits {new_logits.shape}, current_logits {current_logits.shape}")

        update_mask = (new_logits > current_logits) & mask
        
        logits_map[valid_slice_y, valid_slice_x][update_mask] = new_logits[update_mask]
        label_map[valid_slice_y, valid_slice_x][update_mask] = block_to_label[block]
        ring_map[valid_slice_y, valid_slice_x][update_mask] = ring_index

result_image = label_map
ring_image = ring_map

fix_ring = np.where((ring_image >= 1) & (ring_image <= (ring_count-1)), ring_count - ring_image, ring_image)

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)

    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    img_height, img_width = segmented_map.shape

    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    valid_update_mask = (pred[point_indices[valid_point_mask]] == 7)
    
    y_valid = y[valid_point_mask][valid_update_mask]
    x_valid = x[valid_point_mask][valid_update_mask]
    
    bounds_mask = (y_valid >= 0) & (y_valid < img_height) & (x_valid >= 0) & (x_valid < img_width)
    
    final_point_indices = point_indices[valid_point_mask][valid_update_mask][bounds_mask]
    final_y = y_valid[bounds_mask]
    final_x = x_valid[bounds_mask]

    pred[final_point_indices] = segmented_map[final_y, final_x]
    pred_ring[final_point_indices] = instance_map[final_y, final_x]

    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring

    return df_copy

updated_df = project_back_to_point_cloud(result_image, fix_ring, pixel_to_point, df_point_cloud)

os.makedirs(base_dir, exist_ok=True)
updated_df.to_csv(f'{base_dir}/final.csv', index=False)

df_pred = pd.DataFrame()
df_pred['gt_labels'] = updated_df['segment']
df_pred['gt_rings'] = updated_df['ring']
df_pred['pred_labels'] = updated_df['pred']
df_pred['pred_rings'] = updated_df['pred_ring']
df_pred.to_csv(f'{base_dir}/only_label.csv', index=False) 
```

## Input scope
Use **only** the two raw characteristic JSON blobs, the **REFERENCE … PARAMETERS** JSON block above, and the pipeline code. Do not assume unfolded / denoised / enhanced / detected summaries.

## Required final output (must match `parameters_sam.json`)
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
| `AB_height` | number |
| `K_height` | number |
| `angle` | number |
| `processing.crop_margin` | integer |
| `processing.mask_eps` | number |
| `processing.padding` | integer |
| `processing.resolution` | number |
| `processing.y_bounds` | array[2] of integer |
| `prompt_points.ab_blocks.center_ring` | integer |
| `prompt_points.ab_blocks.edge_ring` | number |
| `prompt_points.ab_blocks.edge_spacing` | integer |
| `prompt_points.ab_blocks.fine_spacing` | integer |
| `prompt_points.ab_blocks.inner_ring` | integer |
| `prompt_points.ab_blocks.middle_ring` | number |
| `prompt_points.ab_blocks.outer_ring` | integer |
| `prompt_points.ab_blocks.ultra_fine` | number |
| `prompt_points.ab_blocks.vertical_levels.center` | integer |
| `prompt_points.ab_blocks.vertical_levels.level_1` | number |
| `prompt_points.ab_blocks.vertical_levels.level_2` | number |
| `prompt_points.ab_blocks.vertical_levels.level_3` | number |
| `prompt_points.ab_blocks.vertical_levels.level_4` | number |
| `prompt_points.ab_blocks.vertical_levels.level_5` | number |
| `prompt_points.ab_blocks.vertical_levels.level_6` | number |
| `prompt_points.ab_blocks.vertical_levels.level_7` | number |
| `prompt_points.ab_blocks.vertical_levels.special_levels` | array[15] of number |
| `prompt_points.k_block.center_ring` | integer |
| `prompt_points.k_block.inner_ring` | number |
| `prompt_points.k_block.middle_ring` | integer |
| `prompt_points.k_block.outer_ring` | integer |
| `prompt_points.k_block.spacing_factors.k_block_spacing` | number |
| `prompt_points.k_block.spacing_factors.vertical_spacing` | array[5] of number |
| `prompt_points.template_mask.a_blocks.height` | number |
| `prompt_points.template_mask.a_blocks.width` | integer |
| `prompt_points.template_mask.b1_block.height_bottom_neg` | number |
| `prompt_points.template_mask.b1_block.height_bottom_pos` | number |
| `prompt_points.template_mask.b1_block.height_top` | number |
| `prompt_points.template_mask.b1_block.width` | integer |
| `prompt_points.template_mask.b2_block.height_bottom` | number |
| `prompt_points.template_mask.b2_block.height_top_neg` | number |
| `prompt_points.template_mask.b2_block.height_top_pos` | number |
| `prompt_points.template_mask.b2_block.width` | integer |
| `prompt_points.template_mask.k_block.height_neg` | number |
| `prompt_points.template_mask.k_block.height_pos` | number |
| `prompt_points.template_mask.k_block.width` | integer |
| `segment_order` | array[6] of string |
| `segment_per_ring` | integer |
| `segment_width` | integer |
| `use_original_label_distributions` | boolean |

### Before the code fence
At most a **short** prose note (optional); **no** CoT section headers. The fence must contain the full parameters object so it can be copied into `parameters_sam.json`.
