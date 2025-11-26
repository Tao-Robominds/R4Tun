# Algorithm 4 - Prompt Point Generation for 7-segment tunnels (4-1, 5-1)

import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import math
import json
from tqdm import tqdm
import pickle
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python sam_4+5.py <tunnel_id>")
    print("Example: python sam_4+5.py 4-1")
    sys.exit(1)

tunnel_id = sys.argv[1]
base_dir = f"data/{tunnel_id}/"

# Load parameters from JSON file with improved fallback
param_file = os.path.join(base_dir, "sam_parameters.json")
parameters = {}
param_load_success = False

if os.path.exists(param_file):
    try:
        with open(param_file, 'r') as f:
            parameters = json.load(f)
        param_load_success = True
        print(f"Loaded parameters from {param_file}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading parameter file {param_file}: {e}")
        print("Falling back to default parameters")
else:
    print(f"Parameter file {param_file} not found, using defaults")

# Load parameters with defaults (optimized for 7-segment tunnels)
segment_per_ring = parameters.get('segment_per_ring', 7)  # Default to 7 for 4+5 version
segment_order = parameters.get('segment_order', None)
segment_width = parameters.get('segment_width', 1200)  # Optimal for hardcoded compatibility
K_height = parameters.get('K_height', 1079.92)
AB_height = parameters.get('AB_height', 3239.77)
angle = parameters.get('angle', 7.52)
use_original_label_distributions = parameters.get('use_original_label_distributions', False)  # Enhanced distributions optimal
resolution = parameters.get('processing', {}).get('resolution', 0.005)
padding = parameters.get('processing', {}).get('padding', 300)
crop_margin = parameters.get('processing', {}).get('crop_margin', 50)

def generate_dynamic_segment_order(segment_per_ring):
    """Generate segment order dynamically for 7-segment tunnels"""
    if segment_per_ring == 7:
        # Fixed order for 7-segment tunnels: K, B1, A1, A2, A3, A4, B2
        return ['K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']
    else:
        # Fallback for other configurations
        block_labels = ['K', 'B1']
        num_a_labels = segment_per_ring - 3
        if num_a_labels > 0:
            block_labels += [f'A{i+1}' for i in range(num_a_labels)]
        block_labels += ['B2']
        return block_labels

# Verify required files exist
required_files = [
    "detected.csv",
    "pixel_to_point.pkl", 
    "enhanced.csv",
    "ring_count.txt",
    "depth_map.png"
]

for file in required_files:
    file_path = os.path.join(base_dir, file)
    if not os.path.exists(file_path):
        print(f"Error: Required file {file_path} not found!")
        sys.exit(1)

print(f"Processing tunnel: {tunnel_id}")

initial_prompt_points = pd.read_csv(os.path.join(base_dir, "detected.csv"))
pixel_to_point = pickle.load(open(os.path.join(base_dir, "pixel_to_point.pkl"), "rb"))
df_point_cloud = pd.read_csv(os.path.join(base_dir, "enhanced.csv"))
ring_count = int(open(os.path.join(base_dir, "ring_count.txt"), 'r').read())

# Initialize SAM model
sam_checkpoint = "mes/segment-anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load image
image = cv2.imread(os.path.join(base_dir, 'depth_map.png'))
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
    
    # Use loaded parameters instead of hardcoded values
    half_width = segment_width / 2  # 625 = 1250/2, but use dynamic segment_width
    
    if block == 'K':
        half_height = K_height / 2  # Use dynamic K_height
        # Keep the same asymmetric pattern but with dynamic values
        vertices_real = np.array([
            [x-half_width, y-half_height*0.57],  # -619.16 ≈ -1079.92*0.57
            [x-half_width, y+half_height*0.57],  # +619.16 ≈ +1079.92*0.57
            [x+half_width, y+half_height*0.43],  # +460.77 ≈ +1079.92*0.43
            [x+half_width, y-half_height*0.43]   # -460.77 ≈ -1079.92*0.43
        ])
    elif block == 'B1':
        half_height = AB_height / 2  # Use dynamic AB_height
        angle_offset = math.tan(math.radians(angle)) * half_width
        # Keep the same B1 pattern but with dynamic values
        vertices_real = np.array([
            [x-half_width, y-half_height*0.5],           # -1619.89 ≈ -3239.77*0.5
            [x-half_width, y+half_height*0.48],          # +1540.69 ≈ +3239.77*0.48
            [x+half_width, y+half_height*0.52+angle_offset], # +1699.08 with angle
            [x+half_width, y-half_height*0.5]            # -1619.89 ≈ -3239.77*0.5
        ])
    elif block == 'B2':
        half_height = AB_height / 2  # Use dynamic AB_height
        angle_offset = math.tan(math.radians(angle)) * half_width
        # Keep the same B2 pattern but with dynamic values
        vertices_real = np.array([
            [x-half_width, y-half_height*0.48],          # -1540.69 ≈ -3239.77*0.48
            [x-half_width, y+half_height*0.5],           # +1619.89 ≈ +3239.77*0.5
            [x+half_width, y+half_height*0.5],           # +1619.89 ≈ +3239.77*0.5
            [x+half_width, y-half_height*0.52-angle_offset] # -1699.08 with angle
        ])
    else:  # A blocks
        half_height = AB_height / 2  # Use dynamic AB_height
        # Keep the same rectangular pattern for A blocks
        vertices_real = np.array([
            [x-half_width, y-half_height*0.5],   # -1619.89 ≈ -3239.77*0.5
            [x-half_width, y+half_height*0.5],   # +1619.89 ≈ +3239.77*0.5
            [x+half_width, y+half_height*0.5],   # +1619.89 ≈ +3239.77*0.5
            [x+half_width, y-half_height*0.5]    # -1619.89 ≈ -3239.77*0.5
        ])
        
    vertices = vertices_real / (resolution*1000)
    fill_polygon(mask, vertices)
    return mask

def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution=0.005,
                           segment_width=1200, K_height=1079.92, AB_height=3239.77):
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    map_y = map_y * (resolution*1000)
    
    # Get label distribution preference from parameters loaded at top
    use_original_distributions = use_original_label_distributions
    
    # Calculate scaling factors based on loaded parameters vs original hardcoded values
    width_scale = segment_width / 1250  # Original was based on 1250 width (625*2)
    
    if block == 'K':
        height_scale = K_height / 1079.92  # Original K_height
        # Scale all K block coordinates by the scaling factors
        points_real = np.array([
            [x-700*width_scale,y-732.35*height_scale],[x-700*width_scale,y-505.96*height_scale],[x-700*width_scale,y-310.91*height_scale],[x-700*width_scale,y],[x-700*width_scale,y+310.91*height_scale],[x-700*width_scale,y+505.96*height_scale],[x-700*width_scale,y+732.35*height_scale],
            [x-500*width_scale,y-705.96*height_scale],[x-500*width_scale,y+705.96*height_scale],
            [x-348.16*width_scale,y-685.91*height_scale],[x-348.16*width_scale,y-310.91*height_scale],[x-325*width_scale,y],[x-348.16*width_scale,y+310.91*height_scale],[x-348.16*width_scale,y+685.91*height_scale],
            [x,y-639.96*height_scale],[x,y],[x,y+639.96*height_scale],
            [x+348.16*width_scale,y-594.01*height_scale],[x+348.16*width_scale,y-219.01*height_scale],[x+325*width_scale,y],[x+348.16*width_scale,y+219.01*height_scale],[x+348.16*width_scale,y+594.01*height_scale],
            [x+500*width_scale,y-573.96*height_scale],[x+500*width_scale,y+573.96*height_scale],
            [x+700*width_scale,y-547.57*height_scale],[x+700*width_scale,y-373.96*height_scale],[x+700*width_scale,y-219.01*height_scale],[x+700*width_scale,y],[x+700*width_scale,y+219.01*height_scale],[x+700*width_scale,y+373.96*height_scale],[x+700*width_scale,y+547.57*height_scale],
            [x-500*width_scale,y-505.96*height_scale],[x-511.06*width_scale,y-310.91*height_scale],[x-500*width_scale,y],[x-511.06*width_scale,y+310.91*height_scale],[x-500*width_scale,y+505.96*height_scale],
            [x-348.16*width_scale,y-485.91*height_scale],[x-348.16*width_scale,y+485.91*height_scale],
            [x,y-439.96*height_scale],[x,y+439.96*height_scale],
            [x+348.16*width_scale,y-394.01*height_scale],[x+348.16*width_scale,y+394.01*height_scale],
            [x+500*width_scale,y-373.96*height_scale],[x+511.06*width_scale,y-219.01*height_scale],[x+500*width_scale,y],[x+511.06*width_scale,y+219.01*height_scale],[x+500*width_scale,y+373.96*height_scale]
        ])
        # Use original or new label distribution based on configuration
        if use_original_distributions:
            labels = np.repeat([0, 1], [31, 16])  # Original distribution
        else:
            labels = np.repeat([0, 1], [20, 27])  # New distribution with more positives
    elif block == 'B1':
        height_scale = AB_height / 3239.77  # Original AB_height
        # Scale all B1 block coordinates by the scaling factors
        points_real = np.array([
            [x-700*width_scale,y-1719.89*height_scale],[x-511.06*width_scale,y-1719.89*height_scale],[x-348.16*width_scale,y-1719.89*height_scale],[x,y-1719.89*height_scale],[x+348.16*width_scale,y-1719.89*height_scale],[x+511.06*width_scale,y-1719.89*height_scale],[x+700*width_scale,y-1719.89*height_scale],
            [x-700*width_scale,y-1519.89*height_scale],[x+700*width_scale,y-1519.89*height_scale],
            [x-700*width_scale,y-1344.89*height_scale],[x-348.16*width_scale,y-1344.89*height_scale],[x+348.16*width_scale,y-1344.89*height_scale],[x+700*width_scale,y-1344.89*height_scale],
            [x-700*width_scale,y-1090.09*height_scale],[x-325*width_scale,y-1090.09*height_scale],[x+325*width_scale,y-1090.09*height_scale],[x+700*width_scale,y-1090.09*height_scale],
            [x-700*width_scale,y-817.57*height_scale],[x+700*width_scale,y-817.57*height_scale],
            [x-700*width_scale,y-545.05*height_scale],[x+700*width_scale,y-545.05*height_scale],
            [x-700*width_scale,y-272.52*height_scale],[x+700*width_scale,y-272.52*height_scale],
            [x-700*width_scale,y],[x-325*width_scale,y],[x,y],[x+325*width_scale,y],[x+700*width_scale,y],
            [x-700*width_scale,y+272.52*height_scale],[x+700*width_scale,y+272.52*height_scale],
            [x-700*width_scale,y+545.05*height_scale],[x+700*width_scale,y+545.05*height_scale],
            [x-700*width_scale,y+817.57*height_scale],[x+700*width_scale,y+817.57*height_scale],
            [x-700*width_scale,y+1090.09*height_scale],[x-325*width_scale,y+1090.09*height_scale],[x+325*width_scale,y+1090.09*height_scale],[x+700*width_scale,y+1090.09*height_scale],
            [x-700*width_scale,y+1298.93*height_scale],[x-350*width_scale,y+1298.93*height_scale],[x+350*width_scale,y+1390.84*height_scale],[x+700*width_scale,y+1390.84*height_scale],
            [x-700*width_scale,y+1427.43*height_scale],[x+700*width_scale,y+1612.28*height_scale],
            [x-700*width_scale,y+1627.49*height_scale],[x-511.06*width_scale,y+1652.43*height_scale],[x-350*width_scale,y+1673.69*height_scale],[x,y+1719.89*height_scale],[x+350*width_scale,y+1766.08*height_scale],[x+511.06*width_scale,y+1787.34*height_scale],[x+700*width_scale,y+1812.28*height_scale],
            [x-511.06*width_scale,y-1519.89*height_scale],[x-348.16*width_scale,y-1519.89*height_scale],[x,y-1519.89*height_scale],[x+348.16*width_scale,y-1519.89*height_scale],[x+511.06*width_scale,y-1519.89*height_scale],
            [x-511.06*width_scale,y-1344.89*height_scale],[x,y-1344.89*height_scale],[x+511.06*width_scale,y-1344.89*height_scale],
            [x-500*width_scale,y-1090.09*height_scale],[x,y-1090.09*height_scale],[x+500*width_scale,y-1090.09*height_scale],
            [x-500*width_scale,y-817.57*height_scale],[x-250*width_scale,y-817.57*height_scale],[x,y-817.57*height_scale],[x+250*width_scale,y-817.57*height_scale],[x+500*width_scale,y-817.57*height_scale],
            [x-500*width_scale,y-545.05*height_scale],[x-250*width_scale,y-545.05*height_scale],[x,y-545.05*height_scale],[x+250*width_scale,y-545.05*height_scale],[x+500*width_scale,y-545.05*height_scale],
            [x-500*width_scale,y-272.52*height_scale],[x-250*width_scale,y-272.52*height_scale],[x,y-272.52*height_scale],[x+250*width_scale,y-272.52*height_scale],[x+500*width_scale,y-272.52*height_scale],
            [x-500*width_scale,y],[x-162.5*width_scale,y],[x+162.5*width_scale,y],[x+500*width_scale,y],
            [x-500*width_scale,y+272.52*height_scale],[x-250*width_scale,y+272.52*height_scale],[x,y+272.52*height_scale],[x+250*width_scale,y+272.52*height_scale],[x+500*width_scale,y+272.52*height_scale],
            [x-500*width_scale,y+545.05*height_scale],[x-250*width_scale,y+545.05*height_scale],[x,y+545.05*height_scale],[x+250*width_scale,y+545.05*height_scale],[x+500*width_scale,y+545.05*height_scale],
            [x-500*width_scale,y+817.57*height_scale],[x-250*width_scale,y+817.57*height_scale],[x,y+817.57*height_scale],[x+250*width_scale,y+817.57*height_scale],[x+500*width_scale,y+817.57*height_scale],
            [x-500*width_scale,y+1090.09*height_scale],[x,y+1090.09*height_scale],[x+500*width_scale,y+1090.09*height_scale],
            [x-511.06*width_scale,y+1298.93*height_scale],[x,y+1345.01*height_scale],[x+511.06*width_scale,y+1390.84*height_scale],
            [x-511.06*width_scale,y+1452.43*height_scale],[x-350*width_scale,y+1473.69*height_scale],[x,y+1519.89*height_scale],[x+350*width_scale,y+1566.08*height_scale],[x+511.06*width_scale,y+1587.34*height_scale]      
        ])
        if use_original_distributions:
            labels = np.repeat([0,1],[51,56])  # Original distribution
        else:
            labels = np.repeat([0,1],[35,72])  # New distribution with more positives
    elif block == 'B2':
        height_scale = AB_height / 3239.77  # Original AB_height
        # Scale all B2 block coordinates by the scaling factors
        points_real = np.array([
            [x-700*width_scale,y-1627.49*height_scale],[x-511.06*width_scale,y-1652.43*height_scale],[x-350*width_scale,y-1673.69*height_scale],[x,y-1719.89*height_scale],[x+350*width_scale,y-1766.08*height_scale],[x+511.06*width_scale,y-1787.34*height_scale],[x+700*width_scale,y-1812.28*height_scale],
            [x-700*width_scale,y-1427.43*height_scale],[x+700*width_scale,y-1612.28*height_scale],
            [x-700*width_scale,y-1298.93*height_scale],[x-350*width_scale,y-1298.93*height_scale],[x+350*width_scale,y-1390.84*height_scale],[x+700*width_scale,y-1390.84*height_scale],            
            [x-700*width_scale,y-1090.09*height_scale],[x-325*width_scale,y-1090.09*height_scale],[x+325*width_scale,y-1090.09*height_scale],[x+700*width_scale,y-1090.09*height_scale],
            [x-700*width_scale,y-817.57*height_scale],[x+700*width_scale,y-817.57*height_scale],
            [x-700*width_scale,y-545.05*height_scale],[x+700*width_scale,y-545.05*height_scale],
            [x-700*width_scale,y-272.52*height_scale],[x+700*width_scale,y-272.52*height_scale],
            [x-700*width_scale,y],[x-325*width_scale,y],[x,y],[x+325*width_scale,y],[x+700*width_scale,y],
            [x-700*width_scale,y+272.52*height_scale],[x+700*width_scale,y+272.52*height_scale],
            [x-700*width_scale,y+545.05*height_scale],[x+700*width_scale,y+545.05*height_scale],
            [x-700*width_scale,y+817.57*height_scale],[x+700*width_scale,y+817.57*height_scale],
            [x-700*width_scale,y+1090.09*height_scale],[x-325*width_scale,y+1090.09*height_scale],[x+325*width_scale,y+1090.09*height_scale],[x+700*width_scale,y+1090.09*height_scale],
            [x-700*width_scale,y+1344.89*height_scale],[x-348.16*width_scale,y+1344.89*height_scale],[x+348.16*width_scale,y+1344.89*height_scale],[x+700*width_scale,y+1344.89*height_scale],
            [x-700*width_scale,y+1519.89*height_scale],[x+700*width_scale,y+1519.89*height_scale],
            [x-700*width_scale,y+1719.89*height_scale],[x-511.06*width_scale,y+1719.89*height_scale],[x-348.16*width_scale,y+1719.89*height_scale],[x,y+1719.89*height_scale],[x+348.16*width_scale,y+1719.89*height_scale],[x+511.06*width_scale,y+1719.89*height_scale],[x+700*width_scale,y+1719.89*height_scale],
            [x-511.06*width_scale,y-1452.43*height_scale],[x-350*width_scale,y-1473.69*height_scale],[x,y-1519.89*height_scale],[x+350*width_scale,y-1566.08*height_scale],[x+511.06*width_scale,y-1587.34*height_scale],     
            [x-511.06*width_scale,y-1298.93*height_scale],[x,y-1345.01*height_scale],[x+511.06*width_scale,y-1390.84*height_scale],
            [x-500*width_scale,y-1090.09*height_scale],[x,y-1090.09*height_scale],[x+500*width_scale,y-1090.09*height_scale],
            [x-500*width_scale,y-817.57*height_scale],[x-250*width_scale,y-817.57*height_scale],[x,y-817.57*height_scale],[x+250*width_scale,y-817.57*height_scale],[x+500*width_scale,y+817.57*height_scale],
            [x-500*width_scale,y-545.05*height_scale],[x-250*width_scale,y-545.05*height_scale],[x,y-545.05*height_scale],[x+250*width_scale,y-545.05*height_scale],[x+500*width_scale,y-545.05*height_scale],
            [x-500*width_scale,y-272.52*height_scale],[x-250*width_scale,y-272.52*height_scale],[x,y-272.52*height_scale],[x+250*width_scale,y-272.52*height_scale],[x+500*width_scale,y-272.52*height_scale],
            [x-500*width_scale,y],[x-162.5*width_scale,y],[x+162.5*width_scale,y],[x+500*width_scale,y],
            [x-500*width_scale,y+272.52*height_scale],[x-250*width_scale,y+272.52*height_scale],[x,y+272.52*height_scale],[x+250*width_scale,y+272.52*height_scale],[x+500*width_scale,y+272.52*height_scale],
            [x-500*width_scale,y+545.05*height_scale],[x-250*width_scale,y+545.05*height_scale],[x,y+545.05*height_scale],[x+250*width_scale,y+545.05*height_scale],[x+500*width_scale,y+545.05*height_scale],
            [x-500*width_scale,y+817.57*height_scale],[x-250*width_scale,y+817.57*height_scale],[x,y+817.57*height_scale],[x+250*width_scale,y+817.57*height_scale],[x+500*width_scale,y+817.57*height_scale],
            [x-500*width_scale,y+1090.09*height_scale],[x,y+1090.09*height_scale],[x+500*width_scale,y+1090.09*height_scale],
            [x-511.06*width_scale,y+1344.89*height_scale],[x,y+1344.89*height_scale],[x+511.06*width_scale,y+1344.89*height_scale],
            [x-511.06*width_scale,y+1519.89*height_scale],[x-348.16*width_scale,y+1519.89*height_scale],[x,y+1519.89*height_scale],[x+348.16*width_scale,y+1519.89*height_scale],[x+511.06*width_scale,y+1519.89*height_scale],
        ])
        if use_original_distributions:
            labels = np.repeat([0,1],[51,56])  # Original distribution
        else:
            labels = np.repeat([0,1],[35,72])  # New distribution with more positives
    else:  # A blocks
        height_scale = AB_height / 3239.77  # Original AB_height
        # Scale all A block coordinates by the scaling factors
        points_real = np.array([
            [x-700*width_scale,y-1719.89*height_scale],[x-511.06*width_scale,y-1719.89*height_scale],[x-348.16*width_scale,y-1719.89*height_scale],[x,y-1719.89*height_scale],[x+348.16*width_scale,y-1719.89*height_scale],[x+511.06*width_scale,y-1719.89*height_scale],[x+700*width_scale,y-1719.89*height_scale],
            [x-700*width_scale,y-1519.89*height_scale],[x+700*width_scale,y-1519.89*height_scale],
            [x-700*width_scale,y-1344.89*height_scale],[x-348.16*width_scale,y-1344.89*height_scale],[x+348.16*width_scale,y-1344.89*height_scale],[x+700*width_scale,y-1344.89*height_scale],
            [x-700*width_scale,y-1090.09*height_scale],[x-325*width_scale,y-1090.09*height_scale],[x+325*width_scale,y-1090.09*height_scale],[x+700*width_scale,y-1090.09*height_scale],
            [x-700*width_scale,y-817.57*height_scale],[x+700*width_scale,y-817.57*height_scale],
            [x-700*width_scale,y-545.05*height_scale],[x+700*width_scale,y-545.05*height_scale],
            [x-700*width_scale,y-272.52*height_scale],[x+700*width_scale,y-272.52*height_scale],
            [x-700*width_scale,y],[x-325*width_scale,y],[x,y],[x+325*width_scale,y],[x+700*width_scale,y],
            [x-700*width_scale,y+272.52*height_scale],[x+700*width_scale,y+272.52*height_scale],
            [x-700*width_scale,y+545.05*height_scale],[x+700*width_scale,y+545.05*height_scale],
            [x-700*width_scale,y+817.57*height_scale],[x+700*width_scale,y+817.57*height_scale],
            [x-700*width_scale,y+1090.09*height_scale],[x-325*width_scale,y+1090.09*height_scale],[x+325*width_scale,y+1090.09*height_scale],[x+700*width_scale,y+1090.09*height_scale],
            [x-700*width_scale,y+1344.89*height_scale],[x-348.16*width_scale,y+1344.89*height_scale],[x+348.16*width_scale,y+1344.89*height_scale],[x+700*width_scale,y+1344.89*height_scale],
            [x-700*width_scale,y+1519.89*height_scale],[x+700*width_scale,y+1519.89*height_scale],
            [x-700*width_scale,y+1719.89*height_scale],[x-511.06*width_scale,y+1719.89*height_scale],[x-348.16*width_scale,y+1719.89*height_scale],[x,y+1719.89*height_scale],[x+348.16*width_scale,y+1719.89*height_scale],[x+511.06*width_scale,y+1719.89*height_scale],[x+700*width_scale,y+1719.89*height_scale],
            [x-511.06*width_scale,y-1519.89*height_scale],[x-348.16*width_scale,y-1519.89*height_scale],[x,y-1519.89*height_scale],[x+348.16*width_scale,y-1519.89*height_scale],[x+511.06*width_scale,y-1519.89*height_scale],
            [x-511.06*width_scale,y-1344.89*height_scale],[x,y-1344.89*height_scale],[x+511.06*width_scale,y-1344.89*height_scale],
            [x-500*width_scale,y-1090.09*height_scale],[x,y-1090.09*height_scale],[x+500*width_scale,y-1090.09*height_scale],
            [x-500*width_scale,y-817.57*height_scale],[x-250*width_scale,y-817.57*height_scale],[x,y-817.57*height_scale],[x+250*width_scale,y-817.57*height_scale],[x+500*width_scale,y-817.57*height_scale],
            [x-500*width_scale,y-545.05*height_scale],[x-250*width_scale,y-545.05*height_scale],[x,y-545.05*height_scale],[x+250*width_scale,y-545.05*height_scale],[x+500*width_scale,y-545.05*height_scale],
            [x-500*width_scale,y-272.52*height_scale],[x-250*width_scale,y-272.52*height_scale],[x,y-272.52*height_scale],[x+250*width_scale,y-272.52*height_scale],[x+500*width_scale,y-272.52*height_scale],
            [x-500*width_scale,y],[x-162.5*width_scale,y],[x+162.5*width_scale,y],[x+500*width_scale,y],
            [x-500*width_scale,y+272.52*height_scale],[x-250*width_scale,y+272.52*height_scale],[x,y+272.52*height_scale],[x+250*width_scale,y+272.52*height_scale],[x+500*width_scale,y+272.52*height_scale],
            [x-500*width_scale,y+545.05*height_scale],[x-250*width_scale,y+545.05*height_scale],[x,y+545.05*height_scale],[x+250*width_scale,y+545.05*height_scale],[x+500*width_scale,y+545.05*height_scale],
            [x-500*width_scale,y+817.57*height_scale],[x-250*width_scale,y+817.57*height_scale],[x,y+817.57*height_scale],[x+250*width_scale,y+817.57*height_scale],[x+500*width_scale,y+817.57*height_scale],
            [x-500*width_scale,y+1090.09*height_scale],[x,y+1090.09*height_scale],[x+500*width_scale,y+1090.09*height_scale],
            [x-511.06*width_scale,y+1344.89*height_scale],[x,y+1344.89*height_scale],[x+511.06*width_scale,y+1344.89*height_scale],
            [x-511.06*width_scale,y+1519.89*height_scale],[x-348.16*width_scale,y+1519.89*height_scale],[x,y+1519.89*height_scale],[x+348.16*width_scale,y+1519.89*height_scale],[x+511.06*width_scale,y+1519.89*height_scale],
        ])
        if use_original_distributions:
            labels = np.repeat([0,1],[51,56])  # Original distribution
        else:
            labels = np.repeat([0,1],[35,72])  # New distribution with more positives

    # Use the dynamic parameters in the filtering logic as well
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

    # Ensure the crop has positive dimensions
    if x1 >= x2 or y1 >= y2:
        print(f"Warning: Invalid crop dimensions - x1={x1}, x2={x2}, y1={y1}, y2={y2}")
        print(f"Center: ({cx}, {cy}), Crop size: {crop_width}x{crop_height}, Image size: {img_width}x{img_height}")
        
        # Adjust crop boundaries to ensure minimum size
        min_size = 10  # Minimum crop size
        if x1 >= x2:
            center_x = (x1 + x2) // 2
            x1 = max(center_x - min_size // 2, 0)
            x2 = min(center_x + min_size // 2, img_width)
            if x1 >= x2:  # Still invalid, use minimal valid crop
                x1 = max(0, min(cx - min_size // 2, img_width - min_size))
                x2 = min(x1 + min_size, img_width)
        
        if y1 >= y2:
            center_y = (y1 + y2) // 2
            y1 = max(center_y - min_size // 2, 0)
            y2 = min(center_y + min_size // 2, img_height)
            if y1 >= y2:  # Still invalid, use minimal valid crop
                y1 = max(0, min(cy - min_size // 2, img_height - min_size))
                y2 = min(y1 + min_size, img_height)

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    
    # Final validation - ensure we have a valid crop
    if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
        raise ValueError(f"Failed to create valid crop: final shape is {cropped_image.shape}")
    
    prompt_centre_x = cx - x1
    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x, prompt_centre_y)
    
    cropped_template_mask = generate_template_mask(cropped_image.shape[0], cropped_image.shape[1], prompt_centre, block, resolution)
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    return cropped_image, template_mask_logits, prompt_centre

def compute_logits_from_mask(mask, eps=1e-3):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    # Validate mask dimensions
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError(f"Invalid mask dimensions: {mask.shape}")

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
    """
    Improved block label computation with both JSON and dynamic fallback support
    """
    global segment_order
    
    # If we have a valid segment_order from JSON, use it
    if segment_order is not None and len(segment_order) == segment_per_ring:
        return segment_order
    
    # Otherwise, generate dynamically like the original version
    print(f"Generating dynamic block labels for segment_per_ring={segment_per_ring}")
    return generate_dynamic_segment_order(segment_per_ring)

def sam_prediction(cropped_image, points, labels, template_mask_logit):
    predictor.set_image(cropped_image)
    mask, score, logit = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=template_mask_logit,
            multimask_output=False,
    )
    return mask, score, logit[0]

def process_row(df_row, image, resolution=0.005, segment_per_ring=7, segment_width=1200, 
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

def sam_segment(df, image, resolution=0.005, segment_per_ring=7):
    all_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_row(row, image, resolution, segment_per_ring, 
                           segment_width, K_height, angle, AB_height)
        all_results.append(result)
    return all_results

# Process the data
results = sam_segment(initial_prompt_points, image, resolution=resolution, segment_per_ring=segment_per_ring)

# Project back to point cloud - DYNAMIC block_to_label mapping for 7-segment tunnels
block_to_label = {block: i+1 for i, block in enumerate(compute_block_label(segment_per_ring))}
print(f"Using block_to_label mapping: {block_to_label}")

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

# Fix ring labels
fix_ring = np.where((ring_image >= 1) & (ring_image <= (ring_count-1)), ring_count - ring_image, ring_image)

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)

    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    # Get image dimensions for bounds checking
    img_height, img_width = segmented_map.shape

    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    valid_update_mask = (pred[point_indices[valid_point_mask]] == 7)
    
    # Additional bounds checking for pixel coordinates
    y_valid = y[valid_point_mask][valid_update_mask]
    x_valid = x[valid_point_mask][valid_update_mask]
    
    # Check if pixel coordinates are within image bounds
    bounds_mask = (y_valid >= 0) & (y_valid < img_height) & (x_valid >= 0) & (x_valid < img_width)
    
    # Only update pixels that are within bounds
    final_point_indices = point_indices[valid_point_mask][valid_update_mask][bounds_mask]
    final_y = y_valid[bounds_mask]
    final_x = x_valid[bounds_mask]

    pred[final_point_indices] = segmented_map[final_y, final_x]
    pred_ring[final_point_indices] = instance_map[final_y, final_x]

    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring

    return df_copy

# Save final results
updated_df = project_back_to_point_cloud(result_image, fix_ring, pixel_to_point, df_point_cloud)

# save to base_dir
updated_df.to_csv(os.path.join(base_dir, 'final.csv'), index=False)

df_pred = pd.DataFrame()
df_pred['gt_labels'] = updated_df['segment']
df_pred['gt_rings'] = updated_df['ring']
df_pred['pred_labels'] = updated_df['pred']
df_pred['pred_rings'] = updated_df['pred_ring']
# save to base_dir
df_pred.to_csv(os.path.join(base_dir, 'only_label.csv'), index=False)

print(f"SAM segmentation completed for tunnel {tunnel_id}")
print(f"Results saved to {base_dir}final.csv and {base_dir}only_label.csv")

