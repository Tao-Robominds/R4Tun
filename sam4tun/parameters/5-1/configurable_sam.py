# Algorithm 4-2 - SAM Segmentation (Configurable Version)

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
import json

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python configurable_sam.py <tunnel_id>")
    print("Example: python configurable_sam.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

def load_parameters(tunnel_id):
    """Load parameters from configurable directory where analyst saves parameters"""
    
    # Determine script directory to handle both project root and configurable execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.join(script_dir, tunnel_id, 'parameters_sam.json')
    
    if os.path.exists(param_file):
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"✅ Loaded parameters from configurable/{tunnel_id}/parameters_sam.json")
            return params
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            sys.exit(1)
    else:
        print(f"❌ Error: Parameter file not found at configurable/{tunnel_id}/parameters_sam.json")
        print("Please run the analyst to generate parameters first.")
        sys.exit(1)

# Load configuration parameters
config = load_parameters(tunnel_id)
segment_per_ring = config["segment_per_ring"]
segment_width = config["segment_width"]
K_height = config["K_height"]
AB_height = config["AB_height"]
angle = config["angle"]

# Handle both new parameterized format and old format
if "processing" in config:
    # New parameterized format
    processing = config["processing"]
    resolution = processing["resolution"]
    padding = processing["padding"]
    crop_margin = processing["crop_margin"]
    mask_eps = processing["mask_eps"]
    y_bounds = processing["y_bounds"]
    
    # Extract prompt point parameters
    if "prompt_points" in config:
        prompt_params = config["prompt_points"]
        k_params = prompt_params["k_block"]
        ab_params = prompt_params["ab_blocks"]
        template_params = prompt_params["template_mask"]
        print("✅ Using new fully parameterized format")
    else:
        # Use default hardcoded values for prompt points
        k_params = None
        ab_params = None
        template_params = None
        print("✅ Using new processing format with default prompt point values")
else:
    # Old format - extract directly and use defaults
    resolution = config["resolution"]
    padding = 150  # Default padding used in old format
    crop_margin = 50  # Default crop margin
    mask_eps = 1e-3  # Default mask epsilon
    y_bounds = [4200, 13100]  # Default y bounds
    k_params = None
    ab_params = None
    template_params = None
    print("✅ Using legacy format with default processing parameters")

# SAM model configuration
if "sam_checkpoint" in config:
    sam_checkpoint = config["sam_checkpoint"]
    model_type = config["model_type"]
    device = config["device"]
else:
    # Use default paths
    sam_checkpoint = "sam4tun/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

print(f"Processing tunnel: {tunnel_id}")
print(f"Using parameters: segment_per_ring={segment_per_ring}, segment_width={segment_width}, resolution={resolution}")
print(f"Processing config: padding={padding}, crop_margin={crop_margin}, y_bounds={y_bounds}")

base_dir = f"data/{tunnel_id}/"
initial_prompt_points = pd.read_csv(os.path.join(base_dir, "detected.csv"))
pixel_to_point = pickle.load(open(os.path.join(base_dir, "pixel_to_point.pkl"), "rb"))
df_point_cloud = pd.read_csv(os.path.join(base_dir, "enhanced.csv"))
ring_count = int(open(f'data/{tunnel_id}/ring_count.txt', 'r').read())

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

def generate_template_mask(height, width, prompt_centre, block):
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    
    if template_params is not None:
        # Use parameterized values
        if block == 'K':
            k_mask = template_params["k_block"]
            width_val = k_mask["width"]
            height_pos = k_mask["height_pos"]
            height_neg = k_mask["height_neg"]
            vertices_real = np.array([[x-width_val,y-height_pos],[x-width_val,y+height_pos],[x+width_val,y+height_neg],[x+width_val,y-height_neg]])
        elif block == 'B1':
            b1_mask = template_params["b1_block"]
            width_val = b1_mask["width"]
            height_top = b1_mask["height_top"]
            height_bottom_pos = b1_mask["height_bottom_pos"]
            height_bottom_neg = b1_mask["height_bottom_neg"]
            vertices_real = np.array([[x-width_val,y-height_top],[x-width_val,y+height_bottom_pos],[x+width_val,y+height_bottom_neg],[x+width_val,y-height_top]])
        elif block == 'B2':
            b2_mask = template_params["b2_block"]
            width_val = b2_mask["width"]
            height_top_pos = b2_mask["height_top_pos"]
            height_top_neg = b2_mask["height_top_neg"]
            height_bottom = b2_mask["height_bottom"]
            vertices_real = np.array([[x-width_val,y-height_top_pos],[x-width_val,y+height_bottom],[x+width_val,y+height_bottom],[x+width_val,y-height_top_neg]])
        else:
            a_mask = template_params["a_blocks"]
            width_val = a_mask["width"]
            height_val = a_mask["height"]
            vertices_real = np.array([[x-width_val,y-height_val],[x-width_val,y+height_val],[x+width_val,y+height_val],[x+width_val,y-height_val]])
    else:
        # Use hardcoded default values (legacy)
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

def generate_prompt_points(prompt_centre, initial_x, map_y, block):
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution*1000)
    y = prompt_centre_y * (resolution*1000)
    map_y = map_y * (resolution*1000)
    
    if block == 'K':
        if k_params is not None:
            # Use parameterized K block values
            outer_ring = k_params["outer_ring"]
            middle_ring = k_params["middle_ring"] 
            inner_ring = k_params["inner_ring"]
            center_ring = k_params["center_ring"]
            spacing_factors = k_params["spacing_factors"]
            k_spacing = spacing_factors["k_block_spacing"]
            v_spacing = spacing_factors["vertical_spacing"]
            
            points_real = np.array([
                [x-outer_ring,y-v_spacing[0]],[x-outer_ring,y-v_spacing[1]],[x-outer_ring,y-k_spacing],[x-outer_ring,y],[x-outer_ring,y+k_spacing],[x-outer_ring,y+v_spacing[1]],[x-outer_ring,y+v_spacing[0]],
                [x-middle_ring,y-v_spacing[1]],[x-middle_ring,y+v_spacing[1]],
                [x-inner_ring,y-v_spacing[1]],[x-inner_ring,y-k_spacing],[x-center_ring,y],[x-inner_ring,y+k_spacing],[x-inner_ring,y+v_spacing[1]],
                [x,y-v_spacing[1]],[x,y],[x,y+v_spacing[1]],
                [x+inner_ring,y-v_spacing[1]],[x+inner_ring,y-v_spacing[2]],[x+center_ring,y],[x+inner_ring,y+v_spacing[2]],[x+inner_ring,y+v_spacing[1]],
                [x+middle_ring,y-v_spacing[1]],[x+middle_ring,y+v_spacing[1]],
                [x+outer_ring,y-v_spacing[1]],[x+outer_ring,y-v_spacing[4]],[x+outer_ring,y-v_spacing[3]],[x+outer_ring,y],[x+outer_ring,y+v_spacing[3]],[x+outer_ring,y+v_spacing[4]],[x+outer_ring,y+v_spacing[1]],
                [x-middle_ring,y-k_spacing],[x-middle_ring-11.06,y-k_spacing],[x-middle_ring,y],[x-middle_ring-11.06,y+k_spacing],[x-middle_ring,y+k_spacing],
                [x-inner_ring,y-k_spacing],[x-inner_ring,y+k_spacing],
                [x,y-k_spacing],[x,y+k_spacing],
                [x+inner_ring,y-k_spacing],[x+inner_ring,y+k_spacing],
                [x+middle_ring,y-k_spacing],[x+middle_ring+11.06,y-v_spacing[3]],[x+middle_ring,y],[x+middle_ring+11.06,y+v_spacing[3]],[x+middle_ring,y+k_spacing]
            ])
        else:
            # Use hardcoded default values (legacy)
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
        if ab_params is not None:
            # Use parameterized AB block values
            outer_ring = ab_params["outer_ring"]
            middle_ring = ab_params["middle_ring"]
            inner_ring = ab_params["inner_ring"]
            center_ring = ab_params["center_ring"]
            fine_spacing = ab_params["fine_spacing"]
            ultra_fine = ab_params["ultra_fine"]
            edge_ring = ab_params["edge_ring"]
            edge_spacing = ab_params["edge_spacing"]
            levels = ab_params["vertical_levels"]
            special = levels["special_levels"]
            
            points_real = np.array([
                # Use parameterized values for all coordinates
                [x-outer_ring,y-levels["level_1"]],[x-middle_ring,y-levels["level_1"]],[x-edge_ring,y-levels["level_1"]],[x,y-levels["level_1"]],[x+edge_ring,y-levels["level_1"]],[x+middle_ring,y-levels["level_1"]],[x+outer_ring,y-levels["level_1"]],
                [x-outer_ring,y-levels["level_2"]],[x+outer_ring,y-levels["level_2"]],
                [x-outer_ring,y-levels["level_3"]],[x-edge_ring,y-levels["level_3"]],[x+edge_ring,y-levels["level_3"]],[x+outer_ring,y-levels["level_3"]],
                [x-outer_ring,y-levels["level_4"]],[x-center_ring,y-levels["level_4"]],[x+center_ring,y-levels["level_4"]],[x+outer_ring,y-levels["level_4"]],
                [x-outer_ring,y-levels["level_5"]],[x+outer_ring,y-levels["level_5"]],
                [x-outer_ring,y-levels["level_6"]],[x+outer_ring,y-levels["level_6"]],
                [x-outer_ring,y-levels["level_7"]],[x+outer_ring,y-levels["level_7"]],
                [x-outer_ring,y],[x-center_ring,y],[x,y],[x+center_ring,y],[x+outer_ring,y],
                [x-outer_ring,y+levels["level_7"]],[x+outer_ring,y+levels["level_7"]],
                [x-outer_ring,y+levels["level_6"]],[x+outer_ring,y+levels["level_6"]],
                [x-outer_ring,y+levels["level_5"]],[x+outer_ring,y+levels["level_5"]],
                [x-outer_ring,y+levels["level_4"]],[x-center_ring,y+levels["level_4"]],[x+center_ring,y+levels["level_4"]],[x+outer_ring,y+levels["level_4"]],
                [x-outer_ring,y+special[0]],[x-edge_spacing,y+special[0]],[x+edge_spacing,y+special[1]],[x+outer_ring,y+special[1]],
                [x-outer_ring,y+special[2]],[x+outer_ring,y+special[3]],
                [x-outer_ring,y+special[4]],[x-middle_ring,y+special[5]],[x-edge_spacing,y+special[6]],[x,y+levels["level_1"]],[x+edge_spacing,y+special[7]],[x+middle_ring,y+special[8]],[x+outer_ring,y+special[9]],
                [x-middle_ring,y-levels["level_2"]],[x-edge_ring,y-levels["level_2"]],[x,y-levels["level_2"]],[x+edge_ring,y-levels["level_2"]],[x+middle_ring,y-levels["level_2"]],
                [x-middle_ring,y-levels["level_3"]],[x,y-levels["level_3"]],[x+middle_ring,y-levels["level_3"]],
                [x-inner_ring,y-levels["level_4"]],[x,y-levels["level_4"]],[x+inner_ring,y-levels["level_4"]],
                [x-inner_ring,y-levels["level_5"]],[x-fine_spacing,y-levels["level_5"]],[x,y-levels["level_5"]],[x+fine_spacing,y-levels["level_5"]],[x+inner_ring,y-levels["level_5"]],
                [x-inner_ring,y-levels["level_6"]],[x-fine_spacing,y-levels["level_6"]],[x,y-levels["level_6"]],[x+fine_spacing,y-levels["level_6"]],[x+inner_ring,y-levels["level_6"]],
                [x-inner_ring,y-levels["level_7"]],[x-fine_spacing,y-levels["level_7"]],[x,y-levels["level_7"]],[x+fine_spacing,y-levels["level_7"]],[x+inner_ring,y-levels["level_7"]],
                [x-inner_ring,y],[x-ultra_fine,y],[x+ultra_fine,y],[x+inner_ring,y],
                [x-inner_ring,y+levels["level_7"]],[x-fine_spacing,y+levels["level_7"]],[x,y+levels["level_7"]],[x+fine_spacing,y+levels["level_7"]],[x+inner_ring,y+levels["level_7"]],
                [x-inner_ring,y+levels["level_6"]],[x-fine_spacing,y+levels["level_6"]],[x,y+levels["level_6"]],[x+fine_spacing,y+levels["level_6"]],[x+inner_ring,y+levels["level_6"]],
                [x-inner_ring,y+levels["level_5"]],[x-fine_spacing,y+levels["level_5"]],[x,y+levels["level_5"]],[x+fine_spacing,y+levels["level_5"]],[x+inner_ring,y+levels["level_5"]],
                [x-inner_ring,y+levels["level_4"]],[x,y+levels["level_4"]],[x+inner_ring,y+levels["level_4"]],
                [x-middle_ring,y+special[0]],[x,y+special[10]],[x+middle_ring,y+special[1]],
                [x-middle_ring,y+special[11]],[x-edge_spacing,y+special[12]],[x,y+levels["level_2"]],[x+edge_spacing,y+special[13]],[x+middle_ring,y+special[14]]      
            ])
        else:
            # Use hardcoded default values (legacy)
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
        if ab_params is not None:
            # Use parameterized AB block values (B2 pattern)
            outer_ring = ab_params["outer_ring"]
            middle_ring = ab_params["middle_ring"]
            inner_ring = ab_params["inner_ring"]
            center_ring = ab_params["center_ring"]
            fine_spacing = ab_params["fine_spacing"]
            ultra_fine = ab_params["ultra_fine"]
            edge_ring = ab_params["edge_ring"]
            edge_spacing = ab_params["edge_spacing"]
            levels = ab_params["vertical_levels"]
            special = levels["special_levels"]
            
            points_real = np.array([
                # B2 pattern using parameterized values
                [x-outer_ring,y-special[4]],[x-middle_ring,y-special[5]],[x-edge_spacing,y-special[6]],[x,y-levels["level_1"]],[x+edge_spacing,y-special[7]],[x+middle_ring,y-special[8]],[x+outer_ring,y-special[9]],
                [x-outer_ring,y-special[2]],[x+outer_ring,y-special[3]],
                [x-outer_ring,y-special[0]],[x-edge_spacing,y-special[0]],[x+edge_spacing,y-special[1]],[x+outer_ring,y-special[1]],            
                [x-outer_ring,y-levels["level_4"]],[x-center_ring,y-levels["level_4"]],[x+center_ring,y-levels["level_4"]],[x+outer_ring,y-levels["level_4"]],
                [x-outer_ring,y-levels["level_5"]],[x+outer_ring,y-levels["level_5"]],
                [x-outer_ring,y-levels["level_6"]],[x+outer_ring,y-levels["level_6"]],
                [x-outer_ring,y-levels["level_7"]],[x+outer_ring,y-levels["level_7"]],
                [x-outer_ring,y],[x-center_ring,y],[x,y],[x+center_ring,y],[x+outer_ring,y],
                [x-outer_ring,y+levels["level_7"]],[x+outer_ring,y+levels["level_7"]],
                [x-outer_ring,y+levels["level_6"]],[x+outer_ring,y+levels["level_6"]],
                [x-outer_ring,y+levels["level_5"]],[x+outer_ring,y+levels["level_5"]],
                [x-outer_ring,y+levels["level_4"]],[x-center_ring,y+levels["level_4"]],[x+center_ring,y+levels["level_4"]],[x+outer_ring,y+levels["level_4"]],
                [x-outer_ring,y+levels["level_3"]],[x-edge_ring,y+levels["level_3"]],[x+edge_ring,y+levels["level_3"]],[x+outer_ring,y+levels["level_3"]],
                [x-outer_ring,y+levels["level_2"]],[x+outer_ring,y+levels["level_2"]],
                [x-outer_ring,y+levels["level_1"]],[x-middle_ring,y+levels["level_1"]],[x-edge_ring,y+levels["level_1"]],[x,y+levels["level_1"]],[x+edge_ring,y+levels["level_1"]],[x+middle_ring,y+levels["level_1"]],[x+outer_ring,y+levels["level_1"]],
                [x-middle_ring,y-special[11]],[x-edge_spacing,y-special[12]],[x,y-levels["level_2"]],[x+edge_spacing,y-special[13]],[x+middle_ring,y-special[14]],     
                [x-middle_ring,y-special[0]],[x,y-special[10]],[x+middle_ring,y-special[1]],
                [x-inner_ring,y-levels["level_4"]],[x,y-levels["level_4"]],[x+inner_ring,y-levels["level_4"]],
                [x-inner_ring,y-levels["level_5"]],[x-fine_spacing,y-levels["level_5"]],[x,y-levels["level_5"]],[x+fine_spacing,y-levels["level_5"]],[x+inner_ring,y+levels["level_5"]],
                [x-inner_ring,y-levels["level_6"]],[x-fine_spacing,y-levels["level_6"]],[x,y-levels["level_6"]],[x+fine_spacing,y-levels["level_6"]],[x+inner_ring,y-levels["level_6"]],
                [x-inner_ring,y-levels["level_7"]],[x-fine_spacing,y-levels["level_7"]],[x,y-levels["level_7"]],[x+fine_spacing,y-levels["level_7"]],[x+inner_ring,y-levels["level_7"]],
                [x-inner_ring,y],[x-ultra_fine,y],[x+ultra_fine,y],[x+inner_ring,y],
                [x-inner_ring,y+levels["level_7"]],[x-fine_spacing,y+levels["level_7"]],[x,y+levels["level_7"]],[x+fine_spacing,y+levels["level_7"]],[x+inner_ring,y+levels["level_7"]],
                [x-inner_ring,y+levels["level_6"]],[x-fine_spacing,y+levels["level_6"]],[x,y+levels["level_6"]],[x+fine_spacing,y+levels["level_6"]],[x+inner_ring,y+levels["level_6"]],
                [x-inner_ring,y+levels["level_5"]],[x-fine_spacing,y+levels["level_5"]],[x,y+levels["level_5"]],[x+fine_spacing,y+levels["level_5"]],[x+inner_ring,y+levels["level_5"]],
                [x-inner_ring,y+levels["level_4"]],[x,y+levels["level_4"]],[x+inner_ring,y+levels["level_4"]],
                [x-middle_ring,y+levels["level_3"]],[x,y+levels["level_3"]],[x+middle_ring,y+levels["level_3"]],
                [x-middle_ring,y+levels["level_2"]],[x-edge_ring,y+levels["level_2"]],[x,y+levels["level_2"]],[x+edge_ring,y+levels["level_2"]],[x+middle_ring,y+levels["level_2"]],
            ])
        else:
            # Use hardcoded default values (legacy B2)
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
        # A blocks (A1, A2, A3) - same pattern as B1
        if ab_params is not None:
            # Use parameterized AB block values (A block pattern - same as B1)
            outer_ring = ab_params["outer_ring"]
            middle_ring = ab_params["middle_ring"]
            inner_ring = ab_params["inner_ring"]
            center_ring = ab_params["center_ring"]
            fine_spacing = ab_params["fine_spacing"]
            ultra_fine = ab_params["ultra_fine"]
            edge_ring = ab_params["edge_ring"]
            edge_spacing = ab_params["edge_spacing"]
            levels = ab_params["vertical_levels"]
            special = levels["special_levels"]
            
            points_real = np.array([
                # A blocks use same pattern as B1
                [x-outer_ring,y-levels["level_1"]],[x-middle_ring,y-levels["level_1"]],[x-edge_ring,y-levels["level_1"]],[x,y-levels["level_1"]],[x+edge_ring,y-levels["level_1"]],[x+middle_ring,y-levels["level_1"]],[x+outer_ring,y-levels["level_1"]],
                [x-outer_ring,y-levels["level_2"]],[x+outer_ring,y-levels["level_2"]],
                [x-outer_ring,y-levels["level_3"]],[x-edge_ring,y-levels["level_3"]],[x+edge_ring,y-levels["level_3"]],[x+outer_ring,y-levels["level_3"]],
                [x-outer_ring,y-levels["level_4"]],[x-center_ring,y-levels["level_4"]],[x+center_ring,y-levels["level_4"]],[x+outer_ring,y-levels["level_4"]],
                [x-outer_ring,y-levels["level_5"]],[x+outer_ring,y-levels["level_5"]],
                [x-outer_ring,y-levels["level_6"]],[x+outer_ring,y-levels["level_6"]],
                [x-outer_ring,y-levels["level_7"]],[x+outer_ring,y-levels["level_7"]],
                [x-outer_ring,y],[x-center_ring,y],[x,y],[x+center_ring,y],[x+outer_ring,y],
                [x-outer_ring,y+levels["level_7"]],[x+outer_ring,y+levels["level_7"]],
                [x-outer_ring,y+levels["level_6"]],[x+outer_ring,y+levels["level_6"]],
                [x-outer_ring,y+levels["level_5"]],[x+outer_ring,y+levels["level_5"]],
                [x-outer_ring,y+levels["level_4"]],[x-center_ring,y+levels["level_4"]],[x+center_ring,y+levels["level_4"]],[x+outer_ring,y+levels["level_4"]],
                [x-outer_ring,y+levels["level_3"]],[x-edge_ring,y+levels["level_3"]],[x+edge_ring,y+levels["level_3"]],[x+outer_ring,y+levels["level_3"]],
                [x-outer_ring,y+levels["level_2"]],[x+outer_ring,y+levels["level_2"]],
                [x-outer_ring,y+levels["level_1"]],[x-middle_ring,y+levels["level_1"]],[x-edge_ring,y+levels["level_1"]],[x,y+levels["level_1"]],[x+edge_ring,y+levels["level_1"]],[x+middle_ring,y+levels["level_1"]],[x+outer_ring,y+levels["level_1"]],
                [x-middle_ring,y-levels["level_2"]],[x-edge_ring,y-levels["level_2"]],[x,y-levels["level_2"]],[x+edge_ring,y-levels["level_2"]],[x+middle_ring,y-levels["level_2"]],
                [x-middle_ring,y-levels["level_3"]],[x,y-levels["level_3"]],[x+middle_ring,y-levels["level_3"]],
                [x-inner_ring,y-levels["level_4"]],[x,y-levels["level_4"]],[x+inner_ring,y-levels["level_4"]],
                [x-inner_ring,y-levels["level_5"]],[x-fine_spacing,y-levels["level_5"]],[x,y-levels["level_5"]],[x+fine_spacing,y-levels["level_5"]],[x+inner_ring,y-levels["level_5"]],
                [x-inner_ring,y-levels["level_6"]],[x-fine_spacing,y-levels["level_6"]],[x,y-levels["level_6"]],[x+fine_spacing,y-levels["level_6"]],[x+inner_ring,y-levels["level_6"]],
                [x-inner_ring,y-levels["level_7"]],[x-fine_spacing,y-levels["level_7"]],[x,y-levels["level_7"]],[x+fine_spacing,y-levels["level_7"]],[x+inner_ring,y-levels["level_7"]],
                [x-inner_ring,y],[x-ultra_fine,y],[x+ultra_fine,y],[x+inner_ring,y],
                [x-inner_ring,y+levels["level_7"]],[x-fine_spacing,y+levels["level_7"]],[x,y+levels["level_7"]],[x+fine_spacing,y+levels["level_7"]],[x+inner_ring,y+levels["level_7"]],
                [x-inner_ring,y+levels["level_6"]],[x-fine_spacing,y+levels["level_6"]],[x,y+levels["level_6"]],[x+fine_spacing,y+levels["level_6"]],[x+inner_ring,y+levels["level_6"]],
                [x-inner_ring,y+levels["level_5"]],[x-fine_spacing,y+levels["level_5"]],[x,y+levels["level_5"]],[x+fine_spacing,y+levels["level_5"]],[x+inner_ring,y+levels["level_5"]],
                [x-inner_ring,y+levels["level_4"]],[x,y+levels["level_4"]],[x+inner_ring,y+levels["level_4"]],
                [x-middle_ring,y+levels["level_3"]],[x,y+levels["level_3"]],[x+middle_ring,y+levels["level_3"]],
                [x-middle_ring,y+levels["level_2"]],[x-edge_ring,y+levels["level_2"]],[x,y+levels["level_2"]],[x+edge_ring,y+levels["level_2"]],[x+middle_ring,y+levels["level_2"]],
            ])
        else:
            # Use hardcoded default values (legacy A blocks)
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
            y_cond = points_real[i, 1] + map_y < y_bounds[0] or points_real[i, 1] + map_y > y_bounds[1]
            x_cond = abs(points_real[i, 0] - x) <= segment_width * 0.5
            y_limit = K_height if block == 'K' else AB_height
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
            
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    
    points = points_real / (resolution*1000)

    within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (segment_width*0.5+padding)/(resolution*1000)) <= image.shape[1])
    points = points[within_bounds]
    labels = labels[within_bounds]
        
    return points, labels

def convert_to_pixel_coords(real_dist):
    return int(real_dist / (resolution*1000))

def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block):
    img_height, img_width, _ = image.shape
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)
    y2 = min(cy + crop_height // 2, img_height)

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    prompt_centre_x = cx - x1
    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x,prompt_centre_y)
    
    cropped_template_mask = generate_template_mask(cropped_image.shape[0],cropped_image.shape[1],prompt_centre,block)
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    return cropped_image, template_mask_logits, prompt_centre

def compute_logits_from_mask(mask):
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - mask_eps
    logits[mask == 0] = mask_eps
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

def process_row(df_row, image):
    initial_x, initial_y = df_row['X'], df_row['Y']
    block_labels = compute_block_label(segment_per_ring)

    delta_x = convert_to_pixel_coords(0.5*segment_width + padding)
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
                delta_y = convert_to_pixel_coords(0.5*K_height + math.tan(math.radians(angle))*700+100 + crop_margin)
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + crop_margin)
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height)
                else:
                    map_y = map_y - convert_to_pixel_coords(AB_height)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y,2 * delta_x, 2 * delta_y, block)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block)
        
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
                map_y = initial_y + convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height)
            else:
                map_y = map_y + convert_to_pixel_coords(AB_height)

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(image, initial_x, map_y, 
                                                                                            2 * delta_x, 2 * delta_y, block)
            points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block)

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

def sam_segment(df, image):
    all_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        result = process_row(row, image)
        all_results.append(result)
    return all_results

# Use configurable parameters
results = sam_segment(initial_prompt_points, image)

# Generate block to label mapping from segment_order
if "segment_order" in config and config["use_original_label_distributions"]:
    block_to_label = {}
    for i, block_name in enumerate(config["segment_order"], start=1):
        block_to_label[block_name] = i
    print(f"Using configured segment order: {config['segment_order']}")
    print(f"Block to label mapping: {block_to_label}")
else:
    # Fallback to default mapping
    block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}
    if segment_per_ring == 7:
        block_to_label = {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
    print(f"Using default block mapping: {block_to_label}")

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

print(f"Processing completed for tunnel: {tunnel_id}")
print(f"Results saved to {base_dir}/final.csv")
