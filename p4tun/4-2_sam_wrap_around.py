"""
Algorithm 4-2 - SAM-based Tunnel Segment Segmentation (Fully Parameterized)

All prompt points, template masks, and processing parameters are configurable
via JSON for Bayesian Optimization tuning.

Pipeline:
    4-1_detection.py → detected.csv (K positions)
    4-2_sam.py → final.csv (segmented point cloud)
"""

import os
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

# Add segment_anything to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
# Try p4tun/segment-anything first, then sam4tun/segment-anything
segment_anything_paths = [
    os.path.join(repo_root, "p4tun", "segment-anything"),
    os.path.join(repo_root, "sam4tun", "segment-anything"),
]
for path in segment_anything_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        break

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path

# =============================================================================
# Default Parameter Values (used when not specified in JSON)
# =============================================================================

DEFAULT_PARAMS = {
    "segment_geometry": {
        "segment_width": 1200.0,
        "k_height": 1079.92,
        "ab_height": 3239.77,
        "angle_deg": 7.52
    },
    "image": {
        "resolution": 0.005
    },
    "processing": {
        "padding": 150,
        "crop_margin": 50,
        "mask_eps": 0.001,
        "y_bounds": [4200, 13100],
        "enable_y_wraparound": True  # Enable theta-seam wraparound cropping
    },
    "prompt_points": {
        "k_block": {
            "outer_ring": 700,
            "middle_ring": 500,
            "inner_ring": 348.16,
            "center_ring": 325,
            "spacing_factors": {
                "k_block_spacing": 310.91,
                "vertical_spacing": [732.35, 505.96, 310.91, 219.01, 373.96]
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
                "special_levels": [1298.93, 1390.84, 1427.43, 1612.28, 1627.49, 1652.43, 1673.69, 1766.08, 1787.34, 1812.28, 1345.01, 1452.43, 1473.69, 1566.08, 1587.34]
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
    },
    "pattern_aware": {
        "use_quality_weighting": True,
        "min_quality_threshold": 0.3
    }
}


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str, base_dir: str = "data"):
    """Load parameters from JSON file with defaults fallback."""
    script_dir = os.path.dirname(__file__)
    
    # Try tunnel-specific params first
    params_path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_sam.json")
    if os.path.exists(params_path):
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            return json.load(f)
    
    # Try sample params
    sample_path = os.path.join(script_dir, "parameters", "sample", "parameters_sam.json")
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    print("Using hardcoded default parameters")
    return {}


def get_param(params, *keys, default=None):
    """Get nested parameter value with fallback to defaults."""
    # First try user params
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            # Fall back to defaults
            value = DEFAULT_PARAMS
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    return value


def deep_merge(base, override):
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# Physical Constants and Segment Count Detection
# =============================================================================

K_HEIGHT_MM = 1079.92
AB_HEIGHT_MM = 3239.77
DEFAULT_RESOLUTION = 0.005


def detect_segment_count_from_geometry(tunnel_dir: str, resolution: float = DEFAULT_RESOLUTION) -> int:
    """Detect segment count from tunnel geometry (radius → circumference)."""
    enhanced_path = os.path.join(tunnel_dir, 'enhanced.csv')
    
    if os.path.exists(enhanced_path):
        df = pd.read_csv(enhanced_path)
        if 'r' in df.columns:
            avg_radius = df['r'].mean()
            circumference_mm = 2 * np.pi * avg_radius * 1000
            
            circ_6 = K_HEIGHT_MM + 5 * AB_HEIGHT_MM
            circ_7 = K_HEIGHT_MM + 6 * AB_HEIGHT_MM
            
            segment_count = 6 if abs(circumference_mm - circ_6) < abs(circumference_mm - circ_7) else 7
            print(f"Detected from geometry: {segment_count} segments")
            return segment_count
    
    return None


def detect_segment_count_from_height(image_height: int, resolution: float = 0.005) -> int:
    """Fallback: Auto-detect 6 or 7 segments from image height."""
    height_mm = image_height * resolution * 1000
    circumference_6 = K_HEIGHT_MM + 5 * AB_HEIGHT_MM
    circumference_7 = K_HEIGHT_MM + 6 * AB_HEIGHT_MM
    
    return 6 if abs(height_mm - circumference_6) < abs(height_mm - circumference_7) else 7


# =============================================================================
# Mask Generation Functions (Fully Parameterized)
# =============================================================================

def fill_polygon(mask, vertices):
    """Fill polygon in mask using matplotlib Path."""
    path = Path(vertices)
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)
    mask[mask_inside] = 1


def generate_template_mask(height, width, prompt_centre, block, resolution, template_params):
    """Generate template mask using parameterized dimensions."""
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)
    
    if block == 'K':
        k_mask = template_params.get('k_block', DEFAULT_PARAMS['prompt_points']['template_mask']['k_block'])
        w = k_mask.get('width', 625)
        hp = k_mask.get('height_pos', 619.16)
        hn = k_mask.get('height_neg', 460.77)
        vertices_real = np.array([[x-w, y-hp], [x-w, y+hp], [x+w, y+hn], [x+w, y-hn]])
    elif block == 'B1':
        b1_mask = template_params.get('b1_block', DEFAULT_PARAMS['prompt_points']['template_mask']['b1_block'])
        w = b1_mask.get('width', 625)
        ht = b1_mask.get('height_top', 1619.89)
        hbp = b1_mask.get('height_bottom_pos', 1540.69)
        hbn = b1_mask.get('height_bottom_neg', 1699.08)
        vertices_real = np.array([[x-w, y-ht], [x-w, y+hbp], [x+w, y+hbn], [x+w, y-ht]])
    elif block == 'B2':
        b2_mask = template_params.get('b2_block', DEFAULT_PARAMS['prompt_points']['template_mask']['b2_block'])
        w = b2_mask.get('width', 625)
        htp = b2_mask.get('height_top_pos', 1540.69)
        htn = b2_mask.get('height_top_neg', 1699.08)
        hb = b2_mask.get('height_bottom', 1619.89)
        vertices_real = np.array([[x-w, y-htp], [x-w, y+hb], [x+w, y+hb], [x+w, y-htn]])
    else:  # A blocks
        a_mask = template_params.get('a_blocks', DEFAULT_PARAMS['prompt_points']['template_mask']['a_blocks'])
        w = a_mask.get('width', 625)
        h = a_mask.get('height', 1619.89)
        vertices_real = np.array([[x-w, y-h], [x-w, y+h], [x+w, y+h], [x+w, y-h]])
        
    vertices = vertices_real / (resolution * 1000)
    fill_polygon(mask, vertices)
    return mask


# =============================================================================
# Prompt Point Generation (Fully Parameterized)
# =============================================================================

def generate_prompt_points_k(x, y, k_params, resolution):
    """Generate K-block prompt points using parameterized values."""
    outer = k_params.get('outer_ring', 700)
    middle = k_params.get('middle_ring', 500)
    inner = k_params.get('inner_ring', 348.16)
    center = k_params.get('center_ring', 325)
    
    spacing = k_params.get('spacing_factors', {})
    k_sp = spacing.get('k_block_spacing', 310.91)
    v_sp = spacing.get('vertical_spacing', [732.35, 505.96, 310.91, 219.01, 373.96])
    
    # Ensure v_sp has at least 5 elements
    while len(v_sp) < 5:
        v_sp.append(v_sp[-1] if v_sp else 300)
    
    points_real = np.array([
        # Outer ring points
        [x-outer, y-v_sp[0]], [x-outer, y-v_sp[1]], [x-outer, y-k_sp], [x-outer, y], [x-outer, y+k_sp], [x-outer, y+v_sp[1]], [x-outer, y+v_sp[0]],
        # Middle ring points
        [x-middle, y-v_sp[1]], [x-middle, y+v_sp[1]],
        # Inner ring points
        [x-inner, y-v_sp[1]], [x-inner, y-k_sp], [x-center, y], [x-inner, y+k_sp], [x-inner, y+v_sp[1]],
        # Center column
        [x, y-v_sp[1]], [x, y], [x, y+v_sp[1]],
        # Inner ring (right side)
        [x+inner, y-v_sp[1]], [x+inner, y-v_sp[2]], [x+center, y], [x+inner, y+v_sp[2]], [x+inner, y+v_sp[1]],
        # Middle ring (right side)
        [x+middle, y-v_sp[1]], [x+middle, y+v_sp[1]],
        # Outer ring (right side)
        [x+outer, y-v_sp[1]], [x+outer, y-v_sp[4]], [x+outer, y-v_sp[3]], [x+outer, y], [x+outer, y+v_sp[3]], [x+outer, y+v_sp[4]], [x+outer, y+v_sp[1]],
        # Additional inner points
        [x-middle, y-k_sp], [x-middle-11.06, y-k_sp], [x-middle, y], [x-middle-11.06, y+k_sp], [x-middle, y+k_sp],
        [x-inner, y-k_sp], [x-inner, y+k_sp],
        [x, y-k_sp], [x, y+k_sp],
        [x+inner, y-k_sp], [x+inner, y+k_sp],
        [x+middle, y-k_sp], [x+middle+11.06, y-v_sp[3]], [x+middle, y], [x+middle+11.06, y+v_sp[3]], [x+middle, y+k_sp]
    ])
    labels = np.repeat([0, 1], [31, 16])
    return points_real, labels


def generate_prompt_points_ab(x, y, ab_params, block_type, resolution):
    """Generate A/B block prompt points using parameterized values."""
    outer = ab_params.get('outer_ring', 700)
    middle = ab_params.get('middle_ring', 511.06)
    inner = ab_params.get('inner_ring', 500)
    center = ab_params.get('center_ring', 325)
    fine = ab_params.get('fine_spacing', 250)
    ultra = ab_params.get('ultra_fine', 162.5)
    edge = ab_params.get('edge_ring', 348.16)
    edge_sp = ab_params.get('edge_spacing', 350)
    
    levels = ab_params.get('vertical_levels', {})
    l1 = levels.get('level_1', 1719.89)
    l2 = levels.get('level_2', 1519.89)
    l3 = levels.get('level_3', 1344.89)
    l4 = levels.get('level_4', 1090.09)
    l5 = levels.get('level_5', 817.57)
    l6 = levels.get('level_6', 545.05)
    l7 = levels.get('level_7', 272.52)
    
    if block_type == 'B1':
        points_real = np.array([
            # Top row
            [x-outer, y-l1], [x-middle, y-l1], [x-edge, y-l1], [x, y-l1], [x+edge, y-l1], [x+middle, y-l1], [x+outer, y-l1],
            [x-outer, y-l2], [x+outer, y-l2],
            [x-outer, y-l3], [x-edge, y-l3], [x+edge, y-l3], [x+outer, y-l3],
            [x-outer, y-l4], [x-center, y-l4], [x+center, y-l4], [x+outer, y-l4],
            [x-outer, y-l5], [x+outer, y-l5],
            [x-outer, y-l6], [x+outer, y-l6],
            [x-outer, y-l7], [x+outer, y-l7],
            [x-outer, y], [x-center, y], [x, y], [x+center, y], [x+outer, y],
            [x-outer, y+l7], [x+outer, y+l7],
            [x-outer, y+l6], [x+outer, y+l6],
            [x-outer, y+l5], [x+outer, y+l5],
            [x-outer, y+l4], [x-center, y+l4], [x+center, y+l4], [x+outer, y+l4],
            # Slanted bottom rows
            [x-outer, y+1298.93], [x-edge_sp, y+1298.93], [x+edge_sp, y+1390.84], [x+outer, y+1390.84],
            [x-outer, y+1427.43], [x+outer, y+1612.28],
            [x-outer, y+1627.49], [x-middle, y+1652.43], [x-edge_sp, y+1673.69], [x, y+l1], [x+edge_sp, y+1766.08], [x+middle, y+1787.34], [x+outer, y+1812.28],
            # Inner points
            [x-middle, y-l2], [x-edge, y-l2], [x, y-l2], [x+edge, y-l2], [x+middle, y-l2],
            [x-middle, y-l3], [x, y-l3], [x+middle, y-l3],
            [x-inner, y-l4], [x, y-l4], [x+inner, y-l4],
            [x-inner, y-l5], [x-fine, y-l5], [x, y-l5], [x+fine, y-l5], [x+inner, y-l5],
            [x-inner, y-l6], [x-fine, y-l6], [x, y-l6], [x+fine, y-l6], [x+inner, y-l6],
            [x-inner, y-l7], [x-fine, y-l7], [x, y-l7], [x+fine, y-l7], [x+inner, y-l7],
            [x-inner, y], [x-ultra, y], [x+ultra, y], [x+inner, y],
            [x-inner, y+l7], [x-fine, y+l7], [x, y+l7], [x+fine, y+l7], [x+inner, y+l7],
            [x-inner, y+l6], [x-fine, y+l6], [x, y+l6], [x+fine, y+l6], [x+inner, y+l6],
            [x-inner, y+l5], [x-fine, y+l5], [x, y+l5], [x+fine, y+l5], [x+inner, y+l5],
            [x-inner, y+l4], [x, y+l4], [x+inner, y+l4],
            [x-middle, y+1298.93], [x, y+1345.01], [x+middle, y+1390.84],
            [x-middle, y+1452.43], [x-edge_sp, y+1473.69], [x, y+l2], [x+edge_sp, y+1566.08], [x+middle, y+1587.34]
        ])
    elif block_type == 'B2':
        points_real = np.array([
            # Slanted top rows
            [x-outer, y-1627.49], [x-middle, y-1652.43], [x-edge_sp, y-1673.69], [x, y-l1], [x+edge_sp, y-1766.08], [x+middle, y-1787.34], [x+outer, y-1812.28],
            [x-outer, y-1427.43], [x+outer, y-1612.28],
            [x-outer, y-1298.93], [x-edge_sp, y-1298.93], [x+edge_sp, y-1390.84], [x+outer, y-1390.84],
            [x-outer, y-l4], [x-center, y-l4], [x+center, y-l4], [x+outer, y-l4],
            [x-outer, y-l5], [x+outer, y-l5],
            [x-outer, y-l6], [x+outer, y-l6],
            [x-outer, y-l7], [x+outer, y-l7],
            [x-outer, y], [x-center, y], [x, y], [x+center, y], [x+outer, y],
            [x-outer, y+l7], [x+outer, y+l7],
            [x-outer, y+l6], [x+outer, y+l6],
            [x-outer, y+l5], [x+outer, y+l5],
            [x-outer, y+l4], [x-center, y+l4], [x+center, y+l4], [x+outer, y+l4],
            [x-outer, y+l3], [x-edge, y+l3], [x+edge, y+l3], [x+outer, y+l3],
            [x-outer, y+l2], [x+outer, y+l2],
            # Bottom row
            [x-outer, y+l1], [x-middle, y+l1], [x-edge, y+l1], [x, y+l1], [x+edge, y+l1], [x+middle, y+l1], [x+outer, y+l1],
            # Inner points
            [x-middle, y-1452.43], [x-edge_sp, y-1473.69], [x, y-l2], [x+edge_sp, y-1566.08], [x+middle, y-1587.34],
            [x-middle, y-1298.93], [x, y-1345.01], [x+middle, y-1390.84],
            [x-inner, y-l4], [x, y-l4], [x+inner, y-l4],
            [x-inner, y-l5], [x-fine, y-l5], [x, y-l5], [x+fine, y-l5], [x+inner, y+l5],
            [x-inner, y-l6], [x-fine, y-l6], [x, y-l6], [x+fine, y-l6], [x+inner, y-l6],
            [x-inner, y-l7], [x-fine, y-l7], [x, y-l7], [x+fine, y-l7], [x+inner, y-l7],
            [x-inner, y], [x-ultra, y], [x+ultra, y], [x+inner, y],
            [x-inner, y+l7], [x-fine, y+l7], [x, y+l7], [x+fine, y+l7], [x+inner, y+l7],
            [x-inner, y+l6], [x-fine, y+l6], [x, y+l6], [x+fine, y+l6], [x+inner, y+l6],
            [x-inner, y+l5], [x-fine, y+l5], [x, y+l5], [x+fine, y+l5], [x+inner, y+l5],
            [x-inner, y+l4], [x, y+l4], [x+inner, y+l4],
            [x-middle, y+l3], [x, y+l3], [x+middle, y+l3],
            [x-middle, y+l2], [x-edge, y+l2], [x, y+l2], [x+edge, y+l2], [x+middle, y+l2],
        ])
    else:  # A blocks - rectangular pattern
        points_real = np.array([
            # Top and bottom rows (symmetric)
            [x-outer, y-l1], [x-middle, y-l1], [x-edge, y-l1], [x, y-l1], [x+edge, y-l1], [x+middle, y-l1], [x+outer, y-l1],
            [x-outer, y-l2], [x+outer, y-l2],
            [x-outer, y-l3], [x-edge, y-l3], [x+edge, y-l3], [x+outer, y-l3],
            [x-outer, y-l4], [x-center, y-l4], [x+center, y-l4], [x+outer, y-l4],
            [x-outer, y-l5], [x+outer, y-l5],
            [x-outer, y-l6], [x+outer, y-l6],
            [x-outer, y-l7], [x+outer, y-l7],
            [x-outer, y], [x-center, y], [x, y], [x+center, y], [x+outer, y],
            [x-outer, y+l7], [x+outer, y+l7],
            [x-outer, y+l6], [x+outer, y+l6],
            [x-outer, y+l5], [x+outer, y+l5],
            [x-outer, y+l4], [x-center, y+l4], [x+center, y+l4], [x+outer, y+l4],
            [x-outer, y+l3], [x-edge, y+l3], [x+edge, y+l3], [x+outer, y+l3],
            [x-outer, y+l2], [x+outer, y+l2],
            [x-outer, y+l1], [x-middle, y+l1], [x-edge, y+l1], [x, y+l1], [x+edge, y+l1], [x+middle, y+l1], [x+outer, y+l1],
            # Inner points
            [x-middle, y-l2], [x-edge, y-l2], [x, y-l2], [x+edge, y-l2], [x+middle, y-l2],
            [x-middle, y-l3], [x, y-l3], [x+middle, y-l3],
            [x-inner, y-l4], [x, y-l4], [x+inner, y-l4],
            [x-inner, y-l5], [x-fine, y-l5], [x, y-l5], [x+fine, y-l5], [x+inner, y-l5],
            [x-inner, y-l6], [x-fine, y-l6], [x, y-l6], [x+fine, y-l6], [x+inner, y-l6],
            [x-inner, y-l7], [x-fine, y-l7], [x, y-l7], [x+fine, y-l7], [x+inner, y-l7],
            [x-inner, y], [x-ultra, y], [x+ultra, y], [x+inner, y],
            [x-inner, y+l7], [x-fine, y+l7], [x, y+l7], [x+fine, y+l7], [x+inner, y+l7],
            [x-inner, y+l6], [x-fine, y+l6], [x, y+l6], [x+fine, y+l6], [x+inner, y+l6],
            [x-inner, y+l5], [x-fine, y+l5], [x, y+l5], [x+fine, y+l5], [x+inner, y+l5],
            [x-inner, y+l4], [x, y+l4], [x+inner, y+l4],
            [x-middle, y+l3], [x, y+l3], [x+middle, y+l3],
            [x-middle, y+l2], [x-edge, y+l2], [x, y+l2], [x+edge, y+l2], [x+middle, y+l2],
        ])
    
    labels = np.repeat([0, 1], [51, 56])
    return points_real, labels


def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution,
                           segment_width, K_height, AB_height, image, 
                           k_params, ab_params, y_bounds):
    """
    Generate prompt points for SAM using fully parameterized values.
    """
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)
    map_y_mm = map_y * (resolution * 1000)
    
    if block == 'K':
        points_real, labels = generate_prompt_points_k(x, y, k_params, resolution)
    else:
        points_real, labels = generate_prompt_points_ab(x, y, ab_params, block, resolution)

    # Filter points based on y_bounds
    keep_mask = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 0:
            y_cond = points_real[i, 1] + map_y_mm < y_bounds[0] or points_real[i, 1] + map_y_mm > y_bounds[1]
            x_cond = abs(points_real[i, 0] - x) <= segment_width * 0.5
            y_limit = K_height if block == 'K' else AB_height
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
            
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    
    points = points_real / (resolution * 1000)

    # Filter points outside image bounds
    if image is not None:
        padding = 150  # Default padding
        within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (segment_width*0.5+padding)/(resolution*1000)) <= image.shape[1])
        points = points[within_bounds]
        labels = labels[within_bounds]
        
    return points, labels


# =============================================================================
# Helper Functions
# =============================================================================

def convert_to_pixel_coords(real_dist, resolution=0.005):
    return int(real_dist / (resolution * 1000))


def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution, template_params, mask_eps, enable_y_wraparound=True):
    """
    Crop image around (cx, cy) and build template mask logits.

    IMPORTANT: In our depth-map convention, the circumferential axis (theta) is the IMAGE Y axis.
    Theta is periodic, so Y should be treated as wraparound (top/bottom seam).

    This function performs **Y-wraparound-aware cropping** when enabled:
    - If the crop extends above y=0, it stitches bottom rows above the top.
    - If the crop extends below y=H, it stitches top rows below the bottom.

    Args:
        enable_y_wraparound: If True, enable wraparound stitching. If False, use clamping (old behavior).

    Returns:
        cropped_image, template_mask_logits, prompt_centre, is_wrapped_y
    """
    img_height, img_width, _ = image.shape

    # X is NOT periodic (along-tunnel); clamp as usual
    x1 = max(int(cx - crop_width // 2), 0)
    x2 = min(int(cx + crop_width // 2), img_width)

    # Y (theta) IS periodic; handle based on enable_y_wraparound flag
    y1 = int(cy - crop_height // 2)
    y2 = int(cy + crop_height // 2)

    is_wrapped_y = False

    if not enable_y_wraparound:
        # Old behavior: clamp Y (may cut off segments that span boundary)
        y1 = max(y1, 0)
        y2 = min(y2, img_height)
        cropped_image = image[y1:y2, x1:x2]
        prompt_centre_x = cx - x1
        prompt_centre_y = cy - y1
    elif y1 >= 0 and y2 <= img_height:
        # Normal case: no wraparound needed
        cropped_image = image[y1:y2, x1:x2]
        prompt_centre_x = cx - x1
        prompt_centre_y = cy - y1
    else:
        # Wraparound crop in Y by stitching
        is_wrapped_y = True
        # Clamp slice extents for safe indexing, but stitch the overflow
        if y1 < 0:
            top_overflow = -y1
            # Bottom part that wraps to the top of the crop
            bottom_start = max(0, img_height - top_overflow)
            part1 = image[bottom_start:img_height, x1:x2]
            part2 = image[0:max(0, y2), x1:x2]
            cropped_image = np.concatenate([part1, part2], axis=0)
            # In stitched crop: wrapped part is [0:top_overflow], normal part is [top_overflow:]
            # cy maps to: top_overflow + cy (since cy is in [0, y2] range)
            prompt_centre_x = cx - x1
            prompt_centre_y = top_overflow + cy
        else:
            # y2 > img_height
            bottom_overflow = y2 - img_height
            part1 = image[min(img_height, y1):img_height, x1:x2]
            part2 = image[0:min(img_height, bottom_overflow), x1:x2]
            cropped_image = np.concatenate([part1, part2], axis=0)
            # In stitched crop: normal part is [0:img_height-y1], wrapped part is [img_height-y1:]
            # cy maps to: cy - y1 (since cy is in [y1, img_height] range, this is in [0, img_height-y1])
            prompt_centre_x = cx - x1
            prompt_centre_y = cy - y1

    # prompt_centre is in CROPPED image coordinates.
    # Note: use the *unclamped* y1 for left_top calculation so aggregation can handle wraparound
    prompt_centre = (prompt_centre_x, prompt_centre_y)
    
    cropped_template_mask = generate_template_mask(
        cropped_image.shape[0], cropped_image.shape[1], 
        prompt_centre, block, resolution, template_params
    )
    template_mask_logits = compute_logits_from_mask(cropped_template_mask, mask_eps)

    return cropped_image, template_mask_logits, prompt_centre, is_wrapped_y


def _apply_patch_region(
    logits_map: np.ndarray,
    label_map: np.ndarray,
    ring_map: np.ndarray,
    new_logits: np.ndarray,
    mask: np.ndarray,
    start_x: int,
    start_y: int,
    block_label: int,
    ring_index: int,
):
    """
    Apply (new_logits, mask) onto maps at (start_x, start_y) with max-logit overwrite.
    Assumes the placement region lies fully within [0,H) in Y and [0,W) in X.
    """
    img_height, img_width = label_map.shape[:2]
    mask_h, mask_w = mask.shape

    end_y = min(start_y + mask_h, img_height)
    end_x = min(start_x + mask_w, img_width)
    start_y_clamped = max(0, start_y)
    start_x_clamped = max(0, start_x)

    actual_h = end_y - start_y_clamped
    actual_w = end_x - start_x_clamped
    if actual_h <= 0 or actual_w <= 0:
        return

    # Align mask/logits slices if start_x/start_y were clamped
    mask_y0 = start_y_clamped - start_y
    mask_x0 = start_x_clamped - start_x
    mask_region = mask[mask_y0:mask_y0 + actual_h, mask_x0:mask_x0 + actual_w]
    logits_region = new_logits[mask_y0:mask_y0 + actual_h, mask_x0:mask_x0 + actual_w]

    current = logits_map[start_y_clamped:end_y, start_x_clamped:end_x]
    if mask_region.shape != current.shape or logits_region.shape != current.shape:
        return

    update = (logits_region > current) & mask_region
    if not np.any(update):
        return

    logits_map[start_y_clamped:end_y, start_x_clamped:end_x][update] = logits_region[update]
    label_map[start_y_clamped:end_y, start_x_clamped:end_x][update] = block_label
    ring_map[start_y_clamped:end_y, start_x_clamped:end_x][update] = ring_index


def apply_mask_logits_with_y_wraparound(
    logits_map: np.ndarray,
    label_map: np.ndarray,
    ring_map: np.ndarray,
    new_logits: np.ndarray,
    mask: np.ndarray,
    start_x: int,
    start_y: int,
    block_label: int,
    ring_index: int,
):
    """
    Apply a crop-local (mask, new_logits) back to the full image maps.

    Y is treated as periodic (theta seam): if start_y < 0 or start_y+H_crop > H_img,
    we split the patch and write the overflow to the opposite side.
    """
    img_height, _ = label_map.shape[:2]
    mask_h = mask.shape[0]
    end_y = start_y + mask_h

    if start_y >= 0 and end_y <= img_height:
        _apply_patch_region(logits_map, label_map, ring_map, new_logits, mask, start_x, start_y, block_label, ring_index)
        return

    # Wraparound cases
    if start_y < 0:
        top_overflow = -start_y
        wrap_h = min(top_overflow, mask_h)

        # Part that wraps to bottom of image: mask rows [0:wrap_h] -> y [H-wrap_h:H]
        if wrap_h > 0:
            wrapped_start_y = img_height - wrap_h
            _apply_patch_region(
                logits_map, label_map, ring_map,
                new_logits[:wrap_h, :], mask[:wrap_h, :],
                start_x, wrapped_start_y, block_label, ring_index
            )

        # Remaining part goes to top starting at y=0: mask rows [wrap_h:] -> y [0:...]
        remaining_h = mask_h - wrap_h
        if remaining_h > 0:
            _apply_patch_region(
                logits_map, label_map, ring_map,
                new_logits[wrap_h:, :], mask[wrap_h:, :],
                start_x, 0, block_label, ring_index
            )
        return

    # end_y > img_height
    bottom_overflow = end_y - img_height
    normal_h = max(0, mask_h - bottom_overflow)

    # Normal bottom part: mask rows [0:normal_h] -> y [start_y:H]
    if normal_h > 0:
        _apply_patch_region(
            logits_map, label_map, ring_map,
            new_logits[:normal_h, :], mask[:normal_h, :],
            start_x, start_y, block_label, ring_index
        )

    # Overflow wraps to top: mask rows [normal_h:] -> y [0:bottom_overflow]
    wrap_h = min(bottom_overflow, mask_h - normal_h)
    if wrap_h > 0:
        _apply_patch_region(
            logits_map, label_map, ring_map,
            new_logits[normal_h:normal_h + wrap_h, :], mask[normal_h:normal_h + wrap_h, :],
            start_x, 0, block_label, ring_index
        )


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


def compute_block_label(segment_per_ring, segment_order=None):
    """
    Get block labels in processing order.
    
    If segment_order is provided, use it directly.
    Otherwise, compute default order from segment count.
    """
    if segment_order is not None:
        return segment_order
    
    # Default order: K, B1, A1, ..., An, B2
    block_labels = ['K', 'B1']
    num_a_labels = segment_per_ring - 3
    block_labels += [f'A{i+1}' for i in range(num_a_labels)]
    block_labels += ['B2']
    return block_labels


def compute_block_to_label_map(segment_per_ring, segment_order=None):
    """
    Create mapping from block names to numeric labels.
    
    Labels are assigned based on default order (K=1, B1=2, A1=3, ...),
    NOT the processing order. This ensures consistent labeling regardless
    of segment_order changes.
    """
    if segment_per_ring == 7:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
    else:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}


# =============================================================================
# Row Processing (Fully Parameterized)
# =============================================================================

def process_row(df_row, image, predictor, config):
    """
    Process a single row (ring) with fully parameterized settings.
    
    Uses original algorithm that handles partial visibility correctly.
    segment_order is used for aggregation priority, not position calculation.
    """
    initial_x, initial_y = df_row['X'], df_row['Y']
    quality = df_row.get('quality', 1.0) if hasattr(df_row, 'get') else 1.0
    
    # Extract config
    resolution = config['resolution']
    segment_per_ring = config['segment_per_ring']
    segment_width = config['segment_width']
    K_height = config['K_height']
    AB_height = config['AB_height']
    angle = config['angle']
    padding = config['padding']
    crop_margin = config['crop_margin']
    mask_eps = config['mask_eps']
    y_bounds = config['y_bounds']
    k_params = config['k_params']
    ab_params = config['ab_params']
    template_params = config['template_params']
    enable_y_wraparound = config.get('enable_y_wraparound', True)
    
    # Always use physical order for position calculation
    block_labels = compute_block_label(segment_per_ring, None)

    delta_x = convert_to_pixel_coords(0.5 * segment_width + padding, resolution)
    delta_y = 0

    reverse = False
    stop = False
    map_y = 0
    block_label_index = 0

    results = []
    for i in range(segment_per_ring):
        if not reverse:
            block = block_labels[block_label_index]
            if block_label_index == 0:
                delta_y = convert_to_pixel_coords(0.5 * K_height + math.tan(math.radians(angle)) * 700 + 100 + crop_margin, resolution)
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5 * AB_height + math.tan(math.radians(angle)) * 700 + 100 + crop_margin, resolution)
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * K_height + 0.5 * AB_height, resolution)
                else:
                    map_y = map_y - convert_to_pixel_coords(AB_height, resolution)

            cropped_image, template_mask_logit, prompt_centre, is_wrapped_y = crop_image_and_mask_logits(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution, template_params, mask_eps, enable_y_wraparound)
            points, labels = generate_prompt_points(
                prompt_centre, initial_x, map_y, block, resolution,
                segment_width, K_height, AB_height, image, k_params, ab_params, y_bounds)

            # If crop wrapped in Y (theta seam), points are already in correct cropped coordinates
            # (prompt_centre_y was adjusted during crop, so points relative to it are correct)
            if len(points) > 0 and is_wrapped_y:
                # Just ensure points are within crop bounds (they should already be, but clamp for safety)
                crop_h = cropped_image.shape[0]
                points[:, 1] = np.clip(points[:, 1], 0, crop_h - 1)
                reverse = False
            elif len(points) > 0 and np.any(points[:, 1] < 0):
                within_bounds = (points[:, 1] >= 0)
                points = points[within_bounds]
                labels = labels[within_bounds]
                reverse = True
            
            if len(points) > 0:
                predictor.set_image(cropped_image)
                mask, score, logit = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=template_mask_logit,
                    multimask_output=False,
                )
            
                results.append({
                    'left_top': (initial_x - prompt_centre[0], map_y - prompt_centre[1]),
                    'block': block,
                    'mask': mask,
                    'score': score,
                    'logit': logit[0],
                    'quality': quality
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

            cropped_image, template_mask_logit, prompt_centre, is_wrapped_y = crop_image_and_mask_logits(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution, template_params, mask_eps, enable_y_wraparound)
            points, labels = generate_prompt_points(
                prompt_centre, initial_x, map_y, block, resolution,
                segment_width, K_height, AB_height, image, k_params, ab_params, y_bounds)

            # If crop wrapped in Y (theta seam), points are already in correct cropped coordinates
            if len(points) > 0 and is_wrapped_y:
                crop_h = cropped_image.shape[0]
                points[:, 1] = np.clip(points[:, 1], 0, crop_h - 1)
                stop = False
            elif len(points) > 0 and np.any((points[:, 1] + map_y - delta_y) > image.shape[0]):
                within_bounds = ((points[:, 1] + map_y - delta_y) <= image.shape[0])
                points = points[within_bounds]
                labels = labels[within_bounds]
                stop = True

            if len(points) > 0:
                predictor.set_image(cropped_image)
                mask, score, logit = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=template_mask_logit,
                    multimask_output=False,
                )

                results.append({
                    'left_top': (initial_x - prompt_centre[0], map_y - prompt_centre[1]),
                    'block': block,
                    'mask': mask,
                    'score': score,
                    'logit': logit[0],
                    'quality': quality
                })

            if stop:
                break

            block_label_index = block_label_index - 1
             
    return results


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


# =============================================================================
# Main Pipeline
# =============================================================================

def run_sam(tunnel_id: str, base_dir: str = "data", segment_count: int = None):
    """Run SAM segmentation with fully parameterized settings."""
    
    # Load and merge parameters
    params = load_parameters(tunnel_id, base_dir)
    params = deep_merge(DEFAULT_PARAMS, params)
    
    # Extract parameters
    resolution = params['image']['resolution']
    segment_width = params['segment_geometry']['segment_width']
    K_height = params['segment_geometry']['k_height']
    AB_height = params['segment_geometry']['ab_height']
    angle = params['segment_geometry']['angle_deg']
    
    # Processing parameters
    padding = params['processing']['padding']
    crop_margin = params['processing']['crop_margin']
    mask_eps = params['processing']['mask_eps']
    y_bounds = params['processing']['y_bounds']
    enable_y_wraparound = get_param(params, 'processing', 'enable_y_wraparound', default=True)
    
    # Prompt point parameters
    k_params = params['prompt_points']['k_block']
    ab_params = params['prompt_points']['ab_blocks']
    template_params = params['prompt_points']['template_mask']
    
    # Pattern-aware parameters
    use_quality_weighting = params['pattern_aware']['use_quality_weighting']
    min_quality_threshold = params['pattern_aware']['min_quality_threshold']
    
    # Segment order parameters (for processing priority)
    # Try root level first, then segment_geometry for nested format
    segment_order = params.get('segment_order', None)
    if segment_order is None:
        segment_order = get_param(params, 'segment_geometry', 'segment_order', default=None)
    
    segment_per_ring_from_params = params.get('segment_per_ring', None)
    if segment_per_ring_from_params is None:
        segment_per_ring_from_params = get_param(params, 'segment_geometry', 'segment_per_ring', default=None)
    
    # Load data
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    pattern_csv_path = os.path.join(tunnel_dir, "pattern.csv")
    detected_csv_path = os.path.join(tunnel_dir, "detected.csv")
    
    if os.path.exists(pattern_csv_path):
        print(f"Loading K positions from pattern.csv")
        initial_prompt_points = pd.read_csv(pattern_csv_path)
    elif os.path.exists(detected_csv_path):
        print(f"Warning: pattern.csv not found, using detected.csv")
        initial_prompt_points = pd.read_csv(detected_csv_path)
    else:
        raise FileNotFoundError(f"Neither pattern.csv nor detected.csv found in {tunnel_dir}")
    
    pixel_to_point = pickle.load(open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb"))
    df_point_cloud = pd.read_csv(os.path.join(tunnel_dir, "enhanced.csv"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    
    print(f"Processing tunnel: {tunnel_id}")
    print(f"Segment geometry: width={segment_width}, K_height={K_height}, AB_height={AB_height}, angle={angle}")
    
    # Detect segment count (priority: argument > params > auto-detect)
    if segment_count is None:
        if segment_per_ring_from_params is not None:
            segment_count = segment_per_ring_from_params
        else:
            segment_count = detect_segment_count_from_geometry(tunnel_dir, resolution)
            if segment_count is None:
                image = cv2.imread(os.path.join(tunnel_dir, 'depth_map.png'))
                segment_count = detect_segment_count_from_height(image.shape[0], resolution)
    
    # Load SAM model
    sam_checkpoint = "sam4tun/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Load image
    image = cv2.imread(os.path.join(tunnel_dir, 'depth_map.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Using {segment_count} segments per ring")
    if segment_order:
        print(f"Processing order: {segment_order}")
    
    block_to_label = compute_block_to_label_map(segment_count, segment_order)
    
    # Build config for row processing
    config = {
        'resolution': resolution,
        'segment_per_ring': segment_count,
        'segment_order': segment_order,  # For processing priority
        'segment_width': segment_width,
        'K_height': K_height,
        'AB_height': AB_height,
        'angle': angle,
        'padding': padding,
        'crop_margin': crop_margin,
        'mask_eps': mask_eps,
        'y_bounds': y_bounds,
        'k_params': k_params,
        'ab_params': ab_params,
        'template_params': template_params,
        'enable_y_wraparound': enable_y_wraparound,
    }
    
    # Run SAM segmentation
    all_results = []
    for _, row in tqdm(initial_prompt_points.iterrows(), total=len(initial_prompt_points), desc="Processing rows"):
        result = process_row(row, image, predictor, config)
        all_results.append(result)
    
    # Aggregate results
    logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
    label_map = np.zeros(image.shape[:2], dtype=int)
    ring_map = np.zeros(image.shape[:2], dtype=int)

    for ring_index, ring in enumerate(all_results, start=0):
        for item in ring:
            mask = item['mask'][0]
            logits = item['logit']
            block = item['block']
            start_x, start_y = map(int, item['left_top'])
            quality = item.get('quality', 1.0)

            new_logits = restore_sam_logits(logits, mask.shape)
            
            if use_quality_weighting and quality >= min_quality_threshold:
                new_logits = new_logits * quality
            elif quality < min_quality_threshold:
                continue

            # Apply patch back with Y-wraparound (theta seam) support
            apply_mask_logits_with_y_wraparound(
                logits_map=logits_map,
                label_map=label_map,
                ring_map=ring_map,
                new_logits=new_logits,
                mask=mask,
                start_x=start_x,
                start_y=start_y,
                block_label=block_to_label.get(block, 0),
                ring_index=ring_index,
            )

    # Fix ring numbering
    fix_ring = np.where((ring_map >= 1) & (ring_map <= (ring_count-1)), ring_count - ring_map, ring_map)
    
    # Project back to point cloud
    updated_df = project_back_to_point_cloud(label_map, fix_ring, pixel_to_point, df_point_cloud)
    
    # Save results
    os.makedirs(tunnel_dir, exist_ok=True)
    updated_df.to_csv(os.path.join(tunnel_dir, 'final.csv'), index=False)
    
    if 'segment' in updated_df.columns:
        df_pred = pd.DataFrame()
        df_pred['gt_labels'] = updated_df['segment']
        df_pred['gt_rings'] = updated_df['ring']
        df_pred['pred_labels'] = updated_df['pred']
        df_pred['pred_rings'] = updated_df['pred_ring']
        df_pred.to_csv(os.path.join(tunnel_dir, 'only_label.csv'), index=False)
    
    print(f"Saved results to {tunnel_dir}")
    return updated_df


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAM-based tunnel segmentation (fully parameterized)")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--segments", type=int, default=None, help="Number of segments (auto-detect if omitted)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_sam(args.tunnel_id, base_dir=args.data_dir, segment_count=args.segments)
