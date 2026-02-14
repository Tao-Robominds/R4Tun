"""
Simple Staggered SAM Segmentation (Stage 3)

SAM-based segment segmentation for SIMPLE STAGGERED tunnels only.
For continuous or complex staggered patterns, use the appropriate pipeline.

Simple staggered characteristics:
- Regular K-block positions at alternating Y coordinates
- 6 segments per ring (K, B1, A1, A2, A3, B2)
- Consistent oblique angle across rings

Pipeline:
    1_preprocessing.py → depth_map.png, enhanced.csv, pixel_to_point.pkl
    2_detection.py → detected.csv (K positions)
    3_sam.py → final.csv (segmented point cloud)

CRITICAL PARAMETERS (9 tunable):
    - Segment geometry: segment_width, angle_deg
    - Template masks: k_mask_width, k_mask_height_pos/neg, ab_mask_width, ab_mask_height
    - Processing: padding, crop_margin
    - Quality: min_quality_threshold
    
INHERITED FROM PREPROCESSING:
    - resolution (depth_map_resolution)
    - tunnel_diameter (→ k_height_mm, ab_height_mm)
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
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path


# =============================================================================
# DEFAULT VALUES FOR TUNABLE PARAMETERS
# =============================================================================

# Segment geometry defaults
DEFAULT_SEGMENT_WIDTH = 1200.0
DEFAULT_ANGLE_DEG = 7.5

# Segment physical heights defaults (in mm)
DEFAULT_K_HEIGHT = 1079.92
DEFAULT_AB_HEIGHT = 3239.77

# Template mask defaults (in mm)
DEFAULT_K_MASK_WIDTH = 625.0
DEFAULT_K_MASK_HEIGHT_POS = 620.0
DEFAULT_K_MASK_HEIGHT_NEG = 460.0
DEFAULT_AB_MASK_WIDTH = 625.0
DEFAULT_AB_MASK_HEIGHT = 1620.0

# B1/B2 block template mask defaults (separate heights for slanted edges)
DEFAULT_B1_HEIGHT_TOP = 1619.89
DEFAULT_B1_HEIGHT_BOTTOM_POS = 1540.69
DEFAULT_B1_HEIGHT_BOTTOM_NEG = 1699.08
DEFAULT_B2_HEIGHT_TOP_POS = 1540.69
DEFAULT_B2_HEIGHT_TOP_NEG = 1699.08
DEFAULT_B2_HEIGHT_BOTTOM = 1619.89

# Processing defaults
DEFAULT_PADDING = 150
DEFAULT_CROP_MARGIN = 50

# Quality weighting defaults
DEFAULT_MIN_QUALITY_THRESHOLD = 0.3
DEFAULT_USE_QUALITY_WEIGHTING = True


# =============================================================================
# FIXED PARAMETERS (Not tuned - use hardcoded values)
# =============================================================================

FIXED_MASK_EPS = 0.001

# Fixed prompt point parameters (complex, low individual impact)
FIXED_K_OUTER_RING = 700.0
FIXED_K_MIDDLE_RING = 500.0
FIXED_K_INNER_RING = 348.16
FIXED_K_CENTER_RING = 325.0
FIXED_K_BLOCK_SPACING = 310.91
FIXED_K_VERTICAL_SPACING = [732.35, 505.96, 310.91, 219.01, 373.96]

FIXED_AB_OUTER_RING = 700.0
FIXED_AB_MIDDLE_RING = 511.06
FIXED_AB_INNER_RING = 500.0
FIXED_AB_CENTER_RING = 325.0
FIXED_AB_FINE_SPACING = 250.0
FIXED_AB_ULTRA_FINE = 162.5
FIXED_AB_EDGE_RING = 348.16
FIXED_AB_EDGE_SPACING = 350.0
FIXED_AB_VERTICAL_LEVELS = {
    'level_1': 1719.89,
    'level_2': 1519.89,
    'level_3': 1344.89,
    'level_4': 1090.09,
    'level_5': 817.57,
    'level_6': 545.05,
    'level_7': 272.52,
    'center': 0
}

# SAM model configuration
SAM_CHECKPOINT = "skills/segment-anything/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_DEVICE = "cuda"


# =============================================================================
# PARAMETER LOADING
# =============================================================================

def load_parameters(tunnel_id: str, base_dir: str = "data"):
    """Load SAM parameters from JSON file with fallback."""
    script_dir = os.path.dirname(__file__)
    
    # Try tunnel-specific params first
    params_path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_sam.json")
    if os.path.exists(params_path):
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            return json.load(f), True
    
    # Try sample params
    sample_path = os.path.join(script_dir, "parameters", "sample", "parameters_sam.json")
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
            return json.load(f), True
    
    print("Using default parameters")
    return {}, False


def load_preprocessing_params(tunnel_id: str, base_dir: str = "data"):
    """Load preprocessing parameters to inherit physical constants."""
    script_dir = os.path.dirname(__file__)
    
    # Navigate to preprocessing parameters
    preprocessing_dir = os.path.join(script_dir, "..", "1_preprocessing", "parameters")
    
    # Try tunnel-specific
    params_path = os.path.join(preprocessing_dir, tunnel_id, "parameters_preprocessing.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            return json.load(f)
    
    # Try sample
    sample_path = os.path.join(preprocessing_dir, "sample", "parameters_preprocessing.json")
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    return {}


def get_param(params, key, default=None, allow_default=True):
    """Get parameter value with fallback to default."""
    if key in params:
        return params[key]
    if allow_default and default is not None:
        return default
    raise ValueError(f"Required parameter '{key}' not found and no default allowed")


def calculate_segment_heights(tunnel_diameter: float):
    """Calculate K-block and AB-block heights from tunnel diameter.
    
    Formula: k_height = π × diameter × 1000 / 16
             ab_height = 3 × k_height
    """
    k_height_mm = math.pi * tunnel_diameter * 1000 / 16
    ab_height_mm = 3 * k_height_mm
    return k_height_mm, ab_height_mm


# Default physical heights for segment count detection
DEFAULT_K_HEIGHT_MM = 1079.92
DEFAULT_AB_HEIGHT_MM = 3239.77


# =============================================================================
# SEGMENT COUNT DETECTION
# =============================================================================

def detect_segment_count(tunnel_dir: str, default: int = 6) -> int:
    """Detect segment count from tunnel geometry (radius → circumference).
    
    Compares circumference to expected values for 6 vs 7 segments.
    Default for simple staggered: 6
    """
    enhanced_path = os.path.join(tunnel_dir, 'enhanced.csv')
    
    if os.path.exists(enhanced_path):
        df = pd.read_csv(enhanced_path)
        if 'r' in df.columns:
            avg_radius = df['r'].mean()
            circumference_mm = 2 * np.pi * avg_radius * 1000
            
            circ_6 = DEFAULT_K_HEIGHT_MM + 5 * DEFAULT_AB_HEIGHT_MM
            circ_7 = DEFAULT_K_HEIGHT_MM + 6 * DEFAULT_AB_HEIGHT_MM
            
            segment_count = 6 if abs(circumference_mm - circ_6) < abs(circumference_mm - circ_7) else 7
            print(f"Detected from geometry: {segment_count} segments (radius={avg_radius:.3f}m)")
            return segment_count
    
    print(f"Using default segment count: {default}")
    return default


# =============================================================================
# MASK GENERATION
# =============================================================================

def fill_polygon(mask, vertices):
    """Fill polygon in mask using matplotlib Path."""
    path = Path(vertices)
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)
    mask[mask_inside] = 1


def generate_template_mask(height, width, prompt_centre, block, resolution, template_params):
    """Generate template mask using parameterized dimensions (matching p4tun behavior)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)
    
    if block == 'K':
        w = template_params['k_mask_width']
        hp = template_params['k_mask_height_pos']
        hn = template_params['k_mask_height_neg']
        vertices_real = np.array([[x-w, y-hp], [x-w, y+hp], [x+w, y+hn], [x+w, y-hn]])
    elif block == 'B1':
        w = template_params['ab_mask_width']
        ht = template_params['b1_height_top']
        hbp = template_params['b1_height_bottom_pos']
        hbn = template_params['b1_height_bottom_neg']
        vertices_real = np.array([[x-w, y-ht], [x-w, y+hbp], [x+w, y+hbn], [x+w, y-ht]])
    elif block == 'B2':
        w = template_params['ab_mask_width']
        htp = template_params['b2_height_top_pos']
        htn = template_params['b2_height_top_neg']
        hb = template_params['b2_height_bottom']
        vertices_real = np.array([[x-w, y-htp], [x-w, y+hb], [x+w, y+hb], [x+w, y-htn]])
    else:  # A blocks - rectangular
        w = template_params['ab_mask_width']
        h = template_params['ab_mask_height']
        vertices_real = np.array([[x-w, y-h], [x-w, y+h], [x+w, y+h], [x+w, y-h]])
        
    vertices = vertices_real / (resolution * 1000)
    fill_polygon(mask, vertices)
    return mask


def compute_logits_from_mask(mask, eps=FIXED_MASK_EPS):
    """Convert binary mask to logits for SAM input."""
    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

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
        logits = np.pad(logits, ((0, padh), (0, padw)), mode="constant", constant_values=0)

    logits = logits[None]
    return logits


def restore_sam_logits(logits, original_shape):
    """Restore SAM logits to original image shape."""
    orig_h, orig_w = original_shape
    trafo = ResizeLongestSide(max(orig_h, orig_w))
    resized_logits = trafo.apply_image(logits[..., None])
    resized_logits = resized_logits.squeeze()
    resized_logits = resized_logits[:orig_h, :orig_w]
    return resized_logits


# =============================================================================
# PROMPT POINT GENERATION (FIXED DEFAULTS)
# =============================================================================

def generate_prompt_points_k(x, y):
    """Generate K-block prompt points using FIXED parameters."""
    outer = FIXED_K_OUTER_RING
    middle = FIXED_K_MIDDLE_RING
    inner = FIXED_K_INNER_RING
    center = FIXED_K_CENTER_RING
    k_sp = FIXED_K_BLOCK_SPACING
    v_sp = FIXED_K_VERTICAL_SPACING
    
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


def generate_prompt_points_ab(x, y, block_type):
    """Generate A/B block prompt points using FIXED parameters."""
    outer = FIXED_AB_OUTER_RING
    middle = FIXED_AB_MIDDLE_RING
    inner = FIXED_AB_INNER_RING
    center = FIXED_AB_CENTER_RING
    fine = FIXED_AB_FINE_SPACING
    ultra = FIXED_AB_ULTRA_FINE
    edge = FIXED_AB_EDGE_RING
    edge_sp = FIXED_AB_EDGE_SPACING
    
    l = FIXED_AB_VERTICAL_LEVELS
    l1, l2, l3, l4 = l['level_1'], l['level_2'], l['level_3'], l['level_4']
    l5, l6, l7 = l['level_5'], l['level_6'], l['level_7']
    
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
                           segment_width, K_height, AB_height, image, y_bounds):
    """Generate prompt points for SAM."""
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)
    map_y_mm = map_y * (resolution * 1000)
    
    if block == 'K':
        points_real, labels = generate_prompt_points_k(x, y)
    else:
        points_real, labels = generate_prompt_points_ab(x, y, block)

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
        padding = 150
        within_bounds = (points[:, 0] >= 0) & ((points[:, 0] + initial_x - (segment_width*0.5+padding)/(resolution*1000)) <= image.shape[1])
        points = points[within_bounds]
        labels = labels[within_bounds]
        
    return points, labels


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_to_pixel_coords(real_dist, resolution=0.005):
    """Convert real distance (mm) to pixel coordinates."""
    return int(real_dist / (resolution * 1000))


def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution, template_params):
    """Crop image and generate template mask logits."""
    img_height, img_width, _ = image.shape
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, img_width)
    y2 = min(cy + crop_height // 2, img_height)

    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    prompt_centre_x = cx - x1
    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x, prompt_centre_y)
    
    cropped_template_mask = generate_template_mask(
        cropped_image.shape[0], cropped_image.shape[1], 
        prompt_centre, block, resolution, template_params
    )
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    return cropped_image, template_mask_logits, prompt_centre


def compute_block_label(segment_per_ring):
    """Get block labels for given segment count."""
    if segment_per_ring == 7:
        return ['K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']
    else:
        return ['K', 'B1', 'A1', 'A2', 'A3', 'B2']


def compute_block_to_label_map(segment_per_ring):
    """Get block name to numeric label mapping."""
    if segment_per_ring == 7:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
    else:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}


# =============================================================================
# ROW PROCESSING
# =============================================================================

def process_row(df_row, image, predictor, config):
    """Process a single row (ring) for SAM segmentation."""
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
    y_bounds = config['y_bounds']
    
    # Template params dict (passed to generate_template_mask)
    template_params = {
        'k_mask_width': config['k_mask_width'],
        'k_mask_height_pos': config['k_mask_height_pos'],
        'k_mask_height_neg': config['k_mask_height_neg'],
        'ab_mask_width': config['ab_mask_width'],
        'ab_mask_height': config['ab_mask_height'],
        'b1_height_top': config['b1_height_top'],
        'b1_height_bottom_pos': config['b1_height_bottom_pos'],
        'b1_height_bottom_neg': config['b1_height_bottom_neg'],
        'b2_height_top_pos': config['b2_height_top_pos'],
        'b2_height_top_neg': config['b2_height_top_neg'],
        'b2_height_bottom': config['b2_height_bottom'],
    }
    
    block_labels = compute_block_label(segment_per_ring)

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

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution, template_params)
            points, labels = generate_prompt_points(
                prompt_centre, initial_x, map_y, block, resolution,
                segment_width, K_height, AB_height, image, y_bounds)
        
            if len(points) > 0 and np.any(points[:, 1] < 0):
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

            cropped_image, template_mask_logit, prompt_centre = crop_image_and_mask_logits(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution, template_params)
            points, labels = generate_prompt_points(
                prompt_centre, initial_x, map_y, block, resolution,
                segment_width, K_height, AB_height, image, y_bounds)

            if len(points) > 0 and np.any((points[:, 1] + map_y - delta_y) > image.shape[0]):
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
    """Project segmentation results back to 3D point cloud."""
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
# MAIN PIPELINE
# =============================================================================

def run_sam(tunnel_id: str, base_dir: str = "data"):
    """
    Run SAM segmentation pipeline.
    
    CRITICAL PARAMETERS (9 tunable):
    - segment_width, angle_deg (geometry)
    - k_mask_width, k_mask_height, ab_mask_width, ab_mask_height (template masks)
    - padding, crop_margin (processing)
    - min_quality_threshold (quality weighting)
    
    INHERITED FROM PREPROCESSING:
    - resolution, tunnel_diameter (→ k_height_mm, ab_height_mm)
    """
    print(f"{'=' * 60}")
    print(f"SAM Segmentation Pipeline: {tunnel_id}")
    print(f"{'=' * 60}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    allow_defaults = not params_loaded
    
    # Load preprocessing parameters for inherited values
    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    
    # Resolution: prefer SAM params (matching original p4tun behavior), fallback to preprocessing
    preprocessing_resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    resolution = get_param(params, 'resolution', default=preprocessing_resolution, allow_default=True)
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    
    # K/AB heights: read from params if available, else calculate from tunnel diameter
    calc_K, calc_AB = calculate_segment_heights(tunnel_diameter)
    K_height = get_param(params, 'k_height', default=calc_K, allow_default=True)
    AB_height = get_param(params, 'ab_height', default=calc_AB, allow_default=True)
    
    # Extract CRITICAL tunable parameters
    segment_width = get_param(params, 'segment_width', default=DEFAULT_SEGMENT_WIDTH, allow_default=allow_defaults)
    angle_deg = get_param(params, 'angle_deg', default=DEFAULT_ANGLE_DEG, allow_default=allow_defaults)
    
    # Template mask parameters
    k_mask_width = get_param(params, 'k_mask_width', default=DEFAULT_K_MASK_WIDTH, allow_default=True)
    k_mask_height_pos = get_param(params, 'k_mask_height_pos', default=DEFAULT_K_MASK_HEIGHT_POS, allow_default=True)
    k_mask_height_neg = get_param(params, 'k_mask_height_neg', default=DEFAULT_K_MASK_HEIGHT_NEG, allow_default=True)
    ab_mask_width = get_param(params, 'ab_mask_width', default=DEFAULT_AB_MASK_WIDTH, allow_default=True)
    ab_mask_height = get_param(params, 'ab_mask_height', default=DEFAULT_AB_MASK_HEIGHT, allow_default=True)
    
    # B1/B2 separate mask heights (original p4tun behavior)
    b1_height_top = get_param(params, 'b1_height_top', default=DEFAULT_B1_HEIGHT_TOP, allow_default=True)
    b1_height_bottom_pos = get_param(params, 'b1_height_bottom_pos', default=DEFAULT_B1_HEIGHT_BOTTOM_POS, allow_default=True)
    b1_height_bottom_neg = get_param(params, 'b1_height_bottom_neg', default=DEFAULT_B1_HEIGHT_BOTTOM_NEG, allow_default=True)
    b2_height_top_pos = get_param(params, 'b2_height_top_pos', default=DEFAULT_B2_HEIGHT_TOP_POS, allow_default=True)
    b2_height_top_neg = get_param(params, 'b2_height_top_neg', default=DEFAULT_B2_HEIGHT_TOP_NEG, allow_default=True)
    b2_height_bottom = get_param(params, 'b2_height_bottom', default=DEFAULT_B2_HEIGHT_BOTTOM, allow_default=True)
    
    # Processing parameters
    padding = get_param(params, 'padding', default=DEFAULT_PADDING, allow_default=True)
    crop_margin = get_param(params, 'crop_margin', default=DEFAULT_CROP_MARGIN, allow_default=True)
    
    # Quality weighting
    min_quality_threshold = get_param(params, 'min_quality_threshold', default=DEFAULT_MIN_QUALITY_THRESHOLD, allow_default=True)
    use_quality_weighting = get_param(params, 'use_quality_weighting', default=DEFAULT_USE_QUALITY_WEIGHTING, allow_default=True)
    
    # Print parameters
    print(f"\nInherited from preprocessing:")
    print(f"  resolution:       {resolution}")
    print(f"  tunnel_diameter:  {tunnel_diameter}m")
    k_src = "from params" if 'k_height' in params else "calculated"
    print(f"  K_height:         {K_height:.2f}mm ({k_src})")
    print(f"  AB_height:        {AB_height:.2f}mm ({k_src})")
    
    print(f"\nCritical parameters (tunable):")
    print(f"  segment_width:    {segment_width}")
    print(f"  angle_deg:        {angle_deg}°")
    print(f"  k_mask_width:     {k_mask_width}")
    print(f"  k_mask_height:    +{k_mask_height_pos}/-{k_mask_height_neg}")
    print(f"  ab_mask_width:    {ab_mask_width}")
    print(f"  ab_mask_height:   {ab_mask_height}")
    print(f"  padding:          {padding}")
    print(f"  crop_margin:      {crop_margin}")
    print(f"  min_quality_threshold: {min_quality_threshold}")
    
    # Load input data
    print("\n[Step 1] Loading data...")
    detected_csv_path = os.path.join(tunnel_dir, "detected.csv")
    if not os.path.exists(detected_csv_path):
        raise FileNotFoundError(f"detected.csv not found in {tunnel_dir}. Run detection first.")
    
    initial_prompt_points = pd.read_csv(detected_csv_path)
    pixel_to_point = pickle.load(open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb"))
    df_point_cloud = pd.read_csv(os.path.join(tunnel_dir, "enhanced.csv"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    
    # Auto-detect segment count (default 6 for simple staggered)
    segment_per_ring = detect_segment_count(tunnel_dir, default=6)
    
    print(f"  Detected K positions: {len(initial_prompt_points)}")
    print(f"  Ring count: {ring_count}")
    print(f"  Segments per ring: {segment_per_ring} (auto-detected)")
    
    # Calculate y_bounds from image
    image = cv2.imread(os.path.join(tunnel_dir, 'depth_map.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    y_bounds = [int(image.shape[0] * 0.3 * resolution * 1000), int(image.shape[0] * 0.95 * resolution * 1000)]
    
    # Load SAM model
    print("\n[Step 2] Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=SAM_DEVICE)
    predictor = SamPredictor(sam)
    
    # Build config
    config = {
        'resolution': resolution,
        'segment_per_ring': segment_per_ring,
        'segment_width': segment_width,
        'K_height': K_height,
        'AB_height': AB_height,
        'angle': angle_deg,
        'padding': padding,
        'crop_margin': crop_margin,
        'y_bounds': y_bounds,
        'k_mask_width': k_mask_width,
        'k_mask_height_pos': k_mask_height_pos,
        'k_mask_height_neg': k_mask_height_neg,
        'ab_mask_width': ab_mask_width,
        'ab_mask_height': ab_mask_height,
        'b1_height_top': b1_height_top,
        'b1_height_bottom_pos': b1_height_bottom_pos,
        'b1_height_bottom_neg': b1_height_bottom_neg,
        'b2_height_top_pos': b2_height_top_pos,
        'b2_height_top_neg': b2_height_top_neg,
        'b2_height_bottom': b2_height_bottom,
    }
    
    # Run SAM on each detected K position
    print("\n[Step 3] Running SAM segmentation...")
    all_results = []
    for _, row in tqdm(initial_prompt_points.iterrows(), total=len(initial_prompt_points), desc="Processing rings"):
        result = process_row(row, image, predictor, config)
        all_results.append(result)
    
    # Aggregate results
    print("\n[Step 4] Aggregating results...")
    logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
    label_map = np.zeros(image.shape[:2], dtype=int)
    ring_map = np.zeros(image.shape[:2], dtype=int)
    
    block_to_label = compute_block_to_label_map(segment_per_ring)

    for ring_index, ring in enumerate(all_results, start=0):
        for item in ring:
            mask = item['mask'][0]
            logits = item['logit']
            block = item['block']
            start_x, start_y = map(int, item['left_top'])
            quality = item.get('quality', 1.0)

            end_y, end_x = start_y + mask.shape[0], start_x + mask.shape[1]
            start_y, start_x = max(0, start_y), max(0, start_x)
            end_y, end_x = min(image.shape[0], end_y), min(image.shape[1], end_x)
            
            valid_slice_y = slice(start_y, end_y)
            valid_slice_x = slice(start_x, end_x)

            new_logits = restore_sam_logits(logits, mask.shape)
            
            if use_quality_weighting and quality >= min_quality_threshold:
                new_logits = new_logits * quality
            elif quality < min_quality_threshold:
                continue
            
            current_logits = logits_map[valid_slice_y, valid_slice_x]

            if mask.shape != current_logits.shape:
                continue
            if new_logits.shape != current_logits.shape:
                new_logits = new_logits[:current_logits.shape[0], :current_logits.shape[1]]

            update_mask = (new_logits > current_logits) & mask
            
            logits_map[valid_slice_y, valid_slice_x][update_mask] = new_logits[update_mask]
            label_map[valid_slice_y, valid_slice_x][update_mask] = block_to_label.get(block, 0)
            ring_map[valid_slice_y, valid_slice_x][update_mask] = ring_index

    # Fix ring numbering
    fix_ring = np.where((ring_map >= 1) & (ring_map <= (ring_count-1)), ring_count - ring_map, ring_map)
    
    # Project back to point cloud
    print("\n[Step 5] Projecting to point cloud...")
    updated_df = project_back_to_point_cloud(label_map, fix_ring, pixel_to_point, df_point_cloud)
    
    # Save results
    updated_df.to_csv(os.path.join(tunnel_dir, 'final.csv'), index=False)
    
    if 'segment' in updated_df.columns:
        df_pred = pd.DataFrame()
        df_pred['gt_labels'] = updated_df['segment']
        df_pred['gt_rings'] = updated_df['ring']
        df_pred['pred_labels'] = updated_df['pred']
        df_pred['pred_rings'] = updated_df['pred_ring']
        df_pred.to_csv(os.path.join(tunnel_dir, 'only_label.csv'), index=False)
    
    # Summary
    segment_counts = updated_df['pred'].value_counts()
    print(f"\n{'=' * 60}")
    print(f"SAM segmentation complete!")
    print(f"{'=' * 60}")
    print(f"  Output: {tunnel_dir}/final.csv")
    print(f"  Total points: {len(updated_df)}")
    print(f"  Segments found: {len(segment_counts[segment_counts.index > 0])}")
    
    return updated_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAM-based tunnel segmentation (simplified for BO)")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4, 2-2)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_sam(args.tunnel_id, base_dir=args.data_dir)
