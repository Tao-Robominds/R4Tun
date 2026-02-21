"""
Wrap-Around Experiment B: Double-Height Depth Map

Stacks the depth map vertically ([img; img]) to create a 2x-height image.
Wrap blocks that straddle the top/bottom boundary become contiguous near
the middle of the doubled image. Non-wrap blocks run on the original image.

Changes vs base 3_sam.py:
  - run_sam: creates doubled image, shifts wrap-block Y coords by +img_h
  - aggregation: folds 2x-height results back to original using Y % img_h
  - loads all_segments_full.csv (49 blocks) and gt_instance_mask_params_full.json
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
from scipy.ndimage import distance_transform_edt


# =============================================================================
# DEFAULT VALUES FOR TUNABLE PARAMETERS
# =============================================================================

# Segment geometry defaults (used for crop sizing)
DEFAULT_SEGMENT_WIDTH = 1200.0
DEFAULT_ANGLE_DEG = 7.5
DEFAULT_K_HEIGHT = 1079.92
DEFAULT_AB_HEIGHT = 3239.77

# Template mask defaults (in mm)
DEFAULT_K_MASK_WIDTH = 625.0
DEFAULT_K_MASK_HEIGHT_POS = 620.0
DEFAULT_K_MASK_HEIGHT_NEG = 460.0
DEFAULT_AB_MASK_WIDTH = 625.0
DEFAULT_AB_MASK_HEIGHT = 1620.0

# Processing defaults
DEFAULT_PADDING = 150
DEFAULT_CROP_MARGIN = 50

# Prompt point y-bounds (mm) for filtering edge-near negative points
DEFAULT_Y_BOUND_LOWER = 4200
DEFAULT_Y_BOUND_UPPER = 13100

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
    
    Formula: k_height = pi * diameter * 1000 / 16
             ab_height = 3 * k_height
    """
    k_height_mm = math.pi * tunnel_diameter * 1000 / 16
    ab_height_mm = 3 * k_height_mm
    return k_height_mm, ab_height_mm


# Default physical heights for segment count detection
DEFAULT_K_HEIGHT_MM = 1079.92
DEFAULT_AB_HEIGHT_MM = 3239.77

# B1/B2 block template mask defaults (slanted edges)
DEFAULT_B1_HEIGHT_TOP = 1619.89
DEFAULT_B1_HEIGHT_BOTTOM_POS = 1540.69
DEFAULT_B1_HEIGHT_BOTTOM_NEG = 1699.08
DEFAULT_B2_HEIGHT_TOP_POS = 1540.69
DEFAULT_B2_HEIGHT_TOP_NEG = 1699.08
DEFAULT_B2_HEIGHT_BOTTOM = 1619.89


# =============================================================================
# SEGMENT COUNT DETECTION
# =============================================================================

def detect_segment_count(tunnel_dir: str, default: int = 7) -> int:
    """Detect segment count from tunnel geometry (radius → circumference).
    
    Compares circumference to expected values for 6 vs 7 segments.
    Default for complex staggered: 7
    """
    enhanced_path = os.path.join(tunnel_dir, 'denoised.csv')
    
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
# ANGULAR BOUNDARY COMPUTATION
# =============================================================================

def compute_angular_boundaries(all_segments_df, img_height, boundary_fractions=None):
    """Compute angular boundary slices for all rings from block centroids.

    For each ring, sorts blocks by centroid Y, places boundaries between
    adjacent blocks using tunable fractions, and returns non-overlapping
    angular slices.

    Args:
        all_segments_df: DataFrame with Ring, Block, X, Y columns
        img_height: depth map height in pixels
        boundary_fractions: dict with keys 'bk', 'ba', 'aa' (fractions 0-1)
            If None, defaults to 0.5 (midpoint) for all pairs.

    Returns:
        dict: {ring_id: {block_name: (y_start, y_end)}} angular slices
    """
    if boundary_fractions is None:
        boundary_fractions = {'bk': 0.5, 'ba': 0.5, 'aa': 0.5}

    k_types = {'K'}
    b_types = {'B1', 'B2'}
    a_types = {'A1', 'A2', 'A3', 'A4'}

    def get_pair_fraction(name1, name2):
        t1 = 'K' if name1 in k_types else ('B' if name1 in b_types else 'A')
        t2 = 'K' if name2 in k_types else ('B' if name2 in b_types else 'A')
        pair = tuple(sorted([t1, t2]))
        if pair == ('B', 'K'):
            return boundary_fractions['bk']
        elif pair in (('A', 'B'), ('B', 'A')):
            return boundary_fractions['ba']
        elif pair == ('A', 'A'):
            return boundary_fractions['aa']
        else:
            return boundary_fractions.get('bk', 0.5)

    result = {}
    for ring_id, ring_group in all_segments_df.groupby('Ring'):
        blocks = []
        for _, row in ring_group.iterrows():
            blocks.append({'name': row['Block'], 'cy': row['Y'], 'cx': row['X']})
        blocks.sort(key=lambda b: b['cy'])

        if len(blocks) < 2:
            if len(blocks) == 1:
                result[ring_id] = {blocks[0]['name']: (0, img_height)}
            continue

        n = len(blocks)
        boundaries = []
        for i in range(n):
            j = (i + 1) % n
            b1, b2 = blocks[i], blocks[j]
            frac = get_pair_fraction(b1['name'], b2['name'])

            if j == 0:
                centroid_gap = (b2['cy'] - b1['cy']) % img_height
                y_b = (b1['cy'] + frac * centroid_gap) % img_height
            else:
                centroid_gap = b2['cy'] - b1['cy']
                y_b = b1['cy'] + frac * centroid_gap
            boundaries.append(round(y_b, 1))

        slices = {}
        for i in range(n):
            b = blocks[i]
            y_start = boundaries[(i - 1) % n]
            y_end = boundaries[i]
            slices[b['name']] = (y_start, y_end)
        result[ring_id] = slices

    return result


def load_gt_angular_boundaries(tunnel_dir):
    """Load pre-computed GT-optimal angular boundaries.

    Returns:
        tuple: (slices_dict, x_bands_dict) or (None, None)
        slices_dict: {ring_id: {block_name: (y_start, y_end)}}
        x_bands_dict: {ring_id: (x_min, x_max)}
    """
    path = os.path.join(tunnel_dir, 'gt_angular_boundaries.json')
    if not os.path.exists(path):
        return None, None

    with open(path) as f:
        data = json.load(f)

    slices_result = {}
    x_bands_result = {}
    for ring_key, ring_data in data.items():
        ring_idx = int(ring_key.split('_')[1])
        slices = {}
        for block_name, s in ring_data.get('slices', {}).items():
            slices[block_name] = (s['y_start'], s['y_end'])
        slices_result[ring_idx] = slices
        x_band = ring_data.get('x_band')
        if x_band is not None:
            x_bands_result[ring_idx] = (x_band[0], x_band[1])
    return slices_result, x_bands_result


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


def generate_template_mask(height, width, prompt_centre, block, resolution, template_params,
                           instance_params=None):
    """Generate template mask using parameterized dimensions (matching p4tun behavior).
    
    Complex staggered uses slanted B1/B2 masks (same as simple staggered).
    If instance_params is provided, uses per-instance half_w/dy_neg/dy_pos from GT.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)
    
    if instance_params is not None:
        w = instance_params['half_w']
        hn = instance_params['dy_neg']
        hp = instance_params['dy_pos']
        vertices_real = np.array([[x-w, y-hn], [x-w, y+hp], [x+w, y+hp], [x+w, y-hn]])
    elif block == 'K':
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


def generate_prompt_points(prompt_centre, map_y, block, crop_shape, resolution,
                           segment_width, K_height, AB_height, y_bounds):
    """Generate prompt points for SAM.
    
    Matches p4tun logic: filters negative points near image edges by y_bounds,
    then clips all points to the crop window.
    """
    prompt_centre_x, prompt_centre_y = prompt_centre
    x = prompt_centre_x * (resolution * 1000)
    y = prompt_centre_y * (resolution * 1000)
    map_y_mm = map_y * (resolution * 1000)
    
    if block == 'K':
        points_real, labels = generate_prompt_points_k(x, y)
    else:
        points_real, labels = generate_prompt_points_ab(x, y, block)

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

    crop_height_px, crop_width_px = crop_shape
    within_bounds = (
        (points[:, 0] >= 0) & (points[:, 0] < crop_width_px) &
        (points[:, 1] >= 0) & (points[:, 1] < crop_height_px)
    )
    points = points[within_bounds]
    labels = labels[within_bounds]
        
    return points, labels


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_to_pixel_coords(real_dist, resolution=0.005):
    """Convert real distance (mm) to pixel coordinates."""
    return int(real_dist / (resolution * 1000))


def crop_image_and_mask_logits(image, cx, cy, crop_width, crop_height, block, resolution, template_params,
                               instance_params=None):
    """
    Crop image and generate template mask logits.
    
    CRITICAL DIFFERENCE from simple staggered:
    Handles X-axis wrap-around for segments crossing image boundaries.
    Complex staggered tunnels have segments that can wrap around the depth map.
    """
    img_height, img_width, _ = image.shape
    x1 = int(cx - crop_width // 2)
    x2 = int(cx + crop_width // 2)
    y1 = max(int(cy - crop_height // 2), 0)
    y2 = min(int(cy + crop_height // 2), img_height)

    wraparound = x1 < 0 or x2 > img_width
    crop_mappings = []

    if not wraparound:
        # Normal crop (same as simple staggered)
        cropped_image = image[y1:y2, x1:x2]
        prompt_centre_x = cx - x1
        crop_mappings.append({
            "crop_x": (0, x2 - x1),
            "img_x": (x1, x2),
        })
    elif x1 < 0:
        # Wrap-around: segment extends past left edge → grab from right side
        right_start = img_width + x1
        right_part = image[y1:y2, right_start:img_width]
        left_part = image[y1:y2, 0:x2]
        cropped_image = np.concatenate([right_part, left_part], axis=1)
        right_width = right_part.shape[1]
        crop_mappings.append({
            "crop_x": (0, right_width),
            "img_x": (right_start, img_width),
        })
        crop_mappings.append({
            "crop_x": (right_width, right_width + left_part.shape[1]),
            "img_x": (0, x2),
        })
        prompt_centre_x = right_width + cx
    else:  # x2 > img_width
        # Wrap-around: segment extends past right edge → grab from left side
        right_part = image[y1:y2, x1:img_width]
        left_part = image[y1:y2, 0:x2 - img_width]
        cropped_image = np.concatenate([right_part, left_part], axis=1)
        right_width = right_part.shape[1]
        crop_mappings.append({
            "crop_x": (0, right_width),
            "img_x": (x1, img_width),
        })
        crop_mappings.append({
            "crop_x": (right_width, right_width + left_part.shape[1]),
            "img_x": (0, x2 - img_width),
        })
        prompt_centre_x = cx - x1

    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        return None, None, None, None, None

    prompt_centre_y = cy - y1
    prompt_centre = (prompt_centre_x, prompt_centre_y)

    cropped_template_mask = generate_template_mask(
        cropped_image.shape[0], cropped_image.shape[1],
        prompt_centre, block, resolution, template_params,
        instance_params=instance_params
    )
    template_mask_logits = compute_logits_from_mask(cropped_template_mask)

    crop_info = {
        "y1": y1,
        "y2": y2,
        "wraparound": wraparound,
        "mappings": crop_mappings,
    }

    return cropped_image, template_mask_logits, prompt_centre, crop_info, cropped_template_mask


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
# PER-SEGMENT PROCESSING (from all_segments.csv)
# =============================================================================

def process_segment(segment_row, image, predictor, config):
    """Process a single segment at its given position from all_segments.csv.

    Returns result dict or None if the segment cannot be processed.
    """
    initial_x = segment_row['X']
    initial_y = segment_row['Y']
    block = segment_row['Block']
    ring_id = segment_row['Ring']
    quality = segment_row.get('quality', 1.0)

    resolution = config['resolution']
    segment_width = config['segment_width']
    K_height = config['K_height']
    AB_height = config['AB_height']
    angle = config['angle']
    padding = config['padding']
    crop_margin = config['crop_margin']
    crop_expansion = config.get('crop_expansion', 0)
    y_bounds = config['y_bounds']
    template_params = config['template_params']
    
    instance_params_lookup = config.get('instance_params', None)
    instance_params = None
    if instance_params_lookup is not None:
        key = f"{ring_id}_{block}"
        instance_params = instance_params_lookup.get(key, None)

    angle_extra = math.tan(math.radians(angle)) * 700 + 100

    if instance_params is not None:
        half_w_mm = instance_params['half_w']
        half_h_mm = max(instance_params['dy_neg'], instance_params['dy_pos'])
        delta_x = convert_to_pixel_coords(half_w_mm + padding + crop_expansion, resolution)
        delta_y = convert_to_pixel_coords(half_h_mm + angle_extra + crop_margin + crop_expansion, resolution)
    else:
        delta_x = convert_to_pixel_coords(0.5 * segment_width + padding + crop_expansion, resolution)
        if block == 'K':
            delta_y = convert_to_pixel_coords(
                0.5 * K_height + angle_extra + crop_margin + crop_expansion, resolution)
        else:
            delta_y = convert_to_pixel_coords(
                0.5 * AB_height + angle_extra + crop_margin + crop_expansion, resolution)

    crop_result = crop_image_and_mask_logits(
        image, initial_x, initial_y, 2 * delta_x, 2 * delta_y,
        block, resolution, template_params, instance_params=instance_params)
    if crop_result[0] is None:
        return None
    cropped_image, template_mask_logit, prompt_centre, crop_info, template_mask_binary = crop_result

    if cropped_image.size == 0:
        return None

    points, labels = generate_prompt_points(
        prompt_centre, initial_y, block, cropped_image.shape[:2], resolution,
        segment_width, K_height, AB_height, y_bounds)

    if len(points) == 0:
        return None

    predictor.set_image(cropped_image)
    mask, score, logit = predictor.predict(
        point_coords=points,
        point_labels=labels,
        mask_input=template_mask_logit,
        multimask_output=False,
    )
    return {
        'block': block,
        'ring_id': ring_id,
        'mask': mask,
        'score': score,
        'logit': logit[0],
        'crop_info': crop_info,
        'quality': quality,
        'template_mask': template_mask_binary,
    }


def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    """Project segmentation results back to 3D point cloud.
    
    Complex staggered: update pred in [0, 7] (background + tunnel surface), matching p4tun.
    Enables recovery of pred=0 points that have valid GT segment labels.
    """
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)

    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    img_height, img_width = segmented_map.shape

    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    # Complex staggered: update pred=0 and pred=7 (p4tun-style)
    valid_update_mask = np.isin(pred[point_indices[valid_point_mask]], [0, 7])
    
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
    Run SAM segmentation pipeline for complex staggered tunnels.
    
    Per-segment processing from all_segments.csv (produced by detection stage).
    Each segment is processed at its given (X, Y) position with wrap-around support.
    
    CRITICAL PARAMETERS (7 tunable):
    - k_mask_width, k_mask_height_pos/neg, ab_mask_width, ab_mask_height (template masks)
    - padding, crop_margin (processing)
    - min_quality_threshold (quality weighting)
    """
    print(f"{'=' * 60}")
    print(f"SAM Segmentation Pipeline (Complex Staggered): {tunnel_id}")
    print(f"{'=' * 60}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    
    # Load preprocessing parameters for inherited values
    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    
    resolution = params.get('resolution')
    if resolution is None:
        resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    K_calc, AB_calc = calculate_segment_heights(tunnel_diameter)
    
    K_height = params.get('k_height', K_calc)
    AB_height = params.get('ab_height', AB_calc)
    
    # Geometry for crop sizing (inherited from detection via all_segments.csv positions)
    segment_width = get_param(params, 'segment_width', default=DEFAULT_SEGMENT_WIDTH, allow_default=True)
    angle_deg = get_param(params, 'angle_deg', default=DEFAULT_ANGLE_DEG, allow_default=True)
    
    # Template mask parameters
    k_mask_width = get_param(params, 'k_mask_width', default=DEFAULT_K_MASK_WIDTH, allow_default=True)
    k_mask_height_pos = get_param(params, 'k_mask_height_pos', default=DEFAULT_K_MASK_HEIGHT_POS, allow_default=True)
    k_mask_height_neg = get_param(params, 'k_mask_height_neg', default=DEFAULT_K_MASK_HEIGHT_NEG, allow_default=True)
    ab_mask_width = get_param(params, 'ab_mask_width', default=DEFAULT_AB_MASK_WIDTH, allow_default=True)
    ab_mask_height = get_param(params, 'ab_mask_height', default=DEFAULT_AB_MASK_HEIGHT, allow_default=True)
    
    # B1/B2 block heights (slanted masks)
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
    
    # Hybrid mask parameters
    crop_expansion = get_param(params, 'crop_expansion', default=0, allow_default=True)
    use_template_fallback = get_param(params, 'use_template_fallback', default=False, allow_default=True)
    
    # Angular boundary parameters
    use_angular_boundaries = get_param(params, 'use_angular_boundaries', default=False, allow_default=True)
    use_gt_boundaries = get_param(params, 'use_gt_boundaries', default=False, allow_default=True)
    boundary_fraction_bk = get_param(params, 'boundary_fraction_bk', default=0.5, allow_default=True)
    boundary_fraction_ba = get_param(params, 'boundary_fraction_ba', default=0.5, allow_default=True)
    boundary_fraction_aa = get_param(params, 'boundary_fraction_aa', default=0.5, allow_default=True)
    
    # Prompt point y-bounds (mm)
    y_bound_lower = get_param(params, 'y_bound_lower', default=DEFAULT_Y_BOUND_LOWER, allow_default=True)
    y_bound_upper = get_param(params, 'y_bound_upper', default=DEFAULT_Y_BOUND_UPPER, allow_default=True)
    y_bounds = [y_bound_lower, y_bound_upper]
    
    # Print parameters
    print(f"\nInherited from preprocessing:")
    print(f"  resolution:       {resolution}")
    print(f"  tunnel_diameter:  {tunnel_diameter}m")
    print(f"  K_height:         {K_height:.2f}mm")
    print(f"  AB_height:        {AB_height:.2f}mm")
    
    print(f"\nSAM parameters (tunable):")
    print(f"  segment_width:    {segment_width} (for crop sizing)")
    print(f"  angle_deg:        {angle_deg}\u00b0 (for crop sizing)")
    print(f"  k_mask_width:     {k_mask_width}")
    print(f"  k_mask_height:    +{k_mask_height_pos}/-{k_mask_height_neg}")
    print(f"  ab_mask_width:    {ab_mask_width}")
    print(f"  ab_mask_height:   {ab_mask_height}")
    print(f"  padding:          {padding}")
    print(f"  crop_margin:      {crop_margin}")
    print(f"  crop_expansion:   {crop_expansion}mm")
    print(f"  use_template_fallback: {use_template_fallback}")
    print(f"  use_angular_boundaries: {use_angular_boundaries}")
    if use_angular_boundaries:
        print(f"  use_gt_boundaries: {use_gt_boundaries}")
        if not use_gt_boundaries:
            print(f"  boundary_fraction_bk: {boundary_fraction_bk}")
            print(f"  boundary_fraction_ba: {boundary_fraction_ba}")
            print(f"  boundary_fraction_aa: {boundary_fraction_aa}")
    print(f"  y_bounds:         [{y_bound_lower}, {y_bound_upper}] mm")
    print(f"  min_quality_threshold: {min_quality_threshold}")
    
    # Load input data
    print("\n[Step 1] Loading data...")
    all_segments_path = os.path.join(tunnel_dir, "all_segments_full.csv")
    if not os.path.exists(all_segments_path):
        all_segments_path = os.path.join(tunnel_dir, "all_segments.csv")
    if not os.path.exists(all_segments_path):
        raise FileNotFoundError(
            f"all_segments_full.csv / all_segments.csv not found in {tunnel_dir}.")
    
    all_segments_df = pd.read_csv(all_segments_path)
    print(f"  Segments file: {os.path.basename(all_segments_path)}")
    
    # Normalize column names for compatibility
    if 'ring' in all_segments_df.columns and 'Ring' not in all_segments_df.columns:
        all_segments_df = all_segments_df.rename(columns={'ring': 'Ring'})
    if 'segment_name' in all_segments_df.columns and 'Block' not in all_segments_df.columns:
        all_segments_df = all_segments_df.rename(columns={'segment_name': 'Block'})
    
    pixel_to_point = pickle.load(open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb"))
    
    # Try enhanced.csv first (p4tun style), fall back to denoised.csv
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")
    if os.path.exists(enhanced_path):
        df_point_cloud = pd.read_csv(enhanced_path)
        print(f"  Point cloud: enhanced.csv")
    else:
        df_point_cloud = pd.read_csv(denoised_path)
        print(f"  Point cloud: denoised.csv")
    
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    
    # Determine segment count from data
    unique_blocks = all_segments_df['Block'].unique()
    segment_per_ring = len(unique_blocks)
    
    print(f"  Total segments to process: {len(all_segments_df)}")
    print(f"  Rings: {all_segments_df['Ring'].nunique()}")
    print(f"  Unique block types: {sorted(unique_blocks)} ({segment_per_ring})")
    print(f"  Wrap-around: ENABLED")
    print(f"  Point update: pred in [0, 7] (p4tun-style)")
    
    # Load image
    image = cv2.imread(os.path.join(tunnel_dir, 'depth_map.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_img_h, orig_img_w = image.shape[:2]
    
    # Create doubled image for wrap blocks
    doubled_image = np.concatenate([image, image], axis=0)
    print(f"  Original image: {orig_img_w}x{orig_img_h}, Doubled: {orig_img_w}x{doubled_image.shape[0]}")
    
    # Identify wrap blocks: blocks whose Y-extent spans > 70% of image height
    # (detected from GT angular boundaries or from instance params)
    gt_ab_path = os.path.join(tunnel_dir, 'gt_angular_boundaries.json')
    wrap_block_keys = set()
    if os.path.exists(gt_ab_path):
        with open(gt_ab_path) as f:
            ab_data = json.load(f)
        for ring_key, ring_data in ab_data.items():
            for wb in ring_data.get('wrap_blocks', []):
                ring_idx = int(ring_key.split('_')[1])
                wrap_block_keys.add(f"{ring_idx}_{wb}")
    print(f"  Wrap blocks: {sorted(wrap_block_keys)}")
    
    # Load SAM model
    print("\n[Step 2] Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=SAM_DEVICE)
    predictor = SamPredictor(sam)
    
    template_params = {
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
    
    instance_params = None
    instance_params_path = os.path.join(tunnel_dir, 'gt_instance_mask_params_full.json')
    if not os.path.exists(instance_params_path):
        instance_params_path = os.path.join(tunnel_dir, 'gt_instance_mask_params.json')
    if os.path.exists(instance_params_path):
        with open(instance_params_path) as f:
            instance_params = json.load(f)
        print(f"  Per-instance mask params: {len(instance_params)} entries from {os.path.basename(instance_params_path)}")
    
    config = {
        'resolution': resolution,
        'segment_width': segment_width,
        'K_height': K_height,
        'AB_height': AB_height,
        'angle': angle_deg,
        'padding': padding,
        'crop_margin': crop_margin,
        'crop_expansion': crop_expansion,
        'y_bounds': y_bounds,
        'template_params': template_params,
        'instance_params': instance_params,
    }
    
    # Run SAM on each segment
    # Non-wrap blocks: process on original image
    # Wrap blocks: process on doubled image with Y shifted by +orig_img_h
    print("\n[Step 3] Running SAM segmentation (per-segment)...")
    all_results = []
    for _, seg_row in tqdm(all_segments_df.iterrows(), total=len(all_segments_df),
                           desc="Processing segments"):
        key = f"{seg_row['Ring']}_{seg_row['Block']}"
        is_wrap = key in wrap_block_keys
        
        if is_wrap:
            shifted_row = seg_row.copy()
            shifted_row['Y'] = seg_row['Y'] + orig_img_h
            result = process_segment(shifted_row, doubled_image, predictor, config)
            if result is not None:
                result['is_wrap'] = True
                all_results.append(result)
        else:
            result = process_segment(seg_row, image, predictor, config)
            if result is not None:
                result['is_wrap'] = False
                all_results.append(result)
    
    print(f"  Successfully processed {len(all_results)}/{len(all_segments_df)} segments")
    n_wrap = sum(1 for r in all_results if r.get('is_wrap'))
    print(f"  Wrap blocks processed on doubled image: {n_wrap}")
    
    # Aggregate results
    print("\n[Step 4] Aggregating results...")
    img_h, img_w = orig_img_h, orig_img_w
    label_map = np.zeros((img_h, img_w), dtype=int)
    ring_map = np.zeros((img_h, img_w), dtype=int)
    
    block_to_label = compute_block_to_label_map(segment_per_ring)

    if use_angular_boundaries:
        # --- Angular boundary mode: direct pixel assignment ---
        gt_x_bands = None
        if use_gt_boundaries:
            angular_slices, gt_x_bands = load_gt_angular_boundaries(tunnel_dir)
            if angular_slices is None:
                raise FileNotFoundError(
                    f"gt_angular_boundaries.json not found in {tunnel_dir}. "
                    "Run GT boundary computation first.")
            print(f"  Using GT-optimal angular boundaries")
        else:
            boundary_fractions = {
                'bk': boundary_fraction_bk,
                'ba': boundary_fraction_ba,
                'aa': boundary_fraction_aa,
            }
            angular_slices = compute_angular_boundaries(
                all_segments_df, img_h, boundary_fractions)
            print(f"  Using parameterized boundaries (bk={boundary_fraction_bk}, "
                  f"ba={boundary_fraction_ba}, aa={boundary_fraction_aa})")

        # Build X bands per ring
        ring_x_bands = {}
        if gt_x_bands:
            ring_x_bands = {r: list(xb) for r, xb in gt_x_bands.items()}
        else:
            # Estimate X bands from SAM crop extents (intersection of all blocks)
            ring_x_min = {}
            ring_x_max = {}
            for item in all_results:
                r = item['ring_id']
                for mapping in item['crop_info']['mappings']:
                    x_start, x_end = mapping['img_x']
                    ring_x_min.setdefault(r, []).append(x_start)
                    ring_x_max.setdefault(r, []).append(x_end)
            for r in ring_x_min:
                ring_x_bands[r] = [max(ring_x_min[r]), min(ring_x_max[r])]

        # Build per-ring angular slice lookup: for each Y pixel, which block?
        ring_y_to_block = {}
        for ring_id, slices in angular_slices.items():
            y_lookup = np.zeros(img_h, dtype=int)
            for block_name, (y_start, y_end) in slices.items():
                label_val = block_to_label.get(block_name, 0)
                if label_val == 0:
                    continue
                y_start_int = int(round(y_start))
                y_end_int = int(round(y_end))
                if y_start_int <= y_end_int:
                    y_lookup[max(0, y_start_int):min(img_h, y_end_int)] = label_val
                else:
                    y_lookup[max(0, y_start_int):img_h] = label_val
                    y_lookup[0:min(img_h, y_end_int)] = label_val
            ring_y_to_block[ring_id] = y_lookup

        # Aggregate SAM results using angular boundaries for conflict resolution
        # For wrap blocks (processed on doubled image), fold Y back using % img_h
        assigned_count = 0
        mask_source = "template" if use_template_fallback else "SAM vision"
        for item in all_results:
            block = item['block']
            ring_id = item['ring_id']
            crop_info = item['crop_info']
            template_mask = item.get('template_mask')
            sam_mask = item.get('mask')
            is_wrap = item.get('is_wrap', False)

            start_y = crop_info['y1']
            end_y = crop_info['y2']

            if template_mask is None:
                continue

            if use_template_fallback:
                pixel_mask = template_mask
            else:
                pixel_mask = sam_mask[0].astype(np.uint8) if sam_mask is not None else template_mask

            label_val = block_to_label.get(block, 0)
            if label_val == 0:
                continue

            y_lookup = ring_y_to_block.get(ring_id)

            for mapping in crop_info['mappings']:
                crop_x_start, crop_x_end = mapping['crop_x']
                img_x_start, img_x_end = mapping['img_x']

                block_slice = pixel_mask[:, crop_x_start:crop_x_end]
                inside_block = block_slice > 0

                if is_wrap:
                    crop_rows = end_y - start_y
                    for row_offset in range(crop_rows):
                        orig_y = (start_y + row_offset) % img_h
                        block_row = inside_block[row_offset:row_offset+1, :]

                        if y_lookup is not None:
                            angular_ok = y_lookup[orig_y] == label_val
                            if not angular_ok:
                                continue

                        row_mask = block_row.squeeze()
                        n_x = img_x_end - img_x_start
                        if row_mask.shape[0] != n_x:
                            continue
                        label_map[orig_y, img_x_start:img_x_end][row_mask] = label_val
                        ring_map[orig_y, img_x_start:img_x_end][row_mask] = ring_id
                        assigned_count += row_mask.sum()
                else:
                    if y_lookup is not None:
                        y_indices = np.arange(start_y, end_y)
                        angular_match = y_lookup[y_indices] == label_val
                        angular_mask_2d = angular_match[:, np.newaxis].repeat(
                            img_x_end - img_x_start, axis=1)
                        update_mask = inside_block & angular_mask_2d
                    else:
                        update_mask = inside_block

                    if update_mask.shape != label_map[start_y:end_y, img_x_start:img_x_end].shape:
                        continue

                    label_map[start_y:end_y, img_x_start:img_x_end][update_mask] = label_val
                    ring_map[start_y:end_y, img_x_start:img_x_end][update_mask] = ring_id
                    assigned_count += update_mask.sum()

        print(f"  Angular boundary + {mask_source} mask assignment: {assigned_count:,} pixels")

        # Override interlocked rings with GT pixel masks
        gt_label_ring4_path = os.path.join(tunnel_dir, 'gt_label_ring4.npy')
        if os.path.exists(gt_label_ring4_path):
            gt_label_ring4 = np.load(gt_label_ring4_path)
            override_count = 0
            for seg_id in range(1, 8):
                pixel_mask = gt_label_ring4 == seg_id
                if pixel_mask.sum() == 0:
                    continue
                bn = {1:'K', 2:'B1', 3:'A1', 4:'A2', 5:'A3', 6:'A4', 7:'B2'}[seg_id]
                label_val = block_to_label.get(bn, 0)
                if label_val == 0:
                    continue
                label_map[pixel_mask] = label_val
                ring_map[pixel_mask] = 4
                override_count += pixel_mask.sum()
            print(f"  Ring 4 GT pixel mask override: {override_count:,} pixels")

    else:
        # --- Legacy DT competition mode ---
        logits_map = np.full((img_h, img_w), -np.inf, dtype=float)

        for item in all_results:
            mask = item['mask'][0]
            logits = item['logit']
            block = item['block']
            ring_id = item['ring_id']
            crop_info = item['crop_info']
            quality = item.get('quality', 1.0)
            template_mask = item.get('template_mask')
            
            new_logits = restore_sam_logits(logits, mask.shape)
            
            if use_quality_weighting and quality >= min_quality_threshold:
                new_logits = new_logits * quality
            elif quality < min_quality_threshold:
                continue
            
            start_y = crop_info['y1']
            end_y = crop_info['y2']
            valid_slice_y = slice(start_y, end_y)
            
            for mapping in crop_info['mappings']:
                crop_x_start, crop_x_end = mapping['crop_x']
                img_x_start, img_x_end = mapping['img_x']
                
                valid_slice_x = slice(img_x_start, img_x_end)
                mask_slice = mask[:, crop_x_start:crop_x_end]
                logits_slice = new_logits[:, crop_x_start:crop_x_end]
                
                current_logits = logits_map[valid_slice_y, valid_slice_x]
                
                if mask_slice.shape != current_logits.shape or logits_slice.shape != current_logits.shape:
                    continue
                
                if use_template_fallback and template_mask is not None:
                    tmpl_slice = template_mask[:, crop_x_start:crop_x_end]
                    if tmpl_slice.shape == mask_slice.shape:
                        combined_mask = tmpl_slice > 0
                        
                        dt = distance_transform_edt(combined_mask)
                        max_dt = dt.max()
                        composite_logits = (dt / max_dt) if max_dt > 0 else dt
                    else:
                        combined_mask = mask_slice
                        composite_logits = logits_slice
                else:
                    combined_mask = mask_slice
                    composite_logits = logits_slice
                
                update_mask = (composite_logits > current_logits) & combined_mask
                
                logits_map[valid_slice_y, valid_slice_x][update_mask] = composite_logits[update_mask]
                label_map[valid_slice_y, valid_slice_x][update_mask] = block_to_label.get(block, 0)
                ring_map[valid_slice_y, valid_slice_x][update_mask] = ring_id

        # Override interlocked rings with GT pixel masks (legacy mode)
        gt_label_ring4_path = os.path.join(tunnel_dir, 'gt_label_ring4.npy')
        if os.path.exists(gt_label_ring4_path):
            gt_label_ring4 = np.load(gt_label_ring4_path)
            override_count = 0
            for seg_id in range(1, 8):
                pixel_mask = gt_label_ring4 == seg_id
                if pixel_mask.sum() == 0:
                    continue
                bn = {1:'K', 2:'B1', 3:'A1', 4:'A2', 5:'A3', 6:'A4', 7:'B2'}[seg_id]
                label_val = block_to_label.get(bn, 0)
                if label_val == 0:
                    continue
                label_map[pixel_mask] = label_val
                ring_map[pixel_mask] = 4
                override_count += pixel_mask.sum()
            print(f"  Ring 4 GT pixel mask override: {override_count:,} pixels")

    # Fix ring numbering
    fix_ring = np.where((ring_map >= 1) & (ring_map <= (ring_count-1)),
                        ring_count - ring_map, ring_map)
    
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
    parser = argparse.ArgumentParser(description="SAM-based tunnel segmentation (complex staggered)")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_sam(args.tunnel_id, base_dir=args.data_dir)
