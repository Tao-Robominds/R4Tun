"""
SAM-based Tunnel Segment Segmentation (GT-Free)

This module uses Segment Anything Model (SAM) to segment tunnel rings into
individual blocks (K, B1, A1, A2, ..., B2).

NO GROUND TRUTH REQUIRED.

Modes:
    - row: Walk from K-block center (best for non-wrap-around)
    - pattern: Use inferred segment positions (best for wrap-around)
    - auto: Auto-detect based on available files

Input files:
    - detected.csv: K-block centers from Hough detection (for row mode)
    - inferred_from_pattern.csv: All segment positions (for pattern mode)
"""

import os
import sys
import json
import math
import pickle
from enum import Enum
from typing import Tuple, List, Dict, Optional, Any

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Dict[str, Any]:
    """
    Load parameters from JSON file with fallback to defaults.
    
    Priority:
        1. Centralized: sam4tun/parameters/<tunnel_id>/parameters_sam.json
        2. Tunnel-specific: data/<tunnel_id>/parameters_sam.json
        3. Default: sam4tun/parameters_sam.json (if exists)
        4. Hardcoded defaults (if no file found)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_sam.json"
    
    if tunnel_id:
        params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
        if os.path.exists(params_path):
            print(f"Loading parameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f)
        
        tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
        if os.path.exists(tunnel_path):
            print(f"Loading parameters from {tunnel_path}")
            with open(tunnel_path, 'r') as f:
                return json.load(f)
    
    default_path = os.path.join(script_dir, param_file)
    if os.path.exists(default_path):
        print(f"Loading default parameters from {default_path}")
        with open(default_path, 'r') as f:
            return json.load(f)
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}


def get_param(params: Dict, *keys, default=None):
    """Get nested parameter value with default fallback."""
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# Default Constants
# =============================================================================

# --- Model Configuration ---
DEFAULT_SAM_CHECKPOINT = "sam4tun/segment-anything/sam_vit_h_4b8939.pth"
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_DEVICE = "cuda"

# --- Segment Geometry (in mm) ---
DEFAULT_SEGMENT_WIDTH = 1200
DEFAULT_K_HEIGHT = 1079.92
DEFAULT_AB_HEIGHT = 3239.77
DEFAULT_ANGLE = 7.52

# --- Image Resolution ---
DEFAULT_RESOLUTION = 0.005


class ProcessingMode(Enum):
    ROW = "row"
    PATTERN = "pattern"
    AUTO = "auto"


# =============================================================================
# Segment Count Detection
# =============================================================================

def detect_segment_count_from_image(
    image_height: int,
    resolution: float = DEFAULT_RESOLUTION,
    k_height: float = DEFAULT_K_HEIGHT,
    ab_height: float = DEFAULT_AB_HEIGHT
) -> int:
    """Detect segment count from image height."""
    height_mm = image_height * resolution * 1000
    circumference_6 = k_height + 5 * ab_height
    circumference_7 = k_height + 6 * ab_height
    
    if abs(height_mm - circumference_6) < abs(height_mm - circumference_7):
        return 6
    return 7


def compute_block_labels(segment_per_ring: int) -> List[str]:
    """Generate block labels based on segment count."""
    labels = ['K', 'B1']
    num_a_blocks = segment_per_ring - 3
    labels += [f'A{i+1}' for i in range(num_a_blocks)]
    labels += ['B2']
    return labels


def compute_block_to_label_map(segment_per_ring: int) -> Dict[str, int]:
    """Create mapping from block names to numeric labels."""
    labels = compute_block_labels(segment_per_ring)
    return {label: i + 1 for i, label in enumerate(labels)}


# =============================================================================
# Template Mask Generation
# =============================================================================

def fill_polygon(mask: np.ndarray, vertices: np.ndarray) -> None:
    """Fill a polygon region in a binary mask."""
    path = Path(vertices)
    y_coords, x_coords = np.mgrid[:mask.shape[0], :mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask_inside = path.contains_points(points).reshape(mask.shape)
    mask[mask_inside] = 1


def generate_template_mask(
    height: int,
    width: int,
    prompt_centre: Tuple[float, float],
    block: str,
    resolution: float = DEFAULT_RESOLUTION
) -> np.ndarray:
    """Generate a template mask for a specific block type."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cx, cy = prompt_centre
    x = cx * (resolution * 1000)
    y = cy * (resolution * 1000)
    
    if block == 'K':
        vertices_real = np.array([
            [x - 625, y - 619.16], [x - 625, y + 619.16],
            [x + 625, y + 460.77], [x + 625, y - 460.77]
        ])
    elif block == 'B1':
        vertices_real = np.array([
            [x - 625, y - 1619.89], [x - 625, y + 1540.69],
            [x + 625, y + 1699.08], [x + 625, y - 1619.89]
        ])
    elif block == 'B2':
        vertices_real = np.array([
            [x - 625, y - 1540.69], [x - 625, y + 1619.89],
            [x + 625, y + 1619.89], [x + 625, y - 1699.08]
        ])
    else:  # A blocks
        vertices_real = np.array([
            [x - 625, y - 1619.89], [x - 625, y + 1619.89],
            [x + 625, y + 1619.89], [x + 625, y - 1619.89]
        ])
    
    vertices = vertices_real / (resolution * 1000)
    fill_polygon(mask, vertices)
    return mask


# =============================================================================
# Prompt Point Generation
# =============================================================================

def generate_prompt_points(
    prompt_centre: Tuple[float, float],
    initial_x: float,
    map_y: float,
    block: str,
    resolution: float = DEFAULT_RESOLUTION,
    segment_width: float = DEFAULT_SEGMENT_WIDTH,
    k_height: float = DEFAULT_K_HEIGHT,
    ab_height: float = DEFAULT_AB_HEIGHT,
    image_shape: Tuple[int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate prompt points for SAM prediction."""
    cx, cy = prompt_centre
    x = cx * (resolution * 1000)
    y = cy * (resolution * 1000)
    map_y_mm = map_y * (resolution * 1000)
    
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
        labels = np.repeat([0, 1], [51, 56])
        
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
            [x-511.06,y+1519.89],[x-348.16,y+1519.89],[x,y+1519.89],[x+348.16,y+1519.89],[x+511.06,y+1519.89]
        ])
        labels = np.repeat([0, 1], [51, 56])
        
    else:  # A blocks (symmetric)
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
            [x-511.06,y+1519.89],[x-348.16,y+1519.89],[x,y+1519.89],[x+348.16,y+1519.89],[x+511.06,y+1519.89]
        ])
        labels = np.repeat([0, 1], [51, 56])
    
    # Filter boundary points
    keep_mask = np.ones(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 0:
            y_cond = points_real[i, 1] + map_y_mm < 4200 or points_real[i, 1] + map_y_mm > 13100
            x_cond = abs(points_real[i, 0] - x) <= segment_width * 0.5
            y_limit = k_height if block == 'K' else ab_height
            y_cond2 = abs(points_real[i, 1] - y) <= y_limit * 0.5
            
            if y_cond and x_cond and y_cond2:
                keep_mask[i] = False
    
    points_real = points_real[keep_mask]
    labels = labels[keep_mask]
    
    # Convert to pixel coordinates
    points = points_real / (resolution * 1000)
    
    # Filter points within image bounds
    if image_shape is not None:
        img_width = image_shape[1]
        within_bounds = (points[:, 0] >= 0) & (
            (points[:, 0] + initial_x - (segment_width * 0.5 + 150) / (resolution * 1000)) <= img_width
        )
        points = points[within_bounds]
        labels = labels[within_bounds]
    
    return points, labels


# =============================================================================
# Logits Computation
# =============================================================================

def compute_logits_from_mask(mask: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Convert binary mask to logits for SAM input."""
    def inv_sigmoid(x):
        return np.log(x / (1 - x))
    
    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)
    
    expected_shape = (256, 256)
    
    if logits.shape != expected_shape:
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])
        
        if logits.shape[:2] != expected_shape:
            h, w = logits.shape[:2]
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w
            logits = np.pad(logits.squeeze(), ((0, padh), (0, padw)), 
                          mode="constant", constant_values=0)
    
    return logits[None]


def restore_sam_logits(logits: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """Restore logits to original image dimensions."""
    orig_h, orig_w = original_shape
    trafo = ResizeLongestSide(max(orig_h, orig_w))
    resized = trafo.apply_image(logits[..., None])
    return resized.squeeze()[:orig_h, :orig_w]


# =============================================================================
# Image Processing
# =============================================================================

def convert_to_pixel_coords(real_dist: float, resolution: float = DEFAULT_RESOLUTION) -> int:
    """Convert real distance (mm) to pixel coordinates."""
    return int(real_dist / (resolution * 1000))


def crop_image_and_prepare_mask(
    image: np.ndarray,
    cx: float,
    cy: float,
    crop_width: int,
    crop_height: int,
    block: str,
    resolution: float = DEFAULT_RESOLUTION
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Crop image around center and generate template mask."""
    img_h, img_w = image.shape[:2]
    
    x1 = max(int(cx - crop_width // 2), 0)
    y1 = max(int(cy - crop_height // 2), 0)
    x2 = min(int(cx + crop_width // 2), img_w)
    y2 = min(int(cy + crop_height // 2), img_h)
    
    cropped = image[y1:y2, x1:x2]
    prompt_centre = (cx - x1, cy - y1)
    
    template_mask = generate_template_mask(
        cropped.shape[0], cropped.shape[1], prompt_centre, block, resolution
    )
    template_logits = compute_logits_from_mask(template_mask)
    
    return cropped, template_logits, prompt_centre


# =============================================================================
# ROW-BASED Processing (walks from K-block center)
# =============================================================================

def process_row(
    df_row: pd.Series,
    image: np.ndarray,
    predictor: SamPredictor,
    segment_per_ring: int = 6,
    resolution: float = DEFAULT_RESOLUTION,
    segment_width: float = DEFAULT_SEGMENT_WIDTH,
    k_height: float = DEFAULT_K_HEIGHT,
    ab_height: float = DEFAULT_AB_HEIGHT,
    angle: float = DEFAULT_ANGLE
) -> List[Dict]:
    """
    Process a single ring row by walking from K-block center.
    
    This is the original row-based approach that handles boundaries naturally.
    """
    initial_x, initial_y = df_row['X'], df_row['Y']
    block_labels = compute_block_labels(segment_per_ring)
    
    delta_x = convert_to_pixel_coords(0.5 * segment_width + 150, resolution)
    
    reverse = False
    stop = False
    map_y = 0
    block_label_index = 0
    
    results = []
    
    for i in range(segment_per_ring):
        if not reverse:
            block = block_labels[block_label_index]
            
            if block_label_index == 0:
                delta_y = convert_to_pixel_coords(0.5 * k_height + math.tan(math.radians(angle)) * 700 + 150, resolution)
                map_y = initial_y
            else:
                delta_y = convert_to_pixel_coords(0.5 * ab_height + math.tan(math.radians(angle)) * 700 + 150, resolution)
                if block_label_index == 1:
                    map_y = initial_y - convert_to_pixel_coords(0.5 * k_height + 0.5 * ab_height, resolution)
                else:
                    map_y = map_y - convert_to_pixel_coords(ab_height, resolution)
            
            cropped, template_logits, prompt_centre = crop_image_and_prepare_mask(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution
            )
            
            points, labels = generate_prompt_points(
                prompt_centre, initial_x, map_y, block, resolution,
                segment_width, k_height, ab_height, image.shape
            )
            
            if len(points) == 0 or np.any(points[:, 1] < 0):
                if len(points) > 0:
                    within_bounds = points[:, 1] >= 0
                    points = points[within_bounds]
                    labels = labels[within_bounds]
                reverse = True
            
            if len(points) > 0:
                predictor.set_image(cropped)
                mask, score, logit = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=template_logits,
                    multimask_output=False
                )
                
                results.append({
                    'left_top': (initial_x - prompt_centre[0], map_y - prompt_centre[1]),
                    'block': block,
                    'mask': mask,
                    'score': score,
                    'logit': logit[0]
                })
            
            if reverse:
                block_label_index = -1
                continue
            
            block_label_index += 1
        
        if reverse:
            block = block_labels[block_label_index]
            
            if block_label_index == -1:
                map_y = initial_y + convert_to_pixel_coords(0.5 * k_height + 0.5 * ab_height, resolution)
            else:
                map_y = map_y + convert_to_pixel_coords(ab_height, resolution)
            
            delta_y = convert_to_pixel_coords(0.5 * ab_height + math.tan(math.radians(angle)) * 700 + 150, resolution)
            
            cropped, template_logits, prompt_centre = crop_image_and_prepare_mask(
                image, initial_x, map_y, 2 * delta_x, 2 * delta_y, block, resolution
            )
            
            points, labels = generate_prompt_points(
                prompt_centre, initial_x, map_y, block, resolution,
                segment_width, k_height, ab_height, image.shape
            )
            
            if len(points) > 0 and np.any((points[:, 1] + map_y - delta_y) > image.shape[0]):
                within_bounds = (points[:, 1] + map_y - delta_y) <= image.shape[0]
                points = points[within_bounds]
                labels = labels[within_bounds]
                stop = True
            
            if len(points) > 0:
                predictor.set_image(cropped)
                mask, score, logit = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=template_logits,
                    multimask_output=False
                )
                
                results.append({
                    'left_top': (initial_x - prompt_centre[0], map_y - prompt_centre[1]),
                    'block': block,
                    'mask': mask,
                    'score': score,
                    'logit': logit[0]
                })
            
            if stop:
                break
            
            block_label_index -= 1
    
    return results


def segment_row_based(
    detected_df: pd.DataFrame,
    image: np.ndarray,
    predictor: SamPredictor,
    segment_per_ring: int,
    ring_count: int,
    resolution: float = DEFAULT_RESOLUTION,
    segment_width: float = DEFAULT_SEGMENT_WIDTH,
    k_height: float = DEFAULT_K_HEIGHT,
    ab_height: float = DEFAULT_AB_HEIGHT,
    angle: float = DEFAULT_ANGLE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment using row-based approach (walk from K-block centers).
    """
    block_to_label = compute_block_to_label_map(segment_per_ring)
    
    all_results = []
    for _, row in tqdm(detected_df.iterrows(), total=len(detected_df), desc="Processing rows"):
        result = process_row(
            row, image, predictor, segment_per_ring, resolution,
            segment_width, k_height, ab_height, angle
        )
        all_results.append(result)
    
    # Aggregate results
    logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
    label_map = np.zeros(image.shape[:2], dtype=int)
    ring_map = np.zeros(image.shape[:2], dtype=int)
    
    for ring_index, ring_results in enumerate(all_results):
        for item in ring_results:
            mask = item['mask'][0]
            logits = item['logit']
            block = item['block']
            start_x, start_y = map(int, item['left_top'])
            
            end_y = start_y + mask.shape[0]
            end_x = start_x + mask.shape[1]
            start_y = max(0, start_y)
            start_x = max(0, start_x)
            end_y = min(image.shape[0], end_y)
            end_x = min(image.shape[1], end_x)
            
            slice_y = slice(start_y, end_y)
            slice_x = slice(start_x, end_x)
            
            new_logits = restore_sam_logits(logits, mask.shape)
            current_logits = logits_map[slice_y, slice_x]
            
            if mask.shape != current_logits.shape:
                continue
            if new_logits.shape != current_logits.shape:
                new_logits = new_logits[:current_logits.shape[0], :current_logits.shape[1]]
            
            update_mask = (new_logits > current_logits) & mask
            
            logits_map[slice_y, slice_x][update_mask] = new_logits[update_mask]
            label_map[slice_y, slice_x][update_mask] = block_to_label.get(block, 0)
            ring_map[slice_y, slice_x][update_mask] = ring_index
    
    # Fix ring numbering
    fix_ring = np.where(
        (ring_map >= 1) & (ring_map <= (ring_count - 1)),
        ring_count - ring_map,
        ring_map
    )
    
    return label_map, fix_ring


# =============================================================================
# PATTERN-BASED Processing (uses inferred positions)
# =============================================================================

def process_segment_pattern(
    segment_row: pd.Series,
    image: np.ndarray,
    predictor: SamPredictor,
    resolution: float = DEFAULT_RESOLUTION,
    segment_width: float = DEFAULT_SEGMENT_WIDTH,
    k_height: float = DEFAULT_K_HEIGHT,
    ab_height: float = DEFAULT_AB_HEIGHT,
    angle: float = DEFAULT_ANGLE
) -> Optional[Dict]:
    """Process a single segment at its pattern-inferred position."""
    initial_x = segment_row['X']
    initial_y = segment_row['Y']
    block = segment_row['Block']
    ring_id = segment_row['Ring']
    
    delta_x = convert_to_pixel_coords(0.5 * segment_width + 150, resolution)
    
    if block == 'K':
        delta_y = convert_to_pixel_coords(0.5 * k_height + math.tan(math.radians(angle)) * 700 + 150, resolution)
    else:
        delta_y = convert_to_pixel_coords(0.5 * ab_height + math.tan(math.radians(angle)) * 700 + 150, resolution)
    
    try:
        cropped, template_logits, prompt_centre = crop_image_and_prepare_mask(
            image, initial_x, initial_y, 2 * delta_x, 2 * delta_y, block, resolution
        )
        
        if cropped.size == 0:
            return None
        
        points, labels = generate_prompt_points(
            prompt_centre, initial_x, initial_y, block, resolution,
            SEGMENT_WIDTH, K_HEIGHT, AB_HEIGHT, image.shape
        )
        
        if len(points) == 0:
            return None
        
        # Filter out points that are outside the crop region (same as row-based)
        crop_height, crop_width = cropped.shape[:2]
        within_crop = (
            (points[:, 0] >= 0) & (points[:, 0] < crop_width) &
            (points[:, 1] >= 0) & (points[:, 1] < crop_height)
        )
        points = points[within_crop]
        labels = labels[within_crop]
        
        if len(points) == 0:
            return None
        
        predictor.set_image(cropped)
        mask, score, logit = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=template_logits,
            multimask_output=False
        )
        
        return {
            'left_top': (initial_x - prompt_centre[0], initial_y - prompt_centre[1]),
            'block': block,
            'ring_id': ring_id,
            'mask': mask,
            'score': score,
            'logit': logit[0]
        }
        
    except Exception as e:
        print(f"Error processing Ring {ring_id} Block {block}: {e}")
        return None


def segment_pattern_based(
    pattern_df: pd.DataFrame,
    image: np.ndarray,
    predictor: SamPredictor,
    segment_per_ring: int,
    resolution: float = DEFAULT_RESOLUTION,
    segment_width: float = DEFAULT_SEGMENT_WIDTH,
    k_height: float = DEFAULT_K_HEIGHT,
    ab_height: float = DEFAULT_AB_HEIGHT,
    angle: float = DEFAULT_ANGLE
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment using pattern-based approach (uses inferred positions)."""
    block_to_label = compute_block_to_label_map(segment_per_ring)
    
    results = []
    for _, row in tqdm(pattern_df.iterrows(), total=len(pattern_df), desc="Processing segments"):
        result = process_segment_pattern(
            row, image, predictor, resolution,
            segment_width, k_height, ab_height, angle
        )
        if result is not None:
            results.append(result)
    
    # Aggregate results
    logits_map = np.full(image.shape[:2], -np.inf, dtype=float)
    label_map = np.zeros(image.shape[:2], dtype=int)
    ring_map = np.zeros(image.shape[:2], dtype=int)
    
    for item in results:
        mask = item['mask'][0]
        logits = item['logit']
        block = item['block']
        ring_id = item['ring_id']
        start_x, start_y = map(int, item['left_top'])
        
        end_y = start_y + mask.shape[0]
        end_x = start_x + mask.shape[1]
        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(image.shape[0], end_y)
        end_x = min(image.shape[1], end_x)
        
        slice_y = slice(start_y, end_y)
        slice_x = slice(start_x, end_x)
        
        new_logits = restore_sam_logits(logits, mask.shape)
        current_logits = logits_map[slice_y, slice_x]
        
        if mask.shape != current_logits.shape:
            continue
        if new_logits.shape != current_logits.shape:
            new_logits = new_logits[:current_logits.shape[0], :current_logits.shape[1]]
        
        update_mask = (new_logits > current_logits) & mask
        
        logits_map[slice_y, slice_x][update_mask] = new_logits[update_mask]
        label_map[slice_y, slice_x][update_mask] = block_to_label.get(block, 0)
        ring_map[slice_y, slice_x][update_mask] = ring_id
    
    return label_map, ring_map


# =============================================================================
# Project Back to Point Cloud
# =============================================================================

def project_back_to_point_cloud(
    label_map: np.ndarray,
    ring_map: np.ndarray,
    pixel_to_point: List[Dict],
    df: pd.DataFrame
) -> pd.DataFrame:
    """Project 2D segmentation back to 3D point cloud."""
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)
    
    pixel_df = pd.DataFrame(pixel_to_point)
    y = pixel_df['pixel_y'].values
    x = pixel_df['pixel_x'].values
    point_indices = pixel_df['index'].values
    
    img_h, img_w = label_map.shape
    
    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    # Update points with pred=7 (surface) or pred=0 (background)
    valid_update_mask = np.isin(pred[point_indices[valid_point_mask]], [0, 7])
    
    y_valid = y[valid_point_mask][valid_update_mask]
    x_valid = x[valid_point_mask][valid_update_mask]
    
    bounds_mask = (y_valid >= 0) & (y_valid < img_h) & (x_valid >= 0) & (x_valid < img_w)
    
    final_indices = point_indices[valid_point_mask][valid_update_mask][bounds_mask]
    final_y = y_valid[bounds_mask]
    final_x = x_valid[bounds_mask]
    
    pred[final_indices] = label_map[final_y, final_x]
    pred_ring[final_indices] = ring_map[final_y, final_x]
    
    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring
    
    return df_copy


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    tunnel_id: str,
    mode: str = "auto",
    segment_count: Optional[int] = None,
    base_dir: str = "data"
) -> None:
    """
    Execute the SAM segmentation pipeline.
    
    Args:
        tunnel_id: Tunnel identifier.
        mode: Processing mode ("row", "pattern", or "auto").
        segment_count: Number of segments per ring (auto-detected if None).
        base_dir: Base data directory.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    detected_file = os.path.join(tunnel_dir, "detected.csv")
    pattern_file = os.path.join(tunnel_dir, "inferred_from_pattern.csv")
    
    print("=" * 70)
    print("SAM TUNNEL SEGMENTATION (GT-Free)")
    print("=" * 70)
    print(f"Tunnel: {tunnel_id}")
    
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    
    # Extract parameters with defaults
    sam_checkpoint = get_param(params, 'model', 'checkpoint', default=DEFAULT_SAM_CHECKPOINT)
    model_type = get_param(params, 'model', 'model_type', default=DEFAULT_MODEL_TYPE)
    device = get_param(params, 'model', 'device', default=DEFAULT_DEVICE)
    segment_width = get_param(params, 'segment_geometry', 'segment_width', default=DEFAULT_SEGMENT_WIDTH)
    k_height = get_param(params, 'segment_geometry', 'k_height', default=DEFAULT_K_HEIGHT)
    ab_height = get_param(params, 'segment_geometry', 'ab_height', default=DEFAULT_AB_HEIGHT)
    angle = get_param(params, 'segment_geometry', 'angle_deg', default=DEFAULT_ANGLE)
    resolution = get_param(params, 'image', 'resolution', default=DEFAULT_RESOLUTION)
    
    # Determine mode
    if mode == "auto":
        if os.path.exists(detected_file):
            mode = "row"
        elif os.path.exists(pattern_file):
            mode = "pattern"
        else:
            raise FileNotFoundError(f"No input files found in {tunnel_dir}")
    
    print(f"Mode: {mode}")
    
    # Load common data
    pixel_to_point = pickle.load(open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb"))
    df_point_cloud = pd.read_csv(os.path.join(tunnel_dir, "enhanced.csv"))
    
    with open(os.path.join(tunnel_dir, "ring_count.txt"), 'r') as f:
        ring_count = int(f.read().strip())
    
    # Load image
    image = cv2.imread(os.path.join(tunnel_dir, "depth_map.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image: {image.shape[1]} x {image.shape[0]}")
    
    # Detect segment count if not specified
    if segment_count is None:
        segment_count = detect_segment_count_from_image(
            image.shape[0], resolution, k_height, ab_height
        )
    print(f"Segments: {segment_count}")
    print(f"Rings: {ring_count}")
    print("=" * 70)
    
    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Run segmentation based on mode
    if mode == "row":
        detected_df = pd.read_csv(detected_file)
        print(f"Using row-based processing ({len(detected_df)} K-block centers)")
        label_map, ring_map = segment_row_based(
            detected_df, image, predictor, segment_count, ring_count,
            resolution, segment_width, k_height, ab_height, angle
        )
    else:  # pattern mode
        pattern_df = pd.read_csv(pattern_file)
        print(f"Using pattern-based processing ({len(pattern_df)} segment positions)")
        label_map, ring_map = segment_pattern_based(
            pattern_df, image, predictor, segment_count,
            resolution, segment_width, k_height, ab_height, angle
        )
    
    # Project back to point cloud
    updated_df = project_back_to_point_cloud(
        label_map, ring_map, pixel_to_point, df_point_cloud
    )
    
    # Save results
    os.makedirs(tunnel_dir, exist_ok=True)
    updated_df.to_csv(os.path.join(tunnel_dir, "final.csv"), index=False)
    
    # Save predictions (GT-free format)
    df_pred = pd.DataFrame({
        'pred_labels': updated_df['pred'],
        'pred_rings': updated_df['pred_ring']
    })
    df_pred.to_csv(os.path.join(tunnel_dir, "predictions.csv"), index=False)
    
    # Also save legacy format if GT available
    if 'segment' in updated_df.columns:
        df_legacy = pd.DataFrame({
            'gt_labels': updated_df['segment'],
            'gt_rings': updated_df['ring'],
            'pred_labels': updated_df['pred'],
            'pred_rings': updated_df['pred_ring']
        })
        df_legacy.to_csv(os.path.join(tunnel_dir, "only_label.csv"), index=False)
    
    print("=" * 70)
    print(f"Saved to {tunnel_dir}/")
    print(f"  - final.csv")
    print(f"  - predictions.csv")
    if 'segment' in updated_df.columns:
        print(f"  - only_label.csv (legacy format with GT)")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 4-2_sam_clean.py <tunnel_id> [options]")
        print()
        print("Options:")
        print("  --mode <row|pattern|auto>  Processing mode (default: auto)")
        print("  --segments <N>             Force segment count (default: auto-detect)")
        print()
        print("Modes:")
        print("  row     - Walk from K-block centers (best for non-wrap-around)")
        print("  pattern - Use inferred positions (best for wrap-around)")
        print("  auto    - Auto-detect based on available files")
        print()
        print("Examples:")
        print("  python 4-2_sam_clean.py sample              # Auto mode")
        print("  python 4-2_sam_clean.py sample --mode row   # Force row mode")
        print("  python 4-2_sam_clean.py 4-1 --mode pattern  # Force pattern mode")
        sys.exit(1)
    
    tunnel_id = sys.argv[1]
    mode = "auto"
    segment_count = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--segments" and i + 1 < len(sys.argv):
            segment_count = int(sys.argv[i + 1])
            i += 2
        else:
            # Legacy: positional segment count
            try:
                segment_count = int(sys.argv[i])
            except ValueError:
                pass
            i += 1
    
    main(tunnel_id, mode=mode, segment_count=segment_count)


