"""
Search Space Definition for Bayesian Optimization

Defines the parameter search spaces for detection and SAM stages.
"""

from skopt.space import Real, Integer, Categorical
from typing import Dict, List, Tuple, Any
import json
import os


# =============================================================================
# Detection Parameters Search Space
# =============================================================================

DETECTION_SPACE = {
    # Preprocessing
    'binary_threshold': Integer(100, 150, name='binary_threshold'),
    'dilation_kernel_size': Integer(2, 5, name='dilation_kernel_size'),
    'dilation_iterations': Integer(1, 3, name='dilation_iterations'),
    
    # Hough oblique line detection
    'hough_oblique_threshold': Integer(30, 80, name='hough_oblique_threshold'),
    'hough_oblique_min_length': Integer(80, 150, name='hough_oblique_min_length'),
    'hough_oblique_max_gap': Integer(30, 60, name='hough_oblique_max_gap'),
    'angle_positive_min': Real(5.0, 7.0, name='angle_positive_min'),
    'angle_positive_max': Real(8.0, 10.0, name='angle_positive_max'),
    
    # Hough horizontal line detection  
    'hough_horizontal_threshold': Integer(40, 70, name='hough_horizontal_threshold'),
    'hough_horizontal_min_length': Integer(80, 130, name='hough_horizontal_min_length'),
    'hough_horizontal_max_gap': Integer(5, 20, name='hough_horizontal_max_gap'),
    
    # Hough vertical line detection
    'hough_vertical_threshold': Integer(400, 700, name='hough_vertical_threshold'),
    
    # Line processing
    'merge_distance_threshold': Integer(2, 5, name='merge_distance_threshold'),
}


# =============================================================================
# SAM Parameters Search Space
# =============================================================================
# NOTE: Physical constants (k_height, ab_height, angle_deg) are FIXED
# Only tune processing parameters, not physical geometry

SAM_SPACE = {
    # Segment width can vary slightly due to image processing
    'segment_width': Real(1150.0, 1250.0, name='segment_width'),
    
    # Pattern-aware parameters
    'min_quality_threshold': Real(0.1, 0.5, name='min_quality_threshold'),
}


# =============================================================================
# Combined Search Spaces
# =============================================================================

def get_detection_space() -> Tuple[List, List[str]]:
    """Get detection parameter search space."""
    dimensions = list(DETECTION_SPACE.values())
    names = list(DETECTION_SPACE.keys())
    return dimensions, names


def get_sam_space() -> Tuple[List, List[str]]:
    """Get SAM parameter search space."""
    dimensions = list(SAM_SPACE.values())
    names = list(SAM_SPACE.keys())
    return dimensions, names


def get_combined_space() -> Tuple[List, List[str]]:
    """Get combined detection + SAM search space."""
    combined = {}
    combined.update(DETECTION_SPACE)
    combined.update(SAM_SPACE)
    dimensions = list(combined.values())
    names = list(combined.keys())
    return dimensions, names


def get_search_space(stage: str = 'combined') -> Tuple[List, List[str]]:
    """
    Get search space for specified stage.
    
    Args:
        stage: 'detection', 'sam', or 'combined'
    
    Returns:
        Tuple of (dimensions list, parameter names list)
    """
    if stage == 'detection':
        return get_detection_space()
    elif stage == 'sam':
        return get_sam_space()
    else:
        return get_combined_space()


# =============================================================================
# Parameter Conversion
# =============================================================================

def params_to_detection_dict(params: List, names: List[str]) -> Dict:
    """Convert BO parameters to detection parameters dict structure."""
    param_dict = dict(zip(names, params))
    
    return {
        'preprocessing': {
            'binary_threshold': int(param_dict.get('binary_threshold', 127)),
            'dilation_kernel_size': int(param_dict.get('dilation_kernel_size', 3)),
            'dilation_iterations': int(param_dict.get('dilation_iterations', 1)),
        },
        'hough_oblique': {
            'threshold': int(param_dict.get('hough_oblique_threshold', 50)),
            'min_length': int(param_dict.get('hough_oblique_min_length', 100)),
            'max_gap': int(param_dict.get('hough_oblique_max_gap', 40)),
            'angle_positive_min': float(param_dict.get('angle_positive_min', 6.0)),
            'angle_positive_max': float(param_dict.get('angle_positive_max', 9.0)),
            'angle_negative_min': -float(param_dict.get('angle_positive_max', 9.0)),
            'angle_negative_max': -float(param_dict.get('angle_positive_min', 6.0)),
        },
        'hough_horizontal': {
            'threshold': int(param_dict.get('hough_horizontal_threshold', 50)),
            'min_length': int(param_dict.get('hough_horizontal_min_length', 100)),
            'max_gap': int(param_dict.get('hough_horizontal_max_gap', 10)),
            'angle_tolerance': 1,
        },
        'hough_vertical': {
            'threshold': int(param_dict.get('hough_vertical_threshold', 500)),
            'filter_rings': 5,
        },
        'line_processing': {
            'merge_distance_threshold': int(param_dict.get('merge_distance_threshold', 3)),
        },
        'physical_constants': {
            'resolution': 0.005,
            'k_height_mm': 1079.92,
            'ab_height_mm': 3239.77,
        },
    }


def params_to_sam_dict(params: List, names: List[str]) -> Dict:
    """Convert BO parameters to SAM parameters dict structure.
    
    NOTE: Physical constants (k_height, ab_height, angle_deg) are FIXED.
    """
    param_dict = dict(zip(names, params))
    
    return {
        'segment_geometry': {
            'segment_width': float(param_dict.get('segment_width', 1200.0)),
            # Physical constants - DO NOT TUNE
            'k_height': 1079.92,
            'ab_height': 3239.77,
            'angle_deg': 7.52,
        },
        'image': {
            'resolution': 0.005,
        },
        'pattern_aware': {
            'use_quality_weighting': True,
            'min_quality_threshold': float(param_dict.get('min_quality_threshold', 0.3)),
        },
    }


def save_parameters(params_dict: Dict, tunnel_id: str, stage: str, base_dir: str = 'p4tun/parameters'):
    """Save parameters to JSON file."""
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    os.makedirs(tunnel_dir, exist_ok=True)
    
    filename = f'parameters_{stage}.json'
    filepath = os.path.join(tunnel_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    return filepath


def load_default_parameters(tunnel_id: str, stage: str, base_dir: str = 'p4tun/parameters') -> Dict:
    """Load default parameters from existing JSON file."""
    filepath = os.path.join(base_dir, tunnel_id, f'parameters_{stage}.json')
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # Try sample parameters
    sample_path = os.path.join(base_dir, 'sample', f'parameters_{stage}.json')
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    return {}
