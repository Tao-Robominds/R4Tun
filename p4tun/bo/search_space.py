"""
Search Space Definition for Bayesian Optimization

Defines the parameter search spaces for detection and SAM stages.
Expanded SAM search space includes prompt point positions and template mask dimensions.
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
# SAM Parameters Search Space (Expanded)
# =============================================================================
# Includes prompt point positions, template mask dimensions, and processing params

SAM_SPACE = {
    # Segment geometry - NOW TUNABLE
    'segment_width': Real(1150.0, 1250.0, name='segment_width'),
    'k_height': Real(1000.0, 1160.0, name='k_height'),  # Was fixed at 1079.92
    'ab_height': Real(3100.0, 3380.0, name='ab_height'),  # Was fixed at 3239.77
    'angle_deg': Real(6.5, 8.5, name='angle_deg'),  # Was fixed at 7.52
    
    # Processing parameters
    'padding': Integer(100, 200, name='padding'),
    'crop_margin': Integer(30, 80, name='crop_margin'),
    
    # K-block prompt point parameters
    'k_outer_ring': Real(650.0, 750.0, name='k_outer_ring'),
    'k_middle_ring': Real(450.0, 550.0, name='k_middle_ring'),
    'k_inner_ring': Real(300.0, 400.0, name='k_inner_ring'),
    'k_center_ring': Real(280.0, 370.0, name='k_center_ring'),
    
    # AB-block prompt point parameters
    'ab_outer_ring': Real(650.0, 750.0, name='ab_outer_ring'),
    'ab_middle_ring': Real(460.0, 560.0, name='ab_middle_ring'),
    'ab_inner_ring': Real(450.0, 550.0, name='ab_inner_ring'),
    'ab_center_ring': Real(280.0, 370.0, name='ab_center_ring'),
    'ab_fine_spacing': Real(200.0, 300.0, name='ab_fine_spacing'),
    'ab_ultra_fine': Real(130.0, 200.0, name='ab_ultra_fine'),
    'ab_edge_ring': Real(300.0, 400.0, name='ab_edge_ring'),
    'ab_edge_spacing': Real(300.0, 400.0, name='ab_edge_spacing'),
    
    # AB-block vertical levels - NOW TUNABLE (±10% from defaults)
    'ab_level_1': Real(1550.0, 1890.0, name='ab_level_1'),  # Default 1719.89
    'ab_level_2': Real(1370.0, 1670.0, name='ab_level_2'),  # Default 1519.89
    'ab_level_3': Real(1210.0, 1480.0, name='ab_level_3'),  # Default 1344.89
    'ab_level_4': Real(980.0, 1200.0, name='ab_level_4'),   # Default 1090.09
    'ab_level_5': Real(735.0, 900.0, name='ab_level_5'),    # Default 817.57
    'ab_level_6': Real(490.0, 600.0, name='ab_level_6'),    # Default 545.05
    'ab_level_7': Real(245.0, 300.0, name='ab_level_7'),    # Default 272.52
    
    # Template mask dimensions - K block
    'k_mask_width': Real(575.0, 675.0, name='k_mask_width'),
    'k_mask_height_pos': Real(570.0, 670.0, name='k_mask_height_pos'),
    'k_mask_height_neg': Real(410.0, 510.0, name='k_mask_height_neg'),
    
    # Template mask dimensions - A/B blocks
    'ab_mask_width': Real(575.0, 675.0, name='ab_mask_width'),
    'ab_mask_height': Real(1570.0, 1670.0, name='ab_mask_height'),
    
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
    """Convert BO parameters to SAM parameters dict structure (expanded with physical constants)."""
    param_dict = dict(zip(names, params))
    
    return {
        'segment_geometry': {
            'segment_width': float(param_dict.get('segment_width', 1200.0)),
            # Physical constants - NOW TUNABLE
            'k_height': float(param_dict.get('k_height', 1079.92)),
            'ab_height': float(param_dict.get('ab_height', 3239.77)),
            'angle_deg': float(param_dict.get('angle_deg', 7.52)),
        },
        'image': {
            'resolution': 0.005,
        },
        'processing': {
            'padding': int(param_dict.get('padding', 150)),
            'crop_margin': int(param_dict.get('crop_margin', 50)),
            'mask_eps': 0.001,
            'y_bounds': [4200, 13100],
        },
        'prompt_points': {
            'k_block': {
                'outer_ring': float(param_dict.get('k_outer_ring', 700)),
                'middle_ring': float(param_dict.get('k_middle_ring', 500)),
                'inner_ring': float(param_dict.get('k_inner_ring', 348.16)),
                'center_ring': float(param_dict.get('k_center_ring', 325)),
                'spacing_factors': {
                    'k_block_spacing': 310.91,
                    'vertical_spacing': [732.35, 505.96, 310.91, 219.01, 373.96]
                }
            },
            'ab_blocks': {
                'outer_ring': float(param_dict.get('ab_outer_ring', 700)),
                'middle_ring': float(param_dict.get('ab_middle_ring', 511.06)),
                'inner_ring': float(param_dict.get('ab_inner_ring', 500)),
                'center_ring': float(param_dict.get('ab_center_ring', 325)),
                'fine_spacing': float(param_dict.get('ab_fine_spacing', 250)),
                'ultra_fine': float(param_dict.get('ab_ultra_fine', 162.5)),
                'edge_ring': float(param_dict.get('ab_edge_ring', 348.16)),
                'edge_spacing': float(param_dict.get('ab_edge_spacing', 350)),
                'vertical_levels': {
                    'level_1': float(param_dict.get('ab_level_1', 1719.89)),
                    'level_2': float(param_dict.get('ab_level_2', 1519.89)),
                    'level_3': float(param_dict.get('ab_level_3', 1344.89)),
                    'level_4': float(param_dict.get('ab_level_4', 1090.09)),
                    'level_5': float(param_dict.get('ab_level_5', 817.57)),
                    'level_6': float(param_dict.get('ab_level_6', 545.05)),
                    'level_7': float(param_dict.get('ab_level_7', 272.52)),
                    'center': 0
                }
            },
            'template_mask': {
                'k_block': {
                    'width': float(param_dict.get('k_mask_width', 625)),
                    'height_pos': float(param_dict.get('k_mask_height_pos', 619.16)),
                    'height_neg': float(param_dict.get('k_mask_height_neg', 460.77))
                },
                'b1_block': {
                    'width': float(param_dict.get('ab_mask_width', 625)),
                    'height_top': float(param_dict.get('ab_mask_height', 1619.89)),
                    'height_bottom_pos': 1540.69,
                    'height_bottom_neg': 1699.08
                },
                'b2_block': {
                    'width': float(param_dict.get('ab_mask_width', 625)),
                    'height_top_pos': 1540.69,
                    'height_top_neg': 1699.08,
                    'height_bottom': float(param_dict.get('ab_mask_height', 1619.89))
                },
                'a_blocks': {
                    'width': float(param_dict.get('ab_mask_width', 625)),
                    'height': float(param_dict.get('ab_mask_height', 1619.89))
                }
            }
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
