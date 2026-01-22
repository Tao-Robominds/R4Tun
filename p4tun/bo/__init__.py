"""
Bayesian Optimization for P4Tun Pipeline Parameters

This module provides Bayesian Optimization for tuning the parameters
of the P4Tun tunnel segmentation pipeline.

Usage:
    # Optimize both detection and SAM parameters
    python -m p4tun.bo.optimize --tunnel 4-1 --n-calls 50
    
    # Optimize only detection parameters
    python -m p4tun.bo.optimize --tunnel 4-1 --stage detection --n-calls 30
    
    # Optimize only SAM parameters
    python -m p4tun.bo.optimize --tunnel 2-2 --stage sam --n-calls 30
    
    # Use different metric (OA instead of mIoU)
    python -m p4tun.bo.optimize --tunnel 4-1 --metric OA --n-calls 50

The optimization focuses on detection (4-1_detection.py) and SAM (4-2_sam.py)
stages, using ground truth data for evaluation.

Search Spaces:
    Detection:
        - Preprocessing: binary_threshold, dilation settings
        - Hough line detection: thresholds, min_length, max_gap, angles
        - Pattern detection: tolerances for V-pair spacing, alternation
    
    SAM:
        - Segment geometry: segment_width, k_height, ab_height, angle
        - Pattern-aware: quality weighting threshold
"""

from .search_space import get_search_space, params_to_detection_dict, params_to_sam_dict
from .objective import PipelineObjective
from .optimize import BayesianOptimizer

__all__ = [
    'get_search_space', 
    'params_to_detection_dict',
    'params_to_sam_dict',
    'PipelineObjective', 
    'BayesianOptimizer'
]
