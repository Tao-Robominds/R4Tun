"""
Bayesian Optimization for P4Tun Pipeline Parameters

This module provides Bayesian Optimization for tuning the parameters
of the P4Tun tunnel segmentation pipeline.

Usage:
    # Optimize both detection and SAM parameters
    python -m p4tun.bo.optimize --tunnel 4-1 --n-calls 50
    
    # Optimize only detection parameters
    python -m p4tun.bo.optimize --tunnel 4-1 --stage detection --n-calls 30
    
    # Optimize only SAM parameters (expanded search space)
    python -m p4tun.bo.optimize --tunnel 2-2 --stage sam --n-calls 30
    
    # Use different metric (OA instead of mIoU)
    python -m p4tun.bo.optimize --tunnel 4-1 --metric OA --n-calls 50

The optimization focuses on detection (4-1_detection.py) and SAM (4-2_sam.py)
stages, using ground truth data for evaluation.

Search Spaces:
    Detection:
        - Preprocessing: binary_threshold, dilation settings
        - Hough line detection: thresholds, min_length, max_gap, angles
        - Line processing: merge_distance_threshold
    
    SAM (Expanded):
        - Segment geometry: segment_width
        - Processing: padding, crop_margin
        - K-block prompt points: outer_ring, middle_ring, inner_ring, center_ring
        - AB-block prompt points: outer_ring, middle_ring, inner_ring, etc.
        - Template mask dimensions: width, height for K, A, B blocks
        - Pattern-aware: quality weighting threshold
"""

# Lazy imports to avoid circular import warnings when running as __main__
def __getattr__(name):
    if name == 'get_search_space':
        from .search_space import get_search_space
        return get_search_space
    elif name == 'params_to_detection_dict':
        from .search_space import params_to_detection_dict
        return params_to_detection_dict
    elif name == 'params_to_sam_dict':
        from .search_space import params_to_sam_dict
        return params_to_sam_dict
    elif name == 'PipelineObjective':
        from .objective import PipelineObjective
        return PipelineObjective
    elif name == 'BayesianOptimizer':
        from .optimize import BayesianOptimizer
        return BayesianOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'get_search_space', 
    'params_to_detection_dict',
    'params_to_sam_dict',
    'PipelineObjective', 
    'BayesianOptimizer'
]
