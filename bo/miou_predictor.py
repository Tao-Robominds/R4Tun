#!/usr/bin/env python3
"""
mIoU Predictor Module

Provides functions to predict mIoU from intrinsic metrics or parameters.
Uses trained models from bo4tun/models/

Two prediction modes:
1. From parameters (recommended, higher accuracy with n=198/93 training):
   - predict_from_params(tunnel_id, params_dict)
   
2. From intrinsic metrics (requires pipeline outputs):
   - predict_miou(tunnel_id, detected_csv, final_csv)

Usage:
    from bo4tun.miou_predictor import predict_from_params, predict_miou
    
    # From parameters (before running pipeline)
    result = predict_from_params('2-2', {
        'param_binary_threshold': 127,
        'param_segment_width': 1200,
        ...
    })
    
    # From intrinsic metrics (after running pipeline)
    result = predict_miou('2-2', 'data/2-2/detected.csv', 'data/2-2/final.csv')
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent

# Feature definitions for parameter-based prediction (RECOMMENDED)
SIMPLE_PARAM_FEATURES = [
    'param_binary_threshold',
    'param_hough_oblique_threshold',
    'param_dilation_iterations',
    'param_segment_width',
    'param_k_height',
    'param_angle_deg',
]

COMPLEX_PARAM_FEATURES = [
    'param_segment_width',
    'param_k_height',
    'param_ab_height',
    'param_angle_deg',
]

# Feature definitions for intrinsic metrics prediction (legacy)
SIMPLE_FEATURES = [
    'det_midpoint_ratio',
    'det_real_detection_ratio',
    'det_x_spacing_cv',
    'sam_mask_fill_rate',
    'det_y_std',
]

COMPLEX_FEATURES = [
    'complex_sam_ring_completeness',
    'complex_sam_segment_height_cv',
    'complex_sam_segment_width_cv',
    'complex_sam_segment_area_cv',
    'complex_sam_ring_coverage_cv',
    'complex_sam_aspect_ratio_mean',
]


def load_predictor(pattern_type: str, use_params: bool = True) -> Tuple[object, list]:
    """
    Load trained predictor model.
    
    Args:
        pattern_type: 'simple' or 'complex'
        use_params: If True, use parameter-based predictor (recommended)
                    If False, use intrinsic metrics predictor (legacy)
    """
    model_dir = project_root / 'bo4tun' / 'models'
    
    if use_params:
        # New parameter-based predictors (higher accuracy)
        if pattern_type == 'simple':
            model_path = model_dir / 'simple_predictor_final.pkl'
        else:
            model_path = model_dir / 'complex_predictor_final.pkl'
    else:
        # Legacy intrinsic metrics predictors
        if pattern_type == 'simple':
            model_path = model_dir / 'simple_miou_predictor.pkl'
        else:
            model_path = model_dir / 'complex_miou_predictor.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_miou_predictors.py first.")
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['features']


def predict_from_params(tunnel_id: str, params: Dict) -> Dict:
    """
    Predict mIoU from input parameters (RECOMMENDED).
    
    This uses the improved predictor trained on 198/93 samples.
    Call this BEFORE running the pipeline to estimate expected mIoU.
    
    Args:
        tunnel_id: Tunnel identifier (1-4, 2-2, 3-1, 4-1, 5-1)
        params: Dictionary of parameters (param_* keys)
        
    Returns:
        Dictionary with predicted_mIoU, confidence, features used
    """
    is_complex = tunnel_id in ['4-1', '5-1']
    pattern_type = 'complex' if is_complex else 'simple'
    
    model, features = load_predictor(pattern_type, use_params=True)
    
    # Prepare feature vector
    X = []
    missing = []
    for feat in features:
        if feat in params:
            X.append(float(params[feat]))
        else:
            # Use reasonable defaults
            defaults = {
                'param_binary_threshold': 127,
                'param_hough_oblique_threshold': 50,
                'param_dilation_iterations': 2,
                'param_segment_width': 1200,
                'param_k_height': 1080,
                'param_ab_height': 3240,
                'param_angle_deg': 7.5,
            }
            X.append(defaults.get(feat, 0.0))
            missing.append(feat)
    
    X = np.array(X).reshape(1, -1)
    predicted = float(np.clip(model.predict(X)[0], 0.0, 1.0))
    
    return {
        'predicted_mIoU': predicted,
        'confidence': 1.0 - (len(missing) / len(features)),
        'pattern_type': pattern_type,
        'model_type': 'parameter_based',
        'features_used': dict(zip(features, X[0].tolist())),
        'missing_features': missing,
    }


def predict_miou(
    tunnel_id: str,
    detected_csv: str,
    final_csv: str,
    data_dir: str = 'data'
) -> Dict:
    """
    Predict mIoU from intrinsic metrics.
    
    Args:
        tunnel_id: Tunnel identifier (1-4, 2-2, 3-1, 4-1, 5-1)
        detected_csv: Path to detected.csv
        final_csv: Path to final.csv
        data_dir: Base data directory
        
    Returns:
        Dictionary with predicted_mIoU, confidence, and metrics used
    """
    import sys
    sys.path.insert(0, str(project_root))
    
    from bo4tun.intrinsic_metrics import (
        compute_all_metrics,
        compute_all_complex_metrics,
    )
    
    # Determine pattern type
    is_complex = tunnel_id in ['4-1', '5-1']
    pattern_type = 'complex' if is_complex else 'simple'
    
    # Compute intrinsic metrics
    base_metrics = compute_all_metrics(tunnel_id, detected_csv, final_csv, data_dir)
    
    if is_complex:
        complex_metrics = compute_all_complex_metrics(tunnel_id, detected_csv, final_csv, data_dir)
        # Add complex prefix
        for k, v in complex_metrics.items():
            base_metrics[f'complex_{k}'] = v
    
    # Load predictor
    model, features = load_predictor(pattern_type)
    
    # Prepare feature vector
    X = []
    missing_features = []
    for feat in features:
        if feat in base_metrics:
            X.append(base_metrics[feat])
        else:
            X.append(0.0)  # Default value
            missing_features.append(feat)
    
    X = np.array(X).reshape(1, -1)
    
    # Predict
    predicted_miou = float(model.predict(X)[0])
    
    # Clip to valid range
    predicted_miou = np.clip(predicted_miou, 0.0, 1.0)
    
    # Confidence estimate (based on feature completeness)
    confidence = 1.0 - (len(missing_features) / len(features))
    
    return {
        'predicted_mIoU': predicted_miou,
        'confidence': confidence,
        'pattern_type': pattern_type,
        'features_used': {f: base_metrics.get(f, 0.0) for f in features},
        'missing_features': missing_features,
    }


def predict_from_metrics(metrics: Dict, tunnel_id: str) -> Dict:
    """
    Predict mIoU from pre-computed metrics dictionary.
    
    Args:
        metrics: Dictionary of intrinsic metrics
        tunnel_id: Tunnel identifier
        
    Returns:
        Prediction result dictionary
    """
    is_complex = tunnel_id in ['4-1', '5-1']
    pattern_type = 'complex' if is_complex else 'simple'
    
    model, features = load_predictor(pattern_type)
    
    X = []
    missing = []
    for feat in features:
        if feat in metrics:
            X.append(metrics[feat])
        else:
            X.append(0.0)
            missing.append(feat)
    
    X = np.array(X).reshape(1, -1)
    predicted = float(np.clip(model.predict(X)[0], 0.0, 1.0))
    
    return {
        'predicted_mIoU': predicted,
        'confidence': 1.0 - (len(missing) / len(features)),
        'pattern_type': pattern_type,
        'missing_features': missing,
    }


# =============================================================================
# Predictor Formulas (Parameter-based - RECOMMENDED)
# =============================================================================

SIMPLE_PARAM_FORMULA = """
Simple Patterns (1-4, 2-2, 3-1) - n=198, CV Spearman=0.40

mIoU = -0.000968 * binary_threshold
     + 0.002320 * hough_oblique_threshold
     - 0.044161 * dilation_iterations
     - 0.001336 * segment_width
     + 0.000415 * k_height
     - 0.011179 * angle_deg
     + 1.6919
"""

COMPLEX_PARAM_FORMULA = """
Complex Patterns (4-1, 5-1) - n=93, CV Spearman=0.29

mIoU = -0.000250 * segment_width
     - 0.000225 * k_height
     + 0.000093 * ab_height
     - 0.003805 * angle_deg
     + 0.5812
"""

# Legacy formulas (intrinsic metrics based)
SIMPLE_FORMULA_LEGACY = """
mIoU = 0.0299 * det_midpoint_ratio
     + 0.0090 * det_real_detection_ratio
     - 0.0078 * det_x_spacing_cv
     + 0.0041 * sam_mask_fill_rate
     + 0.0006 * det_y_std
     + 0.4341
"""

COMPLEX_FORMULA_LEGACY = """
mIoU = 0.0707 * sam_ring_completeness
     + 0.0219 * sam_segment_height_cv
     - 0.0217 * sam_segment_width_cv
     - 0.0130 * sam_segment_area_cv
     + 0.0106 * sam_ring_coverage_cv
     + 0.1184 * sam_aspect_ratio_mean
     + 0.1544
"""


if __name__ == '__main__':
    # Test the predictor
    import sys
    
    if len(sys.argv) > 1:
        tunnel_id = sys.argv[1]
    else:
        tunnel_id = '2-2'
    
    detected = f'data/{tunnel_id}/detected.csv'
    final = f'data/{tunnel_id}/final.csv'
    
    if os.path.exists(final):
        result = predict_miou(tunnel_id, detected, final)
        print(f"\n=== Prediction for {tunnel_id} ===")
        print(f"Pattern type: {result['pattern_type']}")
        print(f"Predicted mIoU: {result['predicted_mIoU']:.4f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nFeatures used:")
        for feat, val in result['features_used'].items():
            print(f"  {feat}: {val:.4f}")
    else:
        print(f"File not found: {final}")
