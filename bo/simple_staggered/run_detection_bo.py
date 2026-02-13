"""
Bayesian Optimization for Detection Parameters with K-Position F1 Objective

Optimizes detection parameters (enhancing + line detection) to maximize
K-Position Weighted F1 score against ground truth K positions.

Uses forest_minimize (Random Forest surrogate) for high-dimensional search space (13 params).
Implements depth map caching to avoid re-running enhancing when only detection params change.
"""

import os
import sys
import json
import glob
import time
import argparse
import importlib.util
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real, Integer
from scipy.optimize import linear_sum_assignment

# Add project root to path
# BO script is now in: bo/{agent_type}/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Extract agent_type from directory name
# Script is in bo/{agent_type}/, so parent.name gives agent_type
DEFAULT_AGENT_TYPE = Path(__file__).parent.name

# Import detection functions
# Detection script is in: agents/{agent_type}/2_detection/
detection_dir = PROJECT_ROOT / 'agents' / DEFAULT_AGENT_TYPE / '2_detection'
sys.path.insert(0, str(detection_dir))

spec = importlib.util.spec_from_file_location(
    "detection",
    os.path.join(detection_dir, "2_detection.py")
)
detection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection_module)

run_detection = detection_module.run_detection
load_parameters = detection_module.load_parameters
get_param = detection_module.get_param


# =============================================================================
# Search Space Definition (13 parameters)
# =============================================================================

def get_detection_dimensions(
    tunnel_id: str = None,
    agent_type: str = DEFAULT_AGENT_TYPE
) -> Tuple[List, List[str], Dict]:
    """
    Define search space for detection parameters.
    
    Loads per-tunnel config from bo/{agent_type}/configs/detect_{tunnel_id}.json if it exists.
    If config exists:
        - Excludes parameters listed in fixed_params from search space
        - Overrides bounds from narrowed_bounds
    If no config: returns full 13D default space.
    
    Note: Negative angles are derived symmetrically from positive angles,
    reducing from 15 raw params to 13 tunable (or fewer if some are fixed).
    
    Args:
        tunnel_id: Tunnel identifier (e.g., '1-4')
        agent_type: Agent type (e.g., 'simple_staggered')
    
    Returns:
        Tuple of (dimensions list, parameter names list, fixed_params dict)
    """
    # Default bounds (full 13D space)
    default_bounds = {
        'curvature_neighbors': (8, 30),
        'depth_map_resolution': (0.003, 0.010),
        'interpolation_window': (3, 15),
        'target_distance_1': (0.05, 0.12),
        'binary_threshold': (80, 200),
        'dilation_kernel_size': (2, 5),
        'dilation_iterations': (1, 4),
        'hough_oblique_threshold': (20, 120),
        'hough_oblique_min_length': (40, 150),
        'hough_oblique_max_gap': (20, 80),
        'angle_min': (4.0, 7.0),
        'angle_max': (7.5, 12.0),
        'hough_vertical_threshold': (200, 800),
    }
    
    # Load per-tunnel config if it exists
    fixed_params = {}
    narrowed_bounds = {}
    
    if tunnel_id:
        config_file = PROJECT_ROOT / 'bo' / agent_type / 'configs' / f'detect_{tunnel_id}.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            fixed_params = config.get('fixed_params', {})
            narrowed_bounds = config.get('narrowed_bounds', {})
    
    # Build bounds dict (narrowed overrides default)
    bounds = default_bounds.copy()
    bounds.update(narrowed_bounds)
    
    # Build dimensions and param_names, excluding fixed params
    dimensions = []
    param_names = []
    
    param_defs = [
        ('curvature_neighbors', Integer, bounds['curvature_neighbors']),
        ('depth_map_resolution', Real, bounds['depth_map_resolution']),
        ('interpolation_window', Integer, bounds['interpolation_window']),
        ('target_distance_1', Real, bounds['target_distance_1']),
        ('binary_threshold', Integer, bounds['binary_threshold']),
        ('dilation_kernel_size', Integer, bounds['dilation_kernel_size']),
        ('dilation_iterations', Integer, bounds['dilation_iterations']),
        ('hough_oblique_threshold', Integer, bounds['hough_oblique_threshold']),
        ('hough_oblique_min_length', Integer, bounds['hough_oblique_min_length']),
        ('hough_oblique_max_gap', Integer, bounds['hough_oblique_max_gap']),
        ('angle_min', Real, bounds['angle_min']),
        ('angle_max', Real, bounds['angle_max']),
        ('hough_vertical_threshold', Integer, bounds['hough_vertical_threshold']),
    ]
    
    for name, param_type, (low, high) in param_defs:
        if name not in fixed_params:
            dimensions.append(param_type(low, high, name=name))
            param_names.append(name)
    
    return dimensions, param_names, fixed_params


def params_to_detection_json(params: List, param_names: List[str], fixed_params: Dict = None) -> Dict:
    """
    Convert BO parameters to detection JSON structure.
    
    Derives negative angles symmetrically from positive angles.
    Constructs target_distances as [td1, td1*0.5, 0.02].
    Merges fixed_params into output (fixed params are not in param_names/params).
    
    Args:
        params: List of BO-tuned parameter values
        param_names: List of parameter names corresponding to params
        fixed_params: Dict of fixed parameter values (not in BO search space)
    
    Returns:
        Complete detection parameters dict with all 15 fields
    """
    if fixed_params is None:
        fixed_params = {}
    
    param_dict = dict(zip(param_names, params))
    
    # Merge fixed params into param_dict (fixed params take precedence)
    param_dict.update(fixed_params)
    
    # Construct target_distances from single param
    td1 = param_dict['target_distance_1']
    target_distances = [td1, td1 * 0.5, 0.02]
    
    # Derive negative angles from positive
    angle_min = param_dict['angle_min']
    angle_max = param_dict['angle_max']
    
    return {
        'target_distances': target_distances,
        'curvature_neighbors': int(param_dict['curvature_neighbors']),
        'depth_map_resolution': float(param_dict['depth_map_resolution']),
        'interpolation_window': int(param_dict['interpolation_window']),
        'binary_threshold': int(param_dict['binary_threshold']),
        'dilation_kernel_size': int(param_dict['dilation_kernel_size']),
        'dilation_iterations': int(param_dict['dilation_iterations']),
        'hough_oblique_threshold': int(param_dict['hough_oblique_threshold']),
        'hough_oblique_min_length': int(param_dict['hough_oblique_min_length']),
        'hough_oblique_max_gap': int(param_dict['hough_oblique_max_gap']),
        'angle_positive_min': float(angle_min),
        'angle_positive_max': float(angle_max),
        'angle_negative_min': -float(angle_max),  # Symmetric
        'angle_negative_max': -float(angle_min),  # Symmetric
        'hough_vertical_threshold': int(param_dict['hough_vertical_threshold']),
    }


# =============================================================================
# K-Position F1 Score Computation
# =============================================================================

def compute_kposition_f1(
    detected: pd.DataFrame,
    gt: pd.DataFrame,
    threshold: float = 100.0
) -> Dict:
    """
    Compute K-Position Weighted F1 score using Hungarian matching.
    
    Args:
        detected: DataFrame with 'X', 'Y' columns (detected K positions)
        gt: DataFrame with 'X', 'Y' columns (ground truth K positions)
        threshold: Distance threshold in pixels for a "correct" match
    
    Returns:
        Dictionary with F1, precision, recall, position_bonus, and detailed metrics
    """
    if len(detected) == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'position_bonus': 0.0,
            'weighted_f1': 0.0,
            'tp': 0,
            'fp': len(detected),
            'fn': len(gt),
            'matched_distances': [],
            'mean_distance': 9999.0,  # Sentinel value for no matches
        }
    
    if len(gt) == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'position_bonus': 0.0,
            'weighted_f1': 0.0,
            'tp': 0,
            'fp': len(detected),
            'fn': 0,
            'matched_distances': [],
            'mean_distance': 9999.0,  # Sentinel value for no matches
        }
    
    # Build distance matrix
    n_gt = len(gt)
    n_det = len(detected)
    
    # Compute pairwise distances
    distances = np.zeros((n_gt, n_det))
    for i, (_, gt_row) in enumerate(gt.iterrows()):
        for j, (_, det_row) in enumerate(detected.iterrows()):
            dx = gt_row['X'] - det_row['X']
            dy = gt_row['Y'] - det_row['Y']
            distances[i, j] = np.sqrt(dx**2 + dy**2)
    
    # Hungarian matching (optimal assignment)
    row_indices, col_indices = linear_sum_assignment(distances)
    
    # Classify matches
    tp = 0
    matched_distances = []
    matched_gt_indices = set()
    matched_det_indices = set()
    
    for i, j in zip(row_indices, col_indices):
        dist = distances[i, j]
        if dist <= threshold:
            tp += 1
            matched_distances.append(dist)
            matched_gt_indices.add(i)
            matched_det_indices.add(j)
    
    # False negatives: GT positions not matched or matched beyond threshold
    fn = n_gt - len(matched_gt_indices)
    
    # False positives: Detected positions not matched or matched beyond threshold
    fp = n_det - len(matched_det_indices)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Position bonus: reward tighter matches
    if matched_distances:
        mean_distance = np.mean(matched_distances)
        position_bonus = max(0.0, 1.0 - mean_distance / threshold)
    else:
        position_bonus = 0.0
    
    # Weighted F1: F1 * (0.5 + 0.5 * position_bonus)
    weighted_f1 = f1 * (0.5 + 0.5 * position_bonus)
    
    return {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'position_bonus': float(position_bonus),
        'weighted_f1': float(weighted_f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'matched_distances': [float(d) for d in matched_distances],
        'mean_distance': float(np.mean(matched_distances)) if matched_distances else 9999.0,  # Sentinel value for no matches
    }


# =============================================================================
# Objective Function
# =============================================================================

class DetectionObjective:
    """
    Objective function that evaluates detection parameters using K-Position F1 score.
    Implements depth map caching to avoid re-running enhancing when only detection params change.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
        eval_offset: int = 0,
        agent_type: str = DEFAULT_AGENT_TYPE,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        self.agent_type = agent_type
        
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.params_dir = os.path.join(
            PROJECT_ROOT,
            'agents', agent_type, '2_detection',
            'parameters', tunnel_id
        )
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Load ground truth K positions dynamically from denoised.csv
        # K-block = segment label 1 (confirmed from BLOCK_TO_LABEL mapping)
        denoised_file = os.path.join(self.tunnel_dir, 'denoised.csv')
        if not os.path.exists(denoised_file):
            raise FileNotFoundError(f"denoised.csv not found: {denoised_file}. Run preprocessing first.")
        
        df = pd.read_csv(denoised_file)
        
        # Filter K-block points (segment=1) that were kept by preprocessing
        k_points = df[(df['segment'] == 1) & (df['pred'] != 0)]
        if len(k_points) == 0:
            raise ValueError(f"No K-block points (segment=1) found in denoised.csv for {tunnel_id}")
        
        # Compute per-ring centroids in cylindrical coordinates
        self.gt_cylindrical = k_points.groupby('ring').agg(
            h_mean=('h', 'mean'),
            theta_mean=('theta', 'mean'),
            count=('h', 'count')
        ).reset_index()
        
        # Store coordinate bounds from valid denoised points (used for pixel mapping)
        valid = df[df['pred'] != 0]
        self.h_bounds = (valid['h'].min(), valid['h'].max())
        self.theta_bounds = (valid['theta'].min(), valid['theta'].max())
        
        # Store GT positions will be computed dynamically in __call__ based on current depth map
        self.gt_positions = None  # Will be set dynamically
        
        # Get search space (loads config if exists)
        self.dimensions, self.param_names, self.fixed_params = get_detection_dimensions(
            tunnel_id=tunnel_id,
            agent_type=agent_type
        )
        
        # Tracking
        self.eval_offset = eval_offset
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        self.logs_dir = os.path.join(
            PROJECT_ROOT,
            'bo', agent_type, 'logs'
        )
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Depth map caching
        self.cached_depth_map_hash = None
        self.cached_enhancing_params = None
        
        if verbose:
            print(f"Detection BO for tunnel {tunnel_id}")
            print(f"GT K positions: {len(self.gt_cylindrical)} (from segment=1 centroids in denoised.csv)")
            print(f"GT coordinate bounds: h=[{self.h_bounds[0]:.4f}, {self.h_bounds[1]:.4f}], "
                  f"theta=[{self.theta_bounds[0]:.4f}, {self.theta_bounds[1]:.4f}]")
            print(f"Parameters: {len(self.param_names)} (fixed: {len(self.fixed_params)})")
            if self.fixed_params:
                print(f"Fixed parameters: {list(self.fixed_params.keys())}")
            print(f"Eval numbering starts at: {self.eval_offset + 1}")
            print(f"Logs directory: {self.logs_dir}")
    
    @property
    def global_eval_index(self) -> int:
        """Current global eval index (offset + local count)."""
        return self.eval_offset + self.eval_count
    
    def _get_enhancing_params_hash(self, param_dict: Dict) -> str:
        """Compute hash of enhancing parameters for caching."""
        enhancing_params = {
            'curvature_neighbors': param_dict['curvature_neighbors'],
            'depth_map_resolution': param_dict['depth_map_resolution'],
            'interpolation_window': param_dict['interpolation_window'],
            'target_distances': param_dict['target_distances'],
        }
        param_str = json.dumps(enhancing_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _should_skip_enhancing(self, param_dict: Dict) -> bool:
        """Check if we can skip enhancing (reuse cached depth map)."""
        enhancing_hash = self._get_enhancing_params_hash(param_dict)
        depth_map_file = os.path.join(self.tunnel_dir, 'depth_map_outlier.npy')
        
        if (self.cached_depth_map_hash == enhancing_hash and 
            os.path.exists(depth_map_file)):
            return True
        return False
    
    def __call__(self, params: List) -> float:
        """
        Evaluate detection parameters.
        
        Args:
            params: List of parameter values in order of param_names
        
        Returns:
            Negative weighted F1 score (for minimization)
        """
        self.eval_count += 1
        start_time = time.time()
        
        try:
            # Convert params to dict (merge with fixed params)
            param_dict = params_to_detection_json(params, self.param_names, self.fixed_params)
            
            # Check if we can skip enhancing
            skip_enhancing = self._should_skip_enhancing(param_dict)
            
            # Save parameters
            params_file = os.path.join(self.params_dir, 'parameters_detection.json')
            with open(params_file, 'w') as f:
                json.dump(param_dict, f, indent=4)
            
            # Run detection (suppress output)
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                detected_positions = run_detection(self.tunnel_id, self.data_dir)
            
            # Update cache if enhancing was run
            if not skip_enhancing:
                enhancing_hash = self._get_enhancing_params_hash(param_dict)
                self.cached_depth_map_hash = enhancing_hash
                self.cached_enhancing_params = {
                    k: param_dict[k] for k in ['curvature_neighbors', 'depth_map_resolution',
                                                'interpolation_window', 'target_distances']
                }
            
            # Convert GT from cylindrical (h, theta) to pixel coordinates using current depth map
            depth_map_file = os.path.join(self.tunnel_dir, 'depth_map_outlier.npy')
            if not os.path.exists(depth_map_file):
                raise FileNotFoundError(f"depth_map_outlier.npy not found after detection run: {depth_map_file}")
            
            depth_map = np.load(depth_map_file)
            H, W = depth_map.shape
            
            # Map GT centroids to pixel coordinates
            gt_pixels = pd.DataFrame({
                'X': (self.gt_cylindrical['h_mean'] - self.h_bounds[0]) / 
                     (self.h_bounds[1] - self.h_bounds[0]) * W,
                'Y': (self.gt_cylindrical['theta_mean'] - self.theta_bounds[0]) / 
                     (self.theta_bounds[1] - self.theta_bounds[0]) * H,
            })
            gt_pixels = gt_pixels.sort_values('X').reset_index(drop=True)
            
            # Compute F1 score
            results = compute_kposition_f1(detected_positions, gt_pixels)
            weighted_f1 = results['weighted_f1']
            
            runtime = time.time() - start_time
            
            # Track best
            if weighted_f1 > self.best_score:
                self.best_score = weighted_f1
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.global_eval_index}] New best F1: {weighted_f1:.4f} "
                          f"(P={results['precision']:.4f}, R={results['recall']:.4f}, "
                          f"TP={results['tp']}, FP={results['fp']}, FN={results['fn']}, "
                          f"mean_dist={results['mean_distance']:.1f}px)")
            
            # Log trial
            self._log_trial(
                param_dict,
                results,
                len(detected_positions),
                runtime,
                skip_enhancing,
            )
            
            # Record history
            self.history.append({
                'eval': self.global_eval_index,
                'params': param_dict,
                'weighted_f1': weighted_f1,
                'f1': results['f1'],
                'precision': results['precision'],
                'recall': results['recall'],
                'tp': results['tp'],
                'fp': results['fp'],
                'fn': results['fn'],
            })
            
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.global_eval_index}] F1: {weighted_f1:.4f}, "
                      f"TP={results['tp']}, FP={results['fp']}, FN={results['fn']}")
            
            return -weighted_f1  # Negative for minimization
            
        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose:
                print(f"  [Eval {self.global_eval_index}] Error: {e}")
            # Log failed trial
            self._log_trial(
                params_to_detection_json(params, self.param_names, self.fixed_params),
                None,
                0,
                runtime,
                False,
                error=str(e),
            )
            return 0.0  # Return worst score on error
    
    def _log_trial(
        self,
        params: Dict,
        results: Optional[Dict],
        num_detected: int,
        runtime: float,
        cached: bool,
        error: Optional[str] = None,
    ):
        """Log trial to JSON file."""
        global_idx = self.global_eval_index
        trial_id = f"detect_{self.tunnel_id}_{global_idx:03d}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        log_data = {
            'schema_version': 'r4tun.detection.v1',
            'trial': {
                'trial_id': trial_id,
                'timestamp_utc': timestamp,
                'tunnel_id': self.tunnel_id,
                'assembly_type': self.agent_type,
            },
            'params': params,
        }
        
        if error:
            log_data['trace'] = {'warnings': [f"Error: {error}"]}
            log_data['bo'] = {
                'objective_name': 'kposition_weighted_f1',
                'objective_value': 0.0,
                'eval_index': global_idx,
                'runtime_sec': runtime,
                'is_feasible': False,
                'cached_enhancing': cached,
            }
        else:
            log_data['outputs'] = {
                'num_detected': num_detected,
                'num_gt': len(self.gt_cylindrical),
                'confusion': {
                    'tp': results['tp'],
                    'fp': results['fp'],
                    'fn': results['fn'],
                },
                'metrics': {
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1'],
                    'position_bonus': results['position_bonus'],
                    'weighted_f1': results['weighted_f1'],
                    'mean_distance_px': results['mean_distance'],
                },
                'matched_distances_px': results['matched_distances'],
            }
            log_data['bo'] = {
                'objective_name': 'kposition_weighted_f1',
                'objective_value': float(results['weighted_f1']),
                'eval_index': global_idx,
                'runtime_sec': float(runtime),
                'is_feasible': True,
                'cached_enhancing': cached,
            }
        
        # Save log file
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def save_best_params(self) -> Optional[str]:
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        params_file = os.path.join(self.params_dir, 'parameters_detection.json')
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        return params_file


# =============================================================================
# Utilities
# =============================================================================

def find_max_trial_index(logs_dir: str, tunnel_id: str) -> int:
    """Find the highest trial index from existing log files."""
    pattern = os.path.join(logs_dir, f"detect_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    max_idx = 0
    for f in log_files:
        basename = os.path.basename(f)
        # e.g. detect_1-4_035.json -> 035
        try:
            idx = int(basename.split('_')[-1].replace('.json', ''))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return max_idx


def load_best_from_logs(
    logs_dir: str,
    tunnel_id: str,
    agent_type: str = DEFAULT_AGENT_TYPE,
    fixed_params: Dict = None
) -> Optional[Tuple[List[float], float]]:
    """
    Load the best trial from existing logs to use as warm-start x0/y0.
    Falls back to current parameters_detection.json if no logs exist.
    
    Args:
        logs_dir: Directory containing BO log files
        tunnel_id: Tunnel identifier
        agent_type: Agent type
        fixed_params: Dict of fixed parameters (to exclude from warm-start vector)
    
    Returns:
        Tuple of (param_values_list, negative_weighted_f1) or None if no params found.
        param_values_list is in the order of get_detection_dimensions() (excluding fixed params).
    """
    if fixed_params is None:
        fixed_params = {}
    
    pattern = os.path.join(logs_dir, f"detect_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    best_f1 = -1
    best_params = None
    
    # First, try to load from previous BO logs
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if 'bo' not in data or 'objective_value' not in data['bo']:
            continue
        
        f1 = data['bo']['objective_value']
        if f1 > best_f1:
            best_f1 = f1
            best_params = data.get('params', {})
    
    # If no logs found, try to load from current parameters file
    if best_params is None or best_f1 <= 0:
        params_file = os.path.join(
            PROJECT_ROOT,
            'agents', agent_type, '2_detection',
            'parameters', tunnel_id, 'parameters_detection.json'
        )
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                best_params = json.load(f)
            # Use a conservative F1 estimate (0.5) for parameters file
            # This allows BO to explore but starts from a reasonable point
            best_f1 = 0.5
            print(f"  No previous BO logs found, using current parameters_detection.json as warm-start")
    
    if best_params is None:
        return None
    
    # Build param list in dimension order, EXCLUDING fixed params
    # Map from full param names to extraction logic
    param_extractors = {
        'curvature_neighbors': lambda p: p.get('curvature_neighbors', 10),
        'depth_map_resolution': lambda p: p.get('depth_map_resolution', 0.008),
        'interpolation_window': lambda p: p.get('interpolation_window', 9),
        'target_distance_1': lambda p: p.get('target_distances', [0.08, 0.04, 0.02])[0],
        'binary_threshold': lambda p: p.get('binary_threshold', 149),
        'dilation_kernel_size': lambda p: p.get('dilation_kernel_size', 2),
        'dilation_iterations': lambda p: p.get('dilation_iterations', 1),
        'hough_oblique_threshold': lambda p: p.get('hough_oblique_threshold', 69),
        'hough_oblique_min_length': lambda p: p.get('hough_oblique_min_length', 99),
        'hough_oblique_max_gap': lambda p: p.get('hough_oblique_max_gap', 60),
        'angle_min': lambda p: p.get('angle_positive_min', 5.509),
        'angle_max': lambda p: p.get('angle_positive_max', 8.652),
        'hough_vertical_threshold': lambda p: p.get('hough_vertical_threshold', 574),
    }
    
    # Get current search space param names (to know what to include)
    _, param_names, _ = get_detection_dimensions(tunnel_id=tunnel_id, agent_type=agent_type)
    
    # Build param_values list only for params in current search space
    param_values = []
    for param_name in param_names:
        if param_name in param_extractors:
            param_values.append(param_extractors[param_name](best_params))
        else:
            # Fallback (shouldn't happen if param_names matches param_extractors)
            param_values.append(0.0)
    
    return param_values, -best_f1  # negative for minimization


# =============================================================================
# Main Optimization
# =============================================================================

def run_detection_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 80,
    n_initial_points: int = 15,
    verbose: bool = True,
    agent_type: str = DEFAULT_AGENT_TYPE,
) -> Dict:
    """Run Bayesian Optimization for detection parameters."""
    
    print(f"\n{'='*70}")
    print(f"DETECTION BAYESIAN OPTIMIZATION - Tunnel {tunnel_id} ({agent_type})")
    print(f"{'='*70}")
    
    logs_dir = os.path.join(
        PROJECT_ROOT,
        'bo', agent_type, 'logs'
    )
    os.makedirs(logs_dir, exist_ok=True)
    
    # Determine eval offset from existing logs
    eval_offset = find_max_trial_index(logs_dir, tunnel_id)
    
    # Initialize objective
    objective = DetectionObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
        eval_offset=eval_offset,
        agent_type=agent_type,
    )
    
    print(f"\nSearch space: {len(objective.param_names)} parameters")
    print(f"N calls: {n_calls}, N initial: {n_initial_points}")
    print(f"Objective: K-Position Weighted F1 (threshold=100px)")
    print(f"Algorithm: forest_minimize (Random Forest surrogate)")
    
    # Warm-start from best previous trial or current parameters file
    x0 = None
    y0 = None
    warm_start = load_best_from_logs(logs_dir, tunnel_id, agent_type, objective.fixed_params)
    if warm_start is not None:
        x0_vals, y0_val = warm_start
        # Clamp warm-start values to current search space bounds
        clamped_x0 = []
        for i, (name, val) in enumerate(zip(objective.param_names, x0_vals)):
            dim = objective.dimensions[i]
            # Get bounds from dimension
            if hasattr(dim, 'low') and hasattr(dim, 'high'):
                clamped_val = max(dim.low, min(dim.high, val))
                clamped_x0.append(clamped_val)
            else:
                clamped_x0.append(val)
        
        x0 = [clamped_x0]
        y0 = [y0_val]
        source = "previous BO logs" if eval_offset > 0 else "current parameters_detection.json"
        print(f"\nWarm-starting from {source} (estimated F1={-y0_val:.4f}):")
        for name, val, clamped in zip(objective.param_names, x0_vals, clamped_x0):
            if abs(val - clamped) > 1e-6:
                print(f"  {name}: {val:.4f} -> {clamped:.4f} (clamped to bounds)")
            else:
                print(f"  {name}: {val:.4f}")
    
    # Run optimization
    print(f"\nStarting optimization...")
    result = forest_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        x0=x0,
        y0=y0,
        random_state=42,
        verbose=False,
    )
    
    # Results
    best_params = params_to_detection_json(result.x, objective.param_names, objective.fixed_params)
    best_f1 = -result.fun  # Negate back
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best weighted F1 score: {best_f1:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        if isinstance(value, list):
            print(f"  {name}: {value}")
        elif isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")
    
    # Save best parameters
    filepath = objective.save_best_params()
    if filepath:
        print(f"\nSaved parameters to: {filepath}")
    
    return {
        'tunnel_id': tunnel_id,
        'best_f1': best_f1,
        'best_params': best_params,
        'n_evaluations': objective.eval_count,
        'history': objective.history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection BO with K-Position F1 objective')
    parser.add_argument('tunnel_id', type=str, help='Tunnel identifier (e.g., 1-4)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--n-calls', type=int, default=80, help='Total evaluations')
    parser.add_argument('--n-initial', type=int, default=15, help='Initial random points')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--agent-type', type=str, default='simple_staggered',
                       choices=['simple_staggered', 'continuous', 'complex_staggered'],
                       help='Agent type (default: simple_staggered)')
    
    args = parser.parse_args()
    
    run_detection_bo(
        tunnel_id=args.tunnel_id,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        verbose=args.verbose,
        agent_type=args.agent_type,
    )
