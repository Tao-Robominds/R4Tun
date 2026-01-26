"""
Detection-Only Bayesian Optimization

Optimizes detection parameters to match ground truth K positions.
Does NOT run SAM - only evaluates detection quality against GT.

Usage:
    python -m p4tun.bo.detection_bo --tunnel 3-1 --n-calls 50
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer
from skopt.callbacks import DeltaYStopper

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Detection Search Space (focused on line detection)
# =============================================================================

DETECTION_SEARCH_SPACE = {
    # Preprocessing
    'binary_threshold': Integer(80, 140, name='binary_threshold'),
    'dilation_kernel_size': Integer(2, 5, name='dilation_kernel_size'),
    'dilation_iterations': Integer(1, 4, name='dilation_iterations'),
    
    # Hough oblique line detection
    'hough_oblique_threshold': Integer(20, 60, name='hough_oblique_threshold'),
    'hough_oblique_min_length': Integer(40, 120, name='hough_oblique_min_length'),
    'hough_oblique_max_gap': Integer(30, 80, name='hough_oblique_max_gap'),
    
    # Angle ranges for oblique lines (degrees) - based on analysis showing peaks at 5-10 degrees
    'angle_positive_min': Real(4.0, 7.0, name='angle_positive_min'),
    'angle_positive_max': Real(8.0, 12.0, name='angle_positive_max'),
    
    # Hough horizontal line detection
    'hough_horizontal_threshold': Integer(30, 70, name='hough_horizontal_threshold'),
    'hough_horizontal_min_length': Integer(60, 130, name='hough_horizontal_min_length'),
    'hough_horizontal_max_gap': Integer(5, 25, name='hough_horizontal_max_gap'),
    
    # Hough vertical line detection - lower threshold to detect more vertical lines
    'hough_vertical_threshold': Integer(200, 600, name='hough_vertical_threshold'),
    
    # Line processing
    'merge_distance_threshold': Integer(2, 8, name='merge_distance_threshold'),
    'merge_close_threshold': Integer(4, 12, name='merge_close_threshold'),
}


def get_detection_dimensions():
    """Get search space dimensions and names."""
    dimensions = list(DETECTION_SEARCH_SPACE.values())
    names = list(DETECTION_SEARCH_SPACE.keys())
    return dimensions, names


def params_to_detection_json(params: List, names: List[str]) -> Dict:
    """Convert BO parameters to detection JSON structure."""
    param_dict = dict(zip(names, params))
    
    return {
        'preprocessing': {
            'binary_threshold': int(param_dict.get('binary_threshold', 107)),
            'dilation_kernel_size': int(param_dict.get('dilation_kernel_size', 3)),
            'dilation_iterations': int(param_dict.get('dilation_iterations', 2)),
        },
        'hough_oblique': {
            'threshold': int(param_dict.get('hough_oblique_threshold', 37)),
            'min_length': int(param_dict.get('hough_oblique_min_length', 89)),
            'max_gap': int(param_dict.get('hough_oblique_max_gap', 47)),
            'angle_positive_min': float(param_dict.get('angle_positive_min', 5.24)),
            'angle_positive_max': float(param_dict.get('angle_positive_max', 8.36)),
            'angle_negative_min': -float(param_dict.get('angle_positive_max', 8.36)),
            'angle_negative_max': -float(param_dict.get('angle_positive_min', 5.24)),
        },
        'hough_horizontal': {
            'threshold': int(param_dict.get('hough_horizontal_threshold', 45)),
            'min_length': int(param_dict.get('hough_horizontal_min_length', 108)),
            'max_gap': int(param_dict.get('hough_horizontal_max_gap', 15)),
            'angle_tolerance': 1,
        },
        'hough_vertical': {
            'threshold': int(param_dict.get('hough_vertical_threshold', 500)),
        },
        'line_processing': {
            'merge_distance_threshold': int(param_dict.get('merge_distance_threshold', 5)),
            'merge_close_threshold': int(param_dict.get('merge_close_threshold', 6)),
        },
        'physical_constants': {
            'resolution': 0.005,
            'k_height_mm': 1079.92,
            'ab_height_mm': 3239.77,
        },
    }


# =============================================================================
# Detection Objective Function
# =============================================================================

class DetectionObjective:
    """
    Objective function that evaluates detection against ground truth K positions.
    Uses actual detection script for accurate evaluation.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
        use_script: bool = True,  # Use actual detection script
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        self.use_script = use_script
        
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.params_dir = os.path.join(PROJECT_ROOT, 'p4tun', 'parameters', tunnel_id)
        
        # Load ground truth
        self.gt_positions = self._load_gt_positions()
        self.ring_count = len(self.gt_positions)
        
        # Load depth map
        self.depth_outlier = np.load(os.path.join(self.tunnel_dir, 'depth_map_outlier.npy'))
        self.image_height, self.image_width = self.depth_outlier.shape
        
        # Get search space
        self.dimensions, self.param_names = get_detection_dimensions()
        
        # Detection script path
        self.detection_script = os.path.join(PROJECT_ROOT, 'p4tun', '4-1_detection.py')
        
        # Tracking
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        
        if verbose:
            print(f"Loaded GT with {self.ring_count} K positions")
            print(f"Image size: {self.image_width} x {self.image_height}")
            print(f"GT Y range: {self.gt_positions['Y'].min():.1f} - {self.gt_positions['Y'].max():.1f}")
            print(f"Using {'script' if use_script else 'inline'} detection")
    
    def _load_gt_positions(self) -> pd.DataFrame:
        """Load ground truth K positions."""
        gt_path = os.path.join(self.tunnel_dir, 'detected_gt.csv')
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        
        df = pd.read_csv(gt_path)
        return df.sort_values('X').reset_index(drop=True)
    
    def __call__(self, params: List) -> float:
        """
        Evaluate detection parameters against ground truth.
        
        Returns negative score (for minimization).
        """
        self.eval_count += 1
        
        try:
            # Convert params to detection config
            detection_config = params_to_detection_json(params, self.param_names)
            
            # Save parameters
            self._save_temp_params(detection_config)
            
            # Run detection
            if self.use_script:
                detected_positions = self._run_detection_script()
            else:
                detected_positions = self._run_detection(detection_config)
            
            # Calculate score
            score = self._calculate_score(detected_positions)
            
            # Track best
            param_dict = dict(zip(self.param_names, params))
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.eval_count}] New best: {score:.4f} (n={len(detected_positions)})")
            
            # Record history
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'score': score,
                'n_detected': len(detected_positions),
            })
            
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.eval_count}] Score: {score:.4f}, Detected: {len(detected_positions)}")
            
            return -score  # Negative for minimization
            
        except Exception as e:
            if self.verbose:
                print(f"  [Eval {self.eval_count}] Error: {e}")
            return 0.0
    
    def _save_temp_params(self, config: Dict):
        """Save parameters to JSON file for detection script."""
        os.makedirs(self.params_dir, exist_ok=True)
        filepath = os.path.join(self.params_dir, 'parameters_detection.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _run_detection_script(self) -> pd.DataFrame:
        """Run the actual detection script and read results."""
        import subprocess
        
        # Run detection script
        venv_python = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python')
        cmd = [venv_python, '-m', 'p4tun.4-1_detection', self.tunnel_id, '--data-dir', self.data_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT,
        )
        
        # Read detected.csv
        detected_path = os.path.join(self.tunnel_dir, 'detected.csv')
        if os.path.exists(detected_path):
            return pd.read_csv(detected_path)
        else:
            return pd.DataFrame(columns=['Type', 'X', 'Y'])
    
    def _run_detection(self, config: Dict) -> pd.DataFrame:
        """Run detection with given config (inline implementation)."""
        import cv2
        
        # Extract parameters
        binary_threshold = config['preprocessing']['binary_threshold']
        dilation_kernel_size = config['preprocessing']['dilation_kernel_size']
        dilation_iterations = config['preprocessing']['dilation_iterations']
        
        hough_oblique_threshold = config['hough_oblique']['threshold']
        hough_oblique_min_length = config['hough_oblique']['min_length']
        hough_oblique_max_gap = config['hough_oblique']['max_gap']
        angle_pos_min = config['hough_oblique']['angle_positive_min']
        angle_pos_max = config['hough_oblique']['angle_positive_max']
        angle_neg_min = config['hough_oblique']['angle_negative_min']
        angle_neg_max = config['hough_oblique']['angle_negative_max']
        
        hough_vert_threshold = config['hough_vertical']['threshold']
        merge_distance = config['line_processing']['merge_distance_threshold']
        merge_close = config['line_processing'].get('merge_close_threshold', 6)
        
        resolution = config['physical_constants']['resolution']
        k_height_mm = config['physical_constants']['k_height_mm']
        ab_height_mm = config['physical_constants']['ab_height_mm']
        k_height_px = k_height_mm / (resolution * 1000)
        ab_height_px = ab_height_mm / (resolution * 1000)
        
        L, W = self.image_height, self.image_width
        
        # Preprocessing
        binary_map = np.where(np.isnan(self.depth_outlier), 0, 255).astype(np.uint8)
        _, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Canny edges on normalized depth
        depth_valid = self.depth_outlier[~np.isnan(self.depth_outlier)]
        if len(depth_valid) > 0:
            depth_min, depth_max = depth_valid.min(), depth_valid.max()
            if depth_max > depth_min:
                out = np.zeros_like(self.depth_outlier, dtype=np.float64)
                valid = ~np.isnan(self.depth_outlier)
                out[valid] = (self.depth_outlier[valid] - depth_min) / (depth_max - depth_min) * 255
                depth_normalized = out.astype(np.uint8)
                canny_edges = cv2.Canny(depth_normalized, 50, 150)
                combined_edges = cv2.bitwise_or(binary_image, canny_edges)
            else:
                combined_edges = binary_image
        else:
            combined_edges = binary_image
        
        # Dilation
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)
        
        # Detect oblique lines
        lines_oblique = cv2.HoughLinesP(
            dilated_edges, 1, np.pi/180,
            hough_oblique_threshold,
            minLineLength=hough_oblique_min_length,
            maxLineGap=hough_oblique_max_gap
        )
        
        # Detect vertical lines
        lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi/180, hough_vert_threshold)
        
        # Separate positive and negative slope lines
        positive_lines = []
        negative_lines = []
        
        if lines_oblique is not None:
            for line in lines_oblique:
                x1, y1, x2, y2 = line[0]
                if x1 > x2:
                    x1, x2, y1, y2 = x2, x1, y2, y1
                angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
                
                if angle_pos_min <= angle <= angle_pos_max:
                    positive_lines.append((x1, y1, x2, y2))
                elif angle_neg_min <= angle <= angle_neg_max:
                    negative_lines.append((x1, y1, x2, y2))
        
        # Process vertical lines
        merged_vertical = []
        if lines_vertical is not None:
            lines_vert_2d = lines_vertical[:, 0]
            for rho, theta in lines_vert_2d:
                if abs(theta) <= 0.5 * np.pi / 180:
                    x_pos = rho * np.cos(theta)
                    merged = False
                    for i, (mrho, mtheta) in enumerate(merged_vertical):
                        mx = mrho * np.cos(mtheta)
                        if abs(x_pos - mx) < merge_distance:
                            merged_vertical[i] = ((rho + mrho) / 2, (theta + mtheta) / 2)
                            merged = True
                            break
                    if not merged:
                        merged_vertical.append((rho, theta))
            merged_vertical.sort(key=lambda l: l[0])
        
        # Compute ring centers
        if merged_vertical:
            mid_lines = []
            for i in range(len(merged_vertical) - 1):
                rho1, theta1 = merged_vertical[i]
                rho2, theta2 = merged_vertical[i + 1]
                new_rho = (rho1 + rho2) / 2
                new_theta = (theta1 + theta2) / 2
                x_pos = np.cos(new_theta) * new_rho
                mid_lines.append((x_pos, new_theta))
            
            if mid_lines:
                x_positions = [x for x, _ in mid_lines]
                distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                avg_distance = np.mean(np.abs(distances)) if distances else W / self.ring_count
                
                # Extend to cover all rings
                all_mid = list(mid_lines)
                leftmost_x = mid_lines[0][0]
                x = leftmost_x - avg_distance
                while x >= 0:
                    all_mid.insert(0, (x, mid_lines[0][1]))
                    x -= avg_distance
                
                rightmost_x = mid_lines[-1][0]
                x = rightmost_x + avg_distance
                while x <= W:
                    all_mid.append((x, mid_lines[-1][1]))
                    x += avg_distance
                
                ring_centers = sorted(set(x for x, _ in all_mid))
            else:
                ring_centers = [(i + 0.5) * W / self.ring_count for i in range(self.ring_count)]
        else:
            ring_centers = [(i + 0.5) * W / self.ring_count for i in range(self.ring_count)]
        
        # Calculate K positions
        def line_intersection(vertical_x, segment):
            x1, y1, x2, y2 = segment
            if x1 == x2:
                return None
            if min(x1, x2) <= vertical_x <= max(x1, x2):
                t = (vertical_x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
            return None
        
        def merge_points(points, threshold):
            if not points:
                return []
            pts = np.array(points, dtype=np.float64)
            if len(pts) == 1:
                return [float(pts[0])]
            merged = []
            while len(pts) > 0:
                p = pts[0]
                close_mask = np.abs(pts - p) < threshold
                merged.append(float(np.mean(pts[close_mask])))
                pts = pts[~close_mask]
            return merged
        
        adjusted_points = []
        for vertical_x in ring_centers:
            pos_ints = [line_intersection(vertical_x, seg) for seg in positive_lines]
            pos_ints = [y for y in pos_ints if y is not None]
            
            neg_ints = [line_intersection(vertical_x, seg) for seg in negative_lines]
            neg_ints = [y for y in neg_ints if y is not None]
            
            merge_pos = merge_points(pos_ints, merge_close)
            merge_neg = merge_points(neg_ints, merge_close)
            
            if merge_pos and merge_neg:
                midpoint_y = (merge_pos[0] + merge_neg[0]) / 2
                adjusted_points.append({'Type': 'midpoint', 'X': vertical_x, 'Y': midpoint_y})
            elif merge_pos:
                y = merge_pos[0] - 0.5 * k_height_px
                adjusted_points.append({'Type': 'positive_slope', 'X': vertical_x, 'Y': y})
            elif merge_neg:
                y = merge_neg[0] + 0.5 * k_height_px
                adjusted_points.append({'Type': 'negative_slope', 'X': vertical_x, 'Y': y})
            else:
                if adjusted_points:
                    last_y = adjusted_points[-1]['Y']
                    adjusted_points.append({'Type': 'assume', 'X': vertical_x, 'Y': last_y})
                else:
                    adjusted_points.append({'Type': 'default', 'X': vertical_x, 'Y': L / 2})
        
        return pd.DataFrame(adjusted_points)
    
    def _calculate_score(self, detected: pd.DataFrame) -> float:
        """
        Calculate score based on how well detected positions match GT.
        
        Score components:
        1. Position error (lower is better)
        2. Count penalty (should have exactly ring_count positions)
        """
        if len(detected) == 0:
            return 0.0
        
        gt = self.gt_positions
        
        # Match detected to GT using Hungarian algorithm (or greedy)
        # For simplicity, use nearest neighbor matching
        detected_sorted = detected.sort_values('X').reset_index(drop=True)
        gt_sorted = gt.sort_values('X').reset_index(drop=True)
        
        # Calculate position errors
        n_gt = len(gt_sorted)
        n_det = len(detected_sorted)
        
        # Penalize wrong count
        count_penalty = abs(n_det - n_gt) * 50  # 50 pixel penalty per extra/missing
        
        # Calculate X and Y errors
        if n_det >= n_gt:
            # More detected than GT - match each GT to nearest detected
            total_error = 0
            for i in range(n_gt):
                gt_x, gt_y = gt_sorted.iloc[i]['X'], gt_sorted.iloc[i]['Y']
                
                # Find nearest detected
                distances = np.sqrt(
                    (detected_sorted['X'] - gt_x)**2 + 
                    (detected_sorted['Y'] - gt_y)**2
                )
                min_dist = distances.min()
                total_error += min_dist
        else:
            # Fewer detected than GT - match each detected to nearest GT
            total_error = 0
            for i in range(n_det):
                det_x, det_y = detected_sorted.iloc[i]['X'], detected_sorted.iloc[i]['Y']
                
                distances = np.sqrt(
                    (gt_sorted['X'] - det_x)**2 + 
                    (gt_sorted['Y'] - det_y)**2
                )
                min_dist = distances.min()
                total_error += min_dist
            
            # Add penalty for missing detections
            total_error += (n_gt - n_det) * 100  # 100 pixel penalty per missing
        
        # Average error
        avg_error = total_error / max(n_gt, n_det)
        
        # Add count penalty
        total_error_with_penalty = avg_error + count_penalty
        
        # Convert to score (higher is better)
        # Max error we'd consider reasonable is ~500 pixels
        max_error = 500
        score = max(0, 1 - total_error_with_penalty / max_error)
        
        return score
    
    def save_best_params(self):
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Convert to full config
        config = params_to_detection_json(
            [self.best_params[n] for n in self.param_names],
            self.param_names
        )
        
        filepath = os.path.join(self.params_dir, 'parameters_detection.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        return filepath


# =============================================================================
# Main Optimization
# =============================================================================

def run_detection_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 50,
    n_initial: int = 10,
    verbose: bool = True,
    optimizer: str = 'gp',
) -> Dict:
    """Run Bayesian Optimization for detection parameters."""
    
    print(f"\n{'='*70}")
    print(f"DETECTION BAYESIAN OPTIMIZATION - Tunnel {tunnel_id}")
    print(f"{'='*70}")
    
    # Initialize objective
    objective = DetectionObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
    )
    
    print(f"\nSearch space: {len(objective.param_names)} parameters")
    print(f"N calls: {n_calls}, N initial: {n_initial}")
    
    # Select optimizer
    minimize_func = gp_minimize if optimizer == 'gp' else forest_minimize
    
    # Run optimization
    print(f"\nStarting optimization...")
    result = minimize_func(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=False,
        callback=[DeltaYStopper(delta=0.001, n_best=15)],
    )
    
    # Results
    best_params = dict(zip(objective.param_names, result.x))
    best_score = -result.fun
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best score: {best_score:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    
    # Save best parameters
    filepath = objective.save_best_params()
    if filepath:
        print(f"\nSaved parameters to: {filepath}")
    
    # Save results
    output_dir = os.path.join(PROJECT_ROOT, 'p4tun', 'bo', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'tunnel_id': tunnel_id,
        'stage': 'detection',
        'best_score': best_score,
        'best_params': best_params,
        'n_calls': n_calls,
        'history': objective.history,
        'timestamp': timestamp,
    }
    
    results_path = os.path.join(output_dir, f'{tunnel_id}_detection_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"Saved results to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Detection-only Bayesian Optimization')
    parser.add_argument('--tunnel', '-t', required=True, help='Tunnel ID (e.g., 3-1)')
    parser.add_argument('--n-calls', '-n', type=int, default=50, help='Number of evaluations')
    parser.add_argument('--n-initial', type=int, default=10, help='Initial random points')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--optimizer', '-o', default='gp', choices=['gp', 'forest'])
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    
    run_detection_bo(
        tunnel_id=args.tunnel,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        verbose=not args.quiet,
        optimizer=args.optimizer,
    )


if __name__ == '__main__':
    main()
