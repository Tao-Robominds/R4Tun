"""
Objective Function for Bayesian Optimization

Runs the detection + SAM pipeline and evaluates performance using mIoU.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Lazy imports to avoid circular import issues
_search_space_funcs = None

def _get_search_space_funcs():
    """Lazy import search space functions."""
    global _search_space_funcs
    if _search_space_funcs is None:
        from p4tun.bo.search_space import (
            params_to_detection_dict, 
            params_to_sam_dict, 
            save_parameters,
            get_search_space
        )
        _search_space_funcs = {
            'params_to_detection_dict': params_to_detection_dict,
            'params_to_sam_dict': params_to_sam_dict,
            'save_parameters': save_parameters,
            'get_search_space': get_search_space,
        }
    return _search_space_funcs


class PipelineObjective:
    """
    Objective function for Bayesian Optimization.
    
    Runs detection → SAM → evaluation pipeline and returns negative mIoU
    (negative because skopt minimizes).
    """
    
    def __init__(
        self, 
        tunnel_id: str,
        stage: str = 'combined',
        data_dir: str = 'data',
        metric: str = 'mIoU',
        verbose: bool = True,
        timeout: int = 600,
    ):
        """
        Initialize the objective function.
        
        Args:
            tunnel_id: Tunnel identifier (e.g., '4-1', '2-2')
            stage: Which parameters to optimize ('detection', 'sam', 'combined')
            data_dir: Base data directory
            metric: Evaluation metric ('mIoU', 'OA', 'F1')
            verbose: Print progress information
            timeout: Timeout for each pipeline run (seconds)
        """
        self.tunnel_id = tunnel_id
        self.stage = stage
        self.data_dir = data_dir
        self.metric = metric
        self.verbose = verbose
        self.timeout = timeout
        
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        
        # Get parameter names for this stage (lazy import)
        funcs = _get_search_space_funcs()
        _, self.param_names = funcs['get_search_space'](stage)
        
        # Script paths
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.detection_script = os.path.join(self.script_dir, '4-1_detection.py')
        self.sam_script = os.path.join(self.script_dir, '4-2_sam.py')
        self.eval_script = os.path.join(self.script_dir, 'evaluation.py')
        
        # Path to segment-anything module
        self.segment_anything_path = os.path.join(self.script_dir, 'segment-anything')
        
        # Project root directory
        self.project_root = os.path.dirname(self.script_dir)
        
        # Verify scripts exist
        for script in [self.detection_script, self.sam_script, self.eval_script]:
            if not os.path.exists(script):
                raise FileNotFoundError(f"Script not found: {script}")
    
    def __call__(self, params: List) -> float:
        """
        Evaluate a set of parameters.
        
        Args:
            params: List of parameter values from the optimizer
        
        Returns:
            Negative score (mIoU, OA, or F1) - negative because we minimize
        """
        self.eval_count += 1
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation {self.eval_count}")
            print(f"{'='*60}")
        
        try:
            # Convert parameters to dictionaries
            param_dict = dict(zip(self.param_names, params))
            
            if self.verbose:
                print(f"Parameters: {param_dict}")
            
            # Update parameter files
            self._update_parameters(params)
            
            # Run pipeline
            if self.stage in ['detection', 'combined']:
                self._run_detection()
            
            if self.stage in ['sam', 'combined']:
                self._run_sam()
            
            # Evaluate
            metrics = self._evaluate()
            
            # Get score
            score = metrics.get(self.metric, 0.0)
            
            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"*** New best {self.metric}: {score:.4f} ***")
            
            # Record history
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'score': score,
                'metrics': metrics
            })
            
            if self.verbose:
                print(f"Score ({self.metric}): {score:.4f}")
            
            return -score  # Negative because skopt minimizes
            
        except Exception as e:
            print(f"Error in evaluation {self.eval_count}: {e}")
            return 0.0  # Return worst case (will become 0 after negation)
    
    def _update_parameters(self, params: List):
        """Update parameter JSON files with new values."""
        param_dict = dict(zip(self.param_names, params))
        funcs = _get_search_space_funcs()
        
        if self.stage in ['detection', 'combined']:
            detection_dict = funcs['params_to_detection_dict'](params, self.param_names)
            
            # Load existing parameters and update
            existing = self._load_existing_params('detection')
            existing.update(detection_dict)
            
            filepath = funcs['save_parameters'](existing, self.tunnel_id, 'detection')
            if self.verbose:
                print(f"Updated detection parameters: {filepath}")
        
        if self.stage in ['sam', 'combined']:
            sam_dict = funcs['params_to_sam_dict'](params, self.param_names)
            
            # Load existing parameters and update
            existing = self._load_existing_params('sam')
            existing.update(sam_dict)
            
            filepath = funcs['save_parameters'](existing, self.tunnel_id, 'sam')
            if self.verbose:
                print(f"Updated SAM parameters: {filepath}")
    
    def _load_existing_params(self, stage: str) -> Dict:
        """Load existing parameters from JSON file."""
        params_dir = os.path.join(self.script_dir, 'parameters', self.tunnel_id)
        filepath = os.path.join(params_dir, f'parameters_{stage}.json')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        # Try sample
        sample_path = os.path.join(self.script_dir, 'parameters', 'sample', f'parameters_{stage}.json')
        if os.path.exists(sample_path):
            with open(sample_path, 'r') as f:
                return json.load(f)
        
        return {}
    
    def _run_detection(self):
        """Run detection pipeline."""
        if self.verbose:
            print("Running detection...")
        
        cmd = [
            sys.executable, 
            self.detection_script, 
            self.tunnel_id,
            '--data-dir', self.data_dir
        ]
        
        # Set up environment with segment-anything path
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{self.segment_anything_path}:{pythonpath}" if pythonpath else self.segment_anything_path
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=self.timeout,
            cwd=self.project_root,
            env=env
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"Detection stderr: {result.stderr}")
            raise RuntimeError(f"Detection failed: {result.stderr}")
    
    def _run_sam(self):
        """Run SAM segmentation."""
        if self.verbose:
            print("Running SAM segmentation...")
        
        cmd = [
            sys.executable, 
            self.sam_script, 
            self.tunnel_id,
            '--data-dir', self.data_dir
        ]
        
        # Set up environment with segment-anything path
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{self.segment_anything_path}:{pythonpath}" if pythonpath else self.segment_anything_path
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=self.timeout,
            cwd=self.project_root,
            env=env
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"SAM stderr: {result.stderr}")
            raise RuntimeError(f"SAM failed: {result.stderr}")
    
    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation and parse metrics."""
        if self.verbose:
            print("Running evaluation...")
        
        cmd = [
            sys.executable, 
            self.eval_script, 
            self.tunnel_id,
            '--data-dir', self.data_dir
        ]
        
        # Set up environment
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{self.segment_anything_path}:{pythonpath}" if pythonpath else self.segment_anything_path
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=self.project_root,
            env=env
        )
        
        # Parse metrics from output
        metrics = self._parse_metrics(result.stdout)
        
        # Also try to read from performance.md
        if not metrics:
            metrics = self._read_performance_file()
        
        return metrics
    
    def _parse_metrics(self, output: str) -> Dict[str, float]:
        """Parse metrics from evaluation output."""
        metrics = {}
        
        for line in output.split('\n'):
            if 'OA' in line and 'F1' in line and 'mIoU' in line:
                # Parse line like "OA 0.335  F1 0.237  mIoU 0.142"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'OA' and i + 1 < len(parts):
                        try:
                            metrics['OA'] = float(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'F1' and i + 1 < len(parts):
                        try:
                            metrics['F1'] = float(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'mIoU' and i + 1 < len(parts):
                        try:
                            metrics['mIoU'] = float(parts[i + 1])
                        except ValueError:
                            pass
        
        return metrics
    
    def _read_performance_file(self) -> Dict[str, float]:
        """Read metrics from performance.md file."""
        metrics = {}
        
        perf_path = os.path.join(self.data_dir, self.tunnel_id, 'evaluation', 'performance.md')
        
        if os.path.exists(perf_path):
            with open(perf_path, 'r') as f:
                content = f.read()
            
            for line in content.split('\n'):
                if 'Overall Accuracy' in line or 'OA' in line:
                    try:
                        metrics['OA'] = float(line.split('|')[-1].strip())
                    except:
                        pass
                elif 'F1 Score' in line:
                    try:
                        metrics['F1'] = float(line.split('|')[-1].strip())
                    except:
                        pass
                elif 'Mean IoU' in line or 'mIoU' in line:
                    try:
                        metrics['mIoU'] = float(line.split('|')[-1].strip())
                    except:
                        pass
        
        return metrics
    
    def get_best(self) -> Tuple[Dict, float]:
        """Get best parameters and score found so far."""
        return self.best_params, self.best_score
    
    def save_history(self, filepath: str):
        """Save optimization history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'tunnel_id': self.tunnel_id,
                'stage': self.stage,
                'metric': self.metric,
                'best_score': self.best_score,
                'best_params': self.best_params,
                'history': self.history
            }, f, indent=2, default=float)
        
        if self.verbose:
            print(f"History saved to {filepath}")
