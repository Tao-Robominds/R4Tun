"""
Build Training Dataset for Complex Staggered Patterns (4-1, 5-1)

Runs historical BO configs, collects intrinsic metrics + mIoU for training
a predictor specific to complex staggered patterns.

Usage:
    python -m bo4tun.build_complex_training_data --tunnel 5-1 --sample 20
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bo4tun.intrinsic_metrics import compute_all_metrics


def load_complex_sam_history(tunnel_id: str) -> List[Dict]:
    """Load historical complex SAM BO evaluations with mIoU."""
    results_dir = PROJECT_ROOT / "p4tun" / "bo" / "results"
    all_evals = []
    
    # 5-1 complex SAM
    if tunnel_id == '5-1':
        for f in results_dir.glob('5-1_complex_sam_*.json'):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for h in data.get('history', []):
                    m = h.get('metrics', {})
                    if m.get('mIoU', 0) > 0:
                        all_evals.append({
                            'tunnel_id': '5-1',
                            'mIoU': m.get('mIoU'),
                            'OA': m.get('OA', 0),
                            'F1': m.get('F1', 0),
                            'params': h.get('params', {}),
                        })
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    # 4-1 SAM wraparound
    elif tunnel_id == '4-1':
        for f in results_dir.glob('4-1_sam_wraparound_*_history.json'):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for h in data.get('history', []):
                    m = h.get('metrics', {})
                    miou = m.get('mIoU', h.get('score', 0))
                    if miou > 0:
                        all_evals.append({
                            'tunnel_id': '4-1',
                            'mIoU': miou,
                            'params': h.get('params', {}),
                        })
            except Exception as e:
                print(f"Error loading {f}: {e}")
    
    return all_evals


def run_complex_pipeline(tunnel_id: str, sam_params: Dict, data_dir: str = 'data') -> bool:
    """Run complex detection + SAM pipeline with given SAM parameters."""
    # Save SAM params
    params_dir = PROJECT_ROOT / "p4tun" / "parameters" / tunnel_id
    params_dir.mkdir(parents=True, exist_ok=True)
    
    sam_config = {
        'segment_geometry': {
            'segment_width': sam_params.get('segment_width', 1200.0),
            'k_height': sam_params.get('k_height', 1079.92),
            'ab_height': sam_params.get('ab_height', 3239.77),
            'angle_deg': sam_params.get('angle_deg', 7.52),
        },
        'image': {'resolution': 0.005},
        'pattern_aware': {
            'use_quality_weighting': True,
            'min_quality_threshold': 0.3,
        },
        'prompt_points': {
            'template_mask': {
                'k_block': {
                    'width': sam_params.get('k_mask_width', 625.0),
                    'height_pos': sam_params.get('k_mask_height_pos', 619.16),
                    'height_neg': sam_params.get('k_mask_height_neg', 460.77),
                },
                'b1_block': {
                    'width': sam_params.get('ab_mask_width', 700.0),
                    'height_top': 1500.0,
                    'height_bottom_pos': 1540.69,
                    'height_bottom_neg': 1699.08,
                },
                'b2_block': {
                    'width': sam_params.get('ab_mask_width', 700.0),
                    'height_top_pos': 1540.69,
                    'height_top_neg': 1699.08,
                    'height_bottom': 1500.0,
                },
                'a_blocks': {
                    'width': sam_params.get('ab_mask_width', 700.0),
                    'height': sam_params.get('ab_mask_height', 1619.89),
                },
            },
        },
        'complex_staggered': {
            'template_sizing': {
                'k_block_width_factor': sam_params.get('k_block_width_factor', 1.0),
                'k_block_height_factor': sam_params.get('k_block_height_factor', 1.0),
                'ab_block_width_factor': sam_params.get('ab_block_width_factor', 1.0),
                'ab_block_height_factor': sam_params.get('ab_block_height_factor', 1.0),
            },
            'prompt_density': {
                'k_block_points': sam_params.get('k_block_points', 'standard'),
                'ab_block_points': sam_params.get('ab_block_points', 'standard'),
            },
        },
    }
    
    sam_path = params_dir / "parameters_sam.json"
    with open(sam_path, 'w') as f:
        json.dump(sam_config, f, indent=2)
    
    # Run complex detection (uses existing detection params)
    det_script = PROJECT_ROOT / "p4tun" / "4-1_detection_complex.py"
    cmd = [sys.executable, str(det_script), tunnel_id, '--data-dir', data_dir]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"  Detection failed: {result.stderr.decode()[:100]}")
            return False
    except Exception as e:
        print(f"  Detection error: {e}")
        return False
    
    # Run complex SAM
    sam_script = PROJECT_ROOT / "p4tun" / "4-2_sam_complex.py"
    cmd = [sys.executable, str(sam_script), tunnel_id]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"  SAM failed: {result.stderr.decode()[:100]}")
            return False
    except Exception as e:
        print(f"  SAM error: {e}")
        return False
    
    return True


def evaluate_miou(tunnel_id: str, data_dir: str = 'data') -> float:
    """Evaluate mIoU from final.csv against ground truth."""
    from p4tun.evaluation import calculate_metrics
    
    labels_path = Path(data_dir) / tunnel_id / "only_label.csv"
    if not labels_path.exists():
        return 0.0
    
    try:
        df = pd.read_csv(labels_path)
        gt = df['gt_labels'].values
        pred = df['pred_labels'].values
        
        valid = (gt >= 0) & (pred >= 0)
        gt = gt[valid].astype(int)
        pred = pred[valid].astype(int)
        
        n_segments = max(gt.max(), pred.max()) + 1
        class_names = {i: f'class_{i}' for i in range(n_segments)}
        
        metrics = calculate_metrics(gt, pred, class_names, n_segments)
        return metrics.get('mIoU', 0.0)
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return 0.0


def build_training_data(
    tunnel_id: str,
    sample_size: int = 20,
    data_dir: str = 'data',
    output_dir: str = 'bo4tun/training',
) -> pd.DataFrame:
    """
    Build training dataset for complex staggered predictor.
    
    Samples configs from historical BO, runs pipeline, collects intrinsic metrics + mIoU.
    """
    print(f"\n{'='*70}")
    print(f"Building Complex Staggered Training Data: {tunnel_id}")
    print(f"{'='*70}")
    
    # Load historical evaluations
    history = load_complex_sam_history(tunnel_id)
    print(f"Historical SAM evaluations: {len(history)}")
    
    if len(history) == 0:
        print("No historical data found!")
        return pd.DataFrame()
    
    # Sample diverse configs (stratified by mIoU)
    df_hist = pd.DataFrame(history)
    df_hist['mIoU_bin'] = pd.cut(df_hist['mIoU'], bins=5, labels=['low', 'low-mid', 'mid', 'mid-high', 'high'])
    
    sampled = df_hist.groupby('mIoU_bin', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 5), random_state=42)
    )
    
    if len(sampled) < sample_size:
        remaining = sample_size - len(sampled)
        extra = df_hist[~df_hist.index.isin(sampled.index)].sample(
            min(remaining, len(df_hist) - len(sampled)), random_state=42
        )
        sampled = pd.concat([sampled, extra])
    
    print(f"Sampled {len(sampled)} configs for training")
    
    # Process each config
    training_data = []
    
    for i, (idx, row) in enumerate(sampled.iterrows()):
        print(f"\n[{i+1}/{len(sampled)}] Processing config (historical mIoU: {row['mIoU']:.4f})")
        
        # Run pipeline with these params
        if not run_complex_pipeline(tunnel_id, row['params'], data_dir):
            print("  Skipping due to pipeline failure")
            continue
        
        # Compute intrinsic metrics
        detected_csv = Path(data_dir) / tunnel_id / "detected.csv"
        final_csv = Path(data_dir) / tunnel_id / "final.csv"
        
        try:
            intrinsic = compute_all_metrics(tunnel_id, str(detected_csv), str(final_csv), data_dir)
        except Exception as e:
            print(f"  Metrics error: {e}")
            continue
        
        # Evaluate true mIoU (should match historical)
        true_miou = evaluate_miou(tunnel_id, data_dir)
        
        # Record
        record = {
            'tunnel_id': tunnel_id,
            'mIoU': true_miou,
            'historical_mIoU': row['mIoU'],
            **intrinsic,
            # Include key SAM params for analysis
            'param_segment_width': row['params'].get('segment_width'),
            'param_k_height': row['params'].get('k_height'),
            'param_ab_height': row['params'].get('ab_height'),
            'param_angle_deg': row['params'].get('angle_deg'),
        }
        training_data.append(record)
        
        print(f"  Intrinsic: k_count={intrinsic.get('det_k_count', 0):.0f}, "
              f"fill_rate={intrinsic.get('sam_mask_fill_rate', 0):.3f}")
        print(f"  True mIoU: {true_miou:.4f} (hist: {row['mIoU']:.4f})")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(training_data)
    csv_path = output_path / f"complex_training_{tunnel_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved {len(df)} training samples to {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Build Complex Staggered Training Data')
    parser.add_argument('--tunnel', default='5-1', choices=['4-1', '5-1'], help='Tunnel ID')
    parser.add_argument('--sample', type=int, default=20, help='Number of configs to sample')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='bo4tun/training', help='Output directory')
    
    args = parser.parse_args()
    
    build_training_data(
        tunnel_id=args.tunnel,
        sample_size=args.sample,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
