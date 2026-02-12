"""
Build Intrinsic Metrics Training Data

For each historical config in miou_training_data.csv (detection and SAM stages):
1. Run pipeline with those params
2. Compute intrinsic metrics from outputs
3. Pair with mIoU from BO logs
4. Save to intrinsic_training_data.csv

Usage:
    python -m bo4tun.build_training_data              # Full dataset
    python -m bo4tun.build_training_data --sample 50  # Sample for validation
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List
import pandas as pd

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Stages we care about (detection + SAM focus)
DETECTION_SAM_STAGES = {
    'detection',
    'sam',
    'combined',
    'complex_sam',
    'sam_wraparound',
}

# True detection params that affect det_* metrics
TRUE_DET_PARAMS = [
    'param_binary_threshold',
    'param_hough_oblique_threshold', 
    'param_hough_horizontal_threshold',
    'param_hough_vertical_threshold',
    'param_merge_distance_threshold',
    'param_angle_positive_min',
    'param_angle_positive_max',
]

# SAM params that affect sam_* metrics
TRUE_SAM_PARAMS = [
    'param_segment_width',
    'param_angle_deg',
    'param_k_mask_width',
    'param_k_mask_height_pos',
    'param_k_mask_height_neg',
    'param_min_quality_threshold',
    'param_k_block_width_factor',
    'param_k_block_height_factor',
]


def get_training_dir() -> str:
    """Get the training data directory."""
    return os.path.join(PROJECT_ROOT, 'bo4tun', 'training')


def load_miou_training_data() -> tuple:
    """Load the mIoU training data and metadata."""
    training_dir = get_training_dir()
    csv_path = os.path.join(training_dir, 'miou_training_data.csv')
    meta_path = os.path.join(training_dir, 'miou_training_metadata.json')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run data_loader first. Missing: {csv_path}")
    
    df = pd.read_csv(csv_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return df, metadata


def build_intrinsic_training_data(
    sample_size: int = None,
    per_tunnel: int = None,
    data_dir: str = 'data',
    verbose: bool = True,
    detection_only: bool = False,
    det_variation_only: bool = False,
    sam_variation_only: bool = False
) -> pd.DataFrame:
    """
    Build intrinsic metrics training dataset.
    
    Args:
        sample_size: If set, only process this many configs total (random)
        per_tunnel: If set, sample this many configs per tunnel (stratified)
        data_dir: Base data directory
        verbose: Print progress
        detection_only: Only run detection stage (skip SAM, faster)
        det_variation_only: Only use configs where true detection params vary
        sam_variation_only: Only use configs where SAM params vary
        
    Returns:
        DataFrame with intrinsic_metrics + mIoU
    """
    from bo4tun.config_runner import run_config_and_collect_metrics
    
    df, metadata = load_miou_training_data()
    
    # Filter to detection + SAM stages only
    df = df[df['stage'].isin(DETECTION_SAM_STAGES)].copy()
    stage_params = metadata.get('stage_params', {})
    
    # Filter to configs with true detection param variation
    if det_variation_only:
        valid_indices = []
        for tid in df['tunnel_id'].unique():
            for stage in df[df['tunnel_id'] == tid]['stage'].unique():
                sub = df[(df['tunnel_id'] == tid) & (df['stage'] == stage)]
                has_det_variation = False
                for col in TRUE_DET_PARAMS:
                    if col in sub.columns and sub[col].notna().sum() > 1:
                        if sub[col].nunique() > 1:
                            has_det_variation = True
                            break
                if has_det_variation:
                    valid_indices.extend(sub.index.tolist())
        df = df.loc[valid_indices].copy()
        if verbose:
            print(f"Filtered to {len(df)} configs with true detection param variation")
            print(f"Tunnels: {sorted(df['tunnel_id'].unique())}")
    
    # Filter to configs with SAM param variation
    if sam_variation_only:
        valid_indices = []
        for tid in df['tunnel_id'].unique():
            for stage in df[df['tunnel_id'] == tid]['stage'].unique():
                sub = df[(df['tunnel_id'] == tid) & (df['stage'] == stage)]
                has_sam_variation = False
                for col in TRUE_SAM_PARAMS:
                    if col in sub.columns and sub[col].notna().sum() > 1:
                        if sub[col].nunique() > 1:
                            has_sam_variation = True
                            break
                if has_sam_variation:
                    valid_indices.extend(sub.index.tolist())
        df = df.loc[valid_indices].copy()
        if verbose:
            print(f"Filtered to {len(df)} configs with SAM param variation")
            print(f"Tunnels: {sorted(df['tunnel_id'].unique())}")
    
    if verbose:
        print(f"Processing {len(df)} configs from detection/SAM stages")
        print(f"Stages: {df['stage'].unique().tolist()}")
        if detection_only:
            print("Mode: detection-only (skipping SAM for faster validation)")
    
    records = []
    failed = 0
    
    indices = df.index.tolist()
    
    # Stratified sampling per tunnel
    if per_tunnel:
        import random
        random.seed(42)
        indices = []
        for tid in sorted(df['tunnel_id'].unique()):
            tid_indices = df[df['tunnel_id'] == tid].index.tolist()
            sampled = random.sample(tid_indices, min(per_tunnel, len(tid_indices)))
            indices.extend(sampled)
            if verbose:
                print(f"  {tid}: sampled {len(sampled)}/{len(tid_indices)}")
    elif sample_size:
        import random
        random.seed(42)
        indices = random.sample(indices, min(sample_size, len(indices)))
        if verbose:
            print(f"Sampling {len(indices)} configs")
    
    for i, idx in enumerate(indices):
        row = df.loc[idx]
        tunnel_id = row['tunnel_id']
        stage = row['stage']
        miou = row['mIoU']
        
        param_cols = stage_params.get(stage, [])
        if not param_cols:
            param_cols = [c for c in row.index if c.startswith('param_') and pd.notna(row[c])]
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(indices)}] {tunnel_id} {stage} (mIoU={miou:.3f})")
        
        try:
            metrics = run_config_and_collect_metrics(
                tunnel_id=tunnel_id,
                stage=stage,
                row=row,
                param_cols=param_cols,
                data_dir=data_dir,
                detection_only=detection_only
            )
            
            if metrics:
                record = {
                    'tunnel_id': tunnel_id,
                    'stage': stage,
                    'mIoU': miou,
                }
                record.update(metrics)
                records.append(record)
            else:
                failed += 1
                if verbose:
                    print(f"    Warning: No metrics for {tunnel_id} {stage}")
                    
        except Exception as e:
            failed += 1
            if verbose:
                print(f"    Error: {tunnel_id} {stage}: {e}")
    
    result_df = pd.DataFrame(records)
    
    if verbose:
        print(f"\nCompleted: {len(result_df)} records, {failed} failed")
    
    return result_df


def save_intrinsic_training_data(df: pd.DataFrame, output_dir: str = None) -> Dict[str, str]:
    """Save intrinsic training data and metadata."""
    if output_dir is None:
        output_dir = get_training_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'intrinsic_training_data.csv')
    df.to_csv(csv_path, index=False)
    
    # Build metadata
    metric_cols = [c for c in df.columns if c not in ('tunnel_id', 'stage', 'mIoU')]
    
    metadata = {
        'n_records': len(df),
        'intrinsic_metric_columns': metric_cols,
        'target_column': 'mIoU',
        'tunnels': sorted(df['tunnel_id'].unique().tolist()),
        'stages': sorted(df['stage'].unique().tolist()),
        'created_at': datetime.now().isoformat(),
    }
    
    # Add statistics
    if len(df) > 0:
        metadata['mIoU_stats'] = {
            'mean': float(df['mIoU'].mean()),
            'std': float(df['mIoU'].std()),
            'min': float(df['mIoU'].min()),
            'max': float(df['mIoU'].max()),
        }
        metadata['per_stage'] = df.groupby('stage').size().to_dict()
    
    meta_path = os.path.join(output_dir, 'intrinsic_training_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Write summary markdown
    summary_path = os.path.join(output_dir, 'INTRINSIC_TRAINING_SUMMARY.md')
    _write_summary(df, metadata, summary_path)
    
    return {
        'csv': csv_path,
        'metadata': meta_path,
        'summary': summary_path,
    }


def _write_summary(df: pd.DataFrame, metadata: Dict, filepath: str):
    """Write INTRINSIC_TRAINING_SUMMARY.md."""
    with open(filepath, 'w') as f:
        f.write("# Intrinsic Metrics Training Data Summary\n\n")
        f.write("Dataset for training mIoU predictor from intrinsic metrics.\n\n")
        f.write(f"**Generated:** {metadata.get('created_at', 'unknown')}\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Records:** {metadata.get('n_records', 0)}\n")
        f.write(f"- **Intrinsic Metrics:** {len(metadata.get('intrinsic_metric_columns', []))}\n")
        f.write(f"- **Target:** {metadata.get('target_column', 'mIoU')}\n\n")
        
        f.write("## Purpose\n\n")
        f.write("Train a model: `mIoU = f(intrinsic_metrics)`\n\n")
        f.write("At runtime without GT, we compute intrinsic metrics from pipeline outputs.\n")
        f.write("This model predicts mIoU from those metrics.\n\n")
        
        f.write("## Intrinsic Metrics\n\n")
        for col in metadata.get('intrinsic_metric_columns', []):
            f.write(f"- `{col}`\n")
        f.write("\n")
        
        f.write("## Evaluations by Stage\n\n")
        f.write("| Stage | Count |\n|-------|-------|\n")
        for stage, count in sorted(metadata.get('per_stage', {}).items()):
            f.write(f"| {stage} | {count} |\n")
        f.write("\n")
        
        miou = metadata.get('mIoU_stats', {})
        f.write("## mIoU Statistics\n\n")
        f.write(f"- Mean: {miou.get('mean', 0):.4f}\n")
        f.write(f"- Std: {miou.get('std', 0):.4f}\n")
        f.write(f"- Range: [{miou.get('min', 0):.4f}, {miou.get('max', 0):.4f}]\n")


def main():
    parser = argparse.ArgumentParser(description='Build intrinsic metrics training data')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only N configs total (random sampling)')
    parser.add_argument('--per-tunnel', type=int, default=None,
                        help='Sample N configs per tunnel (stratified, for balanced evaluation)')
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output')
    parser.add_argument('--detection-only', action='store_true',
                        help='Only run detection (faster validation, det_metrics only)')
    parser.add_argument('--det-variation-only', action='store_true',
                        help='Only use configs where true detection params vary (excludes SAM-only tuning)')
    parser.add_argument('--sam-variation-only', action='store_true',
                        help='Only use configs where SAM params vary (for sam_* metric evaluation)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Building Intrinsic Metrics Training Data")
    print("=" * 70)
    
    df = build_intrinsic_training_data(
        sample_size=args.sample,
        per_tunnel=args.per_tunnel,
        data_dir=args.data_dir,
        verbose=not args.quiet,
        detection_only=args.detection_only,
        det_variation_only=args.det_variation_only,
        sam_variation_only=args.sam_variation_only
    )
    
    if len(df) == 0:
        print("No records produced!")
        return 1
    
    saved = save_intrinsic_training_data(df)
    print("\nSaved:")
    for name, path in saved.items():
        print(f"  {name}: {path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
