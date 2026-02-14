"""
BO History Data Loader for No-GT Bayesian Optimization

This module loads and parses all BO history JSON files from p4tun/bo/results/
to build training datasets for the learned mIoU predictor (Layer B).

Key functions:
- load_all_bo_histories(): Load all *_history.json files
- build_training_dataset(): Assemble features and targets for predictor training
- save_training_data(): Save processed data to bo4tun/training/
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd


def get_results_dir() -> str:
    """Get the BO results directory path."""
    return os.path.join(os.path.dirname(__file__), 'results')


def get_training_dir() -> str:
    """Get the training data output directory path."""
    # Save to bo4tun/training/ as requested
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    training_dir = os.path.join(project_root, 'bo4tun', 'training')
    os.makedirs(training_dir, exist_ok=True)
    return training_dir


def _infer_stage_from_filename(filename: str) -> str:
    """Infer the stage name from filename if not present in data."""
    filename_lower = filename.lower()
    
    if 'complex_detection' in filename_lower:
        return 'complex_detection'
    elif 'complex_sam' in filename_lower:
        return 'complex_sam'
    elif 'sam_wraparound' in filename_lower:
        return 'sam_wraparound'
    elif 'detection' in filename_lower:
        return 'detection'
    elif 'sam' in filename_lower:
        return 'sam'
    elif 'combined' in filename_lower:
        return 'combined'
    elif 'preprocessing' in filename_lower:
        return 'preprocessing'
    elif 'unfolding' in filename_lower:
        return 'unfolding'
    elif 'full_pipeline' in filename_lower:
        return 'full_pipeline'
    else:
        return 'unknown'


def load_single_history(filepath: str) -> Optional[Dict]:
    """
    Load a single BO history JSON file.
    
    Args:
        filepath: Path to the history JSON file
        
    Returns:
        Dictionary with parsed data or None if invalid/empty
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Skip files with empty history
        if not data.get('history') or len(data['history']) == 0:
            return None
            
        # Skip files with invalid best_score
        if data.get('best_score') in [None, float('-inf'), float('inf')]:
            if not data.get('history'):
                return None
        
        filename = os.path.basename(filepath)
        
        # Infer stage from filename if not present in data
        stage = data.get('stage')
        if not stage or stage == 'unknown':
            stage = _infer_stage_from_filename(filename)
        
        return {
            'filepath': filepath,
            'filename': filename,
            'tunnel_id': data.get('tunnel_id', 'unknown'),
            'stage': stage,
            'metric': data.get('metric', 'mIoU'),
            'best_score': data.get('best_score'),
            'best_params': data.get('best_params'),
            'history': data.get('history', []),
            'n_evaluations': len(data.get('history', []))
        }
        
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def load_all_bo_histories(results_dir: Optional[str] = None) -> List[Dict]:
    """
    Load all BO history JSON files from the results directory.
    
    Args:
        results_dir: Path to results directory (defaults to p4tun/bo/results/)
        
    Returns:
        List of parsed history dictionaries
    """
    if results_dir is None:
        results_dir = get_results_dir()
    
    # Find all JSON files (both *_history.json and direct result files)
    all_json_pattern = os.path.join(results_dir, '*.json')
    all_json_files = set(glob.glob(all_json_pattern))
    
    # Exclude metadata files and other non-BO files
    exclude_patterns = ['metadata', 'best_extracted', 'TUNING', 'checkpoint']
    candidate_files = []
    for f in all_json_files:
        basename = os.path.basename(f)
        if not any(pat in basename for pat in exclude_patterns):
            candidate_files.append(f)
    
    print(f"Found {len(candidate_files)} candidate JSON files in {results_dir}")
    
    histories = []
    loaded_filenames = set()
    
    for filepath in sorted(candidate_files):
        filename = os.path.basename(filepath)
        
        # Skip if we already loaded the history version
        base_name = filename.replace('_history.json', '.json')
        if base_name in loaded_filenames or filename in loaded_filenames:
            continue
            
        data = load_single_history(filepath)
        if data is not None:
            histories.append(data)
            loaded_filenames.add(filename)
            print(f"  Loaded: {data['filename']} ({data['tunnel_id']}/{data['stage']}, {data['n_evaluations']} evals)")
    
    print(f"Successfully loaded {len(histories)} non-empty history files")
    return histories


def extract_evaluation_records(histories: List[Dict], require_miou: bool = True) -> pd.DataFrame:
    """
    Extract all individual evaluation records from histories.
    
    Args:
        histories: List of loaded history dictionaries
        require_miou: If True, only include records with valid mIoU metrics
        
    Returns:
        DataFrame with one row per evaluation
    """
    records = []
    skipped_no_metrics = 0
    
    for hist in histories:
        tunnel_id = hist['tunnel_id']
        stage = hist['stage']
        source_file = hist['filename']
        
        for eval_record in hist['history']:
            # Check if this record has proper mIoU metrics
            metrics = eval_record.get('metrics', {})
            has_valid_miou = metrics and 'mIoU' in metrics and metrics['mIoU'] > 0
            
            if require_miou and not has_valid_miou:
                skipped_no_metrics += 1
                continue
            
            record = {
                'tunnel_id': tunnel_id,
                'stage': stage,
                'source_file': source_file,
                'eval_num': eval_record.get('eval', 0),
            }
            
            # Extract metrics
            if has_valid_miou:
                record['mIoU'] = metrics['mIoU']
                record['OA'] = metrics.get('OA', 0.0)
                record['F1'] = metrics.get('F1', 0.0)
            else:
                # Only used when require_miou=False
                score = eval_record.get('score', 0.0)
                record['mIoU'] = score if score <= 1.0 else 0.0
                record['OA'] = 0.0
                record['F1'] = 0.0
            
            # Extract parameters as individual columns
            params = eval_record.get('params', {})
            for param_name, param_value in params.items():
                # Skip non-numeric params for training
                if isinstance(param_value, (int, float)):
                    record[f'param_{param_name}'] = param_value
            
            records.append(record)
    
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} evaluation records with valid mIoU")
    if skipped_no_metrics > 0:
        print(f"  (Skipped {skipped_no_metrics} records without valid mIoU metrics)")
    return df


def get_param_columns(df: pd.DataFrame) -> List[str]:
    """Get list of parameter column names from DataFrame."""
    return [col for col in df.columns if col.startswith('param_')]


def get_stage_param_mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get mapping of stage to its parameter columns.
    """
    stage_params = {}
    
    for stage in df['stage'].unique():
        stage_df = df[df['stage'] == stage]
        param_cols = get_param_columns(stage_df)
        
        # Keep only columns that have non-null values for this stage
        used_params = []
        for col in param_cols:
            if stage_df[col].notna().any():
                used_params.append(col)
        
        stage_params[stage] = sorted(used_params)
    
    return stage_params


def build_training_dataset(histories: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Build training dataset from BO histories.
    Only includes records with valid mIoU metrics.
    
    Args:
        histories: List of loaded history dictionaries
        
    Returns:
        Tuple of (DataFrame with features and target, metadata dict)
    """
    # Extract only records with valid mIoU
    df = extract_evaluation_records(histories, require_miou=True)
    
    if len(df) == 0:
        print("Warning: No evaluation records with valid mIoU found!")
        return pd.DataFrame(), {}
    
    # Get stage-param mapping
    stage_params = get_stage_param_mapping(df)
    
    # Add tunnel context features
    tunnel_context = {
        # Simple patterns: 6 segments
        '1-4': {'pattern_hint': 'simple_staggered', 'expected_rings': 10, 'segments_per_ring': 6},
        '2-2': {'pattern_hint': 'simple_staggered', 'expected_rings': 10, 'segments_per_ring': 6},
        '3-1': {'pattern_hint': 'continuous', 'expected_rings': 6, 'segments_per_ring': 6},
        # Complex patterns: 7 segments
        '4-1': {'pattern_hint': 'complex_staggered', 'expected_rings': 10, 'segments_per_ring': 7},
        '5-1': {'pattern_hint': 'complex_staggered', 'expected_rings': 7, 'segments_per_ring': 7},
    }
    
    df['pattern_hint'] = df['tunnel_id'].map(
        lambda x: tunnel_context.get(x, {}).get('pattern_hint', 'unknown')
    )
    df['expected_rings'] = df['tunnel_id'].map(
        lambda x: tunnel_context.get(x, {}).get('expected_rings', 10)
    )
    df['segments_per_ring'] = df['tunnel_id'].map(
        lambda x: tunnel_context.get(x, {}).get('segments_per_ring', 6)
    )
    
    # Build metadata
    metadata = {
        'n_records': len(df),
        'n_histories': len(histories),
        'tunnels': sorted(df['tunnel_id'].unique().tolist()),
        'stages': sorted(df['stage'].unique().tolist()),
        'stage_params': stage_params,
        'param_columns': get_param_columns(df),
        'target_column': 'mIoU',
        'metric_columns': ['mIoU', 'OA', 'F1'],
        'context_columns': ['tunnel_id', 'stage', 'pattern_hint', 'expected_rings', 'segments_per_ring'],
        'created_at': datetime.now().isoformat(),
    }
    
    # Add statistics
    metadata['statistics'] = compute_dataset_statistics(df)
    
    return df, metadata


def compute_dataset_statistics(df: pd.DataFrame) -> Dict:
    """Compute statistics about the training dataset."""
    if len(df) == 0:
        return {}
    
    stats = {
        'total_evaluations': len(df),
        'per_tunnel': df.groupby('tunnel_id').size().to_dict(),
        'per_stage': df.groupby('stage').size().to_dict(),
        'mIoU_stats': {
            'mean': float(df['mIoU'].mean()),
            'std': float(df['mIoU'].std()),
            'min': float(df['mIoU'].min()),
            'max': float(df['mIoU'].max()),
            'median': float(df['mIoU'].median()),
        },
    }
    
    # Per-stage mIoU stats
    stats['mIoU_per_stage'] = {}
    for stage in df['stage'].unique():
        stage_df = df[df['stage'] == stage]
        stats['mIoU_per_stage'][stage] = {
            'mean': float(stage_df['mIoU'].mean()),
            'std': float(stage_df['mIoU'].std()),
            'min': float(stage_df['mIoU'].min()),
            'max': float(stage_df['mIoU'].max()),
            'count': int(len(stage_df)),
        }
    
    return stats


def save_training_data(df: pd.DataFrame, metadata: Dict, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Save training data to a single clean file.
    
    Args:
        df: Training DataFrame
        metadata: Metadata dictionary
        output_dir: Output directory (defaults to bo4tun/training/)
        
    Returns:
        Dictionary with paths to saved files
    """
    if output_dir is None:
        output_dir = get_training_dir()
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save single training CSV
    csv_path = os.path.join(output_dir, 'miou_training_data.csv')
    df.to_csv(csv_path, index=False)
    saved_files['training_csv'] = csv_path
    print(f"Saved training data: {csv_path}")
    
    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, 'miou_training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    saved_files['metadata_json'] = metadata_path
    print(f"Saved metadata: {metadata_path}")
    
    # Save summary report
    report_path = os.path.join(output_dir, 'TRAINING_DATA_SUMMARY.md')
    _write_summary_report(df, metadata, report_path)
    saved_files['summary_report'] = report_path
    print(f"Saved summary report: {report_path}")
    
    return saved_files


def _write_summary_report(df: pd.DataFrame, metadata: Dict, filepath: str):
    """Write a markdown summary report of the training data."""
    stats = metadata.get('statistics', {})
    
    with open(filepath, 'w') as f:
        f.write("# mIoU Predictor Training Data Summary\n\n")
        f.write("Training data for Layer B learned mIoU predictor.\n\n")
        f.write(f"**Generated:** {metadata.get('created_at', 'unknown')}\n\n")
        
        f.write("## Purpose\n\n")
        f.write("This dataset is used to train a model that predicts mIoU from:\n")
        f.write("- Pipeline parameters (detection, SAM, etc.)\n")
        f.write("- Tunnel context (pattern type, ring count)\n")
        f.write("- Eventually: intrinsic metrics computed during pipeline execution\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- **Total Evaluations:** {stats.get('total_evaluations', 0)}\n")
        f.write(f"- **Source Files:** {metadata.get('n_histories', 0)}\n")
        f.write(f"- **Tunnels:** {', '.join(metadata.get('tunnels', []))}\n")
        f.write(f"- **Stages:** {', '.join(metadata.get('stages', []))}\n")
        f.write(f"- **Parameter Columns:** {len(metadata.get('param_columns', []))}\n\n")
        
        f.write("## Data Quality\n\n")
        f.write("All records in this dataset have:\n")
        f.write("- Valid mIoU metrics (from ground truth evaluation)\n")
        f.write("- Complete parameter vectors for their respective stages\n")
        f.write("- Numeric-only parameters (categorical params excluded)\n\n")
        
        f.write("## Evaluations by Tunnel\n\n")
        f.write("| Tunnel | Count | Pattern |\n")
        f.write("|--------|-------|--------|\n")
        tunnel_patterns = {'1-4': 'simple_staggered', '2-2': 'simple_staggered', 
                          '3-1': 'continuous', '4-1': 'simple_staggered', '5-1': 'complex_staggered'}
        for tunnel, count in sorted(stats.get('per_tunnel', {}).items()):
            pattern = tunnel_patterns.get(tunnel, 'unknown')
            f.write(f"| {tunnel} | {count} | {pattern} |\n")
        f.write("\n")
        
        f.write("## Evaluations by Stage\n\n")
        f.write("| Stage | Count | Description |\n")
        f.write("|-------|-------|-------------|\n")
        stage_desc = {
            'detection': 'Line detection for K-block positions',
            'sam': 'SAM segmentation parameters',
            'combined': 'Detection + SAM combined optimization',
            'preprocessing': 'Denoising + Enhancing parameters',
            'unfolding': 'Point cloud unfolding parameters',
            'full_pipeline': 'Full pipeline optimization',
            'complex_sam': 'SAM for complex staggered patterns (5-1)',
            'sam_wraparound': 'SAM with wraparound handling',
        }
        for stage, count in sorted(stats.get('per_stage', {}).items()):
            desc = stage_desc.get(stage, 'Other stage')
            f.write(f"| {stage} | {count} | {desc} |\n")
        f.write("\n")
        
        f.write("## mIoU Statistics\n\n")
        miou_stats = stats.get('mIoU_stats', {})
        f.write(f"- **Mean:** {miou_stats.get('mean', 0):.4f}\n")
        f.write(f"- **Std:** {miou_stats.get('std', 0):.4f}\n")
        f.write(f"- **Min:** {miou_stats.get('min', 0):.4f}\n")
        f.write(f"- **Max:** {miou_stats.get('max', 0):.4f}\n")
        f.write(f"- **Median:** {miou_stats.get('median', 0):.4f}\n\n")
        
        f.write("## mIoU by Stage\n\n")
        f.write("| Stage | Mean | Std | Min | Max | Count |\n")
        f.write("|-------|------|-----|-----|-----|-------|\n")
        for stage, stage_stats in sorted(stats.get('mIoU_per_stage', {}).items()):
            f.write(f"| {stage} | {stage_stats['mean']:.4f} | {stage_stats['std']:.4f} | "
                   f"{stage_stats['min']:.4f} | {stage_stats['max']:.4f} | {stage_stats['count']} |\n")
        f.write("\n")
        
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("import json\n\n")
        f.write("# Load training data\n")
        f.write("df = pd.read_csv('bo4tun/training/miou_training_data.csv')\n")
        f.write("with open('bo4tun/training/miou_training_metadata.json') as f:\n")
        f.write("    metadata = json.load(f)\n\n")
        f.write("# Get param columns for a specific stage\n")
        f.write("stage_params = metadata['stage_params']['detection']\n")
        f.write("```\n")


def load_training_data(training_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load the saved training data.
    
    Args:
        training_dir: Training data directory
        
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    if training_dir is None:
        training_dir = get_training_dir()
    
    csv_path = os.path.join(training_dir, 'miou_training_data.csv')
    metadata_path = os.path.join(training_dir, 'miou_training_metadata.json')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No training data found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return df, metadata


def main():
    """Main entry point for data loading and processing."""
    print("=" * 70)
    print("BO History Data Loader - mIoU Training Data")
    print("=" * 70)
    
    # Load all histories
    histories = load_all_bo_histories()
    
    if not histories:
        print("No valid history files found!")
        return
    
    # Build training dataset (only records with valid mIoU)
    print("\nBuilding training dataset (mIoU records only)...")
    df, metadata = build_training_dataset(histories)
    
    if len(df) == 0:
        print("No valid training data found!")
        return
    
    # Print summary
    stats = metadata.get('statistics', {})
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Total evaluations with valid mIoU: {len(df)}")
    print(f"Tunnels: {metadata.get('tunnels', [])}")
    print(f"Stages: {metadata.get('stages', [])}")
    print(f"\nmIoU Statistics:")
    print(f"  Mean: {stats['mIoU_stats']['mean']:.4f}")
    print(f"  Std:  {stats['mIoU_stats']['std']:.4f}")
    print(f"  Range: [{stats['mIoU_stats']['min']:.4f}, {stats['mIoU_stats']['max']:.4f}]")
    
    # Save training data
    print("\n" + "=" * 70)
    print("Saving Training Data")
    print("=" * 70)
    saved_files = save_training_data(df, metadata)
    
    print("\nData loading complete!")
    print(f"Training data ready at: {get_training_dir()}")
    
    return df, metadata


if __name__ == '__main__':
    main()
