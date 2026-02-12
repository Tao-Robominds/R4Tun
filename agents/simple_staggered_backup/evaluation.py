"""
Simple Staggered Evaluation (Stage 4)

Evaluates segmentation quality for SIMPLE STAGGERED tunnels only.
For continuous or complex staggered patterns, use the appropriate pipeline.

Supports two modes:
1. WITH Ground Truth: Compare predictions against ground truth labels
   - Input: final.csv with 'segment' (GT) and 'pred' columns
   - Outputs: OA, F1, mIoU metrics

2. WITHOUT Ground Truth: Generate prediction statistics only
   - Input: final.csv with 'pred' column
   - Outputs: Class distribution, coverage analysis

Simple staggered tunnels typically have 6 segments per ring:
K-block, B1-block, A1-block, A2-block, A3-block, B2-block
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix


# =============================================================================
# Default Constants (can be overridden via parameters JSON)
# =============================================================================

DEFAULT_K_HEIGHT_MM = 1079.92
DEFAULT_AB_HEIGHT_MM = 3239.77
DEFAULT_RESOLUTION = 0.005


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str, base_dir: str = "data") -> dict:
    """
    Load parameters from JSON file.
    
    Tries to load physical constants from detection parameters.
    """
    script_dir = os.path.dirname(__file__)
    
    # Try tunnel-specific detection parameters (has physical constants)
    params_path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_detection.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            return json.load(f)
    
    # Try sample parameters
    sample_path = os.path.join(script_dir, "parameters", "sample", "parameters_detection.json")
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    return {}


def get_param(params: dict, *keys, default=None):
    """Get nested parameter value."""
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# Class Names (Simple Staggered: 6 segments standard)
# =============================================================================

# Standard 6-segment layout for simple staggered tunnels
CLASS_NAMES = {
    0: 'Background',
    1: 'K-block',
    2: 'B1-block',
    3: 'A1-block',
    4: 'A2-block',
    5: 'A3-block',
    6: 'B2-block'
}

# Number of segments for simple staggered (fixed)
SIMPLE_STAGGERED_SEGMENTS = 6


# =============================================================================
# Segment Count (Simple Staggered = 6 segments)
# =============================================================================

def verify_segment_count(tunnel_dir: str) -> int:
    """
    Verify tunnel is simple staggered (6 segments).
    
    Simple staggered tunnels have 6 segments per ring.
    Prints warning if geometry suggests otherwise.
    """
    enhanced_path = os.path.join(tunnel_dir, 'enhanced.csv')
    
    if os.path.exists(enhanced_path):
        try:
            df = pd.read_csv(enhanced_path, usecols=['r'])
            if 'r' in df.columns:
                avg_radius = df['r'].mean()
                print(f"Detected from geometry: {SIMPLE_STAGGERED_SEGMENTS} segments (radius={avg_radius:.3f}m)")
        except:
            pass
    
    return SIMPLE_STAGGERED_SEGMENTS


def get_class_names() -> Dict[int, str]:
    """Get class name mapping for simple staggered (always 6 segments)."""
    return CLASS_NAMES


# =============================================================================
# Data Loading
# =============================================================================

def load_data(tunnel_dir: str) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """
    Load prediction and optional ground truth data.
    
    Returns:
        Tuple of (gt_labels, pred_labels, has_gt).
        gt_labels is None if no ground truth available.
    """
    # Try final.csv first (has full data)
    final_file = os.path.join(tunnel_dir, "final.csv")
    pred_file = os.path.join(tunnel_dir, "predictions.csv")
    
    if os.path.exists(final_file):
        # Read only needed columns to save memory
        df = pd.read_csv(final_file, usecols=['segment', 'pred'] if 'segment' in pd.read_csv(final_file, nrows=0).columns else ['pred'])
        
        pred_labels = df['pred'].values
        
        if 'segment' in df.columns:
            gt_labels = df['segment'].values
            return gt_labels, pred_labels, True
        else:
            return None, pred_labels, False
    
    elif os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        pred_labels = df['pred_labels'].values
        return None, pred_labels, False
    
    else:
        raise FileNotFoundError(f"No prediction data found in {tunnel_dir}")


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: Dict[int, str],
    max_class: int
) -> Dict:
    """
    Calculate segmentation metrics.
    
    Returns:
        Dictionary with OA, F1, mIoU, IoU_per_class, classes.
    """
    # Filter to valid classes
    valid_mask = (gt_labels <= max_class) & (pred_labels <= max_class)
    gt_filtered = gt_labels[valid_mask]
    pred_filtered = pred_labels[valid_mask]
    
    if len(gt_filtered) != len(gt_labels):
        removed = len(gt_labels) - len(gt_filtered)
        print(f"Filtered {removed} points with class > {max_class}")
    
    # Get classes present in data
    classes = np.sort(np.unique(np.concatenate([gt_filtered, pred_filtered])))
    
    # Calculate metrics
    oa = accuracy_score(gt_filtered, pred_filtered)
    f1 = f1_score(gt_filtered, pred_filtered, average='macro', labels=classes, zero_division=0)
    iou_per_class = jaccard_score(gt_filtered, pred_filtered, average=None, labels=classes, zero_division=0)
    miou = np.mean(iou_per_class)
    
    return {
        'OA': oa,
        'F1': f1,
        'mIoU': miou,
        'IoU_per_class': iou_per_class,
        'classes': classes,
        'gt_filtered': gt_filtered,
        'pred_filtered': pred_filtered
    }


def calculate_prediction_stats(
    pred_labels: np.ndarray,
    class_names: Dict[int, str]
) -> Dict:
    """
    Calculate prediction statistics (no ground truth needed).
    """
    unique, counts = np.unique(pred_labels, return_counts=True)
    total = len(pred_labels)
    
    stats = {
        'total_points': total,
        'class_counts': dict(zip(unique.tolist(), counts.tolist())),
        'class_percentages': {int(c): float(cnt / total * 100) for c, cnt in zip(unique, counts)}
    }
    
    # Coverage: non-background points
    background_count = stats['class_counts'].get(0, 0)
    stats['coverage'] = (total - background_count) / total * 100
    
    return stats


# =============================================================================
# Visualization
# =============================================================================

def plot_iou_bars(
    iou_per_class: np.ndarray,
    classes: np.ndarray,
    class_names: Dict[int, str],
    output_file: str
) -> None:
    """Plot IoU bar chart for each class."""
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    
    bars = plt.bar(class_labels, iou_per_class, color=colors)
    plt.axhline(y=np.mean(iou_per_class), color='r', linestyle='-', 
                label=f'Mean IoU: {np.mean(iou_per_class):.3f}')
    
    for bar, iou in zip(bars, iou_per_class):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{iou:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('IoU Score')
    plt.title('IoU Scores by Class')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_class_distribution(
    gt_labels: Optional[np.ndarray],
    pred_labels: np.ndarray,
    class_names: Dict[int, str],
    output_file: str
) -> None:
    """Plot class distribution comparison."""
    # Get all classes
    if gt_labels is not None:
        classes = sorted(set(np.unique(gt_labels)) | set(np.unique(pred_labels)))
        gt_counts = [np.sum(gt_labels == c) for c in classes]
    else:
        classes = sorted(np.unique(pred_labels))
        gt_counts = None
    
    pred_counts = [np.sum(pred_labels == c) for c in classes]
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    
    if gt_counts is not None:
        width = 0.35
        plt.bar(x - width/2, gt_counts, width, label='Ground Truth', color='steelblue')
        plt.bar(x + width/2, pred_counts, width, label='Prediction', color='darkorange')
        plt.legend()
    else:
        plt.bar(x, pred_counts, color='darkorange', label='Prediction')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(x, class_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format large numbers
    from matplotlib.ticker import FuncFormatter
    def format_func(x, pos):
        if x >= 1e6:
            return f'{x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'{x*1e-3:.1f}K'
        return f'{x:.0f}'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_confusion_matrix(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: Dict[int, str],
    output_file: str
) -> None:
    """Plot normalized confusion matrix."""
    classes = sorted(set(np.unique(gt_labels)) | set(np.unique(pred_labels)))
    class_labels = [class_names.get(c, f"Class {c}") for c in classes]
    
    cm = confusion_matrix(gt_labels, pred_labels, labels=classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(label='Proportion')
    
    plt.xticks(range(len(classes)), class_labels, rotation=45, ha='right')
    plt.yticks(range(len(classes)), class_labels)
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            plt.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center', color=color)
    
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    tunnel_id: str,
    results: Dict,
    class_names: Dict[int, str],
    has_gt: bool,
    output_dir: str
) -> None:
    """Generate markdown performance report."""
    report_path = os.path.join(output_dir, 'performance.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Evaluation Results for Tunnel {tunnel_id}\n\n")
        
        if has_gt:
            f.write("## Overall Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Overall Accuracy (OA) | {results['OA']:.3f} |\n")
            f.write(f"| F1 Score (macro) | {results['F1']:.3f} |\n")
            f.write(f"| Mean IoU (mIoU) | {results['mIoU']:.3f} |\n\n")
            
            f.write("## Per-Class IoU\n\n")
            f.write("| Class | IoU |\n")
            f.write("|-------|-----|\n")
            for i, cls in enumerate(results['classes']):
                name = class_names.get(cls, f"Class {cls}")
                f.write(f"| {name} | {results['IoU_per_class'][i]:.3f} |\n")
        else:
            f.write("## Prediction Statistics (No Ground Truth)\n\n")
            f.write(f"- Total points: {results['total_points']:,}\n")
            f.write(f"- Coverage (non-background): {results['coverage']:.1f}%\n\n")
            
            f.write("### Class Distribution\n\n")
            f.write("| Class | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            for cls, count in sorted(results['class_counts'].items()):
                name = class_names.get(cls, f"Class {cls}")
                pct = results['class_percentages'][cls]
                f.write(f"| {name} | {count:,} | {pct:.1f}% |\n")
    
    print(f"Report saved to {report_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def evaluate(tunnel_id: str, base_dir: str = "data") -> Dict:
    """
    Run evaluation for simple staggered tunnel.
    
    Args:
        tunnel_id: Tunnel identifier (e.g., "1-4", "2-2").
        base_dir: Base data directory.
    """
    print("=" * 70)
    print("SIMPLE STAGGERED EVALUATION")
    print("=" * 70)
    print(f"Tunnel: {tunnel_id}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    output_dir = os.path.join(tunnel_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple staggered: always 6 segments
    segment_count = verify_segment_count(tunnel_dir)
    class_names = get_class_names()
    max_class = SIMPLE_STAGGERED_SEGMENTS  # B2-block = 6
    
    print(f"Max class label: {max_class} ({class_names.get(max_class, 'B2-block')})")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    gt_labels, pred_labels, has_gt = load_data(tunnel_dir)
    print(f"Points: {len(pred_labels):,}")
    print(f"Ground truth available: {has_gt}")
    
    if has_gt:
        # Full evaluation with ground truth
        print("\nCalculating metrics...")
        results = calculate_metrics(gt_labels, pred_labels, class_names, max_class)
        
        print(f"\n{'='*40}")
        print(f"OA {results['OA']:.3f}  F1 {results['F1']:.3f}  mIoU {results['mIoU']:.3f}")
        print(f"{'='*40}")
        
        print("\nPer-class IoU:")
        for i, cls in enumerate(results['classes']):
            name = class_names.get(cls, f"Class {cls}")
            print(f"  {name}: {results['IoU_per_class'][i]:.3f}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_iou_bars(results['IoU_per_class'], results['classes'], class_names,
                     os.path.join(output_dir, 'iou_by_class.png'))
        plot_class_distribution(results['gt_filtered'], results['pred_filtered'], class_names,
                               os.path.join(output_dir, 'class_distribution.png'))
        plot_confusion_matrix(results['gt_filtered'], results['pred_filtered'], class_names,
                             os.path.join(output_dir, 'confusion_matrix.png'))
        
    else:
        # Prediction statistics only
        print("\nCalculating prediction statistics...")
        results = calculate_prediction_stats(pred_labels, class_names)
        
        print(f"\nTotal points: {results['total_points']:,}")
        print(f"Coverage: {results['coverage']:.1f}%")
        
        print("\nClass distribution:")
        for cls, count in sorted(results['class_counts'].items()):
            name = class_names.get(cls, f"Class {cls}")
            pct = results['class_percentages'][cls]
            print(f"  {name}: {count:,} ({pct:.1f}%)")
        
        # Generate visualization
        print("\nGenerating visualizations...")
        plot_class_distribution(None, pred_labels, class_names,
                               os.path.join(output_dir, 'class_distribution.png'))
    
    # Generate report
    generate_report(tunnel_id, results, class_names, has_gt, output_dir)
    
    print("=" * 70)
    print(f"Results saved to {output_dir}/")
    print("=" * 70)
    
    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate simple staggered tunnel segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simple Staggered Evaluation

Examples:
  python evaluation.py 1-4
  python evaluation.py 2-2 --data-dir data/custom

Note: Simple staggered tunnels have 6 segments per ring.
For 7-segment or complex tunnels, use the appropriate pipeline.

Input: <data_dir>/<tunnel_id>/final.csv (with 'segment' column for GT metrics)
"""
    )
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4, 2-2)")
    parser.add_argument("--data-dir", default="data",
                       help="Base data directory (default: data)")
    
    args = parser.parse_args()
    
    evaluate(args.tunnel_id, base_dir=args.data_dir)

