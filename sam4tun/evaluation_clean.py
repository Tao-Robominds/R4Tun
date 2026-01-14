"""
Segmentation Evaluation (GT-Free Compatible)

This module evaluates tunnel segmentation quality. It supports two modes:

1. WITH Ground Truth: Compare predictions against ground truth labels
   - Input: final.csv with 'segment' (GT) and 'pred' columns
   - Outputs: OA, F1, mIoU metrics

2. WITHOUT Ground Truth: Generate prediction statistics only
   - Input: final.csv with 'pred' column (or predictions.csv)
   - Outputs: Class distribution, coverage analysis

Segment count is auto-detected from inferred_from_pattern.csv.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix


# =============================================================================
# Constants
# =============================================================================

# Class names for different segment counts
CLASS_NAMES_6 = {
    0: 'Background',
    1: 'K-block',
    2: 'B1-block',
    3: 'A1-block',
    4: 'A2-block',
    5: 'A3-block',
    6: 'B2-block'
}

CLASS_NAMES_7 = {
    0: 'Background',
    1: 'K-block',
    2: 'B1-block',
    3: 'A1-block',
    4: 'A2-block',
    5: 'A3-block',
    6: 'A4-block',
    7: 'B2-block'
}


# =============================================================================
# Segment Count Detection
# =============================================================================

def detect_segment_count(tunnel_dir: str) -> int:
    """
    Auto-detect segment count from pattern file.
    
    Reads inferred_from_pattern.csv and counts unique Block types.
    """
    pattern_file = os.path.join(tunnel_dir, "inferred_from_pattern.csv")
    
    if os.path.exists(pattern_file):
        df = pd.read_csv(pattern_file)
        if 'Block' in df.columns:
            unique_blocks = df['Block'].unique()
            segment_count = len(unique_blocks)
            print(f"Detected {segment_count} segments from pattern: {sorted(unique_blocks)}")
            return segment_count
    
    # Fallback: try to detect from predictions
    pred_file = os.path.join(tunnel_dir, "predictions.csv")
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        max_label = df['pred_labels'].max()
        if max_label <= 6:
            return 6
        else:
            return 7
    
    print("Warning: Could not detect segment count, defaulting to 6")
    return 6


def get_class_names(segment_count: int) -> Dict[int, str]:
    """Get class name mapping based on segment count."""
    if segment_count <= 6:
        return CLASS_NAMES_6
    else:
        return CLASS_NAMES_7


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

def evaluate(
    tunnel_id: str,
    base_dir: str = "data",
    segment_count: Optional[int] = None
) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        tunnel_id: Tunnel identifier.
        base_dir: Base data directory.
        segment_count: Number of segments (auto-detected if None).
    """
    print("=" * 70)
    print("SEGMENTATION EVALUATION")
    print("=" * 70)
    print(f"Tunnel: {tunnel_id}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    output_dir = os.path.join(tunnel_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect segment count
    if segment_count is None:
        segment_count = detect_segment_count(tunnel_dir)
    else:
        print(f"Using specified segment count: {segment_count}")
    
    class_names = get_class_names(segment_count)
    max_class = segment_count  # B2-block label
    
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
    if len(sys.argv) < 2:
        print("Usage: python evaluation_clean.py <tunnel_id> [segment_count]")
        print()
        print("Arguments:")
        print("  tunnel_id      Tunnel identifier (e.g., 1-4, 4-1)")
        print("  segment_count  Number of segments per ring (auto-detected if omitted)")
        print()
        print("Examples:")
        print("  python evaluation_clean.py 1-4      # Auto-detect segments")
        print("  python evaluation_clean.py 4-1 7    # 7 segments")
        print()
        print("Input files (in order of preference):")
        print("  - data/<tunnel_id>/final.csv (with 'segment' column for GT)")
        print("  - data/<tunnel_id>/predictions.csv")
        sys.exit(1)
    
    tunnel_id = sys.argv[1]
    segment_count = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    evaluate(tunnel_id, segment_count=segment_count)

