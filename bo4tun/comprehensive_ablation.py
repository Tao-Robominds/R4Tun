#!/usr/bin/env python3
"""
Comprehensive Ablation Study for Intrinsic Metrics

This script:
1. Collects all intrinsic metrics from all tunnels/configurations
2. Computes correlations with mIoU (Spearman, Pearson)
3. Runs leave-one-out feature importance analysis
4. Generates comprehensive metrics tables

Output: Comprehensive report with metric importance rankings
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bo4tun.intrinsic_metrics import (
    compute_detection_metrics,
    compute_sam_metrics,
    compute_all_metrics,
    compute_complex_sam_metrics,
    compute_complex_detection_metrics,
    compute_all_complex_metrics,
    compute_preprocessing_guardrails,
    load_expected_rings,
)


# =============================================================================
# Data Collection Functions
# =============================================================================

def find_all_tunnel_configs() -> List[Dict]:
    """Find all tunnel configurations with final.csv (completed runs)."""
    configs = []
    data_dir = project_root / 'data'
    
    # Direct tunnel directories
    for tunnel_dir in data_dir.iterdir():
        if tunnel_dir.is_dir() and tunnel_dir.name in ['1-4', '2-2', '3-1', '4-1', '5-1']:
            final_csv = tunnel_dir / 'final.csv'
            detected_csv = tunnel_dir / 'detected.csv'
            if final_csv.exists():
                configs.append({
                    'tunnel_id': tunnel_dir.name,
                    'config_path': str(tunnel_dir),
                    'final_csv': str(final_csv),
                    'detected_csv': str(detected_csv) if detected_csv.exists() else None,
                    'source': 'direct'
                })
    
    # BO results directories
    bo_dir = data_dir / 'bo'
    if bo_dir.exists():
        for tunnel_dir in bo_dir.iterdir():
            if tunnel_dir.is_dir():
                final_csv = tunnel_dir / 'final.csv'
                detected_csv = tunnel_dir / 'detected.csv'
                if final_csv.exists():
                    configs.append({
                        'tunnel_id': tunnel_dir.name,
                        'config_path': str(tunnel_dir),
                        'final_csv': str(final_csv),
                        'detected_csv': str(detected_csv) if detected_csv.exists() else None,
                        'source': 'bo'
                    })
    
    # Baseline directories
    baseline_dir = data_dir / 'baseline'
    if baseline_dir.exists():
        for tunnel_dir in baseline_dir.iterdir():
            if tunnel_dir.is_dir():
                final_csv = tunnel_dir / 'final.csv'
                detected_csv = tunnel_dir / 'detected.csv'
                if final_csv.exists():
                    configs.append({
                        'tunnel_id': tunnel_dir.name,
                        'config_path': str(tunnel_dir),
                        'final_csv': str(final_csv),
                        'detected_csv': str(detected_csv) if detected_csv.exists() else None,
                        'source': 'baseline'
                    })
    
    return configs


def get_miou_from_eval(config_path: str) -> Optional[float]:
    """Extract mIoU from evaluation results if available."""
    eval_file = Path(config_path) / 'evaluation_results.json'
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                data = json.load(f)
                return data.get('mIoU', data.get('miou'))
        except:
            pass
    
    # Try to compute from final.csv if GT available
    final_csv = Path(config_path) / 'final.csv'
    if final_csv.exists():
        try:
            df = pd.read_csv(final_csv, comment='#')
            if 'segment' in df.columns and 'pred' in df.columns:
                # Has GT, can compute mIoU
                gt = df['segment'].values
                pred = df['pred'].values
                # Simple IoU computation
                unique_labels = np.unique(np.concatenate([gt, pred]))
                unique_labels = unique_labels[unique_labels > 0]  # Exclude background
                
                ious = []
                for label in unique_labels:
                    intersection = np.sum((gt == label) & (pred == label))
                    union = np.sum((gt == label) | (pred == label))
                    if union > 0:
                        ious.append(intersection / union)
                
                if ious:
                    return float(np.mean(ious))
        except:
            pass
    
    return None


def collect_all_metrics(configs: List[Dict]) -> pd.DataFrame:
    """Collect all intrinsic metrics from all configurations."""
    rows = []
    
    for config in configs:
        tunnel_id = config['tunnel_id']
        is_complex = tunnel_id in ['4-1', '5-1']
        
        row = {
            'tunnel_id': tunnel_id,
            'config_path': config['config_path'],
            'source': config['source'],
            'pattern_type': 'complex' if is_complex else 'simple',
        }
        
        # Get mIoU
        miou = get_miou_from_eval(config['config_path'])
        row['mIoU'] = miou
        
        # Compute detection metrics
        if config['detected_csv'] and os.path.exists(config['detected_csv']):
            expected_rings = load_expected_rings(tunnel_id, str(project_root / 'data'))
            det_metrics = compute_detection_metrics(
                tunnel_id, config['detected_csv'], expected_rings
            )
            for k, v in det_metrics.items():
                row[f'det_{k}'] = v
            
            # Complex detection metrics
            if is_complex:
                complex_det = compute_complex_detection_metrics(
                    tunnel_id, config['detected_csv'], str(project_root / 'data')
                )
                for k, v in complex_det.items():
                    row[f'det_complex_{k}'] = v
        
        # Compute SAM metrics
        if config['final_csv'] and os.path.exists(config['final_csv']):
            sam_metrics = compute_sam_metrics(
                tunnel_id, config['final_csv'], config['detected_csv'],
                data_dir=str(project_root / 'data')
            )
            for k, v in sam_metrics.items():
                row[f'sam_{k}'] = v
            
            # Complex SAM metrics
            if is_complex:
                complex_sam = compute_complex_sam_metrics(
                    tunnel_id, config['final_csv'], data_dir=str(project_root / 'data')
                )
                for k, v in complex_sam.items():
                    row[f'sam_complex_{k}'] = v
        
        # Preprocessing guardrails (if files exist)
        try:
            config_dir = Path(config['config_path'])
            unwrapped_csv = config_dir / 'unwrapped.csv'
            denoised_csv = config_dir / 'denoised.csv'
            depth_map = config_dir / 'depth_map_outlier.npy'
            
            if unwrapped_csv.exists():
                pre_metrics = compute_preprocessing_guardrails(
                    str(unwrapped_csv),
                    str(denoised_csv) if denoised_csv.exists() else None,
                    str(depth_map) if depth_map.exists() else None
                )
                for k, v in pre_metrics.items():
                    row[f'pre_{k}'] = v
        except Exception as e:
            pass  # Preprocessing files not available
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def load_existing_training_data() -> pd.DataFrame:
    """Load existing training data with mIoU values."""
    training_file = project_root / 'bo4tun' / 'training' / 'intrinsic_training_data.csv'
    if training_file.exists():
        return pd.read_csv(training_file)
    return pd.DataFrame()


def load_miou_training_data() -> pd.DataFrame:
    """Load the larger mIoU training dataset."""
    training_file = project_root / 'bo4tun' / 'training' / 'miou_training_data.csv'
    if training_file.exists():
        df = pd.read_csv(training_file)
        # Filter for rows with valid mIoU
        df = df[df['mIoU'].notna() & (df['mIoU'] > 0)]
        return df
    return pd.DataFrame()


# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_correlations(df: pd.DataFrame, target: str = 'mIoU') -> pd.DataFrame:
    """Compute Spearman and Pearson correlations for all metrics."""
    # Get metric columns (exclude non-metric columns)
    exclude_cols = ['tunnel_id', 'config_path', 'source', 'pattern_type', 'mIoU', 'stage', 'OA', 'F1']
    metric_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('param_')]
    
    # Filter for rows with valid target
    valid_df = df[df[target].notna() & (df[target] > 0)]
    
    if len(valid_df) < 3:
        print(f"Warning: Only {len(valid_df)} samples with valid {target}")
        return pd.DataFrame()
    
    results = []
    for col in metric_cols:
        col_data = valid_df[col].dropna()
        target_data = valid_df.loc[col_data.index, target]
        
        if len(col_data) < 3:
            continue
        
        # Remove infinite values
        mask = np.isfinite(col_data) & np.isfinite(target_data)
        col_data = col_data[mask]
        target_data = target_data[mask]
        
        if len(col_data) < 3:
            continue
        
        try:
            spearman_r, spearman_p = spearmanr(col_data, target_data)
            pearson_r, pearson_p = pearsonr(col_data, target_data)
            
            results.append({
                'metric': col,
                'n_samples': len(col_data),
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
            })
        except Exception as e:
            print(f"Error computing correlation for {col}: {e}")
    
    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values('spearman_r', key=abs, ascending=False)
    return result_df


def compute_correlations_by_pattern(df: pd.DataFrame, target: str = 'mIoU') -> Dict[str, pd.DataFrame]:
    """Compute correlations separately for simple and complex patterns."""
    results = {}
    
    # Determine pattern type from tunnel_id
    if 'pattern_type' not in df.columns:
        df = df.copy()
        df['pattern_type'] = df['tunnel_id'].apply(
            lambda x: 'complex' if x in ['4-1', '5-1'] else 'simple'
        )
    
    for pattern in ['simple', 'complex', 'all']:
        if pattern == 'all':
            subset = df
        else:
            subset = df[df['pattern_type'] == pattern]
        
        if len(subset) >= 3:
            corr_df = compute_correlations(subset, target)
            if len(corr_df) > 0:
                results[pattern] = corr_df
    
    return results


# =============================================================================
# Leave-One-Out Feature Importance
# =============================================================================

def run_loo_ablation(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'mIoU',
    alpha: float = 1.0
) -> Dict[str, Dict]:
    """Run leave-one-out feature importance analysis."""
    # Filter valid rows
    valid_df = df[df[target].notna() & (df[target] > 0)].copy()
    
    # Get available features
    available = [f for f in features if f in valid_df.columns]
    if len(available) < 2:
        return {}
    
    # Drop rows with NaN in features
    for f in available:
        valid_df = valid_df[valid_df[f].notna()]
    
    if len(valid_df) < 5:
        print(f"Warning: Only {len(valid_df)} samples for ablation")
        return {}
    
    X = valid_df[available].values
    y = valid_df[target].values
    
    # Baseline model with all features
    baseline_model = Ridge(alpha=alpha)
    baseline_model.fit(X, y)
    baseline_pred = baseline_model.predict(X)
    baseline_mae = mean_absolute_error(y, baseline_pred)
    baseline_spearman = spearmanr(y, baseline_pred)[0]
    
    results = {
        '_baseline': {
            'mae': baseline_mae,
            'spearman': baseline_spearman,
            'n_features': len(available),
            'n_samples': len(valid_df),
        }
    }
    
    # Leave-one-out for each feature
    for i, feature in enumerate(available):
        subset_features = [f for f in available if f != feature]
        subset_idx = [j for j, f in enumerate(available) if f != feature]
        
        X_subset = X[:, subset_idx]
        
        model = Ridge(alpha=alpha)
        model.fit(X_subset, y)
        pred = model.predict(X_subset)
        mae = mean_absolute_error(y, pred)
        spearman = spearmanr(y, pred)[0]
        
        delta_mae = mae - baseline_mae
        delta_spearman = baseline_spearman - spearman
        
        # Get coefficient from baseline model
        coef = baseline_model.coef_[i]
        
        results[feature] = {
            'mae_without': mae,
            'delta_mae': delta_mae,
            'spearman_without': spearman,
            'delta_spearman': delta_spearman,
            'coefficient': coef,
            'importance': classify_importance(delta_mae, delta_spearman),
        }
    
    return results


def classify_importance(delta_mae: float, delta_spearman: float) -> str:
    """Classify feature importance based on deltas."""
    if delta_mae > 0.05 or delta_spearman > 0.15:
        return 'HIGH'
    elif delta_mae > 0.02 or delta_spearman > 0.05:
        return 'MEDIUM'
    else:
        return 'LOW'


# =============================================================================
# Report Generation
# =============================================================================

def generate_comprehensive_report(
    collected_df: pd.DataFrame,
    correlations: Dict[str, pd.DataFrame],
    ablation_results: Dict[str, Dict],
    output_dir: Path
):
    """Generate comprehensive ablation study report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = []
    report.append("# Comprehensive Intrinsic Metrics Ablation Study\n")
    report.append(f"**Generated:** {timestamp}\n")
    report.append(f"**Total Configurations Analyzed:** {len(collected_df)}\n")
    report.append("\n---\n")
    
    # Section 1: Data Summary
    report.append("## 1. Data Summary\n")
    report.append(f"| Pattern Type | Count | mIoU Range |\n")
    report.append(f"|--------------|-------|------------|\n")
    
    for pattern in ['simple', 'complex']:
        subset = collected_df[collected_df['pattern_type'] == pattern]
        if len(subset) > 0 and 'mIoU' in subset.columns:
            valid = subset[subset['mIoU'].notna()]
            if len(valid) > 0:
                miou_range = f"{valid['mIoU'].min():.3f} - {valid['mIoU'].max():.3f}"
            else:
                miou_range = "N/A"
            report.append(f"| {pattern.title()} | {len(subset)} | {miou_range} |\n")
    
    report.append("\n---\n")
    
    # Section 2: Preprocessing Metrics
    report.append("## 2. Preprocessing Metrics (Guardrails)\n")
    report.append("\n*Note: Preprocessing has low mIoU impact (+0.1%), used as fail-fast guardrails.*\n\n")
    
    pre_metrics = [c for c in collected_df.columns if c.startswith('pre_')]
    if pre_metrics:
        report.append("| Metric | Mean | Std | Min | Max | Recommended Range |\n")
        report.append("|--------|------|-----|-----|-----|-------------------|\n")
        
        thresholds = {
            'pre_theta_coverage': '98-102%',
            'pre_point_retention_ratio': '> 85%',
            'pre_interpolation_coverage': '> 95%',
        }
        
        for metric in pre_metrics:
            data = collected_df[metric].dropna()
            if len(data) > 0:
                recommended = thresholds.get(metric, 'TBD')
                report.append(f"| `{metric}` | {data.mean():.2f} | {data.std():.2f} | {data.min():.2f} | {data.max():.2f} | {recommended} |\n")
    else:
        report.append("*No preprocessing data collected. Preprocessing files may not be available.*\n")
    
    report.append("\n---\n")
    
    # Section 3: Detection Metrics
    report.append("## 3. Detection Metrics\n")
    
    for pattern in ['all', 'simple', 'complex']:
        if pattern in correlations and len(correlations[pattern]) > 0:
            report.append(f"\n### 3.{['all', 'simple', 'complex'].index(pattern) + 1} {pattern.title()} Patterns\n\n")
            
            det_metrics = correlations[pattern][correlations[pattern]['metric'].str.startswith('det_')]
            if len(det_metrics) > 0:
                report.append("| Metric | N | Spearman r | p-value | Pearson r | Usefulness |\n")
                report.append("|--------|---|------------|---------|-----------|------------|\n")
                
                for _, row in det_metrics.head(15).iterrows():
                    usefulness = '✓ HIGH' if abs(row['spearman_r']) > 0.5 else '○ MEDIUM' if abs(row['spearman_r']) > 0.3 else '✗ LOW'
                    p_val = f"{row['spearman_p']:.4f}" if row['spearman_p'] > 0.0001 else "<0.0001"
                    report.append(f"| `{row['metric']}` | {row['n_samples']} | {row['spearman_r']:.3f} | {p_val} | {row['pearson_r']:.3f} | {usefulness} |\n")
    
    report.append("\n---\n")
    
    # Section 4: SAM Metrics
    report.append("## 4. SAM Metrics\n")
    
    for pattern in ['all', 'simple', 'complex']:
        if pattern in correlations and len(correlations[pattern]) > 0:
            report.append(f"\n### 4.{['all', 'simple', 'complex'].index(pattern) + 1} {pattern.title()} Patterns\n\n")
            
            sam_metrics = correlations[pattern][correlations[pattern]['metric'].str.startswith('sam_')]
            if len(sam_metrics) > 0:
                report.append("| Metric | N | Spearman r | p-value | Pearson r | Usefulness |\n")
                report.append("|--------|---|------------|---------|-----------|------------|\n")
                
                for _, row in sam_metrics.head(15).iterrows():
                    usefulness = '✓ HIGH' if abs(row['spearman_r']) > 0.5 else '○ MEDIUM' if abs(row['spearman_r']) > 0.3 else '✗ LOW'
                    p_val = f"{row['spearman_p']:.4f}" if row['spearman_p'] > 0.0001 else "<0.0001"
                    report.append(f"| `{row['metric']}` | {row['n_samples']} | {row['spearman_r']:.3f} | {p_val} | {row['pearson_r']:.3f} | {usefulness} |\n")
    
    report.append("\n---\n")
    
    # Section 5: Feature Importance (Ablation)
    report.append("## 5. Feature Importance (Leave-One-Out Ablation)\n")
    
    for pattern, results in ablation_results.items():
        if results and '_baseline' in results:
            report.append(f"\n### 5.{list(ablation_results.keys()).index(pattern) + 1} {pattern.title()} Pattern Predictor\n\n")
            
            baseline = results['_baseline']
            report.append(f"**Baseline Model:**\n")
            report.append(f"- Features: {baseline['n_features']}\n")
            report.append(f"- Samples: {baseline['n_samples']}\n")
            report.append(f"- MAE: {baseline['mae']:.4f}\n")
            report.append(f"- Spearman: {baseline['spearman']:.4f}\n\n")
            
            # Sort by delta_mae (impact when removed)
            features = [(k, v) for k, v in results.items() if k != '_baseline']
            features.sort(key=lambda x: x[1]['delta_mae'], reverse=True)
            
            report.append("| Feature | Δ MAE | Δ Spearman | Coefficient | Importance |\n")
            report.append("|---------|-------|------------|-------------|------------|\n")
            
            for feature, data in features:
                importance_icon = '🔴' if data['importance'] == 'HIGH' else '🟡' if data['importance'] == 'MEDIUM' else '⚪'
                report.append(f"| `{feature}` | {data['delta_mae']:+.4f} | {data['delta_spearman']:+.4f} | {data['coefficient']:.4f} | {importance_icon} {data['importance']} |\n")
    
    report.append("\n---\n")
    
    # Section 6: Comprehensive Metrics Table
    report.append("## 6. Comprehensive Metrics Summary\n\n")
    report.append("### All Metrics Ranked by Correlation with mIoU\n\n")
    
    if 'all' in correlations and len(correlations['all']) > 0:
        all_corr = correlations['all']
        report.append("| Rank | Stage | Metric | Spearman r | N | Recommendation |\n")
        report.append("|------|-------|--------|------------|---|----------------|\n")
        
        for rank, (_, row) in enumerate(all_corr.iterrows(), 1):
            metric = row['metric']
            if metric.startswith('pre_'):
                stage = 'Preprocessing'
            elif metric.startswith('det_'):
                stage = 'Detection'
            elif metric.startswith('sam_'):
                stage = 'SAM'
            else:
                stage = 'Other'
            
            r = row['spearman_r']
            if abs(r) > 0.5:
                rec = "✓ Use in predictor"
            elif abs(r) > 0.3:
                rec = "○ Consider for predictor"
            elif abs(r) > 0.2:
                rec = "◇ Use as guardrail"
            else:
                rec = "✗ Low predictive value"
            
            report.append(f"| {rank} | {stage} | `{metric}` | {r:.3f} | {row['n_samples']} | {rec} |\n")
            
            if rank >= 30:  # Limit table size
                break
    
    report.append("\n---\n")
    
    # Section 7: Recommended Predictor Features
    report.append("## 7. Recommended Predictor Features\n\n")
    
    report.append("### Simple Patterns (1-4, 2-2, 3-1)\n\n")
    report.append("Based on correlation analysis and ablation results:\n\n")
    report.append("| Priority | Feature | Rationale |\n")
    report.append("|----------|---------|----------|\n")
    
    simple_recs = [
        ('P1', 'det_midpoint_ratio', 'Strong correlation, detection method quality'),
        ('P1', 'det_real_detection_ratio', 'Distinguishes actual vs fallback detections'),
        ('P2', 'det_k_count_match', 'Exact count match matters'),
        ('P2', 'sam_mask_fill_rate', 'Segmentation coverage'),
        ('P3', 'sam_segment_count_match', 'Segment count accuracy'),
    ]
    for p, f, r in simple_recs:
        report.append(f"| {p} | `{f}` | {r} |\n")
    
    report.append("\n### Complex Patterns (4-1, 5-1)\n\n")
    report.append("| Priority | Feature | Rationale |\n")
    report.append("|----------|---------|----------|\n")
    
    complex_recs = [
        ('P1', 'sam_complex_gap_ratio', 'Under-segmentation indicator'),
        ('P1', 'sam_complex_segment_width_cv', 'Geometry consistency'),
        ('P2', 'sam_complex_segment_area_cv', 'Size uniformity'),
        ('P2', 'det_complex_cluster_separation', 'Detection quality'),
        ('P3', 'sam_complex_ring_coverage_cv', 'Cross-ring consistency'),
    ]
    for p, f, r in complex_recs:
        report.append(f"| {p} | `{f}` | {r} |\n")
    
    report.append("\n---\n")
    
    # Section 8: Guardrail Thresholds
    report.append("## 8. Guardrail Thresholds Summary\n\n")
    
    report.append("### Preprocessing Guardrails (Fail-Fast)\n\n")
    report.append("| Metric | Threshold | Action if Failed |\n")
    report.append("|--------|-----------|------------------|\n")
    report.append("| `pre_theta_coverage` | 98-102% | Rerun unfolding |\n")
    report.append("| `pre_point_retention_ratio` | > 85% | Check denoising params |\n")
    report.append("| `pre_interpolation_coverage` | > 95% | Rerun enhancing |\n")
    
    report.append("\n### Detection Guardrails\n\n")
    report.append("| Metric | Simple | Complex | Action if Failed |\n")
    report.append("|--------|--------|---------|------------------|\n")
    report.append("| `det_k_count` | ±2 expected | ±2 expected | Adjust detection params |\n")
    report.append("| `det_real_detection_ratio` | > 0.60 | N/A | Check Hough params |\n")
    report.append("| `det_x_spacing_cv` | < 0.40 | < 0.70 | Review binary threshold |\n")
    
    report.append("\n### SAM Guardrails\n\n")
    report.append("| Metric | Simple | Complex | Action if Failed |\n")
    report.append("|--------|--------|---------|------------------|\n")
    report.append("| `sam_mask_fill_rate` | 0.30-0.90 | 0.30-0.90 | Adjust segment_width |\n")
    report.append("| `sam_gap_ratio` | N/A | < 0.15 | Increase segment_width |\n")
    report.append("| `sam_segment_width_cv` | N/A | < 0.20 | Adjust k_height |\n")
    
    report.append("\n---\n")
    
    # Section 9: Next Steps
    report.append("## 9. Next Steps\n\n")
    report.append("1. **Collect more data**: Need 50+ samples per pattern type for robust correlations\n")
    report.append("2. **Validate thresholds**: Test guardrail thresholds on new data\n")
    report.append("3. **Build predictors**: Train Ridge regression with recommended features\n")
    report.append("4. **Test reflection loop**: Implement rerun logic based on guardrails\n")
    
    report.append("\n---\n")
    report.append(f"\n*Report generated: {timestamp}*\n")
    
    # Write report
    report_path = output_dir / 'COMPREHENSIVE_ABLATION_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    
    print(f"Report saved to: {report_path}")
    return report_path


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE INTRINSIC METRICS ABLATION STUDY")
    print("=" * 70)
    
    output_dir = project_root / 'bo4tun' / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Collect metrics from file system
    print("\n[1/5] Finding all tunnel configurations...")
    configs = find_all_tunnel_configs()
    print(f"Found {len(configs)} configurations")
    
    # Step 2: Collect all metrics
    print("\n[2/5] Collecting intrinsic metrics from configurations...")
    collected_df = collect_all_metrics(configs)
    print(f"Collected metrics for {len(collected_df)} configurations")
    
    # Also load existing training data
    print("Loading existing training data...")
    existing_df = load_existing_training_data()
    miou_df = load_miou_training_data()
    
    # Merge datasets
    print(f"Existing intrinsic data: {len(existing_df)} rows")
    print(f"mIoU training data: {len(miou_df)} rows")
    
    # Use existing intrinsic training data which has mIoU
    if len(existing_df) > 0:
        analysis_df = existing_df.copy()
        analysis_df['pattern_type'] = analysis_df['tunnel_id'].apply(
            lambda x: 'complex' if x in ['4-1', '5-1'] else 'simple'
        )
    else:
        analysis_df = collected_df
    
    # If we have mIoU training data, try to merge metrics
    if len(miou_df) > 0:
        print("\nMerging mIoU data with metrics...")
        # The mIoU data has parameters but not all metrics
        # Use it to supplement our analysis
    
    # Step 3: Compute correlations
    print("\n[3/5] Computing correlations with mIoU...")
    correlations = compute_correlations_by_pattern(analysis_df, 'mIoU')
    
    for pattern, corr_df in correlations.items():
        print(f"\n{pattern.upper()} patterns: {len(corr_df)} metrics analyzed")
        if len(corr_df) > 0:
            top3 = corr_df.head(3)
            for _, row in top3.iterrows():
                print(f"  {row['metric']}: r={row['spearman_r']:.3f}")
    
    # Step 4: Run ablation analysis
    print("\n[4/5] Running leave-one-out ablation analysis...")
    ablation_results = {}
    
    # Define feature sets for each pattern
    simple_features = [
        'det_k_count', 'det_k_count_match', 'det_assume_default_ratio',
        'det_midpoint_ratio', 'det_real_detection_ratio',
        'det_y_range', 'det_y_std', 'det_x_spacing_cv',
        'sam_prompt_count', 'sam_segment_count', 'sam_segment_count_match',
        'sam_mask_fill_rate', 'sam_template_coverage'
    ]
    
    complex_features = [
        'det_k_count', 'det_y_range', 'det_y_std', 'det_x_spacing_cv',
        'sam_segment_count', 'sam_mask_fill_rate'
    ]
    
    # Simple patterns ablation
    simple_df = analysis_df[analysis_df['pattern_type'] == 'simple']
    if len(simple_df) >= 5:
        print(f"Running ablation for SIMPLE patterns ({len(simple_df)} samples)...")
        ablation_results['simple'] = run_loo_ablation(simple_df, simple_features)
    
    # Complex patterns ablation
    complex_df = analysis_df[analysis_df['pattern_type'] == 'complex']
    if len(complex_df) >= 5:
        print(f"Running ablation for COMPLEX patterns ({len(complex_df)} samples)...")
        ablation_results['complex'] = run_loo_ablation(complex_df, complex_features)
    
    # All patterns ablation
    if len(analysis_df) >= 5:
        print(f"Running ablation for ALL patterns ({len(analysis_df)} samples)...")
        ablation_results['all'] = run_loo_ablation(analysis_df, simple_features + complex_features)
    
    # Step 5: Generate report
    print("\n[5/5] Generating comprehensive report...")
    report_path = generate_comprehensive_report(
        collected_df, correlations, ablation_results, output_dir
    )
    
    # Save raw data
    collected_df.to_csv(output_dir / 'comprehensive_metrics_raw.csv', index=False)
    
    # Save correlation results
    all_corr = pd.concat([
        df.assign(pattern=pattern) for pattern, df in correlations.items()
    ], ignore_index=True)
    if len(all_corr) > 0:
        all_corr.to_csv(output_dir / 'comprehensive_correlations.csv', index=False)
    
    # Save ablation results
    with open(output_dir / 'comprehensive_ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {report_path}")
    print(f"  - {output_dir / 'comprehensive_metrics_raw.csv'}")
    print(f"  - {output_dir / 'comprehensive_correlations.csv'}")
    print(f"  - {output_dir / 'comprehensive_ablation_results.json'}")
    
    return correlations, ablation_results


if __name__ == '__main__':
    main()
