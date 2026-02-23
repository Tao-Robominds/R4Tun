"""
Compare Stage 2A (geometric) vs Stage 2B (SAM) BO results and lock best parameters.

This script:
1. Finds best geometric BO trial (highest mIoU)
2. Finds best SAM BO trial (highest mIoU)
3. Compares and selects the winner
4. Updates parameter files with best values
5. Runs full pipeline end-to-end and reports final mIoU
"""

import os
import sys
import json
import glob
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_best_geometric_trial(logs_dir: str, tunnel_id: str) -> dict:
    """Find best geometric BO trial (highest mIoU)."""
    pattern = os.path.join(logs_dir, f"seg_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    if not log_files:
        return None
    
    best_miou = -1.0
    best_trial = None
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if 'error' in data or 'miou' not in data:
            continue
        
        miou = data['miou']
        if miou > best_miou:
            best_miou = miou
            best_trial = data
    
    return best_trial


def find_best_sam_trial(logs_dir: str, tunnel_id: str) -> dict:
    """Find best SAM BO trial (highest mIoU)."""
    pattern = os.path.join(logs_dir, f"sam_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    if not log_files:
        return None
    
    best_miou = -1.0
    best_trial = None
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if 'error' in data:
            continue
        
        # SAM logs have mIoU in outputs.metrics.mIoU
        miou = data.get('outputs', {}).get('metrics', {}).get('mIoU', 0.0)
        if miou > best_miou:
            best_miou = miou
            best_trial = data
    
    return best_trial


def main(tunnel_id: str, data_dir: str = "data/wrap"):
    """Compare and lock best parameters."""
    geo_logs_dir = str(PROJECT_ROOT / "bo" / "complex_staggered" / f"logs_geo_{tunnel_id}")
    sam_logs_dir = str(PROJECT_ROOT / "bo" / "complex_staggered" / "logs")
    
    print(f"Comparing Stage 2A (geometric) vs Stage 2B (SAM) for {tunnel_id}...")
    
    # Find best trials
    best_geo = find_best_geometric_trial(geo_logs_dir, tunnel_id)
    best_sam = find_best_sam_trial(sam_logs_dir, tunnel_id)
    
    if best_geo is None and best_sam is None:
        raise ValueError("No valid trials found for either method")
    
    geo_miou = best_geo['miou'] if best_geo else 0.0
    # SAM logs have mIoU in outputs.metrics.mIoU
    if best_sam:
        sam_miou = best_sam.get('outputs', {}).get('metrics', {}).get('mIoU', 0.0)
    else:
        sam_miou = 0.0
    
    geo_trial_id = best_geo.get('trial_id', 'N/A') if best_geo else 'N/A'
    sam_trial_id = best_sam.get('trial', {}).get('trial_id', 'N/A') if best_sam else 'N/A'
    print(f"\nBest Geometric mIoU: {geo_miou:.4f} (trial: {geo_trial_id})")
    print(f"Best SAM mIoU:       {sam_miou:.4f} (trial: {sam_trial_id})")
    
    # Select winner
    if geo_miou > sam_miou:
        winner = 'geometric'
        winner_miou = geo_miou
        winner_trial = best_geo
        print(f"\nWinner: Geometric (mIoU = {winner_miou:.4f})")
        
        # Update geometric params
        params_file = os.path.join(
            PROJECT_ROOT, 'agents', 'irregular', '3_segmentation',
            'parameters', tunnel_id, 'parameters_geometric.json'
        )
        with open(params_file, 'w') as f:
            json.dump(winner_trial['params'], f, indent=2)
        print(f"Updated: {params_file}")
        
    else:
        winner = 'sam'
        winner_miou = sam_miou
        winner_trial = best_sam
        print(f"\nWinner: SAM (mIoU = {winner_miou:.4f})")
        
        # Update SAM params
        params_file = os.path.join(
            PROJECT_ROOT, 'agents', 'irregular', '3_segmentation',
            'parameters', tunnel_id, 'parameters_sam.json'
        )
        # Extract params from SAM trial (may be in different format)
        sam_params = winner_trial.get('params', {})
        with open(params_file, 'w') as f:
            json.dump(sam_params, f, indent=2)
        print(f"Updated: {params_file}")
    
    # Run full pipeline end-to-end
    print(f"\nRunning full pipeline with best {winner} parameters...")
    
    # Import and run detection
    detection_dir = PROJECT_ROOT / 'agents' / 'irregular' / '2_detection'
    sys.path.insert(0, str(detection_dir))
    spec = importlib.util.spec_from_file_location("detection", detection_dir / "2_detection.py")
    det_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(det_module)
    
    import io
    from contextlib import redirect_stdout, redirect_stderr
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        det_module.run_detection(tunnel_id, data_dir)
    
    # Run segmentation
    if winner == 'geometric':
        seg_dir = PROJECT_ROOT / 'agents' / 'irregular' / '3_segmentation'
        spec = importlib.util.spec_from_file_location("geo", seg_dir / "3_geometric.py")
        geo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geo_module)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            geo_module.run_geometric(tunnel_id, data_dir)
    else:
        seg_dir = PROJECT_ROOT / 'agents' / 'irregular' / '3_segmentation'
        spec = importlib.util.spec_from_file_location("sam", seg_dir / "3_sam.py")
        sam_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sam_module)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            sam_module.run_sam(tunnel_id, data_dir)
    
    # Evaluate
    eval_dir = PROJECT_ROOT / 'agents' / 'irregular'
    spec = importlib.util.spec_from_file_location("eval", eval_dir / "evaluation.py")
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    results = eval_module.evaluate(tunnel_id, data_dir)
    
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Winner: {winner.upper()}")
    # Evaluation returns different key names
    final_miou = results.get('miou') or results.get('mIoU') or results.get('mean_iou', 0.0)
    final_oa = results.get('overall_accuracy') or results.get('OA', 0.0)
    final_f1 = results.get('f1') or results.get('F1', 'N/A')
    print(f"Final mIoU: {final_miou:.4f}")
    print(f"OA: {final_oa:.4f}")
    print(f"F1: {final_f1}")
    print(f"\nPer-class IoU:")
    per_class = results.get('per_class_iou') or results.get('per_class') or {}
    for cls, iou in per_class.items():
        print(f"  {cls}: {iou:.4f}")
    
    # Save summary
    summary = {
        'winner': winner,
        'winner_miou': winner_miou,
        'final_miou': final_miou,
        'final_oa': final_oa,
        'geometric_miou': geo_miou,
        'sam_miou': sam_miou,
        'trial_ids': {
            'geometric': best_geo.get('trial_id') if best_geo else None,
            'sam': best_sam.get('trial', {}).get('trial_id') if best_sam else None,
        }
    }
    
    summary_file = os.path.join(geo_logs_dir, 'stage2_comparison_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_file}")
    
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare and lock best Stage 2 parameters")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data/wrap")
    args = parser.parse_args()
    
    main(args.tunnel_id, args.data_dir)
