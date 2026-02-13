#!/usr/bin/env python3
"""
End-to-End Pipeline Smoke Test

Runs detection, SAM segmentation, and evaluation on all tunnels
using existing preprocessed outputs in data/.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent

# Tunnel-to-agent mapping
TUNNEL_AGENT_MAP = {
    '1-4': 'simple_staggered',
    '2-2': 'simple_staggered',
    '3-1': 'continuous',
    '4-1': 'complex_staggered',
    '5-1': 'complex_staggered',
}

# Tunnel execution order
TUNNEL_ORDER = ['1-4', '2-2', '3-1', '4-1', '5-1']


def get_python_executable() -> str:
    """Get the Python executable, preferring venv if available."""
    # Check for venv
    venv_python = PROJECT_ROOT / 'venv' / 'bin' / 'python3'
    if venv_python.exists():
        return str(venv_python)
    
    # Check for .venv
    venv_python = PROJECT_ROOT / '.venv' / 'bin' / 'python3'
    if venv_python.exists():
        return str(venv_python)
    
    # Fall back to system Python
    return sys.executable


def run_stage(
    agent_type: str,
    stage: str,
    tunnel_id: str,
    data_dir: str = 'data',
    verbose: bool = True
) -> Tuple[bool, str, Optional[str]]:
    """
    Run a single stage (detection, SAM, or evaluation) for a tunnel.
    
    Args:
        agent_type: Agent type ('simple_staggered', 'continuous', 'complex_staggered')
        stage: Stage name ('detection', 'sam', 'evaluation')
        tunnel_id: Tunnel identifier (e.g., '1-4')
        data_dir: Base data directory
        verbose: Print output
    
    Returns:
        Tuple of (success: bool, stdout: str, stderr: Optional[str])
    """
    if stage == 'detection':
        script_path = PROJECT_ROOT / 'agents' / agent_type / '2_detection' / '2_detection.py'
    elif stage == 'sam':
        script_path = PROJECT_ROOT / 'agents' / agent_type / '3_segmentation' / '3_sam.py'
    elif stage == 'evaluation':
        script_path = PROJECT_ROOT / 'agents' / agent_type / 'evaluation.py'
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    if not script_path.exists():
        return False, "", f"Script not found: {script_path}"
    
    python_exe = get_python_executable()
    cmd = [
        python_exe,
        str(script_path),
        tunnel_id,
        '--data-dir', data_dir
    ]
    
    if verbose:
        print(f"  Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per stage
        )
        
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr if result.returncode != 0 else None
        
        if verbose and stdout:
            print(stdout)
        if stderr:
            print(f"  ERROR: {stderr}", file=sys.stderr)
        
        return success, stdout, stderr
    
    except subprocess.TimeoutExpired:
        return False, "", f"Stage timed out after 1 hour"
    except Exception as e:
        return False, "", f"Exception: {str(e)}"


def extract_evaluation_metrics(tunnel_dir: str) -> Optional[Dict]:
    """
    Extract evaluation metrics from evaluation output markdown file.
    
    Returns:
        Dictionary with metrics (mIoU, OA, F1) or None if not found
    """
    eval_dir = Path(tunnel_dir) / 'evaluation'
    
    # Try to find metrics JSON first
    metrics_file = eval_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    # Try to find report JSON
    report_file = eval_dir / 'report.json'
    if report_file.exists():
        with open(report_file, 'r') as f:
            data = json.load(f)
            if 'metrics' in data:
                return data['metrics']
            return data
    
    # Parse from performance.md (markdown report)
    perf_file = eval_dir / 'performance.md'
    if perf_file.exists():
        import re
        with open(perf_file, 'r') as f:
            content = f.read()
        
        # Extract metrics from markdown table
        metrics = {}
        
        # Overall Accuracy
        oa_match = re.search(r'Overall Accuracy \(OA\)\s*\|\s*([\d.]+)', content)
        if oa_match:
            metrics['OA'] = float(oa_match.group(1))
        
        # F1 Score
        f1_match = re.search(r'F1 Score \(macro\)\s*\|\s*([\d.]+)', content)
        if f1_match:
            metrics['F1'] = float(f1_match.group(1))
        
        # Mean IoU
        miou_match = re.search(r'Mean IoU \(mIoU\)\s*\|\s*([\d.]+)', content)
        if miou_match:
            metrics['mIoU'] = float(miou_match.group(1))
        
        if metrics:
            return metrics
    
    return None


def run_pipeline(
    tunnel_ids: Optional[List[str]] = None,
    data_dir: str = 'data',
    skip_detection: bool = False,
    skip_sam: bool = False,
    skip_evaluation: bool = False
) -> Dict[str, Dict]:
    """
    Run the complete pipeline (detection → SAM → evaluation) for specified tunnels.
    
    Args:
        tunnel_ids: List of tunnel IDs to process (default: all in TUNNEL_ORDER)
        data_dir: Base data directory
        skip_detection: Skip detection stage (use existing detected.csv)
        skip_sam: Skip SAM stage (use existing final.csv)
        skip_evaluation: Skip evaluation stage
    
    Returns:
        Dictionary mapping tunnel_id to results dict
    """
    if tunnel_ids is None:
        tunnel_ids = TUNNEL_ORDER
    
    results = {}
    
    print("=" * 80)
    print("END-TO-END PIPELINE TEST")
    print("=" * 80)
    print(f"Tunnels: {', '.join(tunnel_ids)}")
    print(f"Data directory: {data_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for tunnel_id in tunnel_ids:
        if tunnel_id not in TUNNEL_AGENT_MAP:
            print(f"\n⚠️  Skipping {tunnel_id}: not in tunnel-agent mapping")
            continue
        
        agent_type = TUNNEL_AGENT_MAP[tunnel_id]
        tunnel_dir = Path(data_dir) / tunnel_id
        
        print(f"\n{'=' * 80}")
        print(f"TUNNEL: {tunnel_id} ({agent_type})")
        print(f"{'=' * 80}")
        
        result = {
            'tunnel_id': tunnel_id,
            'agent_type': agent_type,
            'stages': {},
            'metrics': None,
            'success': False
        }
        
        # Stage 1: Detection
        if not skip_detection:
            print(f"\n[1/3] Detection...")
            success, stdout, stderr = run_stage(agent_type, 'detection', tunnel_id, data_dir)
            result['stages']['detection'] = {
                'success': success,
                'error': stderr
            }
            if not success:
                print(f"  ❌ Detection failed: {stderr}")
                results[tunnel_id] = result
                continue
            print(f"  ✓ Detection complete")
        else:
            print(f"\n[1/3] Detection... (skipped)")
            result['stages']['detection'] = {'success': True, 'skipped': True}
        
        # Stage 2: SAM Segmentation
        if not skip_sam:
            print(f"\n[2/3] SAM Segmentation...")
            success, stdout, stderr = run_stage(agent_type, 'sam', tunnel_id, data_dir)
            result['stages']['sam'] = {
                'success': success,
                'error': stderr
            }
            if not success:
                print(f"  ❌ SAM failed: {stderr}")
                results[tunnel_id] = result
                continue
            print(f"  ✓ SAM complete")
        else:
            print(f"\n[2/3] SAM Segmentation... (skipped)")
            result['stages']['sam'] = {'success': True, 'skipped': True}
        
        # Stage 3: Evaluation
        if not skip_evaluation:
            print(f"\n[3/3] Evaluation...")
            success, stdout, stderr = run_stage(agent_type, 'evaluation', tunnel_id, data_dir)
            result['stages']['evaluation'] = {
                'success': success,
                'error': stderr
            }
            if not success:
                print(f"  ❌ Evaluation failed: {stderr}")
            else:
                print(f"  ✓ Evaluation complete")
                
                # Extract metrics
                metrics = extract_evaluation_metrics(str(tunnel_dir))
                result['metrics'] = metrics
        else:
            print(f"\n[3/3] Evaluation... (skipped)")
            result['stages']['evaluation'] = {'success': True, 'skipped': True}
        
        result['success'] = all(
            stage.get('success', False) or stage.get('skipped', False)
            for stage in result['stages'].values()
        )
        
        results[tunnel_id] = result
        
        if result['success']:
            print(f"\n✓ {tunnel_id} pipeline complete")
            if result['metrics']:
                print(f"  Metrics: mIoU={result['metrics'].get('mIoU', 'N/A'):.4f}, "
                      f"OA={result['metrics'].get('OA', 'N/A'):.4f}, "
                      f"F1={result['metrics'].get('F1', 'N/A'):.4f}")
        else:
            print(f"\n❌ {tunnel_id} pipeline failed")
    
    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Tunnel':<10} {'Agent':<20} {'Status':<10} {'mIoU':<8} {'OA':<8} {'F1':<8}")
    print("-" * 80)
    
    for tunnel_id in tunnel_ids:
        if tunnel_id not in results:
            continue
        
        r = results[tunnel_id]
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        metrics = r.get('metrics') or {}
        miou = f"{metrics.get('mIoU', 0):.4f}" if metrics and metrics.get('mIoU') is not None else "N/A"
        oa = f"{metrics.get('OA', 0):.4f}" if metrics and metrics.get('OA') is not None else "N/A"
        f1 = f"{metrics.get('F1', 0):.4f}" if metrics and metrics.get('F1') is not None else "N/A"
        
        print(f"{tunnel_id:<10} {r['agent_type']:<20} {status:<10} {miou:<8} {oa:<8} {f1:<8}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run end-to-end pipeline test on all tunnels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_pipeline.py                    # Run all tunnels
  python run_all_pipeline.py --tunnels 1-4 2-2 # Run specific tunnels
  python run_all_pipeline.py --skip-detection  # Use existing detected.csv
  python run_all_pipeline.py --skip-sam        # Use existing final.csv
        """
    )
    
    parser.add_argument(
        '--tunnels',
        nargs='+',
        default=None,
        help='Tunnel IDs to process (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Base data directory (default: data)'
    )
    parser.add_argument(
        '--skip-detection',
        action='store_true',
        help='Skip detection stage (use existing detected.csv)'
    )
    parser.add_argument(
        '--skip-sam',
        action='store_true',
        help='Skip SAM stage (use existing final.csv)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation stage'
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        tunnel_ids=args.tunnels,
        data_dir=args.data_dir,
        skip_detection=args.skip_detection,
        skip_sam=args.skip_sam,
        skip_evaluation=args.skip_evaluation
    )
