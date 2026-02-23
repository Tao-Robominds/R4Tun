"""
Proxy BO Experiment: Validate that intrinsic-metrics predictor is safe for tuning.

Compares:
  1. Baseline: default params, true mIoU
  2. Proxy BO: optimize for predicted mIoU (no GT), then evaluate best with true mIoU
  3. True BO: optimize for true mIoU (oracle)

If proxy-BO best achieves true mIoU close to true-BO best, the predictor is validated.

Usage:
  python -m p4tun.bo.proxy_bo_experiment --tunnel 2-2 --n-calls 20
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _get_imports():
    from p4tun.bo.search_space import (
        get_search_space,
        params_to_detection_dict,
        params_to_sam_dict,
        save_parameters,
        load_default_parameters,
    )
    from p4tun.bo.objective import PipelineObjective
    return {
        'get_search_space': get_search_space,
        'params_to_detection_dict': params_to_detection_dict,
        'params_to_sam_dict': params_to_sam_dict,
        'save_parameters': save_parameters,
        'load_default_parameters': load_default_parameters,
        'PipelineObjective': PipelineObjective,
    }


class ProxyObjective:
    """
    Objective that uses predicted mIoU (from intrinsic metrics) instead of true mIoU.
    Runs detection + SAM, computes intrinsic metrics, predicts mIoU.
    """

    def __init__(
        self,
        tunnel_id: str,
        stage: str = 'sam',
        data_dir: str = 'data',
        model_path: Path = None,
        verbose: bool = True,
    ):
        self.tunnel_id = tunnel_id
        self.stage = stage
        self.data_dir = data_dir
        self.verbose = verbose

        from p4tun.bo.predictor import load_model, predict
        self.predict = predict
        self.bundle = load_model(model_path)

        funcs = _get_imports()
        self.dimensions, self.param_names = funcs['get_search_space'](stage)
        self._update_params = lambda p: self._do_update_params(p, funcs)
        self._run_detection = lambda: self._run_script('4-1_detection')
        self._run_sam = lambda: self._run_script('4-2_sam')

        self.script_dir = Path(__file__).parent.parent
        self.project_root = self.script_dir.parent
        self.segment_path = str(self.script_dir / 'segment-anything')

        self.eval_count = 0
        self.best_predicted = -np.inf
        self.best_params = None
        self.history = []

    def _do_update_params(self, params, funcs):
        param_dict = dict(zip(self.param_names, params))
        params_dir = self.script_dir / 'parameters' / self.tunnel_id
        params_dir.mkdir(parents=True, exist_ok=True)

        if self.stage in ['detection', 'combined']:
            d = funcs['params_to_detection_dict'](params, self.param_names)
            existing = funcs['load_default_parameters'](self.tunnel_id, 'detection')
            existing.update(d)
            with open(params_dir / 'parameters_detection.json', 'w') as f:
                json.dump(existing, f, indent=4)

        if self.stage in ['sam', 'combined']:
            d = funcs['params_to_sam_dict'](params, self.param_names)
            existing = funcs['load_default_parameters'](self.tunnel_id, 'sam')
            existing.update(d)
            with open(params_dir / 'parameters_sam.json', 'w') as f:
                json.dump(existing, f, indent=4)

    def _run_script(self, script_name: str):
        import subprocess
        script_path = self.script_dir / f'{script_name}.py'
        cmd = [
            sys.executable, str(script_path),
            self.tunnel_id, '--data-dir', self.data_dir
        ]
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{self.segment_path}:{env.get('PYTHONPATH', '')}"
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(self.project_root), env=env
        )
        if r.returncode != 0:
            raise RuntimeError(f"{script_name} failed: {r.stderr}")

    def __call__(self, params) -> float:
        self.eval_count += 1
        try:
            self._update_params(params)
            if self.stage in ['detection', 'combined']:
                self._run_detection()
            if self.stage in ['sam', 'combined']:
                self._run_sam()

            from bo4tun.intrinsic_metrics import (
                compute_detection_metrics,
                compute_sam_metrics,
            )
            tunnel_dir = Path(self.data_dir) / self.tunnel_id
            det_csv = tunnel_dir / 'detected.csv'
            final_csv = tunnel_dir / 'final.csv'

            metrics = {}
            if det_csv.exists():
                dm = compute_detection_metrics(
                    self.tunnel_id, str(det_csv), data_dir=self.data_dir
                )
                metrics.update({f'det_{k}': v for k, v in dm.items()})
            if final_csv.exists():
                sm = compute_sam_metrics(
                    self.tunnel_id, str(final_csv), str(det_csv), data_dir=self.data_dir
                )
                metrics.update({f'sam_{k}': v for k, v in sm.items()})

            pred_miou = self.predict(self.tunnel_id, metrics)

            if pred_miou > self.best_predicted:
                self.best_predicted = pred_miou
                self.best_params = dict(zip(self.param_names, params))
                if self.verbose:
                    print(f"  [Proxy Eval {self.eval_count}] New best predicted mIoU: {pred_miou:.4f}")

            self.history.append({
                'eval': self.eval_count,
                'params': dict(zip(self.param_names, params)),
                'predicted_mIoU': pred_miou,
                'intrinsic_metrics': metrics,
            })
            if self.verbose and self.eval_count % 5 == 0:
                print(f"  [Proxy Eval {self.eval_count}] Predicted mIoU: {pred_miou:.4f}")

            return -pred_miou  # minimize negative = maximize
        except Exception as e:
            if self.verbose:
                print(f"  [Proxy Eval {self.eval_count}] Error: {e}")
            return 1.0  # bad (we minimize; high return = low predicted mIoU)


def evaluate_true_miou(tunnel_id: str, stage: str, data_dir: str = 'data') -> float:
    """Run evaluation script and return mIoU."""
    import subprocess
    script_dir = Path(__file__).parent.parent
    eval_script = script_dir / 'evaluation.py'
    cmd = [
        sys.executable, str(eval_script), tunnel_id, '--data-dir', data_dir
    ]
    env = os.environ.copy()
    segment_path = str(script_dir / 'segment-anything')
    env['PYTHONPATH'] = f"{segment_path}:{env.get('PYTHONPATH', '')}"
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
        cwd=str(PROJECT_ROOT), env=env
    )
    # Parse mIoU from stdout
    for line in (r.stdout or '').splitlines():
        if 'mIoU' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'mIoU' and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        pass
    # Fallback: read performance.md
    perf = Path(data_dir) / tunnel_id / 'evaluation' / 'performance.md'
    if perf.exists():
        for line in perf.read_text().splitlines():
            if 'Mean IoU' in line or 'mIoU' in line:
                try:
                    return float(line.split('|')[-1].strip())
                except (ValueError, IndexError):
                    pass
    return 0.0


def backup_and_restore_params(tunnel_id: str, stage: str, restore: bool = False):
    """Backup or restore parameter files."""
    params_dir = PROJECT_ROOT / 'p4tun' / 'parameters' / tunnel_id
    backup_dir = params_dir / '_proxy_bo_backup'
    if not restore:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        for stage_name in ['detection', 'sam']:
            f = params_dir / f'parameters_{stage_name}.json'
            if f.exists():
                shutil.copy(f, backup_dir / f.name)
    else:
        if backup_dir.exists():
            for f in backup_dir.glob('*.json'):
                shutil.copy(f, params_dir / f.name)


def run_experiment(
    tunnel_id: str = '2-2',
    stage: str = 'sam',
    n_calls: int = 20,
    n_initial: int = 5,
    data_dir: str = 'data',
) -> dict:
    """Run full proxy BO experiment."""
    funcs = _get_imports()
    dims, names = funcs['get_search_space'](stage)

    print("\n" + "=" * 70)
    print("PROXY BO EXPERIMENT")
    print("=" * 70)
    print(f"Tunnel: {tunnel_id}, Stage: {stage}, N calls: {n_calls}")
    print("=" * 70)

    # 1. Baseline: default params (existing param files)
    backup_and_restore_params(tunnel_id, stage, restore=False)
    print("\n[1/4] Baseline: running pipeline with current (default) params...")
    from p4tun.bo.objective import PipelineObjective
    base_obj = PipelineObjective(tunnel_id=tunnel_id, stage=stage, data_dir=data_dir, verbose=False)
    try:
        base_obj._run_detection()
        base_obj._run_sam()
        baseline_miou = base_obj._evaluate().get('mIoU', 0.0)
        print(f"      Baseline true mIoU: {baseline_miou:.4f}")
    except Exception as e:
        print(f"      Baseline failed: {e}")
        baseline_miou = 0.0

    # 2. Proxy BO
    print("\n[2/4] Proxy BO: optimizing for predicted mIoU...")
    proxy_obj = ProxyObjective(
        tunnel_id=tunnel_id, stage=stage, data_dir=data_dir, verbose=True
    )
    proxy_result = gp_minimize(
        proxy_obj,
        dims,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=False,
        callback=[DeltaYStopper(delta=0.001, n_best=10)],
    )
    best_proxy_params = dict(zip(names, proxy_result.x))
    best_predicted = -proxy_result.fun

    # Apply proxy best params and evaluate with true mIoU
    print("\n[3/4] Evaluating proxy-BO best params with true mIoU...")
    params_list = [best_proxy_params[n] for n in names]
    base_obj2 = PipelineObjective(tunnel_id=tunnel_id, stage=stage, data_dir=data_dir, verbose=False)
    try:
        base_obj2._update_parameters(params_list)
        base_obj2._run_detection()
        base_obj2._run_sam()
        proxy_best_true_miou = base_obj2._evaluate().get('mIoU', 0.0)
    except Exception as e:
        print(f"      Proxy best evaluation failed: {e}")
        proxy_best_true_miou = 0.0
    print(f"      Proxy-BO best → true mIoU: {proxy_best_true_miou:.4f}")

    # 3. True mIoU BO
    print("\n[4/4] True BO: optimizing for true mIoU (oracle)...")
    backup_and_restore_params(tunnel_id, stage, restore=True)  # restore baseline
    true_obj = PipelineObjective(
        tunnel_id=tunnel_id, stage=stage, data_dir=data_dir, verbose=True
    )
    try:
        true_result = gp_minimize(
            true_obj,
            dims,
            n_calls=n_calls,
            n_initial_points=n_initial,
            random_state=42,
            verbose=False,
            callback=[DeltaYStopper(delta=0.001, n_best=10)],
        )
        true_best_miou = -true_result.fun
    except Exception as e:
        print(f"      True BO failed: {e}")
        true_best_miou = baseline_miou
    print(f"      True-BO best mIoU: {true_best_miou:.4f}")

    # Restore baseline
    backup_and_restore_params(tunnel_id, stage, restore=True)

    # Report
    results = {
        'tunnel_id': tunnel_id,
        'stage': stage,
        'n_calls': n_calls,
        'baseline_miou': baseline_miou,
        'proxy_best_predicted': best_predicted,
        'proxy_best_true_miou': proxy_best_true_miou,
        'true_bo_best_miou': true_best_miou,
        'proxy_vs_baseline': proxy_best_true_miou - baseline_miou,
        'proxy_vs_true_bo': proxy_best_true_miou - true_best_miou,
        'timestamp': datetime.now().isoformat(),
    }

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline (default params):     true mIoU = {baseline_miou:.4f}")
    print(f"  Proxy BO best (predicted opt): true mIoU = {proxy_best_true_miou:.4f}")
    print(f"  True BO best (oracle):         true mIoU = {true_best_miou:.4f}")
    print(f"  Proxy vs baseline:             {results['proxy_vs_baseline']:+.4f}")
    print(f"  Proxy vs true BO:              {results['proxy_vs_true_bo']:+.4f}")
    if proxy_best_true_miou >= baseline_miou and proxy_best_true_miou >= 0.9 * true_best_miou:
        print("\n  → Predictor is SAFE for tuning: proxy optimization improves over baseline")
        print("    and stays within ~90% of oracle.")
    else:
        print("\n  → Interpret with caution: proxy may need more training data or tuning.")
    print("=" * 70)

    # Save results
    out_dir = PROJECT_ROOT / 'p4tun' / 'bo' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'proxy_bo_experiment_{tunnel_id}_{stage}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


ALL_TUNNELS = ['1-4', '2-2', '3-1', '4-1', '5-1']


def run_experiment_all(
    stage: str = 'detection',
    n_calls: int = 10,
    n_initial: int = 3,
    data_dir: str = 'data',
    subprocess_per_tunnel: bool = True,
) -> dict:
    """Run experiment for all tunnels and return aggregated results.
    Uses subprocess per tunnel to avoid GPU memory accumulation."""
    import subprocess
    all_results = []
    for tunnel_id in ALL_TUNNELS:
        print("\n" + "#" * 70)
        print(f"# TUNNEL {tunnel_id}")
        print("#" * 70)
        if subprocess_per_tunnel:
            # Fresh process per tunnel to free GPU memory
            cmd = [
                sys.executable, '-m', 'p4tun.bo.proxy_bo_experiment',
                '--tunnel', tunnel_id, '--stage', stage,
                '--n-calls', str(n_calls), '--n-initial', str(n_initial),
                '--data-dir', data_dir,
            ]
            try:
                out = subprocess.run(cmd, timeout=3600, cwd=str(PROJECT_ROOT))
                if out.returncode != 0:
                    print(f"Subprocess for {tunnel_id} exited with {out.returncode}")
                # Parse result from last saved JSON
                import glob
                jsons = sorted(glob.glob(str(PROJECT_ROOT / 'p4tun' / 'bo' / 'results' / f'proxy_bo_experiment_{tunnel_id}_{stage}_*.json')))
                if jsons:
                    with open(jsons[-1]) as f:
                        r = json.load(f)
                    all_results.append(r)
                else:
                    all_results.append({'tunnel_id': tunnel_id, 'stage': stage, 'baseline_miou': 0.0,
                        'proxy_best_true_miou': 0.0, 'true_bo_best_miou': 0.0, 'proxy_vs_baseline': 0.0,
                        'proxy_vs_true_bo': 0.0, 'error': 'No results file'})
            except subprocess.TimeoutExpired:
                all_results.append({'tunnel_id': tunnel_id, 'stage': stage, 'error': 'Timeout'})
            except Exception as e:
                all_results.append({'tunnel_id': tunnel_id, 'stage': stage, 'error': str(e)})
        else:
            try:
                r = run_experiment(tunnel_id=tunnel_id, stage=stage, n_calls=n_calls,
                    n_initial=n_initial, data_dir=data_dir)
                all_results.append(r)
            except Exception as e:
                print(f"Tunnel {tunnel_id} failed: {e}")
                all_results.append({'tunnel_id': tunnel_id, 'stage': stage, 'baseline_miou': 0.0,
                    'proxy_best_true_miou': 0.0, 'true_bo_best_miou': 0.0, 'proxy_vs_baseline': 0.0,
                    'proxy_vs_true_bo': 0.0, 'error': str(e)})

    # Aggregate (exclude runs that failed entirely)
    valid = [r for r in all_results if not r.get('error')]
    n = len(valid)
    if n == 0:
        print("\nNo successful tunnel runs.")
        return {'all_results': all_results, 'averages': {}}

    agg = {
        'baseline_miou': sum(r['baseline_miou'] for r in valid) / n,
        'proxy_best_true_miou': sum(r['proxy_best_true_miou'] for r in valid) / n,
        'true_bo_best_miou': sum(r['true_bo_best_miou'] for r in valid) / n,
        'proxy_vs_baseline': sum(r['proxy_vs_baseline'] for r in valid) / n,
        'proxy_vs_true_bo': sum(r['proxy_vs_true_bo'] for r in valid) / n,
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("INDIVIDUAL RESULTS (all tunnels)")
    print("=" * 80)
    print(f"{'Tunnel':<8} {'Baseline':>10} {'Proxy Best':>10} {'True BO':>10} {'Δ vs Base':>10} {'Δ vs Oracle':>10}")
    print("-" * 80)
    for r in all_results:
        tid = r['tunnel_id']
        b = r.get('baseline_miou', 0)
        p = r.get('proxy_best_true_miou', 0)
        t = r.get('true_bo_best_miou', 0)
        db = r.get('proxy_vs_baseline', 0)
        dt = r.get('proxy_vs_true_bo', 0)
        err = " (FAILED)" if r.get('error') else ""
        print(f"{tid:<8} {b:>10.4f} {p:>10.4f} {t:>10.4f} {db:>+10.4f} {dt:>+10.4f}{err}")
    print("-" * 80)
    print(f"{'AVERAGE':<8} {agg['baseline_miou']:>10.4f} {agg['proxy_best_true_miou']:>10.4f} "
          f"{agg['true_bo_best_miou']:>10.4f} {agg['proxy_vs_baseline']:>+10.4f} {agg['proxy_vs_true_bo']:>+10.4f}")
    print("=" * 80)

    # Save aggregated
    out_dir = PROJECT_ROOT / 'p4tun' / 'bo' / 'results'
    out_path = out_dir / f'proxy_bo_experiment_ALL_{stage}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump({'all_results': all_results, 'averages': agg, 'n_tunnels': n}, f, indent=2)
    print(f"\nAggregated results saved to {out_path}")

    return {'all_results': all_results, 'averages': agg}


def main():
    parser = argparse.ArgumentParser(description='Proxy BO experiment')
    parser.add_argument('--tunnel', default=None, help='Tunnel ID (omit with --all)')
    parser.add_argument('--all', action='store_true', help='Run for all tunnels (1-4, 2-2, 3-1, 4-1, 5-1)')
    parser.add_argument('--stage', default='detection', choices=['detection', 'sam', 'combined'],
                        help='Detection is faster (~1s/eval); SAM ~2min/eval')
    parser.add_argument('--n-calls', type=int, default=20,
                        help='Total BO evaluations (try 5-10 for quick test)')
    parser.add_argument('--n-initial', type=int, default=5)
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    if args.all:
        run_experiment_all(
            stage=args.stage,
            n_calls=args.n_calls,
            n_initial=args.n_initial,
            data_dir=args.data_dir,
        )
    else:
        run_experiment(
            tunnel_id=args.tunnel or '1-4',
            stage=args.stage,
            n_calls=args.n_calls,
            n_initial=args.n_initial,
            data_dir=args.data_dir,
        )


if __name__ == '__main__':
    main()
