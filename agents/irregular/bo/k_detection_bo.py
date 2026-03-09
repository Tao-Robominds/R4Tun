"""
Bayesian Optimization for K detection only.
Uses composite objective: count_penalty (200 * n_far) + clipped_mean + missing_penalty.
Per-method search spaces; --method all runs all 7 methods in sequence.
"""

import os
import sys
import json
import glob
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real, Integer
from scipy.optimize import linear_sum_assignment

# Project root (p4tun/bo/ -> p4tun/ -> project root)
P4TUN_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = P4TUN_DIR.parent
sys.path.insert(0, str(P4TUN_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from importlib.util import spec_from_file_location, module_from_spec

# Load unified K detection
_spec = spec_from_file_location(
    "k_detection",
    P4TUN_DIR / "4-1-1_geo_k_detection.py",
)
_k_mod = module_from_spec(_spec)
_spec.loader.exec_module(_k_mod)

run_k_detection = _k_mod.run_k_detection
align_k_to_gt = _k_mod.align_k_to_gt
K_METHODS = _k_mod.K_METHODS


def _get_k_height_px(tunnel_id: str, data_dir: str) -> float:
    """K block height in pixels for regulator search bounds."""
    preproc = _k_mod.load_preprocessing_params(tunnel_id, data_dir)
    td = float(preproc.get("tunnel_diameter", 5.5))
    res = float(preproc.get("depth_map_resolution", 0.005))
    k_mm, _ = _k_mod.calculate_segment_heights(td)
    return k_mm / (res * 1000.0)


def _regulator_dims(tunnel_id: str, data_dir: str) -> List:
    """Four BO-tunable regulator dimensions."""
    kh = _get_k_height_px(tunnel_id, data_dir)
    return [
        Real(0.3 * kh, 1.5 * kh, name="reg_target_gap"),
        Real(0.2, 0.8, name="reg_gap_tolerance"),
        Real(0.0, 1.0, name="reg_blend_weight"),
        Real(0.5 * kh, 2.0 * kh, name="reg_max_det_line_dist"),
    ]


def _wrap_distance(x1: float, y1: float, x2: float, y2: float, img_height: int) -> float:
    dx = x1 - x2
    dy = abs(y1 - y2)
    dy = min(dy, img_height - dy)
    return float(np.sqrt(dx**2 + dy**2))


def k_composite_objective(
    detected_k: pd.DataFrame,
    gt_k: pd.DataFrame,
    img_height: int,
    close_threshold: float = 500.0,
    clip_dist: float = 1000.0,
    count_weight: float = 200.0,
) -> Tuple[float, Dict]:
    """
    Composite objective (minimize).
    Returns (score, info_dict with mean_k_distance, n_far, n_matched, etc.)
    """
    n_gt = len(gt_k)
    n_pred = len(detected_k)
    if n_gt == 0:
        return 0.0, {"mean_k_distance": 0.0, "n_far": 0, "n_matched": 0, "k_distances": []}

    gt_sorted = gt_k.sort_values("Ring").reset_index(drop=True)
    cost = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        gx, gy = float(gt_sorted.loc[i, "X"]), float(gt_sorted.loc[i, "Y"])
        for j in range(n_pred):
            dx = float(detected_k.iloc[j]["X"])
            dy = float(detected_k.iloc[j]["Y"])
            cost[i, j] = _wrap_distance(gx, gy, dx, dy, img_height)

    row_ind, col_ind = linear_sum_assignment(cost)
    dists = [float(cost[r, c]) for r, c in zip(row_ind, col_ind)]
    # Pad for missing GT matches (fewer det than gt)
    for _ in range(n_gt - len(dists)):
        dists.append(clip_dist)

    n_missing = max(0, n_gt - n_pred)
    missing_penalty = n_missing * clip_dist
    n_far = sum(1 for d in dists if d > close_threshold)
    count_penalty = count_weight * n_far
    clipped_dists = [min(d, clip_dist) for d in dists]
    clipped_mean = float(np.mean(clipped_dists)) if clipped_dists else clip_dist
    score = count_penalty + clipped_mean + missing_penalty
    mean_k_distance = float(np.mean(dists)) if dists else clip_dist

    return score, {
        "mean_k_distance": mean_k_distance,
        "n_far": n_far,
        "n_matched": len(dists),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "k_distances": dists,
        "clipped_mean": clipped_mean,
        "count_penalty": count_penalty,
        "missing_penalty": missing_penalty,
    }


# -----------------------------------------------------------------------------
# Search spaces per method
# -----------------------------------------------------------------------------
def get_k_search_space(method: str, tunnel_id: str, data_dir: str) -> Tuple[List, List[str], Dict]:
    """
    Returns (dimensions, param_names, base_params).
    base_params are merged with BO trial params (base_params keys not in param_names are fixed).
    """
    base_params = {}
    params_file = PROJECT_ROOT / "agents" / "irregular" / "2_detection" / "parameters" / tunnel_id / "parameters_detection.json"
    if params_file.exists():
        with open(params_file) as f:
            base_params = json.load(f)

    if method == "dbscan":
        dims = [
            Integer(80, 200, name="binary_threshold"),
            Integer(20, 120, name="hough_threshold"),
            Integer(20, 150, name="hough_min_length"),
            Integer(20, 150, name="hough_max_gap"),
            Real(4.0, 8.0, name="angle_pos_min"),
            Real(8.0, 14.0, name="angle_pos_max"),
            Real(-14.0, -8.0, name="angle_neg_min"),
            Real(-8.0, -4.0, name="angle_neg_max"),
            Real(0.03, 0.15, name="eps"),
            Real(1.0, 2.5, name="complex_subdivision_threshold"),
            Integer(2, 5, name="complex_max_subdivisions"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    if method == "groove_pair":
        dims = [
            Integer(80, 200, name="binary_threshold"),
            Integer(20, 120, name="hough_threshold"),
            Integer(20, 150, name="hough_min_length"),
            Integer(20, 150, name="hough_max_gap"),
            Real(4.0, 8.0, name="angle_pos_min"),
            Real(8.0, 14.0, name="angle_pos_max"),
            Real(-14.0, -8.0, name="angle_neg_min"),
            Real(-8.0, -4.0, name="angle_neg_max"),
            Real(200.0, 400.0, name="k_expected_height_px"),
            Real(80.0, 250.0, name="k_gap_tolerance_px"),
            Integer(4, 12, name="k_candidates_per_ring"),
            Real(30.0, 100.0, name="groove_snap_px"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    if method == "banded":
        dims = [
            Integer(80, 200, name="binary_threshold"),
            Integer(20, 120, name="hough_threshold"),
            Integer(20, 150, name="hough_min_length"),
            Integer(20, 150, name="hough_max_gap"),
            Real(4.0, 8.0, name="angle_pos_min"),
            Real(8.0, 14.0, name="angle_pos_max"),
            Real(-14.0, -8.0, name="angle_neg_min"),
            Real(-8.0, -4.0, name="angle_neg_max"),
            Real(0.4, 0.9, name="band_margin_factor"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    if method == "edge_projection":
        dims = [
            Integer(30, 80, name="ep_canny_low"),
            Integer(100, 200, name="ep_canny_high"),
            Integer(1, 4, name="ep_dilation_size"),
            Real(5.0, 30.0, name="ep_smooth_sigma"),
            Real(0.5, 1.5, name="ep_band_width_factor"),
            Integer(40, 120, name="ep_peak_distance"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    if method == "gradient_direction":
        dims = [
            Integer(3, 9, name="gd_sobel_ksize"),
            Real(4.0, 10.0, name="gd_pos_angle_center"),
            Real(-10.0, -4.0, name="gd_neg_angle_center"),
            Real(2.0, 8.0, name="gd_angle_tolerance"),
            Real(200.0, 400.0, name="gd_k_height_px"),
            Real(20.0, 60.0, name="gd_smooth_sigma"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    if method == "local_hough":
        dims = [
            Integer(15, 50, name="lh_hough_threshold"),
            Integer(20, 80, name="lh_min_length"),
            Integer(40, 120, name="lh_max_gap"),
            Real(4.0, 10.0, name="lh_angle_pos_range"),
            Real(-10.0, -4.0, name="lh_angle_neg_range"),
            Integer(30, 80, name="lh_canny_low"),
            Integer(100, 180, name="lh_canny_high"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    if method == "ensemble":
        dims = [
            Real(0.1, 2.0, name="w_dbscan"),
            Real(0.1, 2.0, name="w_groove_pair"),
            Real(0.1, 2.0, name="w_banded"),
            Real(0.1, 2.0, name="w_edge_projection"),
            Real(0.1, 2.0, name="w_gradient_direction"),
            Real(0.1, 2.0, name="w_local_hough"),
        ]
        dims = dims + _regulator_dims(tunnel_id, data_dir)
        names = [d.name for d in dims]
        return dims, names, base_params

    raise ValueError(f"Unknown method: {method}")


def params_list_to_dict(params: List, param_names: List[str], base_params: Dict) -> Dict:
    """Merge trial params with base_params (base overwrites for keys not in param_names)."""
    out = dict(base_params)
    for name, val in zip(param_names, params):
        if isinstance(val, (np.floating, np.integer)):
            val = float(val) if isinstance(val, np.floating) else int(val)
        out[name] = val
    # Ensure angle_neg derived from angle_pos for detection module if needed
    if "angle_pos_min" in out and "angle_neg_max" not in out:
        out["angle_neg_max"] = -out["angle_pos_min"]
        out["angle_neg_min"] = -out["angle_pos_max"]
    return out


# -----------------------------------------------------------------------------
# Objective and BO run
# -----------------------------------------------------------------------------
class KDetectionObjective:
    def __init__(
        self,
        tunnel_id: str,
        method: str,
        data_dir: str = "data",
        verbose: bool = True,
        eval_offset: int = 0,
    ):
        self.tunnel_id = tunnel_id
        self.method = method
        self.data_dir = data_dir
        self.verbose = verbose
        self.eval_offset = eval_offset
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        depth_path = os.path.join(self.tunnel_dir, "depth_map_outlier.npy")
        ring_path = os.path.join(self.tunnel_dir, "ring_count.txt")
        gt_path = os.path.join(self.tunnel_dir, "all_segments_gt.csv")
        if not os.path.exists(depth_path) or not os.path.exists(ring_path):
            raise FileNotFoundError(f"Missing depth_map_outlier.npy or ring_count.txt in {self.tunnel_dir}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing all_segments_gt.csv in {self.tunnel_dir}")

        self.depth_map = np.load(depth_path)
        self.img_height = self.depth_map.shape[0]
        self.ring_count = int(open(ring_path).read())
        gt = pd.read_csv(gt_path)
        self.gt_k = gt[gt["Block"] == "K"][["Ring", "X", "Y"]].copy()

        self.dimensions, self.param_names, self.base_params = get_k_search_space(method, tunnel_id, data_dir)
        self.eval_count = 0
        self.best_score = np.inf
        self.best_params = None
        self.history = []
        self.logs_dir = Path(__file__).resolve().parent / "k_logs" / method
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def global_eval_index(self) -> int:
        return self.eval_offset + self.eval_count

    def __call__(self, params: List) -> float:
        self.eval_count += 1
        start = time.time()
        try:
            param_dict = params_list_to_dict(params, self.param_names, self.base_params)
            with open(os.devnull, "w") as devnull:
                import io
                from contextlib import redirect_stdout, redirect_stderr
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    k_df = run_k_detection(
                        self.depth_map,
                        self.ring_count,
                        self.method,
                        param_dict,
                        tunnel_id=self.tunnel_id,
                        base_dir=self.data_dir,
                        verbose=False,
                    )
            score, info = k_composite_objective(
                k_df, self.gt_k, self.img_height,
                close_threshold=500.0, clip_dist=1000.0, count_weight=200.0,
            )
            elapsed = time.time() - start
            if score < self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.global_eval_index}] New best: composite={score:.1f} "
                          f"(mean_K_dist={info['mean_k_distance']:.1f}px, n_far={info['n_far']})")
            self.history.append({
                "eval": self.global_eval_index,
                "score": score,
                "mean_k_distance": info["mean_k_distance"],
                "n_far": info["n_far"],
            })
            self._log_trial(param_dict, score, info, len(k_df), elapsed)
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.global_eval_index}] composite={score:.1f}, mean_K_dist={info['mean_k_distance']:.1f}px")
            return score
        except Exception as e:
            elapsed = time.time() - start
            if self.verbose:
                print(f"  [Eval {self.global_eval_index}] Error: {e}")
            self._log_trial(
                params_list_to_dict(params, self.param_names, self.base_params),
                None, None, 0, elapsed, error=str(e),
            )
            return 1e9

    def _log_trial(
        self,
        params: Dict,
        score: Optional[float],
        info: Optional[Dict],
        n_det: int,
        runtime: float,
        error: Optional[str] = None,
    ):
        trial_id = f"k_detect_{self.tunnel_id}_{self.global_eval_index:03d}"
        log_path = self.logs_dir / f"{trial_id}.json"
        log_data = {
            "trial_id": trial_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tunnel_id": self.tunnel_id,
            "method": self.method,
            "params": params,
        }
        if error:
            log_data["error"] = error
            log_data["objective_value"] = 1e9
        else:
            log_data["objective_value"] = score
            log_data["mean_k_distance"] = info.get("mean_k_distance")
            log_data["n_far"] = info.get("n_far")
            log_data["n_matched"] = info.get("n_matched")
        log_data["runtime_sec"] = runtime
        log_data["n_detected"] = n_det
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)


def run_k_detection_bo(
    tunnel_id: str,
    method: str,
    data_dir: str = "data",
    n_calls: int = 200,
    n_initial_points: int = 20,
    verbose: bool = True,
    eval_offset: int = 0,
) -> Dict:
    """Run BO for one K detection method. Returns best score and params."""
    objective = KDetectionObjective(
        tunnel_id=tunnel_id,
        method=method,
        data_dir=data_dir,
        verbose=verbose,
        eval_offset=eval_offset,
    )
    print(f"\nK Detection BO: tunnel={tunnel_id}, method={method}, dims={len(objective.param_names)}")
    print(f"  n_calls={n_calls}, n_initial={n_initial_points}")

    result = forest_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=42,
        verbose=False,
    )

    best_score = float(result.fun)
    best_x = result.x
    best_params = params_list_to_dict(best_x, objective.param_names, objective.base_params)
    print(f"\nBest composite score: {best_score:.2f}")
    print(f"Best mean K distance: {objective.history[-1].get('mean_k_distance', 'N/A')} px (from last eval)")

    # Save best K positions to data/tunnel_id/detected_k_{method}.csv (aligned to GT)
    k_df = run_k_detection(
        objective.depth_map, objective.ring_count, method, best_params,
        tunnel_id=tunnel_id, base_dir=data_dir, verbose=False,
    )
    aligned, dists = align_k_to_gt(k_df, objective.gt_k, objective.img_height)
    out_csv = os.path.join(objective.tunnel_dir, f"detected_k_{method}.csv")
    aligned.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return {
        "tunnel_id": tunnel_id,
        "method": method,
        "best_score": best_score,
        "best_mean_k_distance": float(np.mean(dists)) if dists else 9999.0,
        "best_params": best_params,
        "n_evaluations": objective.eval_count,
        "history": objective.history,
    }


def find_max_trial_index(logs_dir: Path, tunnel_id: str, method: str) -> int:
    pattern = str(logs_dir / f"k_detect_{tunnel_id}_*.json")
    files = glob.glob(pattern)
    max_idx = 0
    for f in files:
        try:
            idx = int(Path(f).stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return max_idx


def main():
    parser = argparse.ArgumentParser(description="K detection BO (composite objective)")
    parser.add_argument("tunnel_id", default="4-1", nargs="?", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--method", default="groove_pair",
                        choices=K_METHODS + ["all"],
                        help="K detection method or 'all'")
    parser.add_argument("--n-calls", type=int, default=200)
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--quiet", action="store_true", help="Less verbose")
    args = parser.parse_args()

    if args.method == "all":
        order = ["edge_projection", "gradient_direction", "local_hough", "banded", "dbscan", "groove_pair", "ensemble"]
        results = {}
        eval_offset = 0
        for method in order:
            logs_dir = Path(__file__).resolve().parent / "k_logs" / method
            eval_offset = find_max_trial_index(logs_dir, args.tunnel_id, method)
            r = run_k_detection_bo(
                args.tunnel_id,
                method,
                data_dir=args.data_dir,
                n_calls=args.n_calls,
                n_initial_points=args.n_initial,
                verbose=not args.quiet,
                eval_offset=eval_offset,
            )
            results[method] = r
            eval_offset += args.n_calls  # next method starts after this one's evals

        print("\n" + "=" * 70)
        print("SUMMARY (all methods)")
        print("=" * 70)
        print(f"{'Method':<22} | {'Best composite':>14} | {'Mean K dist (px)':>16} | Dims | Trials")
        print("-" * 70)
        for method in order:
            r = results.get(method, {})
            score = r.get("best_score", 9999)
            mean_dist = r.get("best_mean_k_distance", 9999)
            dims = len(get_k_search_space(method, args.tunnel_id, args.data_dir)[0])
            print(f"{method:<22} | {score:>14.1f} | {mean_dist:>16.1f} | {dims:>4} | {args.n_calls}")
        return

    logs_dir = Path(__file__).resolve().parent / "k_logs" / args.method
    eval_offset = find_max_trial_index(logs_dir, args.tunnel_id, args.method)
    run_k_detection_bo(
        args.tunnel_id,
        args.method,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        verbose=not args.quiet,
        eval_offset=eval_offset,
    )


if __name__ == "__main__":
    main()
