"""
Bayesian Optimization for A/B block position detection (complex_staggered).

Uses groove-first hybrid: 6-distance structural model + stagger detection + regulators.
Objective: Hungarian-matched mean centroid distance (49 segments: 7 K + 42 A/B).
Evaluate on 4-1; code supports any tunnel with all_segments_gt.csv.
"""

import os
import sys
import json
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

P4TUN_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = P4TUN_DIR.parent
sys.path.insert(0, str(P4TUN_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from importlib.util import spec_from_file_location, module_from_spec

_spec_k = spec_from_file_location("k_detection", P4TUN_DIR / "4-1-1_geo_k_detection.py")
_k_mod = module_from_spec(_spec_k)
_spec_k.loader.exec_module(_k_mod)

_spec_ab = spec_from_file_location("geo_ab_detection", P4TUN_DIR / "geo_ab_detection.py")
_ab_mod = module_from_spec(_spec_ab)
_spec_ab.loader.exec_module(_ab_mod)

run_k_detection = _k_mod.run_k_detection
run_ab_detection = _ab_mod.run_ab_detection


def _wrap_distance(x1: float, y1: float, x2: float, y2: float, circ: int) -> float:
    dx = x1 - x2
    dy = abs(y1 - y2)
    dy = min(dy, circ - dy)
    return float(np.sqrt(dx**2 + dy**2))


def ab_composite_objective(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    circ: int,
    clip_dist: float = 800.0,
) -> Tuple[float, Dict]:
    """
    Hungarian match 49 predicted vs 49 GT centroids. Minimize mean distance.
    pred_df and gt_df must have Ring, Block, X, Y. Sort by (Ring, Block) for consistent order.
    """
    block_order = ["K", "B1", "B2", "A1", "A2", "A3", "A4"]
    gt_sorted = gt_df.sort_values(["Ring", "Block"]).reset_index(drop=True)
    gt_sorted["_order"] = gt_sorted["Ring"].astype(str) + "_" + gt_sorted["Block"]
    pred_sorted = pred_df.sort_values(["Ring", "Block"]).reset_index(drop=True)
    pred_sorted["_order"] = pred_sorted["Ring"].astype(str) + "_" + pred_sorted["Block"]

    n_gt = len(gt_sorted)
    n_pred = len(pred_sorted)
    if n_gt == 0:
        return 0.0, {"mean_distance": 0.0, "n_matched": 0}

    cost = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        gx = float(gt_sorted.loc[i, "X"])
        gy = float(gt_sorted.loc[i, "Y"])
        for j in range(n_pred):
            px = float(pred_sorted.loc[j, "X"])
            py = float(pred_sorted.loc[j, "Y"])
            cost[i, j] = _wrap_distance(gx, gy, px, py, circ)

    row_ind, col_ind = linear_sum_assignment(cost)
    dists = [float(cost[r, c]) for r, c in zip(row_ind, col_ind)]
    if n_gt > n_pred:
        dists.extend([clip_dist] * (n_gt - n_pred))
    mean_dist = float(np.mean(dists))
    clipped_mean = float(np.mean([min(d, clip_dist) for d in dists]))
    n_missing = max(0, n_gt - n_pred)
    score = clipped_mean + n_missing * clip_dist

    return score, {
        "mean_distance": mean_dist,
        "n_matched": len(dists),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "dists": dists,
    }


def get_ab_search_space(tunnel_id: str, data_dir: str) -> Tuple[List, List[str], Dict]:
    """~14 dimensions: d1..d6, merge_dist, groove_blend, groove_radius, dx_global, edge_threshold, edge_scale, b_size_ratio."""
    dimensions = [
        Real(0.05, 0.15, name="d1"),
        Real(0.08, 0.18, name="d2"),
        Real(0.15, 0.35, name="d3"),
        Real(0.20, 0.40, name="d4"),
        Real(0.30, 0.45, name="d5"),
        Real(0.35, 0.50, name="d6"),
        Real(20.0, 150.0, name="merge_dist"),
        Real(0.0, 1.0, name="groove_blend"),
        Real(30.0, 200.0, name="groove_search_radius"),
        Real(0.0, 50.0, name="dx_global"),
        Real(200.0, 600.0, name="edge_threshold"),
        Real(0.5, 1.0, name="edge_scale"),
        Real(0.3, 0.8, name="b_size_ratio"),
    ]
    param_names = [d.name for d in dimensions]
    base_params = {}
    return dimensions, param_names, base_params


def params_to_ab_kwargs(params: List, param_names: List[str]) -> Dict:
    """Convert BO vector to run_ab_detection kwargs. Enforce d1<=d2<=...<=d6 by sorting."""
    d = dict(zip(param_names, params))
    d_fracs = [float(d["d1"]), float(d["d2"]), float(d["d3"]), float(d["d4"]), float(d["d5"]), float(d["d6"])]
    d_fracs.sort()
    return {
        "merge_dist": float(d["merge_dist"]),
        "d_fracs": d_fracs,
        "groove_blend": float(d["groove_blend"]),
        "groove_search_radius": float(d["groove_search_radius"]),
        "dx_global": float(d["dx_global"]),
        "edge_threshold": float(d["edge_threshold"]),
        "edge_scale": float(d["edge_scale"]),
        "b_size_ratio": float(d["b_size_ratio"]),
    }


class ABDetectionObjective:
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = "data",
        verbose: bool = True,
        eval_offset: int = 0,
        k_positions: Optional[pd.DataFrame] = None,
    ):
        self.tunnel_id = tunnel_id
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
        self.circ = int(self.depth_map.shape[0])
        self.ring_count = int(open(ring_path).read())
        gt = pd.read_csv(gt_path)
        if "segment_name" in gt.columns and "Block" not in gt.columns:
            gt = gt.rename(columns={"segment_name": "Block"})
        if "ring" in gt.columns and "Ring" not in gt.columns:
            gt = gt.rename(columns={"ring": "Ring"})
        self.gt_df = gt[["Ring", "Block", "X", "Y"]].copy()

        if k_positions is not None:
            self.k_positions = k_positions
        else:
            for name in ["detected_k_dbscan.csv", "detected_k_groove_pair.csv", "detected_k_banded.csv"]:
                k_path = os.path.join(self.tunnel_dir, name)
                if os.path.exists(k_path):
                    self.k_positions = pd.read_csv(k_path)
                    if "Ring" not in self.k_positions.columns:
                        self.k_positions.insert(0, "Ring", range(len(self.k_positions)))
                    break
            else:
                params, _ = _k_mod.load_parameters(tunnel_id, data_dir)
                if params is None:
                    params = {}
                self.k_positions = run_k_detection(
                    self.depth_map, self.ring_count, "dbscan", params,
                    tunnel_id=tunnel_id, base_dir=data_dir, verbose=False,
                )

        self.dimensions, self.param_names, self.base_params = get_ab_search_space(tunnel_id, data_dir)
        self.eval_count = 0
        self.best_score = np.inf
        self.best_params = None
        self.history = []
        self.logs_dir = Path(__file__).resolve().parent / "ab_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        _det_spec = spec_from_file_location("detection", os.path.join(PROJECT_ROOT, "agents", "irregular", "2_detection", "2_detection.py"))
        _det_mod = module_from_spec(_det_spec)
        _det_spec.loader.exec_module(_det_mod)
        self.detect_params, _ = _det_mod.load_parameters(tunnel_id, data_dir)
        if self.detect_params is None:
            self.detect_params = {}
        self.line_data = _det_mod.detect_lines(self.depth_map, self.detect_params)

    @property
    def global_eval_index(self) -> int:
        return self.eval_offset + self.eval_count

    def __call__(self, params: List) -> float:
        self.eval_count += 1
        start = time.time()
        try:
            kwargs = params_to_ab_kwargs(params, self.param_names)
            pred_df = run_ab_detection(
                self.depth_map,
                self.k_positions,
                line_data=self.line_data,
                params=self.detect_params,
                circ=self.circ,
                **kwargs,
            )
            score, info = ab_composite_objective(pred_df, self.gt_df, self.circ, clip_dist=800.0)
            elapsed = time.time() - start
            if score < self.best_score:
                self.best_score = score
                self.best_params = dict(zip(self.param_names, params))
                self.best_params["d_fracs"] = kwargs["d_fracs"]
                if self.verbose:
                    print(f"  [Eval {self.global_eval_index}] New best: score={score:.1f} "
                          f"(mean_dist={info['mean_distance']:.1f}px)")
            self.history.append({
                "eval": self.global_eval_index,
                "score": score,
                "mean_distance": info["mean_distance"],
            })
            self._log_trial(params, score, info, elapsed)
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.global_eval_index}] score={score:.1f}, mean_dist={info['mean_distance']:.1f}px")
            return score
        except Exception as e:
            elapsed = time.time() - start
            if self.verbose:
                print(f"  [Eval {self.global_eval_index}] Error: {e}")
            self._log_trial(params, None, None, elapsed, error=str(e))
            return 1e9

    def _log_trial(
        self,
        params: List,
        score: Optional[float],
        info: Optional[Dict],
        runtime: float,
        error: Optional[str] = None,
    ):
        trial_id = f"ab_detect_{self.tunnel_id}_{self.global_eval_index:03d}"
        log_path = self.logs_dir / f"{trial_id}.json"
        kwargs = params_to_ab_kwargs(params, self.param_names)
        log_data = {
            "trial_id": trial_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tunnel_id": self.tunnel_id,
            "params": kwargs,
        }
        if error:
            log_data["error"] = error
            log_data["objective_value"] = 1e9
        else:
            log_data["objective_value"] = score
            log_data["mean_distance"] = info.get("mean_distance")
            log_data["n_matched"] = info.get("n_matched")
        log_data["runtime_sec"] = runtime
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)


def run_ab_detection_bo(
    tunnel_id: str,
    data_dir: str = "data",
    n_calls: int = 200,
    n_initial_points: int = 20,
    verbose: bool = True,
    eval_offset: int = 0,
) -> Dict:
    """Run BO for A/B detection. Returns best score and params."""
    objective = ABDetectionObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
        eval_offset=eval_offset,
    )
    print(f"\nA/B Detection BO: tunnel={tunnel_id}, dims={len(objective.param_names)}")
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
    best_kwargs = params_to_ab_kwargs(best_x, objective.param_names)
    best_mean_dist = min((h.get("mean_distance", 9999) for h in objective.history), default=9999)
    print(f"\nBest score: {best_score:.2f}")
    print(f"Best mean centroid distance: {best_mean_dist:.1f} px")

    pred_df = run_ab_detection(
        objective.depth_map,
        objective.k_positions,
        line_data=objective.line_data,
        params=objective.detect_params,
        circ=objective.circ,
        **best_kwargs,
    )
    out_csv = os.path.join(objective.tunnel_dir, "all_segments_geo_ab_bo.csv")
    pred_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    # Also write all_segments.csv for downstream (e.g. 4-2_geo_per_instance.py)
    all_segments_csv = os.path.join(objective.tunnel_dir, "all_segments.csv")
    pred_df.to_csv(all_segments_csv, index=False)
    print(f"Saved: {all_segments_csv} (for per-instance geo / mIoU eval)")

    return {
        "tunnel_id": tunnel_id,
        "best_score": best_score,
        "best_mean_distance": best_mean_dist,
        "best_params": best_kwargs,
        "n_evaluations": objective.eval_count,
        "history": objective.history,
    }


def main():
    parser = argparse.ArgumentParser(description="A/B block detection BO (Hungarian-matched centroid distance)")
    parser.add_argument("tunnel_id", default="4-1", nargs="?", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-calls", type=int, default=200)
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_ab_detection_bo(
        args.tunnel_id,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
