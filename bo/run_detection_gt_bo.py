"""
Bayesian optimization of line-detection parameters using ground truth segment boundaries.

Tunes detection parameters (edge + Hough oblique + Hough horizontal) so that detected
lines (oblique + horizontal) align with GT boundaries from unwrapped.csv. Objective:
maximize fraction of GT boundaries within 20px of a detection, averaged over all rings.
Does not run full pipeline or overwrite data/; saves logs under logs/<tunnel_id>/detection_gt_bo/.

Usage:
  ./venv/bin/python bo/run_detection_gt_bo.py 4-1 --n-calls 40
  ./venv/bin/python bo/run_detection_gt_bo.py 4-1 --ring 0 --n-calls 200 --logs-dir data/4-1/detection_gt_bo_200runs/ring_0
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("TQDM_DISABLE", "1")

AGENTS = os.path.join(PROJECT_ROOT, "agents", "irregular")
DETECTION_PARAMS_DIR = os.path.join(AGENTS, "2_detection", "parameters")


def load_base_detection_params(tunnel_id: str) -> dict:
    path = os.path.join(DETECTION_PARAMS_DIR, tunnel_id, "parameters_detection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base detection params not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_detection_params(tunnel_id: str, params: dict) -> None:
    path = os.path.join(DETECTION_PARAMS_DIR, tunnel_id, "parameters_detection.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def build_params_from_sample(base: dict, x: list, dim_names: list) -> dict:
    """Build full params from a sample vector x; dim_names[i] is the param name for x[i]."""
    params = dict(base)
    for i, name in enumerate(dim_names):
        if name is None:
            continue
        val = x[i]
        if name in ("binary_threshold", "hough_threshold", "hough_min_length", "hough_max_gap",
                    "hough_horizontal_threshold", "hough_horizontal_min_length", "hough_horizontal_max_gap",
                    "canny_low", "canny_high", "dilation_kernel_size", "dilation_iterations"):
            params[name] = int(round(val))
        else:
            params[name] = float(val)
    return params


def main():
    parser = argparse.ArgumentParser(description="BO for line detection using GT segment boundaries")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--ring", type=int, default=None, help="Optimize for this ring only (0-based); saves to logs_dir without overwriting other rings")
    parser.add_argument("--n-calls", type=int, default=40, help="Number of BO trials")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--logs-dir", default=None, help="Logs directory (default: logs/<tunnel_id>/detection_gt_bo or .../ring_N for per-ring)")
    parser.add_argument("--match-thresh", type=float, default=20.0, help="GT boundary match threshold (px)")
    args = parser.parse_args()

    tunnel_id = args.tunnel_id
    base_dir = args.data_dir
    ring_index = args.ring

    if ring_index is not None:
        logs_dir = args.logs_dir or os.path.join(PROJECT_ROOT, "logs", tunnel_id, "detection_gt_bo", f"ring_{ring_index}")
    else:
        logs_dir = args.logs_dir or os.path.join(PROJECT_ROOT, "logs", tunnel_id, "detection_gt_bo")
    os.makedirs(logs_dir, exist_ok=True)

    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_path):
        print(f"Preprocessing required: {depth_path} not found", file=sys.stderr)
        sys.exit(1)
    depth_map = np.load(depth_path)

    import importlib
    _cmp = importlib.import_module("agents.irregular.2_detection.scripts.compare_detect_lines_to_gt")
    if ring_index is not None:
        compute_gt_line_metric = lambda td, dm, p, thresh=args.match_thresh: _cmp.compute_gt_line_metric_one_ring(
            td, dm, p, ring_index, match_thresh_px=thresh
        )
    else:
        compute_gt_line_metric = _cmp.compute_gt_line_metric

    base = load_base_detection_params(tunnel_id)
    original_params = json.loads(json.dumps(base))

    # Search space: line-detection params (focus on horizontal + shared edge/oblique)
    # Order must match x indices in build_params_from_sample
    def _int(name, low, high):
        return Integer(low, high, name=name)

    def _float(name, low, high):
        return Real(low, high, name=name)

    b = base.get("binary_threshold", 195)
    ht = base.get("hough_threshold", 37)
    hml = base.get("hough_min_length", 86)
    hmg = base.get("hough_max_gap", 39)
    aomin = base.get("angle_oblique_min", 6.0)
    aomax = base.get("angle_oblique_max", 9.0)
    hht = base.get("hough_horizontal_threshold", 50)
    hhml = base.get("hough_horizontal_min_length", 100)
    hhmg = base.get("hough_horizontal_max_gap", 10)
    hat = base.get("horizontal_angle_tolerance", 1.0)
    cl = base.get("canny_low", 50)
    ch = base.get("canny_high", 150)
    dks = base.get("dilation_kernel_size", 3)
    di = base.get("dilation_iterations", 1)

    space = [
        _int("binary_threshold", max(1, b - 80), min(255, b + 80)),
        _int("hough_threshold", max(5, ht - 30), min(150, ht + 50)),
        _int("hough_min_length", max(20, hml - 50), min(200, hml + 80)),
        _int("hough_max_gap", max(5, hmg - 30), min(120, hmg + 50)),
        _float("angle_oblique_min", max(3.0, aomin - 3), min(15.0, aomin + 3)),
        _float("angle_oblique_max", max(5.0, aomax - 3), min(18.0, aomax + 3)),
        _int("hough_horizontal_threshold", max(10, hht - 40), min(150, hht + 80)),
        _int("hough_horizontal_min_length", max(30, hhml - 70), min(250, hhml + 100)),
        _int("hough_horizontal_max_gap", max(2, hhmg - 15), min(50, hhmg + 30)),
        _float("horizontal_angle_tolerance", 0.2, 5.0),
        _int("canny_low", max(20, cl - 40), min(150, cl + 50)),
        _int("canny_high", max(80, ch - 80), min(255, ch + 80)),
        _int("dilation_kernel_size", 2, 5),
        _int("dilation_iterations", 1, 3),
    ]
    dim_names = [s.name for s in space]

    trial_count = [0]

    def objective(x: list) -> float:
        trial_count[0] += 1
        n = trial_count[0]
        t0 = time.perf_counter()
        params = build_params_from_sample(base, x, dim_names)
        save_detection_params(tunnel_id, params)
        try:
            if ring_index is not None:
                metrics = compute_gt_line_metric(tunnel_dir, depth_map, params)
            else:
                metrics = compute_gt_line_metric(tunnel_dir, depth_map, params, match_thresh_px=args.match_thresh)
        except Exception as e:
            print(f"Trial {n} failed: {e}", file=sys.stderr)
            return 1.0
        runtime_sec = time.perf_counter() - t0
        matched_frac = metrics["matched_frac"]
        mae_avg = metrics["mae_avg"]
        loss = -matched_frac
        int_params = {
            "binary_threshold", "hough_threshold", "hough_min_length", "hough_max_gap",
            "hough_horizontal_threshold", "hough_horizontal_min_length", "hough_horizontal_max_gap",
            "canny_low", "canny_high", "dilation_kernel_size", "dilation_iterations",
        }
        log_params = {}
        for i, name in enumerate(dim_names):
            v = x[i]
            log_params[name] = int(round(v)) if name in int_params else float(v)
        log = {
            "trial_id": n,
            "tunnel_id": tunnel_id,
            "ring_index": ring_index,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "params": log_params,
            "metrics": metrics,
            "loss": loss,
            "runtime_sec": round(runtime_sec, 2),
        }
        log_path = os.path.join(logs_dir, f"trial_{n:04d}.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Trial {n} matched_frac={matched_frac:.3f} mae={mae_avg:.1f} -> {log_path}")
        gc.collect()
        return loss

    n_initial = min(10, args.n_calls)
    ring_info = f" ring={ring_index}" if ring_index is not None else ""
    print(f"BO: tunnel={tunnel_id}{ring_info} n_calls={args.n_calls} n_initial={n_initial} logs={logs_dir}")
    res = gp_minimize(
        objective,
        space,
        n_calls=args.n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=True,
    )
    save_detection_params(tunnel_id, original_params)
    best_matched = -res.fun
    int_params = {
        "binary_threshold", "hough_threshold", "hough_min_length", "hough_max_gap",
        "hough_horizontal_threshold", "hough_horizontal_min_length", "hough_horizontal_max_gap",
        "canny_low", "canny_high", "dilation_kernel_size", "dilation_iterations",
    }
    best_dict = {dim_names[i]: (int(round(res.x[i])) if dim_names[i] in int_params else float(res.x[i])) for i in range(len(dim_names))}
    best_path = os.path.join(logs_dir, "best_params.json")
    best_payload = {"matched_frac": best_matched, "params": best_dict}
    if ring_index is not None:
        best_payload["ring_index"] = ring_index
    with open(best_path, "w") as f:
        json.dump(best_payload, f, indent=2)
    print(f"Best matched_frac={best_matched:.3f}")
    print(f"Best params -> {best_path}")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
