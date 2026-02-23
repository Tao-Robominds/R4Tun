"""
Two-Stage Bayesian Optimization for Geometric Segmentation Pipeline (4-1)

Stage A — Detection (19D): Optimizes 7 K_Y positions + 12 grouped offsets.
    Geometric params fixed at GT-derived values.
Stage B — Segmentation (15D): Optimizes 7 half_heights + 7 centre_offsets + segment_half_width.
    Detection params fixed from Stage A best or GT.

Both stages use mIoU as the objective (requires GT labels).
Uses forest_minimize (Random Forest surrogate) from scikit-optimize.
"""

import os
import sys
import json
import glob
import time
import argparse
import importlib.util
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real, Integer
from sklearn.metrics import jaccard_score

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load geometric segmentation module
_geo_path = PROJECT_ROOT / "agents" / "irregular" / "3_segmentation" / "3_geometric.py"
_geo_spec = importlib.util.spec_from_file_location("geo_module", str(_geo_path))
_geo_module = importlib.util.module_from_spec(_geo_spec)
sys.modules["geo_module"] = _geo_module
_geo_spec.loader.exec_module(_geo_module)

run_geometric = _geo_module.run_geometric


# =============================================================================
# mIoU Computation
# =============================================================================

def compute_miou(final_csv: str) -> Tuple[float, dict]:
    """
    Compute mIoU from final.csv (pred vs segment columns).

    Returns:
        (miou, details_dict) where details has per-class IoUs and OA.
    """
    df = pd.read_csv(final_csv)
    gt = df["segment"].values
    pr = df["pred"].values

    valid = (~np.isnan(gt)) & (gt >= 1) & (gt <= 7) & (pr >= 0) & (pr <= 7)
    gt_v = gt[valid].astype(int)
    pr_v = pr[valid].astype(int)

    classes = np.arange(1, 8)
    ious = jaccard_score(gt_v, pr_v, labels=classes, average=None, zero_division=0)
    miou = float(np.mean(ious))
    oa = float(np.mean(gt_v == pr_v))

    names = {1: "K", 2: "B1", 3: "B2", 4: "A1", 5: "A2", 6: "A3", 7: "A4"}
    details = {
        "mIoU": miou,
        "OA": oa,
        "per_class": {names[c]: float(v) for c, v in zip(classes, ious)},
        "valid_points": int(valid.sum()),
    }
    return miou, details


# =============================================================================
# Stage A — Detection BO (K_Y + grouped offsets, 19D)
# =============================================================================

class DetectionObjective:
    """Optimize K Y positions + grouped offsets to maximize mIoU."""

    def __init__(
        self,
        config: dict,
        tunnel_id: str,
        data_dir: str,
        logs_dir: str,
        verbose: bool = True,
    ):
        self.config = config
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.logs_dir = logs_dir
        self.verbose = verbose
        self.img_height = config["img_height"]
        self.ring_count = config["ring_count"]
        self.groups = config["stagger_groups"]
        self.k_x = config["detected_k_x"]

        # Fixed geometric params (GT-derived)
        self.geo_params = config["segmentation_search_space"]["warmstart"].copy()
        self.geo_params.update(config["segmentation_search_space"]["fixed_params"])

        # Build search space
        det_space = config["detection_search_space"]
        self.dimensions = []
        self.param_names = []

        # 7 K_Y params
        for i in range(self.ring_count):
            name = f"k_y_r{i}"
            lo, hi = det_space["k_y_bounds"][name]
            self.dimensions.append(Real(lo, hi, name=name))
            self.param_names.append(name)

        # 12 grouped offset params
        for key, (lo, hi) in det_space["group_offset_bounds"].items():
            self.dimensions.append(Real(lo, hi, name=key))
            self.param_names.append(key)

        # Build warmstart
        ws = det_space.get("warmstart", {})
        self.x0 = [ws.get(n, (self.dimensions[i].low + self.dimensions[i].high) / 2)
                    for i, n in enumerate(self.param_names)]

        os.makedirs(logs_dir, exist_ok=True)
        self.eval_count = 0
        self.best_miou = -1.0
        self.best_params = None

        if verbose:
            print(f"Detection BO ({len(self.param_names)}D)")
            print(f"  K Y params: 7, group offset params: 12")
            print(f"  Geometric params fixed at GT-derived values")

    def _build_segments(self, param_dict: dict) -> pd.DataFrame:
        """Build all_segments.csv from K positions + grouped offsets."""
        rows = []
        blocks = ["B1", "B2", "A1", "A2", "A3", "A4"]

        for ring_idx in range(self.ring_count):
            k_x = self.k_x[ring_idx]
            k_y = param_dict[f"k_y_r{ring_idx}"]
            group = "A" if ring_idx in self.groups["A"] else "B"

            rows.append({
                "Ring": ring_idx, "Block": "K",
                "X": k_x, "Y": k_y % self.img_height, "quality": 1.0,
            })

            for block in blocks:
                offset = param_dict[f"{group}_{block}"]
                y = (k_y + offset) % self.img_height
                rows.append({
                    "Ring": ring_idx, "Block": block,
                    "X": k_x, "Y": round(y, 1), "quality": 1.0,
                })

        return pd.DataFrame(rows)

    def __call__(self, params: List) -> float:
        self.eval_count += 1
        t0 = time.time()

        try:
            param_dict = dict(zip(self.param_names, params))
            segments_df = self._build_segments(param_dict)

            # Write temp segments file (absolute path to avoid double-prefix)
            seg_path = os.path.abspath(os.path.join(self.tunnel_dir, "all_segments_bo.csv"))
            segments_df.to_csv(seg_path, index=False)

            # Run geometric segmentation with fixed geometric params
            import io
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_geometric(
                    self.tunnel_id,
                    base_dir=self.data_dir,
                    segments_file=seg_path,
                    override_params=self.geo_params,
                )

            final_csv = os.path.join(self.tunnel_dir, "final.csv")
            miou, details = compute_miou(final_csv)
            runtime = time.time() - t0

            if miou > self.best_miou:
                self.best_miou = miou
                self.best_params = param_dict.copy()
                if self.verbose:
                    per_cls = " ".join(f"{k}={v:.3f}" for k, v in details["per_class"].items())
                    print(f"  [#{self.eval_count}] NEW BEST mIoU={miou:.4f}  {per_cls}  ({runtime:.1f}s)")

            elif self.verbose and (self.eval_count <= 5 or self.eval_count % 20 == 0):
                print(f"  [#{self.eval_count}] mIoU={miou:.4f}  best={self.best_miou:.4f}  ({runtime:.1f}s)")

            # Log
            self._log(param_dict, miou, details, runtime)
            return -miou  # minimize

        except Exception as e:
            runtime = time.time() - t0
            if self.verbose:
                print(f"  [#{self.eval_count}] ERROR: {e}  ({runtime:.1f}s)")
            self._log({}, 0.0, {}, runtime, error=str(e))
            return 0.0

    def _log(self, params, miou, details, runtime, error=None):
        trial_id = f"det_{self.tunnel_id}_{self.eval_count:03d}"
        log_data = {
            "trial_id": trial_id,
            "stage": "detection",
            "params": params,
            "miou": miou,
            "details": details,
            "runtime_sec": runtime,
        }
        if error:
            log_data["error"] = error
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)


# =============================================================================
# Stage B — Segmentation BO (geometric params, 15D)
# =============================================================================

class SegmentationObjective:
    """Optimize geometric segmentation params to maximize mIoU."""

    def __init__(
        self,
        config: dict,
        tunnel_id: str,
        data_dir: str,
        segments_file: str,
        logs_dir: str,
        verbose: bool = True,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.segments_file = segments_file
        self.logs_dir = logs_dir
        self.verbose = verbose

        seg_space = config["segmentation_search_space"]
        self.fixed_params = seg_space.get("fixed_params", {})

        # Build search space
        self.dimensions = []
        self.param_names = []
        for name, (lo, hi) in seg_space["bounds"].items():
            self.dimensions.append(Real(lo, hi, name=name))
            self.param_names.append(name)

        # Warmstart
        ws = seg_space.get("warmstart", {})
        self.x0 = [ws.get(n, (self.dimensions[i].low + self.dimensions[i].high) / 2)
                    for i, n in enumerate(self.param_names)]

        os.makedirs(logs_dir, exist_ok=True)
        self.eval_count = 0
        self.best_miou = -1.0
        self.best_params = None

        if verbose:
            print(f"Segmentation BO ({len(self.param_names)}D)")
            print(f"  Segments file: {segments_file}")
            print(f"  Fixed params: {list(self.fixed_params.keys())}")

    def __call__(self, params: List) -> float:
        self.eval_count += 1
        t0 = time.time()

        try:
            param_dict = dict(self.fixed_params)
            for name, val in zip(self.param_names, params):
                param_dict[name] = float(val)

            import io
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_geometric(
                    self.tunnel_id,
                    base_dir=self.data_dir,
                    segments_file=self.segments_file,
                    override_params=param_dict,
                )

            final_csv = os.path.join(self.tunnel_dir, "final.csv")
            miou, details = compute_miou(final_csv)
            runtime = time.time() - t0

            if miou > self.best_miou:
                self.best_miou = miou
                self.best_params = param_dict.copy()
                if self.verbose:
                    per_cls = " ".join(f"{k}={v:.3f}" for k, v in details["per_class"].items())
                    print(f"  [#{self.eval_count}] NEW BEST mIoU={miou:.4f}  {per_cls}  ({runtime:.1f}s)")

            elif self.verbose and (self.eval_count <= 5 or self.eval_count % 20 == 0):
                print(f"  [#{self.eval_count}] mIoU={miou:.4f}  best={self.best_miou:.4f}  ({runtime:.1f}s)")

            self._log(param_dict, miou, details, runtime)
            return -miou

        except Exception as e:
            runtime = time.time() - t0
            if self.verbose:
                print(f"  [#{self.eval_count}] ERROR: {e}  ({runtime:.1f}s)")
            self._log({}, 0.0, {}, runtime, error=str(e))
            return 0.0

    def _log(self, params, miou, details, runtime, error=None):
        trial_id = f"seg_{self.tunnel_id}_{self.eval_count:03d}"
        log_data = {
            "trial_id": trial_id,
            "stage": "segmentation",
            "params": params,
            "miou": miou,
            "details": details,
            "runtime_sec": runtime,
        }
        if error:
            log_data["error"] = error
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def save_best_params(self) -> Optional[str]:
        if self.best_params is None:
            return None
        out = os.path.join(
            PROJECT_ROOT, "agents", "irregular", "3_segmentation",
            "parameters", self.tunnel_id, "parameters_geometric.json",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.best_params, f, indent=2)
        return out


# =============================================================================
# Main runner
# =============================================================================

def run_bo(
    tunnel_id: str,
    mode: str = "detection",
    data_dir: str = "data/wrap",
    n_calls: int = 100,
    n_initial: int = 15,
    segments_file: str = None,
    verbose: bool = True,
):
    """
    Run staged Bayesian Optimization.

    Args:
        tunnel_id:  e.g. '4-1'
        mode:       'detection' (Stage A, 19D) or 'segmentation' (Stage B, 15D)
        data_dir:   base data directory
        n_calls:    total BO iterations
        n_initial:  random initial points
        segments_file:  path to all_segments.csv for segmentation mode
        verbose:    print progress
    """
    config_path = PROJECT_ROOT / "bo" / "complex_staggered" / "configs" / f"geometric_{tunnel_id}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    logs_dir = str(PROJECT_ROOT / "bo" / "complex_staggered" / f"logs_geo_{tunnel_id}")

    print(f"\n{'=' * 70}")
    print(f"GEOMETRIC BO — Stage {'A (Detection)' if mode == 'detection' else 'B (Segmentation)'}")
    print(f"Tunnel: {tunnel_id}  |  Mode: {mode}  |  N calls: {n_calls}")
    print(f"{'=' * 70}")

    if mode == "detection":
        objective = DetectionObjective(config, tunnel_id, data_dir, logs_dir, verbose)
        x0 = [objective.x0]
        y0 = None  # let BO evaluate warmstart

    elif mode == "segmentation":
        if segments_file is None:
            segments_file = os.path.join(data_dir, tunnel_id, "all_segments_gt.csv")
        if not os.path.isabs(segments_file):
            segments_file = os.path.abspath(segments_file)
        objective = SegmentationObjective(
            config, tunnel_id, data_dir, segments_file, logs_dir, verbose,
        )
        x0 = [objective.x0]
        y0 = None

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\nSearch space: {len(objective.param_names)} dimensions")
    for i, (name, dim) in enumerate(zip(objective.param_names, objective.dimensions)):
        ws = x0[0][i] if x0 else "?"
        print(f"  {name:20s}  [{dim.low:.0f}, {dim.high:.0f}]  warmstart={ws:.1f}")
    print(f"\nStarting optimization ({n_calls} calls, {n_initial} initial)...")

    result = forest_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        x0=x0,
        y0=y0,
        random_state=42,
        verbose=False,
    )

    best_miou = -result.fun
    best_dict = dict(zip(objective.param_names, result.x))

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Best mIoU = {best_miou:.4f}")
    print(f"{'=' * 70}")
    for name, val in best_dict.items():
        print(f"  {name:20s}: {val:.2f}")

    # Save results
    results_file = os.path.join(logs_dir, f"best_{mode}_{tunnel_id}.json")
    with open(results_file, "w") as f:
        json.dump({"mode": mode, "miou": best_miou, "params": {k: float(v) for k, v in best_dict.items()}}, f, indent=2)
    print(f"\nSaved best params to {results_file}")

    if mode == "segmentation" and hasattr(objective, "save_best_params"):
        p = objective.save_best_params()
        if p:
            print(f"Saved geometric params to {p}")

    return {"miou": best_miou, "params": best_dict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage geometric BO for 4-1")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--mode", choices=["detection", "segmentation"], default="detection",
                        help="BO stage: detection (19D) or segmentation (15D)")
    parser.add_argument("--data-dir", default="data/wrap")
    parser.add_argument("--n-calls", type=int, default=100)
    parser.add_argument("--n-initial", type=int, default=15)
    parser.add_argument("--segments-file", default=None,
                        help="Segments CSV for segmentation mode")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    run_bo(
        args.tunnel_id,
        mode=args.mode,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        segments_file=args.segments_file,
        verbose=not args.quiet,
    )
