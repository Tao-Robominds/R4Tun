"""
Bayesian Optimization for template geometric segmentation (3_template_geometric.py).

Optimizes 22 template shape parameters to maximize mIoU. Uses segments from Phase 1
(all_segments.csv from detection). GT-free at inference; GT used only as BO objective.

Uses forest_minimize (Random Forest surrogate) from scikit-optimize.
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real
from sklearn.metrics import jaccard_score

# agents/irregular/bo/ -> agents/irregular -> agents -> project root
IRREGULAR_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = IRREGULAR_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "template_geo",
    str(IRREGULAR_ROOT / "3_segmentation" / "3_template_geometric.py"),
)
_template_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_template_mod)
run_template_geometric = _template_mod.run_template_geometric


def compute_miou(final_csv: str) -> Tuple[float, dict]:
    """
    Compute mIoU from final.csv (pred vs segment columns).
    Returns (miou, details_dict) with per-class IoUs and OA.
    """
    df = pd.read_csv(final_csv)
    gt = np.nan_to_num(df["segment"].values, nan=-1)
    pr = np.nan_to_num(df["pred"].values, nan=-1)
    valid = (gt >= 1) & (gt <= 7) & (pr >= 0) & (pr <= 7)
    gt_v = gt[valid].astype(int)
    pr_v = pr[valid].astype(int)
    classes = np.arange(1, 8)
    ious = jaccard_score(gt_v, pr_v, labels=classes, average=None, zero_division=0)
    miou = float(np.mean(ious))
    oa = float(np.mean(gt_v == pr_v)) if len(gt_v) else 0.0
    names = {1: "K", 2: "B1", 3: "B2", 4: "A1", 5: "A2", 6: "A3", 7: "A4"}
    details = {
        "mIoU": miou,
        "OA": oa,
        "per_class": {names[c]: float(v) for c, v in zip(classes, ious)},
        "valid_points": int(valid.sum()),
    }
    return miou, details


def get_search_space_and_warmstart(tunnel_id: str) -> Tuple[List, List[str], List[float], dict]:
    """
    Build 22D search space, param names, warmstart, and fixed params.
    Fixed: K_half_width, B1_half_width, B2_half_width (set from segment_half_width in override).
    """
    params_path = IRREGULAR_ROOT / "3_segmentation" / "parameters" / tunnel_id / "parameters_geometric_template.json"
    warmstart = {}
    if params_path.exists():
        with open(params_path) as f:
            warmstart = json.load(f)

    # 22D: global 3 + K 3 + B1 4 + B2 4 + A 8
    dimensions = [
        Real(100, 300, name="segment_half_width"),
        Real(0, 20, name="shrink_x"),
        Real(0, 20, name="shrink_y"),
        Real(30, 150, name="K_half_height_pos"),
        Real(20, 120, name="K_half_height_neg"),
        Real(-80, 40, name="K_centre_offset"),
        Real(150, 400, name="B1_half_height_top"),
        Real(140, 380, name="B1_half_height_bottom_pos"),
        Real(150, 400, name="B1_half_height_bottom_neg"),
        Real(-50, 60, name="B1_centre_offset"),
        Real(80, 280, name="B2_half_height_top_pos"),
        Real(90, 300, name="B2_half_height_top_neg"),
        Real(200, 450, name="B2_half_height_bottom"),
        Real(-80, 40, name="B2_centre_offset"),
        Real(200, 450, name="A1_half_height"),
        Real(250, 450, name="A2_half_height"),
        Real(250, 450, name="A3_half_height"),
        Real(250, 450, name="A4_half_height"),
        Real(-80, 60, name="A1_centre_offset"),
        Real(-80, 60, name="A2_centre_offset"),
        Real(-80, 60, name="A3_centre_offset"),
        Real(-80, 60, name="A4_centre_offset"),
    ]
    param_names = [d.name for d in dimensions]
    x0 = [
        warmstart.get(name, (d.low + d.high) / 2)
        for name, d in zip(param_names, dimensions)
    ]
    return dimensions, param_names, x0, warmstart


class TemplateGeoObjective:
    """Optimize template segmentation params to maximize mIoU."""

    def __init__(
        self,
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

        self.dimensions, self.param_names, self.x0, self.base_params = get_search_space_and_warmstart(tunnel_id)
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_count = 0
        self.best_miou = -1.0
        self.best_params = None

        if verbose:
            print(f"Template BO ({len(self.param_names)}D)")
            print(f"  Segments file: {segments_file}")

    def __call__(self, params: List) -> float:
        self.eval_count += 1
        t0 = time.time()
        try:
            param_dict = dict(self.base_params)
            for name, val in zip(self.param_names, params):
                param_dict[name] = float(val)
            # Tie half_widths to segment_half_width for consistency
            sw = param_dict["segment_half_width"]
            param_dict["K_half_width"] = sw
            param_dict["B1_half_width"] = sw
            param_dict["B2_half_width"] = sw

            import io
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_template_geometric(
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
        trial_id = f"template_{self.tunnel_id}_{self.eval_count:03d}"
        log_data = {
            "trial_id": trial_id,
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
        out = str(IRREGULAR_ROOT / "3_segmentation" / "parameters" / self.tunnel_id / "parameters_geometric_template.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.best_params, f, indent=2)
        return out


def run_template_geo_bo(
    tunnel_id: str,
    data_dir: str = "data",
    segments_file: str = None,
    n_calls: int = 300,
    n_initial: int = 25,
    verbose: bool = True,
) -> Dict:
    """Run BO for template geometric segmentation. Returns best mIoU and params."""
    if segments_file is None:
        segments_file = os.path.join(data_dir, tunnel_id, "all_segments.csv")
    if not os.path.isabs(segments_file):
        segments_file = os.path.abspath(segments_file)
    if not os.path.exists(segments_file):
        raise FileNotFoundError(f"Segments file not found: {segments_file}")

    logs_dir = str(IRREGULAR_ROOT / "bo" / f"logs_template_geo_{tunnel_id}")

    print(f"\n{'=' * 70}")
    print(f"TEMPLATE GEOMETRIC BO — Tunnel {tunnel_id}")
    print(f"Segments: {segments_file}  |  N calls: {n_calls}  |  N initial: {n_initial}")
    print(f"{'=' * 70}")

    objective = TemplateGeoObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        segments_file=segments_file,
        logs_dir=logs_dir,
        verbose=verbose,
    )

    print(f"\nSearch space: {len(objective.param_names)} dimensions")
    for i, (name, dim) in enumerate(zip(objective.param_names, objective.dimensions)):
        ws = objective.x0[i]
        print(f"  {name:28s}  [{dim.low:.0f}, {dim.high:.0f}]  warmstart={ws:.2f}")
    print(f"\nStarting optimization...")

    result = forest_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        x0=[objective.x0],
        y0=None,
        random_state=42,
        verbose=False,
    )

    best_miou = -result.fun
    best_dict = dict(zip(objective.param_names, result.x))

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Best mIoU = {best_miou:.4f}")
    print(f"{'=' * 70}")
    for name, val in best_dict.items():
        print(f"  {name:28s}: {val:.2f}")

    results_file = os.path.join(logs_dir, f"best_template_{tunnel_id}.json")
    with open(results_file, "w") as f:
        json.dump({
            "miou": best_miou,
            "params": {k: float(v) for k, v in best_dict.items()},
            "tunnel_id": tunnel_id,
        }, f, indent=2)
    print(f"\nSaved best params to {results_file}")

    out_path = objective.save_best_params()
    if out_path:
        print(f"Saved template params to {out_path}")

    return {"miou": best_miou, "params": best_dict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Template geometric segmentation BO (mIoU objective)")
    parser.add_argument("tunnel_id", default="4-1", nargs="?", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--segments-file", default=None, help="all_segments.csv from Phase 1 (default: data/<tunnel>/all_segments.csv)")
    parser.add_argument("--n-calls", type=int, default=300)
    parser.add_argument("--n-initial", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_template_geo_bo(
        args.tunnel_id,
        data_dir=args.data_dir,
        segments_file=args.segments_file,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        verbose=not args.quiet,
    )
