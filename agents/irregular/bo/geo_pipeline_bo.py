"""
Geo pipeline Bayesian Optimization: tune detection offsets (ring-by-ring) and shrink params.

Phase 1 (offset): Tune 6 per-ring Y offsets per ring (7 rings × 6D), mIoU objective.
Phase 2 (shrink): Tune shrink_x, shrink_y (2D), mIoU objective.

Usage:
  python -m p4tun.bo.geo_pipeline_bo 4-1 --phase offset --ring 0 --n-calls 40
  python -m p4tun.bo.geo_pipeline_bo 4-1 --phase offset --ring all   # ~3.5h for 7 rings × 40 calls
  python -m p4tun.bo.geo_pipeline_bo 4-1 --phase shrink --n-calls 30   # after Phase 1
"""

import os
import sys
import json
import argparse
import importlib.util
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
P4TUN_DIR = os.path.join(PROJECT_ROOT, "p4tun")


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy-load p4tun scripts to avoid circular imports and to run from any cwd
_geo_detection = _load_module("geo_detection", os.path.join(P4TUN_DIR, "4-1_geo_detection.py"))
_geo_seg = _load_module("geo_seg", os.path.join(P4TUN_DIR, "4-2_geo_per_instance.py"))
_evaluation = _load_module("evaluation", os.path.join(P4TUN_DIR, "evaluation.py"))

run_geo_detection_with_offsets = _geo_detection.run_geo_detection_with_offsets
compute_gt_derived_offsets = _geo_detection.compute_gt_derived_offsets
get_k_positions_and_height = _geo_detection.get_k_positions_and_height
expand_k_with_per_ring_offsets = _geo_detection.expand_k_with_per_ring_offsets
EXPANSION_BLOCKS = _geo_detection.EXPANSION_BLOCKS

run_per_instance_geometric = _geo_seg.run_per_instance_geometric
get_miou = _evaluation.get_miou

OFFSETS_JSON = "bo_geo_offsets.json"
SEGMENTS_OUT = "all_segments.csv"


def _merge_offsets(base: Dict[str, float], ring_index: int, x: List[float]) -> Dict[str, float]:
    """Merge base offsets with 6 values for ring_index (order: B1, B2, A1, A2, A3, A4)."""
    out = dict(base)
    for i, block in enumerate(EXPANSION_BLOCKS):
        key = f"{block.lower()}_offset_r{ring_index}"
        out[key] = float(x[i])
    return out


class GeoPipelineObjective:
    """
    Objective for BO: run detection (with given offsets or fixed) -> segmentation -> mIoU.
    Returns -mIoU for minimization.
    """

    def __init__(
        self,
        tunnel_id: str,
        base_dir: str,
        phase: str,
        ring_index: Optional[int] = None,
        fixed_offsets: Optional[Dict[str, float]] = None,
        k_positions: Optional[pd.DataFrame] = None,
        img_height: Optional[int] = None,
        shrink_x: float = 4.0,
        shrink_y: float = 2.0,
        verbose: bool = True,
    ):
        self.tunnel_id = tunnel_id
        self.base_dir = base_dir
        self.phase = phase
        self.ring_index = ring_index
        self.fixed_offsets = fixed_offsets or {}
        self.k_positions = k_positions
        self.img_height = img_height
        self.shrink_x = shrink_x
        self.shrink_y = shrink_y
        self.verbose = verbose
        self.tunnel_dir = os.path.join(base_dir, tunnel_id)
        self.eval_count = 0
        self.best_miou = -1.0
        self.best_x = None

    def __call__(self, x: List) -> float:
        self.eval_count += 1
        try:
            if self.phase == "offset":
                merged = _merge_offsets(self.fixed_offsets, self.ring_index, x)
                all_segments = expand_k_with_per_ring_offsets(
                    self.k_positions, self.img_height, merged, use_gt_x_df=None
                )
                out_path = os.path.join(self.tunnel_dir, SEGMENTS_OUT)
                all_segments.to_csv(out_path, index=False)
                shrink_x, shrink_y = self.shrink_x, self.shrink_y
            else:
                # shrink: all_segments.csv already on disk from Phase 1; use trial x
                shrink_x, shrink_y = float(x[0]), float(x[1])

            run_per_instance_geometric(
                self.tunnel_id,
                base_dir=self.base_dir,
                segments_file=SEGMENTS_OUT,
                shrink_x=shrink_x,
                shrink_y=shrink_y,
            )
            miou = get_miou(self.tunnel_id, base_dir=self.base_dir, segment_count=7)
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_x = list(x)
                if self.verbose:
                    print(f"  [Eval {self.eval_count}] New best mIoU: {miou:.4f}")
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.eval_count}] mIoU: {miou:.4f}")
            return -miou
        except Exception as e:
            if self.verbose:
                print(f"  [Eval {self.eval_count}] Error: {e}")
            return 0.0


def run_offset_phase(
    tunnel_id: str,
    base_dir: str,
    ring_index: int,
    n_calls: int = 40,
    n_initial: int = 15,
    verbose: bool = True,
) -> Tuple[Dict[str, float], float]:
    """Run BO for one ring's 6 offsets. Returns (best_offsets_for_this_ring, best_miou)."""
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    offsets_path = os.path.join(tunnel_dir, OFFSETS_JSON)

    k_positions, img_height = get_k_positions_and_height(tunnel_id, base_dir)
    gt_offsets, _ = compute_gt_derived_offsets(tunnel_dir, img_height)

    # Base offsets: load saved best or GT-derived
    if os.path.exists(offsets_path):
        with open(offsets_path, "r") as f:
            base_offsets = json.load(f)
    else:
        base_offsets = dict(gt_offsets) if gt_offsets else {}

    # Warm-start for this ring from GT
    x0_vals = [base_offsets.get(f"{b.lower()}_offset_r{ring_index}", 0.0) for b in EXPANSION_BLOCKS]
    x0_vals = [max(-2400, min(2400, v)) for v in x0_vals]
    dimensions = [Real(-2400.0, 2400.0, name=f"{b.lower()}_offset_r{ring_index}") for b in EXPANSION_BLOCKS]
    param_names = [f"{b.lower()}_offset_r{ring_index}" for b in EXPANSION_BLOCKS]

    objective = GeoPipelineObjective(
        tunnel_id=tunnel_id,
        base_dir=base_dir,
        phase="offset",
        ring_index=ring_index,
        fixed_offsets=base_offsets,
        k_positions=k_positions,
        img_height=img_height,
        verbose=verbose,
    )

    result = forest_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        x0=[x0_vals],
        random_state=42,
        verbose=False,
    )
    best_x = result.x
    best_miou = -result.fun
    # Update base_offsets with this ring's best
    for i, name in enumerate(param_names):
        base_offsets[name] = float(best_x[i])
    os.makedirs(tunnel_dir, exist_ok=True)
    with open(offsets_path, "w") as f:
        json.dump(base_offsets, f, indent=2)
    return base_offsets, best_miou


def run_shrink_phase(
    tunnel_id: str,
    base_dir: str,
    n_calls: int = 30,
    n_initial: int = 10,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Assume all_segments.csv is already written. Tune shrink_x, shrink_y. Returns (shrink_x, shrink_y, best_miou)."""
    dimensions = [Real(0.0, 15.0, name="shrink_x"), Real(0.0, 15.0, name="shrink_y")]
    objective = GeoPipelineObjective(
        tunnel_id=tunnel_id,
        base_dir=base_dir,
        phase="shrink",
        ring_index=None,
        shrink_x=4.0,
        shrink_y=2.0,
        verbose=verbose,
    )
    result = forest_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        x0=[[4.0, 2.0]],
        random_state=42,
        verbose=False,
    )
    sx, sy = float(result.x[0]), float(result.x[1])
    best_miou = -result.fun
    return sx, sy, best_miou


def main():
    parser = argparse.ArgumentParser(description="Geo pipeline BO: offsets (ring-by-ring) and shrink")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--phase", choices=["offset", "shrink"], required=True)
    parser.add_argument("--ring", default=None, help="Ring index 0-6, or 'all' for offset phase")
    parser.add_argument("--n-calls", type=int, default=40, help="BO evaluations (offset phase)")
    parser.add_argument("--n-initial", type=int, default=15, help="Initial random points (offset)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    base_dir = args.data_dir
    tunnel_id = args.tunnel_id

    if args.phase == "offset":
        if args.ring is None:
            raise SystemExit("--phase offset requires --ring N or --ring all")
        if args.ring == "all":
            rings = list(range(7))
        else:
            rings = [int(args.ring)]
        for r in rings:
            print(f"\n{'='*60}\nOffset BO ring {r}\n{'='*60}")
            best_offsets, best_miou = run_offset_phase(
                tunnel_id, base_dir, r, n_calls=args.n_calls, n_initial=args.n_initial, verbose=args.verbose
            )
            print(f"Ring {r} best mIoU: {best_miou:.4f}")
        print(f"\nSaved offsets to {os.path.join(base_dir, tunnel_id, OFFSETS_JSON)}")
    else:
        # shrink: ensure all_segments.csv exists (run detection with saved or GT offsets)
        tunnel_dir = os.path.join(base_dir, tunnel_id)
        offsets_path = os.path.join(tunnel_dir, OFFSETS_JSON)
        gt_offsets, _ = compute_gt_derived_offsets(tunnel_dir, 4711)
        if os.path.exists(offsets_path):
            with open(offsets_path, "r") as f:
                offsets = json.load(f)
        else:
            offsets = gt_offsets
        if offsets:
            run_geo_detection_with_offsets(
                tunnel_id, base_dir, per_ring_offsets=offsets, k_positions=None,
                output_file=SEGMENTS_OUT, verbose=True
            )
        print(f"\n{'='*60}\nShrink BO\n{'='*60}")
        sx, sy, best_miou = run_shrink_phase(tunnel_id, base_dir, n_calls=30, n_initial=10, verbose=args.verbose)
        print(f"Best shrink_x={sx:.2f}, shrink_y={sy:.2f}, mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
