"""
Per-ring offset BO for irregular tunnels.

Keeps all line detection parameters fixed; only tunes per_ring_offsets
(7 boundary offsets: K, B1, B2, A1, A2, A3, A4) per ring.
Saves each trial to bo/logs/{tunnel_id}/offset_ring{R}_{NNN}.json (intrinsic.v1 schema).

Usage:
  ./venv/bin/python bo/run_offset_bo.py 5-1 --ring 0 --n-calls 50
  ./venv/bin/python bo/run_offset_bo.py 4-1 --ring 0 --n-calls 100   # more iterations
  ./venv/bin/python bo/run_offset_bo.py 4-1 --ring 0 --n-calls 50 --warm-start-offsets bo/warm_start/4-1_offsets.json --offset-margin 100
  ./venv/bin/python bo/run_offset_bo.py 5-1 --combine-best
  ./venv/bin/python bo/run_offset_bo.py 5-1 --validate-logs
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Agents paths
AGENTS = os.path.join(PROJECT_ROOT, "agents", "irregular")
DETECTION_PARAMS_DIR = os.path.join(AGENTS, "2_detection", "parameters")
BLOCKS = ["K", "B1", "B2", "A1", "A2", "A3", "A4"]
DEFAULT_OFFSET_MARGIN_PX = 200


def load_base_detection_params(tunnel_id: str) -> Dict:
    path = os.path.join(DETECTION_PARAMS_DIR, tunnel_id, "parameters_detection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base detection params not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_detection_params(tunnel_id: str, params: Dict) -> None:
    path = os.path.join(DETECTION_PARAMS_DIR, tunnel_id, "parameters_detection.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def build_params_from_sample(base: Dict, ring_idx: int, x: List[float]) -> Dict:
    """Build full detection params from 7D sample x (offsets only).

    x[0:7] = boundary offsets for K, B1, B2, A1, A2, A3, A4.
    All line detection and other params are copied from base unchanged.
    """
    params = dict(base)
    per_ring = dict(params.get("per_ring_offsets", {}))
    ring_key = str(ring_idx)
    if ring_key not in per_ring:
        per_ring[ring_key] = {b: 0.0 for b in BLOCKS}
    for i, block in enumerate(BLOCKS):
        per_ring[ring_key][block] = round(float(x[i]), 1)
    params["per_ring_offsets"] = per_ring
    return params


def _parse_detection_stdout(stdout: str) -> Dict[str, Optional[int]]:
    """Parse detection stdout for line counts (GT-free)."""
    out = {"n_positive_lines": None, "n_negative_lines": None}
    m = re.search(r"Lines:\s*\+(\d+)\s*-(\d+)", stdout)
    if m:
        out["n_positive_lines"] = int(m.group(1))
        out["n_negative_lines"] = int(m.group(2))
    return out


def collect_observables(tunnel_dir: str, detection_stdout: Optional[str] = None) -> Dict[str, Any]:
    """Collect GT-free observables from pipeline output files."""
    obs: Dict[str, Any] = {}

    if detection_stdout:
        parsed = _parse_detection_stdout(detection_stdout)
        obs["n_positive_lines"] = parsed["n_positive_lines"]
        obs["n_negative_lines"] = parsed["n_negative_lines"]
    else:
        obs["n_positive_lines"] = None
        obs["n_negative_lines"] = None

    detected_path = os.path.join(tunnel_dir, "detected.csv")
    if os.path.exists(detected_path):
        df = pd.read_csv(detected_path)
        obs["k_count"] = len(df)
        obs["k_confidence_mean"] = float(df["Confidence"].mean()) if "Confidence" in df.columns else None
        if len(df) >= 2 and "X" in df.columns:
            x = np.sort(df["X"].values)
            spacings = np.diff(x)
            obs["k_x_spacing_cv"] = float(np.std(spacings) / np.mean(spacings)) if np.mean(spacings) != 0 else 0.0
        else:
            obs["k_x_spacing_cv"] = None
    else:
        obs["k_count"] = None
        obs["k_confidence_mean"] = None
        obs["k_x_spacing_cv"] = None

    boundaries_path = os.path.join(tunnel_dir, "boundaries_per_ring.json")
    if os.path.exists(boundaries_path):
        with open(boundaries_path, "r") as f:
            bpr = json.load(f)
        ring_keys = sorted(bpr.keys(), key=int)
        blocks_per_ring = []
        for rk in ring_keys:
            val = bpr[rk]
            blist = val["boundaries"] if isinstance(val, dict) and "boundaries" in val else (val if isinstance(val, list) else [])
            blocks_per_ring.append(len(blist))
        obs["segment_count"] = sum(blocks_per_ring)
        obs["blocks_per_ring"] = blocks_per_ring
    else:
        obs["segment_count"] = None
        obs["blocks_per_ring"] = []

    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    img_height = None
    if os.path.exists(depth_path):
        img_height = int(np.load(depth_path).shape[0])
    if os.path.exists(boundaries_path) and img_height is not None:
        with open(boundaries_path, "r") as f:
            bpr = json.load(f)
        min_gap = None
        coverages = []
        for ring_key in sorted(bpr.keys(), key=int):
            val = bpr[ring_key]
            bounds = val["boundaries"] if isinstance(val, dict) and "boundaries" in val else (val if isinstance(val, list) else [])
            if not bounds:
                continue
            ys = [float(b["y"]) for b in bounds]
            n = len(ys)
            ring_slot_sum = 0
            for i in range(n):
                start = ys[i]
                end = ys[(i + 1) % n]
                if end > start:
                    gap = end - start
                else:
                    gap = (img_height - start) + end
                if min_gap is None or gap < min_gap:
                    min_gap = gap
                ring_slot_sum += gap
            coverages.append(ring_slot_sum / img_height)
        obs["boundary_min_gap_px"] = round(min_gap, 1) if min_gap is not None else None
        obs["boundary_coverage_pct"] = round(100.0 * np.mean(coverages), 2) if coverages else None
    else:
        obs["boundary_min_gap_px"] = None
        obs["boundary_coverage_pct"] = None

    groove_path = os.path.join(tunnel_dir, "groove_alignment.json")
    if os.path.exists(groove_path):
        with open(groove_path, "r") as f:
            g = json.load(f)
        obs["groove_alignment_pct"] = g.get("groove_alignment_pct")
    else:
        obs["groove_alignment_pct"] = None

    final_path = os.path.join(tunnel_dir, "final.csv")
    if os.path.exists(final_path):
        df = pd.read_csv(final_path)
        if "pred" in df.columns:
            pred = df["pred"].values
            n = len(pred)
            nonzero = np.sum(pred > 0)
            obs["pred_nonzero_pct"] = round(100.0 * nonzero / n, 2) if n else None
            unique, counts = np.unique(pred[pred > 0], return_counts=True)
            obs["pred_class_counts"] = {int(u): int(c) for u, c in zip(unique, counts)}
            total_seg = counts.sum()
            k_label = 1
            k_count = counts[unique == k_label].sum() if k_label in unique else 0
            obs["pred_k_fraction"] = round(float(k_count / total_seg), 4) if total_seg else None
        else:
            obs["pred_nonzero_pct"] = None
            obs["pred_class_counts"] = {}
            obs["pred_k_fraction"] = None
    else:
        obs["pred_nonzero_pct"] = None
        obs["pred_class_counts"] = {}
        obs["pred_k_fraction"] = None

    return obs


def run_pipeline(tunnel_id: str, base_dir: str) -> tuple:
    """Run detection -> segmentation -> evaluation. Returns (eval_results, detection_stdout)."""
    import subprocess
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    detection_stdout = ""
    r = subprocess.run(
        [sys.executable, os.path.join(AGENTS, "2_detection", "2_detection.py"), tunnel_id, "--data-dir", base_dir],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"detection failed: {r.stderr[:500]}")
    detection_stdout = r.stdout or ""
    r = subprocess.run(
        [sys.executable, os.path.join(AGENTS, "3_segmentation", "segmentation.py"), tunnel_id, "--data-dir", base_dir],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"segmentation failed: {r.stderr[:500]}")
    from agents.irregular import evaluation as ev
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = ev.evaluate(tunnel_id, base_dir=base_dir, segment_count=7)
    finally:
        sys.stdout = old_stdout
    return results, detection_stdout


def combine_best_and_evaluate(tunnel_id: str, base_dir: str, logs_dir: str, log_prefix: str = "offset_ring") -> None:
    """Load per-ring offset logs, merge best boundary_offsets per ring, run pipeline, print mIoU.
    Line detection params stay from current parameters_detection.json (unchanged).
    """
    base = load_base_detection_params(tunnel_id)
    merged = dict(base)
    merged["per_ring_offsets"] = {}
    n_rings = len(base.get("per_ring_offsets", {}))
    for ring_idx in range(n_rings):
        import glob
        pattern = os.path.join(logs_dir, f"{log_prefix}{ring_idx}_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No logs for ring {ring_idx} in {logs_dir}")
        best_miou = -1.0
        best_log = None
        for p in files:
            with open(p, "r") as f:
                log = json.load(f)
            if log.get("schema") != "intrinsic.v1":
                continue
            m = float(log.get("gt_metrics", {}).get("mIoU", -1))
            if m > best_miou:
                best_miou = m
                best_log = log
        if best_log is None:
            raise ValueError(f"No valid intrinsic.v1 log for ring {ring_idx}")
        merged["per_ring_offsets"][str(ring_idx)] = best_log["params"]["boundary_offsets"]
    save_detection_params(tunnel_id, merged)
    print("Merged best per-ring offsets (line params unchanged); running pipeline...")
    results, _ = run_pipeline(tunnel_id, base_dir)
    print(f"Combined best → mIoU={results['mIoU']:.4f} OA={results['OA']:.4f} F1={results['F1']:.4f}")


def validate_logs(tunnel_id: str, logs_dir: str, log_prefix: str = "offset_ring") -> None:
    """Verify offset BO logs have schema, observables, gt_metrics."""
    import glob
    required_keys = ["schema", "trial_id", "tunnel_id", "ring_idx", "params", "observables", "gt_metrics", "runtime_sec"]
    observable_keys = [
        "k_count", "segment_count", "blocks_per_ring",
        "pred_nonzero_pct", "pred_class_counts", "pred_k_fraction",
    ]
    files = sorted(glob.glob(os.path.join(logs_dir, f"{log_prefix}*_*.json")))
    if not files:
        print(f"No logs in {logs_dir}")
        return
    missing = []
    for p in files:
        with open(p, "r") as f:
            log = json.load(f)
        for k in required_keys:
            if k not in log:
                missing.append(f"{os.path.basename(p)}: missing '{k}'")
        if "params" in log and "boundary_offsets" not in log["params"]:
            missing.append(f"{os.path.basename(p)}: params missing 'boundary_offsets'")
        if "observables" in log:
            for k in observable_keys:
                if k not in log["observables"]:
                    missing.append(f"{os.path.basename(p)}: observables missing '{k}'")
    if missing:
        for m in missing[:20]:
            print(m)
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more")
        raise SystemExit(1)
    print(f"Validated {len(files)} logs: schema, observables, gt_metrics, boundary_offsets present.")


def main():
    parser = argparse.ArgumentParser(description="Per-ring offset BO (line params fixed, tune offsets only)")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 5-1)")
    parser.add_argument("--ring", type=int, default=None, help="Ring index (required for BO)")
    parser.add_argument("--n-calls", type=int, default=50, help="Number of BO trials")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--logs-dir", default=None, help="Logs directory (default: bo/logs/<tunnel_id>)")
    parser.add_argument("--log-prefix", default="offset_ring", help="Log file prefix (default: offset_ring)")
    parser.add_argument("--warm-start-offsets", metavar="PATH", default=None, help="JSON with per_ring_offsets to use as BO center (e.g. from bo/estimate_offsets_from_gt.py)")
    parser.add_argument("--offset-margin", type=float, default=None, help="Search half-width in px (default: 200, or 100 with --warm-start-offsets)")
    parser.add_argument("--combine-best", action="store_true", help="Combine best per-ring offsets and run final evaluation")
    parser.add_argument("--validate-logs", action="store_true", help="Verify logs for post-hoc intrinsic analysis")
    args = parser.parse_args()

    tunnel_id = args.tunnel_id
    base_dir = args.data_dir
    logs_dir = args.logs_dir or os.path.join(PROJECT_ROOT, "bo", "logs", tunnel_id)

    if args.combine_best:
        combine_best_and_evaluate(tunnel_id, base_dir, logs_dir, args.log_prefix)
        return
    if args.validate_logs:
        validate_logs(tunnel_id, logs_dir, args.log_prefix)
        return

    ring_idx = args.ring
    if ring_idx is None:
        parser.error("--ring is required for BO (or use --combine-best / --validate-logs)")
    os.makedirs(logs_dir, exist_ok=True)

    base = load_base_detection_params(tunnel_id)
    original_params = json.loads(json.dumps(base))

    if args.warm_start_offsets:
        with open(args.warm_start_offsets, "r") as f:
            warm = json.load(f)
        ws_offsets = warm.get("per_ring_offsets", warm)
        for rk, offs in ws_offsets.items():
            if rk in base.get("per_ring_offsets", {}):
                base["per_ring_offsets"][rk] = dict(offs)
        margin = args.offset_margin if args.offset_margin is not None else 100
        print(f"Warm start from {args.warm_start_offsets} (offset_margin={margin}px)")
    else:
        margin = args.offset_margin if args.offset_margin is not None else DEFAULT_OFFSET_MARGIN_PX

    ring_key = str(ring_idx)
    if ring_key not in base.get("per_ring_offsets", {}):
        raise ValueError(f"Ring {ring_idx} not in base per_ring_offsets")
    gt_offsets = base["per_ring_offsets"][ring_key]

    space = tuple(
        Real(gt_offsets[b] - margin, gt_offsets[b] + margin, name=f"off_{b}")
        for b in BLOCKS
    )

    from agents.irregular.evaluation import CLASS_NAMES_7
    trial_count = [0]

    def objective(x: List[float]) -> float:
        trial_count[0] += 1
        n = trial_count[0]
        t0 = time.perf_counter()
        params = build_params_from_sample(base, ring_idx, x)
        save_detection_params(tunnel_id, params)
        try:
            results, detection_stdout = run_pipeline(tunnel_id, base_dir)
        except Exception as e:
            print(f"Trial {n} pipeline failed: {e}", file=sys.stderr)
            return 1.0
        tunnel_dir = os.path.join(base_dir, tunnel_id)
        observables = collect_observables(tunnel_dir, detection_stdout)
        runtime_sec = time.perf_counter() - t0
        miou = float(results.get("mIoU", 0.0))
        iou_per_class = results.get("IoU_per_class", [])
        classes = results.get("classes", list(range(len(iou_per_class))))
        iou_per_class_dict = {CLASS_NAMES_7.get(int(c), str(c)): float(iou_per_class[i]) for i, c in enumerate(classes)}
        log = {
            "schema": "intrinsic.v1",
            "trial_id": f"offset_ring{ring_idx}_{n:03d}",
            "tunnel_id": tunnel_id,
            "ring_idx": ring_idx,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "params": {
                "boundary_offsets": {b: round(float(x[i]), 1) for i, b in enumerate(BLOCKS)},
            },
            "observables": observables,
            "gt_metrics": {
                "mIoU": miou,
                "OA": float(results.get("OA", 0)),
                "F1": float(results.get("F1", 0)),
                "iou_per_class": iou_per_class_dict,
            },
            "runtime_sec": round(runtime_sec, 2),
        }
        log_path = os.path.join(logs_dir, f"offset_ring{ring_idx}_{n:03d}.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Trial {n} mIoU={miou:.4f} -> {log_path}")
        return -miou

    n_initial = min(10, args.n_calls)
    print(f"Offset BO: tunnel={tunnel_id} ring={ring_idx} n_calls={args.n_calls} (line params fixed)")
    res = gp_minimize(
        objective, space,
        n_calls=args.n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=True,
    )
    save_detection_params(tunnel_id, original_params)
    print(f"Best -mIoU={res.fun:.4f} (mIoU={-res.fun:.4f})")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
