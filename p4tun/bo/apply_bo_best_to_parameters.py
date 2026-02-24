#!/usr/bin/env python3
"""
Apply BO best parameters from p4tun/bo/results to p4tun/parameters/<tunnel_id>/,
then optionally run the full pipeline to verify we duplicate data/bo performance.

Usage:
  # Apply best params for tunnel 1-4 (detection + SAM from BO result JSONs)
  python -m p4tun.bo.apply_bo_best_to_parameters 1-4 --apply

  # Apply for multiple tunnels
  python -m p4tun.bo.apply_bo_best_to_parameters 1-4 2-2 --apply

  # Apply and run full pipeline for 1-4
  python -m p4tun.bo.apply_bo_best_to_parameters 1-4 --apply --run-pipeline
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Project root (parent of p4tun)
SCRIPT_DIR = Path(__file__).resolve().parent
P4TUN_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = P4TUN_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
PARAMS_BASE = P4TUN_DIR / "parameters"

# Canonical BO result files that produced data/bo performance (see results/README.md).
# Use these for detection so we pair with the SAM best that achieved the reported mIoU.
CANONICAL_DETECTION_FILES = {
    "1-4": "1-4_detection_20260126_125324.json",  # Pairs with 1-4_sam_20260126_best_extracted (mIoU 0.748)
    "2-2": "2-2_detection_20260122_101404.json",
}


def _load_bo_result(tunnel_id: str, stage: str) -> Optional[Tuple[float, Dict]]:
    """Load best score and best_params from BO result JSON. Use canonical files when set."""
    best_score = 0.0
    best_params = None

    # Detection: prefer canonical file so we match the run that achieved data/bo mIoU
    if stage == "detection" and tunnel_id in CANONICAL_DETECTION_FILES:
        path = RESULTS_DIR / CANONICAL_DETECTION_FILES[tunnel_id]
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                best_params = data.get("best_params")
                if best_params:
                    return (data.get("best_score", 0), best_params)
            except Exception:
                pass

    # SAM: prefer *_best_extracted.json (the run that achieved reported mIoU)
    if stage == "sam" and (RESULTS_DIR / f"{tunnel_id}_sam_20260126_best_extracted.json").exists():
        path = RESULTS_DIR / f"{tunnel_id}_sam_20260126_best_extracted.json"
        try:
            with open(path) as f:
                data = json.load(f)
            best_score = data.get("best_score", 0)
            best_params = data.get("best_params")
            if best_params:
                return (best_score, best_params)
        except Exception:
            pass
    pattern = f"{tunnel_id}_{stage}_*.json"
    for p in RESULTS_DIR.glob(pattern):
        if "_best_extracted" in p.name or "_history" in p.name or "proxy" in p.name or "no_gt" in p.name:
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            score = data.get("best_score", 0)
            params = data.get("best_params")
            if params and score > best_score:
                best_score = score
                best_params = params
        except Exception:
            continue
    return (best_score, best_params) if best_params else None


def _detection_dict_from_bo(best_params: Dict[str, Any]) -> Dict:
    from p4tun.bo.search_space import params_to_detection_dict

    names = list(best_params.keys())
    values = [best_params[n] for n in names]
    d = params_to_detection_dict(values, names)
    if "merge_close_threshold" in best_params:
        d.setdefault("line_processing", {})["merge_close_threshold"] = int(best_params["merge_close_threshold"])
    return d


def _sam_dict_from_bo(best_params: Dict[str, Any]) -> Dict:
    from p4tun.bo.search_space import params_to_sam_dict

    # 1-4 best has k_mask_height only; pipeline expects height_pos and height_neg
    bp = dict(best_params)
    if "k_mask_height" in bp and "k_mask_height_neg" not in bp:
        bp["k_mask_height_neg"] = float(bp["k_mask_height"])
    if "k_mask_height" in bp and "k_mask_height_pos" not in bp:
        bp["k_mask_height_pos"] = 619.16  # default

    names = list(bp.keys())
    values = [bp[n] for n in names]
    return params_to_sam_dict(values, names)


def apply_best_params(tunnel_id: str) -> bool:
    """Write BO best detection and SAM params to p4tun/parameters/<tunnel_id>/."""
    out_dir = PARAMS_BASE / tunnel_id
    out_dir.mkdir(parents=True, exist_ok=True)

    applied = False
    res_d = _load_bo_result(tunnel_id, "detection")
    if res_d:
        _, best_d = res_d
        det_dict = _detection_dict_from_bo(best_d)
        det_path = out_dir / "parameters_detection.json"
        with open(det_path, "w") as f:
            json.dump(det_dict, f, indent=2)
        print(f"  Wrote {det_path}")
        applied = True
    else:
        print(f"  No detection BO result for {tunnel_id}")

    res_s = _load_bo_result(tunnel_id, "sam")
    if res_s:
        _, best_s = res_s
        sam_dict = _sam_dict_from_bo(best_s)
        sam_path = out_dir / "parameters_sam.json"
        with open(sam_path, "w") as f:
            json.dump(sam_dict, f, indent=2)
        print(f"  Wrote {sam_path}")
        applied = True
    else:
        print(f"  No SAM BO result for {tunnel_id}")

    return applied


def run_pipeline(tunnel_id: str, base_dir: str = "data") -> bool:
    """Run preprocessing -> detection -> SAM -> evaluation. base_dir defaults to data."""
    base = Path(base_dir)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    base_str = str(base)
    # Do not overwrite data/bo (or any path containing /bo/) — it may hold best BO results.
    if "/bo" in base_str or base_str.rstrip("/").endswith("bo"):
        print("ERROR: Refusing to run pipeline with data-dir that contains 'bo' (e.g. data/bo).")
        print("       This would overwrite best BO results. Use a copy (e.g. data/bo_rerun) instead.")
        return False
    tunnel_dir = base / tunnel_id
    tunnel_dir.mkdir(parents=True, exist_ok=True)

    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    # 1_preprocessing.py has no --data-dir; uses default "data" relative to cwd
    steps = [
        ("Preprocessing", [str(P4TUN_DIR / "1_preprocessing.py"), tunnel_id]),
        ("Detection", [str(P4TUN_DIR / "4-1_detection.py"), tunnel_id, "--data-dir", base_str]),
        ("SAM", [str(P4TUN_DIR / "4-2_sam.py"), tunnel_id, "--data-dir", base_str]),
        ("Evaluation", [str(P4TUN_DIR / "evaluation.py"), tunnel_id, "--data-dir", base_str]),
    ]
    for label, cmd in steps:
        full_cmd = [str(venv_python)] + cmd
        print(f"\n--- {label} ---")
        r = subprocess.run(full_cmd, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            print(f"  Failed: {label}")
            return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply BO best params and optionally run pipeline")
    parser.add_argument("tunnels", nargs="+", help="Tunnel IDs (e.g. 1-4 2-2)")
    parser.add_argument("--apply", action="store_true", help="Write parameters to p4tun/parameters/<tunnel_id>/")
    parser.add_argument("--run-pipeline", action="store_true", help="Run preprocessing, detection, SAM, evaluation")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    args = parser.parse_args()

    if not args.apply and not args.run_pipeline:
        parser.print_help()
        print("\nUse --apply to write parameters and/or --run-pipeline to run the full pipeline.")
        return 0

    for tunnel_id in args.tunnels:
        print(f"\n{'='*60}\nTunnel {tunnel_id}\n{'='*60}")
        if args.apply:
            apply_best_params(tunnel_id)
        if args.run_pipeline:
            run_pipeline(tunnel_id, args.data_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
