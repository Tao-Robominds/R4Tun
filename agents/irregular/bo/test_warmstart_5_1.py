"""
Test all possible warm-start configurations for 5-1 SAM segmentation.

Runs each variant: writes parameters_sam.json, runs SAM, evaluates mIoU.
Results saved to data/5-1/warmstart_test/.

Usage:
  python test_warmstart_5_1.py [--variants "01_current,02_fyr,..."]
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SAM and evaluation
import importlib.util
sam_dir = PROJECT_ROOT / 'agents' / 'complex_staggered' / '3_segmentation'
sys.path.insert(0, str(sam_dir))
spec = importlib.util.spec_from_file_location("sam", sam_dir / "3_sam.py")
sam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sam_module)
run_sam = sam_module.run_sam

eval_dir = PROJECT_ROOT / 'agents' / 'complex_staggered'
spec = importlib.util.spec_from_file_location("evaluation", eval_dir / "evaluation.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
calculate_metrics = eval_module.calculate_metrics
get_class_names = eval_module.get_class_names
detect_segment_count = eval_module.detect_segment_count

import numpy as np
import pandas as pd


# Base params (required fields; B1/B2 from current)
BASE = {
    "resolution": 0.005,
    "b1_height_top": 1500,
    "b1_height_bottom_pos": 1540.69,
    "b1_height_bottom_neg": 1699.08,
    "b2_height_top_pos": 1540.69,
    "b2_height_top_neg": 1699.08,
    "b2_height_bottom": 1500,
    "use_quality_weighting": True,
}
WALK_ORDER = [["K", 0], ["B1", 1], ["A1", 1], ["A2", 1], ["A3", 1], ["A4", 1], ["B2", -1]]


def build_params(**overrides):
    """Merge BASE + WALK_ORDER + overrides."""
    p = dict(BASE)
    p["walk_order"] = WALK_ORDER
    p.update(overrides)
    return p


# Warm-start variants to test
VARIANTS = {
    "01_current": build_params(
        k_height=1113.62, ab_height=3262.27,
        segment_width=1288.96, angle_deg=6.14,
        k_mask_width=694.92, k_mask_height_pos=649.21, k_mask_height_neg=483.75,
        ab_mask_width=524.99, ab_mask_height=1583.70,
        padding=196, crop_margin=73,
        min_quality_threshold=0.46,
    ),
    "02_fyr": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=300, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "03_fyr_padding200": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "04_p4tun_masks": build_params(
        k_height=1113.62, ab_height=3262.27,
        segment_width=1288.96, angle_deg=6.14,
        k_mask_width=583, k_mask_height_pos=638, k_mask_height_neg=412,
        ab_mask_width=679, ab_mask_height=1574,
        padding=196, crop_margin=73,
        min_quality_threshold=0.46,
    ),
    "05_fyr_p4tun_masks": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=583, k_mask_height_pos=638, k_mask_height_neg=412,
        ab_mask_width=679, ab_mask_height=1574,
        padding=200, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "06_fyr_p4tun_full": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=583, k_mask_height_pos=638, k_mask_height_neg=412,
        ab_mask_width=679, ab_mask_height=1574,
        padding=300, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "07_report_aligned": build_params(
        k_height=1000, ab_height=3100,
        segment_width=1150, angle_deg=7.3,
        k_mask_width=583, k_mask_height_pos=638, k_mask_height_neg=412,
        ab_mask_width=679, ab_mask_height=1574,
        padding=150, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "08_defaults": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1200, angle_deg=7.5,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=150, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "09_fyr_angle73": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.3,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "10_fyr_seg1200": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1200, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "11_fyr_qual04": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.4,
    ),
    "12_fyr_qual02": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.2,
    ),
    "13_p4tun_geometry": build_params(
        k_height=1113.62, ab_height=3262.27,
        segment_width=1322.76, angle_deg=7.28,
        k_mask_width=583, k_mask_height_pos=638, k_mask_height_neg=412,
        ab_mask_width=679, ab_mask_height=1574,
        padding=200, crop_margin=50,
        min_quality_threshold=0.3,
    ),
    "14_fyr_crop75": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1300, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=75,
        min_quality_threshold=0.3,
    ),
    # Additional variants to close fyr gap (0.429)
    "15_no_quality_weight": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1200, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.3,
        use_quality_weighting=False,
    ),
    "16_qual01_permissive": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1200, angle_deg=7.52,
        k_mask_width=625, k_mask_height_pos=620, k_mask_height_neg=460,
        ab_mask_width=625, ab_mask_height=1620,
        padding=200, crop_margin=50,
        min_quality_threshold=0.1,
    ),
    "17_report_p4tun_qual01": build_params(
        k_height=1000, ab_height=3100,
        segment_width=1150, angle_deg=7.3,
        k_mask_width=583, k_mask_height_pos=638, k_mask_height_neg=412,
        ab_mask_width=679, ab_mask_height=1574,
        padding=150, crop_margin=50,
        min_quality_threshold=0.1,
    ),
    "18_hybrid_1175": build_params(
        k_height=1079.92, ab_height=3239.77,
        segment_width=1175, angle_deg=7.4,
        k_mask_width=600, k_mask_height_pos=630, k_mask_height_neg=440,
        ab_mask_width=650, ab_mask_height=1600,
        padding=200, crop_margin=50,
        min_quality_threshold=0.25,
    ),
}


def compute_miou(tunnel_id: str, tunnel_dir: str, segment_count: int) -> dict:
    """Compute mIoU from final.csv."""
    final_csv = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_csv):
        raise FileNotFoundError(f"final.csv not found at {final_csv}")
    df = pd.read_csv(final_csv)
    if "segment" not in df.columns or "pred" not in df.columns:
        raise ValueError("final.csv missing segment or pred column")
    gt = np.nan_to_num(df["segment"].values, nan=-1).astype(int)
    pred = np.nan_to_num(df["pred"].values, nan=-1).astype(int)
    class_names = get_class_names(segment_count)
    return calculate_metrics(gt, pred, class_names, segment_count)


def main():
    parser = argparse.ArgumentParser(description="Test SAM warm-start variants for 5-1")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variant names to run (default: all)")
    args = parser.parse_args()

    tunnel_id = "5-1"
    data_dir = "data"
    tunnel_dir = os.path.join(data_dir, tunnel_id)
    params_dir = PROJECT_ROOT / "agents" / "complex_staggered" / "3_segmentation" / "parameters" / tunnel_id
    out_dir = Path(tunnel_dir) / "warmstart_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    params_file = params_dir / "parameters_sam.json"
    backup_file = out_dir / "parameters_sam_backup.json"
    shutil.copy(params_file, backup_file)
    print(f"Backed up params to {backup_file}")

    segment_count = detect_segment_count(tunnel_dir, default=7)
    results = []

    variants_to_run = VARIANTS
    if args.variants:
        names = [s.strip() for s in args.variants.split(",")]
        variants_to_run = {k: v for k, v in VARIANTS.items() if k in names}
        if not variants_to_run:
            print(f"No matching variants. Available: {list(VARIANTS.keys())}")
            return {}

    import io
    from contextlib import redirect_stdout, redirect_stderr

    for name, params in variants_to_run.items():
        print(f"\n--- {name} ---")
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_sam(tunnel_id, data_dir)
            res = compute_miou(tunnel_id, tunnel_dir, segment_count)
            miou = res["mIoU"]
            print(f"  mIoU: {miou:.4f}  OA: {res['OA']:.4f}  F1: {res['F1']:.4f}")
            results.append({"variant": name, "mIoU": miou, "OA": res["OA"], "F1": res["F1"], "params": params})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"variant": name, "mIoU": 0.0, "OA": 0.0, "F1": 0.0, "error": str(e), "params": params})

    # Find best
    valid = [r for r in results if "error" not in r]
    best = max(valid, key=lambda r: r["mIoU"]) if valid else None

    # Save results
    report = {
        "tunnel_id": tunnel_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results,
        "best_variant": best["variant"] if best else None,
        "best_mIoU": best["mIoU"] if best else 0.0,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    # Update params with best
    if best:
        with open(params_file, "w") as f:
            json.dump(best["params"], f, indent=2)
        print(f"\n{'='*60}")
        print(f"BEST: {best['variant']}  mIoU={best['mIoU']:.4f}")
        print(f"Updated {params_file}")
    else:
        shutil.copy(backup_file, params_file)
        print("\nNo successful runs; restored backup.")

    # Summary table
    print(f"\n{'Variant':<25} {'mIoU':>8} {'OA':>8} {'F1':>8}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: -x["mIoU"]):
        err = " (ERROR)" if "error" in r else ""
        print(f"{r['variant']:<25} {r['mIoU']:>8.4f} {r['OA']:>8.4f} {r['F1']:>8.4f}{err}")

    return report


if __name__ == "__main__":
    main()
