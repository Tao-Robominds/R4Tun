"""
Evidence for why mIoU is only 0.594 with GT centres.
Run: python p4tun/scripts/analyze_why_miou_0594.py 4-1 [--data-dir data]
"""
import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tunnel_id", default="4-1", nargs="?")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    base = os.path.join(args.data_dir, args.tunnel_id)
    final_path = os.path.join(base, "final_gt_centres.csv")
    if not os.path.exists(final_path):
        print(f"Not found: {final_path}. Run template with all_segments_gt.csv then cp final.csv final_gt_centres.csv")
        return

    df = pd.read_csv(final_path)
    gt = df["segment"].values.astype(np.float64)
    pr = df["pred"].values.astype(np.float64)
    valid = np.isfinite(gt) & np.isfinite(pr) & (gt >= 1) & (gt <= 7) & (pr >= 0) & (pr <= 7)
    gt_v = np.round(gt[valid]).astype(int)
    pr_v = np.round(pr[valid]).astype(int)
    n = len(gt_v)
    correct = (gt_v == pr_v).sum()
    wrong = n - correct
    pred0 = (pr_v[gt_v != pr_v] == 0).sum()
    print("1. GT block points (segment 1-7): correct vs wrong")
    print(f"   Total block points (valid): {n}")
    print(f"   Correct: {correct} ({100*correct/n:.1f}%)")
    print(f"   Wrong: {wrong} ({100*wrong/n:.1f}%)")
    print(f"   Of wrong, pred=0: {pred0} ({100*pred0/wrong:.1f}%)")
    print()

    enh = pd.read_csv(os.path.join(base, "enhanced.csv"))
    block_mask = (enh["segment"] >= 1) & (enh["segment"] <= 7)
    n_block = block_mask.sum()
    with open(os.path.join(base, "pixel_to_point.pkl"), "rb") as f:
        p2p = pickle.load(f)
    p2p_df = pd.DataFrame(p2p)
    indices_mapped = set(p2p_df["index"].astype(int).values)
    block_indices = set(np.where(block_mask)[0])
    unmapped_block = block_indices - indices_mapped
    mapped_pred0 = pred0 - len(unmapped_block)
    if mapped_pred0 < 0:
        mapped_pred0 = pred0
    print("2. Why pred=0 for block points?")
    print(f"   Unmapped (not in pixel_to_point): {len(unmapped_block)}")
    print(f"   Mapped but template assigned 0:   {mapped_pred0} (templates too small)")
    print()

    k_mask = (gt_v == 1)
    n_k = k_mask.sum()
    k_pr = pr_v[k_mask]
    k_correct = (k_pr == 1).sum()
    k_pred0 = (k_pr == 0).sum()
    print("3. K-block (worst IoU):")
    print(f"   K points: {n_k}, correct: {k_correct} ({100*k_correct/n_k:.1f}%), pred=0: {k_pred0} ({100*k_pred0/n_k:.1f}%)")
    params_path = os.path.join(os.path.dirname(__file__), "..", "parameters", args.tunnel_id, "parameters_geometric_template.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            p = json.load(f)
        k_h = p["K_half_height_pos"] + p["K_half_height_neg"]
        print(f"   K template total height: {k_h:.0f} px (GT median ~286 px -> template ~2.7x too small)")
    print()
    print("See data/4-1/WHY_MIOU_0594_WITH_GT_CENTRES.md for full evidence.")

if __name__ == "__main__":
    main()
