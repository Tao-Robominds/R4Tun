"""
Diagnose segmentation gap for the best method (per-instance geometric).
Reports: per-class breakdown, unmapped GT points by block/ring, misclassification rates.
Usage: python p4tun/scripts/diagnose_segmentation_gap.py 4-1 [--data-dir data]
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tunnel_id", help="e.g. 4-1")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    p2p_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    final_path = os.path.join(tunnel_dir, "final.csv")

    df = pd.read_csv(enhanced_path)
    with open(p2p_path, "rb") as f:
        p2p = pickle.load(f)
    final = pd.read_csv(final_path)

    segment_col = df["segment"].values
    ring_col = df["ring"].values
    mapped_indices = set(r["index"] for r in p2p)
    pred = final["pred"].values
    if len(pred) != len(df):
        pred = np.resize(pred, len(df))  # align by index if needed
    gt = segment_col  # 0=bg, 1-7=blocks

    # GT block points only (segment 1-7)
    gt_mask = (gt >= 1) & (gt <= 7) & np.isfinite(gt)
    gt_indices = np.where(gt_mask)[0]
    gt_seg = gt[gt_mask].astype(int)
    gt_ring = ring_col[gt_mask]
    pred_gt = pred[gt_mask]

    unmapped = np.array([i not in mapped_indices for i in gt_indices])
    mapped = ~unmapped
    correct = mapped & (pred_gt == gt_seg)
    wrong = mapped & (pred_gt != gt_seg)

    seg_names = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
    n_total_gt = len(gt_indices)
    n_unmapped = unmapped.sum()
    n_mapped = mapped.sum()
    n_correct = correct.sum()
    n_wrong = wrong.sum()

    lines = [
        "# Segmentation gap diagnosis (4-1, per-instance geometric)",
        "",
        "## Summary",
        f"- Total GT block points: {n_total_gt}",
        f"- Unmapped (no pixel): {n_unmapped} ({100*n_unmapped/n_total_gt:.2f}%)",
        f"- Mapped: {n_mapped} ({100*n_mapped/n_total_gt:.2f}%)",
        f"- Mapped correct: {n_correct} ({100*n_correct/n_mapped:.2f}% of mapped)",
        f"- Mapped wrong: {n_wrong} ({100*n_wrong/n_mapped:.2f}% of mapped)",
        "",
        "## Unmapped GT points by block",
        "| Block | Unmapped | Total GT | % unmapped |",
        "|-------|----------|----------|------------|",
    ]
    for seg_id, name in enumerate(seg_names, start=1):
        m = gt_seg == seg_id
        u = unmapped[m].sum()
        t = m.sum()
        pct = 100 * u / t if t else 0
        lines.append(f"| {name} | {u} | {t} | {pct:.1f}% |")

    lines += [
        "",
        "## Unmapped GT points by ring",
        "| Ring | Unmapped | Total GT | % unmapped |",
        "|------|----------|----------|------------|",
    ]
    for r in sorted(np.unique(gt_ring)):
        m = gt_ring == r
        u = unmapped[m].sum()
        t = m.sum()
        pct = 100 * u / t if t else 0
        lines.append(f"| {r} | {u} | {t} | {pct:.1f}% |")

    lines += [
        "",
        "## Mapped-but-wrong by block (pred != GT)",
        "| Block | Wrong | Mapped | % wrong (of mapped) |",
        "|-------|-------|--------|---------------------|",
    ]
    for seg_id, name in enumerate(seg_names, start=1):
        m = (gt_seg == seg_id) & mapped
        w = (pred_gt != gt_seg)[m].sum()
        t = m.sum()
        pct = 100 * w / t if t else 0
        lines.append(f"| {name} | {w} | {t} | {pct:.1f}% |")

    out_path = os.path.join(tunnel_dir, "evaluation", "segmentation_gap_diagnosis.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
