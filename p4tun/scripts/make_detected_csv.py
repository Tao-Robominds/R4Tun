"""
Generate detected.csv for 4-2_sam.py (K-only, one row per ring).
Usage:
  python -m p4tun.scripts.make_detected_csv 4-1 --from-gt [--data-dir data]
  python -m p4tun.scripts.make_detected_csv 4-1 --from-k-csv detected_k_dbscan.csv [--data-dir data]
"""
import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Write detected.csv (X, Y, quality) for 4-2_sam.py")
    parser.add_argument("tunnel_id", help="e.g. 4-1")
    parser.add_argument("--data-dir", default="data", help="Base data dir")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-gt", action="store_true", help="Extract K rows from all_segments_gt.csv")
    group.add_argument("--from-k-csv", metavar="FILE", help="Use this CSV (e.g. detected_k_dbscan.csv); needs X, Y, optional Confidence")
    args = parser.parse_args()

    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    out_path = os.path.join(tunnel_dir, "detected.csv")

    if args.from_gt:
        gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(gt_path)
        df = pd.read_csv(gt_path)
        if "Block" not in df.columns and "segment_name" in df.columns:
            df = df.rename(columns={"segment_name": "Block"})
        k = df[df["Block"].str.upper() == "K"].copy()
        k = k.sort_values("Ring").reset_index(drop=True)
        out = k[["X", "Y"]].copy()
        out["quality"] = k["quality"] if "quality" in k.columns else 1.0
    else:
        k_path = os.path.join(tunnel_dir, args.from_k_csv) if not os.path.isabs(args.from_k_csv) else args.from_k_csv
        if not os.path.exists(k_path):
            raise FileNotFoundError(k_path)
        df = pd.read_csv(k_path)
        if "Type" in df.columns and (df["Type"].str.upper() == "K").any():
            df = df[df["Type"].str.upper() == "K"]
        out = df[["X", "Y"]].copy()
        out["quality"] = df["Confidence"] if "Confidence" in df.columns else 0.9

    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
