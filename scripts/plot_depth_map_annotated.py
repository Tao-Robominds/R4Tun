"""
Plot depth map with GT K positions (4-1 style: R0K..R6K). Optionally overlay
detected K from a detected_k_*.csv in a different color.

Uses data/<tunnel_id>/depth_map.png as base if present, else builds from
depth_map_outlier.npy. Overlays K points from all_segments_gt.csv (Block=='K').
If --detected-csv is set, also plots those points (e.g. DBSCAN) in blue.
Saves data/<tunnel_id>/depth_map_annotated.png.

Usage:
  python plot_depth_map_annotated.py --tunnel 4-1 --data-dir data [--detected-csv data/4-1/detected_k_dbscan.csv]
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IRREGULAR_ROOT = os.path.dirname(SCRIPT_DIR)


def load_depth_as_rgb(tunnel_dir: str) -> np.ndarray:
    """Load depth map as RGB (H, W, 3). Prefer depth_map.png, else depth_map_outlier.npy."""
    png_path = os.path.join(tunnel_dir, "depth_map.png")
    npy_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if os.path.exists(png_path):
        img = cv2.imread(png_path)
        if img is None:
            raise RuntimeError(f"Failed to read {png_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if os.path.exists(npy_path):
        depth = np.load(npy_path)
        valid = np.isfinite(depth) & ~np.isnan(depth)
        if not np.any(valid):
            out = np.ones((*depth.shape, 3), dtype=np.uint8) * 255
            return out
        vmin, vmax = np.nanmin(depth[valid]), np.nanmax(depth[valid])
        if vmax <= vmin:
            vmax = vmin + 1
        norm = np.full_like(depth, np.nan, dtype=np.float64)
        norm[valid] = (depth[valid] - vmin) / (vmax - vmin)
        cmap = plt.cm.get_cmap("viridis")
        rgb = cmap(norm)
        rgb = (rgb[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 255
        return rgb
    raise FileNotFoundError(
        f"Need depth_map.png or depth_map_outlier.npy in {tunnel_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Plot depth map with GT K (and optional detected K)")
    parser.add_argument("--tunnel", default="4-1", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--detected-csv", default=None, help="Optional: path to detected_k_*.csv to overlay in blue")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        repo_root = os.path.dirname(os.path.dirname(IRREGULAR_ROOT))
        data_dir = os.path.join(repo_root, data_dir)

    tunnel_dir = os.path.join(data_dir, args.tunnel)
    gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    out_path = os.path.join(tunnel_dir, "depth_map_annotated.png")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT not found: {gt_path}")

    df = pd.read_csv(gt_path)
    k_df = df[df["Block"] == "K"].copy()
    k_df = k_df.sort_values("Ring").reset_index(drop=True)
    if len(k_df) == 0:
        raise ValueError(f"No K rows in {gt_path}")

    detected_df = None
    if args.detected_csv and os.path.exists(args.detected_csv):
        detected_df = pd.read_csv(args.detected_csv).sort_values("Ring").reset_index(drop=True)
    elif args.detected_csv:
        # Maybe path relative to tunnel_dir
        rel = os.path.join(tunnel_dir, os.path.basename(args.detected_csv))
        if os.path.exists(rel):
            detected_df = pd.read_csv(rel).sort_values("Ring").reset_index(drop=True)

    rgb = load_depth_as_rgb(tunnel_dir)
    h, w = rgb.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    for _, row in k_df.iterrows():
        x, y = float(row["X"]), float(row["Y"])
        ring = int(row["Ring"])
        label = f"R{ring}K"
        ax.plot(x, y, "o", color="red", markersize=10, markeredgecolor="black", markeredgewidth=1.5)
        ax.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", edgecolor="none"),
        )

    if detected_df is not None and len(detected_df) > 0:
        for _, row in detected_df.iterrows():
            x, y = float(row["X"]), float(row["Y"])
            ring = int(row["Ring"])
            label = f"D{ring}"
            ax.plot(x, y, "o", color="dodgerblue", markersize=10, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(
                label,
                (x, y),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="dodgerblue", edgecolor="none"),
            )

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()
    msg = f"Saved {out_path} ({len(k_df)} GT K"
    if detected_df is not None:
        msg += f", {len(detected_df)} detected K (blue)"
    msg += ")"
    print(msg)


if __name__ == "__main__":
    main()
