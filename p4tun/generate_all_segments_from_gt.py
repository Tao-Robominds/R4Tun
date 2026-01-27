"""
Generate all_segments.csv from ground truth (final.csv segment labels).

For each ring, computes centroids of each segment type (K, B1, A1, A2, A3, A4, B2)
and maps them to depth-map pixel coordinates (X, Y).

Usage:
  python -m p4tun.generate_all_segments_from_gt 4-1 [--data-dir data] [--out all_segments_gt.csv]
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

# Segment label to block name mapping
SEGMENT_TO_BLOCK = {
    1: 'K',
    2: 'B1',
    3: 'A1',
    4: 'A2',
    5: 'A3',
    6: 'A4',
    7: 'B2'
}

def load_bounds_and_shape(tunnel_dir: str) -> tuple[float, float, float, float, int, int]:
    """Get (h_min, h_max, theta_min, theta_max) from final.csv and depth map shape."""
    final_path = os.path.join(tunnel_dir, "final.csv")
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(final_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"Need {final_path} and {depth_path}")

    df = pd.read_csv(final_path, usecols=["h", "theta"])
    df = df.dropna()
    h_min, h_max = float(df["h"].min()), float(df["h"].max())
    theta_min, theta_max = float(df["theta"].min()), float(df["theta"].max())

    depth = np.load(depth_path)
    H, W = depth.shape[0], depth.shape[1]
    return h_min, h_max, theta_min, theta_max, H, W


def to_pixel(
    h: float,
    theta: float,
    h_min: float,
    h_max: float,
    theta_min: float,
    theta_max: float,
    W: int,
    H: int,
) -> tuple[float, float]:
    """Convert (h, theta) to pixel coordinates (X, Y)."""
    x = (h - h_min) / (h_max - h_min) * (W - 1)
    y = (theta - theta_min) / (theta_max - theta_min) * (H - 1)
    return float(np.clip(x, 0, W - 1)), float(np.clip(y, 0, H - 1))


def _read_ring_count(tunnel_dir: str) -> int | None:
    """Read ring count from ring_count.txt."""
    path = os.path.join(tunnel_dir, "ring_count.txt")
    if os.path.exists(path):
        try:
            return int(open(path, 'r').read().strip())
        except (ValueError, OSError):
            return None
    return None


def generate_all_segments_from_gt(
    tunnel_id: str,
    base_dir: str = "data",
    n_rings: int | None = None,
    out_name: str = "all_segments_gt.csv",
) -> pd.DataFrame:
    """Generate all_segments.csv from ground truth segment labels."""
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    final_path = os.path.join(tunnel_dir, "final.csv")

    if n_rings is None:
        n_rings = _read_ring_count(tunnel_dir)
        if n_rings is None:
            raise ValueError(f"Could not determine ring count for {tunnel_id}")

    # Load data
    df = pd.read_csv(final_path, usecols=["h", "theta", "segment"])
    df = df.dropna(subset=["segment"])
    
    # Filter to valid segments (1-7)
    df = df[df["segment"].isin(SEGMENT_TO_BLOCK.keys())]
    
    if df.empty:
        raise ValueError(f"No valid segment points in {final_path}")

    h_min, h_max, theta_min, theta_max, H, W = load_bounds_and_shape(tunnel_dir)

    # Partition points by h (ring along tunnel)
    h_vals = df["h"].values
    order = np.argsort(h_vals)
    n = len(order)
    edges = np.linspace(0, n, n_rings + 1, dtype=int)

    rows = []
    for ring_id in range(n_rings):
        lo, hi = edges[ring_id], edges[ring_id + 1]
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        
        ring_data = df.iloc[idx]
        
        # For each segment type, compute centroid
        for segment_label, block_name in SEGMENT_TO_BLOCK.items():
            segment_points = ring_data[ring_data["segment"] == segment_label]
            
            if len(segment_points) == 0:
                # Skip if no points for this segment in this ring
                continue
            
            # Compute centroid
            h_mean = float(segment_points["h"].mean())
            theta_mean = float(segment_points["theta"].mean())
            
            # Map to pixel coordinates
            x, y = to_pixel(h_mean, theta_mean, h_min, h_max, theta_min, theta_max, W, H)
            
            rows.append({
                "Ring": ring_id,
                "Block": block_name,
                "X": x,
                "Y": y,
                "quality": 1.0  # Ground truth, so quality is perfect
            })

    out = pd.DataFrame(rows)
    # Sort by Ring, then by Block order
    block_order = ['K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']
    out['Block'] = pd.Categorical(out['Block'], categories=block_order, ordered=True)
    out = out.sort_values(['Ring', 'Block']).reset_index(drop=True)
    
    out_path = os.path.join(tunnel_dir, out_name)
    out.to_csv(out_path, index=False)
    print(f"Generated {out_name} with {len(out)} segments for tunnel {tunnel_id}")
    
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate all_segments.csv from GT (final.csv)")
    ap.add_argument("tunnel_id", help="Tunnel ID (e.g. 4-1)")
    ap.add_argument("--data-dir", default="data", help="Base data directory")
    ap.add_argument("--out", default="all_segments_gt.csv", help="Output filename in tunnel dir")
    ap.add_argument("--rings", type=int, default=None, help="Number of rings (default: from ring_count.txt)")
    
    args = ap.parse_args()
    
    generate_all_segments_from_gt(
        args.tunnel_id,
        base_dir=args.data_dir,
        n_rings=args.rings,
        out_name=args.out,
    )
