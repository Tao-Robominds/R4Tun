"""
Generate detected.csv from ground truth (final.csv segment labels).

Uses K-block points (segment == 1), partitions by h (ring along tunnel),
computes (h, theta) centroids per ring, maps to depth-map pixel (X, Y).

Usage:
  python -m p4tun.generate_detected_from_gt 3-1 [--data-dir data] [--out detected.csv]
  python -m p4tun.generate_detected_from_gt --all [--data-dir data]   # all tunnels except sample
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


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
    """Map (h, theta) to depth-map pixel (X, Y). Linear mapping to [0, W-1] x [0, H-1]."""
    if h_max <= h_min or theta_max <= theta_min:
        return float(W - 1) / 2, float(H - 1) / 2
    x = (h - h_min) / (h_max - h_min) * (W - 1)
    y = (theta - theta_min) / (theta_max - theta_min) * (H - 1)
    return float(np.clip(x, 0, W - 1)), float(np.clip(y, 0, H - 1))


def _theta_to_pixel_y(theta: float, theta_min: float, theta_max: float, H: int) -> float:
    """Map theta to depth-map row Y only."""
    if theta_max <= theta_min:
        return float(H - 1) / 2
    y = (theta - theta_min) / (theta_max - theta_min) * (H - 1)
    return float(np.clip(y, 0, H - 1))


def _read_ring_count(tunnel_dir: str) -> int | None:
    """Read ring_count from tunnel_dir/ring_count.txt if present."""
    p = os.path.join(tunnel_dir, "ring_count.txt")
    if not os.path.exists(p):
        return None
    try:
        return int(open(p).read().strip())
    except (ValueError, OSError):
        return None


def generate_detected_from_gt(
    tunnel_id: str,
    base_dir: str = "data",
    n_segments: int | None = None,
    out_name: str = "detected.csv",
) -> pd.DataFrame:
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    final_path = os.path.join(tunnel_dir, "final.csv")

    if n_segments is None:
        n_segments = _read_ring_count(tunnel_dir) or 6

    df = pd.read_csv(final_path, usecols=["h", "theta", "segment"])
    df = df.dropna(subset=["segment"])
    k = df[df["segment"] == 1].copy()
    if k.empty:
        raise ValueError(f"No K-block points (segment==1) in {final_path}")

    h_min, h_max, theta_min, theta_max, H, W = load_bounds_and_shape(tunnel_dir)

    # Partition K-block points by h (ring along tunnel). Each ring has one K position.
    # Use detection-style evenly spaced X (ring centers) — matches SAM expects.
    # Y from GT: mean theta per ring → pixel Y.
    h_vals = k["h"].values
    order = np.argsort(h_vals)
    n = len(order)
    edges = np.linspace(0, n, n_segments + 1, dtype=int)

    rows = []
    for i in range(n_segments):
        lo, hi = edges[i], edges[i + 1]
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        sub = k.iloc[idx]
        # Use actual GT centroids for both X and Y
        h_mean = float(sub["h"].mean())
        theta_mean = float(sub["theta"].mean())
        # Map to pixel coordinates
        x = (h_mean - h_min) / (h_max - h_min) * (W - 1)
        y = (theta_mean - theta_min) / (theta_max - theta_min) * (H - 1)
        rows.append({"X": float(np.clip(x, 0, W - 1)), "Y": float(np.clip(y, 0, H - 1))})

    out = pd.DataFrame(rows)
    # Already ordered by ring index i; X increasing
    out = out.sort_values("X").reset_index(drop=True)

    # Assign Type based on Y position relative to median.
    # Keep actual GT-derived Y values per ring (no normalization).
    med_y = out["Y"].median()
    out["Type"] = np.where(out["Y"] <= med_y, "positive_slope", "negative_slope")

    out = out[["Type", "X", "Y"]]
    out_path = os.path.join(tunnel_dir, out_name)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} GT-derived K positions to {out_path}")
    return out


def _discover_tunnels(base_dir: str) -> list[str]:
    """List tunnel IDs under base_dir that have final.csv + depth_map_outlier.npy, excluding sample."""
    out = []
    for name in sorted(os.listdir(base_dir)):
        if name == "sample":
            continue
        d = os.path.join(base_dir, name)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "final.csv")) and os.path.exists(
            os.path.join(d, "depth_map_outlier.npy")
        ):
            out.append(name)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate detected.csv from GT (final.csv)")
    ap.add_argument("tunnel_id", nargs="?", help="Tunnel ID (e.g. 3-1). Omit if using --all.")
    ap.add_argument("--data-dir", default="data", help="Base data directory")
    ap.add_argument("--out", default="detected.csv", help="Output filename in tunnel dir")
    ap.add_argument("--segments", type=int, default=None, help="Number of segments (default: from ring_count.txt)")
    ap.add_argument("--all", action="store_true", help="Run for all tunnels except sample")
    args = ap.parse_args()

    if args.all:
        tunnels = _discover_tunnels(args.data_dir)
        if not tunnels:
            print("No tunnels found with final.csv + depth_map_outlier.npy")
            raise SystemExit(1)
        for tid in tunnels:
            try:
                generate_detected_from_gt(tid, base_dir=args.data_dir, n_segments=args.segments, out_name=args.out)
            except Exception as e:
                print(f"  Skip {tid}: {e}")
    elif args.tunnel_id:
        generate_detected_from_gt(
            args.tunnel_id,
            base_dir=args.data_dir,
            n_segments=args.segments,
            out_name=args.out,
        )
    else:
        ap.error("Provide tunnel_id or --all")
