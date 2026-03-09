"""
Re-generate all_segments_gt.csv from ground truth using the pipeline's own coordinates.

Uses unwrapped.csv (same (h, theta) as the depth map build) for segment/ring centroids,
and denoised.csv (pred!=0) extent for grid calibration so (h, theta) -> (X, Y) matches
the depth map. Derives true X, Y for each K and block central point.

Requires: data/<tunnel_id>/unwrapped.csv (h, theta, segment, ring),
          data/<tunnel_id>/denoised.csv (for grid extent),
          data/<tunnel_id>/depth_map*.npy or depth_map.png (for shape).

Usage:
  python build_all_segments_gt.py 4-1 [--data-dir data] [--output all_segments_gt.csv]
  Run from repo root with data-dir default "data", or pass path to data.
"""

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd

# 7-seg irregular: segment 1=K, 2=B1, 3=B2, 4=A1, 5=A2, 6=A3, 7=A4
SEGMENT_TO_BLOCK = {1: "K", 2: "B1", 3: "B2", 4: "A1", 5: "A2", 6: "A3", 7: "A4"}

DEFAULT_DEPTH_MAP_RESOLUTION = 0.005

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IRREGULAR_ROOT = os.path.dirname(SCRIPT_DIR)


def load_parameters_enhancing(tunnel_id: str, base_dir: str) -> dict:
    """Load resolution from preprocessing/enhancing parameters."""
    for path in [
        os.path.join(IRREGULAR_ROOT, "1_preprocessing", "parameters", tunnel_id, "parameters_preprocessing.json"),
        os.path.join(base_dir, tunnel_id, "parameters_preprocessing.json"),
        os.path.join(base_dir, tunnel_id, "parameters_enhancing.json"),
    ]:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    return {}


def get_depth_map_shape(tunnel_dir: str) -> tuple:
    """Return (height, width) of the tunnel depth map."""
    npy_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        return int(arr.shape[0]), int(arr.shape[1])
    png_path = os.path.join(tunnel_dir, "depth_map.png")
    if os.path.exists(png_path):
        import cv2
        img = cv2.imread(png_path)
        if img is not None:
            return img.shape[0], img.shape[1]
    raise FileNotFoundError(f"No depth map found in {tunnel_dir} (depth_map_outlier.npy or depth_map.png)")


def load_depth_map_grid(tunnel_dir: str, tunnel_id: str = None, base_dir: str = None) -> tuple:
    """
    Get (x_min, y_min, resolution) for (h, theta) -> pixel so all blocks fall inside the map.
    Prefer depth_map_grid.json (saved when depth map was built).
    Else infer from pixel_to_point + enhanced or denoised extent.
    Returns (x_min, y_min, resolution).
    """
    grid_path = os.path.join(tunnel_dir, "depth_map_grid.json")
    if os.path.exists(grid_path):
        with open(grid_path, "r") as f:
            grid = json.load(f)
        return float(grid["x_min"]), float(grid["y_min"]), float(grid["resolution"])

    resolution = DEFAULT_DEPTH_MAP_RESOLUTION
    if tunnel_id and base_dir:
        params_enh = load_parameters_enhancing(tunnel_id, base_dir)
        resolution = float(
            params_enh.get("depth_map_resolution") or
            (params_enh.get("depth_map") or {}).get("resolution") or
            DEFAULT_DEPTH_MAP_RESOLUTION
        )

    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    try:
        height, width = get_depth_map_shape(tunnel_dir)
    except FileNotFoundError:
        height, width = None, None
    if height is not None and width is not None and os.path.exists(unwrapped_path):
        uw = pd.read_csv(unwrapped_path, usecols=["h", "theta"])
        h_min, h_max = float(uw["h"].min()), float(uw["h"].max())
        t_min, t_max = float(uw["theta"].min()), float(uw["theta"].max())
        res_h = (h_max - h_min) / max(width - 1, 1)
        res_t = (t_max - t_min) / max(height - 1, 1)
        resolution = max(res_h, res_t, resolution)
        return h_min, t_min, resolution

    p2p_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    if os.path.exists(p2p_path) and os.path.exists(enhanced_path):
        with open(p2p_path, "rb") as f:
            pixel_to_point = pickle.load(f)
        df = pd.read_csv(enhanced_path)
        if "h" in df.columns and "theta" in df.columns:
            df = df.reset_index(drop=True)
            x_min_vals = []
            y_min_vals = []
            for rec in pixel_to_point:
                idx = rec.get("index")
                if idx is None or idx >= len(df):
                    continue
                h = df.loc[idx, "h"]
                theta = df.loc[idx, "theta"]
                if pd.isna(h) or pd.isna(theta):
                    continue
                px = rec.get("pixel_x")
                py = rec.get("pixel_y")
                if px is None or py is None:
                    continue
                x_min_vals.append(float(h) - float(px) * resolution)
                y_min_vals.append(float(theta) - float(py) * resolution)
            if x_min_vals and y_min_vals:
                return float(np.median(x_min_vals)), float(np.median(y_min_vals)), resolution
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")
    if not os.path.exists(denoised_path):
        raise FileNotFoundError(
            f"Need {grid_path}, or {p2p_path}+{enhanced_path}, or {denoised_path} for grid"
        )
    df = pd.read_csv(denoised_path, usecols=["h", "theta", "pred"])
    surface = df[df["pred"] != 0]
    if surface.empty:
        raise ValueError("denoised.csv has no points with pred!=0")
    x_min = float(surface["h"].min())
    y_min = float(surface["theta"].min())
    return x_min, y_min, resolution


def build_all_segments_gt(tunnel_id: str, data_dir: str = "data", out_path: str = None) -> pd.DataFrame:
    """
    Build all_segments_gt.csv from pipeline GT: unwrapped.csv (same (h, theta) as depth map).
    Grid from denoised (pred!=0) extent + resolution so we derive true X, Y for each K/block centre.
    """
    base_dir = os.path.abspath(data_dir)
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    if not os.path.exists(unwrapped_path):
        raise FileNotFoundError(f"Need {unwrapped_path} (run preprocessing first)")

    df = pd.read_csv(unwrapped_path, usecols=["h", "theta", "segment", "ring"])
    df_gt = df[(df["segment"] >= 1) & (df["segment"] <= 7)].copy()
    if df_gt.empty:
        raise ValueError("No points with segment in 1..7 in unwrapped.csv")

    x_min, y_min, resolution = load_depth_map_grid(tunnel_dir, tunnel_id=tunnel_id, base_dir=base_dir)

    rings_raw = sorted(df_gt["ring"].dropna().astype(int).unique())
    ring_to_idx = {r: i for i, r in enumerate(rings_raw)}

    rows = []
    for (seg_num, ring_raw), grp in df_gt.groupby(["segment", "ring"]):
        seg_int = int(seg_num)
        ring_int = int(ring_raw)
        block = SEGMENT_TO_BLOCK.get(seg_int)
        if block is None:
            continue
        ring_idx = ring_to_idx.get(ring_int)
        if ring_idx is None:
            continue
        h_mean = float(grp["h"].mean())
        theta_mean = float(grp["theta"].mean())
        pixel_x = (h_mean - x_min) / resolution
        pixel_y = (theta_mean - y_min) / resolution
        rows.append({
            "Ring": ring_idx,
            "Block": block,
            "X": round(float(pixel_x), 1),
            "Y": round(float(pixel_y), 1),
            "quality": 1.0,
        })

    out_df = pd.DataFrame(rows)
    block_order = ["K", "B1", "B2", "A1", "A2", "A3", "A4"]
    out_df["_ord"] = out_df["Block"].map({b: i for i, b in enumerate(block_order)})
    out_df = out_df.sort_values(["Ring", "_ord"]).drop(columns=["_ord"]).reset_index(drop=True)

    if out_path is None:
        out_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows, rings 0..{len(rings_raw)-1}) from {unwrapped_path}")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Build all_segments_gt.csv from ground truth data/<tunnel_id> (unwrapped.csv)"
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output", default=None, help="Output path (default: data/<tunnel_id>/all_segments_gt.csv)")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        repo_root = os.path.dirname(os.path.dirname(IRREGULAR_ROOT))
        data_dir = os.path.join(repo_root, data_dir)

    build_all_segments_gt(args.tunnel_id, data_dir=data_dir, out_path=args.output)


if __name__ == "__main__":
    main()
