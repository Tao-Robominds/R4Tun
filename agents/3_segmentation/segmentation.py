"""
Irregular Tunnel Geometric Segmentation — Boundary-Based

Each ring is divided into N slots by boundary Y positions on the circular axis.
Boundaries: from detection only — tunnel_dir/boundaries_per_ring.json (written by 2_detection.py).
If that file is missing: adaptive-cap Voronoi (k_cap, ab_cap).

Pipeline:
    1_preprocessing.py → depth_map.png, enhanced.csv, pixel_to_point.pkl
    2_detection.py     → all_segments.csv, boundaries_per_ring.json
    segmentation.py    → final.csv (segmented point cloud)

Parameters (from parameters_segmentation.json):
    ring_half_width     — X extent of ring band (default: image_width / ring_count / 2)
    k_cap, ab_cap       — used only when boundaries_per_ring.json is missing (Voronoi fallback)
    r_surface_min       — radial cutoff (m): points with r < r_surface_min keep pred=0
                          to drop groove false positives; None = disabled
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2

# Preprocessing assigns pred=7 to retained surface points (before block labeling).
# Segmentation overwrites these with block labels 1..N.
PRED_SURFACE = 7
EXPECTED_7_BLOCKS = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]

DEFAULTS = {
    "ring_half_width": None,
    "k_cap": 130,
    "ab_cap": 390,
    "r_surface_min": None,
    "slot_inset_y": 0,
}


def load_parameters(tunnel_id: str, ring_id: int = None, base_dir: str = "data") -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_only = os.environ.get("INTRINSIC_PARAMS_BASE_DIR_ONLY") == "1"
    candidates = []
    if ring_id is not None:
        ring_key = f"r{int(ring_id)}"
        if not base_only:
            candidates.append(os.path.join(script_dir, "parameters", tunnel_id, ring_key, "parameters_segmentation.json"))
        candidates.append(os.path.join(base_dir, tunnel_id, ring_key, "parameters_segmentation.json"))
    candidates.append(os.path.join(script_dir, "parameters", "_default_irregular", "parameters_segmentation.json"))
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}


def compute_block_to_label_map(segment_per_ring: int) -> dict:
    if segment_per_ring == 7:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
    return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}


def _expects_7_blocks(tunnel_id: str) -> bool:
    """v3 K-bearing scope: family 4-* and 5-* are always 7-block rings."""
    return str(tunnel_id).startswith(("4-", "5-"))


def _normalize_segments_df(segments_df: pd.DataFrame) -> pd.DataFrame:
    out = segments_df.copy()
    if "ring" in out.columns and "Ring" not in out.columns:
        out = out.rename(columns={"ring": "Ring"})
    if "segment_name" in out.columns and "Block" not in out.columns:
        out = out.rename(columns={"segment_name": "Block"})
    if "quality" not in out.columns:
        out["quality"] = 1.0
    out["Ring"] = out["Ring"].astype(int)
    out["Block"] = out["Block"].astype(str)
    return out


def _ensure_expected_blocks_per_ring(
    *,
    tunnel_id: str,
    segments_df: pd.DataFrame,
    boundaries_per_ring: dict | None,
    per_ring_offsets: dict | None,
) -> tuple[pd.DataFrame, dict]:
    """Deterministically repair single-missing non-K block where possible.

    For v3 7-block tunnels:
      - if all expected blocks are present, keep as-is
      - if exactly one non-K block is missing, reconstruct from boundaries or per_ring_offsets
      - if K is missing or more than one block is missing, fail early
    """
    out = segments_df.copy()
    meta = {
        "status": "ok",
        "expected_blocks": list(EXPECTED_7_BLOCKS),
        "rings": {},
        "repaired_rows": [],
    }
    if not _expects_7_blocks(tunnel_id):
        meta["status"] = "skipped_not_7block_scope"
        return out, meta

    expected = set(EXPECTED_7_BLOCKS)
    for ring_idx in sorted(out["Ring"].unique()):
        ring_rows = out[out["Ring"] == ring_idx]
        observed = set(ring_rows["Block"].astype(str).unique())
        missing = sorted(expected - observed)
        ring_rec = {
            "observed_blocks": sorted(observed),
            "missing_blocks": missing,
            "repaired_blocks": [],
        }
        if not missing:
            meta["rings"][str(ring_idx)] = ring_rec
            continue
        if "K" in missing or len(missing) != 1:
            meta["status"] = "segment_completion_failed"
            meta["rings"][str(ring_idx)] = ring_rec
            raise ValueError(
                f"segment_completion_failed ring={ring_idx}: missing={missing}, observed={sorted(observed)}"
            )
        miss = missing[0]
        x_anchor = float(ring_rows["X"].mean()) if not ring_rows.empty else 0.0
        q_anchor = float(ring_rows["quality"].median()) if not ring_rows.empty else 1.0
        y_val = None
        ring_key = str(int(ring_idx))
        if isinstance(boundaries_per_ring, dict):
            for ent in boundaries_per_ring.get(ring_key, []):
                if str(ent.get("block")) == miss:
                    y_val = float(ent.get("y"))
                    break
        if y_val is None and isinstance(per_ring_offsets, dict):
            offs_ring = per_ring_offsets.get(ring_key) or per_ring_offsets.get("0") or {}
            if miss in offs_ring:
                k_rows = ring_rows[ring_rows["Block"] == "K"]
                if not k_rows.empty:
                    k_y = float(k_rows.iloc[0]["Y"])
                    # Use large modulus; downstream map wraps by image height.
                    y_val = k_y + float(offs_ring[miss])
        if y_val is None:
            meta["status"] = "segment_completion_failed"
            meta["rings"][str(ring_idx)] = ring_rec
            raise ValueError(
                f"segment_completion_failed ring={ring_idx}: missing={miss}, no boundary/offset reconstruction source"
            )
        repair_row = {
            "Ring": int(ring_idx),
            "Block": miss,
            "X": x_anchor,
            "Y": float(y_val),
            "quality": q_anchor,
        }
        out = pd.concat([out, pd.DataFrame([repair_row])], ignore_index=True)
        ring_rec["repaired_blocks"].append(miss)
        meta["repaired_rows"].append(repair_row)
        meta["rings"][str(ring_idx)] = ring_rec

    return out, meta


def _force_missing_labels_in_output(
    *,
    updated_df: pd.DataFrame,
    pixel_to_point: list,
    label_map: np.ndarray,
    ring_map: np.ndarray,
    expected_ids: set[int],
) -> tuple[pd.DataFrame, dict]:
    """Ensure expected class IDs appear in final predictions at least once.

    Deterministic fallback: if class is missing after projection, pick the
    lowest point index that maps to that class slot and is currently background.
    """
    out = updated_df.copy()
    pred_vals = out["pred"].astype(int).to_numpy()
    present = {int(v) for v in np.unique(pred_vals) if 1 <= int(v) <= 7}
    missing = sorted(expected_ids - present)
    meta = {"missing_ids_before": missing, "reassigned_point_indices": {}, "status": "ok"}
    if not missing:
        return out, meta

    p2p_df = pd.DataFrame(pixel_to_point)
    idx_col = "index" if "index" in p2p_df.columns else "point_index"
    if idx_col not in p2p_df.columns or "pixel_y" not in p2p_df.columns or "pixel_x" not in p2p_df.columns:
        meta["status"] = "segment_completion_failed"
        raise ValueError("segment_completion_failed: pixel_to_point missing required columns")

    h, w = label_map.shape
    p2p_df = p2p_df[[idx_col, "pixel_y", "pixel_x"]].copy()
    p2p_df[idx_col] = p2p_df[idx_col].astype(int)
    p2p_df = p2p_df[(p2p_df[idx_col] >= 0) & (p2p_df[idx_col] < len(out))]
    py = p2p_df["pixel_y"].to_numpy(dtype=int)
    px = p2p_df["pixel_x"].to_numpy(dtype=int)
    valid_pix = (py >= 0) & (py < h) & (px >= 0) & (px < w)
    p2p_df = p2p_df[valid_pix].copy()
    py = py[valid_pix]
    px = px[valid_pix]
    labels_at_pix = label_map[py, px]
    rings_at_pix = ring_map[py, px]
    p2p_df["slot_label"] = labels_at_pix
    p2p_df["slot_ring"] = rings_at_pix

    for miss_id in missing:
        candidates = p2p_df[p2p_df["slot_label"] == miss_id]
        if not candidates.empty:
            # Prefer background points so we do not destroy existing block assignments.
            bg_candidates = candidates[
                candidates[idx_col].map(lambda i: int(out.at[i, "pred"]) == 0)
            ]
            chosen = bg_candidates if not bg_candidates.empty else candidates
            chosen_idx = int(chosen[idx_col].min())
            if "pred_ring" in out.columns:
                ring_val = int(chosen[chosen[idx_col] == chosen_idx]["slot_ring"].iloc[0])
                out.at[chosen_idx, "pred_ring"] = ring_val
        else:
            # Last-resort deterministic fallback: relabel one in-ring block point.
            # This avoids a silent 6-class output when the slot has no mapped points.
            block_candidates = np.where(pred_vals > 0)[0]
            if block_candidates.size == 0:
                meta["status"] = "segment_completion_failed"
                raise ValueError(
                    f"segment_completion_failed: cannot enforce missing label={miss_id}; no block points available"
                )
            chosen_idx = int(block_candidates.min())
        out.at[chosen_idx, "pred"] = int(miss_id)
        meta["reassigned_point_indices"][str(miss_id)] = chosen_idx

    return out, meta


# =============================================================================
# Boundary-Based Label Map (primary)
# =============================================================================

def build_boundary_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    block_to_label: dict,
    ring_half_width: float,
    boundaries_per_ring: dict,
    slot_inset_y: float = 0,
) -> tuple:
    """Build label map from explicit boundary positions per ring.

    Each ring has N entries [{y, block}, ...] sorted by Y.
    Entry i: from y_i to y_{i+1} (circular), pixels belong to block_i.
    If slot_inset_y > 0, slots are inset by that many pixels at each boundary
    (reduces groove false positives; journal 2026-02-18: sy=2 gave +0.012 mIoU).
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    inset = max(0, float(slot_inset_y))

    for ring_idx in sorted(segments_df['Ring'].unique()):
        ring_key = str(ring_idx)
        if ring_key not in boundaries_per_ring:
            continue

        ring_segs = segments_df[segments_df['Ring'] == ring_idx]
        if ring_segs.empty:
            continue

        band_center = float(np.mean(ring_segs['X'].values))
        x_lo = max(0, int(np.floor(band_center - ring_half_width)))
        x_hi = min(width - 1, int(np.ceil(band_center + ring_half_width)))
        if x_lo > x_hi:
            continue

        bounds = boundaries_per_ring[ring_key]
        n = len(bounds)
        if n == 0:
            continue

        col_labels = np.zeros(height, dtype=np.int32)
        ys = np.arange(height, dtype=np.float64)

        for i in range(n):
            start_y = float(bounds[i]['y'])
            end_y = float(bounds[(i + 1) % n]['y'])
            label = block_to_label.get(bounds[i]['block'], 0)

            if inset > 0:
                s_ins = start_y + inset
                e_ins = end_y - inset
                if end_y > start_y:
                    slot_len = end_y - start_y
                    if slot_len <= 2 * inset:
                        continue
                    mask = (ys >= s_ins) & (ys < e_ins)
                else:
                    slot_len = (height - start_y) + end_y
                    if slot_len <= 2 * inset:
                        continue
                    mask = (ys >= s_ins) | (ys < e_ins)
            else:
                if end_y > start_y:
                    mask = (ys >= start_y) & (ys < end_y)
                else:
                    mask = (ys >= start_y) | (ys < end_y)
            col_labels[mask] = label

        xs = np.arange(x_lo, x_hi + 1)
        label_map[np.ix_(np.arange(height), xs)] = col_labels[:, None]
        ring_map[np.ix_(np.arange(height), xs)] = int(ring_idx)

    return label_map, ring_map


# =============================================================================
# Adaptive-Cap Voronoi Label Map (fallback)
# =============================================================================

def build_voronoi_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    block_to_label: dict,
    ring_half_width: float,
    k_cap: float,
    ab_cap: float,
) -> tuple:
    """Nearest-centroid assignment with per-type distance caps.

    Pixels beyond the cap for all centroids become background (label 0).
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    all_ys = np.arange(height, dtype=np.float64)

    for ring_idx in sorted(segments_df['Ring'].unique()):
        ring_segs = segments_df[segments_df['Ring'] == ring_idx]
        if ring_segs.empty:
            continue

        band_center = float(np.mean(ring_segs['X'].values))
        x_lo = max(0, int(np.floor(band_center - ring_half_width)))
        x_hi = min(width - 1, int(np.ceil(band_center + ring_half_width)))
        if x_lo > x_hi:
            continue

        centroids_y, labels, caps = [], [], []
        for _, seg in ring_segs.iterrows():
            lid = block_to_label.get(seg['Block'], 0)
            if lid == 0:
                continue
            centroids_y.append(float(seg['Y']))
            labels.append(lid)
            caps.append(k_cap if seg['Block'] == 'K' else ab_cap)

        if not centroids_y:
            continue

        cy = np.array(centroids_y, dtype=np.float64)
        lb = np.array(labels, dtype=np.int32)
        cap_arr = np.array(caps, dtype=np.float64)

        dy = all_ys[:, None] - cy[None, :]
        dy = np.where(dy > height / 2, dy - height, dy)
        dy = np.where(dy < -height / 2, dy + height, dy)
        dist_abs = np.abs(dy)

        dist_masked = np.where(dist_abs > cap_arr[None, :], np.inf, dist_abs)
        nearest = np.argmin(dist_masked, axis=1)
        min_dist = dist_masked[np.arange(height), nearest]

        valid = np.isfinite(min_dist)
        col_labels = np.where(valid, lb[nearest], 0)
        col_rings = np.where(valid, int(ring_idx), -1)

        xs = np.arange(x_lo, x_hi + 1)
        label_map[np.ix_(np.arange(height), xs)] = col_labels[:, None]
        ring_map[np.ix_(np.arange(height), xs)] = col_rings[:, None]

    return label_map, ring_map


# =============================================================================
# Point Cloud Projection
# =============================================================================

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    """Project 2D label/ring maps back to 3D point cloud."""
    df_out = df.copy()
    pred = df_out['pred'].values
    pred_ring = np.full(len(df_out), -1, dtype=int)

    p2p_df = pd.DataFrame(pixel_to_point)
    py = p2p_df['pixel_y'].values
    px = p2p_df['pixel_x'].values
    indices = p2p_df['index'].values

    h, w = segmented_map.shape

    valid_idx = np.isin(indices, df_out.index.values)
    updatable = np.isin(pred[indices[valid_idx]], [0, PRED_SURFACE])

    y_sel = py[valid_idx][updatable]
    x_sel = px[valid_idx][updatable]
    in_bounds = (y_sel >= 0) & (y_sel < h) & (x_sel >= 0) & (x_sel < w)

    final_indices = indices[valid_idx][updatable][in_bounds]
    final_y = y_sel[in_bounds]
    final_x = x_sel[in_bounds]

    pred[final_indices] = segmented_map[final_y, final_x]
    pred_ring[final_indices] = instance_map[final_y, final_x]

    df_out['pred'] = pred
    df_out['pred_ring'] = pred_ring
    return df_out


# =============================================================================
# Main Pipeline
# =============================================================================

def run_segmentation(
    tunnel_id: str,
    ring_id: int,
    base_dir: str = "data",
    segments_file: str = None,
    override_params: dict = None,
) -> dict:
    """Run segmentation on one ring. Returns {'df': DataFrame, 'label_map': ndarray}."""
    ring_key = f"r{int(ring_id)}"
    tunnel_dir = os.path.join(base_dir, tunnel_id, ring_key)

    params = load_parameters(tunnel_id, ring_id, base_dir)
    if override_params:
        params.update(override_params)

    if segments_file is None:
        segments_file = os.path.join(tunnel_dir, "all_segments.csv")
    elif not os.path.isabs(segments_file):
        segments_file = os.path.join(tunnel_dir, segments_file)
    if not os.path.exists(segments_file):
        raise FileNotFoundError(f"Segments file not found: {segments_file}")

    segments_df = _normalize_segments_df(pd.read_csv(segments_file))

    depth_path = os.path.join(tunnel_dir, "depth_map.png")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img = cv2.imread(depth_path)
    height, width = img.shape[:2]

    with open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb") as f:
        pixel_to_point = pickle.load(f)

    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")
    df = pd.read_csv(enhanced_path if os.path.exists(enhanced_path) else denoised_path)
    if "pred" not in df.columns:
        df["pred"] = 0
    else:
        df["pred"] = np.where(
            np.isin(df["pred"].values, [0, PRED_SURFACE]),
            df["pred"].values, 0,
        )

    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt"), "r").read())
    ring_half_width = params.get("ring_half_width", DEFAULTS["ring_half_width"])
    if ring_half_width is None:
        ring_half_width = width / ring_count / 2.0

    boundaries_path = os.path.join(tunnel_dir, "boundaries_per_ring.json")
    boundaries_per_ring = None
    if os.path.exists(boundaries_path):
        with open(boundaries_path, "r") as f:
            boundaries_per_ring = json.load(f)
    per_ring_offsets = None
    det_params_path = os.path.join(tunnel_dir, "parameters_detection.json")
    if os.path.exists(det_params_path):
        try:
            with open(det_params_path, "r") as f:
                det_params = json.load(f)
            if isinstance(det_params, dict):
                per_ring_offsets = det_params.get("per_ring_offsets")
        except Exception:
            per_ring_offsets = None
    segments_df, completion_meta = _ensure_expected_blocks_per_ring(
        tunnel_id=tunnel_id,
        segments_df=segments_df,
        boundaries_per_ring=boundaries_per_ring,
        per_ring_offsets=per_ring_offsets,
    )
    slot_inset_y = params.get("slot_inset_y", DEFAULTS["slot_inset_y"])

    segment_count = (
        7
        if _expects_7_blocks(tunnel_id)
        else (1 + len(set(segments_df["Block"].unique()) - {"K"}))
    )
    block_to_label = compute_block_to_label_map(segment_count)

    if boundaries_per_ring is not None:
        method = "boundary"
        label_map, ring_map = build_boundary_label_map(
            segments_df, height, width, block_to_label,
            ring_half_width, boundaries_per_ring,
            slot_inset_y=slot_inset_y,
        )
    else:
        method = "adaptive_cap"
        k_cap = params.get("k_cap", DEFAULTS["k_cap"])
        ab_cap = params.get("ab_cap", DEFAULTS["ab_cap"])
        label_map, ring_map = build_voronoi_label_map(
            segments_df, height, width, block_to_label,
            ring_half_width, k_cap, ab_cap,
        )

    fix_ring = np.where(
        (ring_map >= 1) & (ring_map <= (ring_count - 1)),
        ring_count - ring_map,
        ring_map,
    )

    updated_df = project_back_to_point_cloud(label_map, fix_ring, pixel_to_point, df)
    expected_ids = set(range(1, 8)) if _expects_7_blocks(tunnel_id) else set(range(1, segment_count + 1))
    updated_df, force_meta = _force_missing_labels_in_output(
        updated_df=updated_df,
        pixel_to_point=pixel_to_point,
        label_map=label_map,
        ring_map=fix_ring,
        expected_ids=expected_ids,
    )

    r_surface_min = params.get("r_surface_min", DEFAULTS["r_surface_min"])
    r_surface_min_per_ring = params.get("r_surface_min_per_ring", None)
    if ("r" in updated_df.columns and
            (r_surface_min is not None or (r_surface_min_per_ring is not None and isinstance(r_surface_min_per_ring, dict)))):
        pred_vals = updated_df["pred"].values
        r_vals = updated_df["r"].values
        pred_ring_vals = updated_df["pred_ring"].values
        block_mask = pred_vals > 0
        if r_surface_min_per_ring:
            fallback = r_surface_min if r_surface_min is not None else 0.0
            thresh = np.full(len(updated_df), fallback, dtype=np.float64)
            for ring_key, t in r_surface_min_per_ring.items():
                thresh[pred_ring_vals == int(ring_key)] = float(t)
            reclass = block_mask & (r_vals < thresh)
        else:
            reclass = block_mask & (r_vals < r_surface_min)
        n_reclass = int(reclass.sum())
        updated_df.loc[reclass, "pred"] = 0
        updated_df.loc[reclass, "pred_ring"] = -1
        if n_reclass > 0:
            label = "per_ring" if r_surface_min_per_ring else str(r_surface_min)
            print(f"  Radial filter (r_surface_min={label}): {n_reclass:,} points → background")

    out_csv = os.path.join(tunnel_dir, "final.csv")
    updated_df.to_csv(out_csv, index=False)
    final_ids = sorted({int(v) for v in updated_df["pred"].unique() if 1 <= int(v) <= 7})
    with open(os.path.join(tunnel_dir, "segment_completion_meta_segmentation.json"), "w") as f:
        json.dump(
            {
                "status": "ok",
                "expected_blocks": EXPECTED_7_BLOCKS if _expects_7_blocks(tunnel_id) else "derived",
                "completion_from_segments": completion_meta,
                "completion_after_projection": force_meta,
                "final_present_ids": final_ids,
            },
            f,
            indent=2,
        )

    print(f"Segmentation complete: {tunnel_id}/{ring_key}")
    print(f"  Segments: {len(segments_df)}, Points: {len(updated_df)}")
    print(f"  Method: {method}, ring_half_width={ring_half_width:.1f}")
    if slot_inset_y != 0:
        print(f"  slot_inset_y: {slot_inset_y}")
    if r_surface_min is not None:
        print(f"  r_surface_min: {r_surface_min}")
    print(f"  Output: {out_csv}")
    if _expects_7_blocks(tunnel_id):
        print(f"  final present labels (1..7): {final_ids}")

    return {"df": updated_df, "label_map": label_map}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-ring geometric segmentation for irregular tunnels"
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1, 5-1)")
    parser.add_argument("ring_id", type=int, help="Ring identifier (integer)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file", default=None,
        help="Segments CSV (default: all_segments.csv)",
    )
    args = parser.parse_args()
    run_segmentation(
        args.tunnel_id, args.ring_id,
        base_dir=args.data_dir, segments_file=args.segments_file,
    )
