"""SAM-stage hint modes for regular tunnels (1-*, 2-*)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from regular_hint_lib import gt_k_pixel_positions, gt_k_ring_table

SAM_HINT_MODES = (
    "off",
    "geometric",
    "geometric_gt_k",
    "geometric_gt_k_flip",
    "gt_theta",
    "oracle_k",
    "oracle_k_a2_a3",
    "oracle_swap",
    "oracle_blocks",
    "gt_ring_flip",
    "gt_handedness",
)


def compute_block_label(segment_per_ring: int) -> list[str]:
    labels = ["K", "B1"]
    labels += [f"A{i + 1}" for i in range(segment_per_ring - 3)]
    labels += ["B2"]
    return labels


def geometric_segment_regular(
    detected_df: pd.DataFrame,
    image_shape: tuple[int, int],
    ring_count: int,
    K_height: float,
    AB_height: float,
    segment_order: list[str],
    resolution: float,
    *,
    ring_flip: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-template labelling for 6-segment regular tunnels (no SAM masks).

    Circular tiling: K block + (n-1) AB bands along depth-map Y from detected K Y.
    """
    H, W = image_shape[:2]
    scale = resolution * 1000.0
    K_px = max(int(round(K_height / scale)), 1)
    AB_px = max(int(round(AB_height / scale)), 1)
    ring_w = W / float(ring_count)
    n_seg = len(segment_order)

    if n_seg < 2 or segment_order[0] != "K":
        raise ValueError(f"segment_order must start with K, got {segment_order}")

    block_to_label = {name: i + 1 for i, name in enumerate(segment_order)}
    downward_blocks = list(reversed(segment_order[1:]))

    label_map = np.zeros((H, W), dtype=int)
    ring_map = np.zeros((H, W), dtype=int)
    y_coords = np.arange(H, dtype=np.float64)

    if len(detected_df) != ring_count:
        raise ValueError(
            f"geometric_segment_regular: {len(detected_df)} rows vs ring_count={ring_count}"
        )

    for ring_idx, (_, row) in enumerate(detected_df.iterrows()):
        x0 = int(math.floor(ring_idx * ring_w))
        x1 = int(math.ceil(min((ring_idx + 1) * ring_w, W)))
        ky = float(row["Y"])
        flip = bool(ring_flip[ring_idx]) if ring_flip else False

        anchor = ky + K_px / 2.0 if not flip else ky - K_px / 2.0
        pos = (y_coords - anchor) % H
        if flip:
            pos = (anchor - y_coords) % H

        blocks = downward_blocks
        for i, block in enumerate(blocks):
            mask = (pos >= i * AB_px) & (pos < (i + 1) * AB_px)
            label_map[mask, x0:x1] = block_to_label[block]
            ring_map[mask, x0:x1] = ring_idx

        k_start = len(blocks) * AB_px
        k_mask = (pos >= k_start) & (pos < k_start + K_px)
        label_map[k_mask, x0:x1] = block_to_label["K"]
        ring_map[k_mask, x0:x1] = ring_idx

    return label_map, ring_map


def _signed_offset(theta: float, k: float, period: float) -> float:
    return ((theta - k + period / 2.0) % period) - period / 2.0


def ring_flip_flags_from_pred_gt(
    tunnel_dir: str | Path,
    n_rings: int,
) -> list[bool]:
    """
    Per-ring flip flags: True when pred block order is a mirror of GT (C4 swap).
    Uses pred_ring index 0..n-1 aligned with detected.csv row order.
    """
    path = Path(tunnel_dir) / "final.csv"
    df = pd.read_csv(path, usecols=["segment", "pred", "ring", "theta", "pred_ring"])
    df = df[np.isfinite(df["segment"]) & np.isfinite(df["pred"])]

    thetas = df["theta"].to_numpy()
    period = float(np.nanmax(thetas) - np.nanmin(thetas)) if len(thetas) else 10.0
    if period < 1e-3:
        period = 10.0

    def order_of(ring_df, col, k_theta):
        b = ring_df[ring_df[col] > 0]
        med = b.groupby(col)["theta"].median()
        offs = {int(c): _signed_offset(med[c], k_theta, period) for c in med.index}
        return [c for c, _ in sorted(offs.items(), key=lambda kv: kv[1])]

    def is_rotation(a, b):
        if sorted(a) != sorted(b) or not a:
            return False
        aa = a + a
        return any(aa[i : i + len(a)] == b for i in range(len(a)))

    flips: list[bool] = []
    for ring_idx in range(n_rings):
        g = df[df["pred_ring"] == ring_idx]
        if g.empty:
            g = df[df["ring"] == ring_idx]
        if g.empty:
            flips.append(False)
            continue
        kk = g[g["segment"] == 1]["theta"]
        if len(kk) == 0:
            flips.append(False)
            continue
        k_theta = float(kk.median())
        gt_o = order_of(g, "segment", k_theta)
        pr_o = order_of(g, "pred", k_theta)
        common = [c for c in gt_o if c in pr_o]
        gt_c = [c for c in gt_o if c in common]
        pr_c = [c for c in pr_o if c in common]
        if is_rotation(gt_c, pr_c):
            flips.append(False)
        elif is_rotation(gt_c[::-1], pr_c):
            flips.append(True)
        else:
            flips.append(False)
    return flips


def gt_handedness_flip_flags(tunnel_dir: str | Path, n_rings: int) -> list[bool]:
    """
    Per-ring flip flags from GT segment angular order vs ring-0 reference handedness.
    True when this ring's GT block order is a mirror rotation of ring 0 (continuous T3).
    """
    path = Path(tunnel_dir) / "final.csv"
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "pred_ring"])
    df = df[np.isfinite(df["segment"]) & (df["segment"] > 0)]

    thetas = df["theta"].to_numpy()
    period = float(np.nanmax(thetas) - np.nanmin(thetas)) if len(thetas) else 10.0
    if period < 1e-3:
        period = 10.0

    def order_of(ring_df, k_theta):
        med = ring_df.groupby("segment")["theta"].median()
        offs = {int(c): _signed_offset(med[c], k_theta, period) for c in med.index}
        return [c for c, _ in sorted(offs.items(), key=lambda kv: kv[1])]

    def is_rotation(a, b):
        if sorted(a) != sorted(b) or not a:
            return False
        aa = a + a
        return any(aa[i : i + len(a)] == b for i in range(len(a)))

    orders: list[list[int]] = []
    for ring_idx in range(n_rings):
        g = df[df["pred_ring"] == ring_idx]
        if g.empty:
            g = df[df["ring"] == ring_idx]
        if g.empty:
            orders.append([])
            continue
        kk = g[g["segment"] == 1]["theta"]
        if len(kk) == 0:
            orders.append([])
            continue
        k_theta = float(kk.median())
        orders.append(order_of(g, k_theta))

    ref = next((o for o in orders if len(o) >= 3), [])
    flips: list[bool] = []
    for o in orders:
        if not ref or not o or sorted(ref) != sorted(o):
            flips.append(False)
        elif is_rotation(ref, o):
            flips.append(False)
        elif is_rotation(ref[::-1], o):
            flips.append(True)
        else:
            flips.append(False)
    return flips


def gt_ring_flip_flags(tunnel_dir: str | Path, ring_ids: list) -> list[bool]:
    """Per-ring direction flip vs default geometric walk (from GT block order)."""
    path = Path(tunnel_dir) / "final.csv"
    df = pd.read_csv(path, usecols=["segment", "ring", "theta"])
    df = df[np.isfinite(df["segment"]) & np.isfinite(df["ring"])]

    thetas = df["theta"].to_numpy()
    period = float(np.nanmax(thetas) - np.nanmin(thetas)) if len(thetas) else 10.0
    if period < 1e-3:
        period = 10.0
    sector = period / 6.0

    def order_of(ring_df, col, k_theta):
        b = ring_df[ring_df[col] > 0]
        med = b.groupby(col)["theta"].median()
        offs = {int(c): _signed_offset(med[c], k_theta, period) for c in med.index}
        return [c for c, _ in sorted(offs.items(), key=lambda kv: kv[1])]

    def is_rotation(a, b):
        if sorted(a) != sorted(b) or not a:
            return False
        aa = a + a
        return any(aa[i : i + len(a)] == b for i in range(len(a)))

    flips: list[bool] = []
    for ring in ring_ids:
        g = df[df["ring"] == ring]
        kk = g[g["segment"] == 1]["theta"]
        if len(kk) == 0 or len(g[g["segment"] > 0]) < 3:
            flips.append(False)
            continue
        k_theta = float(kk.median())
        gt_o = order_of(g, "segment", k_theta)
        default_o = sorted(gt_o)
        flips.append(not is_rotation(gt_o, default_o) and is_rotation(gt_o[::-1], default_o))

    return flips


def gt_theta_label_map(
    tunnel_dir: str | Path,
    image_shape: tuple[int, int],
    ring_count: int,
    detected_df: pd.DataFrame,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign block labels from GT median theta per block per ring (theta hint)."""
    H, W = image_shape[:2]
    ring_w = W / float(ring_count)
    path = Path(tunnel_dir) / "final.csv"
    df = pd.read_csv(path, usecols=["segment", "ring", "theta", "h"])
    df = df[np.isfinite(df["segment"]) & np.isfinite(df["ring"])]

    label_map = np.zeros((H, W), dtype=int)
    ring_map = np.zeros((H, W), dtype=int)
    y_coords = np.arange(H, dtype=np.float64)

    ring_tbl = gt_k_ring_table(tunnel_dir)
    _, ys_gt, _ = gt_k_pixel_positions(tunnel_dir)

    for ring_idx in range(ring_count):
        if ring_idx >= len(ring_tbl):
            break
        ring_id = ring_tbl.iloc[ring_idx]["ring"]
        g = df[df["ring"] == ring_id]
        kk = g[g["segment"] == 1]["theta"]
        if len(kk) == 0:
            continue
        k_theta = float(kk.median())
        thetas = g["theta"].to_numpy()
        period = float(np.nanmax(thetas) - np.nanmin(thetas)) or 10.0

        block_medians: dict[int, float] = {}
        for seg in sorted(g["segment"].unique()):
            if seg <= 0:
                continue
            block_medians[int(seg)] = float(g[g["segment"] == seg]["theta"].median())

        x0 = int(math.floor(ring_idx * ring_w))
        x1 = int(math.ceil(min((ring_idx + 1) * ring_w, W)))
        ky = float(ys_gt[ring_idx]) if ring_idx < len(ys_gt) else float(detected_df.iloc[ring_idx]["Y"])

        # Map image Y to theta via linear ring unwrap (local)
        y_ring = y_coords[x0:x1] if x1 > x0 else y_coords
        theta_at_y = k_theta + (y_coords - ky) * (period / H)

        for yi in range(H):
            theta_y = float(theta_at_y[yi])
            best_seg = 0
            best_dist = float("inf")
            for seg, med in block_medians.items():
                d = abs(_signed_offset(theta_y, med, period))
                if d < best_dist:
                    best_dist = d
                    best_seg = seg
            if best_seg > 0:
                label_map[yi, x0:x1] = best_seg
                ring_map[yi, x0:x1] = ring_idx

    return label_map, ring_map


def apply_detected_gt_k_y(detected_df: pd.DataFrame, tunnel_dir: str | Path) -> pd.DataFrame:
    """Replace detected Y with GT-calibrated K pixel Y per ring."""
    _, ys_gt, _ = gt_k_pixel_positions(tunnel_dir)
    out = detected_df.copy().reset_index(drop=True)
    n = min(len(out), len(ys_gt))
    out.loc[: n - 1, "Y"] = ys_gt[:n]
    return out


def sam_hint_level_to_mode(level: str) -> str:
    mapping = {
        "S0": "off",
        "S1": "geometric",
        "S2": "geometric_gt_k",
        "S3": "geometric_gt_k_flip",
        "S4": "gt_theta",
        "S5a": "oracle_k_a2_a3",
        "S5b": "oracle_swap",
        "S5c": "oracle_k",
        "S5": "oracle_blocks",
        "T5": "gt_ring_flip",
    }
    return mapping.get(level, level)
