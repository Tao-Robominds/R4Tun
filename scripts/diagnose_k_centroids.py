#!/usr/bin/env python3
"""K centroid, taper-asymmetry, and pure-geometry mirror diagnostics for 3-1-1."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

K_LABEL = 1
Y_PASS_PX = 50.0
X_PASS_PX = 100.0

# Template asymmetry: left vertical span 2*619.16, right 2*460.77 mm
TEMPLATE_LEFT_SPAN_MM = 2 * 619.16
TEMPLATE_RIGHT_SPAN_MM = 2 * 460.77
TEMPLATE_ASYMMETRY_MM = TEMPLATE_LEFT_SPAN_MM - TEMPLATE_RIGHT_SPAN_MM


def mm_to_px(mm: float, resolution: float) -> float:
    return mm / (resolution * 1000)


def px_to_mm(px: float, resolution: float) -> float:
    return px * resolution * 1000


def fill_polygon(mask: np.ndarray, vertices: np.ndarray) -> None:
    path = MplPath(vertices)
    y_coords, x_coords = np.mgrid[: mask.shape[0], : mask.shape[1]]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
    mask[path.contains_points(points).reshape(mask.shape)] = 1


def k_vertices_real(cx_px: float, cy_px: float, resolution: float, mirror: bool = False) -> np.ndarray:
    x = cx_px * resolution * 1000
    y = cy_px * resolution * 1000
    verts = np.array(
        [
            [x - 625, y - 619.16],
            [x - 625, y + 619.16],
            [x + 625, y + 460.77],
            [x + 625, y - 460.77],
        ]
    )
    if mirror:
        verts[:, 0] = 2 * x - verts[:, 0]
    return verts / (resolution * 1000)


def build_gt_segment_map(
    pixel_to_point: list,
    segments: np.ndarray,
    shape: tuple[int, int],
    label: int,
) -> np.ndarray:
    h, w = shape
    gt = np.zeros((h, w), dtype=np.uint8)
    for entry in pixel_to_point:
        idx = int(entry["index"])
        if segments[idx] != label:
            continue
        py = int(entry["pixel_y"])
        px = int(entry["pixel_x"])
        if 0 <= py < h and 0 <= px < w:
            gt[py, px] = 1
    return gt


def ring_band_mask(
    shape: tuple[int, int],
    x_center: float,
    half_width: float,
) -> np.ndarray:
    h, w = shape
    xs = np.arange(w)
    band = (xs >= x_center - half_width) & (xs <= x_center + half_width)
    return np.tile(band, (h, 1))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def centroid_of_mask(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def vertical_span(mask: np.ndarray, x_lo: float, x_hi: float) -> float:
    sub = mask & (np.arange(mask.shape[1])[None, :] >= x_lo) & (
        np.arange(mask.shape[1])[None, :] < x_hi
    )
    ys, _ = np.where(sub)
    if len(ys) == 0:
        return 0.0
    return float(ys.max() - ys.min())


def rasterize_k_template(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    resolution: float,
    mirror: bool,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    verts = k_vertices_real(cx, cy, resolution, mirror=mirror)
    fill_polygon(mask, verts)
    return mask.astype(bool)


def run_diagnostics(
    tunnel_dir: Path,
    out_dir: Path,
    segment_width_mm: float = 1264.0,
    resolution: float = 0.005,
) -> dict:
    tunnel_dir = tunnel_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(tunnel_dir / "pixel_to_point.pkl", "rb") as f:
        pixel_to_point = pickle.load(f)
    final = pd.read_csv(tunnel_dir / "final.csv")
    segments = final["segment"].values
    depth = np.load(tunnel_dir / "depth_map.npy")
    h, w = depth.shape

    detected = pd.read_csv(tunnel_dir / "initial_points.csv")
    det_x = detected["X"].astype(float).values
    det_y = detected["Y"].astype(float).values
    n_rings = len(detected)

    gt_k = build_gt_segment_map(pixel_to_point, segments, (h, w), K_LABEL)
    half_width = mm_to_px(0.5 * segment_width_mm, resolution)

    ring_results = []
    geom_default_ious = []
    geom_mirror_ious = []

    for i in range(n_rings):
        band = ring_band_mask((h, w), det_x[i], half_width)
        gt_ring = gt_k & band
        gt_cent = centroid_of_mask(gt_ring)
        if gt_cent is None:
            ring_results.append(
                {
                    "ring": i,
                    "det_x": det_x[i],
                    "det_y": det_y[i],
                    "gt_x": None,
                    "gt_y": None,
                    "dx": None,
                    "dy": None,
                    "y_pass": False,
                    "x_pass": False,
                }
            )
            continue

        gx, gy = gt_cent
        dx = abs(det_x[i] - gx)
        dy = abs(det_y[i] - gy)

        # Taper asymmetry in pixel space
        span_left = vertical_span(gt_ring, -np.inf, gx)
        span_right = vertical_span(gt_ring, gx, np.inf)
        asym_px = span_left - span_right
        asym_mm = px_to_mm(asym_px, resolution)

        tmpl_default = rasterize_k_template((h, w), det_x[i], det_y[i], resolution, mirror=False)
        tmpl_mirror = rasterize_k_template((h, w), det_x[i], det_y[i], resolution, mirror=True)
        iou_def = mask_iou(tmpl_default & band, gt_ring)
        iou_mir = mask_iou(tmpl_mirror & band, gt_ring)
        geom_default_ious.append(iou_def)
        geom_mirror_ious.append(iou_mir)

        ring_results.append(
            {
                "ring": i,
                "det_x": float(det_x[i]),
                "det_y": float(det_y[i]),
                "gt_x": gx,
                "gt_y": gy,
                "dx": dx,
                "dy": dy,
                "y_pass": bool(dy < Y_PASS_PX),
                "x_pass": bool(dx < X_PASS_PX),
                "span_left_px": span_left,
                "span_right_px": span_right,
                "asymmetry_px": asym_px,
                "asymmetry_mm": asym_mm,
                "asymmetry_matches_template": bool(asym_px > 0),
                "geom_iou_default": iou_def,
                "geom_iou_mirror": iou_mir,
                "geom_mirror_better": bool(iou_mir > iou_def),
            }
        )

    valid = [r for r in ring_results if r.get("gt_x") is not None]
    y_errors = [r["dy"] for r in valid]
    x_errors = [r["dx"] for r in valid]
    gt_ys = [r["gt_y"] for r in valid]

    summary = {
        "tunnel_dir": str(tunnel_dir),
        "n_rings": n_rings,
        "half_width_px": half_width,
        "resolution": resolution,
        "segment_width_mm": segment_width_mm,
        "centroid": {
            "max_dy": max(y_errors) if y_errors else None,
            "mean_dy": float(np.mean(y_errors)) if y_errors else None,
            "max_dx": max(x_errors) if x_errors else None,
            "mean_dx": float(np.mean(x_errors)) if x_errors else None,
            "gt_y_std": float(np.std(gt_ys)) if gt_ys else None,
            "all_y_pass": bool(all(r["y_pass"] for r in valid)),
            "all_x_pass": bool(all(r["x_pass"] for r in valid)),
            "y_pass_threshold_px": Y_PASS_PX,
            "x_pass_threshold_px": X_PASS_PX,
        },
        "taper": {
            "template_asymmetry_mm": TEMPLATE_ASYMMETRY_MM,
            "rings_left_taller": sum(1 for r in valid if r["asymmetry_px"] > 0),
            "rings_right_taller": sum(1 for r in valid if r["asymmetry_px"] < 0),
            "rings_matching_template_sign": sum(1 for r in valid if r["asymmetry_matches_template"]),
            "mean_asymmetry_mm": float(np.mean([r["asymmetry_mm"] for r in valid])) if valid else None,
        },
        "geometry": {
            "mean_iou_default": float(np.mean(geom_default_ious)) if geom_default_ious else None,
            "mean_iou_mirror": float(np.mean(geom_mirror_ious)) if geom_mirror_ious else None,
            "rings_mirror_better": sum(1 for r in valid if r.get("geom_mirror_better")),
            "mirror_wins_majority": bool(
                sum(1 for r in valid if r.get("geom_mirror_better")) > len(valid) / 2
            ),
        },
        "per_ring": ring_results,
    }

    # Overlay: depth map + GT centroids + detected points
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    ax.imshow(depth, cmap="gray")
    for r in valid:
        ax.plot(r["gt_x"], r["gt_y"], "go", markersize=8)
        ax.plot(r["det_x"], r["det_y"], "y*", markersize=12)
        ax.plot([r["gt_x"], r["det_x"]], [r["gt_y"], r["det_y"]], "c-", alpha=0.5, lw=0.8)
    ax.set_title("GT K centroids (green) vs detected (yellow)")
    ax.invert_yaxis()

    ax = axes[0, 1]
    rings = [r["ring"] for r in valid]
    ax.bar(rings, [r["dy"] for r in valid], label="|dY|", alpha=0.7)
    ax.axhline(Y_PASS_PX, color="r", ls="--", label=f"Y pass < {Y_PASS_PX}px")
    ax.set_xlabel("Ring")
    ax.set_ylabel("px")
    ax.set_title("Per-ring |Y det - Y gt|")
    ax.legend()

    ax = axes[1, 0]
    ax.bar(rings, [r["asymmetry_mm"] for r in valid], alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(
        px_to_mm(TEMPLATE_ASYMMETRY_MM / (resolution * 1000) * 0, resolution),
        color="gray",
        ls=":",
        alpha=0,
    )
    ax.set_xlabel("Ring")
    ax.set_ylabel("mm (span_left - span_right)")
    ax.set_title(f"GT K taper asymmetry (template expects left taller, +{TEMPLATE_ASYMMETRY_MM:.0f}mm)")

    ax = axes[1, 1]
    w = 0.35
    xpos = np.array(rings)
    ax.bar(xpos - w / 2, [r["geom_iou_default"] for r in valid], w, label="default template")
    ax.bar(xpos + w / 2, [r["geom_iou_mirror"] for r in valid], w, label="mirrored template")
    ax.set_xlabel("Ring")
    ax.set_ylabel("IoU vs GT K")
    ax.set_title("Pure geometry IoU (no SAM)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "k_centroid_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Zoom on central seam region (rings 4-5)
    if len(valid) >= 6:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.imshow(depth, cmap="gray")
        x0 = min(valid[3]["det_x"], valid[4]["det_x"]) - 200
        x1 = max(valid[3]["det_x"], valid[4]["det_x"]) + 200
        y0 = min(r["gt_y"] for r in valid[3:6]) - 150
        y1 = max(r["gt_y"] for r in valid[3:6]) + 150
        for r in valid[3:6]:
            ax2.plot(r["gt_x"], r["gt_y"], "go", markersize=10)
            ax2.plot(r["det_x"], r["det_y"], "y*", markersize=14)
        ax2.set_xlim(x0, x1)
        ax2.set_ylim(y1, y0)
        ax2.set_title("Zoom: rings 3-5 (θ seam region)")
        fig2.savefig(out_dir / "k_centroid_zoom_seam.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    with open(out_dir / "k_centroid_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tunnel_dir",
        type=Path,
        help="Tunnel data directory (e.g. sam4tun/data/3-1-1)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/<tunnel_id>_k_centroid_check)",
    )
    parser.add_argument("--segment-width", type=float, default=1264.0)
    parser.add_argument("--resolution", type=float, default=0.005)
    args = parser.parse_args()

    tunnel_dir = args.tunnel_dir.resolve()
    tunnel_id = tunnel_dir.name
    out_dir = args.out_dir or (Path("data") / f"{tunnel_id}_k_centroid_check")

    summary = run_diagnostics(
        tunnel_dir,
        out_dir,
        segment_width_mm=args.segment_width,
        resolution=args.resolution,
    )

    c = summary["centroid"]
    g = summary["geometry"]
    t = summary["taper"]
    print(f"Centroid gate: max_dY={c['max_dy']:.1f}px mean_dY={c['mean_dy']:.1f}px all_y_pass={c['all_y_pass']}")
    print(f"Taper: {t['rings_matching_template_sign']}/{len(summary['per_ring'])} rings left-taller (template sign)")
    print(f"Geometry: mean IoU default={g['mean_iou_default']:.3f} mirror={g['mean_iou_mirror']:.3f} mirror_wins={g['rings_mirror_better']}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
