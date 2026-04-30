"""First-principles GT-detection ceiling for one ring.

This is the ceiling of "if we had ground-truth boundaries on the 2D
depth map and we just back-project them to the 3D point cloud". It
deliberately does NOT use preprocessing (no denoising, no enhancement)
and does NOT use any detection result — the raw ring is unwrapped
directly, a per-pixel dominant-GT labelmap is built, then every raw
point is labelled by its own pixel.

Inputs (per ring):

    {data_dir}/{tunnel_id}/r{ring_id}/{tunnel_id}_r{ring_id}.txt
    columns: x y z intensity segment ring   (whitespace-separated)

Outputs (per ring):

    {data_dir}/{tunnel_id}/r{ring_id}/gt_ceiling/labelmap.npy
    {data_dir}/{tunnel_id}/r{ring_id}/gt_ceiling/final.csv
    {data_dir}/{tunnel_id}/r{ring_id}/gt_ceiling/performance.md

Aggregate:

    {data_dir}/gt_ceiling_results.json

Run with the project venv only:

    ./venv/bin/python methods/ablation/scripts/run_gt_ceiling.py \\
        --data-dir data/ablation
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score


CLASS_NAMES = {
    0: "Background",
    1: "K-block",
    2: "B1-block",
    3: "A1-block",
    4: "A2-block",
    5: "A3-block",
    6: "A4-block",
    7: "B2-block",
}
ALL_CLASSES = list(range(8))


def fit_circle_xz(x: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    n = len(x)
    if n > 50000:
        idx = np.linspace(0, n - 1, 50000).astype(int)
        x = x[idx]
        z = z[idx]
    A = np.column_stack([2.0 * x, 2.0 * z, np.ones_like(x)])
    b = x * x + z * z
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cz, c = float(sol[0]), float(sol[1]), float(sol[2])
    R = float(np.sqrt(max(c + cx * cx + cz * cz, 0.0)))
    return cx, cz, R


def unwrap_ring(df: pd.DataFrame, tunnel_diameter: float) -> pd.DataFrame:
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    z = df["z"].to_numpy(dtype=np.float64)
    cx, cz, _R = fit_circle_xz(x, z)
    theta_deg = (np.degrees(np.arctan2(z - cz, x - cx)) + 90.0) % 360.0
    theta = theta_deg * (np.pi * tunnel_diameter / 360.0)
    h = y - y.min()
    out = df.copy()
    out["theta"] = theta.astype(np.float32)
    out["h"] = h.astype(np.float32)
    return out


def build_per_pixel_labelmap(
    df: pd.DataFrame, resolution: float, tunnel_diameter: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build a per-pixel dominant-GT labelmap.

    Pixel-y indexes the cyclic theta-arc axis; the labelmap height is
    locked to the full circumference (pi * tunnel_diameter) so that
    boundaries stay consistent across rings of the same tunnel family.
    Pixel-x indexes the h axis (per-ring origin at h=0).
    """
    H_full = int(round(np.pi * tunnel_diameter / resolution))
    h_min = float(df["h"].min())
    h_max = float(df["h"].max())
    W = max(1, int(round((h_max - h_min) / resolution)) + 1)

    gy = np.clip((df["theta"].to_numpy() / resolution).astype(np.int64), 0, H_full - 1)
    gx = np.clip(((df["h"].to_numpy() - h_min) / resolution).astype(np.int64), 0, W - 1)
    seg = df["segment"].to_numpy(dtype=np.int64)
    pix_id = gy * W + gx

    mixing_pixels = 0
    n_classes = max(int(seg.max()), 7) + 1
    counts = np.zeros((H_full * W, n_classes), dtype=np.int32)
    np.add.at(counts, (pix_id, seg), 1)
    occupied = counts.sum(axis=1) > 0
    multi = (counts > 0).sum(axis=1) > 1
    mixing_pixels = int(multi.sum())

    dominant_flat = counts.argmax(axis=1).astype(np.int16)
    dominant_flat[~occupied] = 0
    labelmap = dominant_flat.reshape(H_full, W)

    return labelmap, gy.astype(np.int32), gx.astype(np.int32), float(mixing_pixels), float(occupied.sum())


def evaluate_labels(gt: np.ndarray, pred: np.ndarray) -> Dict:
    iou = jaccard_score(gt, pred, average=None, labels=ALL_CLASSES, zero_division=0)
    miou = float(np.mean(iou))
    oa = float(accuracy_score(gt, pred))
    f1 = float(f1_score(gt, pred, average="macro", labels=ALL_CLASSES, zero_division=0))
    return {
        "mIoU": miou,
        "OA": oa,
        "F1_macro": f1,
        "IoU_per_class": {int(c): float(v) for c, v in zip(ALL_CLASSES, iou)},
    }


def write_performance_md(
    out_dir: Path,
    tunnel_id: str,
    ring_id: int,
    metrics: Dict,
    n_points: int,
    n_pixels_total: int,
    n_pixels_mixed: int,
    h_w: Tuple[int, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["| class | IoU |", "|---|---:|"]
    for c in ALL_CLASSES:
        rows.append(f"| {CLASS_NAMES[c]} | {metrics['IoU_per_class'][c]:.3f} |")
    pix_mix_pct = (n_pixels_mixed / n_pixels_total * 100) if n_pixels_total else 0.0
    md = f"""# GT-detection ceiling — `{tunnel_id}/r{ring_id}`

First-principles ceiling: per-pixel dominant GT labelmap, back-projected
to every raw point. Preprocessing and detection are bypassed.

## Headline

| metric | value |
|---|---:|
| n raw points | {n_points:,} |
| depth-map size (theta x h) | {h_w[0]:,} x {h_w[1]:,} |
| occupied pixels | {n_pixels_total:,} |
| mixed-GT pixels (≥2 segments) | {n_pixels_mixed:,} ({pix_mix_pct:.2f}%) |
| mIoU | **{metrics['mIoU']:.4f}** |
| OA | {metrics['OA']:.4f} |
| F1 (macro) | {metrics['F1_macro']:.4f} |

## Per-class IoU

{chr(10).join(rows)}
"""
    (out_dir / "performance.md").write_text(md)


def run_one(
    tunnel_id: str,
    ring_id: int,
    data_dir: str,
    resolution: float,
    tunnel_diameter: float,
) -> Dict:
    unit_dir = Path(data_dir) / tunnel_id / f"r{int(ring_id)}"
    raw_path = unit_dir / f"{tunnel_id}_r{int(ring_id)}.txt"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    df = pd.read_csv(
        raw_path, sep=r"\s+", header=None,
        names=["x", "y", "z", "intensity", "segment", "ring"],
        engine="c",
        dtype={
            "x": "float32", "y": "float32", "z": "float32",
            "intensity": "float32", "segment": "int16", "ring": "int32",
        },
    )
    n_points = len(df)
    df = unwrap_ring(df, tunnel_diameter)

    labelmap, gy, gx, n_mixed, n_occupied = build_per_pixel_labelmap(
        df, resolution=resolution, tunnel_diameter=tunnel_diameter
    )

    pred = labelmap[gy, gx].astype(np.int64)
    gt = df["segment"].astype(np.int64).to_numpy()
    metrics = evaluate_labels(gt, pred)

    out_dir = unit_dir / "gt_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "labelmap.npy", labelmap)
    final_csv = pd.DataFrame({
        "segment": gt,
        "pred": pred,
        "gy": gy,
        "gx": gx,
    })
    final_csv.to_csv(out_dir / "final.csv", index=False)

    write_performance_md(
        out_dir, tunnel_id, int(ring_id), metrics,
        n_points=n_points, n_pixels_total=int(n_occupied),
        n_pixels_mixed=int(n_mixed), h_w=labelmap.shape,
    )

    summary = {
        "tunnel_id": tunnel_id,
        "ring_id": int(ring_id),
        "n_points": n_points,
        "labelmap_shape": [int(labelmap.shape[0]), int(labelmap.shape[1])],
        "occupied_pixels": int(n_occupied),
        "mixed_gt_pixels": int(n_mixed),
        "mixing_fraction": (float(n_mixed) / float(n_occupied)) if n_occupied else 0.0,
        **metrics,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ablation")
    p.add_argument("--panel", default=None,
                   help="Panel JSON (default: <data-dir>/reference_panel.json)")
    p.add_argument("--resolution", type=float, default=0.005,
                   help="Depth-map resolution in metres (default 0.005)")
    p.add_argument("--tunnel-diameter", type=float, default=7.5,
                   help="Tunnel diameter in metres (default 7.5)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Optional list of <tid>:<rid> to run a subset")
    args = p.parse_args()

    panel_path = Path(args.panel) if args.panel else Path(args.data_dir) / "reference_panel.json"
    panel = json.loads(panel_path.read_text())
    rings = panel.get("rings", [])
    if args.only:
        wanted = set()
        for tok in args.only:
            tid, _, rid = tok.partition(":")
            wanted.add((tid.strip(), int(rid)))
        rings = [r for r in rings if (r["tunnel_id"], int(r["ring_id"])) in wanted]

    results: List[Dict] = []
    for r in rings:
        try:
            print("=" * 70)
            print(f"[gt-ceiling] {r['tunnel_id']}/r{r['ring_id']}")
            summary = run_one(
                r["tunnel_id"], int(r["ring_id"]),
                data_dir=args.data_dir,
                resolution=args.resolution,
                tunnel_diameter=args.tunnel_diameter,
            )
            print(
                f"[gt-ceiling] mIoU={summary['mIoU']:.4f} "
                f"OA={summary['OA']:.4f} F1={summary['F1_macro']:.4f} "
                f"mix={summary['mixing_fraction']*100:.2f}%"
            )
            results.append(summary)
        except Exception as e:  # noqa: BLE001
            print(f"[gt-ceiling] FAILED {r['tunnel_id']}/r{r['ring_id']}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append({
                "tunnel_id": r["tunnel_id"], "ring_id": int(r["ring_id"]),
                "error": str(e),
            })

    aggregate = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "panel": str(panel_path.resolve()),
        "resolution": args.resolution,
        "tunnel_diameter": args.tunnel_diameter,
        "rings": results,
    }
    (Path(args.data_dir) / "gt_ceiling_results.json").write_text(
        json.dumps(aggregate, indent=2)
    )
    miou_vals = [r["mIoU"] for r in results if "mIoU" in r]
    if miou_vals:
        print(
            f"[gt-ceiling] median mIoU = {float(statistics.median(miou_vals)):.4f}  "
            f"min={min(miou_vals):.4f} max={max(miou_vals):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
