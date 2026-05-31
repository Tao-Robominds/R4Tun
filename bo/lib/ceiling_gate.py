"""Shared GT-derived ceiling gate logic (6- or 7-block rings)."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"
DET_DEFAULT = REPO_ROOT / "agents" / "2_detection" / "parameters" / "_default_irregular" / "parameters_detection.json"
SEG_DEFAULT = REPO_ROOT / "agents" / "3_segmentation" / "parameters" / "_default_irregular" / "parameters_segmentation.json"

BLOCKS_6 = ["K", "B1", "A1", "A2", "A3", "B2"]
BLOCKS_7 = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
BLOCK_TO_LABEL_7 = {"K": 1, "B1": 2, "A1": 3, "A2": 4, "A3": 5, "A4": 6, "B2": 7}
BLOCK_TO_LABEL_6 = {"K": 1, "B1": 2, "A1": 3, "A2": 4, "A3": 5, "B2": 6}

CEILING_THRESHOLD = 0.85

REQUIRED_PRE = [
    "depth_map.png",
    "depth_map.npy",
    "depth_map_outlier.npy",
    "denoised.csv",
    "enhanced.csv",
    "pixel_to_point.pkl",
    "ring_count.txt",
    "parameters_preprocessing.json",
]


def blocks_for_segment_count(segment_count: int) -> list[str]:
    if segment_count == 6:
        return list(BLOCKS_6)
    if segment_count == 7:
        return list(BLOCKS_7)
    raise ValueError(f"Unsupported segment_count={segment_count}; expected 6 or 7")


def label_to_block_map(segment_count: int) -> dict[int, str]:
    m = BLOCK_TO_LABEL_7 if segment_count == 7 else BLOCK_TO_LABEL_6
    return {v: k for k, v in m.items()}


def detect_segment_count(src_ring: Path) -> int:
    enh = pd.read_csv(src_ring / "enhanced.csv", usecols=["segment"])
    s = enh["segment"].dropna()
    if s.empty:
        raise RuntimeError(f"No GT segments in {src_ring / 'enhanced.csv'}")
    labels = [int(x) for x in s.astype(int).unique() if int(x) >= 1]
    if not labels:
        raise RuntimeError(f"No block GT labels (only background) in {src_ring / 'enhanced.csv'}")
    nmax = max(labels)
    if nmax <= 6:
        return 6
    if nmax == 7:
        return 7
    raise RuntimeError(f"Unexpected max segment label {nmax} in {src_ring}")


def _compute_miou(df: pd.DataFrame, max_class: int = 7) -> float | None:
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    tmp = df[["segment", "pred"]].dropna(subset=["segment", "pred"]).copy()
    if tmp.empty:
        return None
    gt = pd.to_numeric(tmp["segment"], errors="coerce").fillna(0).astype(int).to_numpy()
    pred = pd.to_numeric(tmp["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= max_class) & (pred >= 0) & (pred <= max_class)
    gt, pred = gt[valid], pred[valid]
    if gt.size == 0:
        return None
    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    ious = []
    for cls in labels:
        g, p = gt == cls, pred == cls
        union = np.logical_or(g, p).sum()
        if union:
            ious.append(float(np.logical_and(g, p).sum() / union))
    return float(np.mean(ious)) if ious else None


def _per_class_iou(df: pd.DataFrame, max_class: int = 7) -> dict[str, float]:
    gt = pd.to_numeric(df["segment"], errors="coerce").fillna(0).astype(int).to_numpy()
    pred = pd.to_numeric(df["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    out: dict[str, float] = {}
    for cls in range(0, max_class + 1):
        g, p = gt == cls, pred == cls
        u = np.logical_or(g, p).sum()
        if u:
            out[str(cls)] = round(float(np.logical_and(g, p).sum() / u), 4)
    return out


def otsu_threshold(r: np.ndarray, n_bins: int = 256) -> float:
    r = r[np.isfinite(r)]
    hist, edges = np.histogram(r, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    total = w.sum()
    if total == 0:
        return float("nan")
    p = w / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(centers[k])


def setup_sandbox(src_ring: Path, sandbox_ring: Path) -> None:
    sandbox_ring.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_PRE:
        src = src_ring / name
        if not src.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {src}")
        shutil.copy2(src, sandbox_ring / name)
    (sandbox_ring / "logs").mkdir(exist_ok=True)


def derive_gt_layout(src_ring: Path, sandbox_ring: Path, segment_count: int) -> dict[str, Any]:
    label_to_block = label_to_block_map(segment_count)
    H = int(np.load(sandbox_ring / "depth_map.npy").shape[0])
    enh = pd.read_csv(src_ring / "enhanced.csv").reset_index().rename(columns={"index": "row_idx"})
    with open(src_ring / "pixel_to_point.pkl", "rb") as f:
        p2p = pd.DataFrame(pickle.load(f))
    m = p2p.merge(enh[["row_idx", "segment"]], left_on="index", right_on="row_idx", how="inner")
    m = m.dropna(subset=["segment"])
    m["segment"] = m["segment"].astype(int)
    m = m[(m["segment"] >= 1) & (m["segment"] <= segment_count)]

    ys = {c: (m[m["segment"] == c]["pixel_y"].to_numpy() % H) for c in range(1, segment_count + 1)}
    for c, arr in ys.items():
        if arr.size == 0:
            raise RuntimeError(f"GT class {c} has no labeled rows; cannot derive layout")

    def cmean(a: np.ndarray) -> float:
        ang = 2 * np.pi * a / H
        return float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi) * H)

    centers = {c: cmean(ys[c]) for c in range(1, segment_count + 1)}
    order = sorted(range(1, segment_count + 1), key=lambda c: centers[c])

    def opt_cut(p: int, c: int) -> float:
        cp, cc = centers[p], centers[c]
        gap = (cc - cp) % H
        pp = (ys[p] - cp) % H
        pp = np.where(pp > H / 2, pp - H, pp)
        pc = (ys[c] - cp) % H
        cands = np.linspace(0, gap, 400)
        costs = [(pp > v).sum() + (pc < v).sum() for v in cands]
        return float((cp + cands[int(np.argmin(costs))]) % H)

    starts = {order[i]: opt_cut(order[i - 1], order[i]) for i in range(len(order))}
    k_y = float(starts[1])
    offsets = {label_to_block[c]: float((starts[c] - k_y) % H) for c in range(1, segment_count + 1)}
    return {
        "H": H,
        "k_y": k_y,
        "offsets": offsets,
        "arc_starts_by_label": {str(c): round(starts[c], 1) for c in starts},
        "class_centers_by_label": {str(c): round(centers[c], 1) for c in centers},
        "spatial_order_by_label": order,
    }


def run_agents_unfiltered(
    tunnel_id: str,
    ring_id: int,
    sandbox_data: Path,
    sandbox_ring: Path,
    k_y: float,
    offsets: dict[str, float],
    segment_count: int,
    blocks: list[str],
    tag: str,
) -> pd.DataFrame | None:
    H = int(np.load(sandbox_ring / "depth_map.npy").shape[0])
    det = json.loads(DET_DEFAULT.read_text(encoding="utf-8"))
    seg = json.loads(SEG_DEFAULT.read_text(encoding="utf-8"))
    det["segment_count"] = segment_count
    det["enabled_blocks"] = blocks
    det["k_anchor_semantics"] = "boundary_start"
    det["per_ring_offsets"] = {"0": {b: float(offsets[b]) for b in blocks}}
    det["k_y_positions"] = [float(k_y) % H]
    seg["segment_count"] = segment_count
    seg["r_surface_min"] = 0.0
    (sandbox_ring / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
    (sandbox_ring / "parameters_segmentation.json").write_text(json.dumps(seg, indent=2) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    for cli in (DET_CLI, SEG_CLI):
        log = sandbox_ring / "logs" / f"{tag}_{cli.parent.name}.log"
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [str(VENV_PY), str(cli), tunnel_id, str(ring_id), "--data-dir", str(sandbox_data)],
                cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900, check=False,
            )
        if proc.returncode != 0:
            return None
    final = sandbox_ring / "final.csv"
    return pd.read_csv(final) if final.exists() else None


def best_ceiling_over_cutoff(df: pd.DataFrame, max_class: int = 7) -> dict[str, Any]:
    gt = pd.to_numeric(df["segment"], errors="coerce").fillna(0).astype(int).to_numpy()
    pred0 = pd.to_numeric(df["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    r = df["r"].to_numpy()
    base = _compute_miou(df, max_class=max_class) or 0.0
    best_m, best_t = base, None
    lo, hi = np.nanpercentile(r, 1), np.nanpercentile(r, 60)
    for t in np.arange(lo, hi, 0.02):
        p = pred0.copy()
        p[(p > 0) & (r < t)] = 0
        m = _compute_miou(pd.DataFrame({"segment": gt, "pred": p}), max_class=max_class) or 0.0
        if m > best_m:
            best_m, best_t = m, round(float(t), 4)
    p = pred0.copy()
    if best_t is not None:
        p[(p > 0) & (r < best_t)] = 0
    per_class = _per_class_iou(pd.DataFrame({"segment": gt, "pred": p}), max_class=max_class)
    return {
        "ceiling_miou": round(float(best_m), 4),
        "r_surface_min": best_t,
        "ceiling_no_filter_miou": round(float(base), 4),
        "per_class_iou": per_class,
    }
