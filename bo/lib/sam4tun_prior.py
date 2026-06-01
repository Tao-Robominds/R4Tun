"""SAM4Tun static layout prior for honest BO warm-starts (v4).

Resolution-aligned geometric tiling (6-seg: K+5×AB, 7-seg: K+6×AB).
No GT layout; no _bo_v1 warm-start offsets.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import (
    DET_CLI,
    DET_DEFAULT,
    REPO_ROOT,
    VENV_PY,
    blocks_for_segment_count,
)
from lib.search_space import (
    LAYOUT_RECOVERY_PARAMS,
    encode_r_surface_min,
)

if TYPE_CHECKING:
    from lib.layout_bo import RingContext

REF_RESOLUTION = 0.005  # m/px SAM4Tun reference
REF_DIAMETER = 5.5  # m

BLOCKS_6_ORDER = ["K", "B1", "A1", "A2", "A3", "B2"]
BLOCKS_7_ORDER = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]

# BO-searchable subset of default detection JSON keys → layout param name
_LAYOUT_TAIL_FROM_DET = {
    "hough_threshold": "hough_threshold",
    "hough_horizontal_threshold": "hough_horizontal_threshold",
    "merge_distance_threshold": "merge_distance_threshold",
    "single_ring_visual_slot_snap_px": "single_ring_visual_slot_snap_px",
}


@dataclass
class RingPreprocMeta:
    tunnel_diameter: float
    depth_map_resolution: float
    H: int
    segment_count: int
    blocks: list[str]
    scale_px: float
    k_height_px: float
    ab_height_px: float
    k_height_mm: float
    ab_height_mm: float


@dataclass
class Sam4TunPrior:
    case_id: str
    k_y: float
    k_y_centre: float | None
    offsets: dict[str, float]
    layout_params: dict[str, float]
    r_surface_min: float
    search_x: list[float]
    normalized_ab: bool = False
    line_counts: dict[str, int] = field(default_factory=dict)
    resolution_alignment: dict[str, Any] = field(default_factory=dict)


def segment_heights_mm(tunnel_diameter: float) -> tuple[float, float]:
    circ_mm = math.pi * float(tunnel_diameter) * 1000.0
    k_mm = circ_mm / 16.0
    ab_mm = 3.0 * k_mm
    return k_mm, ab_mm


def scale_px_value(value: float, depth_map_resolution: float) -> float:
    scale = REF_RESOLUTION / max(float(depth_map_resolution), 1e-9)
    return float(value) * scale


def load_preproc_meta(src_ring: Path, segment_count: int) -> RingPreprocMeta:
    pre = json.loads((src_ring / "parameters_preprocessing.json").read_text(encoding="utf-8"))
    res = float(pre["depth_map_resolution"])
    diam = float(pre["tunnel_diameter"])
    H = int(np.load(src_ring / "depth_map.npy").shape[0])
    k_mm, ab_mm = segment_heights_mm(diam)
    k_px = k_mm / (res * 1000.0)
    ab_px = ab_mm / (res * 1000.0)
    blocks = blocks_for_segment_count(segment_count)
    return RingPreprocMeta(
        tunnel_diameter=diam,
        depth_map_resolution=res,
        H=H,
        segment_count=segment_count,
        blocks=blocks,
        scale_px=REF_RESOLUTION / max(res, 1e-9),
        k_height_px=k_px,
        ab_height_px=ab_px,
        k_height_mm=k_mm,
        ab_height_mm=ab_mm,
    )


def _block_order(segment_count: int) -> list[str]:
    return list(BLOCKS_7_ORDER if segment_count == 7 else BLOCKS_6_ORDER)


def geometric_boundary_offsets(
    meta: RingPreprocMeta,
    *,
    normalize_to_h: bool = False,
) -> tuple[dict[str, float], float, bool]:
    """Circular boundary offsets from K (K=0), SAM4Tun tiling."""
    H = meta.H
    k_px = meta.k_height_px
    ab_px = meta.ab_height_px
    n_non_k = meta.segment_count - 1
    normalized = False
    if normalize_to_h and meta.segment_count == 7:
        total = k_px + n_non_k * ab_px
        if total > 1e-6 and abs(total - H) / H > 0.05:
            ab_px = max((H - k_px) / n_non_k, 1e-6)
            normalized = True

    order = _block_order(meta.segment_count)
    offsets: dict[str, float] = {order[0]: 0.0}
    pos = k_px
    for block in order[1:]:
        offsets[block] = float(pos % H)
        pos += ab_px
    return offsets, ab_px, normalized


def sam4tun_layout_params_from_default(meta: RingPreprocMeta) -> dict[str, float]:
    det = json.loads(DET_DEFAULT.read_text(encoding="utf-8"))
    out: dict[str, float] = {"slot_inset_y": 0.0}
    for det_key, layout_key in _LAYOUT_TAIL_FROM_DET.items():
        raw = float(det.get(det_key, LAYOUT_RECOVERY_PARAMS[0].default))
        if layout_key in ("hough_threshold", "hough_horizontal_threshold"):
            out[layout_key] = float(int(round(scale_px_value(raw, meta.depth_map_resolution))))
        else:
            out[layout_key] = round(scale_px_value(raw, meta.depth_map_resolution), 4)
    return out


def centre_to_boundary(k_y_centre: float, k_height_px: float, H: int) -> float:
    y = float(k_y_centre) - 0.5 * float(k_height_px)
    return float(y % H)


def build_detection_params(meta: RingPreprocMeta, offsets: dict[str, float]) -> dict[str, Any]:
    det = json.loads(DET_DEFAULT.read_text(encoding="utf-8"))
    det["segment_count"] = meta.segment_count
    det["enabled_blocks"] = meta.blocks
    det["per_ring_offsets"] = {"0": {b: float(offsets[b]) for b in meta.blocks}}
    det.pop("k_y_positions", None)
    for key in ("hough_min_length", "hough_max_gap", "k_expected_height_px", "k_gap_tolerance_px", "groove_snap_px"):
        if key in det:
            det[key] = round(scale_px_value(float(det[key]), meta.depth_map_resolution), 4)
    layout = sam4tun_layout_params_from_default(meta)
    det["hough_threshold"] = int(layout["hough_threshold"])
    det["hough_horizontal_threshold"] = int(layout["hough_horizontal_threshold"])
    det["merge_distance_threshold"] = layout["merge_distance_threshold"]
    if "single_ring_visual_slot_snap_px" in det:
        det["single_ring_visual_slot_snap_px"] = layout.get("single_ring_visual_slot_snap_px", 20.0)
    return det


def run_line_detection(
    ctx: "RingContext",
    det_params: dict[str, Any],
    *,
    tag: str = "sam4tun_line",
) -> tuple[float | None, str | None, dict[str, int]]:
    ctx.sandbox_ring.mkdir(parents=True, exist_ok=True)
    (ctx.sandbox_ring / "parameters_detection.json").write_text(
        json.dumps(det_params, indent=2) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    log = ctx.sandbox_ring / "logs" / f"{tag}_2_detection.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(VENV_PY), str(DET_CLI), ctx.tunnel_id, str(ctx.ring_id), "--data-dir", str(ctx.sandbox_data)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
        check=False,
    )
    log.write_text(proc.stdout or "", encoding="utf-8")
    counts = {"oblique_pos": 0, "oblique_neg": 0, "horizontal": 0, "rc": proc.returncode}
    import re

    m = re.search(r"Lines: \+(\d+) -(\d+) H(\d+)", proc.stdout or "")
    if m:
        counts["oblique_pos"] = int(m.group(1))
        counts["oblique_neg"] = int(m.group(2))
        counts["horizontal"] = int(m.group(3))

    det_csv = ctx.sandbox_ring / "detected.csv"
    if proc.returncode != 0 or not det_csv.exists():
        return None, None, counts

    df = pd.read_csv(det_csv)
    if df.empty:
        return None, None, counts
    k_y = float(df.iloc[0]["Y"])
    k_type = str(df.iloc[0].get("Type", "")) if "Type" in df.columns else None
    return k_y, k_type, counts


def encode_sam4tun_x(
    ctx: "RingContext",
    k_y: float,
    offsets: dict[str, float],
    layout_params: dict[str, float],
    r_surface_min: float,
) -> np.ndarray:
    H = ctx.H
    k_frac = float(k_y) % H / max(H, 1)
    off_fracs = np.array([float(offsets[b]) % H / max(H, 1) for b in ctx.blocks])
    off_fracs[0] = 0.0
    tail = []
    for spec in LAYOUT_RECOVERY_PARAMS:
        val = layout_params.get(spec.name, spec.default)
        tail.append(spec.encode(float(val)))
    r_frac = encode_r_surface_min(r_surface_min, ctx.r_lo, ctx.r_hi)
    return np.concatenate([[k_frac], off_fracs, np.asarray(tail, dtype=float), [r_frac]])


def validate_resolution_alignment(
    meta: RingPreprocMeta,
    offsets: dict[str, float],
    k_y: float,
    x: np.ndarray,
    ctx: "RingContext",
    *,
    ab_height_px_used: float | None = None,
) -> dict[str, Any]:
    from lib.layout_bo import decode_x

    H = meta.H
    n_non_k = meta.segment_count - 1
    ab_px = float(ab_height_px_used if ab_height_px_used is not None else meta.ab_height_px)
    sum_w = meta.k_height_px + n_non_k * ab_px
    rel_err = abs(sum_w - H) / max(H, 1)

    k_dec, off_dec, _, _ = decode_x(ctx, x)
    off_ok = all(abs(float(off_dec[b]) - float(offsets[b]) % H) < 1.5 for b in ctx.blocks)
    k_ok = abs((k_dec - k_y) % H) < 1.5 or abs((k_dec - k_y + H) % H) < 1.5

    passed = (
        math.isfinite(k_y)
        and all(math.isfinite(v) for v in offsets.values())
        and rel_err <= 0.10
        and k_ok
        and off_ok
    )
    return {
        "depth_map_resolution": meta.depth_map_resolution,
        "scale_px": meta.scale_px,
        "k_height_px": round(meta.k_height_px, 4),
        "ab_height_px": round(meta.ab_height_px, 4),
        "H": H,
        "block_width_sum_px": round(sum_w, 2),
        "block_width_rel_err": round(rel_err, 4),
        "round_trip_k_ok": k_ok,
        "round_trip_offsets_ok": off_ok,
        "resolution_alignment_passed": passed,
    }


def compute_sam4tun_prior(ctx: "RingContext", *, normalize_7: bool = True) -> Sam4TunPrior:
    meta = load_preproc_meta(ctx.src_ring, ctx.segment_count)
    offsets, ab_used, normalized = geometric_boundary_offsets(meta, normalize_to_h=normalize_7)
    layout = sam4tun_layout_params_from_default(meta)
    r_surface = float(ctx.r_lo)

    det_params = build_detection_params(meta, offsets)
    k_centre, k_type, line_counts = run_line_detection(ctx, det_params)

    if k_centre is None:
        k_y = float(ctx.H) / 2.0
        k_centre = k_y + 0.5 * meta.k_height_px
    else:
        k_type_l = (k_type or "").lower()
        if "center" in k_type_l or "centre" in k_type_l or "midpoint" in k_type_l:
            k_y = centre_to_boundary(k_centre, meta.k_height_px, meta.H)
        else:
            k_y = float(k_centre) % meta.H

    x = encode_sam4tun_x(ctx, k_y, offsets, layout, r_surface)
    align = validate_resolution_alignment(meta, offsets, k_y, x, ctx, ab_height_px_used=ab_used)
    if meta.segment_count == 7 and normalized:
        align["ab_normalized_to_h"] = True
        align["ab_height_px_used"] = round(ab_used, 4)

    return Sam4TunPrior(
        case_id=ctx.case_id,
        k_y=float(k_y),
        k_y_centre=float(k_centre) if k_centre is not None else None,
        offsets={b: float(offsets[b]) for b in ctx.blocks},
        layout_params=layout,
        r_surface_min=r_surface,
        search_x=[float(v) for v in x.tolist()],
        normalized_ab=normalized,
        line_counts=line_counts,
        resolution_alignment=align,
    )


def prior_to_ring_json(prior: Sam4TunPrior) -> dict[str, Any]:
    return {
        "case_id": prior.case_id,
        "kind": "sam4tun_static",
        "k_y": prior.k_y,
        "k_y_centre": prior.k_y_centre,
        "offsets": prior.offsets,
        "layout_params": prior.layout_params,
        "r_surface_min": prior.r_surface_min,
        "search_x": prior.search_x,
        "normalized_ab": prior.normalized_ab,
        "line_counts": prior.line_counts,
        "resolution_alignment": prior.resolution_alignment,
    }


def load_prior_x_for_ring(prior_root: Path, case_id: str) -> np.ndarray | None:
    path = prior_root / case_id.replace("/", "_") / "sam4tun_prior.json"
    if not path.is_file():
        alt = prior_root / "sam4tun_prior_panel.json"
        if alt.is_file():
            panel = json.loads(alt.read_text(encoding="utf-8"))
            for entry in panel.get("rings", []):
                if entry.get("case_id") == case_id:
                    return np.asarray(entry["search_x"], dtype=float)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["search_x"], dtype=float)
