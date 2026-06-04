"""Detection layout GP-BO over k_y, offsets, line-evidence params, and r_surface_min (v2).

Search vector x = [k_y_frac, off_frac[block_0], ..., layout_param_frac…, r_surface_min_frac]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from lib.ceiling_gate import (
    REPO_ROOT,
    VENV_PY,
    DET_CLI,
    SEG_CLI,
    DET_DEFAULT,
    SEG_DEFAULT,
    best_ceiling_over_cutoff,
    blocks_for_segment_count,
    derive_gt_layout,
    otsu_threshold,
    run_agents_unfiltered,
    setup_sandbox,
)
from lib.direction_select import select_direction_and_segment, write_selection_to_out_dir
from lib.ring_site_params import resolve_ring_site_params, write_ring_site_params
from lib.perturbations import experience_phase_budgets, forced_perturbation_candidates
from lib.ring_regular import ring_is_regular
from lib.sam4tun_prior import load_prior_x_for_ring
from lib.search_space import (
    N_LAYOUT_TAIL,
    decode_layout_params,
    decode_r_surface_frac,
    decode_r_surface_min,
    default_layout_fracs,
    encode_r_surface_min,
    full_to_layout_stream_x,
    d_stream_dim,
    k_stream_dim,
    layout_params_for_log,
    layout_stream_dim,
    layout_stream_to_full_x,
    search_dim,
    search_space_summary,
    v1_search_dim,
)

EXTRACT_INTRINSICS = REPO_ROOT / "agents" / "2_detection" / "scripts" / "extract_intrinsics.py"

# Not used in single-ring layout BO; omit from trial logs and calibration records.
EXCLUDED_TRIAL_METRICS = frozenset({
    "det_groove_alignment_pct",
    "det_groove_alignment_total",
    "det_groove_alignment_max",
    "det_k_x_spacing_cv",
    "det_k_detection_method",
})

# 7-block K-small / AB-large prior (normalized arc widths)
PRIOR_K_SMALL_7 = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.18])
PRIOR_K_SMALL_6 = np.array([0.07, 0.18, 0.18, 0.18, 0.18, 0.21])


@dataclass
class RingContext:
    tunnel_id: str
    ring_id: int
    source_root: Path
    run_root: Path
    segment_count: int = 0
    tunnel_diameter: float = 0.0
    blocks: list[str] = field(default_factory=list)
    H: int = 0
    r_lo: float = 0.0
    r_hi: float = 0.0
    r_otsu: float = float("nan")
    ceiling_miou: float = 0.0
    ceiling_r_surface_min: float | None = None
    src_ring: Path = field(default_factory=Path)
    sandbox_data: Path = field(default_factory=Path)
    sandbox_ring: Path = field(default_factory=Path)
    out_dir: Path = field(default_factory=Path)
    prior_root: Path | None = None
    warm_anchor: str = "sam4tun"
    experience_stream: str = "full"
    frozen_k_y_frac: float | None = None
    layout_handoff_root: Path | None = None
    k_handoff_root: Path | None = None
    k_handoff_path: Path | None = None
    ring_is_regular: bool = True
    frozen_k_y: float | None = None
    direction_tier_gt: str | None = None
    sam_k_y_frac: float | None = None
    line_k_y_frac: float | None = None
    frozen_offsets: dict[str, float] | None = None
    frozen_layout_params: dict[str, float] | None = None
    frozen_r_surface_min: float | None = None
    layout_handoff_path: Path | None = None

    @property
    def ring_key(self) -> str:
        return f"r{int(self.ring_id)}"

    @property
    def case_id(self) -> str:
        return f"{self.tunnel_id}/{self.ring_key}"

    @property
    def search_dim(self) -> int:
        if self.experience_stream == "d":
            return d_stream_dim()
        if self.experience_stream == "k":
            return k_stream_dim()
        if self.experience_stream == "layout":
            return layout_stream_dim(self.segment_count)
        return search_dim(self.segment_count)

    @property
    def full_search_dim(self) -> int:
        return search_dim(self.segment_count)


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


def _import_extract_metrics():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    agents_scripts = REPO_ROOT / "agents" / "2_detection" / "scripts"
    if str(agents_scripts) not in sys.path:
        sys.path.insert(0, str(agents_scripts))
    from extract_intrinsics import extract_detection_metrics  # noqa: WPS433

    return extract_detection_metrics


def build_ring_context(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    segment_count: int | None = None,
    tunnel_diameter: float | None = None,
    manifest_entry: dict | None = None,
    prior_root: Path | None = None,
    warm_anchor: str = "sam4tun",
    experience_stream: str = "full",
    layout_handoff_root: Path | None = None,
    k_handoff_root: Path | None = None,
) -> RingContext:
    src_ring = source_root / tunnel_id / f"r{int(ring_id)}"
    if not src_ring.is_dir():
        raise FileNotFoundError(f"No preprocessing at {src_ring}")

    ring_key = f"{tunnel_id}/r{int(ring_id)}"
    site = resolve_ring_site_params(
        ring_key,
        src_ring,
        segment_count=segment_count,
        tunnel_diameter=tunnel_diameter,
        manifest_entry=manifest_entry,
    )
    seg_n = int(site["segment_count"])
    diam = float(site["tunnel_diameter"])
    blocks = blocks_for_segment_count(seg_n)
    sandbox_data = run_root / "sandbox"
    sandbox_ring = sandbox_data / tunnel_id / f"r{int(ring_id)}"
    out_dir = run_root / tunnel_id / f"r{int(ring_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_sandbox(src_ring, sandbox_ring)
    write_ring_site_params(sandbox_ring, site)
    write_ring_site_params(out_dir, site)
    H = int(np.load(sandbox_ring / "depth_map.npy").shape[0])
    enh = pd.read_csv(src_ring / "enhanced.csv")
    r_vals = enh["r"].to_numpy()
    r_lo = float(np.nanpercentile(r_vals, 1))
    r_hi = float(np.nanpercentile(r_vals, 60))
    r_otsu = float(otsu_threshold(r_vals))

    ctx = RingContext(
        tunnel_id=tunnel_id,
        ring_id=ring_id,
        source_root=source_root,
        run_root=run_root,
        segment_count=seg_n,
        tunnel_diameter=diam,
        blocks=blocks,
        H=H,
        r_lo=r_lo,
        r_hi=r_hi,
        r_otsu=r_otsu,
        src_ring=src_ring,
        sandbox_data=sandbox_data,
        sandbox_ring=sandbox_ring,
        out_dir=out_dir,
        prior_root=prior_root,
        warm_anchor=warm_anchor,
        experience_stream=experience_stream,
        layout_handoff_root=layout_handoff_root,
        k_handoff_root=k_handoff_root,
    )
    ctx.ring_is_regular = ring_is_regular(src_ring, seg_n)
    if experience_stream == "layout":
        ctx = _init_layout_stream_frozen_k(ctx)
    elif experience_stream == "k":
        ctx = _init_k_stream(ctx)
    elif experience_stream == "d":
        ctx = _init_d_stream(ctx)
    elif experience_stream == "full":
        ctx = _init_full_stream_k_anchors(ctx)
    return ctx


def _coerce_full_search_x(ctx: RingContext, x: np.ndarray) -> np.ndarray:
    """Coerce to full (k_y + offsets + tail + r) vector — ignores layout-stream mode."""
    x = np.asarray(x, dtype=float).ravel()
    expected = search_dim(ctx.segment_count)
    v1_dim = v1_search_dim(ctx.segment_count)
    legacy = 1 + ctx.segment_count + 1
    if x.size == expected:
        return x
    if x.size == v1_dim:
        return np.concatenate([x, [_default_r_frac(ctx)]])
    if x.size == legacy:
        r_frac = float(x[-1])
        return np.concatenate([x[:-1], default_layout_fracs(), [r_frac]])
    raise ValueError(f"search_x length {x.size} != expected {expected} (v1 {v1_dim}, legacy {legacy})")


def _init_layout_stream_frozen_k(ctx: RingContext) -> RingContext:
    prior_root = _resolve_prior_root(ctx)
    if prior_root is None:
        raise ValueError(
            f"Stream layout requires SAM4Tun prior under prior_root for {ctx.case_id}"
        )
    full_x = load_prior_x_for_ring(prior_root, ctx.case_id)
    if full_x is None:
        raise ValueError(f"Missing SAM4Tun prior search_x for {ctx.case_id}")
    full_x = _coerce_full_search_x(ctx, full_x)
    ctx.frozen_k_y_frac = float(full_x[0])
    return ctx


def _default_layout_handoff_root() -> Path:
    return REPO_ROOT / "logs" / "proxy4tun" / "stream_l"


def load_layout_handoff_for_ring(
    ctx: RingContext, handoff_root: Path | None = None
) -> dict[str, Any]:
    root = handoff_root or ctx.layout_handoff_root or _default_layout_handoff_root()
    path = root / ctx.tunnel_id / ctx.ring_key / "layout_best_for_stream_k.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stream L handoff: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _default_k_handoff_root() -> Path:
    return REPO_ROOT / "logs" / "proxy4tun" / "stream_k"


def load_k_handoff_for_ring(ctx: RingContext, handoff_root: Path | None = None) -> dict[str, Any]:
    root = handoff_root or ctx.k_handoff_root or _default_k_handoff_root()
    path = root / ctx.tunnel_id / ctx.ring_key / "k_best_for_stream_d.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stream K handoff: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _direction_tier_gt(ctx: RingContext) -> str:
    from lib.held_out_descriptors import _direction_from_spatial_order, _load_spatial_order

    gt_path = ctx.out_dir / "gt_layout.json"
    if not gt_path.is_file():
        gt_path = ctx.src_ring / "gt_layout.json"
    if gt_path.is_file():
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        spatial = gt.get("spatial_order_by_label")
        if spatial:
            tier, _ = _direction_from_spatial_order(spatial, ctx.segment_count)
            return tier
    ring_dir = ctx.src_ring
    spatial = _load_spatial_order(ring_dir, ctx.segment_count)
    tier, _ = _direction_from_spatial_order(spatial, ctx.segment_count)
    return tier


def _init_d_stream(ctx: RingContext) -> RingContext:
    handoff = load_k_handoff_for_ring(ctx)
    root = ctx.k_handoff_root or _default_k_handoff_root()
    ctx.k_handoff_path = root / ctx.tunnel_id / ctx.ring_key / "k_best_for_stream_d.json"
    ctx.frozen_k_y = float(handoff["k_y"])
    ctx.frozen_k_y_frac = float(handoff.get("k_y_frac") or (ctx.frozen_k_y % ctx.H / max(ctx.H, 1)))
    ctx.frozen_offsets = {k: float(v) for k, v in handoff["per_ring_offsets"].items()}
    ctx.frozen_layout_params = dict(handoff["layout_params"])
    ctx.frozen_r_surface_min = float(handoff["r_surface_min"])
    if "ring_is_regular" in handoff:
        ctx.ring_is_regular = bool(handoff["ring_is_regular"])
    ctx.direction_tier_gt = _direction_tier_gt(ctx)
    return ctx


def _init_full_stream_k_anchors(ctx: RingContext) -> RingContext:
    """Load SAM/line K anchors for k_anchor_dist_* logging on joint (--stream full) trials."""
    prior_root = _resolve_prior_root(ctx)
    if prior_root is not None:
        full_x = load_prior_x_for_ring(prior_root, ctx.case_id)
        if full_x is not None:
            ctx.sam_k_y_frac = float(_coerce_full_search_x(ctx, full_x)[0])
        prior_json = prior_root / ctx.case_id.replace("/", "_") / "sam4tun_prior.json"
        if prior_json.is_file():
            pj = json.loads(prior_json.read_text(encoding="utf-8"))
            k_y = float(pj.get("k_y", 0))
            ctx.line_k_y_frac = float(k_y % ctx.H / max(ctx.H, 1))
    return ctx


def _init_k_stream(ctx: RingContext) -> RingContext:
    handoff = load_layout_handoff_for_ring(ctx)
    ctx.layout_handoff_path = (
        (ctx.layout_handoff_root or _default_layout_handoff_root())
        / ctx.tunnel_id
        / ctx.ring_key
        / "layout_best_for_stream_k.json"
    )
    ctx.frozen_offsets = {k: float(v) for k, v in handoff["per_ring_offsets"].items()}
    ctx.frozen_layout_params = dict(handoff["layout_params"])
    ctx.frozen_r_surface_min = float(handoff["r_surface_min"])

    prior_root = _resolve_prior_root(ctx)
    if prior_root is not None:
        full_x = load_prior_x_for_ring(prior_root, ctx.case_id)
        if full_x is not None:
            ctx.sam_k_y_frac = float(_coerce_full_search_x(ctx, full_x)[0])
        prior_json = prior_root / ctx.case_id.replace("/", "_") / "sam4tun_prior.json"
        if prior_json.is_file():
            pj = json.loads(prior_json.read_text(encoding="utf-8"))
            k_y = float(pj.get("k_y", 0))
            ctx.line_k_y_frac = float(k_y % ctx.H / max(ctx.H, 1))
    return ctx


def _parse_det_line_counts(log_path: Path) -> dict[str, int]:
    import re

    counts = {"oblique_pos": 0, "oblique_neg": 0, "horizontal": 0}
    if not log_path.is_file():
        return counts
    m = re.search(
        r"Lines: \+(\d+) -(\d+) H(\d+)",
        log_path.read_text(encoding="utf-8", errors="ignore"),
    )
    if m:
        counts["oblique_pos"] = int(m.group(1))
        counts["oblique_neg"] = int(m.group(2))
        counts["horizontal"] = int(m.group(3))
    return counts


def compute_ceiling_reference(ctx: RingContext) -> dict[str, Any]:
    gt = derive_gt_layout(ctx.src_ring, ctx.sandbox_ring, ctx.segment_count)
    (ctx.out_dir / "gt_layout.json").write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")

    final_df = run_agents_unfiltered(
        ctx.tunnel_id,
        ctx.ring_id,
        ctx.sandbox_data,
        ctx.sandbox_ring,
        gt["k_y"],
        gt["offsets"],
        ctx.segment_count,
        ctx.blocks,
        tag="ceiling_ref",
    )
    if final_df is None:
        ceiling = {"agents_gt_ceiling_miou": None, "failure_reason": "agent_error"}
    else:
        cap = best_ceiling_over_cutoff(final_df, max_class=ctx.segment_count)
        ceiling = {
            "case_id": ctx.case_id,
            "agents_gt_ceiling_miou": cap["ceiling_miou"],
            "ceiling_no_filter_miou": cap["ceiling_no_filter_miou"],
            "r_surface_min_selected": cap["r_surface_min"],
            "r_surface_min_otsu_intrinsic": round(ctx.r_otsu, 4),
            "per_class_iou": cap["per_class_iou"],
            "r_search_bounds": {"r_lo": ctx.r_lo, "r_hi": ctx.r_hi},
        }
        ctx.ceiling_miou = float(cap["ceiling_miou"])
        ctx.ceiling_r_surface_min = cap.get("r_surface_min")

    (ctx.out_dir / "ceiling.json").write_text(json.dumps(ceiling, indent=2) + "\n", encoding="utf-8")
    return ceiling


def widths_to_offset_fracs(blocks: list[str], widths: np.ndarray) -> np.ndarray:
    """Cumulative arc positions (as H-fractions) from normalized block widths."""
    w = np.clip(widths, 1e-3, None)
    w = w / w.sum()
    return np.concatenate([[0.0], np.cumsum(w)[:-1]])


def _default_r_frac(ctx: RingContext) -> float:
    val = ctx.ceiling_r_surface_min if ctx.ceiling_r_surface_min is not None else ctx.r_otsu
    return encode_r_surface_min(val, ctx.r_lo, ctx.r_hi)


def _coerce_search_x(ctx: RingContext, x: np.ndarray) -> np.ndarray:
    """Accept layout-stream, full, v1, and legacy vectors for read/resume."""
    x = np.asarray(x, dtype=float).ravel()
    if ctx.experience_stream == "d":
        return np.asarray([], dtype=float)
    if ctx.experience_stream == "k":
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 1:
            return x
        if x.size == search_dim(ctx.segment_count):
            return np.asarray([float(x[0])], dtype=float)
        raise ValueError(f"k stream search_x length {x.size} != 1")
    if ctx.experience_stream == "layout":
        expected = layout_stream_dim(ctx.segment_count)
        full_n = search_dim(ctx.segment_count)
        if x.size == expected:
            return x
        if x.size == full_n:
            return full_to_layout_stream_x(x, ctx.segment_count)
        raise ValueError(f"layout search_x length {x.size} != expected {expected} (full {full_n})")
    return _coerce_full_search_x(ctx, x)


def decode_x(
    ctx: RingContext, x: np.ndarray
) -> tuple[float, dict[str, float], dict[str, float], float]:
    if ctx.experience_stream == "d":
        if ctx.frozen_k_y is None or ctx.frozen_offsets is None:
            raise ValueError(f"Stream D handoff not loaded for {ctx.case_id}")
        return (
            float(ctx.frozen_k_y),
            dict(ctx.frozen_offsets),
            dict(ctx.frozen_layout_params or {}),
            float(ctx.frozen_r_surface_min),
        )
    if ctx.experience_stream == "k":
        if ctx.frozen_offsets is None or ctx.frozen_layout_params is None:
            raise ValueError(f"Stream K handoff not loaded for {ctx.case_id}")
        k_frac = float(_coerce_search_x(ctx, x)[0])
        k_y = k_frac * ctx.H
        return (
            k_y,
            dict(ctx.frozen_offsets),
            dict(ctx.frozen_layout_params),
            float(ctx.frozen_r_surface_min),
        )
    if ctx.experience_stream == "layout":
        if ctx.frozen_k_y_frac is None:
            raise ValueError(f"frozen_k_y_frac not set for layout stream on {ctx.case_id}")
        x = layout_stream_to_full_x(
            _coerce_search_x(ctx, x), ctx.frozen_k_y_frac, ctx.segment_count
        )
    else:
        x = _coerce_search_x(ctx, x)
    k_y = float(x[0]) * ctx.H
    off_fracs = x[1 : 1 + ctx.segment_count]
    offsets = {ctx.blocks[i]: float(off_fracs[i] % 1.0) * ctx.H for i in range(len(ctx.blocks))}
    offsets[ctx.blocks[0]] = 0.0
    layout = decode_layout_params(x, ctx.segment_count)
    r_frac = decode_r_surface_frac(x, ctx.segment_count)
    r_surface_min = decode_r_surface_min(r_frac, ctx.r_lo, ctx.r_hi)
    return k_y, offsets, layout, r_surface_min


def arc_width_entropy(widths: np.ndarray) -> float:
    w = np.clip(widths, 1e-9, None)
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def offsets_to_arc_widths(blocks: list[str], offsets: dict[str, float], H: int) -> np.ndarray:
    ys = [float(offsets[b]) % H for b in blocks]
    widths = []
    for i in range(len(blocks)):
        w = (ys[(i + 1) % len(blocks)] - ys[i]) % H
        widths.append(w)
    return np.array(widths, dtype=float)


def encode_gt_layout_x(
    ctx: RingContext,
    gt_layout: dict[str, Any],
    *,
    r_surface_min: float | None = None,
) -> np.ndarray:
    """Encode GT k_y + offsets + default layout params + r_surface_min frac."""
    k_y_frac = float(gt_layout["k_y"]) % ctx.H / max(ctx.H, 1)
    off_fracs = np.array([float(gt_layout["offsets"][b]) % ctx.H / max(ctx.H, 1) for b in ctx.blocks])
    off_fracs[0] = 0.0
    r_val = r_surface_min
    if r_val is None:
        r_val = ctx.ceiling_r_surface_min if ctx.ceiling_r_surface_min is not None else ctx.r_otsu
    r_frac = encode_r_surface_min(r_val, ctx.r_lo, ctx.r_hi)
    return np.concatenate([[k_y_frac], off_fracs, default_layout_fracs(), [r_frac]])


ORACLE_TRIAL_KINDS = frozenset({"gt_layout", "gt_layout_otsu_r", "gt_layout_ceiling_r"})

WARM_ANCHOR_CHOICES = frozenset({"sam4tun", "geometric", "gt_derived"})


def _load_gt_layout(ctx: RingContext) -> dict[str, Any]:
    gt_path = ctx.out_dir / "gt_layout.json"
    if gt_path.is_file():
        return json.loads(gt_path.read_text(encoding="utf-8"))
    src = ctx.src_ring / "gt_layout.json"
    if src.is_file():
        return json.loads(src.read_text(encoding="utf-8"))
    return {}


def gt_derived_reference_x(ctx: RingContext) -> np.ndarray:
    """GT k_y + offsets + ceiling r_surface_min — perturbation anchor for gt_derived mode."""
    gt_layout = _load_gt_layout(ctx)
    if not gt_layout:
        raise ValueError(f"Missing gt_layout.json for {ctx.case_id}")
    r_val = ctx.ceiling_r_surface_min if ctx.ceiling_r_surface_min is not None else ctx.r_otsu
    return _coerce_search_x(ctx, encode_gt_layout_x(ctx, gt_layout, r_surface_min=r_val))


def _resolve_prior_root(ctx: RingContext) -> Path | None:
    if ctx.prior_root is not None:
        return ctx.prior_root
    env = os.environ.get("SAM4TUN_PRIOR_ROOT")
    if env:
        return Path(env)
    for candidate in (
        REPO_ROOT / "logs" / "proxy4tun" / "sam4tun_prior",
        REPO_ROOT / "logs" / "sam4tun_prior_v1",
    ):
        if candidate.is_dir():
            return candidate
    return None


def experience_warm_seeds(ctx: RingContext, rng: np.random.Generator) -> list[tuple[np.ndarray, str]]:
    """Warm-start candidates for experience collection (policy set by ctx.warm_anchor)."""
    seeds: list[tuple[np.ndarray, str]] = []

    if ctx.experience_stream == "k":
        if ctx.sam_k_y_frac is not None:
            seeds.append((np.asarray([ctx.sam_k_y_frac], dtype=float), "sam4tun_k"))
        if ctx.line_k_y_frac is not None and ctx.line_k_y_frac != ctx.sam_k_y_frac:
            seeds.append((np.asarray([ctx.line_k_y_frac], dtype=float), "line_k"))
        x = rng.random(1)
        seeds.append((x, "random_k"))
        return seeds

    if ctx.warm_anchor == "gt_derived":
        gt_layout = _load_gt_layout(ctx)
        if gt_layout:
            r_ceil = ctx.ceiling_r_surface_min if ctx.ceiling_r_surface_min is not None else ctx.r_otsu
            seeds.append((
                _coerce_search_x(ctx, encode_gt_layout_x(ctx, gt_layout, r_surface_min=r_ceil)),
                "gt_layout_ceiling_r",
            ))
            seeds.append((
                _coerce_search_x(ctx, encode_gt_layout_x(ctx, gt_layout, r_surface_min=ctx.r_otsu)),
                "gt_layout_otsu_r",
            ))
        else:
            print(f"  WARN: no gt_layout for {ctx.case_id}; gt_derived warm-start degraded")
    elif ctx.warm_anchor == "geometric":
        for i, x in enumerate(geometric_priors(ctx)):
            seeds.append((_coerce_search_x(ctx, x), f"geometric_{i}"))
    else:
        prior_root = _resolve_prior_root(ctx)
        if prior_root is not None:
            x = load_prior_x_for_ring(prior_root, ctx.case_id)
            if x is not None:
                seeds.append((_coerce_search_x(ctx, x), "sam4tun_static"))
        if not seeds:
            print(f"  WARN: no SAM4Tun prior for {ctx.case_id}; falling back to geometric_0")
            seeds.append((geometric_priors(ctx)[0], "geometric_0"))

    if ctx.experience_stream == "layout":
        intrinsic_x = rng.random(ctx.search_dim)
        intrinsic_x[-1] = encode_r_surface_min(ctx.r_otsu, ctx.r_lo, ctx.r_hi)
        seeds.append((intrinsic_x, "intrinsic_r_otsu"))
    elif ctx.experience_stream == "full":
        intrinsic_x = rng.random(ctx.search_dim)
        intrinsic_x[-1] = encode_r_surface_min(ctx.r_otsu, ctx.r_lo, ctx.r_hi)
        seeds.append((intrinsic_x, "intrinsic_r_otsu"))
    return seeds


def default_perturbation_reference_x(ctx: RingContext, rng: np.random.Generator) -> np.ndarray:
    """Fallback perturbation anchor — policy-specific; gt_derived uses GT ceiling-r layout."""
    if ctx.warm_anchor == "gt_derived":
        return gt_derived_reference_x(ctx)
    for x, _kind in experience_warm_seeds(ctx, rng):
        if _kind in ("sam4tun_static", "sam4tun_k"):
            return x
    return experience_warm_seeds(ctx, rng)[0][0]


def write_search_space_spec(ctx: RingContext) -> None:
    spec = search_space_summary(
        ctx.segment_count,
        experience_stream=ctx.experience_stream,
        r_lo=ctx.r_lo,
        r_hi=ctx.r_hi,
        r_otsu_ref=ctx.r_otsu,
        r_ceiling_ref=ctx.ceiling_r_surface_min,
    )
    spec["ring_is_regular"] = bool(ctx.ring_is_regular)
    if ctx.experience_stream == "k" and ctx.layout_handoff_path is not None:
        spec["layout_handoff"] = str(ctx.layout_handoff_path)
    if ctx.experience_stream == "d" and ctx.k_handoff_path is not None:
        spec["k_handoff"] = str(ctx.k_handoff_path)
        spec["direction_tier_gt"] = ctx.direction_tier_gt
    (ctx.out_dir / "search_space.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def ceiling_push_priors(ctx: RingContext) -> list[tuple[np.ndarray, str]]:
    """Geometric priors only — no GT-layout trial seeds."""
    return [(x, f"geometric_{i}") for i, x in enumerate(geometric_priors(ctx))]


def load_trial_history(ctx: RingContext) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]], int, float]:
    """Load existing bo_trials.csv for resume. Returns X, Y, rows, next_idx, best_y."""
    trials_path = ctx.out_dir / "bo_trials.csv"
    if not trials_path.exists():
        return [], [], [], 0, -1.0
    df = pd.read_csv(trials_path)
    if df.empty:
        return [], [], [], 0, -1.0
    X, Y, rows = [], [], []
    for _, row in df.iterrows():
        x = _coerce_search_x(ctx, np.asarray(json.loads(row["search_x"]), dtype=float))
        y = float(row["gt_miou"])
        X.append(x)
        Y.append(y)
        rows.append(row.to_dict())
    next_idx = int(df["trial_id"].max()) + 1
    best_y = float(df["gt_miou"].max())
    return X, Y, rows, next_idx, best_y


def _regret_at_checkpoints(trials_df: pd.DataFrame, ceiling: float, checkpoints: list[int]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if trials_df.empty or not ceiling:
        return {str(c): None for c in checkpoints}
    for c in checkpoints:
        sub = trials_df[trials_df["trial_id"] < c]
        if sub.empty:
            out[str(c)] = None
        else:
            best = float(sub["gt_miou"].max())
            out[str(c)] = round(float(ceiling - best), 4)
    return out


def write_ceiling_push_report(
    ctx: RingContext,
    trials_df: pd.DataFrame,
    *,
    target_regret: float,
    stop_reason: str,
    n_iterations: int,
    order_branch: str,
    oracle_ceiling_miou: float | None = None,
) -> dict[str, Any]:
    best_bo = float(trials_df["gt_miou"].max()) if not trials_df.empty else 0.0
    target_miou = float(ctx.ceiling_miou) - float(target_regret) if ctx.ceiling_miou else None
    regret = float(ctx.ceiling_miou) - best_bo if ctx.ceiling_miou else None
    target_reached = target_miou is not None and best_bo >= target_miou

    checkpoints = [128, 256, 384, 512, 768, 1024]
    oracle_ref = oracle_ceiling_miou if oracle_ceiling_miou is not None else ctx.ceiling_miou
    report = {
        "objective": "minimize_regret_vs_gt_ceiling",
        "case_id": ctx.case_id,
        "ceiling_miou_reference": ctx.ceiling_miou,
        "oracle_ceiling_miou": round(float(oracle_ref), 4) if oracle_ref is not None else None,
        "best_bo_miou": round(best_bo, 4),
        "regret_vs_ceiling": round(regret, 4) if regret is not None else None,
        "target_miou": round(target_miou, 4) if target_miou is not None else None,
        "target_regret": float(target_regret),
        "target_reached": bool(target_reached),
        "stop_reason": stop_reason,
        "total_evals": int(len(trials_df)),
        "n_iterations": int(n_iterations),
        "regret_at_checkpoints": _regret_at_checkpoints(trials_df, ctx.ceiling_miou, checkpoints),
        "order_branch": order_branch,
    }
    (ctx.out_dir / "ceiling_push_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def geometric_priors(ctx: RingContext) -> list[np.ndarray]:
    n = ctx.segment_count
    equal_w = np.full(n, 1.0 / n)
    k_small = PRIOR_K_SMALL_7 if n == 7 else PRIOR_K_SMALL_6
    if len(k_small) != n:
        k_small = np.concatenate([k_small[: n - 1], [k_small[-1]]])[:n]
        k_small = k_small / k_small.sum()
    layout = default_layout_fracs()
    r_mid = encode_r_surface_min(0.5 * (ctx.r_lo + ctx.r_hi), ctx.r_lo, ctx.r_hi)
    tail = np.concatenate([layout, [r_mid]])
    return [
        np.concatenate([[0.0], widths_to_offset_fracs(ctx.blocks, equal_w), tail]),
        np.concatenate([[0.0], widths_to_offset_fracs(ctx.blocks, k_small), tail]),
    ]


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def make_gp(seed: int, n_dims: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(n_dims),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    )


def evaluate_trial(
    ctx: RingContext,
    k_y: float,
    offsets: dict[str, float],
    layout_params: dict[str, float],
    r_surface_min: float,
    tag: str,
    order_branch: str = "plus",
    force_branch: str | None = None,
) -> dict[str, Any]:
    """Run det+seg with injected layout + line-evidence + r_surface_min params."""
    H = ctx.H
    det = json.loads(DET_DEFAULT.read_text(encoding="utf-8"))
    seg = json.loads(SEG_DEFAULT.read_text(encoding="utf-8"))
    det["segment_count"] = ctx.segment_count
    det["enabled_blocks"] = ctx.blocks
    det["k_anchor_semantics"] = "boundary_start"
    det["per_ring_offsets"] = {"0": {b: float(offsets[b]) for b in ctx.blocks}}
    det["k_y_positions"] = [float(k_y) % H]
    branch = str(order_branch).lower()
    det["reverse_ring_order"] = branch == "minus"
    for key, val in layout_params.items():
        if key == "slot_inset_y":
            continue
        if key in ("hough_threshold", "hough_horizontal_threshold"):
            det[key] = int(round(val))
        else:
            det[key] = val
    seg["segment_count"] = ctx.segment_count
    seg["r_surface_min"] = float(r_surface_min)
    seg["slot_inset_y"] = float(layout_params.get("slot_inset_y", 0.0))

    ctx.sandbox_ring.mkdir(parents=True, exist_ok=True)
    (ctx.sandbox_ring / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
    (ctx.sandbox_ring / "parameters_segmentation.json").write_text(json.dumps(seg, indent=2) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"

    log = ctx.sandbox_ring / "logs" / f"{tag}_2_detection.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [str(VENV_PY), str(DET_CLI), ctx.tunnel_id, str(ctx.ring_id), "--data-dir", str(ctx.sandbox_data)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )
    if proc.returncode != 0:
        return {"gt_miou": 0.0, "agent_error": True}

    direction_sel = select_direction_and_segment(
        tunnel_id=ctx.tunnel_id,
        ring_id=ctx.ring_id,
        sandbox_data=ctx.sandbox_data,
        ring_dir=ctx.sandbox_ring,
        tag=tag,
        prefer_branch=branch,
        segment_count=ctx.segment_count,
        force_branch=force_branch,
        log_twin_gt_miou=ctx.experience_stream == "d",
    )
    write_selection_to_out_dir(ctx.sandbox_ring, ctx.out_dir)
    if direction_sel.get("agent_error"):
        return {"gt_miou": 0.0, "agent_error": True}

    final_path = ctx.sandbox_ring / "final.csv"
    if not final_path.exists():
        return {"gt_miou": 0.0, "agent_error": True}

    final_df = pd.read_csv(final_path)
    gt_miou = _compute_miou(final_df, max_class=ctx.segment_count) or 0.0

    n_reclass = 0
    seg_log = ctx.sandbox_ring / "logs" / f"{tag}_3_segmentation.log"
    if seg_log.exists():
        for line in seg_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "Radial filter" in line and "background" in line:
                try:
                    part = line.split(":")[1].split("points")[0].strip().replace(",", "")
                    n_reclass = int(part)
                except (IndexError, ValueError):
                    pass
                break

    intrinsics: dict[str, Any] = {}
    try:
        extract_detection_metrics = _import_extract_metrics()
        raw = extract_detection_metrics(str(ctx.sandbox_ring))
        for k, v in raw.items():
            if k == "det_guardrail_violations":
                intrinsics[k] = json.dumps(v) if v else "[]"
            else:
                intrinsics[k] = v
    except Exception as exc:
        intrinsics["det_extract_error"] = str(exc)

    intrinsics = {k: v for k, v in intrinsics.items() if k not in EXCLUDED_TRIAL_METRICS}

    r_surface_min = float(seg["r_surface_min"])
    arc_widths = offsets_to_arc_widths(ctx.blocks, offsets, H)
    k_frac = float((k_y % H) / max(H, 1))

    extra: dict[str, Any] = {}
    if ctx.experience_stream in ("k", "full"):
        from lib.line_reliability import compute_line_evidence  # noqa: WPS433

        line_counts = _parse_det_line_counts(log)
        line_ev = compute_line_evidence(
            ctx,
            k_y=float(k_y) % H,
            k_type=None,
            line_counts=line_counts,
            log_path=log,
        )
        extra.update(line_ev.to_dict())
        extra["layout_k_center_norm"] = round(k_frac, 6)
        if ctx.sam_k_y_frac is not None:
            extra["k_anchor_dist_sam_frac"] = round(abs(k_frac - ctx.sam_k_y_frac), 6)
        if ctx.line_k_y_frac is not None and line_ev.valid_line_anchor:
            extra["k_anchor_dist_line_frac"] = round(abs(k_frac - ctx.line_k_y_frac), 6)
        else:
            extra["k_anchor_dist_line_frac"] = None
        extra["ring_is_regular"] = bool(ctx.ring_is_regular)

    if ctx.experience_stream == "d":
        miou_plus = direction_sel.get("gt_miou_plus")
        miou_minus = direction_sel.get("gt_miou_minus")
        oracle = None
        if miou_plus is not None and miou_minus is not None:
            oracle = "plus" if float(miou_plus) >= float(miou_minus) else "minus"
        extra.update({
            "gt_miou_plus": miou_plus,
            "gt_miou_minus": miou_minus,
            "gt_miou_oracle_branch": oracle,
            "oracle_branch_hit": (
                direction_sel.get("selected_branch") == oracle if oracle else None
            ),
            "template_match_score_plus": direction_sel.get("template_match_score_plus"),
            "template_match_score_minus": direction_sel.get("template_match_score_minus"),
            "prefer_branch": direction_sel.get("prefer_branch", branch),
            "direction_tier_gt": ctx.direction_tier_gt,
        })

    return {
        "gt_miou": float(gt_miou),
        "agent_error": False,
        "n_reclassified_by_r_filter": n_reclass,
        "r_surface_min": round(r_surface_min, 4),
        "r_surface_min_frac": round(encode_r_surface_min(r_surface_min, ctx.r_lo, ctx.r_hi), 6),
        "r_surface_min_ceiling_ref": round(float(ctx.ceiling_r_surface_min), 4) if ctx.ceiling_r_surface_min is not None else None,
        "r_surface_min_otsu_ref": round(ctx.r_otsu, 4),
        "k_y_frac": k_frac,
        "arc_width_entropy": arc_width_entropy(arc_widths),
        "order_branch": direction_sel.get("selected_branch", branch),
        "branch_is_minus": direction_sel.get("selected_branch", branch) == "minus",
        "direction_score_plus": direction_sel.get("score_plus"),
        "direction_score_minus": direction_sel.get("score_minus"),
        "direction_margin": direction_sel.get("margin"),
        "template_margin_minus_plus": direction_sel.get("template_margin_minus_plus"),
        "template_match_score_plus": direction_sel.get("template_match_score_plus"),
        "template_match_score_minus": direction_sel.get("template_match_score_minus"),
        "direction_select_enabled": direction_sel.get("direction_select_enabled"),
        **layout_params_for_log(layout_params),
        **intrinsics,
        **extra,
    }


def order_trial_schedule(n_evals: int, rng: np.random.Generator) -> list[tuple[str, str, str | None]]:
    """(kind, prefer_branch, force_branch)."""
    trials: list[tuple[str, str, str | None]] = [("twin_baseline", "plus", None)]
    n_rest = max(0, n_evals - 1)
    n_each = n_rest // 2
    trials.extend([("force_plus", "plus", "plus")] * n_each)
    trials.extend([("force_minus", "minus", "minus")] * (n_rest - n_each))
    while len(trials) < n_evals:
        pref = "plus" if rng.random() < 0.5 else "minus"
        trials.append(("random_prefer", pref, None))
    return trials[:n_evals]


def run_order_stream_bo(
    ctx: RingContext,
    *,
    n_evals: int = 32,
    seed: int = 7,
    resume: bool = False,
) -> dict[str, Any]:
    """Stream D: fixed L+K layout; vary order selection policy across trials."""
    rng = np.random.default_rng(seed)
    write_search_space_spec(ctx)
    frozen_x = np.asarray([], dtype=float)

    full_schedule = order_trial_schedule(n_evals, rng)
    if resume and (ctx.out_dir / "bo_trials.csv").exists():
        old_df = pd.read_csv(ctx.out_dir / "bo_trials.csv")
        start_idx = len(old_df)
        best_y = float(old_df["gt_miou"].max()) if not old_df.empty else -1.0
        best_row = old_df.loc[old_df["gt_miou"].idxmax()].to_dict() if not old_df.empty else None
        trial_rows: list[dict[str, Any]] = []
        schedule = full_schedule[start_idx:n_evals]
    else:
        start_idx = 0
        best_y, best_row = -1.0, None
        trial_rows = []
        schedule = full_schedule

    if not schedule and resume and (ctx.out_dir / "bo_trials.csv").exists():
        trials_df = pd.read_csv(ctx.out_dir / "bo_trials.csv")
        best_row = trials_df.loc[trials_df["gt_miou"].idxmax()].to_dict()
        k_y_b, offs_b, layout_b, r_surf_b = decode_x(ctx, frozen_x)
        best_payload = {
            "case_id": ctx.case_id,
            "best_bo_miou": float(best_row["gt_miou"]),
            "best_k_y": k_y_b,
            "best_offsets": offs_b,
            "best_layout_params": layout_b,
            "best_r_surface_min": r_surf_b,
            "n_evals": len(trials_df),
            "ceiling_miou_reference": ctx.ceiling_miou,
            "order_branch": best_row.get("order_branch"),
        }
        return {
            "trials_df": trials_df,
            "best_payload": best_payload,
            "best_row": best_row,
            "chunk_trials": 0,
            "agent_error_stop": False,
        }

    consecutive_errors = 0
    for sched_i, (kind, prefer, force) in enumerate(schedule):
        trial_idx = start_idx + sched_i
        k_y, offs, layout, r_surf = decode_x(ctx, frozen_x)
        tag = f"trial{trial_idx:03d}"
        metrics = evaluate_trial(
            ctx, k_y, offs, layout, r_surf, tag=tag,
            order_branch=prefer, force_branch=force,
        )
        y = float(metrics["gt_miou"])
        if metrics.get("agent_error"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if y > best_y:
            best_y = y
            best_row = dict(metrics)

        row: dict[str, Any] = {
            "trial_id": trial_idx,
            "tunnel_id": ctx.tunnel_id,
            "ring_id": ctx.ring_id,
            "case_id": ctx.case_id,
            "experience_stream": "d",
            "ring_is_regular": bool(ctx.ring_is_regular),
            "kind": kind,
            "k_y": k_y,
            "per_ring_offsets": json.dumps({"0": offs}),
            "r_surface_min_ceiling_ref": ctx.ceiling_r_surface_min,
            "gt_miou": y,
            "best_so_far": best_y,
            "regret_vs_ceiling": ctx.ceiling_miou - best_y if ctx.ceiling_miou else None,
            "search_x": "[]",
            **layout_params_for_log(layout),
        }
        row.update(metrics)
        trial_rows.append(row)
        regret_s = f"{ctx.ceiling_miou - best_y:.4f}" if ctx.ceiling_miou else "n/a"
        print(f"  trial{trial_idx:03d} [{kind}] branch={row.get('order_branch')} miou={y:.4f} best={best_y:.4f} regret={regret_s}")

    if resume and (ctx.out_dir / "bo_trials.csv").exists():
        old_df = pd.read_csv(ctx.out_dir / "bo_trials.csv")
        trials_df = pd.concat([old_df, pd.DataFrame(trial_rows)], ignore_index=True)
    else:
        trials_df = pd.DataFrame(trial_rows)

    trials_path = ctx.out_dir / "bo_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    conv_cols = [c for c in [
        "trial_id", "kind", "gt_miou", "best_so_far", "regret_vs_ceiling",
        "order_branch", "gt_miou_plus", "gt_miou_minus", "direction_margin",
    ] if c in trials_df.columns]
    trials_df[conv_cols].to_csv(ctx.out_dir / "convergence.csv", index=False)

    k_y_b, offs_b, layout_b, r_surf_b = decode_x(ctx, frozen_x)
    best_payload = {
        "case_id": ctx.case_id,
        "best_bo_miou": best_y,
        "best_k_y": k_y_b,
        "best_offsets": offs_b,
        "best_layout_params": layout_b,
        "best_r_surface_min": r_surf_b,
        "r_surface_min_ceiling_ref": ctx.ceiling_r_surface_min,
        "n_evals": int(len(trials_df)),
        "ceiling_miou_reference": ctx.ceiling_miou,
        "order_branch": best_row.get("order_branch") if best_row else prefer,
        "direction_selected_branch": best_row.get("order_branch") if best_row else None,
        "gt_miou_plus": best_row.get("gt_miou_plus") if best_row else None,
        "gt_miou_minus": best_row.get("gt_miou_minus") if best_row else None,
    }
    (ctx.out_dir / "best_bo_trial.json").write_text(json.dumps(best_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "trials_df": trials_df,
        "best_payload": best_payload,
        "best_row": best_row,
        "oracle_ceiling_miou": ctx.ceiling_miou if ctx.ceiling_miou else None,
        "chunk_trials": len(trial_rows),
        "agent_error_stop": consecutive_errors > 5,
    }


def run_gp_bo(
    ctx: RingContext,
    *,
    n_evals: int = 64,
    seed: int = 7,
    order_branch: str = "plus",
    resume: bool = False,
    gt_layout: dict[str, Any] | None = None,
    ceiling: dict[str, Any] | None = None,
    experience_mode: bool = False,
) -> dict[str, Any]:
    """GP-BO + EI over layout-recovery search space. Appends n_evals new trials (resume skips priors)."""
    rng = np.random.default_rng(seed)
    dim = ctx.search_dim
    write_search_space_spec(ctx)

    if resume:
        X, Y, prior_rows, idx, best_y = load_trial_history(ctx)
        best_x = None
        best_row: dict[str, Any] | None = None
        if prior_rows:
            best_idx = int(np.argmax(Y))
            best_x = X[best_idx]
            best_row = prior_rows[best_idx]
        trial_rows: list[dict[str, Any]] = []
    else:
        X, Y, trial_rows = [], [], []
        idx, best_y = 0, -1.0
        best_x, best_row = None, None

    end_idx = idx + n_evals
    consecutive_errors = 0

    def record(x: np.ndarray, trial_idx: int, kind: str, gp_meta: dict[str, Any] | None = None) -> bool:
        nonlocal best_y, best_x, best_row, consecutive_errors
        k_y, offs, layout, r_surf = decode_x(ctx, x)
        tag = f"trial{trial_idx:03d}"
        metrics = evaluate_trial(ctx, k_y, offs, layout, r_surf, tag=tag, order_branch=order_branch)
        y = float(metrics["gt_miou"])
        if metrics.get("agent_error"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        X.append(np.asarray(x, dtype=float))
        Y.append(y)
        if y > best_y:
            best_y, best_x = y, np.asarray(x, dtype=float)
            best_row = dict(metrics)

        row: dict[str, Any] = {
            "trial_id": trial_idx,
            "tunnel_id": ctx.tunnel_id,
            "ring_id": ctx.ring_id,
            "case_id": ctx.case_id,
            "experience_stream": ctx.experience_stream,
            "ring_is_regular": bool(ctx.ring_is_regular),
            "kind": kind,
            "k_y": k_y,
            "per_ring_offsets": json.dumps({"0": offs}),
            "r_surface_min_ceiling_ref": ctx.ceiling_r_surface_min,
            "gt_miou": y,
            "best_so_far": best_y,
            "regret_vs_ceiling": ctx.ceiling_miou - best_y if ctx.ceiling_miou else None,
            "search_x": json.dumps(x.tolist()),
            **layout_params_for_log(layout),
        }
        row.update(metrics)
        if gp_meta:
            row.update(gp_meta)
        trial_rows.append(row)
        regret_s = f"{ctx.ceiling_miou - best_y:.4f}" if ctx.ceiling_miou else "n/a"
        print(f"  trial{trial_idx:03d} [{kind}] miou={y:.4f} best={best_y:.4f} regret={regret_s}")
        return consecutive_errors > 5

    if not resume:
        if gt_layout is None:
            gt_path = ctx.out_dir / "gt_layout.json"
            gt_layout = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
        if ceiling is None:
            ceil_path = ctx.out_dir / "ceiling.json"
            ceiling = json.loads(ceil_path.read_text(encoding="utf-8")) if ceil_path.exists() else {}

        if experience_mode:
            budgets = experience_phase_budgets(n_evals)
            warm_end = idx + budgets["warm"]
            perturb_end = warm_end + budgets["perturb"]
            gp_end = end_idx

            for x, kind in experience_warm_seeds(ctx, rng):
                if idx >= warm_end:
                    break
                if record(x, idx, kind):
                    break
                idx += 1

            while idx < warm_end:
                if record(rng.random(dim), idx, "random"):
                    break
                idx += 1

            if ctx.warm_anchor == "gt_derived":
                ref_x = gt_derived_reference_x(ctx)
            else:
                ref_x = best_x if best_x is not None else default_perturbation_reference_x(ctx, rng)
            for x, kind in forced_perturbation_candidates(
                ctx, ref_x, rng=rng, n_target=budgets["perturb"]
            ):
                if idx >= perturb_end:
                    break
                if record(x, idx, kind):
                    break
                idx += 1

            while idx < perturb_end:
                if record(rng.random(dim), idx, "perturb_fill"):
                    break
                idx += 1

            end_idx = gp_end
        else:
            for x, kind in ceiling_push_priors(ctx):
                if idx >= end_idx:
                    break
                if record(x, idx, kind):
                    break
                idx += 1

            n_random_target = min(32, max(0, (end_idx - idx) // 8), max(0, end_idx - idx))
            for _ in range(n_random_target):
                if idx >= end_idx:
                    break
                if record(rng.random(dim), idx, "random"):
                    break
                idx += 1

    while idx < end_idx:
        if len(X) < 2:
            if record(rng.random(dim), idx, "random"):
                break
            idx += 1
            continue
        gp = make_gp(seed, dim)
        X_arr = np.vstack(X)
        y_arr = np.asarray(Y)
        gp.fit(X_arr, y_arr)
        pool = rng.random((4096, dim))
        mu, std = gp.predict(pool, return_std=True)
        ei = expected_improvement(mu, std, y_best=float(np.max(y_arr)))
        j = int(np.argmax(ei))
        chosen = pool[j]
        gp_meta = {
            "bo_surrogate_mean": float(mu[j]),
            "bo_surrogate_std": float(std[j]),
            "ei_value": float(ei[j]),
        }
        if record(chosen, idx, "bo", gp_meta=gp_meta):
            break
        idx += 1

    if resume and (ctx.out_dir / "bo_trials.csv").exists():
        old_df = pd.read_csv(ctx.out_dir / "bo_trials.csv")
        new_df = pd.DataFrame(trial_rows)
        trials_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        trials_df = pd.DataFrame(trial_rows)

    trials_path = ctx.out_dir / "bo_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    conv_cols = [
        "trial_id", "kind", "gt_miou", "best_so_far", "regret_vs_ceiling",
        "k_y", "r_surface_min", "hough_oblique_threshold", "hough_horizontal_threshold",
        "line_merge_distance", "line_snap_tolerance_px", "segmentation_slot_inset_y",
        "r_surface_min_ceiling_ref",
    ]
    conv_cols = [c for c in conv_cols if c in trials_df.columns]
    trials_df[conv_cols].to_csv(ctx.out_dir / "convergence.csv", index=False)

    k_y_b, offs_b, layout_b, r_surf_b = decode_x(ctx, best_x) if best_x is not None else (0.0, {}, {}, 0.0)
    best_payload = {
        "case_id": ctx.case_id,
        "best_bo_miou": best_y,
        "best_k_y": k_y_b,
        "best_offsets": offs_b,
        "best_layout_params": layout_b,
        "best_r_surface_min": r_surf_b,
        "r_surface_min_ceiling_ref": ctx.ceiling_r_surface_min,
        "n_evals": int(len(trials_df)),
        "ceiling_miou_reference": ctx.ceiling_miou,
        "order_branch": best_row.get("order_branch", order_branch) if best_row else order_branch,
        "direction_selected_branch": best_row.get("order_branch") if best_row else None,
    }
    (ctx.out_dir / "best_bo_trial.json").write_text(json.dumps(best_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "trials_df": trials_df,
        "best_payload": best_payload,
        "best_row": best_row,
        "oracle_ceiling_miou": ctx.ceiling_miou if ctx.ceiling_miou else None,
        "chunk_trials": len(trial_rows),
        "agent_error_stop": consecutive_errors > 5,
    }


def run_iterative_ceiling_push(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    segment_count: int | None = None,
    manifest_entry: dict | None = None,
    order_branch: str = "plus",
    eval_chunk: int = 128,
    max_total_evals: int = 1024,
    target_regret: float = 0.05,
    min_improvement: float = 0.005,
    seed: int = 7,
    skip_ceiling_if_exists: bool = True,
) -> dict[str, Any]:
    """Iteratively run BO chunks until target regret, ceiling_bind, or max budget."""
    ctx = build_ring_context(
        tunnel_id,
        ring_id,
        source_root=source_root,
        run_root=run_root,
        segment_count=segment_count,
        manifest_entry=manifest_entry,
    )
    ceil_path = ctx.out_dir / "ceiling.json"
    if skip_ceiling_if_exists and ceil_path.exists():
        ceiling = json.loads(ceil_path.read_text(encoding="utf-8"))
        ctx.ceiling_miou = float(ceiling.get("agents_gt_ceiling_miou") or 0.0)
        ctx.ceiling_r_surface_min = ceiling.get("r_surface_min_selected")
        print(f"== ceiling reference (cached): {ctx.case_id} = {ctx.ceiling_miou:.4f} ==")
    else:
        print(f"== ceiling reference: {ctx.case_id} ==")
        ceiling = compute_ceiling_reference(ctx)
        print(f"  ceiling mIoU (reference) = {ctx.ceiling_miou:.4f}")

    gt_path = ctx.out_dir / "gt_layout.json"
    gt_layout = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

    target_miou = float(ctx.ceiling_miou) - float(target_regret)
    iteration_summaries: list[dict[str, Any]] = []
    stop_reason = "max_budget"
    n_iterations = 0
    while True:
        existing = 0
        if (ctx.out_dir / "bo_trials.csv").exists():
            existing = len(pd.read_csv(ctx.out_dir / "bo_trials.csv"))
        if existing >= max_total_evals:
            stop_reason = "max_budget"
            break

        chunk = min(eval_chunk, max_total_evals - existing)
        if chunk <= 0:
            stop_reason = "max_budget"
            break

        n_iterations += 1
        resume = existing > 0
        print(f"== iteration {n_iterations}: +{chunk} evals (total cap {max_total_evals}, resume={resume}): {ctx.case_id} ==")
        prev_best = -1.0
        if resume:
            _, _, _, _, prev_best = load_trial_history(ctx)

        bo_result = run_gp_bo(
            ctx,
            n_evals=chunk,
            seed=seed + n_iterations,
            order_branch=order_branch,
            resume=resume,
            gt_layout=gt_layout if not resume else None,
            ceiling=ceiling if not resume else None,
        )

        trials_df = bo_result["trials_df"]
        best_bo = float(trials_df["gt_miou"].max())
        regret = float(ctx.ceiling_miou) - best_bo if ctx.ceiling_miou else None
        chunk_gain = best_bo - prev_best if prev_best >= 0 else best_bo

        iter_dir = ctx.out_dir / f"iteration_{n_iterations:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        chunk_df = pd.DataFrame(trials_df.tail(bo_result["chunk_trials"]))
        chunk_df.to_csv(iter_dir / "chunk_trials.csv", index=False)
        iter_summary = {
            "iteration": n_iterations,
            "chunk_evals": chunk,
            "total_evals": int(len(trials_df)),
            "best_bo_miou": round(best_bo, 4),
            "regret_vs_ceiling": round(regret, 4) if regret is not None else None,
            "chunk_gain": round(chunk_gain, 4),
            "target_miou": round(target_miou, 4),
        }
        (iter_dir / "iteration_summary.json").write_text(json.dumps(iter_summary, indent=2) + "\n", encoding="utf-8")
        iteration_summaries.append(iter_summary)

        if bo_result.get("agent_error_stop"):
            stop_reason = "agent_errors"
            break
        if best_bo >= target_miou:
            stop_reason = "target_reached"
            break
        if regret is not None and regret <= target_regret and chunk_gain < min_improvement:
            stop_reason = "ceiling_bind"
            break
        if len(trials_df) >= max_total_evals:
            stop_reason = "max_budget"
            break

    trials_df = pd.read_csv(ctx.out_dir / "bo_trials.csv") if (ctx.out_dir / "bo_trials.csv").exists() else pd.DataFrame()
    report = write_ceiling_push_report(
        ctx,
        trials_df,
        target_regret=target_regret,
        stop_reason=stop_reason,
        n_iterations=n_iterations,
        order_branch=order_branch,
        oracle_ceiling_miou=ctx.ceiling_miou if ctx.ceiling_miou else None,
    )
    return {
        "ctx": ctx,
        "report": report,
        "iteration_summaries": iteration_summaries,
        "trials_df": trials_df,
    }


def write_experience_gate(
    ctx: RingContext,
    trials_df: pd.DataFrame,
    bo_result: dict[str, Any],
    *,
    target_n_evals: int | None = None,
) -> dict[str, Any]:
    if ctx.warm_anchor == "gt_derived":
        return _write_gt_experience_gate(ctx, trials_df, bo_result, target_n_evals=target_n_evals)
    return _write_honest_experience_gate(ctx, trials_df, bo_result, target_n_evals=target_n_evals)


def _write_gt_experience_gate(
    ctx: RingContext,
    trials_df: pd.DataFrame,
    bo_result: dict[str, Any],
    *,
    target_n_evals: int | None = None,
) -> dict[str, Any]:
    miou_std = float(trials_df["gt_miou"].std()) if len(trials_df) > 1 else 0.0
    gt_ceiling_rows = trials_df[trials_df["kind"] == "gt_layout_ceiling_r"]
    gt_ceiling_miou = float(gt_ceiling_rows["gt_miou"].iloc[0]) if not gt_ceiling_rows.empty else None
    n_oracle = int(trials_df["kind"].isin(ORACLE_TRIAL_KINDS).sum()) if "kind" in trials_df.columns else 0
    dir_ok = bool(
        "direction_select_enabled" in trials_df.columns
        and trials_df["direction_select_enabled"].fillna(False).astype(bool).all()
    )
    perturb_kinds = trials_df[trials_df["kind"].astype(str).str.startswith("perturb")]
    n_perturb = int(len(perturb_kinds))
    perturb_frac = n_perturb / max(len(trials_df), 1)
    intrinsic_cols = [c for c in trials_df.columns if c.startswith("det_") or c in (
        "k_y_frac", "arc_width_entropy", "n_reclassified_by_r_filter",
    )]
    has_intrinsic = any(trials_df[c].notna().any() for c in intrinsic_cols if c in trials_df.columns)
    best_bo = float(trials_df["gt_miou"].max())
    min_trials = target_n_evals if target_n_evals is not None else 64
    min_perturb_frac = 0.15 if target_n_evals is not None and target_n_evals >= 12 else 0.0

    gate = {
        "case_id": ctx.case_id,
        "objective": "GT-labeled experience collection (GT-anchor warm-start)",
        "warm_anchor": ctx.warm_anchor,
        "n_oracle_trials": n_oracle,
        "target_n_evals": target_n_evals,
        "n_evals": int(len(trials_df)),
        "n_perturb_trials": n_perturb,
        "perturb_trial_frac": round(perturb_frac, 4),
        "ceiling_miou_reference": ctx.ceiling_miou,
        "gt_layout_ceiling_r_miou": gt_ceiling_miou,
        "best_bo_miou": best_bo,
        "regret_vs_ceiling": ctx.ceiling_miou - best_bo if ctx.ceiling_miou else None,
        "miou_std": miou_std,
        "criteria": {
            "n_trials_complete": bool(len(trials_df) >= min_trials),
            "trial_schema_ok": bool(
                "gt_miou" in trials_df.columns
                and "r_surface_min" in trials_df.columns
                and "hough_oblique_threshold" in trials_df.columns
            ),
            "miou_spread_ok": bool(miou_std > 0.02),
            "gt_ceiling_warm_present": bool(not gt_ceiling_rows.empty),
            "gt_ceiling_warm_miou_ok": bool(gt_ceiling_miou is not None and gt_ceiling_miou >= 0.80),
            "intrinsics_populated": bool(has_intrinsic),
            "perturb_fraction_ok": bool(perturb_frac >= min_perturb_frac),
            "direction_select_populated": bool(dir_ok),
        },
        "passed": bool(
            len(trials_df) >= min_trials
            and miou_std > 0.02
            and not gt_ceiling_rows.empty
            and gt_ceiling_miou is not None
            and gt_ceiling_miou >= 0.80
            and has_intrinsic
            and perturb_frac >= min_perturb_frac
            and dir_ok
        ),
        "outputs": {
            "bo_trials": str(ctx.out_dir / "bo_trials.csv"),
            "best_bo_trial": str(ctx.out_dir / "best_bo_trial.json"),
            "ceiling": str(ctx.out_dir / "ceiling.json"),
        },
    }
    (ctx.out_dir / "experience_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def _write_honest_experience_gate(
    ctx: RingContext,
    trials_df: pd.DataFrame,
    bo_result: dict[str, Any],
    *,
    target_n_evals: int | None = None,
) -> dict[str, Any]:
    miou_std = float(trials_df["gt_miou"].std()) if len(trials_df) > 1 else 0.0
    warm_kinds = {
        "prior", "random", "geometric_0", "geometric_1", "intrinsic_r_otsu", "sam4tun_static",
    }
    random_phase = trials_df[trials_df["kind"].isin(warm_kinds)]
    n_oracle = int(trials_df["kind"].isin(ORACLE_TRIAL_KINDS).sum()) if "kind" in trials_df.columns else 0
    dir_ok = bool(
        "direction_select_enabled" in trials_df.columns
        and trials_df["direction_select_enabled"].fillna(False).astype(bool).all()
    )
    perturb_kinds = trials_df[trials_df["kind"].astype(str).str.startswith("perturb")]
    n_perturb = int(len(perturb_kinds))
    perturb_frac = n_perturb / max(len(trials_df), 1)
    intrinsic_cols = [c for c in trials_df.columns if c.startswith("det_") or c in (
        "k_y_frac", "arc_width_entropy", "n_reclassified_by_r_filter",
    )]
    has_intrinsic = any(trials_df[c].notna().any() for c in intrinsic_cols if c in trials_df.columns)

    best_bo = float(trials_df["gt_miou"].max())
    random_median = float(random_phase["gt_miou"].median()) if not random_phase.empty else 0.0
    min_trials = target_n_evals if target_n_evals is not None else 64
    min_perturb_frac = 0.15 if target_n_evals is not None and target_n_evals >= 12 else 0.0

    if ctx.experience_stream == "d":
        base = trials_df[trials_df["kind"] == "twin_baseline"]
        twin_spread = 0.0
        if not base.empty and "gt_miou_plus" in base.columns:
            mp, mm = base.iloc[0].get("gt_miou_plus"), base.iloc[0].get("gt_miou_minus")
            if pd.notna(mp) and pd.notna(mm):
                twin_spread = abs(float(mp) - float(mm))
        n_force = int(trials_df["kind"].isin({"force_plus", "force_minus"}).sum())
        gate = {
            "case_id": ctx.case_id,
            "experience_stream": ctx.experience_stream,
            "ring_is_regular": bool(ctx.ring_is_regular),
            "objective": "honest order-axis experience (frozen L+K; plus/minus twin eval)",
            "n_oracle_trials": n_oracle,
            "target_n_evals": target_n_evals,
            "n_evals": int(len(trials_df)),
            "ceiling_miou_reference": ctx.ceiling_miou,
            "best_bo_miou": best_bo,
            "regret_vs_ceiling": ctx.ceiling_miou - best_bo if ctx.ceiling_miou else None,
            "miou_std": miou_std,
            "twin_miou_spread_baseline": twin_spread,
            "criteria": {
                "n_trials_complete": bool(len(trials_df) >= min_trials),
                "trial_schema_ok": bool(
                    "gt_miou" in trials_df.columns
                    and "gt_miou_plus" in trials_df.columns
                    and "order_branch" in trials_df.columns
                ),
                "miou_spread_ok": bool(miou_std > 0.02),
                "twin_spread_ok": bool(twin_spread >= 0.02),
                "force_branch_trials_ok": bool(n_force >= max(2, min_trials // 4)),
                "no_oracle_trials": bool(n_oracle == 0),
                "direction_select_populated": bool(dir_ok),
                "intrinsics_populated": bool(has_intrinsic),
            },
            "passed": bool(
                len(trials_df) >= min_trials
                and miou_std > 0.02
                and twin_spread >= 0.02
                and n_force >= max(2, min_trials // 4)
                and n_oracle == 0
                and dir_ok
                and has_intrinsic
            ),
            "outputs": {
                "bo_trials": str(ctx.out_dir / "bo_trials.csv"),
                "best_bo_trial": str(ctx.out_dir / "best_bo_trial.json"),
                "order_best_for_v6": str(ctx.out_dir / "order_best_for_v6.json"),
            },
        }
        (ctx.out_dir / "experience_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        return gate

    gate = {
        "case_id": ctx.case_id,
        "experience_stream": ctx.experience_stream,
        "ring_is_regular": bool(ctx.ring_is_regular),
        "objective": "honest GT-labeled experience collection (no oracle layout trials)",
        "n_oracle_trials": n_oracle,
        "target_n_evals": target_n_evals,
        "n_evals": int(len(trials_df)),
        "n_perturb_trials": n_perturb,
        "perturb_trial_frac": round(perturb_frac, 4),
        "ceiling_miou_reference": ctx.ceiling_miou,
        "best_bo_miou": best_bo,
        "regret_vs_ceiling": ctx.ceiling_miou - best_bo if ctx.ceiling_miou else None,
        "miou_std": miou_std,
        "random_phase_median_miou": random_median,
        "bo_beats_random_median": bool(best_bo > random_median),
        "criteria": {
            "n_trials_complete": bool(len(trials_df) >= min_trials),
            "trial_schema_ok": bool(
                "gt_miou" in trials_df.columns
                and "r_surface_min" in trials_df.columns
                and "hough_oblique_threshold" in trials_df.columns
                and "line_snap_tolerance_px" in trials_df.columns
            ),
            "miou_spread_ok": bool(miou_std > 0.02),
            "bo_improves_over_random": bool(best_bo > random_median),
            "intrinsics_populated": bool(has_intrinsic),
            "perturb_fraction_ok": bool(perturb_frac >= min_perturb_frac),
            "no_oracle_trials": bool(n_oracle == 0),
            "direction_select_populated": bool(dir_ok),
        },
        "passed": bool(
            len(trials_df) >= min_trials
            and miou_std > 0.02
            and best_bo > random_median
            and has_intrinsic
            and perturb_frac >= min_perturb_frac
            and n_oracle == 0
            and dir_ok
        ),
        "outputs": {
            "bo_trials": str(ctx.out_dir / "bo_trials.csv"),
            "best_bo_trial": str(ctx.out_dir / "best_bo_trial.json"),
            "ceiling": str(ctx.out_dir / "ceiling.json"),
        },
    }
    (ctx.out_dir / "experience_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def write_panel_ceiling_push_summary(run_root: Path, ring_results: list[dict[str, Any]]) -> None:
    """Write panel_summary.csv, iteration_log.json, merged bo_trials.csv, ceiling_push_summary.md."""
    run_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    iteration_log: dict[str, Any] = {}
    all_trials: list[pd.DataFrame] = []

    for res in ring_results:
        report = res["report"]
        ctx = res["ctx"]
        ring_key = ctx.case_id
        summaries.append({
            "ring_key": ring_key,
            "segment_count": ctx.segment_count,
            "order_branch": report.get("order_branch"),
            "ceiling_reference": report.get("ceiling_miou_reference"),
            "target_miou": report.get("target_miou"),
            "target_regret": report.get("target_regret"),
            "best_bo_miou": report.get("best_bo_miou"),
            "regret_vs_ceiling": report.get("regret_vs_ceiling"),
            "target_reached": report.get("target_reached"),
            "stop_reason": report.get("stop_reason"),
            "total_evals": report.get("total_evals"),
            "n_iterations": report.get("n_iterations"),
        })
        iteration_log[ring_key] = res.get("iteration_summaries", [])
        trials_path = ctx.out_dir / "bo_trials.csv"
        if trials_path.exists():
            all_trials.append(pd.read_csv(trials_path))

    panel_df = pd.DataFrame(summaries)
    panel_df.to_csv(run_root / "panel_summary.csv", index=False)
    (run_root / "iteration_log.json").write_text(json.dumps(iteration_log, indent=2) + "\n", encoding="utf-8")

    if all_trials:
        merged = pd.concat(all_trials, ignore_index=True)
        merged.to_csv(run_root / "bo_trials.csv", index=False)

    lines = [
        "# Ceiling-push BO summary",
        "",
        f"Run root: `{run_root.relative_to(REPO_ROOT)}`",
        "",
        "| Ring | Ceiling | Target | Best BO | Regret | Stop | Evals | Iters |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['ring_key']} | {s['ceiling_reference']} | {s['target_miou']} | {s['best_bo_miou']} | "
            f"{s['regret_vs_ceiling']} | {s['stop_reason']} | {s['total_evals']} | {s['n_iterations']} |"
        )
    lines.extend(["", "## Targets", "", "- Default success: regret ≤ 0.05 vs GT ceiling", ""])
    (run_root / "ceiling_push_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_order_best_for_v6(ctx: RingContext, best_payload: dict[str, Any]) -> Path:
    out = {
        "case_id": ctx.case_id,
        "experience_stream": "d",
        "ring_is_regular": bool(ctx.ring_is_regular),
        "order_branch": best_payload.get("order_branch"),
        "gt_miou": best_payload.get("best_bo_miou"),
        "gt_miou_plus": best_payload.get("gt_miou_plus"),
        "gt_miou_minus": best_payload.get("gt_miou_minus"),
        "k_handoff": str(ctx.k_handoff_path) if ctx.k_handoff_path else None,
        "direction_tier_gt": ctx.direction_tier_gt,
        "k_y": best_payload.get("best_k_y"),
        "per_ring_offsets": best_payload.get("best_offsets"),
        "layout_params": best_payload.get("best_layout_params"),
        "r_surface_min": best_payload.get("best_r_surface_min"),
        "best_bo_miou": best_payload.get("best_bo_miou"),
        "source": str(ctx.out_dir / "best_bo_trial.json"),
    }
    path = ctx.out_dir / "order_best_for_v6.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path


def write_k_best_for_stream_d(ctx: RingContext, best_payload: dict[str, Any]) -> Path:
    k_y = best_payload.get("best_k_y")
    k_frac = float((k_y % ctx.H) / max(ctx.H, 1)) if k_y is not None else None
    out = {
        "case_id": ctx.case_id,
        "experience_stream": "k",
        "ring_is_regular": bool(ctx.ring_is_regular),
        "k_y": k_y,
        "k_y_frac": k_frac,
        "layout_handoff": str(ctx.layout_handoff_path) if ctx.layout_handoff_path else None,
        "per_ring_offsets": best_payload.get("best_offsets"),
        "layout_params": best_payload.get("best_layout_params"),
        "r_surface_min": best_payload.get("best_r_surface_min"),
        "best_bo_miou": best_payload.get("best_bo_miou"),
        "source": str(ctx.out_dir / "best_bo_trial.json"),
    }
    path = ctx.out_dir / "k_best_for_stream_d.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path


def write_ring_regular_manifest(run_root: Path, entries: list[dict[str, Any]]) -> Path:
    path = run_root / "ring_regular_manifest.json"
    path.write_text(json.dumps({"rings": entries}, indent=2) + "\n", encoding="utf-8")
    return path


def write_layout_best_for_stream_k(ctx: RingContext, best_payload: dict[str, Any]) -> Path:
    """Handoff JSON for Stream K (layout frozen; k_y from SAM4Tun in layout-only BO)."""
    out = {
        "case_id": ctx.case_id,
        "experience_stream": "layout",
        "frozen_k_y_frac": ctx.frozen_k_y_frac,
        "k_y": best_payload.get("best_k_y"),
        "per_ring_offsets": best_payload.get("best_offsets"),
        "layout_params": best_payload.get("best_layout_params"),
        "r_surface_min": best_payload.get("best_r_surface_min"),
        "best_bo_miou": best_payload.get("best_bo_miou"),
        "source": str(ctx.out_dir / "best_bo_trial.json"),
    }
    path = ctx.out_dir / "layout_best_for_stream_k.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path


def run_ring_bo(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    n_evals: int = 64,
    seed: int = 7,
    segment_count: int | None = None,
    tunnel_diameter: float | None = None,
    manifest_entry: dict | None = None,
    order_branch: str = "plus",
    prior_root: Path | None = None,
    warm_anchor: str = "sam4tun",
    experience_stream: str = "full",
    layout_handoff_root: Path | None = None,
    k_handoff_root: Path | None = None,
) -> dict[str, Any]:
    ctx = build_ring_context(
        tunnel_id,
        ring_id,
        source_root=source_root,
        run_root=run_root,
        segment_count=segment_count,
        tunnel_diameter=tunnel_diameter,
        manifest_entry=manifest_entry,
        prior_root=prior_root,
        warm_anchor=warm_anchor,
        experience_stream=experience_stream,
        layout_handoff_root=layout_handoff_root,
        k_handoff_root=k_handoff_root,
    )
    ceil_src = ctx.src_ring / "ceiling.json"
    gt_src = ctx.src_ring / "gt_layout.json"
    if gt_src.exists():
        (ctx.out_dir / "gt_layout.json").write_text(gt_src.read_text(encoding="utf-8"), encoding="utf-8")

    if ceil_src.exists():
        ceiling = json.loads(ceil_src.read_text(encoding="utf-8"))
        ctx.ceiling_miou = float(ceiling.get("agents_gt_ceiling_miou") or 0.0)
        ctx.ceiling_r_surface_min = ceiling.get("r_surface_min_selected")
        (ctx.out_dir / "ceiling.json").write_text(json.dumps(ceiling, indent=2) + "\n", encoding="utf-8")
        print(f"== ceiling reference (corpus): {ctx.case_id} = {ctx.ceiling_miou:.4f} ==")
    else:
        print(f"== ceiling reference: {ctx.case_id} ==")
        ceiling = compute_ceiling_reference(ctx)
        print(f"  ceiling mIoU (reference) = {ctx.ceiling_miou:.4f}")

    gt_path = ctx.out_dir / "gt_layout.json"
    gt_layout = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

    stream_s = f", stream={experience_stream}" if experience_stream != "full" else ""
    print(
        f"== experience BO ({n_evals} evals, anchor={warm_anchor}{stream_s}, "
        f"direction_select per trial): {ctx.case_id} =="
    )
    if experience_stream == "layout" and ctx.frozen_k_y_frac is not None:
        print(f"  frozen k_y_frac={ctx.frozen_k_y_frac:.4f}")
    if experience_stream == "k":
        print(
            f"  ring_is_regular={ctx.ring_is_regular} "
            f"sam_k_frac={ctx.sam_k_y_frac} layout_handoff={ctx.layout_handoff_path}"
        )
    if experience_stream == "d":
        print(
            f"  ring_is_regular={ctx.ring_is_regular} "
            f"k_handoff={ctx.k_handoff_path} direction_tier_gt={ctx.direction_tier_gt}"
        )
    if experience_stream == "d":
        bo_result = run_order_stream_bo(ctx, n_evals=n_evals, seed=seed, resume=True)
    else:
        bo_result = run_gp_bo(
            ctx,
            n_evals=n_evals,
            seed=seed,
            order_branch=order_branch,
            ceiling=ceiling,
            experience_mode=True,
        )
    gate = write_experience_gate(ctx, bo_result["trials_df"], bo_result, target_n_evals=n_evals)
    if experience_stream == "layout":
        handoff = write_layout_best_for_stream_k(ctx, bo_result["best_payload"])
        print(f"  wrote {handoff}")
    if experience_stream == "k":
        handoff = write_k_best_for_stream_d(ctx, bo_result["best_payload"])
        print(f"  wrote {handoff}")
    if experience_stream == "d":
        handoff = write_order_best_for_v6(ctx, bo_result["best_payload"])
        print(f"  wrote {handoff}")
    print(f"== experience gate: passed={gate['passed']} ==")
    return {"ctx": ctx, "gate": gate, "bo_result": bo_result}
