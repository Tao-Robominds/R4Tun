"""Line-derived K/A/B layout anchor (GT-free)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lib.layout_bo import RingContext, decode_x, geometric_priors, widths_to_offset_fracs
from lib.line_reliability import LineEvidence
from lib.sam4tun_prior import (
    centre_to_boundary,
    load_preproc_meta,
    sam4tun_layout_params_from_default,
)


@dataclass
class LineAnchor:
    valid: bool
    k_y: float
    k_center_norm: float
    k_width_norm: float
    offsets: dict[str, float]
    ab_offset_norm: dict[str, float]
    layout_params: dict[str, float]
    r_surface_min: float
    search_x: np.ndarray
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "k_y": self.k_y,
            "k_center_norm": self.k_center_norm,
            "k_width_norm": self.k_width_norm,
            "offsets": self.offsets,
            "ab_offset_norm": self.ab_offset_norm,
            "reason": self.reason,
        }


def _equal_offsets(ctx: RingContext, k_y: float, k_width_px: float) -> dict[str, float]:
    H = ctx.H
    n = ctx.segment_count
    ab_w = max((H - k_width_px) / max(n - 1, 1), 1.0)
    offsets = {ctx.blocks[0]: 0.0}
    pos = k_y + k_width_px
    for block in ctx.blocks[1:]:
        offsets[block] = float(pos % H)
        pos += ab_w
    return offsets


def build_line_anchor(ctx: RingContext, evidence: LineEvidence, *, sam_k_y: float, sam_layout: dict[str, float], sam_r: float) -> LineAnchor:
    H = ctx.H
    meta = load_preproc_meta(ctx.src_ring, ctx.segment_count)
    k_width = meta.k_height_px

    if not evidence.valid_line_anchor or evidence.k_y is None:
        x = geometric_priors(ctx)[0]
        k_y, offs, layout, r_s = decode_x(ctx, x)
        return LineAnchor(
            valid=False,
            k_y=k_y,
            k_center_norm=float(k_y / max(H, 1)),
            k_width_norm=float(k_width / max(H, 1)),
            offsets=offs,
            ab_offset_norm={b: float(v % H / max(H, 1)) for b, v in offs.items()},
            layout_params=layout,
            r_surface_min=r_s,
            search_x=x,
            reason="rho_K below threshold or missing K detection",
        )

    k_centre = float(evidence.k_y)
    k_type_l = (evidence.k_type or "").lower()
    if "center" in k_type_l or "centre" in k_type_l or "midpoint" in k_type_l:
        k_y = centre_to_boundary(k_centre, k_width, H)
    else:
        k_y = float(k_centre) % H

    horiz = sorted(evidence.horizontal_y)
    if len(horiz) >= ctx.segment_count - 1:
        ys = [0.0] + [float(y % H) for y in horiz[: ctx.segment_count - 1]]
        ys = sorted(ys)
        widths = []
        for i in range(len(ctx.blocks)):
            w = (ys[(i + 1) % len(ys)] - ys[i]) % H if i + 1 < len(ys) else (H / len(ctx.blocks))
            widths.append(max(w, 0.04 * H))
        widths_arr = np.array(widths[: len(ctx.blocks)], dtype=float)
        if widths_arr.sum() > 0:
            widths_arr = widths_arr / widths_arr.sum() * H
        off_fracs = widths_to_offset_fracs(ctx.blocks, widths_arr / H)
        offsets = {ctx.blocks[i]: float(off_fracs[i] % 1.0) * H for i in range(len(ctx.blocks))}
        offsets[ctx.blocks[0]] = 0.0
    else:
        offsets = _equal_offsets(ctx, k_y, k_width)

    layout = dict(sam_layout)
    from lib.sam4tun_prior import encode_sam4tun_x

    x = encode_sam4tun_x(ctx, k_y, offsets, layout, sam_r)
    ab_norm = {b: float(offsets[b] % H / max(H, 1)) for b in ctx.blocks}
    return LineAnchor(
        valid=True,
        k_y=float(k_y),
        k_center_norm=float(k_y / max(H, 1)),
        k_width_norm=float(k_width / max(H, 1)),
        offsets=offsets,
        ab_offset_norm=ab_norm,
        layout_params=layout,
        r_surface_min=sam_r,
        search_x=np.asarray(x, dtype=float),
        reason="line_derived",
    )
