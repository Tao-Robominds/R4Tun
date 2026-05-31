"""Pre-flight GT layout encode/decode round-trip smoke test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lib.layout_bo import (
    build_ring_context,
    compute_ceiling_reference,
    decode_x,
    encode_gt_layout_x,
    evaluate_trial,
)


def verify_ring(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    order_branch: str,
    segment_count: int | None,
    tol_px: float = 1.0,
    miou_tol: float = 0.05,
) -> dict[str, Any]:
    ctx = build_ring_context(
        tunnel_id, ring_id, source_root=source_root, run_root=run_root, segment_count=segment_count
    )
    ceiling = compute_ceiling_reference(ctx)
    gt = json.loads((ctx.out_dir / "gt_layout.json").read_text(encoding="utf-8"))
    r = ceiling.get("r_surface_min_selected") or ctx.r_lo
    x = encode_gt_layout_x(ctx, gt, float(r))
    ky, dec_offs, dr = decode_x(ctx, x)
    roundtrip = all(abs((gt["offsets"][b] % ctx.H) - dec_offs[b]) < tol_px for b in ctx.blocks) and abs(
        ky - gt["k_y"]
    ) < tol_px
    metrics = evaluate_trial(ctx, gt["k_y"], gt["offsets"], float(r), tag="verify", order_branch=order_branch)
    ceil_miou = float(ceiling["agents_gt_ceiling_miou"])
    eval_miou = float(metrics["gt_miou"])
    regret = ceil_miou - eval_miou
    miou_ok = regret <= max(miou_tol, 0.07)
    passed = roundtrip and miou_ok and not metrics.get("agent_error")
    return {
        "ring_key": ctx.case_id,
        "roundtrip_ok": roundtrip,
        "miou_ok": miou_ok,
        "ceiling_miou": round(ceil_miou, 4),
        "evaluate_miou": round(eval_miou, 4),
        "regret_vs_ceiling": round(regret, 4),
        "passed": passed,
    }
