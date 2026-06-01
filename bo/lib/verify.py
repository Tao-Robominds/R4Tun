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
from lib.search_space import LAYOUT_RECOVERY_PARAMS, default_layout_fracs


def verify_ring(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    order_branch: str,
    segment_count: int | None,
    manifest_entry: dict[str, Any] | None = None,
    tol_px: float = 1.0,
    miou_tol: float = 0.05,
) -> dict[str, Any]:
    ctx = build_ring_context(
        tunnel_id,
        ring_id,
        source_root=source_root,
        run_root=run_root,
        segment_count=segment_count,
        manifest_entry=manifest_entry,
    )
    ceiling = compute_ceiling_reference(ctx)
    gt = json.loads((ctx.out_dir / "gt_layout.json").read_text(encoding="utf-8"))
    x = encode_gt_layout_x(ctx, gt)
    ky, dec_offs, layout, r_surf = decode_x(ctx, x)
    roundtrip = all(abs((gt["offsets"][b] % ctx.H) - dec_offs[b]) < tol_px for b in ctx.blocks) and abs(
        ky - gt["k_y"]
    ) < tol_px
    default_layout = {
        p.name: p.decode(float(f)) for p, f in zip(LAYOUT_RECOVERY_PARAMS, default_layout_fracs())
    }
    layout_ok = all(abs(layout[k] - default_layout[k]) < 1e-3 for k in layout)
    metrics = evaluate_trial(
        ctx, gt["k_y"], gt["offsets"], layout, r_surf, tag="verify", order_branch=order_branch
    )
    ceil_miou = float(ceiling["agents_gt_ceiling_miou"])
    eval_miou = float(metrics["gt_miou"])
    regret = ceil_miou - eval_miou
    miou_ok = regret <= max(miou_tol, 0.07)
    passed = roundtrip and layout_ok and miou_ok and not metrics.get("agent_error")
    return {
        "ring_key": ctx.case_id,
        "roundtrip_ok": roundtrip,
        "layout_defaults_ok": layout_ok,
        "miou_ok": miou_ok,
        "ceiling_miou": round(ceil_miou, 4),
        "evaluate_miou": round(eval_miou, 4),
        "regret_vs_ceiling": round(regret, 4),
        "passed": passed,
    }
