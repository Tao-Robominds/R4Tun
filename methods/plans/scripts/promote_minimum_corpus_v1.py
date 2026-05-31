#!/usr/bin/env python3
"""Gate and promote minimum 2-ring BO calib corpus to data/minimum/."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_BO = _REPO / "bo"
if str(_BO) not in sys.path:
    sys.path.insert(0, str(_BO))

from lib.ceiling_gate import CEILING_THRESHOLD, detect_segment_count  # noqa: E402
from lib.layout_bo import build_ring_context, compute_ceiling_reference  # noqa: E402

# 3a QA (inline — same thresholds as agents/1_preprocessing/knowledge.md)
def depth_3a(ring_dir: Path) -> dict:
    dm = np.load(ring_dir / "depth_map.npy")
    finite = np.isfinite(dm)
    valid = finite & (dm > 0.0)
    h = dm.shape[0]
    row_ok = np.mean(valid, axis=1) > 0.01
    row_nonempty_ratio = float(np.count_nonzero(row_ok) / max(h, 1))
    finite_ratio = float(np.count_nonzero(finite) / max(int(dm.size), 1))
    empty = ~row_ok
    max_gap = cur = 0
    for v in empty:
        if v:
            cur += 1
            max_gap = max(max_gap, cur)
        else:
            cur = 0
    gap_frac = float(max_gap / max(h, 1))
    passed = finite_ratio >= 0.60 and row_nonempty_ratio >= 0.90 and gap_frac <= 0.08
    return {
        "finite_ratio": finite_ratio,
        "row_nonempty_ratio": row_nonempty_ratio,
        "largest_empty_vertical_gap_frac": gap_frac,
        "passed_3a": passed,
    }


def pattern_score(ring_dir: Path, segment_count: int) -> float:
    sys.path.insert(0, str(_REPO / "stages" / "v7" / "bo" / "proxy"))
    from build_ring_descriptors_v1 import _pattern_score  # noqa: WPS433

    return float(_pattern_score(ring_dir, segment_count))


def ensure_params_json(tunnel_id: str, ring_id: int, ring_dir: Path) -> None:
    dst = ring_dir / "parameters_preprocessing.json"
    if dst.exists():
        return
    src = _REPO / "agents" / "1_preprocessing" / "parameters" / tunnel_id / f"r{ring_id}" / "parameters_preprocessing.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing parameters template: {src}")
    shutil.copy2(src, dst)


def gate_ring(
    tunnel_id: str,
    ring_id: int,
    *,
    sandbox_root: Path,
    gate_root: Path,
    order_branch_default: str,
) -> dict:
    src_ring = sandbox_root / tunnel_id / f"r{ring_id}"
    ensure_params_json(tunnel_id, ring_id, src_ring)
    seg = detect_segment_count(src_ring)
    pat = pattern_score(src_ring, seg)
    order_from_gt = "plus" if pat >= 0.99 else ("minus" if pat <= 0.01 else "ambiguous")
    qa = depth_3a(src_ring)

    ctx = build_ring_context(
        tunnel_id, ring_id, source_root=sandbox_root, run_root=gate_root, segment_count=seg
    )
    ceiling = compute_ceiling_reference(ctx)
    ceil_miou = ceiling.get("agents_gt_ceiling_miou")
    passed_3b = ceil_miou is not None and float(ceil_miou) >= CEILING_THRESHOLD

    pp = json.loads((src_ring / "parameters_preprocessing.json").read_text(encoding="utf-8"))
    summary = json.loads((_REPO / "data" / "rings" / "summary.json").read_text(encoding="utf-8"))
    meta = next(
        (
            s
            for s in summary["samples"]
            if s["file"].replace("_", "-") == tunnel_id and int(s["ring_id"]) == ring_id
        ),
        {},
    )

    payload = {
        "ring_key": f"{tunnel_id}/r{ring_id}",
        "segment_count": seg,
        "order_branch_default": order_branch_default,
        "pattern_score": pat,
        "order_from_gt_pattern_score": order_from_gt,
        "intrinsic_quality": qa,
        "passed_3a": qa["passed_3a"],
        "ceiling_miou": ceil_miou,
        "passed_3b": passed_3b,
        "ceiling": ceiling,
        "tunnel_diameter": float(pp["tunnel_diameter"]),
        "n_points": meta.get("n_points"),
        "density_tier": meta.get("reason"),
        "walking_order": meta.get("walking_order"),
        "sandbox_lineage": str(src_ring.relative_to(_REPO)),
    }
    payload["passed"] = payload["passed_3a"] and payload["passed_3b"]
    out = gate_root / "gates" / f"{tunnel_id}_r{ring_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def promote_ring(tunnel_id: str, ring_id: int, *, src_root: Path, dst_root: Path) -> None:
    src = src_root / tunnel_id / f"r{ring_id}"
    dst = dst_root / tunnel_id / f"r{ring_id}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox-root", type=Path, default=_REPO / "logs" / "v7_minimum_calib_v1")
    ap.add_argument("--gate-root", type=Path, default=_REPO / "logs" / "v7_minimum_calib_v1")
    ap.add_argument("--out-root", type=Path, default=_REPO / "data" / "minimum")
    ap.add_argument(
        "--force-promote",
        action="store_true",
        help="Copy to data/minimum/ even if 3a/3b gates fail (gate status recorded in MANIFEST)",
    )
    args = ap.parse_args()

    rings = [
        ("1-4", 206, "plus"),
        ("4-4", 210, "minus"),
    ]
    results = []
    for tid, rid, branch in rings:
        print(f"Gating {tid}/r{rid}...")
        results.append(
            gate_ring(tid, rid, sandbox_root=args.sandbox_root, gate_root=args.gate_root, order_branch_default=branch)
        )

    panel = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rings": results,
        "all_passed": all(r["passed"] for r in results),
    }
    panel_path = args.gate_root / "gates" / "panel_gate.json"
    panel_path.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(panel, indent=2))

    if not panel["all_passed"] and not args.force_promote:
        print("GATE FAILED — not promoting to data/minimum/ (use --force-promote to copy anyway)", file=sys.stderr)
        return 1

    if not panel["all_passed"]:
        print("GATE FAILED — promoting anyway (--force-promote)", file=sys.stderr)

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_rings = []
    for tid, rid, branch in rings:
        promote_ring(tid, rid, src_root=args.sandbox_root, dst_root=args.out_root)
        r = next(x for x in results if x["ring_key"] == f"{tid}/r{rid}")
        manifest_rings.append(
            {
                "ring_key": r["ring_key"],
                "segment_count": r["segment_count"],
                "diameter_bin": round(r["tunnel_diameter"], 1),
                "order_branch_default": branch,
                "pattern_score": r["pattern_score"],
                "n_points": r["n_points"],
                "density_tier": r["density_tier"],
                "walking_order": r["walking_order"],
                "source_txt": f"data/rings/{tid.replace('-', '_')}_ring{rid}.txt",
                "sandbox_lineage": r["sandbox_lineage"],
                "ceiling_miou": r["ceiling_miou"],
                "ceiling_passed": r["passed_3b"],
                "intrinsic_quality": r["intrinsic_quality"],
            }
        )

    manifest = {
        "description": "2-ring minimum BO calibration corpus (sparse 6-block plus + 7-block minus)",
        "output_root": "data/minimum",
        "user_copy_hint": "Canonical minimum-calib corpus. Read-only; BO/eval outputs go to logs/ only.",
        "all_gates_passed": panel["all_passed"],
        "force_promoted": bool(not panel["all_passed"] and args.force_promote),
        "rings": manifest_rings,
        "gate_evidence": str(panel_path.relative_to(_REPO)),
    }
    (args.out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Promoted to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
