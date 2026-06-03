#!/usr/bin/env python3
"""Audit GT K vs regular prior band for held-out families 1, 2, 3 (read-only corpus)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bo"))
from lib.ceiling_gate import derive_gt_layout, detect_segment_count  # noqa: E402

HELD = REPO / "data/held-out"
PANEL = REPO / "data/held-out/_manifests/data_v6_50ring_calibration_panel.csv"
OUT = Path(__file__).resolve().parent

LOW_FRAC = 1150.0 / 2777.0
HIGH_FRAC = 1580.0 / 2777.0
LOW_PARITY = 0


def circ_dist(a: float, b: float, h: float) -> float:
    return abs((a - b + h / 2) % h - h / 2)


def prior_k(fr: float, h: int) -> float:
    return float(fr * h) % float(h)


def main() -> int:
    panel = pd.read_csv(PANEL)
    panel = panel[panel["family"].isin([1, 2, 3])].copy()

    rows = []
    failures = []
    for _, ent in panel.iterrows():
        ring_key = str(ent["ring_key"])
        tid = str(ent["tunnel_id"])
        rid = int(ent["ring_id"])
        fam = int(ent["family"])
        ring_dir = HELD / tid / f"r{rid}"
        if not (ring_dir / "enhanced.csv").is_file():
            failures.append({"ring_key": ring_key, "error": "missing enhanced.csv"})
            continue
        try:
            seg = detect_segment_count(ring_dir)
            layout = derive_gt_layout(ring_dir, ring_dir, seg)
        except Exception as exc:
            failures.append({"ring_key": ring_key, "error": str(exc)})
            continue

        h = int(layout["H"])
        k_gt = float(layout["k_y"])
        k_gt_frac = k_gt / h
        k_low = prior_k(LOW_FRAC, h)
        k_high = prior_k(HIGH_FRAC, h)
        parity = rid % 2
        active_frac = LOW_FRAC if parity == LOW_PARITY else HIGH_FRAC
        k_active = prior_k(active_frac, h)
        in_band_frac = LOW_FRAC <= k_gt_frac <= HIGH_FRAC
        in_band_px = min(k_low, k_high) <= k_gt <= max(k_low, k_high)
        rows.append(
            {
                "ring_key": ring_key,
                "family": fam,
                "ring_id": rid,
                "parity": parity,
                "H": h,
                "k_gt": round(k_gt, 2),
                "k_gt_frac": round(k_gt_frac, 6),
                "k_low_px": round(k_low, 2),
                "k_high_px": round(k_high, 2),
                "k_active_px": round(k_active, 2),
                "active_frac": round(active_frac, 6),
                "in_prior_frac_band": in_band_frac,
                "in_prior_px_band": in_band_px,
                "gt_to_active_px": round(circ_dist(k_gt, k_active, h), 2),
                "gt_to_nearest_anchor_px": round(min(circ_dist(k_gt, k_low, h), circ_dist(k_gt, k_high, h)), 2),
            }
        )

    df = pd.DataFrame(rows)
    frac_min = float(df["k_gt_frac"].min())
    frac_max = float(df["k_gt_frac"].max())
    # pad slightly so all held-out family 1-3 GT K are inside [low, high]
    pad = 0.005
    proposed_low = max(0.0, frac_min - pad)
    proposed_high = min(1.0, frac_max + pad)

    summary = {
        "corpus": "data/held-out (read-only)",
        "families": [1, 2, 3],
        "n_rings": len(df),
        "current_prior": {
            "regular_k_prior_low_frac": LOW_FRAC,
            "regular_k_prior_high_frac": HIGH_FRAC,
            "regular_k_prior_low_ring_parity": LOW_PARITY,
        },
        "in_current_frac_band": {
            "pass_count": int(df["in_prior_frac_band"].sum()),
            "fail_count": int((~df["in_prior_frac_band"]).sum()),
            "fail_ring_keys": df.loc[~df["in_prior_frac_band"], "ring_key"].tolist(),
        },
        "in_current_px_band": {
            "pass_count": int(df["in_prior_px_band"].sum()),
            "fail_count": int((~df["in_prior_px_band"]).sum()),
        },
        "parity_anchor_vs_gt": {
            "mean_gt_to_active_px": round(float(df["gt_to_active_px"].mean()), 2),
            "max_gt_to_active_px": round(float(df["gt_to_active_px"].max()), 2),
            "mean_gt_to_nearest_anchor_px": round(float(df["gt_to_nearest_anchor_px"].mean()), 2),
        },
        "gt_k_frac_by_family": {
            str(f): {
                "min": round(float(g["k_gt_frac"].min()), 6),
                "max": round(float(g["k_gt_frac"].max()), 6),
                "mean": round(float(g["k_gt_frac"].mean()), 6),
            }
            for f, g in df.groupby("family")
        },
        "proposed_widen_to_cover_all_gt": {
            "regular_k_prior_low_frac": round(proposed_low, 6),
            "regular_k_prior_high_frac": round(proposed_high, 6),
            "note": f"Padded min/max of GT k_gt_frac over families 1-3 (pad={pad})",
        },
        "failures": failures,
    }

    df.to_csv(OUT / "heldout_families_123_k_audit.csv", index=False)
    (OUT / "heldout_families_123_k_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["in_current_frac_band"]["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
