#!/usr/bin/env python3
"""K-position diagnostics for continuous T3 tunnels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

K_SPREAD_PASS_PX = 50.0
FALLBACK_FRAC_PASS = 0.20

_GOOD = frozenset({"midpoint", "positive_slope", "negative_slope", "horizontal"})
_FALLBACK = frozenset({"default", "assume"})


def analyze_detected(detected_path: Path) -> dict:
    df = pd.read_csv(detected_path)
    if "Type" in df.columns:
        types = df["Type"].astype(str).tolist()
        ys = df["Y"].astype(float).tolist()
    elif "type" in df.columns:
        types = df["type"].astype(str).tolist()
        ys = df["y"].astype(float).tolist()
    else:
        raise ValueError(f"No Type column in {detected_path}")

    y_min, y_max = float(min(ys)), float(max(ys))
    spread = y_max - y_min
    n = len(types)
    fallback_n = sum(1 for t in types if t in _FALLBACK)
    fallback_frac = fallback_n / n if n else 1.0

    per_ring = []
    for i, (t, y) in enumerate(zip(types, ys)):
        per_ring.append({"ring": i, "type": t, "y": y})

    k_pass = spread < K_SPREAD_PASS_PX
    recs: list[str] = []
    if spread >= K_SPREAD_PASS_PX:
        recs.append("Lower hough_threshold_horizontal/oblique (55→45→40)")
        recs.append("Widen maxLineGap_horizontal (10→12→15)")
    if fallback_frac > FALLBACK_FRAC_PASS:
        recs.append("More joint lines needed — lower Hough thresholds, widen gaps")
        recs.append("Enable k_pattern_outlier_tol_px 120–150 for continuous consensus")

    return {
        "y_min": y_min,
        "y_max": y_max,
        "y_spread_px": spread,
        "k_spread_pass": k_pass,
        "fallback_count": fallback_n,
        "fallback_frac": round(fallback_frac, 3),
        "type_counts": {t: types.count(t) for t in sorted(set(types))},
        "per_ring": per_ring,
        "recommendations": recs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T3 K diagnostics from detected.csv")
    parser.add_argument("path", help="Tunnel dir or detected.csv path")
    parser.add_argument("--json-out", type=Path, help="Write diagnostics JSON here")
    args = parser.parse_args()

    p = Path(args.path)
    detected = p / "detected.csv" if p.is_dir() else p
    if not detected.is_file():
        print(f"Missing {detected}", file=sys.stderr)
        sys.exit(1)

    result = analyze_detected(detected)
    print(f"Y spread: {result['y_spread_px']:.1f} px (pass < {K_SPREAD_PASS_PX})")
    print(f"Fallback fraction: {result['fallback_frac']:.1%} (pass <= {FALLBACK_FRAC_PASS:.0%})")
    print(f"K gate: {'PASS' if result['k_spread_pass'] else 'FAIL'}")
    print(f"Types: {result['type_counts']}")
    if result["recommendations"]:
        print("Recommendations:")
        for r in result["recommendations"]:
            print(f"  - {r}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.json_out}")

    sys.exit(0 if result["k_spread_pass"] else 1)


if __name__ == "__main__":
    main()
