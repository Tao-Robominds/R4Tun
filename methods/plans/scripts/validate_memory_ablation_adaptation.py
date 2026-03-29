#!/usr/bin/env python3
"""
Post–E2E check: characteristic drift vs sample vs parameter adaptation.

FAIL (exit 1): drift above threshold and all five stage parameter JSONs match sample
    (no meaningful adaptation when the tunnel differs from sample).

See configurable/ablation/memory/process.md for workflow context.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

STAGE_FILES = (
    "parameters_unfolding.json",
    "parameters_denoising.json",
    "parameters_enhancing.json",
    "parameters_detecting.json",
    "parameters_sam.json",
)

# JSON paths under point_cloud_analysis (must exist in raw_characteristics.json)
DRIFT_NUMERIC_PATHS = (
    "point_cloud_analysis.basic_statistics.total_points",
    "point_cloud_analysis.tunnel_geometry.estimated_diameter",
    "point_cloud_analysis.point_density.median_nearest_neighbor_distance",
)


def _get_at(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def drift_score(sample_raw: Mapping[str, Any], tunnel_raw: Mapping[str, Any]) -> tuple[float, list[str]]:
    """Mean relative absolute difference for available numeric pairs; skips missing/null."""
    contribs: list[float] = []
    notes: list[str] = []
    for path in DRIFT_NUMERIC_PATHS:
        s = _get_at(sample_raw, path)
        t = _get_at(tunnel_raw, path)
        if s is None or t is None:
            notes.append(f"{path}: skip (missing/null)")
            continue
        if not _is_number(s) or not _is_number(t):
            notes.append(f"{path}: skip (non-numeric)")
            continue
        sf, tf = float(s), float(t)
        denom = max(abs(sf), 1e-12)
        contribs.append(abs(tf - sf) / denom)
        notes.append(f"{path}: rel={contribs[-1]:.4f}")
    if not contribs:
        return 0.0, notes
    return sum(contribs) / len(contribs), notes


def _deep_diff_leaves(
    a: Any,
    b: Any,
    prefix: str,
    eps_abs: float,
    eps_rel: float,
) -> list[tuple[str, Any, Any]]:
    """Return list of (path, a, b) where values differ beyond numeric tolerance."""
    out: list[tuple[str, Any, Any]] = []
    if type(a) != type(b) and not (_is_number(a) and _is_number(b)):
        out.append((prefix or "<root>", a, b))
        return out
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = sorted(set(a) | set(b))
        for k in keys:
            p = f"{prefix}.{k}" if prefix else k
            if k not in a or k not in b:
                out.append((p, a.get(k, "<missing>"), b.get(k, "<missing>")))
                continue
            out.extend(_deep_diff_leaves(a[k], b[k], p, eps_abs, eps_rel))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((prefix, a, b))
            return out
        for i, (ai, bi) in enumerate(zip(a, b)):
            p = f"{prefix}[{i}]"
            out.extend(_deep_diff_leaves(ai, bi, p, eps_abs, eps_rel))
        return out
    if _is_number(a) and _is_number(b):
        af, bf = float(a), float(b)
        tol = max(eps_abs, eps_rel * max(abs(af), abs(bf), 1e-12))
        if abs(af - bf) > tol:
            out.append((prefix, a, b))
        return out
    if a != b:
        out.append((prefix, a, b))
    return out


def stage_diff_vs_sample(
    tunnel_stage_path: Path,
    sample_stage_path: Path,
    eps_abs: float,
    eps_rel: float,
) -> list[tuple[str, Any, Any]]:
    t_data = _load_json(tunnel_stage_path)
    s_data = _load_json(sample_stage_path)
    return _deep_diff_leaves(t_data, s_data, "", eps_abs, eps_rel)


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    c = Path.cwd() / path
    if c.exists():
        return c.resolve()
    r = REPO_ROOT / path
    return r.resolve()


def discover_tunnel_ids(chars_root: Path) -> list[str]:
    ids: list[str] = []
    if not chars_root.is_dir():
        return ids
    for child in sorted(chars_root.iterdir()):
        if not child.is_dir():
            continue
        raw = child / "characteristics" / "raw_characteristics.json"
        if raw.is_file():
            ids.append(child.name)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tunnel-ids",
        nargs="*",
        default=[],
        help="Tunnel ids to check (e.g. 1-4 4-1)",
    )
    ap.add_argument(
        "--tunnel-ids-file",
        type=str,
        default="",
        help="File with one tunnel id per line",
    )
    ap.add_argument(
        "--discover",
        action="store_true",
        help="Use all tunnel ids under --characteristics-root with raw_characteristics.json",
    )
    ap.add_argument(
        "--sample-raw",
        type=str,
        default="data/sample/characteristics/raw_characteristics.json",
    )
    ap.add_argument(
        "--characteristics-root",
        type=str,
        default="data/ablation/memory",
        help="Parent of <id>/characteristics/raw_characteristics.json",
    )
    ap.add_argument(
        "--configurable-root",
        type=str,
        default="configurable",
    )
    ap.add_argument(
        "--sample-params-dir",
        type=str,
        default="configurable/sample",
    )
    ap.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="Mean relative drift above this triggers strict adaptation check (default 5%%)",
    )
    ap.add_argument(
        "--eps-abs",
        type=float,
        default=1e-9,
        help="Absolute tolerance for numeric parameter equality vs sample",
    )
    ap.add_argument(
        "--eps-rel",
        type=float,
        default=1e-6,
        help="Relative tolerance for numeric parameter equality vs sample",
    )
    args = ap.parse_args()

    sample_raw_path = _resolve(args.sample_raw)
    if not sample_raw_path.is_file():
        print(f"ERROR: sample raw not found: {sample_raw_path}", file=sys.stderr)
        return 2
    sample_raw = _load_json(sample_raw_path)

    chars_root = _resolve(args.characteristics_root)
    cfg_root = _resolve(args.configurable_root)
    sample_dir = _resolve(args.sample_params_dir)

    tids: list[str] = list(args.tunnel_ids)
    if args.tunnel_ids_file:
        tf = _resolve(args.tunnel_ids_file)
        if not tf.is_file():
            print(f"ERROR: --tunnel-ids-file not found: {tf}", file=sys.stderr)
            return 2
        tids.extend(line.strip() for line in tf.read_text(encoding="utf-8").splitlines() if line.strip())
    if args.discover:
        tids.extend(discover_tunnel_ids(chars_root))
    tids = sorted(set(tids))
    if not tids:
        print("ERROR: no tunnel ids (use --tunnel-ids, --tunnel-ids-file, or --discover)", file=sys.stderr)
        return 2

    exit_fail = False
    for tid in tids:
        raw_path = chars_root / tid / "characteristics" / "raw_characteristics.json"
        if not raw_path.is_file():
            print(f"{tid}\tSKIP\tno raw_characteristics at {raw_path}")
            continue
        tunnel_raw = _load_json(raw_path)
        drift, dnotes = drift_score(sample_raw, tunnel_raw)

        all_match = True
        stage_summary: list[str] = []
        per_stage_report: dict[str, list[tuple[str, Any, Any]]] = {}
        for stage in STAGE_FILES:
            tp = cfg_root / tid / stage
            sp = sample_dir / stage
            if not tp.is_file():
                all_match = False
                stage_summary.append(f"{stage}:missing_tunnel")
                continue
            if not sp.is_file():
                print(f"ERROR: sample parameter missing: {sp}", file=sys.stderr)
                return 2
            diffs = stage_diff_vs_sample(tp, sp, args.eps_abs, args.eps_rel)
            clean = len(diffs) == 0
            per_stage_report[stage] = diffs
            if clean:
                stage_summary.append(f"{stage}:same")
            else:
                all_match = False
                stage_summary.append(f"{stage}:diff({len(diffs)})")

        status = "PASS"
        if drift <= args.drift_threshold:
            status = "PASS"
        elif all_match:
            status = "FAIL"
            exit_fail = True
        else:
            warn_rel = 1e-4
            substantive = False
            for diffs in per_stage_report.values():
                for _path, va, vb in diffs:
                    if _is_number(va) and _is_number(vb):
                        fa, fb = float(va), float(vb)
                        if abs(fa - fb) / max(abs(fa), abs(fb), 1e-12) > warn_rel:
                            substantive = True
                            break
                    else:
                        substantive = True
                        break
                if substantive:
                    break
            if substantive:
                status = "PASS"
            else:
                status = "WARN"
                print(
                    f"  WARN: drift>{args.drift_threshold} but only tiny numeric diffs vs sample (<{warn_rel:g} rel).",
                    file=sys.stderr,
                )

        print(f"{tid}\t{status}\tdrift_mean_rel={drift:.6f}\tthreshold={args.drift_threshold}")
        print(f"  drift_detail: {', '.join(dnotes)}")
        print(f"  stages: {', '.join(stage_summary)}")
        if status == "FAIL":
            print("  interpretation: high characteristic drift vs sample but parameters still match sample — no adaptation.")
        for stage, diffs in per_stage_report.items():
            if not diffs:
                continue
            show = diffs[:8]
            for path, va, vb in show:
                print(f"  diff {stage} `{path}`: {va!r} vs {vb!r}")
            if len(diffs) > 8:
                print(f"  ... {len(diffs) - 8} more paths in {stage}")
        print()

    return 1 if exit_fail else 0


if __name__ == "__main__":
    sys.exit(main())
