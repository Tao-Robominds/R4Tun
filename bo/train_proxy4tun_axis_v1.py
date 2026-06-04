#!/usr/bin/env python3
"""Train Proxy4Tun L / K / L+K concat / L+K joint Ridge proxies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BO = Path(__file__).resolve().parent
REPO_ROOT = _BO.parent
if str(_BO) not in sys.path:
    sys.path.insert(0, str(_BO))

from lib.proxy4tun_train import (  # noqa: E402
    train_all_proxies,
    train_lk_enriched_sweep,
    train_gated_weighted_lk_proxy,
    train_weighted_lk_proxy,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("full", "lk-enriched", "weighted-lk", "weighted-lk-gated"),
        default="full",
    )
    ap.add_argument("--stream-l-root", type=Path, default=REPO_ROOT / "logs/proxy4tun/stream_l")
    ap.add_argument("--stream-k-root", type=Path, default=REPO_ROOT / "logs/proxy4tun/stream_k")
    ap.add_argument("--stream-full-root", type=Path, default=REPO_ROOT / "logs/proxy4tun/stream_full")
    ap.add_argument("--records-concat", type=Path, default=None)
    ap.add_argument("--records-joint", type=Path, default=None)
    ap.add_argument("--v1-gate", type=Path, default=REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/proxy_training_gate.json")
    ap.add_argument("--skip-joint", action="store_true", help="Skip LK_joint if stream_full missing")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--top-k-sweep", default="", help="e.g. 4,8,12 for lk-enriched mode")
    ap.add_argument(
        "--nested-loro",
        action="store_true",
        help="Re-fit L/K sub-proxies per held-out ring (weighted-lk mode)",
    )
    ap.add_argument(
        "--blend",
        choices=("ridge", "alpha"),
        default="ridge",
        help="ridge = learned w_L,w_K on sub-scores; alpha = LORO convex blend",
    )
    ap.add_argument(
        "--model-l",
        type=Path,
        default=REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/models/proxy_L.json",
    )
    ap.add_argument(
        "--model-k",
        type=Path,
        default=REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/models/proxy_K.json",
    )
    args = ap.parse_args()

    if args.mode == "weighted-lk-gated":
        out_dir = args.out_dir.resolve()
        concat_path = args.records_concat or (
            REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/records_LK_concat.csv"
        )
        manifest = train_gated_weighted_lk_proxy(
            records_concat=concat_path.resolve(),
            model_l_path=args.model_l.resolve(),
            model_k_path=args.model_k.resolve(),
            out_dir=out_dir,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.mode == "weighted-lk":
        out_dir = args.out_dir.resolve()
        concat_path = args.records_concat or (
            REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/records_LK_concat.csv"
        )
        records_l = out_dir.parent / "proxy_train_lk_v1/records_L.csv"
        if not records_l.is_file():
            records_l = REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/records_L.csv"
        records_k = out_dir.parent / "proxy_train_lk_v1/records_K.csv"
        if not records_k.is_file():
            records_k = REPO_ROOT / "logs/proxy4tun/proxy_train_lk_v1/records_K.csv"
        manifest = train_weighted_lk_proxy(
            records_concat=concat_path.resolve(),
            model_l_path=args.model_l.resolve(),
            model_k_path=args.model_k.resolve(),
            out_dir=out_dir,
            alpha=args.alpha,
            blend=args.blend,
            nested_loro=args.nested_loro,
            records_l=records_l.resolve() if args.nested_loro else None,
            records_k=records_k.resolve() if args.nested_loro else None,
            top_k=args.top_k,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.mode == "lk-enriched":
        out_dir = args.out_dir.resolve()
        concat_path = args.records_concat or (out_dir / "records_LK_concat_enriched.csv")
        joint_path = args.records_joint or (out_dir / "records_LK_joint_enriched.csv")
        if not concat_path.is_file() or not joint_path.is_file():
            raise SystemExit(f"Missing enriched records: {concat_path} / {joint_path}")
        sweep = tuple(int(x) for x in args.top_k_sweep.split(",") if x.strip()) or (4, 8, 12)
        manifest = train_lk_enriched_sweep(
            records_concat=concat_path,
            records_joint=joint_path,
            out_dir=out_dir,
            alpha=args.alpha,
            top_k_values=sweep,
            v1_gate_path=args.v1_gate.resolve() if args.v1_gate else None,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    full_root = None if args.skip_joint else args.stream_full_root
    if full_root is not None and not (full_root / "bo_trials.csv").is_file():
        print(f"WARN: missing {full_root / 'bo_trials.csv'}; training without LK_joint")
        full_root = None

    gate = train_all_proxies(
        stream_l_root=args.stream_l_root.resolve(),
        stream_k_root=args.stream_k_root.resolve(),
        stream_full_root=full_root.resolve() if full_root else None,
        out_dir=args.out_dir.resolve(),
        alpha=args.alpha,
        top_k=args.top_k,
    )
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
