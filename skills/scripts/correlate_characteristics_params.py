#!/usr/bin/env python
"""
Spearman correlation between characteristic fields and output parameters.

Loads characteristics (30 tunnels) and parameter files (30 tunnels x N models)
from an ablation data root, computes Spearman rank correlation for every
(characteristic field, output parameter) pair.

Usage:
    ./venv/bin/python skills/scripts/correlate_characteristics_params.py
    ./venv/bin/python skills/scripts/correlate_characteristics_params.py --chars-root data/ablation_gpt/memory+state+knowledge --models gpt5.4
    ./venv/bin/python skills/scripts/correlate_characteristics_params.py --rho-threshold 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


DEFAULT_CHARS_ROOT = "data/ablation_anthropic/memory+state+knowledge"
PARAM_ROOT = Path("agents/ablation/memory+state+knowledge/parameters")
ALL_MODELS = ["opus4.6", "gpt5.4", "gemini3flash"]

STAGE_PARAM_PREFIX = {
    "unfolding": "parameters_unfolding_m_s_k_",
    "denoising": "parameters_denoising_m_s_k_",
    "enhancing": "parameters_enhancing_m_s_k_",
    "detecting": "parameters_detecting_m_s_k_",
    "sam":       "parameters_sam_m_s_k_",
}

CHARS_FOR_STAGE = {
    "unfolding": ["raw_characteristics.json"],
    "denoising": ["raw_characteristics.json", "unfolded_characteristics.json"],
    "enhancing": ["raw_characteristics.json", "unfolded_characteristics.json",
                  "denoised_characteristics.json"],
    "detecting": ["raw_characteristics.json", "unfolded_characteristics.json",
                  "denoised_characteristics.json", "enhanced_characteristics.json"],
    "sam":       ["raw_characteristics.json", "unfolded_characteristics.json",
                  "denoised_characteristics.json", "enhanced_characteristics.json",
                  "detected_characteristics.json"],
}

SKIP_PARAMS = {
    "batch_size", "n_jobs", "t_extrapolation_start", "t_extrapolation_end",
    "segment_order", "use_original_label_distributions", "processing",
    "prompt_points", "morphological_kernel_size",
}

TUNNEL_IDS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
    "4-1", "4-2", "4-3", "4-4", "4-5", "4-6", "4-7", "4-8", "4-9", "4-10",
    "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7",
]


def flatten_json(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            out.update(flatten_json(v, new_key))
    elif isinstance(obj, list):
        if all(isinstance(x, (int, float)) for x in obj) and len(obj) == 2:
            out[f"{prefix}[0]"] = obj[0]
            out[f"{prefix}[1]"] = obj[1]
        else:
            for i, v in enumerate(obj):
                out.update(flatten_json(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


MIN_TUNNELS = 10


def load_chars_matrix(tunnel_ids: list[str], char_files: list[str],
                      chars_root: Path):
    """Load characteristics for all tunnels.

    Returns {field: {tid: value}} so we can align with params per-tunnel,
    even when some tunnels lack state characteristics.
    """
    fields: dict[str, dict[str, float]] = {}
    for tid in tunnel_ids:
        merged = {}
        for cf in char_files:
            p = chars_root / tid / "characteristics" / cf
            if not p.exists():
                continue
            with open(p) as f:
                flat = flatten_json(json.load(f), cf.replace(".json", ""))
            merged.update(flat)
        for k, v in merged.items():
            if not isinstance(v, (int, float)):
                continue
            fields.setdefault(k, {})[tid] = v

    valid = {k: v for k, v in fields.items() if len(v) >= MIN_TUNNELS}
    return valid


def load_params_matrix(tunnel_ids: list[str], stage: str, models: list[str]):
    """Load parameter values. Average across models for each tunnel.

    Returns {param_key: {tid: mean_value}}.
    """
    prefix = STAGE_PARAM_PREFIX[stage]
    all_keys: set[str] = set()
    per_tunnel: dict[str, dict[str, list[float]]] = {tid: {} for tid in tunnel_ids}

    for tid in tunnel_ids:
        for model in models:
            p = PARAM_ROOT / tid / f"{prefix}{model}.json"
            if not p.exists():
                continue
            with open(p) as f:
                flat = flatten_json(json.load(f))
            for k, v in flat.items():
                if not isinstance(v, (int, float)):
                    continue
                leaf = k.rsplit(".", 1)[-1] if "." in k else k
                leaf = leaf.rstrip("[]0123456789")
                if leaf in SKIP_PARAMS or k in SKIP_PARAMS:
                    continue
                all_keys.add(k)
                per_tunnel[tid].setdefault(k, []).append(v)

    result: dict[str, dict[str, float]] = {}
    for k in sorted(all_keys):
        vals = {}
        for tid in tunnel_ids:
            v_list = per_tunnel[tid].get(k)
            if v_list:
                vals[tid] = float(np.mean(v_list))
        if len(vals) >= MIN_TUNNELS:
            result[k] = vals
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rho-threshold", type=float, default=0.5)
    p.add_argument("--chars-root", type=str, default=DEFAULT_CHARS_ROOT,
                   help="Data root containing {tunnel_id}/characteristics/ dirs")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated model list (default: all 3). E.g. gpt5.4")
    args = p.parse_args()

    chars_root = Path(args.chars_root)
    models = args.models.split(",") if args.models else ALL_MODELS

    print(f"Chars root : {chars_root}")
    print(f"Param root : {PARAM_ROOT}")
    print(f"Models     : {models}")
    print(f"Rho thresh : {args.rho_threshold}")

    stages = ["unfolding", "denoising", "enhancing", "detecting", "sam"]
    all_correlations: dict[str, dict] = {}

    for stage in stages:
        print(f"\n{'='*70}")
        print(f"STAGE: {stage}")
        print(f"{'='*70}")

        chars = load_chars_matrix(TUNNEL_IDS, CHARS_FOR_STAGE[stage], chars_root)
        params = load_params_matrix(TUNNEL_IDS, stage, models)

        if not chars or not params:
            print("  Insufficient data")
            continue

        char_coverage = {k: len(v) for k, v in chars.items()}
        raw_count = sum(1 for k in chars if k.startswith("raw_"))
        state_count = len(chars) - raw_count
        print(f"  Characteristics fields: {len(chars)} (raw: {raw_count}, state: {state_count})")
        print(f"  Parameter fields: {len(params)}")
        if state_count > 0:
            min_cov = min(len(v) for k, v in chars.items() if not k.startswith("raw_"))
            max_cov = max(len(v) for k, v in chars.items() if not k.startswith("raw_"))
            print(f"  State char tunnel coverage: {min_cov}-{max_cov}/30")

        significant = []
        for char_field, char_tid_vals in chars.items():
            for param_field, param_tid_vals in params.items():
                shared = sorted(set(char_tid_vals.keys()) & set(param_tid_vals.keys()))
                if len(shared) < MIN_TUNNELS:
                    continue
                c_arr = np.array([char_tid_vals[t] for t in shared])
                p_arr = np.array([param_tid_vals[t] for t in shared])
                if np.std(c_arr) < 1e-12 or np.std(p_arr) < 1e-12:
                    continue
                rho, pval = stats.spearmanr(c_arr, p_arr)
                if abs(rho) >= args.rho_threshold and pval < 0.05:
                    significant.append((char_field, param_field, rho, pval, len(shared)))
                    key = char_field
                    if key not in all_correlations:
                        all_correlations[key] = {"max_rho": 0, "stages": set(),
                                                 "pairs": [], "n_tunnels": len(shared)}
                    if abs(rho) > abs(all_correlations[key]["max_rho"]):
                        all_correlations[key]["max_rho"] = rho
                        all_correlations[key]["n_tunnels"] = len(shared)
                    all_correlations[key]["stages"].add(stage)
                    all_correlations[key]["pairs"].append(
                        (stage, param_field, rho, pval))

        significant.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n  Significant correlations (|rho| >= {args.rho_threshold}, p < 0.05): {len(significant)}")
        for cf, pf, rho, pval, n in significant[:30]:
            cf_short = cf.rsplit("::", 1)[-1] if "::" in cf else cf
            if len(cf_short) > 48:
                cf_short = "..." + cf_short[-45:]
            print(f"    {cf_short:<52} -> {pf:<35} rho={rho:+.3f}  p={pval:.4f}  n={n}")

    print(f"\n\n{'='*70}")
    print("RANKED CHARACTERISTIC FIELDS BY MAX |rho|")
    print(f"{'='*70}")
    ranked = sorted(all_correlations.items(), key=lambda x: abs(x[1]["max_rho"]), reverse=True)
    print(f"{'Field':<70} | {'|rho|':>6} | {'N':>3} | {'stages':>8} | correlated params")
    print("-" * 140)
    for field, info in ranked:
        short = field
        if len(short) > 68:
            short = "..." + short[-65:]
        stage_list = ",".join(sorted(info["stages"]))
        param_list = ", ".join(set(p[1] for p in info["pairs"][:3]))
        if len(info["pairs"]) > 3:
            param_list += f" (+{len(info['pairs'])-3} more)"
        print(f"{short:<70} | {abs(info['max_rho']):>6.3f} | {info['n_tunnels']:>3} | {stage_list:>8} | {param_list}")

    uncorrelated_chars = set()
    for stage in stages:
        chars = load_chars_matrix(TUNNEL_IDS, CHARS_FOR_STAGE[stage], chars_root)
        for cf in chars:
            if cf not in all_correlations:
                uncorrelated_chars.add(cf)

    print(f"\n\nUNCORRELATED fields (no |rho| >= {args.rho_threshold} with any param): {len(uncorrelated_chars)}")
    for f in sorted(uncorrelated_chars):
        print(f"  {f}")


if __name__ == "__main__":
    main()
