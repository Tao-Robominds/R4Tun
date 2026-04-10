#!/usr/bin/env python3
"""Analyze parameter adaptations across 30 tunnels, 3 LLMs, and 3 ablation conditions.

Compares all adapted parameter JSONs against the sam4tun baseline to find:
1. All parameters whose values have been adapted
2. The most critical parameters that trigger every time tunnel conditions change
"""

import json
import os
from collections import defaultdict
from pathlib import Path

ABLATION_ROOT = Path("configurable/ablation")
BASELINE_DIR = ABLATION_ROOT / "sam4tun"
CONDITIONS = ["memory", "memory+state", "memory+state+knowledge"]
CONDITION_CODES = {"memory": "m", "memory+state": "m_s", "memory+state+knowledge": "m_s_k"}
LLMS = ["gemini3flash", "gpt5.4", "opus4.6"]
STAGES = ["unfolding", "denoising", "enhancing", "detecting", "sam"]

TUNNELS = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
    "4-1", "4-2", "4-3", "4-4", "4-5", "4-6", "4-7", "4-8", "4-9", "4-10",
    "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7",
]


def flatten_json(obj, prefix=""):
    """Flatten nested JSON into dot-separated keys."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(flatten_json(v, new_key))
    elif isinstance(obj, list):
        items[prefix] = obj
    else:
        items[prefix] = obj
    return items


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def values_differ(baseline_val, adapted_val):
    if isinstance(baseline_val, list) and isinstance(adapted_val, list):
        return baseline_val != adapted_val
    if isinstance(baseline_val, (int, float)) and isinstance(adapted_val, (int, float)):
        return abs(baseline_val - adapted_val) > 1e-9
    return baseline_val != adapted_val


def main():
    # Load baseline parameters per stage
    baseline = {}
    for stage in STAGES:
        path = BASELINE_DIR / f"parameters_{stage}.json"
        data = load_json(path)
        if data:
            baseline[stage] = flatten_json(data)
        else:
            print(f"WARNING: Missing baseline for {stage}")

    # Track adaptations
    # param_changes[stage][param_key] = list of {condition, llm, tunnel, baseline_val, adapted_val}
    param_changes = defaultdict(lambda: defaultdict(list))
    # Track which (tunnel, llm, condition) combos each param changed in
    param_tunnel_changes = defaultdict(lambda: defaultdict(set))
    # Track files found/missing
    files_found = 0
    files_missing = 0
    # Per-LLM tracking
    param_changes_by_llm = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # New params not in baseline
    new_params = defaultdict(lambda: defaultdict(list))

    for condition in CONDITIONS:
        code = CONDITION_CODES[condition]
        for tunnel in TUNNELS:
            for llm in LLMS:
                for stage in STAGES:
                    fname = f"parameters_{stage}_{code}_{llm}.json"
                    path = ABLATION_ROOT / condition / "parameters" / tunnel / fname
                    data = load_json(path)
                    if data is None:
                        files_missing += 1
                        continue
                    files_found += 1
                    flat = flatten_json(data)
                    bl = baseline.get(stage, {})

                    for key, val in flat.items():
                        bl_val = bl.get(key)
                        if bl_val is None:
                            new_params[stage][key].append({
                                "condition": condition, "llm": llm, "tunnel": tunnel, "value": val
                            })
                            continue
                        if values_differ(bl_val, val):
                            rec = {
                                "condition": condition, "llm": llm, "tunnel": tunnel,
                                "baseline": bl_val, "adapted": val
                            }
                            param_changes[stage][key].append(rec)
                            param_tunnel_changes[stage][key].add((tunnel, llm, condition))
                            param_changes_by_llm[llm][stage][key].append(rec)

    print("=" * 100)
    print("PARAMETER ADAPTATION ANALYSIS")
    print(f"Conditions: {len(CONDITIONS)} | LLMs: {len(LLMS)} | Tunnels: {len(TUNNELS)} | Stages: {len(STAGES)}")
    print(f"Files found: {files_found} | Files missing: {files_missing}")
    print(f"Max possible adaptations per param: {len(CONDITIONS) * len(LLMS) * len(TUNNELS)} = {len(CONDITIONS)*len(LLMS)*len(TUNNELS)}")
    print("=" * 100)

    # ========== PART 1: All parameters that were adapted ==========
    print("\n" + "=" * 100)
    print("PART 1: ALL PARAMETERS WHOSE VALUES HAVE BEEN ADAPTED (vs baseline)")
    print("=" * 100)

    total_adapted_params = 0
    for stage in STAGES:
        changed = param_changes[stage]
        if not changed:
            print(f"\n--- {stage.upper()}: No parameters adapted ---")
            continue
        print(f"\n{'─' * 80}")
        print(f"  STAGE: {stage.upper()} ({len(changed)} parameters adapted)")
        print(f"{'─' * 80}")
        sorted_params = sorted(changed.items(), key=lambda x: -len(x[1]))
        for key, records in sorted_params:
            total_adapted_params += 1
            unique_tunnels = len(set(r["tunnel"] for r in records))
            unique_llms = len(set(r["llm"] for r in records))
            unique_conds = len(set(r["condition"] for r in records))
            vals = [r["adapted"] for r in records]
            bl_val = records[0]["baseline"]

            if isinstance(bl_val, (int, float)):
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if numeric_vals:
                    min_v, max_v = min(numeric_vals), max(numeric_vals)
                    mean_v = sum(numeric_vals) / len(numeric_vals)
                    print(f"  {key:50s} | changes: {len(records):3d} | tunnels: {unique_tunnels:2d}/30 | LLMs: {unique_llms}/3 | conds: {unique_conds}/3")
                    print(f"  {'':50s} | baseline: {bl_val} → range: [{min_v}, {max_v}], mean: {mean_v:.4f}")
                else:
                    print(f"  {key:50s} | changes: {len(records):3d} | tunnels: {unique_tunnels:2d}/30 | LLMs: {unique_llms}/3 | conds: {unique_conds}/3")
                    print(f"  {'':50s} | baseline: {bl_val}")
            else:
                print(f"  {key:50s} | changes: {len(records):3d} | tunnels: {unique_tunnels:2d}/30 | LLMs: {unique_llms}/3 | conds: {unique_conds}/3")
                print(f"  {'':50s} | baseline: {bl_val}")
                unique_vals = list(set(str(v) for v in vals))[:5]
                print(f"  {'':50s} | sample adapted values: {unique_vals}")

    print(f"\n  TOTAL ADAPTED PARAMETERS: {total_adapted_params}")

    # ========== PART 2: Most critical parameters ==========
    print("\n" + "=" * 100)
    print("PART 2: MOST CRITICAL PARAMETERS (trigger every time tunnel conditions change)")
    print("Ranked by number of unique tunnels where the parameter was adapted")
    print("=" * 100)

    all_params = []
    for stage in STAGES:
        for key, records in param_changes[stage].items():
            unique_tunnels = set(r["tunnel"] for r in records)
            unique_llms = set(r["llm"] for r in records)
            unique_conds = set(r["condition"] for r in records)
            all_params.append({
                "stage": stage, "key": key,
                "n_changes": len(records),
                "n_tunnels": len(unique_tunnels),
                "n_llms": len(unique_llms),
                "n_conds": len(unique_conds),
                "tunnels": unique_tunnels,
                "records": records,
            })

    all_params.sort(key=lambda x: (-x["n_tunnels"], -x["n_changes"]))

    print(f"\n{'Rank':>4s} | {'Stage':<12s} | {'Parameter':<50s} | {'Tunnels':>7s} | {'LLMs':>4s} | {'Conds':>5s} | {'Total':>5s}")
    print("-" * 100)
    for i, p in enumerate(all_params[:50], 1):
        print(f"{i:4d} | {p['stage']:<12s} | {p['key']:<50s} | {p['n_tunnels']:>4d}/30 | {p['n_llms']:>1d}/3  | {p['n_conds']:>2d}/3  | {p['n_changes']:>5d}")

    # ========== PART 2b: Parameters adapted in ALL 30 tunnels ==========
    print("\n" + "=" * 100)
    print("PART 2b: PARAMETERS ADAPTED IN ALL OR NEARLY ALL 30 TUNNELS (>=28)")
    print("These are the 'always-trigger' parameters")
    print("=" * 100)

    always_trigger = [p for p in all_params if p["n_tunnels"] >= 28]
    for p in always_trigger:
        recs = p["records"]
        bl_val = recs[0]["records"]["baseline"] if "records" in recs[0] else recs[0]["baseline"]
        vals = [r["adapted"] for r in recs]
        print(f"\n  {p['stage'].upper()}.{p['key']}")
        print(f"  Tunnels: {p['n_tunnels']}/30 | LLMs: {p['n_llms']}/3 | Conditions: {p['n_conds']}/3 | Total changes: {p['n_changes']}")
        print(f"  Baseline: {bl_val}")

        if isinstance(bl_val, (int, float)):
            numeric_vals = [v for v in vals if isinstance(v, (int, float))]
            if numeric_vals:
                print(f"  Adapted range: [{min(numeric_vals)}, {max(numeric_vals)}], mean: {sum(numeric_vals)/len(numeric_vals):.4f}, std: {(sum((v - sum(numeric_vals)/len(numeric_vals))**2 for v in numeric_vals)/len(numeric_vals))**0.5:.4f}")

        # Show per-LLM breakdown
        for llm in LLMS:
            llm_recs = [r for r in recs if r["llm"] == llm]
            llm_tunnels = set(r["tunnel"] for r in llm_recs)
            if llm_recs:
                llm_vals = [r["adapted"] for r in llm_recs if isinstance(r["adapted"], (int, float))]
                if llm_vals:
                    print(f"    {llm:15s}: {len(llm_tunnels):2d} tunnels, range [{min(llm_vals):.4f}, {max(llm_vals):.4f}], mean {sum(llm_vals)/len(llm_vals):.4f}")
                else:
                    print(f"    {llm:15s}: {len(llm_tunnels):2d} tunnels adapted")

    # ========== PART 3: Per-LLM summary ==========
    print("\n" + "=" * 100)
    print("PART 3: PER-LLM ADAPTATION SUMMARY")
    print("=" * 100)

    for llm in LLMS:
        print(f"\n--- {llm} ---")
        for stage in STAGES:
            changes = param_changes_by_llm[llm][stage]
            if not changes:
                print(f"  {stage}: no adaptations")
                continue
            n_params = len(changes)
            total_changes = sum(len(v) for v in changes.values())
            tunnel_coverage = set()
            for recs in changes.values():
                for r in recs:
                    tunnel_coverage.add(r["tunnel"])
            print(f"  {stage:12s}: {n_params:3d} params adapted, {total_changes:4d} total changes, {len(tunnel_coverage):2d}/30 tunnels")

    # ========== PART 4: Per-tunnel variation ==========
    print("\n" + "=" * 100)
    print("PART 4: PARAMETER VALUE VARIATION ACROSS TUNNELS (for top 'always-trigger' params)")
    print("Shows how much the adapted value changes between different tunnels")
    print("=" * 100)

    for p in always_trigger[:15]:
        recs = p["records"]
        bl_val = recs[0]["baseline"]
        if not isinstance(bl_val, (int, float)):
            continue

        print(f"\n  {p['stage'].upper()}.{p['key']} (baseline: {bl_val})")

        # Per-tunnel mean across LLMs and conditions
        tunnel_means = defaultdict(list)
        for r in recs:
            if isinstance(r["adapted"], (int, float)):
                tunnel_means[r["tunnel"]].append(r["adapted"])

        tunnel_avg = {t: sum(vs)/len(vs) for t, vs in tunnel_means.items()}
        sorted_tunnels = sorted(tunnel_avg.items(), key=lambda x: x[1])

        min_t, min_v = sorted_tunnels[0]
        max_t, max_v = sorted_tunnels[-1]
        all_vals = list(tunnel_avg.values())
        overall_mean = sum(all_vals) / len(all_vals)
        cv = ((sum((v - overall_mean)**2 for v in all_vals) / len(all_vals)) ** 0.5) / abs(overall_mean) if overall_mean != 0 else 0

        print(f"  CV (coefficient of variation): {cv:.4f}")
        print(f"  Min: tunnel {min_t} = {min_v:.4f} | Max: tunnel {max_t} = {max_v:.4f} | Mean: {overall_mean:.4f}")
        print(f"  Per-tunnel values: ", end="")
        for t, v in sorted_tunnels:
            pct = ((v - bl_val) / abs(bl_val) * 100) if bl_val != 0 else 0
            print(f"{t}={v:.3f}({pct:+.0f}%) ", end="")
        print()

    # ========== PART 5: New parameters (not in baseline) ==========
    if any(new_params[s] for s in STAGES):
        print("\n" + "=" * 100)
        print("PART 5: NEW PARAMETERS (present in adapted but NOT in baseline)")
        print("=" * 100)
        for stage in STAGES:
            if new_params[stage]:
                print(f"\n  {stage.upper()}:")
                for key, recs in sorted(new_params[stage].items()):
                    print(f"    {key}: {len(recs)} occurrences")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
