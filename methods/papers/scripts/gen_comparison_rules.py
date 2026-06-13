#!/usr/bin/env python3
"""Generate methods/journals/comparison_rules.md

Compares rules baseline vs all 3 LLM models' m+s+k condition across 30 tunnels.
Includes per-tunnel detail table and family-level summary with t-tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

REPO = Path(__file__).resolve().parents[3]

RULES = {
    "1-1": 0.370, "1-2": 0.317, "1-3": 0.404, "1-4": 0.275, "1-5": 0.484,
    "2-1": 0.341, "2-2": 0.401, "2-3": 0.286, "2-4": 0.418, "2-5": 0.224,
    "3-1-1": 0.088, "3-1-2": 0.080, "3-1-3": 0.029,
    "4-1": 0.143, "4-2": 0.000, "4-3": 0.146, "4-4": 0.268,
    "4-5": 0.170, "4-6": 0.144, "4-7": 0.155, "4-8": 0.203,
    "4-9": 0.092, "4-10": 0.135,
    "5-1": 0.155, "5-2": 0.144, "5-3": 0.231, "5-4": 0.142,
    "5-5": 0.000, "5-6": 0.197, "5-7": 0.000,
}

JOURNALS = [
    ("Opus-4.6", REPO / "methods" / "journals" / "comparison_anthropic.md"),
    ("GPT-5.4", REPO / "methods" / "journals" / "comparison_openai.md"),
    ("Gemini-3-Flash", REPO / "methods" / "journals" / "comparison_gemini.md"),
]

TUNNEL_ORDER = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
    "4-1", "4-2", "4-3", "4-4", "4-5", "4-6", "4-7", "4-8", "4-9", "4-10",
    "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7",
]

TYPE_MAP = {
    **{f"{f}-{i}": "reg" for f in (1, 2) for i in range(1, 6)},
    "3-1-1": "con", "3-1-2": "con", "3-1-3": "con",
    **{f"4-{i}": "com" for i in range(1, 11)},
    **{f"5-{i}": "com" for i in range(1, 8)},
}

FAMILIES = {
    "Overall": lambda t: True,
    "Regular (reg+con)": lambda t: t in ("reg", "con"),
    "Complex": lambda t: t == "com",
}


def parse_journal(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or "tunnel_id" in line or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 8:
            continue
        tid, typ = parts[0], parts[1]
        if typ not in ("reg", "con", "com"):
            continue
        try:
            rows[tid] = {
                "type": typ,
                "sam4tun": float(parts[2]),
                "memory": float(parts[3]),
                "memory+state": float(parts[5]),
                "m_s_k": float(parts[7]),
            }
        except (ValueError, IndexError):
            continue
    return rows


def fmt_p(p: float) -> str:
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"


def main() -> None:
    model_data: dict[str, dict[str, dict]] = {}
    for name, path in JOURNALS:
        model_data[name] = parse_journal(path)
        assert len(model_data[name]) == 30, f"{name}: {len(model_data[name])} tunnels"

    lines: list[str] = []
    w = lines.append

    w("# Rules vs LLM (m+s+k) Comparison Report")
    w("")
    w("Rules baseline (field-observable heuristics, n=30, 3 failed tunnels scored as 0) "
      "vs each LLM's best ablation condition (memory+state+knowledge).")
    w("")

    # --- Per-tunnel table ---
    w("## Per-tunnel mIoU")
    w("")
    w("| tunnel_id | type | sam4tun | rules | Opus-4.6 | GPT-5.4 | Gemini-3-Flash | 3-model avg | best |")
    w("| --------- | ---- | ------- | ----- | -------- | ------- | -------------- | ----------- | ---- |")

    for tid in TUNNEL_ORDER:
        typ = TYPE_MAP[tid]
        sam = model_data["Opus-4.6"][tid]["sam4tun"]
        rules_v = RULES[tid]
        msk = {name: model_data[name][tid]["m_s_k"] for name, _ in JOURNALS}
        avg3 = np.mean(list(msk.values()))
        best = "rules" if rules_v > avg3 else ("tie" if abs(rules_v - avg3) < 0.0005 else "LLM")
        w(f"| {tid:<9} | {typ}  | {sam:.3f}   | {rules_v:.3f} | "
          f"{msk['Opus-4.6']:.3f}    | {msk['GPT-5.4']:.3f}   | "
          f"{msk['Gemini-3-Flash']:.3f}          | {avg3:.3f}       | {best} |")

    w("")

    # --- Family-level summary ---
    w("## Family-level Summary")
    w("")

    for fam_name, fam_pred in FAMILIES.items():
        tids = [t for t in TUNNEL_ORDER if fam_pred(TYPE_MAP[t])]
        n = len(tids)
        w(f"### {fam_name} (n={n})")
        w("")
        w("| condition | mean mIoU | delta vs sam4tun | std (delta) | p-value |")
        w("| --------- | --------- | ---------------- | ----------- | ------- |")

        sam_arr = np.array([model_data["Opus-4.6"][t]["sam4tun"] for t in tids])
        sam_mean = np.mean(sam_arr)
        w(f"| sam4tun (baseline) | {sam_mean:.3f} | — | — | — |")

        rules_arr = np.array([RULES[t] for t in tids])
        rules_mean = np.mean(rules_arr)
        rules_delta = rules_arr - sam_arr
        rd_mean = np.mean(rules_delta)
        rd_std = np.std(rules_delta, ddof=1)
        _, rp = ttest_rel(rules_arr, sam_arr)
        w(f"| rules | {rules_mean:.3f} | {rd_mean:+.3f} | {rd_std:.3f} | {fmt_p(rp)} |")

        for model_name, _ in JOURNALS:
            msk_arr = np.array([model_data[model_name][t]["m_s_k"] for t in tids])
            msk_mean = np.mean(msk_arr)
            delta = msk_arr - sam_arr
            d_mean = np.mean(delta)
            d_std = np.std(delta, ddof=1)
            _, p = ttest_rel(msk_arr, sam_arr)
            w(f"| {model_name} m+s+k | {msk_mean:.3f} | {d_mean:+.3f} | {d_std:.3f} | {fmt_p(p)} |")

        avg3_arr = np.mean(
            [np.array([model_data[mn][t]["m_s_k"] for t in tids]) for mn, _ in JOURNALS],
            axis=0,
        )
        avg3_mean = np.mean(avg3_arr)
        avg3_delta = avg3_arr - sam_arr
        a3d_mean = np.mean(avg3_delta)
        a3d_std = np.std(avg3_delta, ddof=1)
        _, a3p = ttest_rel(avg3_arr, sam_arr)
        w(f"| **3-model avg m+s+k** | **{avg3_mean:.3f}** | **{a3d_mean:+.3f}** | **{a3d_std:.3f}** | **{fmt_p(a3p)}** |")

        w("")

    # --- Wins table ---
    w("## Rules vs 3-model avg m+s+k — wins per family")
    w("")
    w("| family | rules wins | LLM wins | ties |")
    w("| ------ | ---------- | -------- | ---- |")
    for fam_name, fam_pred in FAMILIES.items():
        tids = [t for t in TUNNEL_ORDER if fam_pred(TYPE_MAP[t])]
        rw = lw = tw = 0
        for t in tids:
            rv = RULES[t]
            avg = np.mean([model_data[mn][t]["m_s_k"] for mn, _ in JOURNALS])
            if rv > avg + 0.0005:
                rw += 1
            elif avg > rv + 0.0005:
                lw += 1
            else:
                tw += 1
        w(f"| {fam_name} | {rw} | {lw} | {tw} |")

    w("")

    out_path = REPO / "methods" / "journals" / "comparison_rules.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
