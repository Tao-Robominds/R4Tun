"""Step 7 — Render paper-ready LaTeX tables and a narrative-ready summary.

Reads only the frozen JSON artefacts in ``data/v3/calibration/`` (and
``baseline_vs_bo.csv`` for the per-ring scoreboard), no recomputation,
and writes:

* ``papers/calibration_tables.tex`` — four ``table`` floats with
  ``tabular`` blocks (sensitive parameters, diagnostic intrinsics,
  guardrail bundles, per-ring scoreboard).
* Appends a "Calibration outcomes summary" block to
  ``data/v3/calibration/calibration_report.md``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = REPO_ROOT / "data" / "v3" / "calibration"
TEX_OUT = REPO_ROOT / "papers" / "calibration_tables.tex"
REPORT_OUT = CAL_DIR / "calibration_report.md"
SUMMARY_BEGIN = "<!-- BEGIN auto-generated calibration outcomes summary -->"
SUMMARY_END = "<!-- END auto-generated calibration outcomes summary -->"


def _esc(s: Any) -> str:
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("_", r"\_")
         .replace("%", r"\%")
         .replace("&", r"\&")
         .replace("#", r"\#")
         .replace("$", r"\$")
    )


def _fmt_int_or_float(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    if f.is_integer() and abs(f) < 1e6:
        return f"{int(f):d}"
    return f"{f:.{decimals}f}"


def _fmt_default(v: Any) -> str:
    """Format a tunable parameter's default that may be int or float."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v:d}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    if abs(f) >= 1.0:
        return f"{f:.2f}"
    if abs(f) >= 0.01:
        return f"{f:.3f}"
    return f"{f:.4f}"


def _render_tab1_sensitive(frozen: dict[str, Any]) -> str:
    rows = []
    for tp in frozen["tunable_parameters"]:
        seed = _fmt_default(tp.get("default_r4tun_seed"))
        deploy = _fmt_default(tp.get("default_deployable"))
        soft = tp.get("soft_bounds_p25_p75") or [None, None]
        hard = tp.get("hard_bounds_min_max") or [None, None]
        soft_s = f"[{_fmt_default(soft[0])}, {_fmt_default(soft[1])}]"
        hard_s = f"[{_fmt_default(hard[0])}, {_fmt_default(hard[1])}]"
        rho = float(tp["pooled_spearman_vs_miou"])
        rho_s = f"{abs(rho):.3f}"
        n_rings = len(tp.get("evidence_rings") or [])
        deploy_disp = seed if seed == deploy else f"{seed} ($\\to$ {deploy})"
        rows.append(
            f"{_esc(tp['name'])} & {deploy_disp} & {soft_s} & {hard_s} & {rho_s} & {n_rings}/6 \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[!t]\n"
        "\\caption{Sensitive preprocessing parameters retained for the deployment-time LLM loop. "
        "Selection rule: pooled $|\\rho_{\\mathrm{Spearman}}|\\geq 0.20$ vs canonical mIoU and "
        "evidence in $\\geq 3$ of 6 calibration rings. The R4Tun-seed default is the value the regular-tunnel "
        "reference produces after schema mapping; values shown as $a (\\to b)$ are clipped at deployment to the "
        "BO hard upper bound. Soft bounds are the $[p_{25}, p_{75}]$ envelope of the upper-quartile-mIoU regime "
        "across 180 successful preprocessing trials; hard bounds are the BO search-space limits.}\n"
        "\\label{tab:cal-sensitive-params}\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Parameter & Default & Soft bounds $[p_{25}, p_{75}]$ & Hard bounds & $|\\rho|$ & Rings \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _render_tab2_diagnostics(frozen: dict[str, Any]) -> str:
    rows = []
    for d in frozen["diagnostic_intrinsics"]:
        units = d.get("units") or ""
        # Trim long units to the parenthetical-free leading clause if present.
        units_short = units.split(";")[0]
        thr_strict = d.get("min_good_threshold_p25_top_quartile")
        thr_perm = d.get("permissive_threshold_top_quartile_min")
        rho = float(d["pooled_spearman_vs_miou"])
        rows.append(
            f"{_esc(d['name'])} & {_esc(d['stage_source'])} & {_esc(units_short)} & "
            f"{rho:+.3f} & {_fmt_int_or_float(thr_strict, 4)} & {_fmt_int_or_float(thr_perm, 4)} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[!t]\n"
        "\\caption{Diagnostic intrinsics that the LLM reflection loop reads at deployment. "
        "Selection rule: pooled $\\rho_{\\mathrm{Spearman}}\\geq 0.50$ vs canonical mIoU on the "
        "preprocessing-stage corpus and $\\geq 90\\%$ non-null across 360 successful trials. "
        "Strict threshold is the $p_{25}$ of the upper-quartile-mIoU regime; permissive threshold is the "
        "minimum value seen in the upper-quartile regime. Booleans use $0/1$.}\n"
        "\\label{tab:cal-diag-intrinsics}\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{p{0.27\\columnwidth}lp{0.30\\columnwidth}rrr}\n"
        "\\toprule\n"
        "Intrinsic & Stage & Units & $\\rho$ & Strict & Permissive \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _render_tab3_guardrails(frozen: dict[str, Any]) -> str:
    g = frozen["guardrails"]
    rows: list[str] = []
    for gname in ("G_pre", "G_layout", "G_stability"):
        bundle = g[gname]
        ix_list = ", ".join(_esc(i) for i in bundle["intrinsics"])
        rule_raw = bundle["rule"]
        rule = _esc(rule_raw)
        thr_s = bundle.get("thresholds_strict") or {}
        thr_p = bundle.get("thresholds_permissive") or {}
        op = "$\\geq$" if rule_raw == "all_of_min" else "$\\leq$"
        thr_s_str = "; ".join(f"{_esc(k)}{op}{_fmt_int_or_float(v, 4)}"
                              for k, v in thr_s.items())
        thr_p_str = "; ".join(f"{_esc(k)}{op}{_fmt_int_or_float(v, 4)}"
                              for k, v in thr_p.items())
        # Format bundle name as G_{subscript} in math mode.
        if "_" in gname:
            head, sub = gname.split("_", 1)
            label = f"\\textbf{{${head}_{{\\mathrm{{{sub}}}}}$}}"
        else:
            label = f"\\textbf{{{_esc(gname)}}}"
        rows.append(
            f"{label} & {ix_list} & \\texttt{{{rule}}} & {thr_s_str} & {thr_p_str} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[!t]\n"
        "\\caption{Guardrail bundles emitted by the calibration. Each bundle returns a Boolean verdict from "
        "its rule (\\texttt{all\\_of\\_min}: every constituent intrinsic must clear its lower threshold; "
        "\\texttt{max\\_below}: the constituent must be below its upper threshold). G\\_stability uses the "
        "permutation-vs-fixed mIoU gap on 360 pooled trials; strict is the corpus median, permissive the $p_{75}$.}\n"
        "\\label{tab:cal-guardrails}\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\begin{tabular}{p{0.10\\columnwidth}p{0.30\\columnwidth}lp{0.22\\columnwidth}p{0.22\\columnwidth}}\n"
        "\\toprule\n"
        "Bundle & Constituent intrinsics & Rule & Strict thresholds & Permissive thresholds \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _render_tab4_scoreboard(frozen: dict[str, Any]) -> str:
    sb = frozen["calibration_scoreboard"]
    rows = []
    for r in sb:
        ring = _esc(r["ring_key"])
        regime_short = _esc(r["regime_label"]).replace("\\_", "/")
        bf = float(r["baseline_fixed"])
        pre = float(r["preproc_bo_fixed"])
        det = float(r["detection_bo_fixed"])
        best = max(bf, pre, det)
        delta = best - bf
        rows.append(
            f"{ring} & {regime_short} & {bf:.3f} & {pre:.3f} & {det:.3f} & {best:.3f} & {delta:+.3f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{table}[!t]\n"
        "\\caption{Per-ring fixed-class canonical mIoU on the 6 BO calibration rings. The baseline "
        "applies the gravity-anchored R4Tun seed parameters; the preprocessing-BO column reports the best "
        "preprocessing trial (detection at seed values), and the detection-BO column reports the best "
        "detection trial (preprocessing at seed values). $\\Delta$ is the best-of-three minus baseline; the "
        "scoreboard motivates LLM tuning of preprocessing only.}\n"
        "\\label{tab:cal-scoreboard}\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llrrrrr}\n"
        "\\toprule\n"
        "Ring & Regime & Baseline & Preproc.~BO & Detection BO & Best & $\\Delta$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _summary_block(frozen: dict[str, Any]) -> str:
    ev = frozen["evidence"]
    fm_pre = ev["failure_modes"].get("preprocessing", {})
    fm_det = ev["failure_modes"].get("detection", {})
    det_top = ev.get("detection_top_pooled_abs_spearman", 0.0)
    g_stab = frozen["guardrails"]["G_stability"]
    median_gap = (g_stab.get("thresholds_strict") or {}).get("miou_perm_minus_fixed_gap")
    p75_gap = (g_stab.get("thresholds_permissive") or {}).get("miou_perm_minus_fixed_gap")
    p90_gap = (g_stab.get("warning_threshold_p90") or {}).get("miou_perm_minus_fixed_gap")

    sb = frozen["calibration_scoreboard"]
    n_det_beats_baseline = sum(
        1 for r in sb if float(r["detection_bo_fixed"]) > float(r["baseline_fixed"])
    )
    n_pre_beats_baseline = sum(
        1 for r in sb if float(r["preproc_bo_fixed"]) > float(r["baseline_fixed"])
    )
    n_total = len(sb)

    bullets = [
        (
            f"**Detection params are frozen at deployment.** All 21 detection knobs have pooled "
            f"$|\\rho_{{\\mathrm{{Spearman}}}}|\\leq {det_top:.3f}$ versus canonical mIoU; calibration evidence "
            "does not support LLM tuning of detection. On the calibration panel the detection-BO winner beat "
            f"baseline on {n_det_beats_baseline}/{n_total} rings (per-ring numbers in Table~\\ref{{tab:cal-scoreboard}}); "
            "the per-ring records are preserved in `baseline\\_vs\\_bo.csv` but not promoted to deployment policy."
        ),
        (
            f"**Preprocessing failure mode characterised.** Of {fm_pre.get('n_attempted', 0)} preprocessing trials, "
            f"{fm_pre.get('n_failed', 0)} ({fm_pre.get('failure_rate_pct', 0):.1f}%) failed with "
            f"modes={fm_pre.get('modes') or {}}; detection had "
            f"{fm_det.get('n_failed', 0)} failures over {fm_det.get('n_attempted', 0)} trials. The G\\_pre "
            "guardrail is the loop-side analogue of these failures: it gates downstream tuning on a usable depth "
            "map. Failed trials are excluded from the GP, included in the failure-mode evidence."
        ),
        (
            f"**A 0.176 median permutation-vs-fixed mIoU gap on the calibration corpus motivates G\\_stability "
            "and frames the held-out anchored-vs-unanchored comparison the paper will report later.** "
            f"Pooled distribution: median {median_gap:.3f}, $p_{{75}}$ {p75_gap:.3f}, $p_{{90}}$ {p90_gap:.3f}. "
            "G\\_stability's strict cut uses the median and the operational cut uses $p_{75}$; numbers above "
            "$p_{90}$ flag the LLM loop with a 'residual canonical-anchoring failure' warning. Preprocessing "
            f"BO improved the fixed-class mIoU above baseline on {n_pre_beats_baseline}/{n_total} calibration rings "
            "without removing the gap, which is the empirical case for treating canonical anchoring as a "
            "separate failure mode the LLM loop must monitor."
        ),
    ]
    body = "\n\n".join(f"- {b}" for b in bullets)
    return (
        f"{SUMMARY_BEGIN}\n"
        "## Calibration outcomes summary (auto-generated)\n\n"
        f"{body}\n\n"
        f"_Source: `data/v3/calibration/llm_loop_frozen.json`. Rendered by `bo/v3/render_calibration_tables.py`._\n"
        f"{SUMMARY_END}\n"
    )


def _replace_or_append(report: Path, block: str) -> None:
    if not report.exists():
        report.write_text(block)
        return
    text = report.read_text()
    if SUMMARY_BEGIN in text and SUMMARY_END in text:
        before = text.split(SUMMARY_BEGIN, 1)[0].rstrip() + "\n\n"
        after_pieces = text.split(SUMMARY_END, 1)
        after = ("\n" + after_pieces[1].lstrip()) if len(after_pieces) == 2 else "\n"
        report.write_text(before + block + after)
    else:
        report.write_text(text.rstrip() + "\n\n" + block)


def main() -> int:
    frozen = json.loads((CAL_DIR / "llm_loop_frozen.json").read_text())
    parts = [
        "% Auto-generated by bo/v3/render_calibration_tables.py from data/v3/calibration/llm_loop_frozen.json.",
        "% Do not edit by hand; rerun the renderer instead.",
        "",
        _render_tab1_sensitive(frozen),
        _render_tab2_diagnostics(frozen),
        _render_tab3_guardrails(frozen),
        _render_tab4_scoreboard(frozen),
    ]
    TEX_OUT.write_text("\n".join(parts))
    print(f"wrote {TEX_OUT}")

    block = _summary_block(frozen)
    _replace_or_append(REPORT_OUT, block)
    print(f"appended summary to {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
