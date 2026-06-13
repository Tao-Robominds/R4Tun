# Defending Plan: Additional Experiments for R4Tun Paper

**Claim:** "LLM can improve robustness under varying tunnel conditions by adapting stage parameters from reference parameter values."

**Defence requirement:** Claims based on models need sensitivity analysis, recognition of limitations, summaries of the quality of the underlying evidence, and assessments of confidence in the whole analysis.

---

## Experiment 1: Run-to-run variance (PAUSED --- GPU intensive)

**Purpose:** Quantify pipeline non-determinism (primarily SAM GPU variance) to confirm that reported condition deltas exceed run-to-run noise.

**Method:** Re-run the full pipeline 5 times on all 30 tunnels under two conditions: (a) sam4tun baseline, (b) m_s_k with Opus 4.6. Reuse existing parameter JSONs (no new LLM calls).

**Cost:** 300 full pipeline runs, ~25-50 hours GPU time. No API calls.

**Deliverable:** Per-tunnel mean +/- std mIoU. 95% confidence intervals. Confirm delta > noise.

---

## Experiment 2: LLM stochasticity (IN PROGRESS --- 2-run protocol)

**Purpose:** Quantify within-LLM variance in parameter inference.

**Method:** Run 1 = primary params (snapshot). Run 2 = one fresh m+s+k inference per tunnel×LLM (temperature 0), with run 1 seeded to skip GPU when params match. See `methods/plans/repeatability_run2_plan.md`.

**Scripts:** `bootstrap_repeatability_run1.py`, `run_repeatability.py`, `reproducibility_analysis.py --layout repeatability`.

**Status:** 9/90 pairs from harvested complex-tunnel reruns (Appendix I); remaining 81 via `run_repeatability.py --skip-existing`.

**Deliverable:** `methods/papers/output/repeatability_summary.md`, Appendix I Table.

---

## Experiment 3: Per-class IoU extraction (DONE)

**Script:** `methods/papers/scripts/extract_per_class_iou.py`

**Outputs:** `methods/papers/output/per_class_iou_summary.md`, `per_class_iou_long.csv`

**Note:** Only lines under `## Per-class IoU` are parsed (avoids OA/F1 from other sections).

---

## Experiment 4: Wilcoxon signed-rank test (DONE)

**Script:** `methods/papers/scripts/wilcoxon_test.py`

**Output:** `methods/papers/output/wilcoxon_vs_ttest.md`

---

## Integration into paper (done)

- Exp 3: subsection under Section 4.1 + links to output files; Appendix A documents the script.
- Exp 4: Section 3.5.3 + link to `wilcoxon_vs_ttest.md`; Appendix A documents the script.
- Appendix A: BO and parameter-level ablation removed; deferred experiments listed under A.2.
- Exp 1 / Exp 2: still deferred (see A.2 in `methods/papers/r4tun.md`).
