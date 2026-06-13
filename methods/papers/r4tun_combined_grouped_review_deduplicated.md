# R4Tun Combined Grouped Review (Deduplicated)

Source basis:
- `r4tun_final_manuscript_sentence_review.md`
- `r4tun_review_v4_pre_submission_review.md`

Scope: deduplicated, grouped, non-redundant issue list that preserves all unique points raised across both reviews.

## 1) Numerical And Consistency Issues (Highest Priority)

### 1. Abstract OA clause is grammatically incomplete
**Original**
> ... increased mean mIoU by at least 110.7% and 26.8% Overall Accuracy (OA).

**Issue**
Reads as if OA itself is 26.8% rather than OA improvement.

**Suggested revision**
> ... increased mean mIoU by 110.7%–128.0% and overall accuracy (OA) by 26.8%–34.1% relative to the static SAM4Tun baseline.

### 2. Relative-improvement ranges conflict across sections
**Issue**
Conclusions reports values recomputed from rounded numbers, while Highlights/Abstract/Results use unrounded values.

**Conflicts to align**
- Overall: `113.3%–126.7%` vs `110.7%–128.0%`
- Regular: `+72.4%–86.2%` vs `+70.0%–83.8%`
- Complex: `+350.0%–375.0%` vs `+323.1%–361.7%`

**Suggested revision**
Use one canonical set everywhere (preferably Section 4.1 values from unrounded means).

### 3. Four-stage description conflicts with five API calls
**Original**
- Four stage agents are described in methodology.
- Appendix reports `LLM API calls per tunnel: 5` and totals based on 5.

**Issue**
Either an unmentioned stage exists (e.g., detecting), or API-call counts are incorrect.

**Suggested revision**
- If four adapted stages: `4 calls per tunnel`, `120 per condition per LLM`.
- If five adapted stages: explicitly add the fifth stage in architecture/method sections and figures.

### 4. Reference diameter definitions are inconsistent (5.5 / 5.60 / 5.32)
**Issue**
Nominal and estimated diameters are mixed without explicit definition, which also causes `+34%` vs `+39%` confusion.

**Suggested revision**
Define once:
- Nominal diameter = design value (e.g., 5.60, 7.50)
- Estimated diameter = measured from point-cloud processing (e.g., 5.32, 7.41)

### 5. Parameter-count drift ("approximately 80" vs "81")
**Issue**
TeX and PDF differ.

**Suggested revision**
Use one exact count and cross-check against appendix parameter tables.

### 6. "Default API configuration" conflicts with `Temperature = 0`
**Issue**
Temperature zero is an explicit override.

**Suggested revision**
> Other than setting temperature to 0, all other API settings used vendor defaults.

### 7. Citation [50] does not support all listed models
**Issue**
[50] is GPT-4o system card, while table lists Opus-4.6, GPT-5.4, Gemini-3-Flash.

**Suggested revision**
Remove [50] from that sentence, or add correct per-vendor references.

### 8. Terminology and formatting drift
**Issue**
Inconsistent use of:
- `Non-LLM` vs `non-LLM`
- `tunnel family` vs `tunnel category`
- `regular-category` vs `regular category`
- `95 %` vs `95%`

**Suggested revision**
Standardize globally (recommended: `non-LLM`, `tunnel category`, noun form without hyphen, compact `%`).

## 2) Overstatements To Soften

### 9. "strong adaptability" and similar broad claims
**Original**
> Existing tunnel-segmentation approaches do not jointly provide strong adaptability...

**Issue**
Too categorical/subjective.

**Suggested revision**
> ...have not yet demonstrated, in a single workflow, both robust adaptation to changed conditions and an auditable adaptation process.

### 10. "self-verifying intermediate steps"
**Issue**
Overstates reliability and conflicts with later caveat that constraints are not guaranteed.

**Suggested revision**
Use "prompted consistency checks" and acknowledge probabilistic nature.

### 11. "objective tunnel conditions" in cross-model analysis
**Issue**
Cannot isolate "objective condition" causality from shared prompt/knowledge/model priors.

**Suggested revision**
Phrase as consistency under same inputs/prompt structure.

### 12. "context design is shown to help"
**Issue**
Too assertive for evidence level.

**Suggested revision**
Use "ablation results indicate/suggest".

### 13. LLM-vs-rule conclusion hedging is inconsistent
**Issue**
Some sections use cautious wording ("consistent with additional contribution"), others sound definitive.

**Suggested revision**
Use cautious causal language throughout.

### 14. Practical-impact sentence sounds promotional
**Original**
"reallocates expert effort", "one-off authoring step", "affordable APIs".

**Issue**
Reads as promotional and overgeneralized.

**Suggested revision**
Use "may shift" language; call API costs "indicative under tested settings".

## 3) Internal Logic / Reasoning Consistency

### 15. "strictly bounded" conflicts with "does not guarantee constraints"
**Issue**
Deterministic wording clashes with non-deterministic implementation.

**Suggested revision**
Replace "strictly bounded" with "instructed to stay within" and clarify non-guaranteed enforcement.

### 16. Convergence sentence is internally mixed
**Original**
Claims rule+LLM "converge" yet "LLMs still score higher" in same long sentence.

**Suggested revision**
Split into two sentences:
- Both remain low in absolute performance on complex category.
- LLM still has small mean advantage.

## 4) AI-Like / Vague / Promotional Phrasing

### 17. Replace vague or hype-like wording
Target phrases:
- "handle varying tunnel conditions"
- "degrades silently"
- "diagnostic feedback"
- "explicit rationale"
- "leverage ... foundation models"
- "actual geometric reality"
- "the agent attempts to..."

**Suggested style**
Use concrete, observable language tied to measured effects and pipeline behavior.

### 18. Circular wording in dataset description
**Original**
> complex interleaved key-block arrangement

**Issue**
"complex" for complex category is circular.

**Suggested revision**
> interleaved key-block arrangement that does not repeat ring-to-ring.

## 5) Repetition To Reduce

### 19. Repeated triplet and near-identical claims
Repeated heavily:
- "memory, state, and knowledge"
- "reference configuration"
- "per-tunnel expert re-tuning/retuning"
- duplicated "state provides quantitative evidence..." sentence
- duplicated per-class IoU summary sentence

**Suggested revision**
Define once, then use shorthand (`m+s+k`) and cross-references.

### 20. Keep one strong version of each finding
Examples:
- keep per-class summary once in Results, brief reference in Conclusions.
- keep state-mechanism explanation once in Ablation, brief callback in Discussion.

## 6) Grammar, Typo, And Formatting Fixes

### 21. Typo and spacing
- `JSON ouput` -> `JSON output`
- `protocol(Algorithm 1)` -> `protocol (Algorithm 1)`
- `The mean increment are narrow` -> `The mean increments are small`

### 22. Figure 6 caption grammar
**Issue**
Missing grammatical subject for "constructs" / "feeds".

**Suggested revision**
Use a clear subject and two-step sentence structure.

### 23. Appendix heading grammar
`Non-LLM rules-based` -> `Non-LLM rule-based`.

### 24. Unit and symbols
- `pts/m3` -> `pts/m³` (or journal-preferred equivalent)
- normalize `%` spacing
- verify math rendering in PDF (`ar{v}`, summation bounds) is typographically correct.

### 25. Source artefact in TeX
Remove stray `t d` in the sentence beginning "The pipeline contains...".

## 7) Structure And Organization Fixes

### 26. Move metrics/sensitivity out of Non-LLM subsection
**Issue**
`3.5.1 Evaluation metrics` and `3.5.2 Sensitivity analysis` are global methods, not specific to non-LLM baseline.

**Suggested revision**
Renumber as standalone methodology subsections.

### 27. Align Highlights and Contributions
Use identical headline claims and numbers in both places after numeric reconciliation.

### 28. Verify Algorithm 1 float placement/caption
Ensure caption and body render together and are not detached across columns/pages.

## 8) Consolidated High-Quality Abstract (Optional Replacement)

> Automated inspection of segmental tunnel linings requires segmentation pipelines that adapt to varying point-cloud geometry, yet expert-tuned pipelines often degrade when tunnel conditions depart from the reference setting. This paper presents R4Tun, a large language model (LLM)-driven adaptation framework that extends an expert-designed pipeline (SAM4Tun) with bounded parameter tuning informed by structured context (memory, state, and knowledge; m+s+k). On 30 selected Seg2Tunnel subsets (13 regular, 17 complex) and three LLMs (Opus-4.6, GPT-5.4, Gemini-3-Flash), the m+s+k design increased mean Intersection-over-Union (mIoU) from 0.150 to 0.316–0.342 (+110.7%–128.0%) and overall accuracy (OA) from 0.411 to 0.522–0.552 (+26.8%–34.1%). On regular tunnels mIoU rose from 0.291 to 0.495–0.535 (+70.0%–83.8%); on complex tunnels mIoU rose from 0.042 to 0.178–0.194 (+323.1%–361.7%). Across 270 adapted runs (30 tunnels × 3 LLMs × 3 context settings), the three LLMs followed similar parameter-adjustment trends and consistently selected a shared set of critical parameters. These results suggest that LLM-driven adaptation supplied with structured context can improve transfer of a fixed segmentation pipeline across changed tunnel conditions and reduce the need for per-tunnel expert retuning, particularly when target tunnels remain close to the reference configuration.

## 9) One-Pass Final Checklist

1. Unify all numeric ranges in Highlights/Abstract/Results/Conclusions.
2. Resolve 4-vs-5 stage inconsistency and update architecture, methods, tables.
3. Define nominal vs estimated diameter once and reuse consistently.
4. Standardize terms/capitalization/percent style globally.
5. Replace unsupported or mismatched citations.
6. Soften overstatements; keep causal claims cautious.
7. Remove duplicated wording; keep one canonical statement per finding.
8. Apply typo/grammar/unit/math-format fixes.
9. Reorganize subsection numbering for methodological clarity.
10. Verify final PDF rendering (figures, algorithm floats, equations).
