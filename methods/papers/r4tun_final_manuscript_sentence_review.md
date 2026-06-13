# R4Tun Final Manuscript Sentence Review

Source reviewed: `R4Tun_Journal_Review-4.pdf`

Scope: inconsistencies, overstatements, AI-like or promotional phrasing, repetition, grammar, and sentence-level academic style. The revisions below keep the paper's current claims but make them more cautious, internally consistent, and formal.

## Highest-Priority Inconsistencies

### 1. Abstract: mIoU and OA improvement phrasing is grammatically incomplete

**Original sentence**

> Evaluated on 30 selected Seg2Tunnel subsets (13 regular, 17 complex) across three LLMs, this design increased mean Intersection-over-Union (mIoU) by at least 110.7% and 26.8% Overall Accuracy (OA).

**Issue**

The phrase "and 26.8% Overall Accuracy" is grammatically incomplete. It also reads as if OA itself is 26.8%, rather than the relative increase in OA.

**Suggested revision**

> Evaluated on 30 selected Seg2Tunnel subsets (13 regular, 17 complex) across three LLMs, the full memory + state + knowledge design increased mean Intersection-over-Union (mIoU) by 110.7%--128.0% and overall accuracy (OA) by 26.8%--34.1% relative to the static SAM4Tun baseline.

### 2. Conclusion: relative improvement values do not match the abstract/results

**Original sentence**

> Evaluated on 30 subsets spanning 13 regular and 17 complex tunnels with three independent LLMs, the full memory + state + knowledge design raised mean mIoU from 0.15 to 0.32–0.34 (a relative increase of 113.3%–126.7%; bootstrap 95 % CIs exclude zero for all three LLMs), with per-class IoU improving broadly across structural classes for regular tunnels and recovering progressively from near-zero baselines for complex tunnels (regular: 0.29 → 0.50–0.54, +72.4%–86.2%; complex: 0.04 → 0.18–0.19, +350.0%–375.0%).

**Issue**

The relative increases differ from the abstract and Section 4.1, which report 110.7%--128.0% overall, 70.0%--83.8% for regular tunnels, and 323.1%--361.7% for complex tunnels. Use one set of values throughout.

**Suggested revision**

> Evaluated on 30 Seg2Tunnel subsets (13 regular and 17 complex) with three independent LLMs, the full memory + state + knowledge design raised mean mIoU from 0.15 to 0.32--0.34 (a relative increase of 110.7%--128.0%; bootstrap 95% CIs exclude zero for all three LLMs). Improvements were observed for both regular tunnels (0.29 to 0.50--0.54; +70.0%--83.8%) and complex tunnels (0.04 to 0.18--0.19; +323.1%--361.7%).

### 3. Methodology/Appendix: four stage agents vs five API calls

**Original sentences / table text**

> Each of the four stage agents is a self-contained unit comprising (i) a context (Section 3.3.1; Fig. 7a) and (ii) a Python analyst that constructs the full LLM prompt and parses the returned JSON.

> LLM API calls per tunnel 5 (one per pipeline stage)

**Issue**

The manuscript describes four stage agents (unfolding, denoising, enhancing, and segmenting), but Appendix F reports five API calls per tunnel. If a separate detecting agent is used, it should be introduced consistently in the architecture and stage descriptions. If not, the API-call count should be corrected.

**Suggested revision if there are four LLM calls**

> LLM API calls per tunnel: 4 (one per adapted stage).

**Suggested revision if there are five LLM calls**

> Each of the five stage agents is a self-contained unit comprising (i) a context (Section 3.3.1; Fig. 7a) and (ii) a Python analyst that constructs the full LLM prompt and parses the returned JSON.

Then update the architecture text and figures to include the additional detecting stage.

### 4. CoT validation description conflicts with "strictly bounded" adaptation

**Original sentences**

> These updates are strictly bounded by the empirical ranges defined in the Knowledge component.

> If a constraint is violated, the value is clipped to the nearest valid bound. This check-and-correction is executed as an LLM reasoning step condition, rather than a deterministic software routine, and therefore does not guarantee constraint satisfaction.

**Issue**

"Strictly bounded" and "clipped" imply deterministic enforcement, but the following sentence says constraint satisfaction is not guaranteed. This is an internal inconsistency.

**Suggested revision**

> The agent is instructed to keep proposed updates within the empirical ranges defined in the Knowledge component.

> If a constraint is violated, the agent is instructed to revise the value toward the nearest valid bound. Because this validation is implemented as a prompted reasoning step rather than a deterministic software routine, it reduces but does not guarantee constraint violations.

### 5. Table 3 citation appears inconsistent with the listed models

**Original sentence**

> All three LLMs were accessed via their respective commercial APIs (Table 3) [50].

**Issue**

Reference [50] is listed as an OpenAI GPT-4o system card, but Table 3 lists Opus-4.6, GPT-5.4, and Gemini-3-Flash. The citation does not appear to support all three APIs or the stated model versions.

**Suggested revision**

> All three LLMs were accessed via their respective commercial APIs using the configuration in Table 3.

Add separate vendor/model references only if they are available and accurate.

## Overstatements To Soften

### 6. Introduction: broad claim about existing tunnel-segmentation approaches

**Original sentence**

> Existing tunnel-segmentation approaches do not jointly provide strong adaptability to new conditions and an auditable adaptation process.

**Issue**

"Do not jointly provide" is defensible but broad. "Strong adaptability" is subjective.

**Suggested revision**

> Existing tunnel-segmentation approaches have not yet demonstrated, in a single workflow, both reliable adaptation to changed tunnel conditions and an auditable process for parameter adjustment.

### 7. Related work: feature-engineering and deep-learning comparison is too categorical

**Original sentence**

> Feature-engineering rules are interpretable but lack robustness under changed conditions.

**Issue**

This reads as a blanket statement. Some rule-based systems can be robust within their design envelope.

**Suggested revision**

> Feature-engineering rules are interpretable, but their robustness often depends on how closely new tunnel conditions match the assumptions encoded during design.

### 8. Related work: deep-learning auditability claim is too broad

**Original sentence**

> Supervised deep-learning models generalise through data but require large labelled datasets and periodic retraining, while lacking auditability.

**Issue**

"Lacking auditability" is too absolute. Some interpretability methods exist, even if they do not provide engineer-readable parameter logic.

**Suggested revision**

> Supervised deep-learning models can generalise through data, but they often require large labelled datasets, may require retraining under domain shift, and provide limited engineer-readable audit trails for individual segmentation decisions.

### 9. Related work: LLM self-verification claim may overstate current capabilities

**Original sentence**

> Reasoning-oriented training [36, 37, 38, 39, 40, 41] produces models capable of decomposing complex problems and self-verifying intermediate steps.

**Issue**

"Self-verifying" can overclaim reliability. The paper later acknowledges that LLM validation does not guarantee constraint satisfaction.

**Suggested revision**

> Reasoning-oriented training [36, 37, 38, 39, 40, 41] has produced models that can decompose complex problems and perform prompted consistency checks over intermediate steps.

### 10. Cross-model consistency: "objective tunnel conditions" is too strong

**Original sentence**

> To assess whether the adaptation is model-dependent or driven by objective tunnel conditions, Table 6 compares the three LLMs on the full m+s+k condition.

**Issue**

The experiment can show similar model behaviour under the same inputs, but it cannot prove that adaptation is "driven by objective tunnel conditions" rather than prompt design, knowledge documents, or shared model priors.

**Suggested revision**

> To assess the extent to which adaptation patterns are consistent across models under the same tunnel inputs and prompt structure, Table 6 compares the three LLMs on the full m+s+k condition.

### 11. Discussion: "context design is shown to help" is stronger than necessary

**Original sentence**

> First, as noted in the ablation study (Section 4.2), the context design is shown to help adaptation across models and tunnel categories, with state providing the main gains (+0.103 to +0.176 mIoU on top of memory).

**Issue**

"Is shown to" is assertive. A more formal phrasing should tie the claim directly to the observed results.

**Suggested revision**

> First, the ablation study indicates that structured context improves adaptation across models and tunnel categories, with state providing the largest incremental gains (+0.103 to +0.176 mIoU on top of memory).

### 12. Discussion: explanation of static rules may overstate manual effort

**Original sentence**

> One plausible explanation is that the gain associated with state depends on how the model maps raw numeric summaries (percentiles, counts, ratios) to parameter values in light of parameter semantics.

**Issue**

This sentence is fine, but the following sentence becomes somewhat speculative.

**Original follow-up sentence**

> We did not test alternative mapping strategies (e.g., regression models); however, the continuous, multidimensional nature of the characteristic space and the non-trivial parameter interactions suggest that capturing this mapping through static rules alone would require considerable manual engineering effort per tunnel category.

**Suggested revision**

> We did not test alternative mapping strategies (e.g., regression models); however, the continuous, multidimensional characteristic space and interacting parameters suggest that equivalent static-rule coverage would likely require substantial manual design and validation for each tunnel category.

### 13. Conclusion: practice implication is promotional

**Original sentence**

> For construction practice, R4Tun reallocates expert effort from per-tunnel intervention to a one-off authoring step (reference calibration plus per-stage knowledge documents), makes each parameter change auditable via a logged rationale, and runs via affordable APIs without retraining on labelled domain data.

**Issue**

"Reallocates expert effort" and "affordable APIs" sound promotional. The cost evidence is indicative, not a general affordability claim.

**Suggested revision**

> For construction practice, R4Tun may shift part of the expert effort from repeated per-tunnel retuning to reference calibration and per-stage knowledge authoring, while logging a rationale for each parameter change and avoiding retraining on labelled domain data. The indicative API costs reported here suggest that this adaptation step can be run at low per-tunnel inference cost under the tested settings.

## AI-Like Or Awkward Phrasing

### 14. Abstract: "handle varying tunnel conditions" is vague

**Original sentence**

> These results suggest that LLM-guided adaptation with structured context can help a fixed segmentation pipeline handle varying tunnel conditions and reduce per-tunnel expert re-tuning, particularly for tunnels close to the reference configuration.

**Issue**

"Handle varying tunnel conditions" is vague and slightly promotional.

**Suggested revision**

> These results suggest that LLM-guided adaptation with structured context can improve the transfer of a fixed segmentation pipeline across changed tunnel conditions and reduce the need for per-tunnel expert retuning, particularly when target tunnels remain close to the reference configuration.

### 15. Discussion: awkward and internally mixed sentence

**Original sentence**

> Near the reference, the LLM has sufficient contextual information to reason from, and its advantage over deterministic lookup becomes visible; far from the reference (the complex-category), neither approach has a matching anchor, and the two converge toward similarly low absolute performance, and the LLMs still score higher.

**Issue**

The sentence is long, repetitive, and ends with a loosely attached clause. It also partly contradicts itself by saying the methods converge while the LLMs still score higher.

**Suggested revision**

> Near the reference configuration, the LLM has sufficient contextual information to produce parameter changes beyond deterministic lookup. For complex tunnels, neither the rule-based nor LLM-guided method has a closely matched reference anchor; both therefore remain at low absolute mIoU, although the LLM-guided runs still achieve higher mean scores.

### 16. Methodology: "actual geometric reality" is informal/overstated

**Original sentence**

> If signals conflict, the agent is instructed to prioritise State over Memory, as State reflects the actual geometric reality following upstream processing.

**Issue**

"Actual geometric reality" sounds informal and absolute.

**Suggested revision**

> If signals conflict, the agent is instructed to prioritise State over Memory, because State summarises the observed point-cloud distribution after upstream processing.

### 17. Agent design: "explicit rationale" may be too strong for LLM-generated text

**Original sentence**

> We therefore introduce an LLM-based reasoning framework that retunes parameter-sensitive pipelines per tunnel using expert-guided context and explicit rationale.

**Issue**

"Explicit rationale" can imply verified causal reasoning. "Generated rationale" or "logged rationale" is more precise.

**Suggested revision**

> We therefore introduce an LLM-based framework that retunes parameter-sensitive pipelines per tunnel using expert-guided context and logged, engineer-reviewable rationales.

### 18. Discussion: "single-reference design" explanation should be cleaner

**Original sentence**

> This concentration is due to the single-reference design: both comparators start from a single configuration tuned on a tunnel that is similar to the regular-category.

**Issue**

"Is due to" states causality more strongly than the evidence supports.

**Suggested revision**

> This concentration is consistent with the single-reference design: both comparators start from a single configuration tuned on a tunnel similar to the regular category.

### 19. Conclusion: "recovering progressively" is vague

**Original sentence**

> Improvements were observed for both regular tunnels (0.29 to 0.50--0.54; +70.0%--83.8%) and complex tunnels (0.04 to 0.18--0.19; +323.1%--361.7%).

**Issue**

If retaining the original "recovering progressively" phrase elsewhere, consider replacing it because it is vague. The numerical statement above is clearer.

**Suggested revision**

> Improvements were observed in both tunnel categories, although absolute mIoU remained substantially lower for complex tunnels than for regular tunnels.

## Grammar, Typos, And Formal Style

### 20. Figure 7 typo

**Original text**

> JSON ouput

**Suggested revision**

> JSON output

### 21. Missing space before algorithm reference

**Original sentence**

> The agent follows a strict five-step protocol(Algorithm 1).

**Suggested revision**

> The agent follows a strict five-step protocol (Algorithm 1).

### 22. Subject-verb agreement

**Original sentence**

> The mean increment are narrow, but the per-tunnel direction is nevertheless positive.

**Suggested revision**

> The mean increments are small, but the per-tunnel direction is nevertheless positive.

### 23. Table 3 grammar

**Original table text**

> Failure handling JSON extraction fail → log & error

**Suggested revision**

> Failure handling: JSON extraction failure -> log error and stop run

### 24. Figure 6 caption grammar

**Original sentence**

> Figure 6: Stage 4-Segmenting: Hough-transform line detection [49] to identify ring boundaries, constructs template-based prompts, and feeds them to SAM [11], which produces 2D segment masks that are reprojected into 3D.

**Issue**

The caption lacks a grammatical subject for "constructs" and "feeds".

**Suggested revision**

> Figure 6: Stage 4-Segmenting. Hough-transform line detection [49] identifies ring boundaries, after which template-based prompts are constructed and passed to SAM [11]. SAM produces 2D segment masks that are reprojected into 3D.

### 25. Section heading: "Non-LLM rules-based" should be "rule-based"

**Original heading**

> C. Non-LLM rules-based pseudocode

**Suggested revision**

> C. Non-LLM rule-based pseudocode

### 26. Unit formatting

**Original text**

> density 2,466 pts/m3

**Suggested revision**

> density 2,466 pts/m³

If the journal style discourages superscript units in body text, use:

> density 2,466 points m^-3

### 27. Percentage spacing is inconsistent

**Original examples**

> bootstrap 95 % CIs

> 60 % of the LLM gain

**Issue**

The manuscript uses both "95 %" and "110.7%" styles. Use one style consistently. Automation in Construction generally accepts compact percentage formatting in running text.

**Suggested revision**

> bootstrap 95% CIs

> 60% of the LLM gain

## Repetition To Reduce

### 28. Repeated "reference configuration" in the conclusion

**Original sentences**

> Reliable segmentation of segmental tunnel linings from point clouds remains challenging when tunnel conditions diverge from the expert-tuned reference. This paper presented R4Tun, a multi-agent framework that extends a fixed segmentation pipeline with LLM-guided parameter adaptation. Rather than replacing the underlying geometric operators, R4Tun adapts their parameters by comparing each new tunnel with the reference configuration and by using compact summaries of intermediate pipeline outputs.

**Issue**

"Reference" appears repeatedly in a short span. The meaning is clear but the wording feels repetitive.

**Suggested revision**

> Reliable segmentation of segmental tunnel linings from point clouds remains challenging when tunnel conditions diverge from those used for expert tuning. This paper presented R4Tun, a multi-agent framework that extends a fixed segmentation pipeline with LLM-guided parameter adaptation. Rather than replacing the underlying geometric operators, R4Tun adapts their parameters using comparisons with the calibrated baseline and compact summaries of intermediate pipeline outputs.

### 29. Repeated "LLM-guided adaptation" in abstract/introduction

**Original sentence**

> These results suggest that LLM-guided adaptation with structured context can help a fixed segmentation pipeline handle varying tunnel conditions and reduce per-tunnel expert re-tuning, particularly for tunnels close to the reference configuration.

**Issue**

The phrase "LLM-guided adaptation" is already used frequently in the abstract. A more specific phrase avoids repetition.

**Suggested revision**

> These results suggest that structured, model-assisted parameter adaptation can improve transfer of a fixed segmentation pipeline across changed tunnel conditions and reduce per-tunnel expert retuning, particularly for tunnels close to the reference configuration.

## Additional Consistency Checks Before Resubmission

1. Verify whether the system has four or five LLM-adapted stages, then update the architecture figure, Section 3.2, Section 3.3, Table 15, and the API-call total consistently.
2. Use one set of relative improvement values throughout the highlights, abstract, Section 4.1, and conclusion.
3. Check all model names and references. If Opus-4.6, GPT-5.4, and Gemini-3-Flash are experimental/vendor-specific names, ensure the reference list supports them or remove unsupported citations.
4. Keep the Non-LLM control phrasing cautious: the current evidence supports "consistent with an additional LLM contribution", not proof that reasoning alone caused the gain.
5. Consider renumbering Section 3.5.1 and 3.5.2 if they are intended to be general evaluation sections rather than subsections of "Non-LLM adaptation".
