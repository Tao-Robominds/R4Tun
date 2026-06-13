# R4Tun Pre-Submission Review (Review-4 PDF)

Source reviewed: `R4Tun_Journal_Review-4.pdf`

Companion file: `r4tun_final_manuscript_sentence_review.md` (earlier review; overlapping items are noted but not duplicated below).

Scope: numerical inconsistencies, overstatements, AI-like or promotional phrasing, repetition, terminology and style drift, and grammar. Suggested revisions retain the paper's claims while bringing them closer to formal academic style.

## Highest-Priority Inconsistencies

### 1. Same overall mIoU improvement reported with two different percentage ranges

**Original sentence (Conclusions, Section 6)**

> Evaluated on 30 subsets spanning 13 regular and 17 complex tunnels with three independent LLMs, the full memory + state + knowledge design raised mean mIoU from 0.15 to 0.32–0.34 (a relative increase of 113.3%–126.7%; bootstrap 95 % CIs exclude zero for all three LLMs).

**Issue**

The Highlights, Abstract, and Section 4.1 report a relative mIoU increase of 110.7%–128.0%, computed from the unrounded means (0.342/0.150 and 0.316/0.150). The Conclusions recompute the same quantity from the rounded display values (0.32–0.34 over 0.15) and obtain 113.3%–126.7%. The two ranges describe the same result and should be identical.

**Suggested revision**

> Evaluated on 30 subsets spanning 13 regular and 17 complex tunnels with three independent LLMs, the full memory + state + knowledge design raised mean mIoU from 0.15 to 0.32–0.34 (a relative increase of 110.7%–128.0%; bootstrap 95% CIs exclude zero for all three LLMs).

### 2. Regular-tunnel relative increase quoted as both 70.0%–83.8% and +72.4%–86.2%

**Original sentence (Conclusions, Section 6)**

> regular: 0.29 → 0.50–0.54, +72.4%–86.2%

**Issue**

Section 4.1 reports `70.0%–83.8%` for regular tunnels (computed against the unrounded baseline 0.291). The Conclusions recompute the same quantity against the rounded baseline 0.29 and obtain `+72.4%–86.2%`. The two ranges describe the same result.

**Suggested revision**

> regular: 0.29 → 0.50–0.54 (+70.0%–83.8%)

### 3. Complex-tunnel relative increase quoted as both 323.1%–361.7% and +350.0%–375.0%

**Original sentence (Conclusions, Section 6)**

> complex: 0.04 → 0.18–0.19, +350.0%–375.0%

**Issue**

The Highlights, Abstract, and Section 4.1 report a relative complex-tunnel mIoU increase of 323.1%–361.7% (computed from the unrounded baseline 0.042). The Conclusions recompute the same quantity from the rounded values 0.04 and 0.18–0.19 and obtain +350.0%–375.0%. The two ranges describe the same result.

**Suggested revision**

> complex: 0.04 → 0.18–0.19 (+323.1%–361.7%)

### 4. Complex-tunnel effect-size range easily confused with the overall figure

**Original sentence (Section 4.1)**

> Complex tunnels, where the baseline drops to mIoU = 0.04 because several pipeline assumptions no longer match the tunnel conditions, improve to 0.18–0.19 (a relative increase of 323.1%–361.7%; 𝑑 = 1.2–2.5), an absolute increase of 0.14–0.15 mIoU.

**Issue**

The complex-tunnel Cohen's d range (1.2–2.5) is reported alongside the overall m+s+k effect size of 1.51–1.95 (Table 4 and Abstract). Without an explicit subset label, readers can confuse the two ranges and conclude that the m+s+k effect size is larger than reported.

**Suggested revision**

> Complex tunnels, whose baseline drops to mIoU = 0.04 because several pipeline assumptions no longer match the tunnel conditions, improve to 0.18–0.19 (a relative increase of 323.1%–361.7%; paired Cohen's 𝑑 = 1.2–2.5 across the three LLMs on this subset), an absolute increase of 0.14–0.15 mIoU.

### 5. Reference-tunnel diameter appears as 5.5 m, 5.60 m, and 5.32 m

**Original sentences**

> All parameters were expert-tuned on a reference tunnel (diameter 5.60 m, density 2,466 pts/m3, mIoU = 0.531) (Section 3.2)

> Estimated diameter (m) | 5.32 | 7.41 | +39% (Appendix D, Table 13)

> inner diameter increases by 34% (7.5 m) (Section 3.4.1)

**Issue**

Three different diameters denote the same reference tunnel: 5.60 m (nominal), 5.5 m (Table 1 and abstract), and 5.32 m (estimated diameter in the Memory excerpt). The +34% and +39% increases for tunnel 4-1 use different bases (nominal 5.60 → 7.50 = +34%; estimated 5.32 → 7.41 = +39%). The distinction is never explicitly defined for the reader.

**Suggested revision**

> Throughout, "nominal diameter" refers to the design value (5.60 m for the regular reference and 7.50 m for the complex category), while "estimated diameter" refers to the value recovered from the unfolding stage on the point cloud (e.g., 5.32 m for the reference tunnel and 7.41 m for tunnel 4-1). The +34% and +39% diameter increases reported for the complex category therefore use the nominal and estimated bases, respectively.

### 6. Four stage agents but five LLM API calls per tunnel

**Original sentences**

> Each of the four stages (unfolding, denoising, enhancing, and segmenting) has a dedicated LLM agent that adapts its parameters. (Section 3.2)

> Each of the four stage agents is a self-contained unit comprising (i) a context (Section 3.3.1; Fig. 7a) and (ii) a Python analyst that constructs the full LLM prompt and parses the returned JSON. (Section 3.3)

> LLM API calls per tunnel | 5 (one per pipeline stage)
> Total API calls (full ablation) | 150 per condition per LLM (Appendix F, Table 15)

**Issue**

The architecture description specifies four LLM-adapted stages, but Appendix F reports five LLM API calls per tunnel (and a corresponding total of 150 = 30 × 5 calls per condition). Either there is an additional adapted stage (for example, detecting) that is not introduced consistently, or Table 15 should report four calls per tunnel and 120 calls per condition.

**Suggested revision (if four LLM-adapted stages)**

> LLM API calls per tunnel | 4 (one per adapted stage)
> Total API calls (full ablation) | 120 per condition per LLM

**Suggested revision (if five LLM-adapted stages)**

> Each of the five stages (unfolding, denoising, enhancing, detecting, and segmenting) has a dedicated LLM agent that adapts its parameters. The architecture figure and the agent-design section have been updated accordingly.

### 7. Pipeline parameter count: "approximately 80" vs "81"

**Original sentence (Section 3.2)**

> The pipeline contains 81 tuneable processing parameters, spanning geometric defaults, filtering and interpolation settings, and boundary-detection thresholds.

**Issue**

The TeX source (`r4tun_review_v4.tex`, line 186) reads "approximately 80 tuneable processing parameters", whereas the submitted PDF reads "81". Adopt the precise count and ensure that Tables 8–11 sum to that value.

**Suggested revision**

> The pipeline contains 81 tuneable processing parameters (Tables 8–11), spanning geometric defaults, filtering and interpolation settings, and boundary-detection thresholds.

### 8. "Default" temperature versus temperature 0

**Original sentence (Section 3.5)**

> We used each API's default configuration (Table 3).

**Original entry (Table 3)**

> Temperature | 0 (minimal stochasticity)

**Issue**

Setting temperature to 0 is an explicit override of the API default for all three vendors. The two statements contradict each other.

**Suggested revision**

> Other than setting temperature to 0 to minimise stochasticity, all API parameters were left at their vendor defaults (Table 3).

### 9. Abstract: mIoU and OA improvement phrasing is grammatically incomplete

**Original sentence (Abstract)**

> Evaluated on 30 selected Seg2Tunnel subsets (13 regular, 17 complex) across three LLMs, this design increased mean Intersection-over-Union (mIoU) by at least 110.7% and 26.8% Overall Accuracy (OA).

**Issue**

The conjunction misparses as "increased mIoU by … 26.8% Overall Accuracy". The 26.8% figure is itself a relative increase in OA, not an absolute OA value. (Item also flagged as #1 in the earlier sentence-level review; included here for completeness because it is one of the highest-priority abstract fixes.)

**Suggested revision**

> Evaluated on 30 selected Seg2Tunnel subsets (13 regular, 17 complex) across three LLMs, the full memory + state + knowledge design increased mean mIoU by 110.7%–128.0% and overall accuracy (OA) by 26.8%–34.1% relative to the static SAM4Tun baseline.

### 10. Capitalisation drift: "Non-LLM" versus "non-LLM"

**Original sentences**

> Comparison against the Non-LLM adaptation. (Section 4.1)

> A non-LLM rule-based adaptation reaches overall mIoU 0.20 … (Section 6)

> The non-LLM baseline (Section 3.5) reads the same per-stage knowledge documents… (Appendix C)

**Issue**

The manuscript alternates between "Non-LLM" and "non-LLM" within and across sections. Adopt one form (preferably "non-LLM" except at the start of a sentence) and apply it throughout.

**Suggested revision**

> Comparison against the non-LLM adaptation.

### 11. Terminology drift: "tunnel family" versus "tunnel category"

**Original sentences**

> The improvement holds across both tunnel families (Fig. 9). (Section 4.1)

> The performance distribution varies markedly across tunnel categories. (Section 4.1)

> baseline corrections … take the same improved value on every tunnel regardless of geometry … cluster by tunnel family (Section 3.5.2)

> tunnel-category adaptation patterns (Section 6)

**Issue**

The same concept is referred to as both "tunnel family" and "tunnel category" within and across sections. Adopt one term throughout. "Tunnel category" appears more often in v4 and is the recommended choice.

**Suggested revision**

> The improvement holds across both tunnel categories (Fig. 9).

### 12. Hyphenation drift: "regular-category" versus "regular category"

**Original sentences**

> The gap is largest on the regular-category … (Section 4.1)

> Both the LLM and rule-based adaptations converge … on the complex-category. (Section 4.1)

> regular tunnel category (various)

**Issue**

The hyphenated forms "regular-category" and "complex-category" are used inconsistently with the unhyphenated "regular category" and "complex category". Use the unhyphenated form when the term is used as a noun, and reserve the hyphenated form for compound modifiers (for example, "regular-category baseline").

**Suggested revision**

> The gap is largest on the regular category, where the non-LLM rule table does not improve beyond the SAM4Tun configuration.

### 13. Inconsistent percentage spacing

**Original sentences**

> bootstrap 95 % CIs exclude zero for all three LLMs (Section 6)

> 60 % of the LLM gain (Section 4.1)

> 110.7%–128.0% (Section 4.1)

**Issue**

The manuscript mixes "95 %" and "110.7%" spacings. Automation in Construction style accepts the compact form. Apply the compact form throughout.

**Suggested revision**

> bootstrap 95% CIs exclude zero for all three LLMs

## Overstatements To Soften

### 14. "is shown to help adaptation across models and tunnel categories"

**Original sentence (Section 5.1)**

> First, as noted in the ablation study (Section 4.2), the context design is shown to help adaptation across models and tunnel categories, with state providing the main gains (+0.103 to +0.176 mIoU on top of memory).

**Issue**

"Is shown to help" is assertive given that the mean Δ for memory alone is small or negative for some models on regular tunnels and that 95% CIs straddle zero for some increments.

**Suggested revision**

> The ablation results suggest that structured context contributes to adaptation across models and tunnel categories, with state providing the largest incremental gain (+0.103 to +0.176 mIoU on top of memory).

### 15. "objective tunnel conditions"

**Original sentence (Section 4.3)**

> To assess whether the adaptation is model-dependent or driven by objective tunnel conditions, Table 6 compares the three LLMs on the full m+s+k condition.

**Issue**

Cross-model agreement is consistent with shared inputs and shared knowledge documents; it does not establish that adaptation is "driven by objective tunnel conditions" rather than by prompt design or shared model priors.

**Suggested revision**

> To assess the extent to which adaptation patterns depend on the choice of LLM under the same tunnel inputs and prompt structure, Table 6 compares the three LLMs on the full m+s+k condition.

### 16. "self-verifying intermediate steps"

**Original sentence (Section 2.3)**

> Reasoning-oriented training [36, 37, 38, 39, 40, 41] produces models capable of decomposing complex problems and self-verifying intermediate steps.

**Issue**

"Self-verifying" overstates current capabilities and conflicts with Section 3.3.2, which acknowledges that the validation step does not guarantee constraint satisfaction.

**Suggested revision**

> Reasoning-oriented training [36, 37, 38, 39, 40, 41] has produced models that can decompose complex problems and apply prompted consistency checks to intermediate steps, although such checks remain probabilistic.

### 17. "strictly bounded" coexists with "does not guarantee constraint satisfaction"

**Original sentences (Section 3.3.2)**

> These updates are strictly bounded by the empirical ranges defined in the Knowledge component.

> If a constraint is violated, the value is clipped to the nearest valid bound. This check-and-correction is executed as an LLM reasoning step condition, rather than a deterministic software routine, and therefore does not guarantee constraint satisfaction.

**Issue**

"Strictly bounded" and "clipped" both imply deterministic enforcement, which the next sentence withdraws. (Item also flagged as #4 in the earlier sentence-level review; included here because the wording remains in the v4 PDF.)

**Suggested revision**

> The agent is instructed to keep proposed values within the empirical ranges defined in the Knowledge component.

> If a value is detected as out of range, the agent is prompted to revise it toward the nearest valid bound. Because validation is implemented as a prompted reasoning step rather than a deterministic post-processing routine, it reduces but does not guarantee out-of-range outputs.

### 18. "actual geometric reality"

**Original sentence (Section 3.3.2)**

> If signals conflict, the agent is instructed to prioritise State over Memory, as State reflects the actual geometric reality following upstream processing.

**Issue**

"Actual geometric reality" is informal and absolute. (Item also flagged as #16 in the earlier sentence-level review; included here because the wording remains in the v4 PDF.)

**Suggested revision**

> If signals conflict, the agent is instructed to prioritise State over Memory, since State summarises the observed point distribution after upstream processing.

### 19. "explicit rationale"

**Original sentence (Section 2.3)**

> We therefore introduce an LLM-based reasoning framework that retunes parameter-sensitive pipelines per tunnel using expert-guided context and explicit rationale.

**Issue**

"Explicit rationale" can imply verified causal reasoning. The framework only logs an LLM-generated textual rationale that an engineer can review.

**Suggested revision**

> We therefore introduce an LLM-based framework that retunes parameter-sensitive pipelines per tunnel using expert-guided context and a logged, engineer-reviewable rationale for each parameter change.

### 20. "diagnostic task in which the model receives context and proposes bounded adjustments"

**Original sentence (Section 1)**

> Given that recent LLMs can follow multi-step instructions over structured inputs, parameter adaptation can be formulated as a diagnostic task in which the model receives context and proposes bounded adjustments.

**Issue**

Combined with the validation caveat in Section 3.3.2, the phrase "bounded adjustments" can be misread as an algorithmic guarantee.

**Suggested revision**

> Given that recent LLMs can follow multi-step instructions over structured inputs, parameter adaptation can be framed as a diagnostic task in which the model receives context and proposes adjustments within prompt-specified empirical ranges.

### 21. "strong adaptability"

**Original sentence (Section 1)**

> Existing tunnel-segmentation approaches do not jointly provide strong adaptability to new conditions and an auditable adaptation process.

**Issue**

"Strong adaptability" is subjective and absolute. (Item also flagged as #6 in the earlier sentence-level review; included here because the wording remains in the v4 PDF.)

**Suggested revision**

> Existing tunnel-segmentation approaches have not yet demonstrated, in a single workflow, both robust adaptation to changed tunnel conditions and an auditable mechanism for adjusting the parameters that control segmentation.

### 22. Promotional language in the conclusion

**Original sentence (Section 6)**

> For construction practice, R4Tun reallocates expert effort from per-tunnel intervention to a one-off authoring step (reference calibration plus per-stage knowledge documents), makes each parameter change auditable via a logged rationale, and runs via affordable APIs without retraining on labelled domain data.

**Issue**

"Reallocates expert effort", "one-off authoring step", and "affordable APIs" sound promotional. The cost figures in Table 15 are indicative, not a generalisable affordability claim, and "one-off" ignores the periodic re-authoring of knowledge documents.

**Suggested revision**

> For construction practice, R4Tun aims to shift expert effort from per-tunnel parameter intervention to an upfront authoring step (reference calibration plus per-stage knowledge documents), with a logged rationale accompanying each parameter change. The indicative API costs reported in Appendix F suggest that this adaptation step can be applied at low per-tunnel inference cost under the tested settings, and without retraining on labelled domain data.

### 23. Internally inconsistent sentence about LLM and rule-based convergence

**Original sentence (Section 5.1)**

> Near the reference, the LLM has sufficient contextual information to reason from, and its advantage over deterministic lookup becomes visible; far from the reference (the complex-category), neither approach has a matching anchor, and the two converge toward similarly low absolute performance, and the LLMs still score higher.

**Issue**

A sentence claiming that two methods converge cannot then add that one of them remains higher. The wording is also long and repetitive. (Item also flagged as #15 in the earlier sentence-level review; included here because the wording remains in the v4 PDF.)

**Suggested revision**

> Near the reference configuration, the agent's context contains enough relevant information to support adjustments beyond a deterministic family-level lookup, and the corresponding mIoU gains over the rule-based control are visible. For the complex category, neither the rule-based nor the LLM-guided method has a closely matched reference anchor; both therefore remain at low absolute mIoU, although the LLM-guided runs still achieve a small advantage in mean score.

### 24. "is therefore consistent with an additional LLM contribution" used inconsistently

**Original sentences**

> Under this experimental setup, the additional gain is therefore consistent with an LLM contribution beyond the deterministic control. (Section 4.1)

> Mean mIoU more than doubled overall (bootstrap 95% CIs), with per-class IoU improving across most structural classes. (Highlight 4)

> Evidence that the observed mIoU increase cannot be reproduced by a deterministic, rule-based adaptation, and is therefore consistent with an additional contribution from the LLM-based adaptation process. (Section 1, contribution 4)

**Issue**

Section 4.1 uses appropriately cautious phrasing ("consistent with an additional LLM contribution"), while the Highlights and the contribution list state the same finding more strongly. Align hedging across the manuscript.

**Suggested revision (Section 1, contribution 4)**

> Evidence that, under the single-reference design and dataset tested here, the observed mIoU increase on regular tunnels is not reproduced by a deterministic, rule-based adaptation derived from the same per-stage knowledge documents, supporting an additional contribution from the LLM-based step.

## AI-Like Or Awkward Phrasing

### 25. "leverage mature vision foundation models"

**Original sentence (Section 2.2)**

> practical tunnel pipelines often project 3D data into 2D representations to leverage mature vision foundation models pre-trained on large-scale image datasets, thereby bypassing the annotation bottleneck.

**Issue**

"Leverage" and "bypassing the annotation bottleneck" are common AI-style framings. Plain wording is preferable.

**Suggested revision**

> practical tunnel pipelines often project 3D data into 2D representations so that they can use mature vision foundation models pre-trained on large-scale image datasets, which avoids the annotation cost of training a 3D model from scratch.

### 26. "no diagnostic feedback when parameters become misspecified"

**Original sentence (Section 3.1; similar wording in Section 1)**

> As noted above, the pipeline provides no diagnostic feedback when parameters become misspecified.

**Issue**

"Diagnostic feedback" is borrowed from medical and AI-tooling vocabulary. The pipeline simply lacks an internal signal indicating which parameters have failed.

**Suggested revision**

> As noted above, the pipeline produces no internal signal indicating which parameters have become misspecified.

### 27. "handle varying tunnel conditions"

**Original sentence (Abstract)**

> These results suggest that LLM-guided adaptation with structured context can help a fixed segmentation pipeline handle varying tunnel conditions and reduce per-tunnel expert re-tuning, particularly for tunnels close to the reference configuration.

**Issue**

"Handle varying tunnel conditions" is vague and slightly promotional. (Item also flagged as #14 in the earlier sentence-level review; included here because the wording remains in the v4 PDF.)

**Suggested revision**

> These results suggest that LLM-guided parameter adaptation supplied with structured context improves transfer of a fixed segmentation pipeline across changed tunnel conditions and reduces the need for per-tunnel expert retuning, particularly when target tunnels remain close to the reference configuration.

### 28. "the pipeline degrades silently"

**Original sentence (Section 2.2)**

> When a new tunnel departs from the reference, these parameters become misspecified and the pipeline degrades silently, without signalling which parameters fail or why.

**Issue**

"Degrades silently" is evocative; the mIoU values themselves are observed. The pipeline lacks an internal flag, not all feedback.

**Suggested revision**

> When a new tunnel departs from the reference, these parameters become misspecified and the pipeline produces lower-quality outputs without an internal signal identifying which parameters are responsible.

### 29. "the agent attempts to adapt parameters but cannot verify…"

**Original sentence (Section 4.2)**

> Without intermediate feedback, the agent attempts to adapt parameters but cannot verify whether adjustments improve or degrade the pipeline outputs.

**Issue**

"The agent attempts to" anthropomorphises the LLM. The empirical observation is that, without state, the m condition is small or negative on average.

**Suggested revision**

> Without intermediate-stage statistics, the memory-only condition lacks the post-stage signals needed to relate parameter changes to pipeline outputs, and its mean Δ is small or negative.

### 30. "sufficient contextual information to reason from"

**Original sentence (Section 5.1)**

> Near the reference, the LLM has sufficient contextual information to reason from, and its advantage over deterministic lookup becomes visible.

**Suggested revision**

> Near the reference configuration, the agent's context contains enough relevant information to support adjustments beyond a deterministic family-level lookup, and the corresponding mIoU gains become visible.

### 31. "complex interleaved key-block arrangement"

**Original sentence (Section 3.4.1)**

> each ring adds a seventh segment with a complex interleaved key-block arrangement.

**Issue**

Using "complex" to describe complex tunnels is circular; "interleaved" alone is informative.

**Suggested revision**

> each ring adds a seventh segment with an interleaved key-block arrangement that does not repeat ring-to-ring.

### 32. "most useful as an automated adaptation tool for tunnels close to the reference configuration"

**Original sentence (Highlights, Section 5.1)**

> In its current form, R4Tun is most useful as an automated adaptation tool for tunnels close to the reference configuration, potentially reducing per-tunnel expert re-tuning.

**Suggested revision**

> Within the tested setup, R4Tun is most effective for tunnels that are close to the reference configuration; in this regime, it can reduce the need for per-tunnel expert retuning.

### 33. "JSON output: the selected parameters and their reasoning trace are packaged…"

**Original sentence (Section 3.3.2, step 5)**

> JSON output: The selected parameters and their reasoning trace are packaged into a single schema-conformant JSON object.

**Issue**

"Packaged into" is informal in this technical context.

**Suggested revision**

> JSON output: the agent emits a single schema-conformant JSON object containing the selected parameter values and the textual reasoning trace.

## Repetition To Reduce

### 34. The triplet "memory, state, and knowledge" is repeated many times

**Original**

The phrase appears in the Highlights (twice), the Abstract (twice), Section 1 (three times), Sections 3.3 and 4 (multiple), and Section 6 (twice).

**Suggested revision**

After the first definition (Abstract), abbreviate to **m+s+k** when referring to the full design and to "structured context" when describing the framework concept.

### 35. "expert-tuned reference", "expert-tuned configuration", and "reference configuration" used interchangeably

**Original sentence (Section 6)**

> Reliable segmentation of segmental tunnel linings from point clouds remains challenging when tunnel conditions diverge from the expert-tuned reference. … by comparing each new tunnel with the reference configuration and by using compact summaries of intermediate pipeline outputs.

**Issue**

Three phrases denote the same object in close proximity.

**Suggested revision**

> Reliable segmentation of segmental tunnel linings from point clouds remains challenging when tunnel conditions diverge from those used to tune the reference pipeline. R4Tun adapts SAM4Tun's parameters by comparing each new tunnel with this reference configuration and by drawing on compact summaries of intermediate pipeline outputs.

### 36. State is described identically in Section 4.2 and Section 5.1

**Original sentence (repeated in Sections 4.2 and 5.1)**

> State provides the agent with explicit quantitative evidence of how each stage has transformed the data: radial percentiles for mask bounds, retention rates for denoising aggressiveness, coverage uniformity for upsampling targets.

**Issue**

The same sentence appears in two places. State the description once and back-reference it.

**Suggested revision (Section 5.1)**

> As reported in Section 4.2, state provides per-stage quantitative summaries that the agent can map to specific parameters. The size of the m → m+s gain (+0.103 to +0.176 mIoU) suggests that these summaries supply information that pre-pipeline statistics alone do not.

### 37. Per-class IoU summary appears almost verbatim in Section 4.1 and Section 6

**Original sentence (repeated)**

> per-class IoU improving broadly across structural classes for regular tunnels and recovering progressively from near-zero baselines for complex tunnels

**Issue**

The same observation is stated twice.

**Suggested revision (Section 6)**

> Per-class IoU breakdowns (Appendix G) confirm that the gains are not concentrated on a single structural class.

### 38. "highly sensitive to its many processing parameters" repeated across Sections 1 and 2

**Original sentence (Section 1)**

> However, the pipeline is highly sensitive to its many processing parameters.

**Issue**

The same content is elaborated in Section 2.2 ("any multi-stage pipeline that chains domain-specific preprocessing with a foundation model inherits it") and again later in Section 2.2 ("without alleviating the need for expert parameter tuning"). Trim Section 1 and rely on Section 2.2.

**Suggested revision (Section 1)**

> However, the pipeline depends on a large number of stage-specific parameters; Section 2.2 discusses the consequences of this dependency for new tunnel conditions.

### 39. "no model retraining or labelled domain data" appears in Sections 5.1 and 6

**Original sentences**

> The framework also supports post-hoc review by logging a rationale alongside each parameter change, and requires no model retraining or labelled domain data. (Section 5.1)

> [R4Tun] runs via affordable APIs without retraining on labelled domain data. (Section 6)

**Issue**

The same advantage is stated twice in close proximity.

**Suggested revision**

Keep the Section 6 occurrence (after the toning revisions in #22 above) and remove the duplicate sentence from Section 5.1.

### 40. "per-tunnel expert re-tuning" appears with multiple spellings

**Original**

The manuscript uses "per-tunnel expert re-tuning", "per-tunnel expert retuning", and "per-tunnel intervention" in close proximity (Highlights, Abstract, Sections 1, 5.1, 6).

**Suggested revision**

Standardise on "per-tunnel expert retuning" (one word, no hyphen) and vary occasionally with "manual reconfiguration" where appropriate.

## Grammar, Typos, And Formal Style

### 41. Subject–verb agreement in Section 4.2

**Original sentence**

> The mean increment are narrow, but the per-tunnel direction is nevertheless positive.

**Suggested revision**

> The mean increments are small, but the per-tunnel direction is nevertheless positive.

### 42. Missing space before Algorithm 1 reference

**Original sentence (Section 3.3.2)**

> The agent follows a strict five-step protocol(Algorithm 1).

**Suggested revision**

> The agent follows a strict five-step protocol (Algorithm 1).

### 43. Figure 7 caption typo

**Original text**

> JSON ouput

**Suggested revision**

> JSON output

### 44. Figure 6 caption is grammatically incomplete

**Original sentence**

> Figure 6: Stage 4-Segmenting: Hough-transform line detection [49] to identify ring boundaries, constructs template-based prompts, and feeds them to SAM [11], which produces 2D segment masks that are reprojected into 3D.

**Issue**

"Hough-transform line detection … constructs … and feeds them" lacks a grammatical subject for the two predicates. (Item also flagged as #24 in the earlier sentence-level review; included here for completeness.)

**Suggested revision**

> Figure 6: Stage 4-Segmenting. A Hough-transform line detector [49] identifies ring boundaries; template-based prompts are then constructed and passed to SAM [11], whose 2D segment masks are reprojected into 3D. Rotation is shown for visualisation purposes only.

### 45. Section heading: "Non-LLM rules-based pseudocode"

**Original heading (Appendix C)**

> C. Non-LLM rules-based pseudocode

**Suggested revision**

> C. Non-LLM rule-based adaptation: pseudocode

### 46. Density unit formatting

**Original text (Section 3.2)**

> density 2,466 pts/m3

**Suggested revision**

> density 2,466 pts/m³

If the journal style discourages superscript units in body text, use:

> density 2,466 points·m⁻³

### 47. Awkward parenthetical in the Abstract

**Original sentence**

> Across 270 (30 tunnels × 3 LLMs × 3 context settings) adapted runs, the LLMs showed similar parameter-adjustment trends and consistently adjusted a shared set of critical parameters.

**Issue**

The parenthetical is embedded inside the count. Move it after "runs".

**Suggested revision**

> Across 270 adapted runs (30 tunnels × 3 LLMs × 3 context settings), the LLMs showed similar parameter-adjustment trends and consistently selected a shared set of critical parameters.

### 48. "Yet they require…" reads informally in Section 2.1

**Original sentence**

> Yet they require large labelled tunnel datasets that remain scarce [9], demand retraining for new tunnel conditions, and provide no mechanism for an engineer to trace or override a segmentation decision [8].

**Suggested revision**

> However, supervised models require large labelled tunnel datasets, which remain scarce [9], retraining when tunnel conditions change, and provide limited mechanisms for an engineer to trace or override a segmentation decision [8].

### 49. Conclusion: stacked prepositional phrases

**Original sentence (Section 6)**

> Second, adding an iterative self-evaluation step based on multiple intrinsic criteria, jointly considering coverage, class balance, and geometric continuity, may capture aspects missed by a single coverage metric.

**Issue**

Two consecutive prepositional phrases ("based on", "jointly considering") make the sentence hard to parse.

**Suggested revision**

> Second, an iterative self-evaluation step that jointly considers coverage, class balance, and geometric continuity may capture aspects missed by a single coverage metric.

### 50. Reference [50] does not match Table 3

**Original sentence (Section 3.5)**

> All three LLMs were accessed via their respective commercial APIs (Table 3) [50].

**Issue**

Reference [50] is "OpenAI, GPT-4o system card", which does not support Opus-4.6 or Gemini-3-Flash. (Item also flagged as #5 in the earlier sentence-level review; included here because the citation remains in the v4 PDF.)

**Suggested revision**

> All three LLMs were accessed via their respective commercial APIs using the configurations in Table 3.

Add separate vendor and model references only if accurate citations are available.

### 51. Source-file artefact: "t dThe pipeline contains…"

**Original sentence (TeX source `r4tun_review_v4.tex`, line 186)**

> t dThe pipeline contains approximately 80 tuneable processing parameters.

**Issue**

The leading `t d` is a stray edit artefact. The PDF body renders as "The pipeline contains 81 tuneable processing parameters", but the TeX source still contains the artefact (and the parameter count to be reconciled with item #7 above).

**Suggested revision (TeX source)**

> The pipeline contains 81 tuneable processing parameters (Tables 8–11), spanning geometric defaults, filtering and interpolation settings, and boundary-detection thresholds.

### 52. Algorithm 1 mathematical formatting

**Original (Section 3.5.2 as extracted from the PDF)**

> wherē 𝑣 = 1 𝑁 ∑𝑁 𝑖=1 𝑣𝑖 and 𝑠 = √ 1 𝑁−1 ∑𝑁 𝑖=1(𝑣𝑖 −̄𝑣) 2

**Issue**

The PDF text extraction shows broken combining marks for `\bar{v}`, fractional layout for `1/N`, and stray spacing in the summation. This is most likely a PDF-to-text artefact, but verify the rendered PDF directly to confirm that the bar accent on `v` and the summation bounds typeset correctly.

**Suggested revision**

> Verify the rendered formula in §3.5.2 to ensure that `\bar{v}` displays a bar accent over `v`, that the summation index `i = 1` to `N` is typeset above and below `Σ`, and that the surrounding spacing matches the inline mathematical style used elsewhere.

## Section-Level Structural Issues

### 53. §3.5 placement and sub-numbering

**Original outline**

> 3.5 Non-LLM adaptation
>   3.5.1 Evaluation metrics
>   3.5.2 Sensitivity analysis

**Issue**

The two child sections describe the evaluation metrics and sensitivity analysis used throughout the experiments, not just for the non-LLM control. Their current placement under 3.5 implies a narrower scope.

**Suggested revision**

> 3.5 Non-LLM adaptation
> 3.6 Evaluation metrics  (former §3.5.1)
> 3.7 Sensitivity analysis  (former §3.5.2)

### 54. Highlights and Section 1 contributions list overlap

**Original**

The Highlights box and the four-item contributions list in Section 1 restate the same headline claims (mean mIoU change, dominant context component, cross-LLM consistency, non-LLM control).

**Issue**

After fixing the numerical inconsistencies (items #1–#4), check that the four bullets in the Highlights and the four numbered contributions in Section 1 use exactly the same wording for the headline claims.

**Suggested revision**

After applying the numerical fixes above, copy the four headline claims verbatim from one location to the other and remove any residual divergence (for example, "more than doubled overall" in the Highlights versus "110.7%–128.0%" in Section 4.1).

### 55. Algorithm 1 caption rendering

**Original (Section 3.3.2)**

> The agent follows a strict five-step protocol(Algorithm 1). A worked example of the full five-step trace is provided in Appendix E.

**Issue**

In the PDF text extraction, the body of Algorithm 1 appears interleaved with surrounding text (lines 527–560) but no caption is visible. Verify in the rendered PDF that "Algorithm 1: LLM-driven Parameter Adaptation via CoT" displays inside the float and is not detached from §3.3.2.

**Suggested revision**

Confirm that Algorithm 1 floats with both its caption and its body adjacent to §3.3.2, and that the algorithm body does not break across columns or pages.

## Suggested Wording For The Abstract (Consolidated)

Combining items #1, #2, #3, #9, #22, #27, #34, #35, #40, and #47, a single tightened abstract paragraph might read:

> Automated inspection of segmental tunnel linings requires segmentation pipelines that adapt to varying point-cloud geometry, yet expert-tuned pipelines often degrade when tunnel conditions depart from the reference setting. This paper presents R4Tun, a large language model (LLM)-driven adaptation framework that extends an expert-designed pipeline (SAM4Tun) with bounded parameter tuning informed by structured context (memory, state, and knowledge; m+s+k). On 30 selected Seg2Tunnel subsets (13 regular, 17 complex) and three LLMs (Opus-4.6, GPT-5.4, Gemini-3-Flash), the m+s+k design increased mean Intersection-over-Union (mIoU) from 0.150 to 0.316–0.342 (+110.7%–128.0%) and overall accuracy (OA) from 0.411 to 0.522–0.552 (+26.8%–34.1%). On regular tunnels mIoU rose from 0.291 to 0.495–0.535 (+70.0%–83.8%); on complex tunnels mIoU rose from 0.042 to 0.178–0.194 (+323.1%–361.7%). Across 270 adapted runs (30 tunnels × 3 LLMs × 3 context settings), the three LLMs followed similar parameter-adjustment trends and consistently selected a shared set of critical parameters. These results suggest that LLM-driven adaptation supplied with structured context can improve transfer of a fixed segmentation pipeline across changed tunnel conditions and reduce the need for per-tunnel expert retuning, particularly when target tunnels remain close to the reference configuration.

## Pre-Resubmission Checklist

1. Reconcile items #1–#4 (mIoU and OA percentage ranges, complex-tunnel effect size) so that the Highlights, Abstract, Section 4, Section 5, and Section 6 use one set of values throughout.
2. Define nominal versus estimated diameter once (item #5) and update the +34% / +39% statements accordingly.
3. Resolve the four-versus-five LLM-adapted stages question (item #6) and propagate the count to Section 3.2, Section 3.3, Figs. 2 and 7, and Table 15.
4. Replace "approximately 80" / "81" with one verified count (item #7); confirm Tables 8–11 sum to that value.
5. Reconcile the temperature wording (item #8).
6. Sweep capitalisation, terminology, and formatting drift (items #10–#13).
7. Replace or remove citation [50] next to Table 3 (item #50).
8. Apply the hedging revisions (items #14, #15, #16, #24) consistently.
9. Compress repetitions (items #34–#40) so that each finding is stated once at full length and back-referenced thereafter.
10. Fix typographical and grammatical items #41–#49 and #51.
11. Renumber Section 3.5.1 and 3.5.2 (item #53).
12. After the numerical fixes, align the Highlights and Section 1 contributions list (item #54) and verify Algorithm 1 rendering (items #52 and #55).
