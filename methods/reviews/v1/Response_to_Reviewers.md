# Response to Reviewers

**Manuscript:** R4Tun: LLM-guided adaptive segmental tunnel lining segmentation in point clouds
**Authors:** Xinghui Tao, Guangming Wang, Jelena Ninić, Brian Sheil
**Journal:** Automation in Construction

We thank the Associate Editor and reviewers for their feedback. We have responded to each query below and modified the manuscript accordingly. Changes are highlighted in yellow.

The most consequential changes are summarised first:

- The evaluation has been re-run from scratch on **30 Seg2Tunnel subsets** (13 regular + 17 complex), giving **270 adapted pipeline runs** (30 tunnels × 3 LLMs × 3 ablation conditions) plus 30 SAM4Tun baseline runs and 30 non-LLM rule-based runs (Section 3.5; Tables 1, 2).
- All results are reported across **three independent LLMs** (Opus-4.6, GPT-5.4, Gemini-3-Flash) under identical prompts (Section 3.5.2; Table 3). The reflective agent and DeepSeek-R1 results from the previous version have been removed (see Reviewer 2, comment 4).
- A **deterministic non-LLM rule-based adaptation** has been added as a control (Section 3.3; Python pseudocode in Appendix C), so the LLM contribution can be read directly from the gap between the *Non-LLM* and LLM columns of Table 4.
- All paired comparisons now report **p-values, paired Cohen's *d*, and bootstrap 95 % CIs** (Section 3.5.3, Tables 4–6).
- A dedicated **Related Work** section (Section 2) replaces the previous in-introduction discussion, with three sub-sections; references have been converted to consecutive numbered style starting at [1].
- Supplementary material is now included as appendices after the references: A baseline parameters, B characteriser fields, C non-LLM rules pseudocode, D context-components example, E worked CoT trace, F runtime/API/USD cost, G per-class IoU, H performance distribution, I LLM inference repeatability.

---

## Associate Editor

**Comment 1.** *The journal uses a consecutive numbered reference style, starting at number [1]. Please review the author guide and try to comply from the first submission.*

**Response:** All references have been converted to the consecutive numbered style ([1], [2], …) starting from the first citation in the Introduction.

**Comment 2.** *Aim to include a dedicated Related Works section in Section 2. Do not limit yourself to add related works only briefly in the Introduction.*

**Response:** A dedicated **Section 2 "Related work"** has been added, with three sub-sections: 2.1 Feature engineering versus deep learning, 2.2 Foundation-model pipelines, and 2.3 LLM reasoning. The Introduction now states only the gap and contributions and defers the literature discussion to Section 2.

**Comment 3.** *The article uses quite large page-wide figures. As a result, references to those Figures come several pages earlier. The authors may want to try to use the journal template in 2 columns, and see if they can fit the figures in the columns.*

**Response:** The revised manuscript is rebuilt in the two-column `cas-dc` journal template. Figures are placed within one page of their first textual reference; the four pipeline-stage diagrams (Figs. 2–5) and the cross-family bar plot (Fig. 9) remain page-wide so that axis labels stay legible. Supporting parameter and characteriser tables have been moved to Appendices A and B so the main body now contains only the seven tables that directly support the result narrative.

**Comment 4.** *The supplementary material can be included directly in the article, after the references, as regular appendices.*

**Response:** All supplementary material is now placed directly after the references as Appendices A–H (see the executive summary above for the contents of each).

---

## Reviewer 1

**Reviewer's comment 1.** *The study's objectives and rationale are generally clear but could be more sharply focused. … Please consider consolidating the objectives into a distinct subsection in the Introduction, and explicitly listing the aims of R4Tun more clearly.*

**Response:** The Introduction has been restructured around a clear gap → motivation → contributions arc, ending with an explicit numbered list of four contributions at the end of the Introduction: (i) the framework design with three context components, (ii) the cumulative ablation across 30 tunnels, (iii) cross-LLM validation across three independent models, and (iv) the comparison against the deterministic non-LLM control. The wider literature discussion has been moved to the new Section 2.

**Reviewer's comment 2.** *The prompt templates or system instructions applied in the method should be given in the supplementary material, which support a small, fully worked example of a CoT trace for a stage agent and the reflective agent.*

**Response:** A worked example for the denoising agent on Tunnel 4-1 is provided in **Appendices D and E**. Appendix D (Tables 13–14) shows the actual *memory* and *state* delivered to the agent and reproduces the verbatim denoising *knowledge* document; Appendix E gives the agent's full five-step CoT trace and the JSON it returns. The reflective agent has been removed (see Reviewer 2, comment 4), so a separate trace for it is no longer required. Full prompt templates, adapted JSONs, and per-tunnel CoT logs for all 270 adapted runs are released alongside the source code at the GitHub repository cited under "Data availability".

**Reviewer's comment 3.** *Due to the small sample size, increase the sample size if possible, or employ cross-validation to strengthen generalizability.*

**Response:** The evaluation has been expanded six-fold from 5 tunnel subsets to **30 Seg2Tunnel subsets** spanning the published tunnel families: 13 regular (10 staggered + 3 continuous) and 17 complex (large-diameter, 7-segment, off-axis scanning). Selection criteria are described in Section 3.5.1 and Table 1. Every paired comparison now reports two-sided paired *t*-test *p*-values (α = 0.05), paired Cohen's *d*, and bootstrap 95 % CIs (1000 resamples) on the mean per-tunnel improvement; see Section 3.5.3 and Tables 4–6. We did not perform *k*-fold cross-validation because R4Tun is a non-trained adaptation procedure (the LLMs are queried per-tunnel under fixed prompts), so the 30 tunnels function as a held-out test set.

**Reviewer's comment 4.** *Consider adding a comprehensive flow to make the agent's operation more immediately understandable.*

**Response:** Two new figures trace the agent's operation end-to-end: **Fig. 6** shows the multi-agent architecture (the four stage agents and the data flowing between them), and **Fig. 7** shows the inside of one stage agent (memory + state + knowledge feeding the LLM analyst) together with the five CoT phases the analyst executes. Both are introduced in Section 3.4 *before* the detailed CoT description.

**Reviewer's comment 6.** *The study uses only five tunnel subsets … This small sample size restricts statistical power and the ability to generalize the findings … the claimed "transparency" and "engineer-guided" nature are not validated with user studies.*

**Response:** Sample size has been increased to 30 tunnels (see comment 3 above). Regarding the user-study point, the reviewer is correct that the auditability of the generated reasoning traces has not been validated through a formal study with practising engineers. We have added the following sentence to Section 5.2 (Limitations): *"The auditability of the generated reasoning traces has not been validated through a formal user study with practising engineers."* A user study is identified as future work in Section 6.

**Reviewer's comment 7.** *Please consider introducing the overall R4Tun workflow (Fig. 2) and agent architecture (Fig. 3) before diving into the detailed CoT explanation. Move the detailed mathematical formulation (Eq. 1) and specific agent operations (Sec. 2.3) to a sub-section after the core concepts are established.*

**Response:** Section 3 is now ordered as the reviewer suggests: 3.1 Task definition, 3.2 SAM4Tun baseline, 3.3 Non-LLM adaptation, 3.4 R4Tun (architecture first via Fig. 6, then 3.4.1 Agent design, 3.4.2 Context design, 3.4.3 CoT design). Equations and detailed mathematical content are deferred to Section 3.5 after the architecture and dataset have been introduced.

**Reviewer's comment 8.** *Language editing.*

**Response:** The whole manuscript has been re-edited for clarity and concision; repeated phrases have been removed and acronyms (CoT, mIoU, CV) are introduced once and reused consistently.

---

## Reviewer 2

**General comment.** *The manuscript proposes a reasoning-based multi-agent framework for segmental tunnel analysis in point cloud data. The topic fits the scope of the journal. The idea is interesting. However, the current investigation is not sufficient to support the claimed robustness and methodological contribution.*

**Response:** The single biggest change is the inclusion of a deterministic non-LLM rule-based adaptation (Section 3.3, Appendix C), which directly addresses comments 2 and 3 about isolating the LLM's contribution.

**Reviewer's comment 1.** *The authors should provide the complete parameter set for each stage, including default values and bounds. The state, knowledge, and memory should be clearly defined in an implementable way. The LLM configuration should also be reported, such as model version, decoding settings, and failure handling.*

**Response:** Three additions address this comment:
- **Complete parameter set with defaults.** Tables 8–11 (Appendix A) list every tunable parameter for the four stages (Unfolding, Denoising, Enhancing, Segmenting) with the SAM4Tun baseline value. The 18 critical parameters that all three LLMs consistently adjust, with their adapted ranges or correction values, are reported in Table 7.
- **Memory, state and knowledge defined implementably.** Section 3.4.2 defines each context component; Table 12 (Appendix B) enumerates every characteriser field by stage; and the worked example in Appendix D shows the actual numeric content of memory and state for one specific tunnel together with the verbatim denoising knowledge document.
- **LLM configuration.** Table 3 reports model versions (Opus-4.6, GPT-5.4, Gemini-3-Flash), max tokens (16,384), temperature (vendor default), timeout (300 s per call), prompt format, and failure handling (JSON-extraction failure → log and error). Section 3.5.2 explicitly notes that no model-specific prompt tuning was performed and that all reported numbers are single-run outcomes under vendor-default API settings.

**Reviewer's comment 2.** *In the current version, the authors only conducted the static comparison under the off-reference subsets. mIoU is improved by R4Tun based on the adaptive parameters, but it does not isolate the contribution of the reasoning-enabled LLM acting as a white-box controller. Table 2 suggests that state updating provides the largest gain. Therefore the paper needs stronger evidence that the improvement is specifically due to the LLM-based reasoning.*

**Response:** To isolate the LLM-specific contribution, R4Tun is now compared against a deterministic Python rule table (Section 3.3; pseudocode in Appendix C) that maps the same characterisation fields the LLMs see (diameter, ring length, segments-per-ring, joint type, point density, station configuration) to the same 18 critical parameters using explicit `if … then …` rules, fixed for all 30 tunnels with no evaluation-set tuning. Two empirical observations follow (Section 4.1, Table 4 *Non-LLM* column; Fig. 9):

- On the **regular family** (n = 13), the rule-based adaptation does **not** improve over the static SAM4Tun baseline (≈ 0.29; *p* = 0.79). The LLM (m+s+k) reaches 0.50–0.54 on the same tunnels , under this experimental setup, the additional gain is consistent with an LLM-specific contribution beyond deterministic rule lookup.
- On the **complex family** (n = 17), the rules recover roughly 60 % of the LLM gain (static 0.04 → rules 0.137 → LLM 0.184). Both adaptations converge at a similarly low absolute mIoU, which we attribute to the single-reference design rather than to LLM reasoning quality (Section 5.1).

We also observed that **state contributes the largest single increment** within the LLM ablation (Table 5: m → m+s adds +0.10 to +0.18 mIoU across the three LLMs, vs. +0.014 to +0.022 for m+s → m+s+k). This is now reported as the first key finding in Section 5.1, and we additionally note that the m+s condition exceeds the non-LLM rule baseline by 0.10–0.12 mIoU overall , meaning the LLM gain over deterministic rules holds even before knowledge text is added.

**Reviewer's comment 3.** *The paper defines knowledge as concise parameter definitions and tuning rules based on expert-written tuning guidelines. However, no deterministic execution of these tuning rules is provided as a baseline. … The experiment that applies the same tuning rules without LLM inference is required to support the claimed role of the LLM-based agents.*

**Response:** This experiment has now been added: the per-stage knowledge documents that the LLM agents read have been transcribed into a deterministic Python lookup, described in Section 3.3 with the denoising-stage pseudocode reproduced in **Appendix C**. Results appear in the *Non-LLM* column of Table 4 and as the dotted bars in Fig. 9, and the corresponding "Baseline → Non-LLM" transition is reported alongside the LLM transitions in Table 5. The gap between the *Non-LLM* and LLM columns of Table 4 is the quantity the reviewer asked us to expose, and it is now the central comparison in Section 4.1.

**Reviewer's comment 4.** *mIoU is a relevant but not sufficient factor to assess the performance of the proposed framework. The authors should report IoU of each class and provide stronger evidence on boundary quality near joints and segment edges. The paper should also report runtime, number of calls of LLM and sensitivity to the reflection step, because these factors affect practical applications.*

**Response:** Three additions address this comment:
- **Per-class IoU.** Tables 16 and 17 (Appendix G) report per-class IoU for regular (6-class) and complex (7-class) tunnels under all conditions; the overall mIoU distribution per condition is summarised in Table 18 (Appendix H).
- **Runtime, API calls, and cost.** Table 15 (Appendix F) reports 5 LLM API calls per tunnel (one per pipeline stage), 150 calls per condition per LLM, ~24 min wall-clock time per tunnel on a single NVIDIA RTX 5060, no retraining required, and labelled data needed only for evaluation. The same table reports per-tunnel input/output token counts and indicative USD cost under m+s+k (Opus-4.6: \$1.47; GPT-5.4 high-effort reasoning: \$0.43; Gemini-3-Flash: \$0.03), with the unit pricing assumptions noted in the table footnote.
- **Reflection step.** The reflective agent has been **removed entirely** in this revision. Its contribution to mIoU was not isolated in the previous draft, and on re-examination its role was more interpretive than empirically demonstrated; the supporting DeepSeek-R1 results have also been removed. We acknowledge this in Section 5.2: *"parameters are adapted in a single forward pass without iterative self-correction, leaving potential gains from closed-loop refinement unexplored."* Closed-loop self-evaluation is flagged as future work in Section 6.

**Reviewer's comment 5.** *Most figures are blurred. High-quality figures are required, and figure labels should be readable at column width.*

**Response:** All figures have been regenerated as vector graphics (PDF).

---

## Reviewer 3

Reviewer 3 indicated *"No further comment"* in the free-text section. The structured questionnaire responses are addressed below.

**Q1 , Objectives and rationale.** *The rationale for designing an inference-based multi-agent framework integrated with engineer-designed prompt-driven pipelines,especially the unique research gaps it addresses in tunnel detection with LLMs,is not fully elaborated.*

**Response:** The closing paragraphs of the Introduction and the new Section 2.3 now state the gap explicitly:

> "In tunnelling segmentation, no prior work has applied LLM-based reasoning to adapt a parameter-sensitive pipeline to new conditions while keeping each decision inspectable. A deterministic rule-based adaptation of the same knowledge documents is included as a Non-LLM control to quantify the LLM-specific contribution." (Section 2.3)

The four numbered contributions at the end of the Introduction make the unique aims explicit: framework design, cumulative ablation across 30 tunnels, cross-LLM validation, and the LLM-vs-rule comparison.

**Q2 , Replicability.** *Critical details are missing, including the specific implementation logic of how inference agents delegate parameter tuning, the unreported pipeline parameter selection process of SAM4Tun (which should be demonstrated via CoT), and the lack of ablation analyses to verify the contribution of core modules.*

**Response:** Each sub-point has been addressed:
- **Agent delegation logic.** Section 3.4 explains that each pipeline stage has its own agent, that each agent runs the same five-step CoT protocol on its own context, and that the output JSON is consumed by the unmodified stage script.
- **CoT-based parameter selection.** Appendix D shows the actual numeric content of memory, state, and knowledge supplied to the denoising agent on Tunnel 4-1; Appendix E reproduces the agent's full five-step trace and the JSON it returns.
- **Ablation analyses.** A cumulative ablation across the LLM-driven conditions (baseline → m → m+s → m+s+k) and the separate non-LLM control is reported across all three LLMs and both tunnel families in Section 4.2 (Table 5, Fig. 9). Each transition is accompanied by *p*-values, Cohen's *d*, and bootstrap 95 % CIs.
- **LLM comparison.** Three independent LLMs are evaluated under identical prompts; the cross-model summary is in Table 6.

**Q3 , Statistical analyses, controls, sampling.** *The study does not include key statistical indicators such as P-values, confidence intervals (CIs), or effect sizes … the LLM comparison is limited to only two models with no unified control of variables … and the sampling mechanism for the Seg2Tunnel dataset is not clearly described.*

**Response:**
- **Statistics.** Every paired comparison now reports two-sided paired *t*-test *p*-values (α = 0.05), paired Cohen's *d*, and bootstrap 95 % CIs (1000 resamples); see Section 3.5.3 for definitions and Tables 4–6 for the per-condition, per-transition, and cross-model summaries. Bootstrap CIs and *p*-values are presented as complementary statistics.
- **Controls.** A deterministic non-LLM rule-based adaptation (Section 3.3, Appendix C) provides the missing control. Pipeline code, evaluation script, prompt structure, and characteriser fields are held constant across the three LLMs and across all conditions; only the context content varies by ablation level.
- **LLM comparison.** Three LLMs are now compared (Opus-4.6, GPT-5.4, Gemini-3-Flash), not two.
- **Sampling.** Section 3.5.1 and Table 1 describe the stratification: 13 regular (5.60 m diameter, 1.2 m ring spacing, 6 segments/ring, 6-class evaluation) versus 17 complex (7.5 m diameter, 1.8 m ring spacing, 7 segments/ring with interleaved key-block layout, off-axis single-station scanning, 7-class evaluation). Ground truth is used for evaluation only.

**Q4 , Tables and figures.** *Additional visualizations are needed: CoT-based figures demonstrating SAM4Tun's pipeline parameter selection (and comparison with R4Tun), ablation curves for core hyperparameters … and a comparative table of multiple representative LLMs.*

**Response:**
- **CoT-based parameter selection.** Fig. 7(b) summarises the five-step CoT protocol; Appendices D and E give the numeric inputs and the worked trace for one stage agent (denoising) on one tunnel (4-1).
- **Multi-LLM comparative table.** Table 6 summarises the three LLMs on accuracy (mean ΔmIoU), uncertainty (95 % CI) and effect size (Cohen's *d*) under m+s+k; Table 4 breaks down results by ablation level; Table 15 reports their reasoning efficiency in tokens and indicative USD cost per tunnel.
- **Ablation curves.** Fig. 9 reports the cumulative ablation step-by-step for both tunnel families and all three LLMs; Table 5 reports the same as a numeric increment table. We have not swept the number of agents or reasoning steps because the architectural choice (one agent per stage, a fixed five-step CoT) is part of the method definition; this is acknowledged in Section 5.2.
- **Figure placement.** All figures now appear within one page of the paragraph that introduces them, in column-width format where possible.

**Q5 , Interpretation and conclusions.** *The conclusions about the framework's superiority, adaptability, and ability to prevent baseline collapse are not fully substantiated.*

**Response:** The conclusions in the revised manuscript are scoped to match the evidence:
- We observed that adding state to memory yields a robust improvement across all three LLMs (Cohen's *d* > 1.3, *p* < 0.0001, 95 % CIs exclude zero) , Section 5.1, first finding.
- We observed that on the regular family the LLM adaptation outperforms the non-LLM rule baseline by roughly 0.21–0.25 mIoU; we describe this as *"consistent with an LLM-related contribution"* rather than overclaim it.
- We removed that R4Tun "prevents baseline collapse" on complex tunnels. Section 5.1 says explicitly: *"On the complex family, the LLM and rule-based adaptations converge at low absolute mIoU, which we attribute to the single-reference design rather than to LLM reasoning quality"*, and Section 5.2 lists this as the primary limitation.
- The Highlights and Abstract have been rewritten to reflect this scoped framing , e.g. *"In its current form, R4Tun is most useful as an automated adaptation tool for tunnels close to the reference configuration"*.

**Q6 , Strengths.** *The strengths are only implied rather than systematically articulated and contrasted with state-of-the-art methods.*

**Response:** The strengths are now stated explicitly: the four numbered contributions at the end of the Introduction; the parameter-sensitivity discussion in Section 2.3; the auditability point in Section 5.1; and the operational summary in Section 6 (*"R4Tun reallocates expert effort from per-tunnel intervention to a one-off authoring step (reference calibration plus per-stage knowledge documents), makes each parameter change auditable via a logged rationale, and runs via affordable APIs without retraining on labelled domain data"*).

**Q7 , Limitations.** *The authors have mentioned some general limitations of the study, but the statement is not sufficiently clear, specific, or well-integrated with the research design and experimental results.*

**Response:** Section 5.2 has been rewritten as a focused paragraph naming four specific limitations and tying each to the experimental result that exposes it:
1. **Pipeline and dataset scope.** *"R4Tun was evaluated exclusively on the SAM4Tun pipeline and the Seg2Tunnel dataset; as a result, its transferability to other point-cloud processing pipelines and datasets requires further testing."*
2. **Single-reference design and complex-family ceiling.** *"All parameter adaptation is anchored to a single expert-tuned configuration, which creates a low ceiling for both the LLM-guided and the rule-based methods. … Expanding R4Tun to encode multiple expert-tuned reference configurations may further narrow the absolute gap."*
3. **No formal user study.** *"The auditability of the generated reasoning traces has not been validated through a formal user study with practising engineers."*
4. **Single-pass adaptation.** *"Parameters are adapted in a single forward pass without iterative self-correction, leaving potential gains from closed-loop refinement unexplored."*

Each limitation is matched by a corresponding future-work item in Section 6.

**Q8 , Manuscript structure and writing.** *The logical flow is disrupted by inappropriate chart placement … key sections lack hierarchical subheadings … redundant and ambiguous writing,such as the repeated, semantically unclear statement about mIoU in off-reference cases,needs to be revised.*

**Response:**
- **Hierarchical subheadings.** Section 3 now uses 3.1–3.5 with deeper sub-subheadings (3.4.1 Agent design, 3.4.2 Context design, 3.4.3 CoT design; 3.5.1 Dataset, 3.5.2 Experimental design, 3.5.3 Evaluation metrics, 3.5.4 Sensitivity analysis). Section 4 mirrors this with 4.1 Overall performance, 4.2 Ablation analysis, 4.3 Cross-model consistency, 4.4 Parameter sensitivity. Section 5 uses 5.1 Key findings and 5.2 Limitations.
- **Figure placement.** All figures now appear within one page of their first textual reference.
- **Ambiguous mIoU statements.** Every numeric performance claim now reports the absolute baseline value, the absolute adapted value, the paired Δ, the *p*-value or Cohen's *d*, and the tunnel family on which the comparison is made , for example *"Complex tunnels improve from 0.04 to 0.18–0.19 (d = 1.2–2.5), an absolute increase of 0.14–0.15 mIoU"* (Section 4.1).

**Q9 , Language editing.** *Yes.*

**Response:** The full manuscript has been re-edited for clarity, redundancy, and consistency of terminology.
---

**LLM stochasticity / repeatability.** *Please discuss whether adapted parameters remain stable across repeated API calls, given possible LLM stochasticity.*

**Response:** We added a repeatability protocol (Section 3.5.2; **Appendix I**, Table I). The primary m+s+k parameters (run 1) are compared to a second inference pass (run 2) with temperature set to 0 and run 1 parameters seeded so unchanged stages skip the GPU pipeline. On the first nine held-out complex-tunnel pairs recovered from logged reruns, median identity on the 18 critical parameters was **100%** (mean **88%**); mean **|ΔmIoU|** was **0.029**, below the paired adaptation gain (ΔmIoU ≈ 0.17–0.19 vs baseline). Scripts and logs: `methods/papers/scripts/run_repeatability.py`, `logs/{tunnel}/repeatability/`. The full 90-pair batch (30 tunnels × 3 LLMs) uses the same protocol.

---

We hope these revisions address all reviewer concerns. The revised manuscript is anchored in 270 adapted runs across 30 tunnels and three LLMs, plus 30 baseline and 30 non-LLM rule-based runs as controls.
