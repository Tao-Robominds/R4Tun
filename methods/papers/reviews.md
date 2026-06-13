Dear Dr Wang,

Thank you for submitting your manuscript to Automation in Construction. I regret to inform you that our reviewers have advised against publishing your manuscript, and we must therefore reject it. Please refer to the comments listed at the end of this letter for details of why we reached this decision.

Thank you for giving us the opportunity to consider your work. 

Kind regards, 

Daniel Castro-Lacouture, Ph.D  
Editor-in-Chief  
Automation in Construction 

Editor and Reviewer comments:  


Reviewer's Responses to Questions

Note: In order to effectively convey your recommendations for improvement to the author(s), and help editors make well-informed and efficient decisions, we ask you to answer the following specific questions about the manuscript and provide additional suggestions where appropriate.

1. Are the objectives and the rationale of the study clearly stated?

Please provide suggestions to the author(s) on how to improve the clarity of the objectives and rationale of the study. Please number each suggestion so that author(s) can more easily respond.

Reviewer #1: The study's objectives and rationale are generally clear but could be more sharply focused. The introduction and abstract adequately outline the need for adaptive tunnel inspection and the limitations of current methods, yet the specific research goals are somewhat dispersed. Please consider consolidating the objectives into a distinct subsection in the Introduction, and explicitly listing the aims of R4Tun more clearly.

Reviewer #2: No

Reviewer #3: The study’s basic research objectives and rationale are presented, but they are not articulated with sufficient clarity and specificity to highlight the novelty of the work. While the goal of developing the SAM4Tun framework for tunnel point cloud segmentation is mentioned, the rationale for designing an inference-based multi-agent framework integrated with engineer-designed prompt-driven pipelines—especially the unique research gaps it addresses in tunnel detection with large language models (LLMs)—is not fully elaborated. This lack of in-depth articulation weakens the clarity of the study’s core objectives and the motivation behind its design.

2. If applicable, is the application/theory/method/study reported in sufficient detail to allow for its replicability and/or reproducibility?

Please provide suggestions to the author(s) on how to improve the replicability/reproducibility of their study. Please number each suggestion so that the author(s) can more easily respond.

Reviewer #1: Mark as appropriate with an X:
Yes [] No [X] N/A []
Provide further comments here:

The study is largely replicable due to the open-source code and dataset, and the method is described with reasonable detail. However, some implementation specifics are omitted.
The prompt templates or system instructions applied in the method should be given in the supplementary material, which support a small, fully worked example of a CoT trace for a stage agent and the reflective agent.

Reviewer #2: Mark as appropriate with an X:
Yes [] No [X] N/A []
Provide further comments here:

Reviewer #3: Mark as appropriate with an X:
Yes [] No [X] N/A []
Provide further comments here:
No, the methodological details are insufficient to ensure full replicability and reproducibility. Critical details are missing, including the specific implementation logic of how inference agents delegate parameter tuning, the unreported pipeline parameter selection process of SAM4Tun (which should be demonstrated via Chain-of-Thought (CoT)), and the lack of ablation analyses to verify the contribution of core modules (e.g., multi-agent collaboration, LLM-driven prompt tuning). Additionally, the limited LLM comparison and ambiguous performance descriptions further reduce the transparency of the method, making it difficult for other researchers to replicate the study’s experiments and results.

3. If applicable, are statistical analyses, controls, sampling mechanism, and statistical reporting (e.g., P-values, CIs, effect sizes) appropriate and well described?

Please clearly indicate if the manuscript requires additional peer review by a statistician. Kindly provide suggestions to the author(s) on how to improve the statistical analyses, controls, sampling mechanism, or statistical reporting. Please number each suggestion so that the author(s) can more easily respond.

Reviewer #1: Due to the small sample size, increase the sample size if possible, or employ cross-validation to strengthen generalizability.

Reviewer #2: No

Reviewer #3: Statistical analyses and reporting are incomplete and lack rigor. While basic performance metrics (e.g., mIoU) are reported, the study does not include key statistical indicators such as P-values, confidence intervals (CIs), or effect sizes to validate the significance of performance improvements—especially for the mIoU change in off-reference cases. The experimental controls are also inadequate: the LLM comparison is limited to only two models with no unified control of variables (e.g., prompt templates, reasoning steps), and the sampling mechanism for the Seg2Tunnel dataset (e.g., stratification by tunnel scenario, point cloud characteristics) is not clearly described, undermining the credibility of the statistical results.

4. Could the manuscript benefit from additional tables or figures, or from improving or removing (some of the) existing ones?

Please provide specific suggestions for improvements, removals, or additions of figures or tables. Please number each suggestion so that author(s) can more easily respond.

Reviewer #1: Consider adding a comprehensive flow to make the agent's operation more immediately understandable.

Reviewer #2: No

Reviewer #3: The manuscript would benefit significantly from adding key figures/tables and optimizing the placement and presentation of existing ones. First, additional visualizations are needed: CoT-based figures demonstrating SAM4Tun’s pipeline parameter selection (and comparison with R4Tun), ablation curves for core hyperparameters (e.g., number of inference agents, LLM reasoning steps), and a comparative table of multiple representative LLMs (including accuracy, reasoning efficiency, and robustness metrics). Second, the existing charts suffer from inappropriate placement that disrupts logical flow; these need to be reorganized to follow the "figure follows text" principle. No existing figures/tables require removal, but all need structural and contextual optimization for clarity.

5. If applicable, are the interpretation of results and study conclusions supported by the data?

Please provide suggestions (if needed) to the author(s) on how to improve, tone down, or expand the study interpretations/conclusions. Please number each suggestion so that the author(s) can more easily respond.

Reviewer #1: Mark as appropriate with an X:
Yes [X] No [] N/A []
Provide further comments here:
N/A

Reviewer #2: Mark as appropriate with an X:
Yes [] No [X] N/A []
Provide further comments here:

Reviewer #3: Mark as appropriate with an X:
Yes [X] No [] N/A []
Provide further comments here:
The interpretation of results and study conclusions are partially supported by basic experimental data but lack sufficient and rigorous evidence to validate their claims. While the manuscript reports SAM4Tun’s performance on the Seg2Tunnel dataset, the limited LLM comparisons, missing ablation analyses, and ambiguous mIoU performance statements (in off-reference cases) mean the conclusions about the framework’s superiority, adaptability, and ability to prevent baseline collapse are not fully substantiated. Additionally, the lack of interpretable analyses (e.g., CoT parameter comparison) prevents a deep interpretation of why the framework performs well, leading to superficial result interpretation that is not fully aligned with comprehensive data support.

6. Have the authors clearly emphasized the strengths of their study/theory/methods/argument?

Please provide suggestions to the author(s) on how to better emphasize the strengths of their study. Please number each suggestion so that the author(s) can more easily respond.

Reviewer #1: Yes.

Reviewer #2: No

Reviewer #3: No, the authors have not clearly or sufficiently emphasized the study’s strengths. While the SAM4Tun framework integrates innovative design elements (e.g., inference-based multi-agent collaboration, engineer-designed prompt-driven pipelines for tunnel point cloud segmentation), the manuscript fails to explicitly highlight these unique strengths—such as the framework’s ability to address LLM-driven point cloud segmentation challenges in tunnel detection, its advantages in off-reference scenario robustness, or the practical value of its intelligent visualization and parameter tuning. The strengths are only implied rather than systematically articulated and contrasted with state-of-the-art methods, making the study’s contributions unremarkable.

7. Have the authors clearly stated the limitations of their study/theory/methods/argument?

Please list the limitations that the author(s) need to add or emphasize. Please number each limitation so that author(s) can more easily respond.

Reviewer #1: The study uses only five tunnel subsets for evaluation. This small sample size restricts statistical power and the ability to generalize the findings across the full diversity of real-world tunnel conditions.
Besides, the claimed "transparency" and "engineer-guided" nature are not validated with user studies. It remains unproven whether engineers can or will effectively use the logged traces to oversee or correct the system.

Reviewer #2: No

Reviewer #3: The authors have mentioned some general limitations of the study, but the statement is not sufficiently clear, specific, or well-integrated with the research design and experimental results.

8. Does the manuscript structure, flow or writing need improving (e.g., the addition of subheadings, shortening of text, reorganization of sections, or moving details from one section to another)?

Please provide suggestions to the author(s) on how to improve the manuscript structure and flow. Please number each suggestion so that author(s) can more easily respond.

Reviewer #1: Please consider introducing the overall R4Tun workflow (Fig. 2) and agent architecture (Fig. 3) before diving into the detailed CoT explanation. Move the detailed mathematical formulation (Eq. 1) and specific agent operations (Sec. 2.3) to a sub-section after the core concepts are established.

Reviewer #2: Yes

Reviewer #3: Yes, the manuscript’s structure, flow, and writing require substantial improvement. First, the logical flow is disrupted by inappropriate chart placement, which disconnects visual evidence from textual analysis. Second, key sections lack hierarchical subheadings (e.g., for the CoT parameter selection process, ablation analysis, and LLM comparison experiments), leading to disorganized presentation of critical content. Third, there is redundant and ambiguous writing—such as the repeated, semantically unclear statement about mIoU in off-reference cases—that needs to be revised for conciseness and accuracy. Additionally, the discussion section needs reorganization to better link experimental results to conclusions, and methodological details (e.g., agent parameter tuning) should be moved to a dedicated subsection for clarity and readability.

9. Could the manuscript benefit from language editing?

Reviewer #1: Yes

Reviewer #2: Yes

Reviewer #3: Yes

 


Associate Editor: Thank you for considering Automation in Construction for the publication of your manuscript. The reviewers have evaluated your work and based on those reviews, the work unfortunately needs to be rejected at this point. Several strong points are made by the reviewers. Please consider the provided reviewer comments for helping you to improve your work. After significant revision, the authors of course still have the option to resubmit their work. In that case, no track changes need to be included, but the cover letter (1-2 pages) preferably indicates which changes have been made compared to the current version.

In addition to addressing the reviewer comments, the authors could also do the following to improve the article quality:
- The journal uses a consecutive numbered reference style, starting at number [1]. Please review the author guide and try to comply from the first submission.
- Aim to include a dedicated Related Works section in Section 2. Do not limit yourself to add related works only briefly in the Introduction. This is typically insufficient to show the added contributions of the article.
- The article uses quite large page-wide figures. As a result, references to those Figures come several pages earlier. The authors may want to try to use the journal template in 2 columns, and see if they can fit the figures in the columns. In case of too many Figures, a solution might be to reduce the number or figures and/or use the Appendix space to include Figures and further details.
- Section 3 is typically included in the Results section (Section 4).
- The supplementary material can be included directly in the article, after the references, as regular appendices.


Reviewer #2:
The manuscript proposes a reasoning-based multi-agent framework for segmental tunnel analysis in point cloud data. The topic fits the scope of the journal. The idea is interesting. However, the current investigation is not sufficient to support the claimed robustness and methodological contribution. The following specific comments should be addressed before publication.
1. The authors should provide the complete parameter set for each stage, including default values and bounds. The state, knowledge, and memory should be clearly defined in an implementable way. The LLM configuration should also be reported, such as model version, decoding settings, and failure handling.
2. In the current version, the authors only conducted the static comparison under the off-reference subsets. mIoU is improved by R4Tun based on the adaptive parameters, but it does not isolate the contribution of the reasoning enabled LLM acting as a white box controller. Table 2 suggests that state updating provides the largest gain. Therefore the paper needs stronger evidence that the improvement is specifically due to the LLM-based reasoning.
3. The paper defines knowledge as concise parameter definitions and tuning rules based on expert-written tuning guidelines. However, no deterministic execution of these tuning rules is provided as a baseline. It is ambiguous how much of the reported gain comes from the rules themselves or the LLM generated updates. The experiment that applies the same tuning rules without LLM inference is required to support the claimed role of the LLM based agents.
4. mIoU is a relevant but not sufficient factor to assess the performance of the proposed framework. The authors should report IoU of each class and provide stronger evidence on boundary quality near joints and segment edges. The paper should also report runtime, number of calls of LLM and sensitivity to the reflection step, bacause these factors affect practical applications.
5. Most figures are blurred. High-quality figures are required, and figure labels should be readable at column width.


Reviewer #3: No further comment
 



FAQ: How can I reset a forgotten password?
https://service.elsevier.com/app/answers/detail/a_id/28452/supporthub/publishing/
For further assistance, please visit our customer service site: https://service.elsevier.com/app/home/supporthub/publishing/
Here you can search for solutions on a range of topics, find answers to frequently asked questions, and learn more about Editorial Manager via interactive tutorials. You can also talk 24/7 to our customer support team by phone and 24/7 by live chat and email

At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

#AU_AUTCON#

To ensure this email reaches the intended recipient, please do not delete the above code

 

 