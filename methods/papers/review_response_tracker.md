# Review Response Tracker

## Objectives & Motivation


| #   | Reviewers     | Feedback                                                             | Response in Revision                                                                                                         | Checked |
| --- | ------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------- |
| 1   | R1, R3, Brian | Objectives/rationale unclear; novelty and strengths underarticulated | Intro rewritten with 3 new contributions (Sec 1, lines 104–109); Highlights section; Abstract opens with adaptation problem. | ✅       |
| 2   | Brian         | State labelled-data scarcity explicitly in problem framing           | Sec 1: "require large labelled datasets" (line 98); Sec 2.1: "large labelled tunnel datasets that remain scarce" (line 119)  | ✅       |


## Literature & Structure


| #   | Reviewers         | Feedback                                                                                                               | Response in Revision                                                                                                                                                       | Checked |
| --- | ----------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 3   | AE                | Add dedicated Related Works section (not just intro)                                                                   | Sec 2 with 3 subsections: Sec 2.1 Feature-eng vs DL, Sec 2.2 Foundation-model pipelines, Sec 2.3 LLM reasoning                                                             | ✅       |
| 4   | R1, R3, AE, Brian | Structure needs reorganisation; LLM background too long; m/s/k defined too late; pipeline should precede agent details | Restructured: Sec 2 (Related work), Sec 3 (Methodology) with Sec 3.2 (Baseline: SAM4Tun) before Sec 3.3 (R4Tun agents); m/s/k defined in Sec 1 (line 102); Sec 2.3 trimmed | ✅       |
| 5   | AE                | Use consecutive numbered reference style starting at [1]                                                               | Using natbib with model1-num-names (numbered style)                                                                                                                        | ✅       |


## Experiments & Analysis


| #   | Reviewers | Feedback                                                                                 | Response in Revision                                                                                                                                    | Checked |
| --- | --------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 6   | R2, R3    | No ablation analysis to verify component contributions                                   | Cumulative 4-level ablation (Table 2), incremental gains per component (Table 5, Sec 4.2), ablation bar chart (Fig 9)                                   | ✅       |
| 7   | R1, R3    | Missing p-values, CIs, effect sizes; small sample size                                   | Paired t-tests, bootstrap 95% CIs, Cohen's d reported throughout; expanded to n=30 tunnels; methodology in Sec 3.4.2; main results in Table 4           | ✅       |
| 8   | R2, R3    | Limited LLM comparison (was 2 models); no unified variable control                       | Expanded to 3 LLMs (Opus-4.6, GPT-5.4, Gemini-3-Flash), identical prompts (Sec 3.4.1, Table 3); cross-model summary (Table 6, Sec 4.3)                  | ✅       |
| 9   | R2        | State gives largest gain—need evidence LLM reasoning (not just rules) drives improvement | Sec 4.2: State is the dominant driver.                                                                                                                  | ✅       |
| 10  | R2        | No deterministic rule execution baseline (rules without LLM)                             | 3.4.2. Experimental design: SAM4Tun baseline = deterministic expert rules; run fixed prarameters on all 30 tunnels                                      | ✅       |
| 11  | R2        | Report complete parameter set with defaults/bounds and LLM config                        | Baseline parameters for all 4 stages in App A (Tables 8–11); LLM config in Table 3 (model, tokens, temp, timeout)                                       | ✅       |
| 12  | R2        | Report per-class IoU, runtime, LLM call count                                            | Per-class IoU in App F (Tables 13–14); practical metrics incl. runtime & API calls in App G (Table 15); performance distribution in App E (Table 12)    | ✅       |
| 13  | R2        | Sensitivity analysis and parameter robustness                                            | CV analysis of 1,350 parameter files (Sec 3.4.3, Eq 2); 18 critical parameters identified (Table 7, Sec 4.4); tunnel-responsive vs baseline corrections | ✅       |


## Figures & Presentation


| #   | Reviewers | Feedback                                                            | Response in Revision                                                                       | Checked |
| --- | --------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| 14  | R1, R3    | Add comprehensive flow diagram, CoT visualisations, ablation curves | Stage figures (Figs 2–5), architecture (Fig 7), agent design (Fig 8), ablation bar (Fig 9) | ✅       |
| 15  | R2        | Figures are blurred and labels unreadable                           | Regenerated all figures as high-quality vector PDFs                                        | ✅       |
| 16  | R3, AE    | Figure placement disrupts flow; use 2-column journal template       | Adopted cas-dc 2-column template; figures placed near first reference                      | ✅       |


## Writing & Terminology


| #   | Reviewers         | Feedback                                                                                         | Response in Revision                                                                                       | Checked                     |
| --- | ----------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | --------------------------- |
| 17  | Brian             | Terminology inconsistent ("structured reasoning layer" vs "single-step reflective agent")        | Consistent "stage agents" + "context (m, s, k)" throughout text and figures                                | ✅                           |
| 18  | Brian             | Figures read as full setup, not examples—mark examples clearly                                   | App C: "condensed excerpt"; App D: "excerpts illustrate"; Sec 3.3 references appendices as sample excerpts | full examples are very long |
| 19  | R1, R2, R3, Brian | Language editing needed; reasoning description too abstract; technical wording needs simplifying | Rewriting; CoT anchoring in plain change to "referencing"(Sec 3.3.3);                                      | in progress                 |


## Limitations


| #   | Reviewers | Feedback                                                        | Response in Revision                                                                                                                        | Checked |
| --- | --------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 20  | R1, R3    | Limitations not specific; no user study for transparency claims | Specific limitations in Sec 5.4: single pipeline/dataset, single reference, no user study, single forward pass; framed as future directions | ✅       |


## Supplementary Material


| #   | Reviewers  | Feedback                                                                      | Response in Revision                                                                                                    | Checked |
| --- | ---------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------- |
| 21  | R1, R2, R3 | Insufficient replicability: missing CoT traces, prompts, implementation logic | CoT worked example (App C, Table), context excerpts (App D, Table), full parameter tables (App A), LLM config (Table 3) | ✅       |
| 22  | AE         | Include supplementary material as appendices, not separate                    | 7 appendices (A–G) in appendices.tex, included via \input{appendices}                                                   | ✅       |


