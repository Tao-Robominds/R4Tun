# Stage-wise BO + Intrinsic Reflection Workflow

## Purpose
Primary paper methodology for stage-wise BO, fixed intrinsic proxy validation, threshold-based reflection triggers, and held-out generalisation.

## Scope boundaries
- Include: preprocessing BO, detection/boundary BO, tuning memory, fixed stage proxies, Spearman sanity checks, threshold-trigger validation, generalisation.
- Exclude: full mIoU predictor fitting, Ridge/Platt calibration, leave-one-out metric ablation, LLM routing, RL routing/policy learning, adaptive backtracking, learned correction sequencing.

## Folder map
- `steps/`: paper-scope workflow manuals (00 to 07).
- `output/`: canonical summary stubs aligned with `steps/`; do not use as experiment sandboxes.
- `preparation/`: design-time reverse-engineering history retained for traceability.
- `templates/`: reusable artifact templates.
- `scripts/`: helper scripts (for example, detection parameter dependency graph).

## Step order and dependencies

Per-run runtime artifacts should be written under:
`logs/{run_id}/{tunnel_id}/r{ring_id}/{step_dir}/`

| # | Step | Depends on | Runtime artifact | Canonical output |
|---|------|-----------|------------------|------------------|
| 01 | Stage panel discovery | — | `01_ring_regime_discovery/` | `output/01_ring_regime_discovery_output.md` |
| 02 | Stage-wise BO calibration | 01 | `02_bo_calibration/` | `output/02_bo_calibration_output.md` |
| 03 | Trial dataset + tuning memory | 02 | `03_tuning_memory/` | `output/03_tuning_memory_output.md` |
| 04 | Fixed intrinsic proxies | 02, 03 | `04_intrinsics_and_ontology/` | `output/04_intrinsics_and_ontology_output.md` |
| 05 | Spearman + threshold selection | 04 | `05_proxy_and_calibration/` | `output/05_proxy_and_calibration_output.md` |
| 06 | Reflection trigger validation | 03, 04, 05 | `06_reflection_ablation/` | `output/06_reflection_ablation_output.md` |
| 07 | Generalisation test | 01, 03, 04, 05, 06 | `07_generalisation_test/` | `output/07_generalisation_test_output.md` |

## Fixed proxy strategy
- Preprocessing proxy: `S_depth = S_coverage * S_empty`.
- Detection/boundary proxy: `S_boundary = S_continuity * S_K * S_spacing * S_layout`.
- Validation: Spearman correlation between each combined proxy and final mIoU.
- Deployment trigger: reflect when `S_depth < T_depth` or `S_boundary < T_boundary`.
- Primary trigger metric: bad-case recall for `mIoU < tau`.

## Reflection policy (fixed only)
- low `S_depth` -> rerun preprocessing reflection.
- low `S_boundary` -> rerun boundary/detection reflection.
- Optional sub-reason logging maps low boundary components to continuity, K, spacing, or layout diagnostics.

Ridge, Platt calibration, and leave-one-out ablation are optional appendix analyses only when the fixed proxy fails or reviewers require stronger justification.

## Coding plans
Implementation-level step-by-step coding guidance is maintained in:
`methods/ablation/steps/`
