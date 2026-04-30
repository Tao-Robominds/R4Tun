# Ring-wise BO + Fixed-Rule Reflection Workflow

## Purpose
Primary paper methodology for ring-wise BO, consolidated proxy learning, fixed-rule reflection ablation, and held-out generalisation.

## Scope boundaries
- Include: ring-wise BO, regime-based calibration, tuning memory, pipeline intrinsics + ontology features, ridge+Platt proxy, fixed-rule reflection, ablation, generalisation.
- Exclude: LLM routing, RL routing/policy learning, adaptive backtracking, learned correction sequencing.

## Folder map
- `steps/`: paper-scope workflow manuals (01 to 07).
- `output/`: canonical output stubs aligned with `steps/`.
- `preparation/`: design-time reverse-engineering history retained for traceability.
- `templates/`: reusable artifact templates.
- `scripts/`: helper scripts (for example, detection parameter dependency graph).

## Step order and dependencies

Per-run runtime artifacts should be written under:
`data/{tunnel_id}/workflow/{run_id}/{step_dir}/`

| # | Step | Depends on | Runtime artifact | Canonical output |
|---|------|-----------|------------------|------------------|
| 01 | Ring regime discovery | — | `01_ring_regime_discovery/` | `output/01_ring_regime_discovery_output.md` |
| 02 | BO calibration | 01 | `02_bo_calibration/` | `output/02_bo_calibration_output.md` |
| 03 | Tuning memory | 02 | `03_tuning_memory/` | `output/03_tuning_memory_output.md` |
| 04 | Intrinsics and ontology | 02 | `04_intrinsics_and_ontology/` | `output/04_intrinsics_and_ontology_output.md` |
| 05 | Proxy and calibration | 04 | `05_proxy_and_calibration/` | `output/05_proxy_and_calibration_output.md` |
| 06 | Reflection ablation | 03, 04, 05 | `06_reflection_ablation/` | `output/06_reflection_ablation_output.md` |
| 07 | Generalisation test | 01, 03, 04, 05, 06 | `07_generalisation_test/` | `output/07_generalisation_test_output.md` |

## Consolidated proxy strategy
- Feature blocks:
  - `x_P`: pipeline intrinsic metrics
  - `x_O`: ontology/structural plausibility metrics
- Model: ridge regression on `x = [x_P; x_O]` to predict mIoU.
- Calibration: Platt scaling on `s = y_hat - tau` to compute `p_good`.
- Acceptance: `y_hat >= tau` and `p_good >= p_min`.

## Reflection policy (fixed only)
- poor ring boundary quality -> rerun boundary detection
- poor oblique line quality -> adjust K-line detection
- invalid segment count -> adjust geometry segmentation
- high spacing irregularity -> rerun ring boundary detection

No template/mask action and no RL/LLM routing.

## Coding plans
Implementation-level step-by-step coding guidance is maintained in:
`methods/ablation/steps/`
