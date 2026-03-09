# No-GT Portable Workflow

## Purpose
Portable workflow pack for irregular tunnel adaptation.

## Dataset
- `data/4-1.txt` — 583k pts, 6 rings (120–125), 7 block types, diameter 7.5 m, ring spacing 1.816 m.
- `data/5-1.txt` — 1.5M pts, 7 rings (107–113), 7 block types, diameter 7.5 m, ring spacing 1.816 m.

## Folders
- `steps/`: abstract step manuals. Define what each step must do, its inputs, outputs, and verification.
- `output/`: canonical filled outputs for the steps. Concrete artifacts with evidence from 4-1 and 5-1.
- `templates/`: artifact templates.
- `scripts/`: scaffold and verify helpers.

## How to read
- `steps/` = the plan (what to do).
- `output/` = the results (what was found).
- Per-run runtime artifacts go under `data/{tunnel_id}/workflow/{run_id}/`.

## Step order and dependencies

Steps are **strictly sequential**. Each step requires the output of all previous steps. Do not start step N until step N−1 is verified.

| # | Step | Depends on | Runtime artifact | Canonical output |
|---|------|-----------|-----------------|-----------------|
| 01 | Assumptions | — | `01_assumptions/` | `output/01_assumptions_output.md` |
| 02 | Challenge map | 01 | `02_challenge_map/` | `output/02_challenge_map_output.md` |
| 03 | Upgrade solution | 01, 02 | `03_upgrade_solution/` | — |
| 04 | Critical params for BO | 02, 03 | `04_critical_params_for_bo/` | `critical_params.yaml` |
| 05 | GT warm start | 04 | `05_gt_warm_start/` | — |
| 06 | BO runs | 04, 05 | `06_bo_runs/` | — |
| 07 | Intrinsic analysis | 06 | `07_intrinsic_analysis/` | — |
| 08 | Proxy training | 07 | `08_proxy_training/` | — |
| 09 | Reflection agent | 07, 08 | `09_reflection_agent/` | — |

**Rule:** Without the previous step's verified output, the current step cannot proceed.

## SOP Table
| Step | Verification condition | Fail action |
|---|---|---|
| `01` | scope + assumptions + gaps + evidence present | stop; restate baseline and assumptions |
| `02` | every assumption marked stable/broken with evidence, class, failure mode, response | stop; compare GT and classify gaps |
| `03` | edits + structural limits + risks present | stop; split code vs structural |
| `04` | critical params: selected + excluded + safe fixed + rule documented | stop; complete inventory + data-flow + selection |
| `05` | warm start + fixed + bounds + priors present | stop; redo GT reverse-engineering |
| `06` | metadata + params + GT + artefacts + feature bank + reflection logs present | stop; fix log schema |
| `07` | metric bank + selected intrinsics + ranges + guardrails + knowledge write-back present | stop; refine intrinsic analysis |
| `08` | validation + calibration + uncertainty + confidence bank + trust rule present | stop; do not deploy proxy |
| `09` | case rules + actions + fallback + uncertainty logic present | stop; rewrite policy prompt |

## Non-Negotiable
- Proxy uncertainty is mandatory.
- Do not trust predicted mIoU without holdout validation, calibration, and confidence signals.
- Do not skip steps. Without the previous step's output, the current step cannot start.

## Quick Start
```bash
python plans/scripts/scaffold_run.py --tunnel 5-1 --run pilot_001
python plans/scripts/verify_step.py --root data/5-1/workflow/pilot_001 --step 01
python plans/scripts/verify_step.py --root data/5-1/workflow/pilot_001 --step all
```
