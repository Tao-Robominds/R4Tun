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
| 04 | Parameter inventory | 03 | `04_parameter_inventory/` | — |
| 05 | Data-flow graph | 04 | `05_data_flow_graph/` | — |
| 06 | Critical param set | 02, 04, 05 | `06_critical_param_set/` | — |
| 07 | GT warm start | 06 | `07_gt_warm_start/` | — |
| 08 | BO runs | 06, 07 | `08_bo_runs/` | — |
| 09 | Intrinsic analysis | 08 | `09_intrinsic_analysis/` | — |
| 10 | Proxy training | 09 | `10_proxy_training/` | — |
| 11 | Reflection agent | 09, 10 | `11_reflection_agent/` | — |

**Rule:** Without the previous step's verified output, the current step cannot proceed.

## SOP Table
| Step | Verification condition | Fail action |
|---|---|---|
| `01` | scope + assumptions + gaps + evidence present | stop; restate baseline and assumptions |
| `02` | every assumption marked stable/broken with evidence, class, failure mode, response | stop; compare GT and classify gaps |
| `03` | edits + structural limits + risks present | stop; split code vs structural |
| `04` | all params have stage + type + source + value | stop; complete inventory |
| `05` | nodes + edges + critical path present | stop; rebuild graph |
| `06` | rule + selected + excluded + safe fixed present | stop; restate selection rule |
| `07` | warm start + fixed + bounds + priors present | stop; redo GT reverse-engineering |
| `08` | metadata + params + GT + artefacts + feature bank + reflection logs present | stop; fix log schema |
| `09` | metric bank + selected intrinsics + ranges + guardrails + knowledge write-back present | stop; refine intrinsic analysis |
| `10` | validation + calibration + uncertainty + confidence bank + trust rule present | stop; do not deploy proxy |
| `11` | case rules + actions + fallback + uncertainty logic present | stop; rewrite policy prompt |

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
