# V5 Archive

This folder is the canonical archive for the v5 experiment cycle and contains
the held-out panel definition, archived v5 run outputs, and handoff metadata
for starting v6 work.

## Scope

- 50-ring held-out panel: 10 rings per tunnel family (`1` through `5`).
- K-bearing segmentation ontology for stage validation and comparison.
- Deployment/runtime logic remains GT-free; GT appears only in offline mIoU/OA
  audit artifacts.

## Canonical Panel

- `stages/v5/panels/v5_50ring_panel.csv` is the canonical ring list.

## Archived V5 Logs

- Archived root: `stages/v5/logs/`
- Archive manifest: `stages/v5/archive_manifest_v5.json`
- Historical source location before archive: `logs/v5*/`

All top-level `logs/v5*` directories were moved into `stages/v5/logs/` for
v6 handoff.

## Key Result Pointers

- Stage validation:
  - `stages/v5/logs/v5_stage_validation_v1/v5_50ring_scoreboard.csv`
  - `stages/v5/logs/v5_stage_validation_v1/v5_stage_summary.json`
- Static baseline:
  - `stages/v5/logs/v5_static_r4tun_baseline_v1/static_r4tun_50ring_scoreboard.csv`
  - `stages/v5/logs/v5_static_r4tun_baseline_v1/static_r4tun_summary.json`
- Final proxy-selected refined outputs:
  - `stages/v5/logs/v5_balance_norm_proxy_v1/selected_scoreboard_50rings.csv`
  - `stages/v5/logs/v5_balance_norm_proxy_v1/summary.json`
- Depth contract paper audit:
  - `stages/v5/logs/v5_depth_contract_paper_audit_v1/all_50_depth_quality_summary.json`
  - `stages/v5/logs/v5_depth_contract_paper_audit_v1/all_50_depth_quality_report.md`

## Notes For V6

- Several `bo/v5` scripts still reference `logs/v5_*` paths directly. Keep
  them as historical provenance and use the archive manifest for path mapping.
- Regressions where refined < stabilised should be analyzed from:
  - `stages/v5/logs/v5_stage_validation_v1/v5_50ring_scoreboard.csv`
  - `stages/v5/logs/v5_balance_norm_proxy_v1/selected_scoreboard_50rings.csv`

