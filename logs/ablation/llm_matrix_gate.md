# LLM matrix validation gate

- **Status:** PASS
- **Started:** 2026-07-07 (gate retry after WORK_DIR fix)
- **Command:** `venv/bin/python run_memory_state_knowledge.py 1-1 --model opus4.6`
- **Tunnel:** 1-1
- **Condition:** m+s+k (memory+state+knowledge)
- **Model:** Opus-4.6 (opus4.6)

## Results

| Metric | Value |
|--------|-------|
| OA | 0.810 |
| mIoU | 0.641 |
| Wall time | 445 s |

## Pass criteria

1. All 5 stages complete with JSON extraction — **PASS**
2. `evaluation/performance.md` written — **PASS**
3. mIoU plausible (prior Opus ~0.6 on 1-1) — **PASS** (0.641)

## Output path

`data/ablation/memory+state+knowledge/1-1_opus4.6/` (renamed after gate)

## Notes

- Required `data/sample/characteristics/raw_characteristics.json` (generated via `raw_characteristics.py --tunnel_id sample`)
- Required `R4TUN_PIPELINE_WORK_DIR=data/ablation/{condition}` in orchestrator `_setup_env` so characterisers find pipeline CSVs
