# LLM repeatability experiment (run 2, skip-max)

## Objective

Answer reviewer Q2: whether adapted parameters remain stable across repeated API calls under m+s+k (temperature 0), on 30 tunnels × 3 LLMs, without overwriting `data/ablation/` or `agents/ablation/`.

## Artifacts

| Artifact | Path |
|----------|------|
| Run 1 snapshots | `logs/{tunnel}/repeatability/run1/{model}/` |
| Run 2 (new inference) | `logs/{tunnel}/repeatability/run2_{TS}/{model}/` |
| Run 2 (harvested reruns) | `logs/{tunnel}/repeatability/run2_harvested/{model}/` |
| Batch summary CSV | `logs/repeatability_{TS}_summary.csv` |
| Analysis output | `methods/papers/output/repeatability_summary.md` |
| Bootstrap script | `methods/papers/scripts/bootstrap_repeatability_run1.py` |
| Run script | `methods/papers/scripts/run_repeatability.py` |
| Analysis script | `methods/papers/scripts/reproducibility_analysis.py --layout repeatability` |
| Shared helpers | `methods/papers/scripts/repeatability_common.py` |

## Execution

```bash
python3 methods/papers/scripts/bootstrap_repeatability_run1.py
python3 methods/papers/scripts/run_repeatability.py --harvest-only
python3 methods/papers/scripts/run_repeatability.py --tunnel 1-1 --model opus4.6
python3 methods/papers/scripts/run_repeatability.py --skip-existing
python3 methods/papers/scripts/reproducibility_analysis.py --layout repeatability
```

## Verification

- [ ] 90 rows in summary CSV (30 tunnels × 3 models; run2 source = `harvested` or `inference`)
- [ ] `agents/ablation/` and `data/ablation/` restored after each combo
- [ ] `repeatability_summary.md` reports median critical-param identity and mean mIoU range
- [ ] Appendix I + Section 3.5.2 cite measured stats (not pre-run claims)

## Skip rules

1. Run 1 = disk snapshot (no API/GPU).
2. Harvest 9 combos from `logs/{4-4,5-3,5-4}/rerun_*/m_s_k/`.
3. Seed run 1 params before orchestrator so per-stage pipeline skip applies when LLM output matches.
4. `--skip-existing` for resumable queue.
