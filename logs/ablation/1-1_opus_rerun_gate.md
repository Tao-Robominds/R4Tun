# 1-1 Opus Rerun Gate — sam4tun/agents pipeline

**Date:** 2026-07-07  
**Tunnel:** 1-1  
**Model:** opus4.6  
**Pipeline:** sam4tun/agents (unchanged)  
**Reference:** `1-1_opus4.6_sam4tunpipe` — mIoU **0.826**, OA **0.917**

## Gate status: **FAIL**

Fresh m+s+k Opus API run did **not** reproduce reference performance. Do not scale matrix until resolved.

## Results

| Condition | Command | Status | mIoU | OA | Output |
|-----------|---------|--------|------|-----|--------|
| m | `./venv/bin/python run_memory.py 1-1` | PASS | 0.4684 | 0.6840 | `data/ablation/memory/1-1_opus4.6/` |
| m+s | `./venv/bin/python run_memory_state.py 1-1` | **FAIL** (SAM stage) | — | — | partial: `data/ablation/memory+state/1-1/` (characteristics only) |
| m+s+k | `./venv/bin/python run_memory_state_knowledge.py 1-1` | complete, **gate FAIL** | **0.3048** | 0.5364 | `data/ablation/memory+state+knowledge/1-1_opus4.6/` |

**Gate criterion:** mIoU 0.826 / OA 0.917 (exact match expected with temperature=0).  
**Observed:** mIoU 0.3048 — far below 0.80 conditional-pass floor.

## m+s+k param diff vs replay (known-good)

| Stage | Match replay? | Notes |
|-------|---------------|-------|
| unfolding | YES | MD5 `eaa2c6cffedaafde255e0b5790143e7d` |
| denoising | **NO** | Replay: `y_step=0.5, z_step=0.001, default_cutoff_z=2.95, grad_threshold=0.2, smoothing_window_size=3`. Fresh: `y_step=0.4, z_step=0.005, default_cutoff_z=2.85, grad_threshold=0.15, smoothing_window_size=5` |
| enhancing | **NO** | Replay: stage distances `0.06/0.03/0.015`, `inter_radius=0.03`. Fresh: `0.08/0.04/0.02`, `inter_radius=0.06` |
| detecting | YES | MD5 `bd98dc316e1d739d40993a4e42974139` |
| sam | YES | MD5 `75e0fcfa357f8727034c2e152d029ab1` |

Param paths: `sam4tun/agents/parameters/memory+state+knowledge/1-1/parameters_*_m_s_k_opus4.6.json`

## m+s failure

SAM crashed at ring 6/10:

```
cv2.error: (-215:Assertion failed) !ssize.empty() in function 'resize'
```

Likely bad detecting prompts (Y values ~2700 out of image bounds). Log: `logs/ablation/1-1_m_s_opus4.6_rerun.log`

## Conclusion

Pipeline code was **not modified**. Failure is due to **Opus emitting different LLM parameters** on fresh API calls (denoising + enhancing differ from replay), not pipeline regression. To reproduce 0.826 exactly, re-run pipeline-only with the archived replay params (or fix param persistence / seeding before matrix scale).

## Logs

- `logs/ablation/1-1_m_opus4.6_rerun.log`
- `logs/ablation/1-1_m_s_opus4.6_rerun.log`
- `logs/ablation/1-1_m_s_k_opus4.6_rerun.log`
