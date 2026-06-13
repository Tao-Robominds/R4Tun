# Repeatability summary (m+s+k, temperature 0)

Pairs analysed: **27** (target 90 = 30 tunnels × 3 LLMs).

## Aggregate

| Metric | Value |
|--------|-------|
| Median critical-param identity (18 params) | 100.0% |
| Mean critical-param identity | 90.9% |
| Mean \|ΔmIoU\| (run1 vs run2) | 0.0274 |
| Median \|ΔmIoU\| | 0.0000 |

## Per combo

| Tunnel | Model | Run2 source | mIoU run1 | mIoU run2 | \|ΔmIoU\| | Critical identical |
|--------|-------|-------------|-----------|-----------|---------|-------------------|
| 1-2 | opus4.6 | inference | 0.608 | 0.608 | 0.0000 | 18/18 (100%) |
| 1-3 | opus4.6 | inference | 0.658 | 0.658 | 0.0000 | 18/18 (100%) |
| 1-4 | opus4.6 | inference | 0.436 | 0.436 | 0.0000 | 18/18 (100%) |
| 1-5 | opus4.6 | inference | 0.629 | 0.629 | 0.0000 | 18/18 (100%) |
| 2-1 | opus4.6 | inference | 0.674 | 0.674 | 0.0000 | 18/18 (100%) |
| 2-2 | opus4.6 | inference | 0.685 | 0.685 | 0.0000 | 18/18 (100%) |
| 2-3 | opus4.6 | inference | 0.606 | 0.606 | 0.0000 | 18/18 (100%) |
| 2-5 | opus4.6 | inference | 0.669 | 0.669 | 0.0000 | 18/18 (100%) |
| 4-1 | opus4.6 | inference | 0.172 | 0.093 | 0.0790 | 15/18 (83%) |
| 4-10 | opus4.6 | inference | 0.190 | 0.144 | 0.0460 | 15/18 (83%) |
| 4-2 | opus4.6 | inference | 0.166 | 0.271 | 0.1050 | 15/18 (83%) |
| 4-4 | gemini3flash | harvested | 0.108 | 0.108 | 0.0000 | 18/18 (100%) |
| 4-4 | gpt5.4 | harvested | 0.133 | 0.075 | 0.0580 | 8/18 (44%) |
| 4-4 | opus4.6 | harvested | 0.245 | 0.047 | 0.1980 | 13/18 (72%) |
| 4-5 | opus4.6 | inference | 0.254 | 0.257 | 0.0030 | 16/18 (89%) |
| 4-6 | opus4.6 | inference | 0.139 | 0.146 | 0.0070 | 16/18 (89%) |
| 4-8 | opus4.6 | inference | 0.115 | 0.107 | 0.0080 | 16/18 (89%) |
| 4-9 | opus4.6 | inference | 0.070 | 0.130 | 0.0600 | 16/18 (89%) |
| 5-1 | opus4.6 | inference | 0.150 | 0.170 | 0.0200 | 16/18 (89%) |
| 5-3 | gemini3flash | harvested | 0.143 | 0.143 | 0.0000 | 18/18 (100%) |
| 5-3 | gpt5.4 | harvested | 0.124 | 0.124 | 0.0000 | 18/18 (100%) |
| 5-3 | opus4.6 | harvested | 0.178 | 0.178 | 0.0000 | 18/18 (100%) |
| 5-4 | gemini3flash | harvested | 0.122 | 0.118 | 0.0040 | 13/18 (72%) |
| 5-4 | gpt5.4 | harvested | 0.098 | 0.098 | 0.0000 | 18/18 (100%) |
| 5-4 | opus4.6 | harvested | 0.207 | 0.207 | 0.0000 | 18/18 (100%) |
| 5-5 | opus4.6 | inference | 0.342 | 0.246 | 0.0960 | 15/18 (83%) |
| 5-6 | opus4.6 | inference | 0.123 | 0.068 | 0.0550 | 16/18 (89%) |

## Reviewer response snippet

Under m+s+k with temperature set to 0, a second LLM inference pass was compared to the primary run on 27 tunnel–model pairs. Median identity on the 18 critical parameters was 100% (mean 91%); mean |ΔmIoU| was 0.027 (median 0.000), smaller than the paired adaptation gain (ΔmIoU ≈ 0.17–0.19 vs baseline).

## LaTeX table row (fill when n=90)

```
Median critical-parameter identity & 100\% \\ Mean $|\Delta$mIoU| (run1 vs run2) & 0.027 \\ Pairs analysed & 27/90
```
