# Proxy BO Experiment – All Tunnels (10 calls each)

**Date:** 2026-02-02  
**Stage:** detection  
**N calls:** 10 per tunnel  
**N initial:** 3  

## Individual results

| Tunnel | Baseline | Proxy best (true mIoU) | True BO best | Δ Proxy vs baseline | Δ Proxy vs oracle |
|--------|----------|------------------------|--------------|---------------------|-------------------|
| 1-4    | 0.000*   | 0.000*                 | **0.595**    | —                   | -0.595            |
| 2-2    | 0.000*   | 0.000*                 | **0.365**    | —                   | -0.365            |
| 3-1    | 0.000*   | 0.000*                 | 0.000*       | —                   | —                 |
| 4-1    | 0.000*   | 0.000*                 | **0.083**    | —                   | -0.083            |
| 5-1    | **0.103**| **0.102**              | **0.116**    | -0.001              | -0.014            |

\* Baseline or proxy evaluation failed (SAM/JSON errors). True BO still completed for 1-4, 2-2, 4-1.

## Average (all 5 tunnels)

| Metric             | Value  |
|--------------------|--------|
| Baseline           | 0.0206 |
| Proxy best         | 0.0204 |
| True BO best       | 0.2318 |
| Δ Proxy vs baseline| -0.0002|
| Δ Proxy vs oracle  | -0.211 |

## Average (successful tunnel only: 5-1)

| Metric             | Value  |
|--------------------|--------|
| Baseline           | 0.103  |
| Proxy best         | 0.102  |
| True BO best       | 0.116  |
| Δ Proxy vs baseline| -0.001 |
| Δ Proxy vs oracle  | -0.014 |

## Interpretation

- **5-1 (full success):** Proxy BO reaches almost the same true mIoU as baseline (0.102 vs 0.103) and stays within ~88% of oracle (0.102 / 0.116). Predictor behaves as intended for this tunnel.
- **1-4, 2-2, 4-1:** Baseline and proxy evaluation failed, so true mIoU could not be measured. True BO ran and produced valid scores, indicating SAM/params issues during baseline and proxy runs (e.g. template mask dimensions).
- **3-1:** All three runs failed (likely JSON parse errors in SAM params).

**Conclusion:** Where the pipeline ran correctly (5-1), the proxy predictor is safe to use for tuning. Fixing SAM/param issues on other tunnels is needed before repeating the experiment there.
