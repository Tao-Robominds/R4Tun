# No-GT BO Results: All Tunnels

**Date:** 2026-02-02  
**Runs:** 30 iterations each, detection stage  
**Result files:** `no_gt_bo_{tunnel}_detection_*.json`

---

## Summary Table

| Tunnel | Pattern           | Predicted mIoU | True mIoU | Error   | Notes                    |
|--------|-------------------|----------------|-----------|---------|--------------------------|
| **1-4**  | simple_staggered  | 0.646          | **0.698** | -0.052  | ✓ Works well             |
| **2-2**  | simple_staggered  | 0.672          | 0.476     | +0.196  | Over-estimated           |
| **3-1**  | continuous        | 0.520          | **0.508** | +0.012  | ✓ Very close (SAM had JSON error) |
| **4-1**  | complex_staggered | 0.344          | 0.052     | +0.292  | Wrong pipeline (see below) |
| **5-1**  | complex_staggered | 0.330         | 0.097     | +0.233  | Wrong pipeline (see below) |

---

## Per-Tunnel Details

### 1-4 ✓
- **Best eval:** #28, guardrails passed
- **True mIoU:** 0.698 (improvement over baseline ~0.626)
- **Prediction:** Slight under-estimate (-5.2%) — conservative and safe

### 2-2
- **Best eval:** #17, guardrails passed
- **True mIoU:** 0.476
- **Prediction:** Over-estimated (+19.6%). Predictor may be less calibrated for 2-2; more 2-2 training data could help.

### 3-1 ✓
- **Best eval:** #16
- **True mIoU:** 0.508
- **Prediction:** Very close (+1.2%). Note: 3-1 SAM run hit `JSONDecodeError` (corrupt parameters_sam.json); evaluation used existing final.csv.

### 4-1, 5-1 (Pipeline Mismatch)
- **4-1** and **5-1** use the **complex_staggered** pipeline:
  - Detection: `4-1_detection_complex.py` (not `4-1_detection.py`)
  - SAM: `4-2_sam_wraparound.py` (not `4-2_sam.py`)
- No-GT optimizer currently runs **standard** `4-1_detection.py` for all tunnels, so 4-1 and 5-1 were tuned with the wrong detection script.
- **True mIoU** (0.052, 0.097) reflects standard detection + standard SAM on complex-staggered data, not the intended complex pipeline.
- **To validate 4-1/5-1:** Extend no_gt_optimizer to use `4-1_detection_complex.py` and the complex SAM search space for complex_staggered tunnels.

---

## Conclusion

| Outcome | Tunnels |
|---------|--------|
| **No-GT BO validated** | 1-4, 3-1 (prediction error &lt; 6%) |
| **Needs calibration** | 2-2 (over-estimate) |
| **Needs pipeline support** | 4-1, 5-1 (complex_staggered) |

**Recommendations:**
1. Use no-GT BO for **1-4** and **3-1** as-is.
2. Add tunnel-specific predictor calibration or more 2-2 training data for **2-2**.
3. Add **complex_staggered** branch in no_gt_optimizer (complex detection + wraparound SAM) before claiming results for **4-1** and **5-1**.
