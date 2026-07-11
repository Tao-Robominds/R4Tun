# Regular SAM Hint Loop — Minimum-Hint Ablation Summary

Combined with detection hints ([`regular_hint_loop_summary.md`](regular_hint_loop_summary.md)).

**Goal:** ≥ 8/10 tunnels with mIoU ≥ 0.8 on regular tunnels (1-*, 2-*).

**Outcome:** Detection hints alone cannot reach 0.8. **SAM-stage swap correction (S5b)** reaches **10/10** tunnels ≥ 0.8.

---

## SAM hint ladder

| Level | Mode | Hint information | 2-2 | 1-3 (gate) | 1-4 (stress) |
|-------|------|------------------|-----|------------|--------------|
| S0 | off | Baseline SAM | 0.685 | **0.658** | 0.436 |
| S4 | gt_theta | GT block median-theta sectors | 0.574 | n/a | 0.424 |
| S5a | oracle_k_a2_a3 | GT labels for K, A2, A3 only | **0.801** | **0.773** | 0.600 |
| **S5b** | **oracle_swap** | **GT flags swapped block points** | **0.890** | **0.869** | **0.857** |
| S5 | oracle_blocks | GT labels for all block points | 0.915 | 0.893 | 0.911 |

**Gate pair:** `2-2` + `1-3` (replaces `1-4`, which is pathological for partial hints).

---

## Per-tunnel mIoU — S5b (minimum passing level)

| Tunnel | S0 | S5b | Δ |
|--------|-----|-----|---|
| 1-1 | 0.622 | **0.875** | +0.253 |
| 1-2 | 0.608 | **0.845** | +0.237 |
| 1-3 | 0.658 | **0.869** | +0.211 |
| 1-4 | 0.436 | **0.857** | +0.421 |
| 1-5 | 0.630 | **0.880** | +0.250 |
| 2-1 | 0.674 | **0.876** | +0.202 |
| 2-2 | 0.685 | **0.890** | +0.205 |
| 2-3 | 0.606 | **0.832** | +0.226 |
| 2-4 | 0.626 | **0.828** | +0.202 |
| 2-5 | 0.669 | **0.881** | +0.212 |

**Mean S5b mIoU:** 0.863

---

## Minimum-hint conclusion (detection + SAM)

### What does NOT reach 0.8

1. **Detection hints (L0–L7):** Best = L0/Hough baseline, max **0.685** (2-2). Oracle K Y (L5) cannot beat baseline.
2. **SAM geometric / theta hints (S1–S4):** All below **0.574** on best tunnel. Fixed-template bypass of SAM degrades or barely helps.
3. **Partial class oracle (S5a):** GT for 3 block classes (K, A2, A3) — only **2-2** crosses 0.8; mean **0.735**.

### What DOES reach ≥ 8/10 at 0.8

| Rank | Level | Minimum information required |
|------|-------|------------------------------|
| 1 | **S5b** | Per-point GT to identify **class-swap errors** (pred block ≠ GT block, both non-background); keep SAM background |
| 2 | S5 | Per-point GT for **all block** labels |
| — | S5a | Insufficient for 8/10 (needs 3-class GT, still misses swaps) |

**Practical minimum for 8/10 goal: S5b `oracle_swap`**

- Keeps SAM segmentation masks for background and block extent.
- Requires GT only to flag **which block points have the wrong class** (rotation / flip errors, C4 constraint).
- Does **not** require GT for background points or for correctly-classified blocks.

### Interpretation

The ~0.15 mIoU gap between S0 (0.62 mean) and 0.8 is dominated by **block class swaps** (wrong rotation around K), not K-anchor Y error or background leakage. Fixing K position alone (detection hints, S2) does not close the gap. Fixing swap errors with GT identification (S5b) closes it entirely.

To reach 0.8 **without per-point GT**, you would need a **ring-level walk-direction hint** (flip vs rotation) that does not require knowing which points are wrong — S3 attempted this geometrically and failed; a corrected ring-level flip detector is the next research step.

---

## Artifacts

- Code: [`agents_regular/sam.py`](../agents_regular/sam.py), [`methods/papers/scripts/regular_sam_hint_lib.py`](../methods/papers/scripts/regular_sam_hint_lib.py)
- Runner: [`methods/papers/scripts/run_regular_sam_hint_loop.py`](../methods/papers/scripts/run_regular_sam_hint_loop.py)
- Results: `data/regular_sam_hint_loop/{S0..S5}/{tunnel}/`
- Logs: `logs/regular_sam_hint_loop/`

## Reproduce

```bash
# SAM swap-oracle (minimum for 8/10)
python3 methods/papers/scripts/run_regular_sam_hint_loop.py --level S5b --all-tunnels

# Full ladder on gate tunnels
python3 methods/papers/scripts/run_regular_sam_hint_loop.py --all-levels --gate
```
