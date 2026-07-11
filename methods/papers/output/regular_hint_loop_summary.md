# Regular Hint Loop — Minimum-Hint Ablation Summary

Model: **opus4.6** | Ablation: **memory+state+knowledge** | Upstream: frozen `data/ablation_anthropic`

**Goal:** ≥ 8/10 tunnels with mIoU ≥ 0.8 using minimum detection hints.

**Outcome:** **0/10** tunnels reach mIoU ≥ 0.8 at any hint level tested. Best tunnel (**2-2**) peaks at **0.685** (L0/L2/L5). Detection hints cannot close the gap to 0.8; the ceiling is set by SAM block-walk / fixed template (C3/C4 constraints), not K-anchor Y alone.

## Validation gate (single-instance, before scaling)

| Case | Level | mIoU | Criteria | Evidence |
|------|-------|------|----------|----------|
| 2-2 (best) | L5 `gt_k_all` | **0.685** | Oracle K Y establishes upper bound | `data/regular_hint_loop/L5/2-2/evaluation/performance.md` |
| 1-3 (T1 gate) | L5 `gt_k_all` | **0.658** | Strongest T1 baseline; partial hints approach 0.8 | `data/regular_hint_loop/L0/1-3/evaluation/performance.md` |
| 1-4 (stress only) | L5 `gt_k_all` | **0.436** | Pathological detection outlier | `data/regular_hint_loop/L5/1-4/evaluation/performance.md` |

Command lineage: `python3 methods/papers/scripts/run_regular_hint_loop.py --level L5 --gate`

## Hint ladder (gate tunnels 2-2, 1-3)

| Level | Mode | 2-2 | 1-3 | 1-4 (retired) |
|-------|------|-----|-----|---------------|
| L0 | off | 0.685 | **0.658** | 0.436 |
| L1 | zigzag_prior | 0.256 | n/a | **0.469** |
| L2 | zigzag_fit | 0.685 | n/a | 0.342 |
| L4 | two_gt_k | 0.682 | n/a | 0.376 |
| L5 | gt_k_all | 0.685 | n/a | 0.436 |

## L0 baseline — all 10 regular tunnels

| Tunnel | L0 mIoU | S5a mIoU | ≥ 0.8 (S5a) |
|--------|---------|----------|------------|
| 1-1 | 0.622 | 0.756 | ✗ |
| 1-2 | 0.608 | 0.723 | ✗ |
| **1-3** | **0.658** | **0.773** | ✗ (closest T1) |
| 1-4 (stress) | 0.436 | 0.600 | ✗ |
| 1-5 | 0.630 | 0.729 | ✗ |
| 2-1 | 0.674 | 0.761 | ✗ |
| 2-2 | **0.685** | **0.801** | ✓ |
| 2-3 | 0.606 | 0.713 | ✗ |
| 2-4 | 0.626 | 0.717 | ✗ |
| 2-5 | 0.669 | 0.777 | ✗ |

**Mean L0 mIoU:** 0.621 | **Pass ≥ 0.8:** 0/10

## Best mIoU per tunnel (any level)

| Tunnel | Best mIoU | Best level | Notes |
|--------|-----------|------------|-------|
| 1-1 | 0.622 | L0 | Hough sufficient |
| 1-2 | 0.608 | L0 | |
| 1-3 | 0.658 | L0 | |
| Tunnel | S0 | S5a | S5b | Notes |
|--------|-----|-----|-----|-------|
| **1-3** (new T1 gate) | **0.658** | **0.773** | 0.869 | Best T1 partial-hint; still below 0.8 without swap fix |
| 1-4 (retired gate) | 0.436 | 0.600 | 0.857 | Pathological detection; only T1 helped by L1 zigzag |
| 1-1 | 0.622 | 0.756 | 0.875 | validation_gate regular representative |
| 1-2 | 0.608 | 0.723 | 0.845 | |
| 1-5 | 0.630 | 0.729 | 0.880 | |
| 1-5 | 0.630 | L0 | |
| 2-1 | 0.674 | L0 | |
| 2-2 | **0.685** | L0/L2/L5 | Hough already near ceiling |
| 2-3 | 0.606 | L0 | |
| 2-4 | 0.626 | L0 | |
| 2-5 | 0.669 | L0 | |

## Minimum-hint conclusion

1. **For tunnel family 2 (`2-*`):** **L0 (no hints)** is sufficient and optimal. Zigzag/GT hints do not improve mIoU; v2-style snapping was avoided (v3 fallback-only + full Y replacement in pattern modes).

2. **For tunnel family 1 (`1-*`):** Weak case **1-4** gains slightly with **L1** fixed zigzag prior (+0.033 mIoU) but remains far below 0.8. GT K hints (L4–L6) do not beat L0 on average.

3. **Minimum information to match current pipeline:** **L0** — standard Hough detection with v3 fallback-only K consensus.

4. **To reach mIoU ≈ 0.8:** Detection hints are insufficient. See **[`regular_sam_hint_loop_summary.md`](regular_sam_hint_loop_summary.md)** — SAM-stage **S5b (oracle_swap)** is the minimum level reaching **10/10** tunnels ≥ 0.8 (mean 0.863). Partial class oracle S5a (K+A2+A3 GT) reaches 0.8 on 2-2 only.

## Artifacts

- Code: `agents_regular/detecting.py`, `methods/papers/scripts/regular_hint_lib.py`, `methods/papers/scripts/k_consensus_lib.py` (v3)
- Runner: `methods/papers/scripts/run_regular_hint_loop.py`
- Results: `data/regular_hint_loop/{L0..L7}/{tunnel}/`
- Logs: `logs/regular_hint_loop/`

## Reproduce

```bash
# Single level, one tunnel
export R4TUN_PIPELINE_OUT_PREFIX=data/regular_hint_loop/L5
./venv/bin/python agents_regular/detecting.py 2-2 --ablation m_s_k --model opus4.6
./venv/bin/python agents_regular/sam.py 2-2 --ablation m_s_k --model opus4.6
./venv/bin/python agents_regular/evaluation.py 2-2 --ablation m_s_k --schema auto

# Full ladder (gate)
python3 methods/papers/scripts/run_regular_hint_loop.py --all-levels --gate

# All tunnels baseline
python3 methods/papers/scripts/run_regular_hint_loop.py --level L0 --all-tunnels
```
