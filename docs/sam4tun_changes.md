# SAM4Tun.py changes (vs notebook) and evaluation

| Change | Notebook / old | SAM4Tun.py / new | Effect |
|--------|----------------|------------------|--------|
| Template logits resize | `ResizeLongestSide(256)` | `cv2.resize` (scale longest side) + zero-pad | More predictable SAM `mask_input` for non-square K (and other) crops |
| Tunnel axis | No swap | `center1, center2 = center2, center1` | Unfolding direction matches shield forward / ring order |
| Evaluation | `evaluation.py`: sklearn OA/F1/mIoU, 6/7-class schema, writes `evaluation/performance.md` | `6_evaluation.py`: ring-instance mAP@50–95 + binary foreground metrics, stdout only | Notebook-faithful instance mAP; ablation gates still use `evaluation.py` / `evaluate_static.py` for semantic mIoU |

## T1/T2 ablation results (`1-1`, `2-1`)

Source: `data/ablation/{memory,memory+state,memory+state+knowledge}/{tunnel}/evaluation/performance.md` (sklearn semantic metrics, schema auto).

| Tunnel | Ablation | OA | F1 | mIoU | Δ mIoU (vs m) |
|--------|----------|-----|-----|------|---------------|
| 1-1 | m | 0.560 | 0.496 | 0.339 | — |
| 1-1 | m+s | 0.879 | 0.830 | 0.727 | +0.388 |
| 1-1 | m+s+k | 0.912 | 0.897 | 0.815 | +0.476 |
| 2-1 | m | 0.549 | 0.453 | 0.305 | — |
| 2-1 | m+s | 0.943 | 0.938 | 0.884 | +0.579 |
| 2-1 | m+s+k | 0.944 | 0.938 | 0.885 | +0.580 |

T1/T2 gate threshold: mIoU ≥ 0.70 — all **m+s** and **m+s+k** runs pass; **m** alone does not. Gate record: `data/ablation/t1_t2_gate.md` (PASS, 2026-07-01).

## T3 results (`3-1-1` gate tunnel)

Source: `data/ablation/…`, `data/static/…`, `sam4tun/data/3-1-1/…`, `methods/papers/output/t3_tune_summary.md`, `methods/papers/output/t3_hint_loop_summary.md`.

| Pipeline | mIoU | Notes |
|----------|------|-------|
| Ablation **m** / **m+s** | 0.045–0.047 | State alone does not help |
| Ablation **m+s+k** | 0.548 | Large lift from knowledge; below 0.60 single-tunnel target |
| SAM4Tun.py (axis + resize fixes) | 0.562 | `sam4tun/data/3-1-1` |
| Static baseline | 0.045 | `data/static/3-1-1` |
| GT-free tune best single (`hough_low`) | 0.582 | `3-1-1` only; panel mean 0.306 |
| GT-free tune best panel (`hough_low_flip`) | 0.601 on `3-1-1` | Panel mean 0.331 |
| Hint loop best panel (T5) | 0.456 on `3-1-1` | Panel mean 0.380 |

T3 gate: panel mean mIoU ≥ **0.60** across `3-1-1`, `3-1-2`, `3-1-3` — **not met**. Per-tunnel gates on `3-1-1`: T1 ≥ 0.45 ✓, scale ≥ 0.55 ✓ for ablation m+s+k and SAM4Tun fixes. Bottleneck: K detection on `3-1-2`/`3-1-3` (Y-spread 121–159 px); `3-1-1` largely fixed.

## T4 results (`4-1` gate tunnel)

| Pipeline | mIoU | Notes |
|----------|------|-------|
| Static baseline | 0.038 | `data/static/4-1` |
| Rules-adapted (7.5 m geometry) | 0.157 | ~+0.12 vs static (`data/rules/4-1`) |
| Ablation **m+s+k** | 0.104 | `data/ablation/memory+state+knowledge/4-1` |

No formal T4 gate defined. Rules-adapted geometry outperforms LLM ablation on this tunnel.

## T5 results (`5-1` gate tunnel)

| Pipeline | mIoU | Notes |
|----------|------|-------|
| Static baseline | 0.037 | `data/static/5-1` |
| Rules-adapted | 0.142 | `data/rules/5-1` |
| Ablation **m+s+k** | 0.166 | Best ablation result |

No formal T5 gate defined. Same pattern as T4 — very low static baseline, modest gains from rules and ablation.

## Summary at a glance

| Family | Gate tunnel | Best current mIoU | Primary gate | Pass? |
|--------|-------------|-------------------|--------------|-------|
| **T1** | `1-1` | 0.815 (m+s+k) | ≥ 0.70 | ✓ |
| **T2** | `2-1` | 0.885 (m+s+k) | ≥ 0.70 | ✓ |
| **T3** | `3-1-1` | 0.562 (SAM4Tun fixes) / 0.548 (m+s+k) | panel ≥ 0.60 | ✗ |
| **T4** | `4-1` | 0.157 (rules) | none | ✗ |
| **T5** | `5-1` | 0.166 (m+s+k) | none | ✗ |

Reference: `sample` on SAM4Tun.py reaches mIoU **0.892** (`sam4tun/data/sample`) — upper bound for the fixed-template pipeline.

**Takeaway:** T1/T2 are solved by agent ablation (memory+state, especially +knowledge). T3 is partially recovered on `3-1-1` (0.55+) but the full T3 panel still fails. T4/T5 remain near-random on static params and only reach ~0.10–0.17 with rules or ablation.
