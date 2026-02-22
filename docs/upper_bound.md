# Theoretical Upper Bound: Direct GT Label Projection

**Definition:** For each 3D point, assign as "prediction" the GT segment label that ends up at that point's projected pixel after painting every point's GT label onto the depth map (last-write-wins). Evaluate mIoU at point level. This is the ceiling for any pixel-based segmentation: no method can exceed it without changing the mapping or the point set.

**Why not 100%:** (1) **Multi-point→same-pixel conflicts** — several 3D points project to the same pixel; only one label is kept (last write), so the others are counted wrong. (2) **Sparse pixel coverage** — only a fraction of depth-map pixels receive any projection; evaluation is on mapped points only, but conflict rate grows with points-per-pixel. (3) **BG/segment co-projection** — background and block points can land on the same pixel, so one overwrites the other.

---

## Per-tunnel direct GT projection ceiling

| Tunnel | mIoU (ceiling) | Why below 100% |
|--------|----------------|----------------|
| **1-4** | 0.977 | ~1.78M point-pixel conflicts; 13.6% of pixels receive a projection → high point density per pixel. |
| **2-2** | 0.947 | ~2.57M conflicts; 34.2% pixel coverage → more pixels used but still many points per pixel. |
| **3-1** | 0.946 | ~260k conflicts; 20.2% pixel coverage. |
| **4-1** | 0.986 | ~129k conflicts; 6.9% pixel coverage. |
| **5-1** | 0.988 | ~233k conflicts; 9.2% pixel coverage → fewer conflicts than 1-4/2-2. |

**Computed with:** `scripts/theoretical_ceiling_gt_projection.py` (paint GT to pixels, last-write-wins; evaluate on points in `pixel_to_point.pkl`).

---

## Best performance so far (journal 2026-02-21)

Pipeline: **complex_agents_wrap** — 100% GT coverage, GT-derived params, Approach A (Periodic Y crop), template fallback. Tunnels 1-4, 2-2, 3-1, 4-1 from journal multi-tunnel rollout; 5-1 from same pipeline run (`final_wrap_a.csv` / `performance_wrap_a.md`).

| Tunnel            | Type   | mIoU  | OA    |
|-------------------|--------|-------|-------|
| 1-4               | simple | 0.908 | 0.950 |
| 2-2               | simple | 0.911 | 0.952 |
| 3-1               | simple | 0.861 | 0.924 |
| 4-1               | complex| 0.864 | 0.925 |
| 5-1 (no-wrap)     | complex| 0.795 | 0.876 |
| 5-1 (with-wrap)   | complex| 0.746 | 0.840 |

5-1 (with-wrap) is lower than 4-1 because 5-1 has **wrap-around blocks** (8 of 49); 4-1 has no wrap in the evaluated setup.

---

## 5-1 only: with-wrap vs without-wrap ceiling

**Only tunnel 5-1** has blocks that span the periodic Y boundary of the depth map (8 wrap blocks). So 5-1 has two evaluation views:

| Scope        | mIoU (best) | Ceiling / notes |
|-------------|-------------|-----------------|
| **5-1 no-wrap** (easy subset: no ring 4, no wrap blocks) | **0.795** | Point-level mIoU from `scripts/evaluate_5_1_easy_subset.py` on `data/5-1/final_wrap_a.csv`. OA 0.876. Direct GT projection ceiling ~0.99; pixel-majority ceiling 0.993. |
| **5-1 with-wrap** (49 blocks, full) | **0.746** | Point-level mIoU from same run (performance_wrap_a.md). Pipeline ceiling ~0.872 if non-wrap perfect and 8 wrap blocks left as pred=0. OA 0.840. |

**Why the with-wrap ceiling is lower:** Wrap blocks suffer angular-boundary mismatch and template misalignment in the stitched coordinate system; even with Y-wrap crop, template/aggregation are not yet correct for wrap geometry. So full (all-blocks) mIoU is capped until wrap is fully solved. Current best 5-1 results: `data/5-1/evaluation/performance_wrap_*.md`.
