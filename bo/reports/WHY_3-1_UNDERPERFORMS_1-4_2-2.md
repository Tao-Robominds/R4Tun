# Why 3-1 Underperforms 1-4 and 2-2

**Observed metrics (same pipeline, standard SAM):**

| Tunnel | Pattern           | mIoU  | OA    | F1    | Points  | Rings |
|--------|-------------------|-------|-------|-------|---------|-------|
| 1-4    | simple_staggered  | 0.626 | 0.846 | 0.734 | ~3.3M   | 10    |
| 2-2    | simple_staggered  | 0.775 | 0.890 | 0.872 | ~3.7M   | 10    |
| 3-1    | continuous        | 0.457 | 0.671 | 0.604 | ~5.0M   | 6     |

Continuous is often assumed easier than simple-staggered, and 3-1 has a denser point cloud. Below are **likely reasons** 3-1 still does worse.

---

## 1. **Shorter along-tunnel span (h) → fewer rings, different geometry**

| Tunnel | h span (m) | Depth map H×W    | Aspect (H/W) |
|--------|------------|------------------|--------------|
| 1-4    | 3.65       | 2622×2410        | 1.09         |
| 2-2    | 3.64       | 2752×2419        | 1.14         |
| 3-1    | **1.37**   | **2925×1495**    | **1.96**     |

3-1 has **~2.7× shorter** h-span. The depth map is **narrower** (width = h) and **taller** (height = θ). Result:

- **6 rings** vs 10 → **fewer SAM rows** (6 vs 10). Fewer vertical (X) prompts → less context for mask aggregation.
- **Same** K_height / AB_height in mm → similar vertical extent per block in pixels. But **ring spacing** along h is **larger** (fewer rings in shorter span). Template placement may fit 1-4/2-2 better than 3-1.

---

## 2. **Sparser depth-map boundaries (fewer valid pixels)**

| Tunnel | Valid depth pixels | Rows with ≥1 valid | Coverage |
|--------|--------------------|--------------------|----------|
| 1-4    | 14,328             | 2,001 / 2,622      | **76%**  |
| 2-2    | 18,086             | 2,011 / 2,752      | **73%**  |
| 3-1    | **12,387**         | **1,549 / 2,925**  | **53%**  |

3-1 has **fewest** valid depth pixels and **lowest** row-wise coverage. The depth map is built from **outlier (boundary)** points. So:

- **Denser** raw point cloud (5M) but **sparser** boundary projection.
- SAM segments the **depth map**. More gaps → weaker edges → worse segmentation, especially at block boundaries.

---

## 3. **K positions: two flat bands vs staggered spread**

| Tunnel | K Y range (px)     | Spread |
|--------|--------------------|--------|
| 1-4    | 1148 – 1477        | ~330   |
| 2-2    | 1209 – 1551        | ~342   |
| 3-1    | **1449 – 1459**    | **~10**|

3-1 uses **two horizontal bands** (continuous, GT-derived). 1-4/2-2 have **staggered** Y with much larger spread.

- For 3-1, **all** blocks are placed **below** K (map_y − Δ). K sits around **mid-height** (≈1462). The full block stack (K + 5×AB) extends **~3300 px** upward → **crosses the top seam (Y=0)**.
- **No wraparound** (standard SAM) → crops are **clamped** at Y=0. Blocks that cross the theta-seam get **cut**.
- For 1-4/2-2, staggered K Y **spreads** blocks vertically; crossing patterns differ and may be less damaging than 3-1’s **systematic** top-seam crossing.

So **continuous + no wraparound** likely **amplifies seam-cutting** for 3-1.

---

## 4. **Pattern confidence and classifier output**

- **pattern_type**: 3-1 → `continuous`, confidence **0.25**; 1-4/2-2 → `simple_staggered`, confidence **0.7**.
- **pattern_gt** (3-1): `6seg_constant`, confidence **0.83**.

Classification is **less confident** for 3-1 as “continuous,” and GT suggests a **constant** layout. Mismatch or ambiguity can affect downstream choices (e.g. detection strategy, fallbacks). Not necessarily the main cause, but it suggests 3-1 is **less clear-cut** than 1-4/2-2.

---

## 5. **Per-class IoU: where 3-1 fails**

| Class      | 1-4   | 2-2   | 3-1   |
|------------|-------|-------|-------|
| Background | 0.829 | 0.858 | 0.708 |
| K-block    | 0.485 | 0.621 | **0.267** |
| B1-block   | 0.733 | 0.797 | 0.666 |
| A1-block   | 0.741 | 0.810 | 0.649 |
| A2-block   | 0.087 | 0.725 | **0.331** |
| A3-block   | 0.757 | 0.820 | **0.238** |
| B2-block   | 0.748 | 0.798 | **0.339** |

3-1 is worst on **K**, **A2**, **A3**, **B2**. A2 is often **at or near the theta-seam**; K drives all block placement. Systematic seam clipping and **weaker depth boundaries** would hit exactly these classes.

---

## Summary: most plausible causes

1. **Theta-seam clipping**: Continuous layout + blocks below K + no wraparound → **systematic** crossing of the top seam and **clamping**, cutting blocks (especially A2, and indirectly K/A3/B2).
2. **Sparser depth map**: **Fewer valid boundary pixels** and **~53%** row coverage vs **~73–76%** for 1-4/2-2 → worse input for SAM.
3. **Fewer rings (6 vs 10)**: Less vertical sampling → **fewer** SAM prompts and **coarser** aggregation.
4. **Different h-geometry**: Shorter h-span, different aspect ratio, larger ring spacing → template placement and camera geometry **less well matched** than 1-4/2-2.

**Recommended next steps**

- Test **selective wraparound** (e.g. `sam_continuous`) **only** for 3-1, or **A2-only** wraparound, to reduce seam clipping without changing layout.
- **Improve depth-map coverage** for 3-1 (e.g. enhancing/outlier config, resolution) so boundaries are denser.
- **Revisit K-position placement** (including GT-derived two-band setup) and **ring count** (e.g. whether 6 rings are sufficient or we need different h-sampling).
