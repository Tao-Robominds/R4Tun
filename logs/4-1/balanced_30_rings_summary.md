# Balanced 30 Rings: Data Summary and Variety Explained

## 1. What the data is

**Source:** Tunnel point clouds from irregular (8-segment) tunnels (tunnel families 4 and 5). Each point has `x, y, z, intensity, segment, ring`.

**Ring catalog:** We build a catalog of all rings from subset files in `data/subsets/*.txt`. For each ring we compute: number of points, angular coverage, ellipse fit, walking order (cyclic segment order), K angle and K span, and related metrics. Output: `data/subsets/ring_catalog.csv`.

**Balanced 30:** From the catalog we select **30 rings** that (a) have full angular coverage, (b) contain the key segment K, and (c) are not no_K/BG-only. Selection is **balanced** so that the 30 rings spread across several dimensions (see below). At least 5 rings come from tunnel 5; the rest from tunnel 4. These 30 are exported as individual point-cloud files in `data/subsets/rings/{tunnel_id}-{ring_id}.txt` for preprocessing and evaluation.

So the “random” aspect is limited: we do **not** pick 30 rings at random. We pick them so that **variety** in the sample is controlled and interpretable.

---

## 2. What “variety” means

**Variety** = the dimensions we use to spread the 30 rings so the set is **representative** of different conditions (scan density, K position, K size). The selection algorithm tries to balance counts across these dimensions.

| Dimension | Meaning | Why it varies |
|-----------|--------|----------------|
| **Density** | How many points the ring has (sparse / low / medium / dense). Bins: &lt;10k = sparse, &lt;50k = low, &lt;200k = medium, ≥200k = dense. | Subset files differ in length and sampling; rings from different tunnels or positions have very different point counts. |
| **K quadrant** | Where the K segment sits on the ring in angle: q0 = 0–90°, q1 = 90–180°, q2 = 180–270°, q3 = 270–360°. | In staggered tunnels, K shifts from ring to ring; quadrant tells us “which side” of the cross-section K is on. |
| **K span tier** | Size of the K segment *relative to the catalog*: narrow = bottom 20% of K spans, normal = middle 60%, wide = top 20%. (Based on `k_span_deg` percentiles across all irregular rings.) | K can be a small key or span most of the ring; the tier is a distribution-based category, not a fixed angle. |

So in the **Variety in this set** table we report, for the 30 chosen rings:

- **Density:** Counts in sparse / low / medium / dense (e.g. 8, 8, 7, 7).
- **K quadrant:** Counts in q0–q3 (e.g. 8, 8, 7, 7).
- **K span tier:** Counts in narrow / normal / wide (e.g. 10, 10, 10).

That is the **meaning** of variety: we are describing how the 30 rings are distributed along these three axes.

---

## 3. Why we have this variety

We want the 30 rings to be a **small but representative sample** for:

1. **Preprocessing and segmentation:** So we test on rings that differ in point density (sparse vs dense), K position (all quadrants), and K size (narrow vs wide). If we only picked dense rings with K in one quadrant, we would not know how the pipeline behaves on sparse or differently oriented rings.

2. **Fair comparison across conditions:** Balancing density, quadrant, and K span tier avoids a set that is dominated by one type (e.g. all dense, or all “K in quadrant 1”). That makes downstream metrics (e.g. mIoU) more interpretable across scan quality and geometry.

3. **Tunnel 4 and 5:** We enforce at least 5 rings from tunnel 5 so that both irregular tunnel families are represented, not only tunnel 4.

So **we have variety** so that the 30 rings are not “random” in the sense of arbitrary, but **deliberately spread** across density, K position, and K size (and across tunnels), to support robust evaluation and tuning.

---

## 4. Quick reference

- **Table:** `logs/4-1/balanced_30_rings_table.md` (tunnel_id, ring_id, density, K_quadrant, K_span_tier, K_span_deg, n_points).
- **Catalog:** `data/subsets/ring_catalog.csv`.
- **Ring point clouds:** `data/subsets/rings/{tunnel_id}-{ring_id}.txt`.
- **Selection logic:** `data/subsets/extract_sample_rings.py` (`select_balanced_irregular`), export: `data/subsets/export_balanced_30_rings.py`.
