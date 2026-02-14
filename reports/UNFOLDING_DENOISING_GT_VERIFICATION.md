# Unfolding & Denoising vs Ground Truth — Verification

Script: `python -m p4tun.verify_unfolding_denoising_gt 1-4 2-2 3-1 --data-dir data`

---

## 1. Unfolding

### Ring count (algorithm vs GT)

| Tunnel | GT ring range | GT ring count | Algo ring_count | Match? |
|--------|----------------|---------------|------------------|--------|
| 1-4    | 188–198        | 11            | 10               | **No** |
| 2-2    | 128–137        | 10            | 10               | Yes    |
| 3-1    | 1–6            | 6             | 6                | Yes    |

- **1-4**: Algorithm **underestimates** rings (10 vs GT 11). Likely unfolding/ring_spacing issue.
- **2-2, 3-1**: Ring count matches GT.

### h-span vs raw scan extent

| Tunnel | h span (unfolded) | Raw z span |
|--------|-------------------|------------|
| 1-4    | 12.06             | 5.21       |
| 2-2    | 12.10             | 5.99       |
| 3-1    | **7.49**          | **5.73**   |

- Raw z span is **similar** across tunnels (5.2–6.0).
- **3-1** has **much smaller h span** (7.49) than 1-4/2-2 (12.06, 12.10).
- So **unfolding gives a shorter along-tunnel extent for 3-1** than for 1-4/2-2, despite similar raw extent ⇒ **unfolding for 3-1 is likely wrong or inconsistent** (centerline, ring_spacing, or h mapping).

### h vs GT ring

- GT ring vs h correlation ≈ −0.99 for all: **higher ring → lower h** (ordering consistent).
- Monotonicity: median h per ring is monotonic in ring (up to numerical noise).

**Unfolding verdict:**
- **3-1**: **Incorrect** — h-span too short vs 1-4/2-2 and vs raw z span. Suggests better unfolding (centerline, extent, ring_spacing) needed for 3-1.
- **1-4**: **Incorrect** — ring count mismatch (10 vs 11). Adjust ring_spacing / extent so algo ring_count matches GT.

---

## 2. Denoising

### Retention by GT segment (%)

| Segment | 1-4   | 2-2   | 3-1    |
|---------|-------|-------|--------|
| 0 (bg)  | 13.2  | 12.3  | 22.5   |
| 1 (K)   | 98.3  | 99.1  | 98.8   |
| 2 (B1)  | 98.7  | 98.2  | **93.1** |
| 3 (A1)  | 98.1  | 97.3  | **85.8** |
| 4 (A2)  | 92.0  | 96.0  | 99.9   |
| 5 (A3)  | 97.9  | 98.0  | **92.7** |
| 6 (B2)  | 99.2  | 98.4  | 99.2   |

- **3-1** has **lower retention** for **B1, A1, A3** (86–93%) than 1-4/2-2 (97–99%).
- **Over-removal** of these boundaries in 3-1 ⇒ fewer boundary points ⇒ **sparser depth map**, consistent with observed 3-1 depth coverage (~53% vs ~73–76%).

### Retention by GT ring

- **End rings** generally have **lowest** retention (e.g. 1-4 ring 198 ~30.7%; 3-1 rings 1 and 6 ~59–62%).
- 3-1 ring 1: 62.2%, ring 6: 59.5% — **boundary rings** most affected.

### Radius (r)

| Tunnel | r min | r max |
|--------|-------|-------|
| 1-4    | 1.80  | 2.88  |
| 2-2    | 1.91  | 3.94  |
| 3-1    | **0.12** | 3.20  |

- **3-1** has **r_min ≈ 0.12** (points very close to axis). Denoising uses `radius_min = 2.8` ⇒ those points are **radius-filtered out**. Could remove valid structure or change boundary composition.

**Denoising verdict:**
- **3-1**: **Incorrect** — (1) Over-removal of B1, A1, A3 vs GT; (2) end rings under-retained; (3) strong radius filter with r_min 2.8 vs actual r down to 0.12. Suggests **tuning denoising** (and possibly radius filter) for 3-1 to preserve boundary segments and end rings.

---

## 3. Summary

| Stage     | 1-4       | 2-2   | 3-1        |
|-----------|-----------|-------|------------|
| Unfolding | Ring count wrong | OK   | **h-span too short** |
| Denoising | —         | —     | **Over-removal B1/A1/A3, end rings; r_min vs data** |

**Next steps:**  
Fix **unfolding** for 3-1 (h-span, centerline, ring_spacing) and **denoising** for 3-1 (retention by segment/ring, radius filter vs actual r). Re-run verification after changes.
