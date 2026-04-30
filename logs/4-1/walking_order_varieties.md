# Walking order: why so many varieties?

## What we measure

- **Walking order** = full sequence of 8 segment names by **centroid angle** (e.g. `K-A2-A3-A4-B2-BG-A1-B1`).
- One string per ring; any change in order gives a different "variety".

## Numbers (irregular rings only)

| Metric | Value |
|--------|--------|
| Irregular rings | 438 |
| Unique walking orders | 400 |
| Orders that appear in 2+ rings | 31 |
| Rings with a "unique" order (only one ring has it) | 369 |
| Max frequency of any order | 4 rings |

So most rings have an order that no other ring shares.

## Is it normal?

**Yes.** Reason: we use the **full permutation** of 8 segments. Small changes in geometry (ring shape, segment size, noise) shift centroid angles; when two segments sit close in angle, a tiny shift **swaps** their order and produces a new string. So:

- **Fine-grained (current)**: 8 segments → up to 8! orderings; small noise → many distinct orders → **400 varieties**.
- **Coarse (before/after K only)**: only 42 distinct (before_K, after_K) pairs, and they **are** shared (e.g. 25 rings with `BG→K→B2`, 22 with `BG→K→B1`).

## Implication

- If you need **"a few walking orders shared among rings"**: use a **coarser** descriptor, e.g. (segment before K, segment after K), or "first block after K" (already in `walking_order_family`), or K quadrant + first-after-K. Then you get dozens of patterns, not hundreds.
- If you need **exact** centroid order for some algorithm, then many varieties are expected and not a bug.

## Files

- `data/subsets/extract_sample_rings.py`: `walking_order` = full order; `walking_order_family()` = first block after K.
- Catalog: `data/subsets/ring_catalog.csv`.
