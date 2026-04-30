# Walking order: boundary-adjacency validation

After replacing centroid-angle ordering with boundary-adjacency (sort points by angle, first occurrence of each segment = cyclic order):

| Metric | Before (centroid) | After (boundary) |
|--------|-------------------|-------------------|
| Unique cyclic-normalized (K first) | 257 | 169 |
| Unique cyclic+mirror-normalized | 187 | 140 |
| Top template count | 7 | 16 |

Rings now cluster more: top cyclic template has 16 rings (was 7). Remaining variation may be tunnel-specific (4-x vs 5-x), boundary noise, or mirror. Catalog rebuilt: `data/subsets/ring_catalog.csv`.
