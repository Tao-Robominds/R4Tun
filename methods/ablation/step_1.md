# Step 1 — BO calibration ring selection

Six diversity-selected labelled rings for design-time BO / proxy calibration. All rings are disjoint from `data/held-out/`.

**Pass criterion:** Step 0 ceiling mIoU ≥ 0.85.

---

## Final BO panel

| Slot | Ring | Segments | Ceiling mIoU |
|------|------|:--------:|-------------:|
| Dense 6-block | 1-5/r271 | 6 | 0.895 |
| Medium 6-block | 1-1/r20 | 6 | 0.883 |
| Sparse 6-block | 1-4/r206 | 6 | 0.935 |
| Medium 7-block | 5-5/r258 | 7 | 0.875 |
| Sparse 7-block | 4-6/r283 | 7 | 0.869 |
| Partial / irregular | 4-1/r116 | 7 | 0.905 |

**6 / 6 pass**

| Stat | Value |
|------|------:|
| Mean | 0.894 |
| Min | 0.869 (4-6/r283) |
| Max | 0.935 (1-4/r206) |
