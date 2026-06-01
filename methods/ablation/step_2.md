# Step 2 — BO search space (layout recovery only)

BO searches **layout-recovery variables only** — not preprocessing or full SAM4Tun.

**Implementation:** `bo/lib/search_space.py`, `bo/lib/layout_bo.py`

---

## Search variables

| Ablation slot | Parameter | Log column |
|---------------|-----------|------------|
| K position | `k_y_positions` | `k_y` |
| A/B offsets | `per_ring_offsets` | `per_ring_offsets` |
| Oblique-line Hough threshold | `hough_threshold` | `hough_oblique_threshold` |
| Horizontal-line Hough threshold | `hough_horizontal_threshold` | `hough_horizontal_threshold` |
| Line merge distance | `merge_distance_threshold` | `line_merge_distance` |
| Line snapping tolerance | `single_ring_visual_slot_snap_px` | `line_snap_tolerance_px` |
| Segmentation padding / crop | `slot_inset_y` | `segmentation_slot_inset_y` |

## Fixed (not searched)

- Preprocessing parameters
- Full SAM4Tun / binary threshold / angle gates
- `r_surface_min` — fixed per ring from ceiling reference (`r_surface_min_fixed` in logs)

## Search dimension

`1 + segment_count + 5` (e.g. **13** for 7-block, **12** for 6-block)

## Log outputs per ring

`logs/<run_id>/<tunnel>/r<ring>/`:

- `search_space.json` — bounds and variable list
- `bo_trials.csv` — all layout parameters per trial
- `best_bo_trial.json` — best layout params + offsets
