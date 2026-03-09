# 03 Upgrade Solution — Output

Canonical output for Step 03.  Compares the new non-GT pipeline against the
original `sam4tun` scripts, maps every challenge ID, and lists remaining
structural gaps.

## 1. Architecture Shifts from sam4tun

### A. Walk-order elimination

| Aspect | sam4tun (`4-1_detection.py` + `4-2_sam.py`) | New pipeline (`2_detection.py` + `segmentation.py`) |
|--------|----------------------------------------------|------------------------------------------------------|
| Detection output | K-only (`detected.csv`) | All 7 segments (`all_segments.csv`) via grouped offsets |
| Segmentation input | Recomputes A/B offsets from K inside SAM loop | Reads pre-computed `all_segments.csv` |
| Walk order | Fixed sequential: K→B1→A1→A2→A3→A4→B2 | None — each segment is independent |
| Cascading errors | A/B errors compound from K | Each block positioned independently from K + offset |

### B. Multi-method K detection → Combined fusion

| Aspect | sam4tun | New `2_detection.py` |
|--------|---------|----------------------|
| K method | Single: vertical/oblique line intersection | Combined: DBSCAN clusters + groove-pair crossings |
| Fallback | None | Banded (evenly-spaced ring bands) |
| Intrinsic quality metric | None | Groove alignment score (expand K → A/B, count groove crossings) |
| Per-ring handling | Uniform across rings | Per-ring band centre via `ring_offset` + `ring_spacing_px` |

### C. SAM replaced by geometric segmentation

| Aspect | sam4tun (`4-2_sam.py`) | New `segmentation.py` |
|--------|------------------------|-----------------------|
| Engine | SAM (GPU, ~30s/ring) | Polygon point-in-polygon (CPU, <1s total) |
| Template shape | Fixed rectangular crop + SAM mask | Trapezoid (K, B1, B2) / Rectangle (A blocks) |
| Wrap handling | None — crops fail at Y boundaries | Cylindrical wrap in template assignment |
| Overlap resolution | Last-writer wins (walk order) | Nearest-centre (distance-based) |
| GT dependency | `segment` column for fallback | None |

### D. Consolidated preprocessing

| Aspect | sam4tun (3 scripts) | New `1_preprocessing.py` |
|--------|---------------------|--------------------------|
| Scripts | `1_upfolding.py` + `2_denoising.py` + `3_enhancing.py` | Single `1_preprocessing.py` |
| Parameter files | Hardcoded per script | One JSON (`parameters_preprocessing.json`) |
| Removed | — | `classify_tunnel_pattern` (dead: kmeans2-based pattern classifier) |
| Removed import | — | `scipy.cluster.vq.kmeans2` |

## 2. Detailed Changes per File

### `1_preprocessing.py`

| Change | Lines saved |
|--------|-------------|
| Removed `classify_tunnel_pattern` function | ~84 |
| Removed call site + `pattern_type.json` save | ~4 |
| Removed `from scipy.cluster.vq import kmeans2` | 1 |

### `2_detection.py`

| Removed | Reason | Lines saved |
|---------|--------|-------------|
| `mm_to_px` | Dead code | ~8 |
| `compute_ring_centers` | Dead code (ring centres now from `ring_offset + ring_spacing_px`) | ~64 |
| `_line_crossing_y_at_x` (top-level) | Combined has own inline copy | ~7 |
| `_cluster_1d_gap` | Dead code | ~22 |
| `calculate_k_positions_complex_staggered` | Standalone DBSCAN; superseded by combined | ~140 |
| `calculate_k_positions_groove_pair` | Standalone groove-pair; superseded by combined | ~216 |
| Method selector in `run_detection` | Always uses `combined` now | ~14 |
| `AgglomerativeClustering` import | No longer needed | 1 |
| **Total** | | **~472** |

Other changes:
- `dilation_kernel_size`, `dilation_iterations`, `canny_low`, `canny_high` wired to `params.get()` with `DEFAULT_*` fallbacks.
- `k_detection_method` parameter removed; always calls `calculate_k_positions_combined`.
- `expansion_method` check removed (only `grouped_offsets` exists).
- Return type annotation corrected: `-> Tuple[pd.DataFrame, pd.DataFrame]`.
- 57 print statements consolidated to ~10.

### `segmentation.py` (new file, replaces 6 experimental files)

| Source | What was taken | Lines |
|--------|---------------|-------|
| `3_template_geometric.py` | Template shapes, label map builder, main pipeline | ~200 |
| `3_sam.py` | `project_back_to_point_cloud`, `compute_block_to_label_map` | ~70 |
| **Total new file** | | **~270** |

Removed from `3_template_geometric.py`:
- Dynamic `importlib` import of `3_sam.py` (replaced by inlined functions).
- GT-dependent unmapped fallback (`if "segment" in updated_df.columns`).

### Deleted files (6 experimental segmentation variants)

| File | Size | Reason |
|------|------|--------|
| `3_sam.py` | 50 KB | SAM-based; functions inlined into `segmentation.py` |
| `3_sam_wrap.py` | 53 KB | Per-ring SAM; GT-dependent (`gt_angular_boundaries.json`) |
| `3_sam_wrap_a.py` | 57 KB | SAM wrap variant A |
| `3_sam_wrap_b.py` | 58 KB | SAM wrap variant B |
| `3_geometric.py` | 14 KB | Bounding-box geometric; superseded by template |
| `3_template_geometric.py` | 12 KB | Superseded by new `segmentation.py` |
| **Total removed** | **~244 KB** | |

## 3. Challenge Coverage

| ID | Challenge | Addressed by | Method |
|----|-----------|-------------|--------|
| A1 | Diameter 5.5→7.5 | Preprocessing | `tunnel_diameter` param |
| A2 | Ring spacing 1.2→1.816 | Preprocessing | `ring_spacing` param |
| A3 | Elliptical cross-section | Stable | — |
| A4 | MBR axis | Stable | — |
| A6/A7 | Ring count ≠ slicing | Preprocessing | `num_slicing_planes` decoupled |
| B1 | Surface band r | Preprocessing | `radius_min/max` params |
| B2 | Gradient 0.2 | Preprocessing | `gradient_threshold` param |
| C5 | High-density band | Preprocessing | `outlier_high_density_ring_*` params |
| D1 | Vertical lines = rings | Detection | Per-ring band in combined |
| D2 | Oblique ±6-9° | Detection | `angle_pos/neg_min/max` params |
| D4 | K height fixed | Detection | Derived from `tunnel_diameter` |
| D5 | AB height fixed | Detection | Derived from `tunnel_diameter` |
| D6 | Ring spacing 1.2 in det | Detection | Real `ring_spacing` used |
| D8 | K evenly spaced | Detection | Per-ring K in combined |
| E1 | 6 segments → 7 | Detection + Seg | 7-block support (`SEGMENT_COUNT=7`) |
| E2 | Width=1200mm | Segmentation | Template params per type |
| E4/E5 | Fixed walk order | Detection | `all_segments.csv` (no walk) |
| E6/E7 | Fixed template | Segmentation | Per-type configurable sizes |
| E8 | One group offset | Detection | `group_offsets` (12D, BO-tunable) |
| E9 | SAM fails | Segmentation | Geometric only (no SAM) |
| E11 | Only pred=7 | Segmentation | pred in [0, 7] updated |

### Remaining structural gaps (deferred)

| Gap | Why | Potential fix |
|-----|-----|---------------|
| E8 per-ring | `group_offsets` shared across rings; A2/A3 ~800 px error | Groove-based direct A/B detection |
| E6/E7 per-ring | One size per block type; K varies 579–13,438 pts | Per-ring scaling from centroid distances |
| E11 unmapped | 18k points without pixel mapping stay pred=0 | Retroactive projection in preprocessing |

## 4. Non-GT Compliance

The final pipeline has **zero GT dependencies**:

- No reads of `segment` or `ring` columns for any assignment logic.
- No `gt_angular_boundaries.json`.
- No SAM model.
- The `only_label.csv` convenience export (GT vs pred comparison) is gated on
  `if "segment" in df.columns` and does not affect segmentation results.

## 5. Fix evaluation.py

| Change | Detail |
|--------|--------|
| `CLASS_NAMES_7` ordering | Fixed to match GT and segmentation: K=1, B1=2, A1=3, A2=4, A3=5, A4=6, B2=7. Previously had B2 at position 3. |
| NaN GT handling | Enhanced/upsampled points (no GT label) now filtered in `load_data` with explicit NaN check and accurate log message, instead of relying on `<= max_class` which silently dropped NaN rows and printed misleading "class > 7" count. |

## 6. Verification Run: Tunnel 5-1

Full pipeline executed on `data/irregular/5-1/` (ring_count=7, preprocessing
outputs already existed).

### Detection

```
Lines: +249 -96 H367 V37
K positions: 7 found (dbscan: 4, groove_pair: 3)
Groove alignment: 73.5/84 (87.5%)
Segments: 7 K → 49 total
```

### Segmentation

```
Segments: 49, Points: 4,980,447
Output: data/irregular/5-1/final.csv
```

### Evaluation (baseline, after evaluation.py fix)

```
Points with GT: 1,504,524 (3,475,923 enhanced/upsampled excluded)
OA 0.225  F1 0.208  mIoU 0.130
```

| Class | IoU |
|-------|-----|
| Background | 0.432 |
| K-block | 0.192 |
| B1-block | 0.204 |
| A1-block | 0.052 |
| A2-block | 0.051 |
| A3-block | 0.034 |
| A4-block | 0.067 |
| B2-block | 0.008 |

**Notes**: The mIoU is low because group_offsets and template sizes use defaults
(not BO-tuned for 5-1). This is expected — the verification confirms the
workflow runs end-to-end without errors and without GT dependency. Tuning
happens in later steps (04–07).

## 7. Generality Audit

All four pipeline files were audited to ensure they work for **any irregular
tunnel** (not just 5-1 or 4-1). The following hardcoded values were removed:

| File | Issue | Fix |
|------|-------|-----|
| `segmentation.py` | `SEGMENT_COUNT = 7` hardcoded | Derived from `all_segments.csv` unique block names |
| `detection.py` | `EXPANSION_BLOCKS = ['B1','B2','A1','A2','A3','A4']` in 2 places | New `_derive_expansion_blocks()` extracts blocks from `group_offsets` keys; falls back to defaults based on `segment_count` |
| `detection.py` | `groove_max = 12.0 * ring_count` hardcoded (assumed 6 blocks) | Now `2.0 * len(expansion_blocks) * ring_count` |
| `segmentation.py` | `[0, 7]` magic number for surface pred | Named constant `SURFACE_PRED = 7` |
| `preprocessing.py` | `df['pred'] = 7` magic number | Named constant `SURFACE_PRED = 7` |
| `evaluation.py` | Docstring said "COMPLEX STAGGERED tunnels (4-1, 5-1)" | Generalized to "irregular tunnels" |

### Methodology compliance

The pipeline is now **tunnel-agnostic**: all tunnel-specific values come from
parameter JSON files. The workflow is:

1. **Understand on 5-1**: run pipeline with default params, identify which
   parameters need tuning (Step 04: Parameter Inventory).
2. **Tune on 4-1**: run BO to find optimal parameter values, prove the
   solutions are adaptable without touching code (Steps 05–07).
3. **Deploy on new tunnels**: supply a new parameter JSON, run the same code.
