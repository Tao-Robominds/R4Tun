# GT segment centres and depth-map annotation scripts

These scripts build **all_segments_gt.csv** and **depth_map_annotated.png** from pipeline outputs (complex staggered tunnels 4-1, 5-1).

## 1. Build all_segments_gt.csv

**Script:** `scripts/build_all_segments_gt.py`  
**Implementation:** `p4tun/build_all_segments_gt.py`

Builds `data/<tunnel_id>/all_segments_gt.csv` (Ring, Block, X, Y, quality) from:

- **GT centroids:** `unwrapped.csv` (h, theta, segment, ring) — one row per (segment, ring) with mean (h, theta) in depth-map pixel space.
- **Grid:** Prefer `depth_map_grid.json` (saved at depth-map build time); else fit unwrapped extent into depth map shape so every block lies inside the image.

**Requirements:** Preprocessing must have been run so `unwrapped.csv`, depth map (or shape), and optionally `depth_map_grid.json` exist under `data/<tunnel_id>/`.

**Usage (from repo root):**

```bash
python scripts/build_all_segments_gt.py <tunnel_id> [--data-dir data] [--output all_segments_gt.csv]
```

Example:

```bash
python scripts/build_all_segments_gt.py 4-1
```

## 2. Plot depth map with K positions (annotation map)

**Script:** `scripts/plot_depth_map_k_annotated.py`  
**Implementation:** `p4tun/plot_depth_map_k_annotated.py`

Draws the depth map and overlays **only K** positions from `all_segments_gt.csv` (red circles + labels R0K..R6K), saves `data/<tunnel_id>/depth_map_annotated.png`.

**Requirements:** `all_segments_gt.csv` and `depth_map.png` (or `depth_map_outlier.npy`) in `data/<tunnel_id>/`.

**Usage (from repo root):**

```bash
python scripts/plot_depth_map_k_annotated.py [--tunnel 4-1] [--data-dir data]
```

Example:

```bash
python scripts/plot_depth_map_k_annotated.py --tunnel 4-1
```

## One-shot: build GT then plot

```bash
python scripts/build_all_segments_gt.py 4-1
python scripts/plot_depth_map_k_annotated.py --tunnel 4-1
```

Outputs:

- `data/4-1/all_segments_gt.csv`
- `data/4-1/depth_map_annotated.png`
