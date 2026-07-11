# T3 Manual Param Tune Summary

**Target:** panel mean mIoU ≥ 0.60; K Y-spread &lt; 50 px per tunnel.

## Per-variant mIoU

| Variant | 3-1-1 | 3-1-2 | 3-1-3 | Mean | K-spread (3-1-1) | Pass ≥0.60? |
|---------|-------|-------|-------|------|------------------|-------------|
| base_v3 | 0.132 | — | — | 0.132 | 0 | ✗ |
| center_walk_312 | — | 0.547 | — | 0.547 | — | ✗ |
| center_walk_313 | — | — | 0.222 | 0.222 | — | ✗ |
| center_walk_313_mirror | — | — | 0.361 | 0.361 | — | ✗ |
| center_walk_313_nosnap | — | — | 0.361 | 0.361 | — | ✗ |
| consensus_tight | 0.132 | — | — | 0.132 | 0 | ✗ |
| cross_311_313 | — | — | 0.373 | 0.373 | — | ✗ |
| cross_311_313_snap | — | — | 0.369 | 0.369 | — | ✗ |
| cross_312_313 | — | — | 0.225 | 0.225 | — | ✗ |
| flip_on | 0.132 | — | — | 0.132 | 0 | ✗ |
| gap_wide | 0.494 | — | — | 0.494 | 8 | ✗ |
| geo_313 | — | — | 0.224 | 0.224 | — | ✗ |
| geo_313_flip | — | — | 0.361 | 0.361 | — | ✗ |
| hough_low | 0.582 | 0.180 | 0.156 | 0.306 | 0 | ✗ |
| hough_low_flip | 0.601 | 0.189 | 0.204 | 0.331 | 0 | ✗ |
| oracle_313_nosnap | — | — | 0.829 | 0.829 | — | ✗ |
| oracle_313_solo | — | — | 0.829 | 0.829 | — | ✗ |
| per_tunnel_313 | — | — | 0.222 | 0.222 | — | ✗ |
| per_tunnel_v3 | 0.582 | 0.248 | 0.162 | 0.331 | 0 | ✗ |
| t1_detect | 0.132 | — | — | 0.132 | 0 | ✗ |
| t2_best | 0.132 | — | — | 0.132 | 0 | ✗ |
| t2_detect | 0.132 | — | — | 0.132 | 0 | ✗ |

## Conclusion

**Target not met.** Best panel: **hough_low_flip** mean **0.331** (gap 0.269).
Best gate tunnel: **hough_low** on `3-1-1` mIoU **0.582** (K-spread 0 px). Panel limited by `3-1-2`/`3-1-3` K detection (spread 121–159 px). Lowering Hough to 40/40 fixes `3-1-1`; per-tunnel v3 detecting needed for siblings.

## Artifacts

- Results: `data/t3_tune/{variant}/{tunnel}/`
- Sweep logs: `logs/t3_tune/sweep_*.csv`

