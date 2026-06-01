"""Proxy feature catalog for Stage A candidate scoring."""
from __future__ import annotations

PRE7_FEATURES = [
    "finite_ratio",
    "row_nonempty_ratio",
    "largest_empty_vertical_gap_frac",
    "pre_theta_coverage_pct",
    "pre_depth_map_valid_pixels",
    "pre_point_retention_pct",
    "pre_depth_map_max_empty_row_run",
]

SEG_REPLAY_FEATURES = [
    "seg_segment_type_completeness",
    "seg_ring_completeness_avg",
    "seg_mask_coverage_pct",
    "seg_k_size_ratio",
    "seg_block_size_variance_ratio",
    "seg_groove_score",
    "seg_ready_for_evaluation",
]
