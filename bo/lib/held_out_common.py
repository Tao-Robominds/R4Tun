"""Shared constants and paths for held-out Stage A proxy scoring."""
from __future__ import annotations

from lib.ceiling_gate import REPO_ROOT

HELD_OUT_ROOT = REPO_ROOT / "data" / "held-out"
A3_SLIM_MANIFEST = REPO_ROOT / "logs" / "bo_feature_enrichment_v1" / "PROXY_A3_SLIM_MANIFEST.json"
A3_V5_P11_MANIFEST = REPO_ROOT / "logs" / "bo_v5_proxy_v1" / "PROXY_P11_MANIFEST.json"

A3_SLIM_FEATURE_COLUMNS = [
    "feat_pre_row_nonempty_ratio",
    "feat_pre_largest_empty_vertical_gap_frac",
    "feat_pre_pre_depth_map_valid_pixels",
    "feat_pre_pre_depth_map_max_empty_row_run",
    "feat_intrinsic_n_reclassified_by_r_filter",
    "feat_intrinsic_arc_width_entropy",
    "param_k_y_frac",
    "param_hough_oblique_threshold",
]

P11_FEATURE_COLUMNS = [
    "feat_pre_row_nonempty_ratio",
    "feat_pre_pre_depth_map_valid_pixels",
    "feat_intrinsic_n_reclassified_by_r_filter",
    "feat_intrinsic_arc_width_entropy",
    "param_k_y_frac",
    "param_hough_oblique_threshold",
    "v5_balance_norm",
    "v5_geom_boundary_gap_cv",
    "v5_S_boundary",
    "seg_k_size_ratio",
    "seg_groove_score",
]

RELATIVE_MARGIN = 0.02
