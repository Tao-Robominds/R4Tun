#!/usr/bin/env python3
"""
Build a directed graph of detection parameters and intermediates (data-flow).
Rank parameters by how many other nodes they affect (downstream reachability).

Usage:
  python methods/plans/scripts/detection_param_graph.py [--output path]

Output: writes detection_critical_params_graph.md (or --output path) with
  - Graph summary (nodes, edges)
  - Params ranked by downstream_impact (number of nodes reachable downstream)
  - Optional: same technique can be used for preprocessing if we add that graph.
"""

import argparse
import os
import sys

try:
    import networkx as nx
except ImportError:
    print("networkx required: pip install networkx", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# Detection pipeline data-flow (from 2_detection.py)
# Nodes: params (lowercase with _) and intermediates (no leading param name)
# Edge A -> B means "A affects B" (B depends on A)
# -----------------------------------------------------------------------------

def build_detection_graph():
    G = nx.DiGraph()

    # Input
    G.add_node("depth_map_outlier", kind="input")

    # Edge pipeline: binary_threshold, canny_*, dilation_* -> combined -> dilated
    G.add_edge("binary_threshold", "binary_image")
    G.add_edge("depth_map_outlier", "binary_image")
    G.add_edge("canny_low", "canny_edges")
    G.add_edge("canny_high", "canny_edges")
    G.add_edge("depth_map_outlier", "canny_edges")
    G.add_edge("binary_image", "combined_edges")
    G.add_edge("canny_edges", "combined_edges")
    G.add_edge("dilation_kernel_size", "dilated_edges")
    G.add_edge("dilation_iterations", "dilated_edges")
    G.add_edge("combined_edges", "dilated_edges")

    # Hough: dilated_edges -> line sets
    G.add_edge("dilated_edges", "lines_oblique")
    G.add_edge("hough_threshold", "lines_oblique")
    G.add_edge("hough_min_length", "lines_oblique")
    G.add_edge("hough_max_gap", "lines_oblique")

    G.add_edge("dilated_edges", "lines_horizontal")
    G.add_edge("hough_horizontal_threshold", "lines_horizontal")
    G.add_edge("hough_horizontal_min_length", "lines_horizontal")
    G.add_edge("hough_horizontal_max_gap", "lines_horizontal")

    G.add_edge("dilated_edges", "lines_vertical")
    G.add_edge("hough_vertical_threshold", "lines_vertical")

    # Filter oblique by angle -> positive_lines, negative_lines
    G.add_edge("lines_oblique", "positive_lines")
    G.add_edge("lines_oblique", "negative_lines")
    G.add_edge("angle_oblique_min", "positive_lines")
    G.add_edge("angle_oblique_max", "positive_lines")
    G.add_edge("angle_oblique_min", "negative_lines")
    G.add_edge("angle_oblique_max", "negative_lines")
    G.add_edge("angle_min_oblique_deg", "positive_lines")
    G.add_edge("angle_min_oblique_deg", "negative_lines")
    G.add_edge("max_line_length_factor", "positive_lines")  # via max_line_length_px
    G.add_edge("max_line_length_factor", "negative_lines")

    # Horizontal filter
    G.add_edge("lines_horizontal", "horizontal_lines")
    G.add_edge("horizontal_angle_tolerance", "horizontal_lines")

    # Vertical -> x_bounds (ring boundaries)
    G.add_edge("lines_vertical", "merged_vertical")
    G.add_edge("merge_distance_threshold", "merged_vertical")
    G.add_edge("merged_vertical", "x_bounds")

    # line_data (bundle used by K detection and regulator)
    G.add_edge("positive_lines", "line_data")
    G.add_edge("negative_lines", "line_data")
    G.add_edge("horizontal_lines", "line_data")
    G.add_edge("lines_vertical", "line_data")
    G.add_edge("x_bounds", "line_data")

    # K detection (DBSCAN)
    G.add_edge("line_data", "k_positions_raw")
    G.add_edge("eps", "k_positions_raw")
    G.add_edge("subdivision_threshold", "k_positions_raw")
    G.add_edge("max_subdivisions", "k_positions_raw")
    G.add_edge("k_expected_height_px", "k_positions_raw")

    # Regulator
    G.add_edge("k_positions_raw", "k_positions")
    G.add_edge("line_data", "k_positions")
    G.add_edge("k_expected_height_px", "k_positions")
    G.add_edge("reg_target_gap_frac", "k_positions")
    G.add_edge("reg_gap_tolerance", "k_positions")
    G.add_edge("reg_blend_weight", "k_positions")
    G.add_edge("reg_max_det_line_dist_frac", "k_positions")
    G.add_edge("merge_close_fraction", "k_positions")
    G.add_edge("max_k_gap_factor", "k_positions")
    G.add_edge("angle_oblique_min", "k_positions")
    G.add_edge("angle_oblique_max", "k_positions")

    # Final output
    G.add_edge("k_positions", "boundaries")
    G.add_edge("x_bounds", "boundaries")
    G.add_edge("line_data", "boundaries")
    G.add_node("per_ring_offsets", kind="param")  # not in JSON scalar list; affects boundaries
    G.add_edge("per_ring_offsets", "boundaries")

    return G


def is_param(name: str) -> bool:
    """True if node is a tunable param (typically snake_case, not an intermediate)."""
    intermediates = {
        "depth_map_outlier", "binary_image", "canny_edges", "combined_edges", "dilated_edges",
        "lines_oblique", "lines_horizontal", "lines_vertical", "positive_lines", "negative_lines",
        "horizontal_lines", "merged_vertical", "x_bounds", "line_data", "k_positions_raw",
        "k_positions", "boundaries",
    }
    return name not in intermediates


def main():
    ap = argparse.ArgumentParser(description="Detection parameter dependency graph; rank by downstream impact.")
    ap.add_argument("--output", "-o", default=None, help="Output .md path (default: stdout or detection_critical_params_graph.md)")
    args = ap.parse_args()

    G = build_detection_graph()

    # Downstream impact: for each node, count how many nodes are reachable downstream (descendants)
    param_nodes = [n for n in G.nodes() if is_param(n)]
    impact = {}
    for n in G.nodes():
        try:
            descendants = nx.descendants(G, n)
            impact[n] = len(descendants)
        except Exception:
            impact[n] = 0

    # Rank params by impact (then by out-degree as tie-breaker)
    out_deg = dict(G.out_degree())
    ranked = sorted(
        param_nodes,
        key=lambda p: (-impact.get(p, 0), -out_deg.get(p, 0), p),
    )

    lines = [
        "# Detection parameter dependency graph (networkx)",
        "",
        "Data-flow: param/intermediate → downstream. Built from `agents/irregular/2_detection/2_detection.py`.",
        "",
        "## Graph summary",
        "",
        f"- Nodes: {G.number_of_nodes()}",
        f"- Edges: {G.number_of_edges()}",
        f"- Params (tunable): {len(param_nodes)}",
        "",
        "## Params ranked by downstream impact",
        "",
        "Downstream impact = number of nodes reachable from this param (how many others it affects).",
        "",
        "| Rank | Parameter | Downstream impact | Out-degree |",
        "|------|-----------|-------------------|------------|",
    ]
    for i, p in enumerate(ranked, 1):
        lines.append(f"| {i} | `{p}` | {impact.get(p, 0)} | {out_deg.get(p, 0)} |")

    lines.extend([
        "",
        "## Intermediates (not tuned)",
        "",
        "| Node | Downstream impact |",
        "|------|-------------------|",
    ])
    for n in sorted(G.nodes()):
        if not is_param(n):
            lines.append(f"| {n} | {impact.get(n, 0)} |")

    text = "\n".join(lines)
    if args.output:
        outpath = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        with open(outpath, "w") as f:
            f.write(text)
        print(f"Wrote {outpath}")
    else:
        print(text)


if __name__ == "__main__":
    main()
