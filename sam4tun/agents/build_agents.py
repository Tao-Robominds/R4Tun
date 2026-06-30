#!/usr/bin/env python3
"""Build sam4tun/agents/*.py from modular sources + parameter wiring."""

from __future__ import annotations

import os
import re
import textwrap

SAM4TUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(SAM4TUN, "agents")
REPO = os.path.dirname(SAM4TUN)

BOOTSTRAP = textwrap.dedent('''
    import sys
    import os
    import matplotlib
    matplotlib.use("Agg")

    _AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
    _SAM4TUN_ROOT = os.path.dirname(_AGENTS_DIR)
    for _p in (_AGENTS_DIR, _SAM4TUN_ROOT):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from pipeline_data import (
        parse_pipeline_args, load_stage_parameters, require_keys, resolve_param_file, setup_sam4tun_path,
    )
    setup_sam4tun_path()
    from helpers.pipeline_io import ensure_dir
    from helpers.pipeline_state import load_state, save_state
''').strip()


def write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)
    print(f"wrote {path} ({len(content)} bytes)")


def build_unfolding() -> None:
    src = open(os.path.join(SAM4TUN, "1_upfolding.py")).read()
    # strip modular header through bbox viz setup - keep from generate_slicing
    idx = src.find("def generate_slicing_planes_point_cloud")
    body = src[idx:]

    subs = [
        ("round(l / 1.2)", "round(l / slice_spacing_factor)"),
        ("abs(l - 1.2 * n)", "abs(l - slice_spacing_factor * n)"),
        ("abs(l - 1.2 * candidate_n)", "abs(l - slice_spacing_factor * candidate_n)"),
        ("delta = 0.005  # Thickness of slices / 2\n\n", ""),
        ("<= 4.5)", "<= vertical_filter_window)"),
        ("self.items = 999  # Number of iterations", "self.items = initial_iterations  # Number of iterations"),
        ("inliers = np.where(Z < 0.8 * delta)[0]", "inliers = np.where(Z < inlier_threshold_multiplier * delta)[0]"),
        (
            "class RANSAC:\n    def __init__(self, data, threshold, P, S, N):",
            "class RANSAC:\n    def __init__(self, data, threshold, P, S, N, initial_iterations=999, inlier_threshold_multiplier=0.8):",
        ),
        (
            "ransac = RANSAC(data=points_data, threshold=1.0, P=0.9, S=0.75, N=5)",
            "ransac = RANSAC(data=points_data, threshold=ransac_threshold, P=ransac_probability, "
            "S=ransac_inlier_ratio, N=ransac_sample_size, initial_iterations=ransac_initial_iterations, "
            "inlier_threshold_multiplier=ransac_inlier_threshold_multiplier)",
        ),
        ("degree = 3", "degree = polynomial_degree"),
        ("ring_count * 1210", "ring_count * num_samples_factor"),
        (
            "t_samples = np.linspace(-20, ring_count + 20, num_samples)",
            "t_samples = np.linspace(t_extrapolation_start, ring_count + t_extrapolation_end, num_samples)",
        ),
        ("batch_size = 1000000  # Adjust batch size based on memory constraints\n\n", ""),
        ("Parallel(n_jobs=12)", "Parallel(n_jobs=n_jobs)"),
        ("diameter = 5.5\n", ""),
        ("from tqdm.notebook import tqdm", "from tqdm.auto import tqdm"),
    ]
    for a, b in subs:
        body = body.replace(a, b)

    header = f'''#!/usr/bin/env python3
# Parameterized unfolding — algorithm from sam4tun/1_upfolding.py
# Deferred JSON: none
{BOOTSTRAP}

import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull
        from shapely.geometry import Polygon
        import cv2
        import random
        import time
        import math
        from tqdm.auto import tqdm
        from sklearn.linear_model import RANSACRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from numba import njit, prange
        import faiss
        from joblib import Parallel, delayed

        tunnel_id = parse_pipeline_args("unfolding")
        params = load_stage_parameters(tunnel_id, "unfolding")
        param_file = resolve_param_file(tunnel_id, "unfolding")
        expected_keys = [
            "delta", "slice_spacing_factor", "vertical_filter_window",
            "ransac_threshold", "ransac_probability", "ransac_inlier_ratio",
            "ransac_sample_size", "ransac_initial_iterations", "ransac_inlier_threshold_multiplier",
            "polynomial_degree", "num_samples_factor", "t_extrapolation_start", "t_extrapolation_end",
            "diameter", "batch_size", "n_jobs", "swap_tunnel_centers",
        ]
        require_keys(params, expected_keys, param_file)
        delta = params["delta"]
        slice_spacing_factor = params["slice_spacing_factor"]
        vertical_filter_window = params["vertical_filter_window"]
        ransac_threshold = params["ransac_threshold"]
        ransac_probability = params["ransac_probability"]
        ransac_inlier_ratio = params["ransac_inlier_ratio"]
        ransac_sample_size = params["ransac_sample_size"]
        ransac_initial_iterations = params["ransac_initial_iterations"]
        ransac_inlier_threshold_multiplier = params["ransac_inlier_threshold_multiplier"]
        polynomial_degree = params["polynomial_degree"]
        num_samples_factor = params["num_samples_factor"]
        t_extrapolation_start = params["t_extrapolation_start"]
        t_extrapolation_end = params["t_extrapolation_end"]
        diameter = params["diameter"]
        batch_size = params["batch_size"]
        n_jobs = params["n_jobs"]
        swap_tunnel_centers = params["swap_tunnel_centers"]

        paths = ensure_dir(tunnel_id)
        point_cloud_data = np.loadtxt(paths["input_txt"])
        print(point_cloud_data.shape)
        points_xyz = point_cloud_data[:, :3]
        intensity = point_cloud_data[:, 3]
        segment = point_cloud_data[:, 4].astype(int)
        ring = point_cloud_data[:, 5].astype(int)
        df_point_cloud = pd.DataFrame({{
            'x': points_xyz[:, 0], 'y': points_xyz[:, 1], 'z': points_xyz[:, 2],
            'intensity': intensity, 'segment': segment, 'ring': ring,
        }})

        points_2d_xoy = points_xyz[:, :2]
        convex_hull = ConvexHull(points_2d_xoy)
        convex_hull_points = points_2d_xoy[convex_hull.vertices]
        convex_polygon = Polygon(convex_hull_points)
        min_bounding_rect = convex_polygon.minimum_rotated_rectangle
        rect_vertices = np.array(min_bounding_rect.exterior.coords)[:-1]
        edges = [np.linalg.norm(rect_vertices[i] - rect_vertices[(i + 1) % 4]) for i in range(4)]
        short_edge_index = np.argmin(edges)
        center1 = (rect_vertices[short_edge_index] + rect_vertices[(short_edge_index + 1) % 4]) / 2
        center2 = (rect_vertices[(short_edge_index + 2) % 4] + rect_vertices[(short_edge_index + 3) % 4]) / 2
        if swap_tunnel_centers:
            center1, center2 = center2, center1
        vector = center2 - center1
        print(vector)

        plt.figure(figsize=(8, 8))
        sample_size = 10000
        indices = np.random.choice(len(points_2d_xoy), size=sample_size, replace=False)
        sampled_points = points_2d_xoy[indices]
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], s=1, c='blue', label='Projected Points')
        for simplex in convex_hull.simplices:
            plt.plot(points_2d_xoy[simplex, 0], points_2d_xoy[simplex, 1], 'k-')
        rect_x, rect_y = zip(*(np.array(min_bounding_rect.exterior.coords)))
        plt.plot(rect_x, rect_y, 'r-', label='Minimum Bounding Rectangle')
        plt.plot(center1[0], center1[1], 'go', label='Center 1 of Short Edge')
        plt.plot(center2[0], center2[1], 'mo', label='Center 2 of Short Edge')
        plt.arrow(center1[0], center1[1], vector[0], vector[1], head_width=1, head_length=1, fc='green', ec='green', label='Direction Vector')
        plt.xlabel('X-axis'); plt.ylabel('Y-axis')
        plt.title('Projected Point Cloud and Bounding Rectangle')
        plt.legend(); plt.axis('equal'); plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(paths["state"]), "projected_point_cloud_bbox.png"), dpi=150, bbox_inches='tight')

    ''')
    write(os.path.join(AGENTS, "unfolding.py"), header + body)


def _replace_root_bootstrap(src: str, stage: str, extra_imports: str = "") -> str:
    """Remove root agents bootstrap through first algorithm comment after param load."""
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# Algorithm") or line.startswith("def ") and "parse_pipeline" not in line:
            if line.startswith("# Algorithm"):
                break
        if line.startswith("def ") and i > 50:
            break
        i += 1
    algo = "".join(lines[i:])
    return algo


def build_from_root_agent(
    stage: str,
    root_name: str | None = None,
    extra_header: str = "",
    extra_imports: str = "",
    tail_patch: str = "",
) -> str:
    root_name = root_name or stage
    src = open(os.path.join(REPO, "agents", f"{root_name}.py")).read()
    # extract from expected_keys through end
    m = re.search(r"expected_keys\s*=\s*\[", src)
    if not m:
        raise ValueError(f"no expected_keys in agents/{root_name}.py")
    algo = src[m.start():]
    # remove root I/O at end for denoising etc - we'll append sam4tun I/O
    algo = re.sub(
        r"base_dir = resolve_output_base_dir.*?\n",
        "",
        algo,
        count=1,
    )
    algo = re.sub(
        r"unwrapped_file = os\.path\.join.*?\n",
        "",
        algo,
    )
    algo = re.sub(
        r"df_point_cloud = pd\.read_csv\(unwrapped_file\)\n",
        "",
        algo,
    )
    algo = re.sub(
        r"ring_count_file = os\.path\.join.*?\n",
        "",
        algo,
    )
    algo = re.sub(
        r"ring_count = int\(open\(ring_count_file.*?\n",
        "",
        algo,
    )
    algo = re.sub(
        r"print\(f\"Processing tunnel:.*?\n",
        "",
        algo,
        count=1,
    )
    # strip trailing save for denoising
    algo = re.sub(
        r"# Save results\n.*?df_point_cloud\.to_csv\(denoised_file.*?\n",
        "",
        algo,
        flags=re.DOTALL,
    )
    algo = re.sub(
        r"denoised_file = os\.path\.join\(base_dir.*?\n",
        "",
        algo,
    )
    algo = re.sub(
        r"os\.makedirs\(base_dir.*?\n",
        "",
        algo,
    )
    header = f'''#!/usr/bin/env python3
# Parameterized {stage} — algorithm from agents/{root_name}.py with sam4tun state I/O
{BOOTSTRAP}
{extra_imports}
tunnel_id = parse_pipeline_args("{stage}")
params = load_stage_parameters(tunnel_id, "{stage}")
param_file = resolve_param_file(tunnel_id, "{stage}")
'''
    return header + extra_header + algo + tail_patch


def build_denoising() -> None:
    extra = textwrap.dedent('''
        import numpy as np
        import pandas as pd
        from scipy.interpolate import interp1d
        from scipy.ndimage import uniform_filter1d
        from numba import njit, prange

        expected_keys = [
            "mask_r_low", "mask_r_high", "y_step", "z_step", "grad_threshold",
            "smoothing_window_size", "smoothing_offset", "default_cutoff_z",
        ]
        require_keys(params, expected_keys, param_file)
        mask_r_low = params["mask_r_low"]
        mask_r_high = params["mask_r_high"]
        y_step = params["y_step"]
        z_step = params["z_step"]
        grad_threshold = params["grad_threshold"]
        smoothing_window_size = params["smoothing_window_size"]
        smoothing_offset = params["smoothing_offset"]
        default_cutoff_z = params["default_cutoff_z"]
        print(f"Using parameters: mask_r_low={mask_r_low}, mask_r_high={mask_r_high}")

        paths = ensure_dir(tunnel_id)
        state = load_state(paths["state"])
        df_point_cloud = state["df_point_cloud"]
        ring_count = state["ring_count"]

    ''')
    tail = textwrap.dedent('''
        df_point_cloud.to_csv(paths["denoised_csv"], index=False)
        state["df_point_cloud"] = df_point_cloud
        save_state(paths["state"], state)
        print(f"Denoising complete -> {paths['denoised_csv']}")
    ''')
    write(os.path.join(AGENTS, "denoising.py"), build_from_root_agent("denoising", extra_header=extra, tail_patch=tail))


def build_enhancing() -> None:
    src = open(os.path.join(REPO, "agents", "enhancing.py")).read()
    idx = src.find("# Cell 1")
    if idx < 0:
        idx = src.find("df_support_filtered = df_point_cloud")
    algo = src[idx:]
    # cut root agents tail saves - use sam4tun modular tail
    cut = algo.find("os.makedirs(base_dir")
    if cut < 0:
        cut = algo.find("# Save results")
    if cut > 0:
        algo = algo[:cut]
    mod_tail = open(os.path.join(SAM4TUN, "3_enhancing.py")).read()
    tail_start = mod_tail.find("df_point_cloud.to_csv(paths[\"enhanced_csv\"]")
    tail = mod_tail[tail_start:]

    header = f'''#!/usr/bin/env python3
# Parameterized enhancing — agents/enhancing.py + sam4tun state I/O
# Deferred JSON: none (ring_spacing_factor wired)
{BOOTSTRAP}

import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree, cKDTree
import numba as nb
from numba import njit, prange
from scipy.interpolate import griddata
from tqdm.auto import tqdm
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import time

tunnel_id = parse_pipeline_args("enhancing")
params = load_stage_parameters(tunnel_id, "enhancing")
param_file = resolve_param_file(tunnel_id, "enhancing")
'''
    extra = textwrap.dedent('''
        expected_keys = [
            "upsampling_stage1_target_distance", "upsampling_stage2_target_distance",
            "upsampling_stage3_target_distance", "curvature_threshold",
            "depth_threshold_low", "depth_threshold_high", "inter_radius",
            "duplicate_threshold", "n_segment_start", "n_segment_end",
            "num_neighbors", "num_interpolations", "resolution", "window_size",
            "ring_spacing_factor",
        ]
        require_keys(params, expected_keys, param_file)
        upsampling_stage1_target_distance = params["upsampling_stage1_target_distance"]
        upsampling_stage2_target_distance = params["upsampling_stage2_target_distance"]
        upsampling_stage3_target_distance = params["upsampling_stage3_target_distance"]
        curvature_threshold = params["curvature_threshold"]
        depth_threshold_low = params["depth_threshold_low"]
        depth_threshold_high = params["depth_threshold_high"]
        inter_radius = params["inter_radius"]
        duplicate_threshold = params["duplicate_threshold"]
        n_segment_start = params["n_segment_start"]
        n_segment_end = params["n_segment_end"]
        num_neighbors = params["num_neighbors"]
        num_interpolations = params["num_interpolations"]
        resolution = params["resolution"]
        window_size = params["window_size"]
        ring_spacing_factor = params["ring_spacing_factor"]

        paths = ensure_dir(tunnel_id)
        state = load_state(paths["state"])
        df_point_cloud = state["df_point_cloud"]
        ring_count = state["ring_count"]

    ''')
    algo = algo.replace("1.2 * n_segment[0]", "ring_spacing_factor * n_segment[0]")
    algo = algo.replace("1.2 * n_segment[1]", "ring_spacing_factor * n_segment[1]")
    write(os.path.join(AGENTS, "enhancing.py"), header + extra + algo + "\n" + tail)


def build_detecting() -> None:
    src = open(os.path.join(REPO, "agents", "detecting.py")).read()
    idx = src.find("# Cell 4")
    if idx < 0:
        idx = src.find("binary_map = np.where")
    algo = src[idx:]
    for pat in [
        r"base_dir = resolve_output_base_dir.*?\n",
        r"_base = resolve_output_base_dir.*?\n",
        r"base_dir = _base\.rstrip.*?\n",
        r"depth_map_outlier = np\.load.*?\n",
        r"ring_count_file = os\.path\.join.*?\n",
        r"ring_count = int\(open\(ring_count_file.*?\n",
        r'print\(f"Processing tunnel:.*?\n',
    ]:
        algo = re.sub(pat, "", algo, count=1)
    cut = algo.find("os.makedirs(base_dir")
    if cut < 0:
        cut = algo.find("df_loc.to_csv")
    if cut > 0:
        algo = algo[:cut]

    mod = open(os.path.join(SAM4TUN, "4_detection.py")).read()
    tail_start = mod.find("state[\"df_loc\"]")
    if tail_start < 0:
        tail_start = mod.find("save_state(paths[\"state\"]")
    tail = mod[tail_start:]

    header = f'''#!/usr/bin/env python3
# Parameterized detecting — agents/detecting.py + sam4tun state I/O
# Deferred JSON: K_height/AB_height in prompt heuristics (hardcoded 1079.92/3239.77)
{BOOTSTRAP}

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tunnel_id = parse_pipeline_args("detecting")
params = load_stage_parameters(tunnel_id, "detecting")
param_file = resolve_param_file(tunnel_id, "detecting")
'''
    extra = textwrap.dedent('''
        expected_keys = [
            "binary_threshold", "morphological_kernel_size", "dilation_iterations",
            "hough_threshold_oblique", "minLineLength_oblique", "maxLineGap_oblique",
            "hough_threshold_horizontal", "minLineLength_horizontal", "maxLineGap_horizontal",
            "hough_threshold_vertical", "angle_range_oblique_positive", "angle_range_oblique_negative",
            "merge_distance", "ring_spacing_constant", "resolution",
        ]
        require_keys(params, expected_keys, param_file)
        binary_threshold = params["binary_threshold"]
        morphological_kernel_size = params["morphological_kernel_size"]
        dilation_iterations = params["dilation_iterations"]
        hough_threshold_oblique = params["hough_threshold_oblique"]
        minLineLength_oblique = params["minLineLength_oblique"]
        maxLineGap_oblique = params["maxLineGap_oblique"]
        hough_threshold_horizontal = params["hough_threshold_horizontal"]
        minLineLength_horizontal = params["minLineLength_horizontal"]
        maxLineGap_horizontal = params["maxLineGap_horizontal"]
        hough_threshold_vertical = params["hough_threshold_vertical"]
        angle_range_oblique_positive = params["angle_range_oblique_positive"]
        angle_range_oblique_negative = params["angle_range_oblique_negative"]
        merge_distance = params["merge_distance"]
        ring_spacing_constant = params["ring_spacing_constant"]
        resolution = params["resolution"]

        paths = ensure_dir(tunnel_id)
        state = load_state(paths["state"])
        df_point_cloud = state["df_point_cloud"]
        df_enhance_segment = state["df_enhance_segment"]
        df_enhance_joint = state["df_enhance_joint"]
        ring_count = state["ring_count"]
        depth_map_outlier = state["depth_map_outlier"]

    ''')
  # replace hardcoded angle ranges in algo - agents already uses variables
    write(os.path.join(AGENTS, "detecting.py"), header + extra + algo + "\n" + tail)


def build_sam() -> None:
    # Start from sam4tun/5_sam.py for state I/O + depth_map regen; inject params from agents/sam.py header
    mod = open(os.path.join(SAM4TUN, "5_sam.py")).read()
    # strip modular header through state load
    idx = mod.find("# Regenerate depth_map")
    body = mod[idx:]

    header = f'''#!/usr/bin/env python3
# Parameterized sam — sam4tun/5_sam.py + JSON top-level/processing params
# Deferred JSON: prompt_points, segment_order, use_original_label_distributions, processing.mask_eps
{BOOTSTRAP}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import math
import pickle
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.path import Path

tunnel_id = parse_pipeline_args("sam")
config = load_stage_parameters(tunnel_id, "sam")
param_file = resolve_param_file(tunnel_id, "sam")
expected_keys = [
    "segment_per_ring", "segment_width", "K_height", "AB_height", "angle", "processing",
]
require_keys(params, expected_keys, param_file)
segment_per_ring = config["segment_per_ring"]
segment_width = config["segment_width"]
K_height = config["K_height"]
AB_height = config["AB_height"]
angle = config["angle"]
processing = config["processing"]
resolution = processing["resolution"]
padding = processing["padding"]
crop_margin = processing["crop_margin"]
y_bounds = processing["y_bounds"]

paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
df_loc = state["df_loc"]
pixel_to_point = state["pixel_to_point"]
ring_count = state["ring_count"]

'''
    # fix require_keys call - used wrong variable name
    header = header.replace("require_keys(params,", "require_keys(config,")

    # patch body for parameterized calls
    body = body.replace(
        "def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution=0.005,\n"
        "                           segment_width=1200, K_height=1079.92, AB_height=3239.77):",
        "def generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution,\n"
        "                           segment_width, K_height, AB_height, y_bounds):",
    )
    body = body.replace(
        "y_cond = points_real[i, 1] + map_y < 4200 or points_real[i, 1] + map_y > 13100",
        "y_cond = points_real[i, 1] + map_y < y_bounds[0] or points_real[i, 1] + map_y > y_bounds[1]",
    )
    body = body.replace(
        "(segment_width*0.5+150)/(resolution*1000)",
        "(segment_width*0.5+padding)/(resolution*1000)",
    )
    body = body.replace(
        "def process_row(df_row, image, resolution=0.005, segment_per_ring=6, segment_width=1200, \n"
        "                K_height=1079.92, angle=7.52, AB_height=3239.77):",
        "def process_row(df_row, image, resolution, segment_per_ring, segment_width,\n"
        "                K_height, angle, AB_height, padding, crop_margin, y_bounds):",
    )
    body = body.replace(
        "delta_x = convert_to_pixel_coords(0.5*segment_width + 150, resolution)",
        "delta_x = convert_to_pixel_coords(0.5*segment_width + padding, resolution)",
    )
    body = body.replace(
        "delta_y = convert_to_pixel_coords(0.5*K_height + math.tan(math.radians(angle))*700+100 + 50, resolution) # K-block",
        "delta_y = convert_to_pixel_coords(0.5*K_height + math.tan(math.radians(angle))*700+100 + crop_margin, resolution) # K-block",
    )
    body = body.replace(
        "delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + 50, resolution) # other block",
        "delta_y = convert_to_pixel_coords(0.5*AB_height + math.tan(math.radians(angle))*700+100 + crop_margin, resolution) # other block",
    )
    body = body.replace(
        "points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution)",
        "points, labels = generate_prompt_points(prompt_centre, initial_x, map_y, block, resolution, segment_width, K_height, AB_height, y_bounds)",
    )
    body = body.replace(
        "def sam_segment(df, image, resolution=0.005, segment_per_ring=6):",
        "def sam_segment(df, image, resolution, segment_per_ring, segment_width, K_height, AB_height, angle, padding, crop_margin, y_bounds):",
    )
    body = body.replace(
        "result = process_row(row, image, resolution, segment_per_ring)",
        "result = process_row(row, image, resolution, segment_per_ring, segment_width, K_height, angle, AB_height, padding, crop_margin, y_bounds)",
    )
    body = body.replace(
        "results = sam_segment(df_loc, image)",
        "results = sam_segment(df_loc, image, resolution, segment_per_ring, segment_width, K_height, AB_height, angle, padding, crop_margin, y_bounds)",
    )
    write(os.path.join(AGENTS, "sam.py"), header + body)


def main() -> None:
    build_unfolding()
    build_denoising()
    build_enhancing()
    build_detecting()
    build_sam()
    print("done")


if __name__ == "__main__":
    main()
