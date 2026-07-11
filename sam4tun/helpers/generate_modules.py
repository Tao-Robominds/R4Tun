#!/usr/bin/env python3
"""Generate stage modules by extracting verbatim line ranges from SAM4Tun.py."""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONOLITH = os.path.join(ROOT, "SAM4Tun.py")

COMMON_IMPORTS = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""

STAGES = [
    (
        "1_upfolding.py",
        1,
        728,
        """
import sys
import os
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import save_state

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
""",
        """
df_point_cloud.to_csv(paths["unwrapped_csv"], index=False)
save_state(paths["state"], {
    "df_point_cloud": df_point_cloud,
    "ring_count": ring_count,
    "resolution": 0.005,
})
print(f"Upfolding complete -> {paths['unwrapped_csv']}")
""",
    ),
    (
        "2_denoising.py",
        729,
        882,
        """
import sys
import os
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import load_state, save_state

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
ring_count = state["ring_count"]
resolution = state.get("resolution", 0.005)
""",
        """
df_point_cloud.to_csv(paths["denoised_csv"], index=False)
state["df_point_cloud"] = df_point_cloud
save_state(paths["state"], state)
print(f"Denoising complete -> {paths['denoised_csv']}")
""",
    ),
    (
        "3_enhancing.py",
        883,
        1402,
        """
import sys
import os
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import load_state, save_state

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
ring_count = state["ring_count"]
resolution = state.get("resolution", 0.005)
""",
        """
import pickle
import numpy as np

np.save(os.path.join(os.path.dirname(paths["depth_map"]), "depth_map.npy"), depth_map)
df_point_cloud.to_csv(paths["enhanced_csv"], index=False)
with open(paths["pixel_to_point"], "wb") as f:
    pickle.dump(pixel_to_point, f)
state.update({
    "df_point_cloud": df_point_cloud,
    "df_enhance_segment": df_enhance_segment,
    "df_enhance_joint": df_enhance_joint,
    "depth_map": depth_map,
    "depth_map_outlier": depth_map_outlier,
    "pixel_to_point": pixel_to_point,
    "resolution": resolution,
})
save_state(paths["state"], state)
print(f"Enhancing complete -> {paths['enhanced_csv']}, {paths['depth_map']}")
""",
    ),
    (
        "4_detection.py",
        1403,
        1836,
        """
import sys
import os
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import load_state, save_state

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
df_enhance_segment = state["df_enhance_segment"]
df_enhance_joint = state["df_enhance_joint"]
ring_count = state["ring_count"]
resolution = state.get("resolution", 0.005)
depth_map_outlier = state["depth_map_outlier"]
""",
        """
df_loc.to_csv(paths["initial_points"], index=False)
state["df_loc"] = df_loc
save_state(paths["state"], state)
print(f"Detection complete -> {paths['initial_points']}")
""",
    ),
    (
        "5_sam.py",
        1837,
        2502,
        """
import sys
import os
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir
from helpers.pipeline_state import load_state

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
state = load_state(paths["state"])
df_point_cloud = state["df_point_cloud"]
df_loc = state["df_loc"]
pixel_to_point = state["pixel_to_point"]
ring_count = state["ring_count"]
resolution = state.get("resolution", 0.005)

# Regenerate depth_map.png from stored float array (I/O only; same save_depth_map_exact as monolith)
_depth_npy = os.path.join(os.path.dirname(paths["depth_map"]), "depth_map.npy")
if os.path.exists(_depth_npy):
    from matplotlib import pyplot as _plt

    def save_depth_map_exact(depth_map, resolution, filename="depth_map.png"):
        height, width = depth_map.shape
        dpi = 1.0 / resolution
        fig = _plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(depth_map, cmap="viridis", vmin=2.70, vmax=2.80)
        _plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
        _plt.close()

    save_depth_map_exact(np.load(_depth_npy), resolution, filename=paths["depth_map"])
""",
        """
import pickle

with open(paths["results_pkl"], "wb") as file:
    pickle.dump(results, file)
with open(paths["results_pkl"], "rb") as file:
    results = pickle.load(file)

updated_df.to_csv(paths["final_csv"], index=False)
df_pred.to_csv(paths["only_label"], index=False)
print(f"SAM complete -> {paths['only_label']}")
""",
    ),
    (
        "6_evaluation.py",
        2503,
        2738,
        """
import sys
import os
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from helpers.pipeline_io import ensure_dir

tunnel_id = sys.argv[1]
paths = ensure_dir(tunnel_id)
""",
        "",
    ),
]

REPLACEMENTS = [
    (r'np\.loadtxt\("sample\.txt"\)', 'np.loadtxt(paths["input_txt"])'),
    (r"df_point_cloud\.to_csv\('sample_unwrapped\.csv',index=False\)", 'df_point_cloud.to_csv(paths["unwrapped_csv"], index=False)'),
    (r'filename="depth_map\.png"', 'filename=paths["depth_map"]'),
    (r"np\.save\('depth_map_outlier\.npy', depth_map_outlier\)", 'np.save(paths["depth_map_outlier"], depth_map_outlier)'),
    (r"plt\.savefig\('detected_lines\.png'", 'plt.savefig(paths["detected_lines"]'),
    (r"plt\.savefig\('initial_prompt_points\.png'", 'plt.savefig(os.path.join(os.path.dirname(paths["initial_points"]), "initial_prompt_points.png")'),
    (r"df_loc\.to_csv\('initial_points\.csv',index=False\)", 'df_loc.to_csv(paths["initial_points"], index=False)'),
    (r'sys\.path\.append\("segment-anything"\)', 'sys.path.append(paths["segment_anything"])'),
    (r'sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"', 'sam_checkpoint = paths["sam_checkpoint"]'),
    (r"image = cv2\.imread\('depth_map\.png'\)", 'image = cv2.imread(paths["depth_map"])'),
    (r"with open\('results\.pkl', 'wb'\)", 'with open(paths["results_pkl"], "wb")'),
    (r"with open\('results\.pkl', 'rb'\)", 'with open(paths["results_pkl"], "rb")'),
    (r"updated_df\.to_csv\('final\.csv',index=False\)", 'updated_df.to_csv(paths["final_csv"], index=False)'),
    (r"df_pred\.to_csv\('only_label\.csv',index=False\)", 'df_pred.to_csv(paths["only_label"], index=False)'),
    (r"test = pd\.read_csv\('only_label\.csv'\)", 'test = pd.read_csv(paths["only_label"])'),
    (r'filename=_out\("depth_map\.png"\)', 'filename=paths["depth_map"]'),
    (r"np\.save\(_out\('depth_map_outlier\.npy'\)", 'np.save(paths["depth_map_outlier"]'),
    (r"df_loc\.to_csv\(_out\('initial_points\.csv'\)", 'df_loc.to_csv(paths["initial_points"]'),
    (r"df_point_cloud\.to_csv\(_out\('sample_unwrapped\.csv'\)", 'df_point_cloud.to_csv(paths["unwrapped_csv"]'),
    (r"updated_df\.to_csv\(_out\('final\.csv'\)", 'updated_df.to_csv(paths["final_csv"]'),
    (r"df_pred\.to_csv\(_out\('only_label\.csv'\)", 'df_pred.to_csv(paths["only_label"]'),
    (r"test = pd\.read_csv\(_out\('only_label\.csv'\)\)", 'test = pd.read_csv(paths["only_label"])'),
    (r"image = cv2\.imread\(_out\('depth_map\.png'\)\)", 'image = cv2.imread(paths["depth_map"])'),
    (r"with open\(_out\('results\.pkl'\), 'wb'\)", 'with open(paths["results_pkl"], "wb")'),
    (r"with open\(_out\('results\.pkl'\), 'rb'\)", 'with open(paths["results_pkl"], "rb")'),
    (r"plt\.savefig\(_out\('detected_lines\.png'\)", 'plt.savefig(paths["detected_lines"]'),
    (r"plt\.savefig\(_out\('initial_prompt_points\.png'\)", 'plt.savefig(os.path.join(os.path.dirname(paths["initial_points"]), "initial_prompt_points.png")'),
    (r"_out\('([^']+)'\)", r'os.path.join(os.path.dirname(paths["state"]), "\1")'),
    (r"plt\.show\(\)", "# plt.show()"),
]


def patch_body(body: str) -> str:
    for pattern, repl in REPLACEMENTS:
        body = re.sub(pattern, repl, body)
    return body


def main() -> None:
    with open(MONOLITH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, (filename, start, end, header, footer) in enumerate(STAGES):
        body = "".join(lines[start - 1 : end])
        body = patch_body(body)
        stage_header = header
        if i > 0:
            stage_header = header.replace(
                'matplotlib.use("Agg")\n',
                'matplotlib.use("Agg")\n' + COMMON_IMPORTS,
            )
        content = (
            '#!/usr/bin/env python3\n'
            '# AUTO-GENERATED from SAM4Tun.py — do not edit body; re-run generate_modules.py\n'
            f"{stage_header}\n"
            f"{body}\n"
            f"{footer}\n"
        )
        out = os.path.join(ROOT, filename)
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote {filename} ({end - start + 1} lines body)")


if __name__ == "__main__":
    main()
