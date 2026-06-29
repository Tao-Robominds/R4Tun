"""Artifact paths for the modular SAM4Tun pipeline."""

import os

SAM4TUN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pipeline_dir(tunnel_id: str) -> str:
    return os.path.join(SAM4TUN_ROOT, "data", tunnel_id)


def artifact_paths(tunnel_id: str) -> dict[str, str]:
    d = pipeline_dir(tunnel_id)
    mono = os.path.join(SAM4TUN_ROOT, "data", "monolith")
    return {
        "input_txt": os.path.join(SAM4TUN_ROOT, "data", f"{tunnel_id}.txt"),
        "state": os.path.join(d, "state.pkl"),
        "unwrapped_csv": os.path.join(d, "unwrapped.csv"),
        "denoised_csv": os.path.join(d, "denoised.csv"),
        "enhanced_csv": os.path.join(d, "enhanced.csv"),
        "pixel_to_point": os.path.join(d, "pixel_to_point.pkl"),
        "depth_map": os.path.join(d, "depth_map.png"),
        "depth_map_outlier": os.path.join(d, "depth_map_outlier.npy"),
        "detected_lines": os.path.join(d, "detected_lines.png"),
        "initial_points": os.path.join(d, "initial_points.csv"),
        "results_pkl": os.path.join(d, "results.pkl"),
        "final_csv": os.path.join(d, "final.csv"),
        "only_label": os.path.join(d, "only_label.csv"),
        "evaluation_dir": os.path.join(d, "evaluation"),
        "monolith_dir": mono,
        "sam_checkpoint": os.path.join(SAM4TUN_ROOT, "segment-anything", "sam_vit_h_4b8939.pth"),
        "segment_anything": os.path.join(SAM4TUN_ROOT, "segment-anything"),
    }


def ensure_dir(tunnel_id: str) -> dict[str, str]:
    paths = artifact_paths(tunnel_id)
    os.makedirs(pipeline_dir(tunnel_id), exist_ok=True)
    os.makedirs(paths["evaluation_dir"], exist_ok=True)
    return paths


def monolith_data_dir() -> str:
    path = os.path.join(SAM4TUN_ROOT, "data", "monolith")
    os.makedirs(path, exist_ok=True)
    return path
