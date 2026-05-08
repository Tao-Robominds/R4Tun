"""Official fixed B+C+D ring-level preprocessing module.

B: observed-theta cropped target outputs
C: tunnel-global coordinates where available
D: neighbor-ring context preprocessing with target-ring-only reporting
"""

from __future__ import annotations

import gc
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from _ring_denoising import denoise_ring  # noqa: E402
from _ring_enhancing import (  # noqa: E402
    canonical_theta_pixels,
    compute_curvature,
    enhance_outlier_points_ring,
    enhance_segment_surface,
    project_to_depth_map_inter,
    save_depth_map_exact,
)


def _load_preprocessing_mod():
    path = MODULE_DIR / "1_preprocessing.py"
    spec = importlib.util.spec_from_file_location("preprocessing_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preprocessing module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ring_txt_path(tunnel_id: str, ring_id: int) -> Path:
    return REPO_ROOT / "data" / "rings" / f"{tunnel_id.replace('-', '_')}_ring{ring_id}.txt"


def existing_context_rings(tunnel_id: str, ring_id: int, context_radius: int) -> List[int]:
    out: List[int] = []
    for rid in range(ring_id - context_radius, ring_id + context_radius + 1):
        if rid <= 0:
            continue
        if _ring_txt_path(tunnel_id, rid).is_file():
            out.append(rid)
    if ring_id not in out:
        raise FileNotFoundError(f"Target ring file missing: {_ring_txt_path(tunnel_id, ring_id)}")
    return sorted(out)


def load_context_raw(tunnel_id: str, context_rings: List[int]) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for rid in context_rings:
        txt = _ring_txt_path(tunnel_id, rid)
        arr = np.loadtxt(txt)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 6:
            raise ValueError(f"{txt} must contain at least 6 columns")
        df = pd.DataFrame(
            {
                "x": arr[:, 0],
                "y": arr[:, 1],
                "z": arr[:, 2],
                "intensity": arr[:, 3],
                "segment": arr[:, 4].astype(int),
                "ring": arr[:, 5].astype(int),
            }
        )
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def load_context_unwrapped_global(tunnel_id: str, context_rings: List[int]) -> Tuple[pd.DataFrame, str]:
    """Prefer r4tun tunnel-global coordinates; fallback to current data/<tunnel>/r*/unwrapped."""
    r4tun_unwrapped = REPO_ROOT / "r4tun" / "data" / "ablation_rules" / tunnel_id / "unwrapped.csv"
    if r4tun_unwrapped.is_file():
        df = pd.read_csv(r4tun_unwrapped)
        need = {"x", "y", "z", "intensity", "segment", "ring", "r", "theta", "h"}
        if not need.issubset(set(df.columns)):
            raise ValueError(f"{r4tun_unwrapped} missing required columns")
        sub = df[df["ring"].astype(int).isin(context_rings)].copy().reset_index(drop=True)
        if len(sub) > 0:
            return sub, "r4tun/data/ablation_rules global unwrapped"

    per_ring: List[pd.DataFrame] = []
    for rid in context_rings:
        p = REPO_ROOT / "data" / tunnel_id / f"r{rid}" / "unwrapped.csv"
        if not p.is_file():
            p = REPO_ROOT / "data" / "ablation" / "baseline" / tunnel_id / f"r{rid}" / "unwrapped.csv"
        if not p.is_file():
            raise FileNotFoundError(
                "No r4tun global unwrapped found and fallback per-ring unwrapped missing: "
                f"{p}"
            )
        d = pd.read_csv(p)
        d = d[d["ring"].astype(int) == int(rid)].copy()
        per_ring.append(d)
    return pd.concat(per_ring, ignore_index=True), "data/<tunnel>/r*/unwrapped fallback"


def _assign_ring_nearest(add_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    if len(add_df) == 0:
        add_df["ring"] = pd.Series(dtype=np.int64)
        return add_df
    if len(ref_df) == 0:
        add_df["ring"] = -1
        return add_df
    tree = cKDTree(ref_df[["h", "theta"]].to_numpy(dtype=np.float64))
    q = add_df[["h", "theta"]].to_numpy(dtype=np.float64)
    _, idx = tree.query(q, k=1)
    rings = ref_df["ring"].to_numpy(dtype=np.int64)[idx]
    out = add_df.copy()
    out["ring"] = rings
    return out


def build_enhancing_with_ring(
    df_d: pd.DataFrame,
    enhancing_params: Dict[str, Any],
    outlier_hd_disabled: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Custom wrapper around ring enhancing to preserve ring ownership for new points."""
    e = enhancing_params
    s1 = float(e["upsampling_stage1_target_distance"])
    s2 = float(e["upsampling_stage2_target_distance"])
    s3 = float(e["upsampling_stage3_target_distance"])
    curvature_threshold = float(e["curvature_threshold"])
    depth_threshold_low = float(e["depth_threshold_low"])
    depth_threshold_high = float(e["depth_threshold_high"])
    inter_radius = float(e["inter_radius"])
    duplicate_threshold = float(e["duplicate_threshold"])
    n_segment_start = int(e["n_segment_start"])
    n_segment_end = int(e["n_segment_end"])
    num_neighbors = int(e["num_neighbors"])
    num_interpolations = int(e["num_interpolations"])
    resolution = float(e["resolution"])

    df_support_filtered = df_d[df_d["pred"] != 0].copy()
    df_support_filtered_curva = compute_curvature(df_support_filtered)

    df_upsampling_all = df_support_filtered_curva.copy()
    for td in (s1, s2, s3):
        df_up = enhance_segment_surface(
            df_upsampling_all,
            target_distance=td,
            curvature_threshold_param=curvature_threshold,
            num_neighbors_param=num_neighbors,
        )
        if len(df_up) > 0:
            df_up = _assign_ring_nearest(df_up, df_upsampling_all)
            if "segment" not in df_up.columns and "segment" in df_upsampling_all.columns:
                tree = cKDTree(df_upsampling_all[["h", "theta"]].to_numpy(dtype=np.float64))
                q = df_up[["h", "theta"]].to_numpy(dtype=np.float64)
                _, idx = tree.query(q, k=1)
                df_up["segment"] = df_upsampling_all["segment"].to_numpy(dtype=np.int64)[idx]
        df_upsampling_all = pd.concat([df_upsampling_all, df_up], ignore_index=False)

    df_enhance_segment = df_upsampling_all.copy()

    x_min_ref = float(df_support_filtered_curva["h"].min())
    meaningful_df, new_df = enhance_outlier_points_ring(
        df_support_filtered_curva,
        depth_threshold_low=depth_threshold_low,
        depth_threshold_high=depth_threshold_high,
        inter_radius=inter_radius,
        num_interpolations=num_interpolations,
        duplicate_threshold=duplicate_threshold,
        resolution=resolution,
        num_neighbors=num_neighbors,
        hd_disabled=outlier_hd_disabled or (n_segment_end < 0 or n_segment_start < 0),
        n_segment=(float(n_segment_start), float(n_segment_end)),
        x_min_ref=x_min_ref,
    )
    if len(new_df) > 0:
        new_df = _assign_ring_nearest(new_df, meaningful_df if len(meaningful_df) else df_support_filtered_curva)
        if "segment" not in new_df.columns and "segment" in df_support_filtered_curva.columns:
            tree = cKDTree(df_support_filtered_curva[["h", "theta"]].to_numpy(dtype=np.float64))
            q = new_df[["h", "theta"]].to_numpy(dtype=np.float64)
            _, idx = tree.query(q, k=1)
            new_df["segment"] = df_support_filtered_curva["segment"].to_numpy(dtype=np.int64)[idx]

    df_enhance_joint = pd.concat([meaningful_df, new_df], ignore_index=False)

    df_d2 = df_d.copy()
    if len(meaningful_df) > 0:
        df_d2.loc[meaningful_df.index, "pred"] = 0
    return df_d2, df_enhance_segment, df_enhance_joint


def largest_empty_row_band(valid_mask: np.ndarray) -> int:
    row_valid = valid_mask.sum(axis=1)
    largest = 0
    cur = 0
    for x in row_valid == 0:
        if x:
            cur += 1
            largest = max(largest, cur)
        else:
            cur = 0
    return int(largest)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_trial_outputs(
    out_dir: Path,
    tunnel_diameter: float,
    resolution: float,
    window_size: int,
    df_enhance_segment: pd.DataFrame,
    df_enhance_joint: pd.DataFrame,
    target_ring: int,
) -> Dict[str, Any]:
    can_h = canonical_theta_pixels(tunnel_diameter, resolution)

    data_segment = {
        "index": df_enhance_segment.index,
        "x": df_enhance_segment["h"],
        "y": df_enhance_segment["theta"],
        "z": df_enhance_segment["r"],
        "pred": df_enhance_segment["pred"],
    }
    data_joint = {
        "x": df_enhance_joint["h"],
        "y": df_enhance_joint["theta"],
        "z": df_enhance_joint["r"],
        "pred": df_enhance_joint["pred"],
    }

    depth_map_context, pixel_to_point_context = project_to_depth_map_inter(
        data_segment,
        data_joint,
        resolution=resolution,
        window_size=window_size,
        outlier_mode=False,
        canonical_height_px=can_h,
    )
    np.save(out_dir / "context_depth_map.npy", depth_map_context)
    save_depth_map_exact(depth_map_context, resolution=resolution, filename=str(out_dir / "context_depth_map.png"))
    with (out_dir / "context_pixel_to_point.pkl").open("wb") as f:
        pickle.dump(pixel_to_point_context, f)
    context_pixel_support = int(len(pixel_to_point_context))
    del depth_map_context, pixel_to_point_context

    target_seg = df_enhance_segment[df_enhance_segment["ring"].astype(int) == int(target_ring)].copy()
    target_joint = df_enhance_joint[df_enhance_joint["ring"].astype(int) == int(target_ring)].copy()
    if len(target_seg) == 0:
        raise ValueError(f"No target-ring segment points after enhancing for r{target_ring}")

    t_data_segment = {
        "index": target_seg.index,
        "x": target_seg["h"],
        "y": target_seg["theta"],
        "z": target_seg["r"],
        "pred": target_seg["pred"],
    }
    t_data_joint = {
        "x": target_joint["h"],
        "y": target_joint["theta"],
        "z": target_joint["r"],
        "pred": target_joint["pred"],
    }

    depth_map_target, pixel_to_point_target = project_to_depth_map_inter(
        t_data_segment,
        t_data_joint,
        resolution=resolution,
        window_size=window_size,
        outlier_mode=False,
        canonical_height_px=None,
    )
    np.save(out_dir / "depth_map.npy", depth_map_target)
    save_depth_map_exact(depth_map_target, resolution=resolution, filename=str(out_dir / "depth_map.png"))
    with (out_dir / "pixel_to_point.pkl").open("wb") as f:
        pickle.dump(pixel_to_point_target, f)

    t_joint2 = {
        "x": target_joint["h"],
        "y": target_joint["theta"],
        "z": target_joint["r"],
        "pred": target_joint["pred"],
    }
    depth_map_target_outlier, _ = project_to_depth_map_inter(
        t_data_segment,
        t_joint2,
        resolution=resolution,
        window_size=1,
        outlier_mode=True,
        canonical_height_px=None,
    )
    np.save(out_dir / "depth_map_outlier.npy", depth_map_target_outlier)

    valid = np.isfinite(depth_map_target) & (depth_map_target > 0)
    valid_out = np.isfinite(depth_map_target_outlier) & (depth_map_target_outlier > 0)
    summary = {
        "depth_shape_h": int(depth_map_target.shape[0]),
        "depth_shape_w": int(depth_map_target.shape[1]),
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "valid_ratio_outlier": float(valid_out.mean()) if valid_out.size else 0.0,
        "largest_empty_row_band": largest_empty_row_band(valid),
        "largest_empty_row_band_outlier": largest_empty_row_band(valid_out),
        "target_pixel_support": int(len(pixel_to_point_target)),
        "context_pixel_support": context_pixel_support,
    }
    del depth_map_target, depth_map_target_outlier, pixel_to_point_target, valid, valid_out
    return summary


def run_context_trial(
    tunnel_id: str,
    ring_id: int,
    context_radius: int,
    output_root: Path,
    reference_base_dir: str,
    params_override: Dict[str, Any] | None = None,
) -> Path:
    out_dir = output_root / tunnel_id / f"r{ring_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prep_mod = _load_preprocessing_mod()
    if params_override is not None:
        params = dict(params_override)
    else:
        params, _ = prep_mod.load_parameters(
            tunnel_id=tunnel_id,
            ring_id=ring_id,
            regime_label=None,
            base_dir=reference_base_dir,
        )
    enh = prep_mod.build_enhancing_params(params)
    tunnel_diameter = float(prep_mod.get_param(params, "tunnel_diameter", default=7.5, allow_default=True))

    context_rings = existing_context_rings(tunnel_id, ring_id, context_radius)
    df_raw_context = load_context_raw(tunnel_id, context_rings)
    df_u_context, coord_source = load_context_unwrapped_global(tunnel_id, context_rings)
    need_cols = ["x", "y", "z", "intensity", "segment", "ring", "r", "theta", "h"]
    miss = [c for c in need_cols if c not in df_u_context.columns]
    if miss:
        raise ValueError(f"Context unwrapped missing columns: {miss}")
    df_u_context = df_u_context[need_cols].copy()
    df_u_context = df_u_context[df_u_context["ring"].astype(int).isin(context_rings)].reset_index(drop=True)

    df_raw_context.to_csv(out_dir / "context_raw.csv", index=False)
    df_u_context.to_csv(out_dir / "context_unwrapped.csv", index=False)
    (out_dir / "ring_count.txt").write_text(str(len(context_rings)))

    mask_lo = float(prep_mod.get_param(params, "radius_min", default=2.37, allow_default=True))
    mask_hi = float(prep_mod.get_param(params, "radius_max", default=3.8, allow_default=True))
    y_step = float(prep_mod.get_param(params, "y_step", default=0.4, allow_default=True))
    z_step = float(prep_mod.get_param(params, "z_step", default=0.005, allow_default=True))
    grad_thr = float(prep_mod.get_param(params, "gradient_threshold", default=0.15, allow_default=True))
    sm_win = int(prep_mod.get_param(params, "smoothing_window_size", default=5, allow_default=True))
    sm_off = float(prep_mod.get_param(params, "smoothing_offset", default=-0.002, allow_default=True))
    def_cut = float(prep_mod.get_param(params, "default_cutoff_z", default=tunnel_diameter / 2.0, allow_default=True))

    df_d = denoise_ring(
        df_u_context,
        ring_count=max(1, len(context_rings)),
        mask_r_low=mask_lo,
        mask_r_high=mask_hi,
        y_step=y_step,
        z_step=z_step,
        grad_threshold=grad_thr,
        smoothing_window_size=sm_win,
        smoothing_offset=sm_off,
        default_cutoff_z=def_cut,
    )
    df_d.to_csv(out_dir / "context_denoised.csv", index=False)

    hd_start = int(params.get("outlier_high_density_ring_start", -1))
    hd_end = int(params.get("outlier_high_density_ring_end", -1))
    outlier_hd_disabled = hd_start < 0 or hd_end < 0

    df_d2, df_enhance_segment, df_enhance_joint = build_enhancing_with_ring(
        df_d, enh, outlier_hd_disabled=outlier_hd_disabled
    )

    df_context_enhanced = pd.concat(
        [
            df_d2,
            df_enhance_segment[df_enhance_segment["pred"] == 8],
            df_enhance_joint[df_enhance_joint["pred"] == 8],
        ],
        ignore_index=True,
    )
    df_context_enhanced.to_csv(out_dir / "context_enhanced.csv", index=False)

    summary = write_trial_outputs(
        out_dir=out_dir,
        tunnel_diameter=tunnel_diameter,
        resolution=float(enh["resolution"]),
        window_size=int(enh["window_size"]),
        df_enhance_segment=df_enhance_segment,
        df_enhance_joint=df_enhance_joint,
        target_ring=ring_id,
    )

    df_d_target = df_d2[df_d2["ring"].astype(int) == int(ring_id)].copy()
    df_d_target.to_csv(out_dir / "denoised.csv", index=False)

    meta = {
        "tunnel_id": tunnel_id,
        "ring_id": int(ring_id),
        "context_radius": int(context_radius),
        "context_rings": [int(x) for x in context_rings],
        "coordinate_source": coord_source,
        "reference_base_dir": reference_base_dir,
        "output_dir": str(out_dir),
        "summary": summary,
    }
    _save_json(out_dir / "trial_meta.json", meta)
    del (
        df_raw_context,
        df_u_context,
        df_d,
        df_d2,
        df_enhance_segment,
        df_enhance_joint,
        df_context_enhanced,
        df_d_target,
    )
    gc.collect()
    return out_dir
