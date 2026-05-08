#!/usr/bin/env python3
"""Landmark-aware K anchor (depth-signature based).

Idea
----
After gravity-alignment, the K block sits at a tunnel-specific y_frac
that's stable across rings of the same tunnel (e.g. 4-3 K ≈ 0.118).
Tiny residual rotation errors can still misplace it. We don't need a
per-boundary depth-discontinuity scorer (that didn't work; see
``logs/gravity_unwrap_v1/gravity_and_depth_anchor_report.md``); instead
we exploit a 1D depth signature unique to K and look for the best
match in held-out depth maps.

Workflow
--------
1. **Calibration**: for the BO-best calibration ring of each tunnel,
   extract the depth-signature ``S_K_calib(t)`` — the per-row mean
   depth (over valid pixels) inside K's gravity-aligned y_frac range,
   stretched to a fixed length.
2. **Held-out**: slide ``S_K_calib`` across the held-out depth map
   (gravity-aligned), compute normalised cross-correlation per starting
   row, and pick the peak. The peak's center y_frac is the *measured K
   anchor*.
3. **Use as intrinsic signal**:
     - Compute the offset between measured K anchor and the template's
       predicted K y_frac.
     - Define ``S_K_anchor = exp(-offset^2 / sigma^2)`` as an additional
       intrinsic axis (peak strength can also gate it).
     - Optionally, propose a template shift candidate that aligns K.

This is a pure-intrinsic deployable signal: at inference we know the
calibration K signature for each tunnel and the held-out depth map.
No GT is required.

Outputs
-------
``logs/gravity_v1/calibration/<tunnel>/k_signature.npy``
``logs/gravity_v1/k_anchor_summary.csv``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAVITY_ROOT = REPO_ROOT / "logs" / "gravity_v1"
CALIB_ROOT = GRAVITY_ROOT / "calibration"
HELDOUT_ROOT = GRAVITY_ROOT / "heldout"
CALIB_BASE = REPO_ROOT / "logs" / "detection_boundary_structural_panel_v3" / "artifacts"

DEFAULT_CALIB_MAP = {
    "4-3": "r179",
    "4-4": "r215",
    "4-5": "r249",
    "4-6": "r283",
    "5-1": "r116",
    "5-6": "r285",
    "5-7": "r315",
}

K_SIG_LEN = 80  # fixed length for K signature (pixels)
SIGMA_FRAC = 0.05  # Gaussian sigma for offset->score conversion (fraction of H)


def _row_signature(depth_map: np.ndarray, y_start_frac: float, y_end_frac: float) -> np.ndarray:
    """Per-row mean depth in the y_frac range [y_start_frac, y_end_frac]."""
    H = int(depth_map.shape[0])
    y0 = int(round(y_start_frac * H)) % H
    y1 = int(round(y_end_frac * H)) % H
    if y1 <= y0:
        # Wrap around
        sub = np.concatenate([depth_map[y0:], depth_map[:y1]], axis=0)
    else:
        sub = depth_map[y0:y1]
    valid = np.isfinite(sub) & (sub > 0)
    sig = np.array([
        float(np.mean(row[mask])) if mask.any() else float("nan")
        for row, mask in zip(sub, valid)
    ])
    # Linear interp NaNs (cyclic-safe within window)
    if np.any(np.isnan(sig)):
        idx = np.where(~np.isnan(sig))[0]
        if len(idx):
            sig = np.interp(np.arange(len(sig)), idx, sig[idx])
        else:
            sig = np.zeros_like(sig)
    # Resample to K_SIG_LEN
    if len(sig) > 1:
        sig_rs = np.interp(
            np.linspace(0, len(sig) - 1, K_SIG_LEN),
            np.arange(len(sig)),
            sig,
        )
    else:
        sig_rs = np.zeros(K_SIG_LEN)
    # Standardise (0-mean unit-var) so cross-correlation is scale-free
    if np.std(sig_rs) > 1e-9:
        sig_rs = (sig_rs - sig_rs.mean()) / sig_rs.std()
    else:
        sig_rs = sig_rs - sig_rs.mean()
    return sig_rs


def build_k_signature_for_tunnel(tunnel: str, calib_ring: str, *, force: bool = False) -> dict[str, Any]:
    out_dir = CALIB_ROOT / tunnel
    sig_path = out_dir / "k_signature.npy"
    meta_path = out_dir / "k_signature.json"
    if sig_path.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    template_path = out_dir / "template.json"
    if not template_path.exists():
        raise FileNotFoundError(f"calibration template missing for {tunnel}")
    template_data = json.loads(template_path.read_text())
    template = template_data["template"]
    n = len(template)
    k_idx = next((i for i, b in enumerate(template) if b.get("block") == "K"), None)
    if k_idx is None:
        raise ValueError(f"no K in template for {tunnel}")

    y_start = float(template[k_idx]["y_frac"])
    if k_idx + 1 < n:
        y_end = float(template[k_idx + 1]["y_frac"])
    else:
        y_end = float(template[0]["y_frac"]) + 1.0  # wrap

    # Use calibration depth_map.npy (NON-gravity-aligned). We need to
    # GRAVITY-shift it because calibration's template was shifted to
    # gravity coordinates.
    calib_dir = CALIB_BASE / tunnel / calib_ring / "best" / tunnel / calib_ring
    dm = np.load(calib_dir / "depth_map.npy")
    # Apply the same row_shift that template promotion used
    H = int(dm.shape[0])
    row_shift = int(template_data.get("calib_row_shift", 0)) % H
    dm_g = np.roll(dm, -row_shift, axis=0)

    sig = _row_signature(dm_g, y_start, y_end)
    np.save(sig_path, sig)
    meta = {
        "tunnel": tunnel,
        "calib_ring": calib_ring,
        "k_y_frac_start": y_start,
        "k_y_frac_end": y_end,
        "k_y_frac_center": float((y_start + y_end) / 2.0),
        "k_height_frac": float(y_end - y_start),
        "k_sig_len": K_SIG_LEN,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def measure_k_anchor(depth_map: np.ndarray, k_signature: np.ndarray, k_height_frac: float) -> dict[str, Any]:
    """Slide K signature across depth map; return best-match meta."""
    H = int(depth_map.shape[0])
    win_h = int(round(k_height_frac * H))
    if win_h < 4:
        win_h = max(4, win_h)
    sig_len = len(k_signature)

    # For each candidate start row y0, compute mean depth per row in [y0, y0+win_h),
    # resample to sig_len, normalize, compute correlation.
    valid = np.isfinite(depth_map) & (depth_map > 0)
    row_means = np.full(H, np.nan, dtype=np.float64)
    for y in range(H):
        m = valid[y]
        if m.any():
            row_means[y] = float(np.mean(depth_map[y, m]))
    # Linear interp NaN rows
    nan_mask = np.isnan(row_means)
    if nan_mask.any() and (~nan_mask).any():
        idx = np.where(~nan_mask)[0]
        row_means = np.interp(np.arange(H), idx, row_means[idx])

    # Slide window cyclically
    corrs = np.zeros(H, dtype=np.float64)
    for y in range(H):
        if y + win_h <= H:
            sub = row_means[y:y + win_h]
        else:
            sub = np.concatenate([row_means[y:], row_means[:(y + win_h) % H]])
        sub_rs = np.interp(np.linspace(0, len(sub) - 1, sig_len), np.arange(len(sub)), sub)
        if np.std(sub_rs) > 1e-9:
            sub_rs = (sub_rs - sub_rs.mean()) / sub_rs.std()
        else:
            sub_rs = sub_rs - sub_rs.mean()
        corrs[y] = float(np.dot(sub_rs, k_signature) / sig_len)

    best_y = int(np.argmax(corrs))
    best_corr = float(corrs[best_y])
    best_y_frac = float(best_y) / float(H)
    # K center is at start + half height
    best_center_y_frac = (best_y_frac + k_height_frac / 2.0) % 1.0
    return {
        "best_start_y_frac": best_y_frac,
        "best_center_y_frac": best_center_y_frac,
        "best_correlation": best_corr,
        "win_h_pixels": int(win_h),
        "H": int(H),
    }


def k_anchor_score_from_template(
    depth_map: np.ndarray,
    template: list[dict[str, Any]],
    k_signature: np.ndarray,
    k_height_frac: float,
    *,
    sigma_frac: float = SIGMA_FRAC,
) -> dict[str, Any]:
    """Compute S_K_anchor for a held-out ring given its template."""
    k_idx = next((i for i, b in enumerate(template) if b.get("block") == "K"), None)
    if k_idx is None:
        return {"S_K_anchor": 0.0, "reason": "no_K_in_template"}
    n = len(template)
    y_start_pred = float(template[k_idx]["y_frac"])
    if k_idx + 1 < n:
        y_end_pred = float(template[k_idx + 1]["y_frac"])
    else:
        y_end_pred = float(template[0]["y_frac"]) + 1.0
    pred_center = ((y_start_pred + y_end_pred) / 2.0) % 1.0

    meas = measure_k_anchor(depth_map, k_signature, k_height_frac)
    meas_center = meas["best_center_y_frac"]
    # Cyclic offset (fraction)
    diff = abs(meas_center - pred_center)
    offset = min(diff, 1.0 - diff)
    s = float(np.exp(-(offset ** 2) / (sigma_frac ** 2)))
    # Weight by best correlation strength (so weak matches don't dominate)
    s *= float(np.clip(meas["best_correlation"], 0.0, 1.0))
    return {
        "S_K_anchor": float(s),
        "offset_frac": float(offset),
        "predicted_center_y_frac": float(pred_center),
        "measured_center_y_frac": float(meas_center),
        "best_correlation": float(meas["best_correlation"]),
    }


# ---------------------------------------------------------------------------
# CLI

def cmd_build_signatures(args: argparse.Namespace) -> int:
    out: list[dict[str, Any]] = []
    for tunnel, calib_ring in DEFAULT_CALIB_MAP.items():
        try:
            meta = build_k_signature_for_tunnel(tunnel, calib_ring, force=args.force)
            print(f"{tunnel}: K@y={meta['k_y_frac_center']:.3f} h={meta['k_height_frac']:.3f}")
            out.append({"tunnel": tunnel, **meta})
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {tunnel}: {exc}")
    pd.DataFrame(out).to_csv(GRAVITY_ROOT / "k_signature_summary.csv", index=False)
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    rings = []
    if args.rings:
        for s in args.rings.split(","):
            t, r = s.strip().split("/", 1)
            rings.append((t, r))
    else:
        for tdir in sorted(HELDOUT_ROOT.iterdir()):
            if not tdir.is_dir():
                continue
            for rdir in sorted(tdir.iterdir()):
                if not rdir.is_dir():
                    continue
                if (rdir / "depth_map.npy").exists():
                    rings.append((tdir.name, rdir.name))

    rows: list[dict[str, Any]] = []
    for tunnel, ring in rings:
        sig_path = CALIB_ROOT / tunnel / "k_signature.npy"
        meta_path = CALIB_ROOT / tunnel / "k_signature.json"
        ring_dir = HELDOUT_ROOT / tunnel / ring
        if not sig_path.exists() or not (ring_dir / "depth_map.npy").exists():
            print(f"{tunnel}/{ring}: missing inputs")
            continue
        sig = np.load(sig_path)
        meta = json.loads(meta_path.read_text())
        dm = np.load(ring_dir / "depth_map.npy")
        # Load template from parameters_detection.json
        det_params_path = ring_dir / "parameters_detection.json"
        template = []
        if det_params_path.exists():
            try:
                template = json.loads(det_params_path.read_text()).get("single_ring_visual_slot_template", [])
            except Exception:  # noqa: BLE001
                template = []
        result = k_anchor_score_from_template(
            dm, template, sig, meta["k_height_frac"]
        )
        rows.append({
            "ring": f"{tunnel}/{ring}",
            "tunnel": tunnel,
            **result,
        })
        print(f"{tunnel}/{ring}: S_K_anchor={result['S_K_anchor']:.3f} "
              f"pred_K_y={result.get('predicted_center_y_frac', float('nan')):.3f} "
              f"meas_K_y={result.get('measured_center_y_frac', float('nan')):.3f} "
              f"corr={result.get('best_correlation', float('nan')):.3f}")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(GRAVITY_ROOT / "k_anchor_summary.csv", index=False)
    print(f"\nSaved: {GRAVITY_ROOT / 'k_anchor_summary.csv'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("build-signatures", help="Build per-tunnel K signatures")
    pa.add_argument("--force", action="store_true")
    pa.set_defaults(func=cmd_build_signatures)
    pb = sub.add_parser("measure", help="Measure K anchor on held-out rings")
    pb.add_argument("--rings", type=str, default=None)
    pb.set_defaults(func=cmd_measure)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
