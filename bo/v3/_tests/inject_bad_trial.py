"""Smoke test: inject a deliberately broken params dict to validate the
failure sandbox (run_trial returns a structured failure, not a crash).

Run from repo root via:
    ./venv/bin/python -m bo.v3._tests.inject_bad_trial

This writes its trial sandbox under
``logs/v3/bo_calibration/_smoke/failure_sandbox/`` (a non-protected location).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

from bo.v3 import r4tun_seed
from bo.v3.objectives import run_trial


def main() -> int:
    out = REPO_ROOT / "logs" / "v3" / "bo_calibration" / "_smoke" / "failure_sandbox"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    seed_pre = r4tun_seed.load_r4tun_preprocessing(target_tunnel_diameter=7.5)
    seed_det = r4tun_seed.load_r4tun_detection(target_tunnel_diameter=7.5)

    # Deliberately broken: annulus far above the tunnel radius (~3.75 m)
    # so no points survive filtering and depth-map generation collapses.
    bad_params = {
        "radius_min": 9.0,
        "radius_max": 12.0,
        "gradient_threshold": 0.5,
        "smoothing_offset": 0.0,
        "curvature_neighbors": 5,
        "interpolation_window": 5,
        "target_distance_1": 0.07,
        "target_distance_2": 0.035,
        "target_distance_3": 0.018,
        "outlier_interpolation_radius": 0.03,
        "outlier_num_interpolations": 2,
        "outlier_depth_map_window": 1,
        "outlier_neighbors": 20,
    }

    result = run_trial(
        tunnel_id="4-9",
        ring_id=365,
        stage="preprocessing",
        trial_dir=out / "trial_bad",
        seed_pre_params=seed_pre,
        seed_det_params=seed_det,
        trial_params=bad_params,
        timeout_sec=60.0,
        mem_cap_bytes=8 * (1024**3),
    )

    payload = {
        "status": result.status,
        "failure_mode": result.failure_mode,
        "failure_detail": (result.failure_detail or "")[:400],
        "miou_fixed_class": result.miou_fixed_class,
        "miou_permutation": result.miou_permutation,
        "elapsed_sec": result.elapsed_sec,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if result.status != "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
