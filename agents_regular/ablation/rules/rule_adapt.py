"""
On-site rules baseline: generate parameter JSONs from field-observable inputs only.

Allowed inputs per tunnel (from site_inputs.json):
    diameter_m, family, ring_length_m, segment_per_ring

Everything not explicitly overridden keeps the sam4tun default value.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
SAM4TUN_DIR = os.path.join(REPO_ROOT, "agents", "ablation", "sam4tun")
SITE_INPUTS_PATH = os.path.join(SCRIPT_DIR, "site_inputs.json")
OUT_DIR = os.path.join(SCRIPT_DIR, "parameters")

STAGES = ["unfolding", "denoising", "enhancing", "detecting", "sam"]

REF_DIAMETER = 5.5
REF_RING_LENGTH = 1.2


def load_sam4tun_defaults() -> dict[str, dict]:
    defaults = {}
    for stage in STAGES:
        path = os.path.join(SAM4TUN_DIR, f"parameters_{stage}.json")
        with open(path) as f:
            defaults[stage] = json.load(f)
    return defaults


def load_site_inputs() -> dict[str, dict]:
    with open(SITE_INPUTS_PATH) as f:
        data = json.load(f)
    return data["tunnels"]


def adapt_unfolding(params: dict, site: dict) -> dict:
    params["diameter"] = site["diameter_m"]
    return params


def adapt_denoising(params: dict, site: dict) -> dict:
    r = site["diameter_m"] / 2.0
    params["mask_r_low"] = round(r - 0.15, 2)
    params["mask_r_high"] = round(r + 0.15, 2)
    params["default_cutoff_z"] = round(params["mask_r_high"] + 0.05, 2)
    return params


def adapt_enhancing(params: dict, site: dict) -> dict:
    # n_segment_end follows from segment count (engineer knows how many
    # segments per ring); sam4tun default 5 is for 6-seg tunnels.
    params["n_segment_end"] = site["segment_per_ring"] - 1
    return params


def adapt_detecting(params: dict, site: dict) -> dict:
    params["ring_spacing_constant"] = site["ring_length_m"]
    return params


def adapt_sam(params: dict, site: dict) -> dict:
    seg = site["segment_per_ring"]
    params["segment_per_ring"] = seg

    if seg == 7:
        params["segment_order"] = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
    else:
        params["segment_order"] = ["K", "B1", "A1", "A2", "A3", "B2"]

    d = site["diameter_m"]
    rl = site["ring_length_m"]

    if site["family"] == "T4_T5":
        scale_d = d / REF_DIAMETER
        scale_ring = rl / REF_RING_LENGTH

        params["segment_width"] = round(params["segment_width"] * scale_ring)
        params["K_height"] = round(params["K_height"] * scale_d, 2)
        params["AB_height"] = round(params["AB_height"] * scale_d, 2)

        if "processing" in params:
            proc = params["processing"]
            proc["padding"] = round(proc["padding"] * scale_d)
            proc["crop_margin"] = round(proc["crop_margin"] * scale_d)
            y_lo, y_hi = proc["y_bounds"]
            proc["y_bounds"] = [round(y_lo * scale_d), round(y_hi * scale_d)]

    return params


ADAPTERS = {
    "unfolding": adapt_unfolding,
    "denoising": adapt_denoising,
    "enhancing": adapt_enhancing,
    "detecting": adapt_detecting,
    "sam": adapt_sam,
}


def generate_for_tunnel(tunnel_id: str, site: dict, defaults: dict[str, dict]):
    out_dir = os.path.join(OUT_DIR, tunnel_id)
    os.makedirs(out_dir, exist_ok=True)

    for stage in STAGES:
        params = copy.deepcopy(defaults[stage])
        params = ADAPTERS[stage](params, site)
        out_path = os.path.join(out_dir, f"parameters_{stage}_rules.json")
        with open(out_path, "w") as f:
            json.dump(params, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate rules-baseline parameter JSONs from site inputs"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate for all tunnels in site_inputs.json",
    )
    parser.add_argument(
        "tunnel_ids", nargs="*",
        help="Generate for specific tunnel(s)",
    )
    args = parser.parse_args()

    defaults = load_sam4tun_defaults()
    site_inputs = load_site_inputs()

    if args.all:
        tunnel_ids = sorted(site_inputs.keys())
    elif args.tunnel_ids:
        tunnel_ids = args.tunnel_ids
    else:
        parser.error("Provide tunnel IDs or --all")
        return

    for tid in tunnel_ids:
        if tid not in site_inputs:
            print(f"  [skip] {tid} — not in site_inputs.json")
            continue
        generate_for_tunnel(tid, site_inputs[tid], defaults)
        print(f"  [ok] {tid}")

    print(f"\nGenerated {len(tunnel_ids)} tunnel(s) under {OUT_DIR}/")


if __name__ == "__main__":
    main()
