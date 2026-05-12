from __future__ import annotations

import argparse

from t123_depth_contract import (
    GATE_RINGS,
    RUN_ROOT,
    assert_all_depth_maps_pass,
    assert_writable,
    audit_many,
    load_t123_panel,
    preprocess_ring,
    write_summary,
)


def _run_depth_contract(ring_keys: list[str], *, label: str) -> None:
    for ring_key in ring_keys:
        preprocess_ring(ring_key)
    audit = audit_many(ring_keys).sort_values("ring_key").reset_index(drop=True)
    audit.to_csv(RUN_ROOT / f"{label}_depth_quality_audit.csv", index=False)
    write_summary(label, audit)
    assert_all_depth_maps_pass(audit, label=label)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard-gated r4tun-style depth-map contract for T1/T2/T3.")
    parser.add_argument(
        "--scope",
        choices=["gate", "all", "audit"],
        default="all",
        help="Run only the one-ring-per-family gate, gate plus all 30 T1/T2/T3 rings, or audit existing outputs.",
    )
    args = parser.parse_args()

    assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    panel = load_t123_panel()
    panel.to_csv(RUN_ROOT / "panel_t123_segment6.csv", index=False)

    all_keys = panel["ring_key"].astype(str).tolist()
    if args.scope == "audit":
        audit = audit_many(all_keys).sort_values("ring_key").reset_index(drop=True)
        audit.to_csv(RUN_ROOT / "all_30_depth_gate_depth_quality_audit.csv", index=False)
        write_summary("all_30_depth_gate", audit)
        assert_all_depth_maps_pass(audit, label="all_30_depth_gate")
        return 0

    _run_depth_contract(GATE_RINGS, label="single_instance_depth_gate")
    if args.scope == "gate":
        return 0

    _run_depth_contract(all_keys, label="all_30_depth_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
