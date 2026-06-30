#!/usr/bin/env python3
"""Read-only audit: compare JSON keys vs expected_keys in agent stage scripts."""

from __future__ import annotations

import ast
import json
import os
import re
import sys

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
STAGES = ("unfolding", "denoising", "enhancing", "detecting", "sam")


def _expected_keys_from_script(stage: str) -> list[str]:
    path = os.path.join(_AGENTS_DIR, f"{stage}.py")
    if not os.path.isfile(path):
        return []
    src = open(path).read()
    m = re.search(r"expected_keys\s*=\s*(\[[^\]]*\])", src, re.DOTALL)
    if not m:
        return []
    return ast.literal_eval(m.group(1))


def _nested_keys(d: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        keys.add(path)
        if isinstance(v, dict):
            keys |= _nested_keys(v, path)
    return keys


def audit_tunnel(tunnel_id: str) -> int:
    issues = 0
    for stage in STAGES:
        param_file = os.path.join(_AGENTS_DIR, "parameters", tunnel_id, f"parameters_{stage}.json")
        script = os.path.join(_AGENTS_DIR, f"{stage}.py")
        if not os.path.isfile(param_file):
            print(f"❌ missing {param_file}")
            issues += 1
            continue
        if not os.path.isfile(script):
            print(f"❌ missing {script}")
            issues += 1
            continue
        with open(param_file) as f:
            params = json.load(f)
        expected = _expected_keys_from_script(stage)
        body = open(script).read()
        print(f"\n=== {stage} ===")
        for key in expected:
            if key not in params:
                print(f"  ❌ JSON missing expected key: {key}")
                issues += 1
            elif key not in body:
                print(f"  ⚠️  key loaded but not referenced in script body: {key}")
        # Top-level JSON keys not in expected_keys (informational)
        for key in params:
            if key not in expected:
                print(f"  ℹ️  JSON key not in expected_keys (may be nested/deferred): {key}")
    return issues


def main() -> None:
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "sample"
    n = audit_tunnel(tunnel_id)
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
