#!/usr/bin/env python
"""GLM ablation orchestrator — memory+state+knowledge condition (m_s_k)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_CMD = [sys.executable, str(_REPO / "run_ablation_glm.py"), "--ablation", "m_s_k", *sys.argv[1:]]


def main() -> None:
    raise SystemExit(subprocess.run(_CMD, cwd=str(_REPO)).returncode)


if __name__ == "__main__":
    main()
