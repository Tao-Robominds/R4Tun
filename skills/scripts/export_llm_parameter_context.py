#!/usr/bin/env python
"""Write parameters_<stage>.md next to archived JSON — same text the memory analysts send to the LLM.

Run from repository root (requires ``data/sample/characteristics/raw_characteristics.json``
and tunnel ``raw_characteristics.json`` under ``data/ablation/memory/<tunnel_id>/characteristics/``):

  ./venv/bin/python skills/scripts/export_llm_parameter_context.py 1-4

Repo root is prepended to ``sys.path`` automatically.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    # This file lives at repo/skills/scripts/… → repo root is parents[2].
    return Path(__file__).resolve().parents[2]


def _load_analyst(relpath: str):
    root = _repo_root()
    path = root / relpath
    name = f"ablation_{path.parent.name}_analyst"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "1-4"
    out_dir = root / "configurable" / "ablation" / "memory" / "parameters" / tunnel_id
    if not out_dir.is_dir():
        print(f"Missing parameters directory: {out_dir}", file=sys.stderr)
        return 1

    base = "configurable/ablation/memory/agents"
    stages: list[tuple[str, str, str]] = [
        (f"{base}/unfolding/analyst.py", "UnfoldingAnalyser", "parameters_unfolding.md"),
        (f"{base}/denoising/analyst.py", "DenoisingAnalyser", "parameters_denoising.md"),
        (f"{base}/enhancing/analyst.py", "EnhancingAnalyser", "parameters_enhancing.md"),
        (f"{base}/detecting/analyst.py", "DetectingAnalyser", "parameters_detecting.md"),
        (f"{base}/segmenting/analyst.py", "SegmentingAnalyser", "parameters_sam.md"),
    ]

    header = f"""# Memory-ablation LLM context — tunnel `{tunnel_id}`

This document is the **same user message** the memory-ablation stage analyst builds (raw characteristics only). Use it for copy-paste into any chat or API.

Regenerate after updating raw characteristics or the tunnel archive under `configurable/ablation/memory/parameters/<tunnel_id>/` (else falls back to `configurable/sample/`):

```bash
./venv/bin/python skills/scripts/export_llm_parameter_context.py {tunnel_id}
```

---

"""

    for rel, cls_name, filename in stages:
        mod = _load_analyst(rel)
        cls = getattr(mod, cls_name)
        body = cls(tunnel_id).build_llm_prompt_markdown()
        out_path = out_dir / filename
        out_path.write_text(header + body + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
