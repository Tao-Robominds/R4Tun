"""Generate per-regime warm-start params with LLM APIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.ablation.scripts._warm_start_schema import (
    DET_PASS_THROUGH_KEYS,
    DET_SCHEMA,
    PRE_SCHEMA,
    build_prompt,
    force_canonical_constraints,
    validate_and_clamp,
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response does not contain JSON object")
    return json.loads(s[start : end + 1])


def _regime_label_from_row(row: pd.Series) -> str:
    k_span_tier = row.get("k_span_tier")
    if pd.isna(k_span_tier) or k_span_tier in ("", None):
        k_span_tier = "na"
    return f"{row['density_tier']}_{row['coverage_tier']}_{k_span_tier}_{row['pattern_type']}"


def _compute_regime_stats(descriptors: pd.DataFrame, regime_label: str) -> Dict[str, Any]:
    tmp = descriptors.copy()
    if "regime_label" not in tmp.columns:
        tmp["regime_label"] = tmp.apply(_regime_label_from_row, axis=1)
    reg = tmp[tmp["regime_label"] == regime_label]
    if reg.empty:
        raise ValueError(f"No descriptors found for regime {regime_label}")
    cols = [
        "n_points",
        "angular_gap_frac",
        "k_angle_deg",
        "k_span_deg",
        "segment_balance_cv",
    ]
    stats = {c: float(reg[c].median()) for c in cols if c in reg.columns}
    stats["regime_count"] = int(len(reg))
    stats["family_modes"] = sorted({str(x) for x in reg.get("family", pd.Series(dtype=str)).dropna().astype(str)})
    return stats


def _load_defaults() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pre = json.loads(
        (REPO_ROOT / "agents" / "1_preprocessing" / "parameters" / "_default_irregular" / "parameters_preprocessing.json").read_text()
    )
    det = json.loads(
        (REPO_ROOT / "agents" / "2_detection" / "parameters" / "_default_irregular" / "parameters_detection.json").read_text()
    )
    return pre, det


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float):
        self.provider = provider
        self.model = model
        self.temperature = temperature

        if provider == "anthropic":
            import anthropic  # type: ignore

            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY missing")
            self.client = anthropic.Anthropic(api_key=key)
        elif provider == "openai":
            from openai import OpenAI  # type: ignore

            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY missing")
            self.client = OpenAI(api_key=key)
        elif provider == "gemini":
            import google.generativeai as genai  # type: ignore

            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY missing")
            genai.configure(api_key=key)
            self.client = genai.GenerativeModel(model_name=model)
        else:
            raise ValueError(f"Unsupported provider {provider}")

    def generate(self, prompt: str) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for b in resp.content:
                txt = getattr(b, "text", None)
                if txt:
                    parts.append(txt)
            return "\n".join(parts).strip()
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        # gemini
        resp = self.client.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        return (getattr(resp, "text", None) or "").strip()


def _provider_default_model(provider: str) -> str:
    if provider == "anthropic":
        return "claude-sonnet-4-6"
    if provider == "openai":
        return "gpt-4o"
    return "gemini-1.5-pro"


def _iter_unique_regimes(panel: Dict[str, Any]) -> List[str]:
    seen = []
    for r in panel.get("rings", []):
        reg = str(r["regime_label"])
        if reg not in seen:
            seen.append(reg)
    return seen


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/ablation/reference_panel.json")
    p.add_argument(
        "--descriptors",
        default="data/subsets/workflow/regime_v1/01_ring_regime_discovery/ring_regimes.csv",
    )
    p.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default="anthropic")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--run-id", default="warm_start_v1")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    panel = json.loads((REPO_ROOT / args.panel).read_text())
    descriptors = pd.read_csv(REPO_ROOT / args.descriptors)
    pre_default, det_default = _load_defaults()

    model = args.model or _provider_default_model(args.provider)
    llm = LLMClient(args.provider, model, args.temperature)

    regimes = _iter_unique_regimes(panel)
    out_root = REPO_ROOT / "methods" / "ablation" / "output" / args.run_id
    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": model,
        "temperature": args.temperature,
        "panel": str((REPO_ROOT / args.panel).resolve()),
        "descriptors": str((REPO_ROOT / args.descriptors).resolve()),
        "regimes": [],
    }

    for regime_label in regimes:
        stats = _compute_regime_stats(descriptors, regime_label)
        prompt = build_prompt(regime_label, stats, pre_default, det_default)
        raw = llm.generate(prompt)
        parsed = _extract_json_object(raw)
        pre_in = parsed.get("preprocessing", {})
        det_in = parsed.get("detection", {})

        pre_out, pre_log = validate_and_clamp(pre_in, PRE_SCHEMA)
        det_out, det_log = validate_and_clamp(det_in, DET_SCHEMA)
        for k in DET_PASS_THROUGH_KEYS:
            if k in det_in and isinstance(det_in[k], dict):
                det_out[k] = det_in[k]
        canonical_log = force_canonical_constraints(pre_out)

        safe_reg = regime_label.replace("/", "_")
        pre_path = (
            REPO_ROOT
            / "agents"
            / "1_preprocessing"
            / "parameters"
            / "_warm_start"
            / safe_reg
            / "parameters_preprocessing.json"
        )
        det_path = (
            REPO_ROOT
            / "agents"
            / "2_detection"
            / "parameters"
            / "_warm_start"
            / safe_reg
            / "parameters_detection.json"
        )
        _write_json(pre_path, pre_out)
        _write_json(det_path, det_out)

        response_meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": args.provider,
            "model": model,
            "temperature": args.temperature,
            "regime_label": regime_label,
            "regime_stats": stats,
            "prompt": prompt,
            "raw_response": raw,
            "parsed": {"preprocessing": pre_out, "detection": det_out},
            "clamp_log": pre_log + det_log + canonical_log,
            "output_paths": {
                "preprocessing": str(pre_path.resolve()),
                "detection": str(det_path.resolve()),
            },
        }
        _write_json(out_root / safe_reg / "llm_response.json", response_meta)
        manifest["regimes"].append(
            {
                "regime_label": regime_label,
                "stats": stats,
                "preprocessing_path": str(pre_path.resolve()),
                "detection_path": str(det_path.resolve()),
                "response_path": str((out_root / safe_reg / "llm_response.json").resolve()),
            }
        )
        print(f"[warm-start] {regime_label} -> params written")

    _write_json(out_root / "manifest.json", manifest)
    print(f"[warm-start] manifest: {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
