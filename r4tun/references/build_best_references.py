#!/usr/bin/env python3
"""Select best mIoU r4tun ablation run per subset tunnel and copy to references.

Copies full pipeline outputs (not moves) from ``r4tun/data/ablation_*`` and the
matching five stage parameter JSONs into ``r4tun/references/data/<tunnel_id>/``.

Run with the project venv only::

    ./venv/bin/python r4tun/references/build_best_references.py --clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

VALID_TUNNEL_IDS = {
    *(f"1-{i}" for i in range(1, 6)),
    *(f"2-{i}" for i in range(1, 6)),
    "3-1-1",
    "3-1-2",
    "3-1-3",
    *(f"4-{i}" for i in range(1, 11)),
    *(f"5-{i}" for i in range(1, 8)),
}

# Tie-break when mIoU equal: lower index wins first.
METHOD_PRIORITY: Dict[str, int] = {
    "ablation_anthropic": 0,
    "ablation_gpt": 1,
    "ablation_gemini": 2,
    "ablation_rules": 3,
}

METHOD_TO_PARAM_SUFFIX: Dict[str, str] = {
    "ablation_rules": "rules",
    "ablation_anthropic": "m_s_k_opus4.6",
    "ablation_gpt": "m_s_k_gpt5.4",
    "ablation_gemini": "m_s_k_gemini3flash",
}

PARAM_ROOT_RULES = REPO_ROOT / "r4tun" / "agents" / "ablation" / "rules" / "parameters"
PARAM_ROOT_MSK = (
    REPO_ROOT / "r4tun" / "agents" / "ablation" / "memory+state+knowledge" / "parameters"
)

STAGES = ("unfolding", "denoising", "enhancing", "detecting", "sam")

REQUIRED_OUTPUT_FILES = [
    "unwrapped.csv",
    "denoised.csv",
    "enhanced.csv",
    "detected.csv",
    "final.csv",
    "only_label.csv",
    "depth_map.png",
    "depth_map_outlier.npy",
    "ring_count.txt",
    "evaluation/performance.md",
]


@dataclass
class Candidate:
    tunnel_id: str
    data_folder: str  # e.g. ablation_gpt
    miou: float
    oa: Optional[float]
    f1: Optional[float]
    data_path: Path

    @property
    def param_suffix(self) -> str:
        return METHOD_TO_PARAM_SUFFIX[self.data_folder]

    def param_dir(self) -> Path:
        if self.data_folder == "ablation_rules":
            return PARAM_ROOT_RULES / self.tunnel_id
        return PARAM_ROOT_MSK / self.tunnel_id

    def param_filenames(self) -> List[str]:
        suf = self.param_suffix
        if suf == "rules":
            return [f"parameters_{st}_rules.json" for st in STAGES]
        return [f"parameters_{st}_{suf}.json" for st in STAGES]


def _parse_performance_md(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    oa = f1 = miou = None
    m = re.search(r"Overall Accuracy \(OA\):\s*([0-9.]+)", text)
    if m:
        oa = float(m.group(1))
    m = re.search(r"F1 Score:\s*([0-9.]+)", text)
    if m:
        f1 = float(m.group(1))
    m = re.search(r"Mean IoU \(mIoU\):\s*([0-9.]+)", text)
    if m:
        miou = float(m.group(1))
    return oa, f1, miou


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checksum_selected_under_outputs(outputs_dir: Path, ref_dir: Path) -> Tuple[Dict[str, str], List[Dict]]:
    """Hash small verification files; hash CSV/pkl only when <= 50 MiB; else record size."""
    large_meta: List[Dict] = []
    # Always hash these (bounded size or critical)
    rel_for_hash = [
        "evaluation/performance.md",
        "ring_count.txt",
        "detected.csv",
        "depth_map.png",
        "depth_map_outlier.npy",
    ]
    sha_dict: Dict[str, str] = {}
    for rel in rel_for_hash:
        p = outputs_dir / rel
        if p.is_file():
            key = str(p.relative_to(ref_dir)).replace("\\", "/")
            sha_dict[key] = _sha256_file(p)

    large_suffixes = (".csv", ".pkl")
    for p in sorted(outputs_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in large_suffixes:
            continue
        if p.name in ("detected.csv",):
            continue
        size = p.stat().st_size
        if size <= 50 * 1024 * 1024:  # <= 50 MiB: hash
            key = str(p.relative_to(ref_dir)).replace("\\", "/")
            if key not in sha_dict:
                sha_dict[key] = _sha256_file(p)
        else:
            key = str(p.relative_to(ref_dir)).replace("\\", "/")
            large_meta.append({"path": key, "size_bytes": size})
    return sha_dict, large_meta


def _checksum_tree_small(root: Path, rel_root: Path) -> Dict[str, str]:
    """SHA-256 all files under root (expected small, e.g. parameters/)."""
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(rel_root)).replace("\\", "/")
        out[rel] = _sha256_file(p)
    return out


def _collect_candidates(data_root: Path) -> List[Candidate]:
    candidates: List[Candidate] = []
    for folder in METHOD_TO_PARAM_SUFFIX:
        base = data_root / folder
        if not base.is_dir():
            continue
        for perf in base.glob("*/evaluation/performance.md"):
            tunnel_id = perf.parts[-3]
            if tunnel_id not in VALID_TUNNEL_IDS:
                continue
            text = perf.read_text(encoding="utf-8")
            oa, f1, miou = _parse_performance_md(text)
            if miou is None:
                print(f"[warn] no mIoU in {perf}", file=sys.stderr)
                continue
            data_path = base / tunnel_id
            candidates.append(
                Candidate(
                    tunnel_id=tunnel_id,
                    data_folder=folder,
                    miou=miou,
                    oa=oa,
                    f1=f1,
                    data_path=data_path,
                )
            )
    return candidates


def _pick_winners(candidates: Iterable[Candidate]) -> Dict[str, Candidate]:
    by_tunnel: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_tunnel.setdefault(c.tunnel_id, []).append(c)

    winners: Dict[str, Candidate] = {}
    for tid in sorted(VALID_TUNNEL_IDS):
        opts = by_tunnel.get(tid, [])
        if not opts:
            raise SystemExit(f"[error] no evaluation candidates for tunnel {tid}")
        best = min(
            opts,
            key=lambda c: (
                -c.miou,
                METHOD_PRIORITY.get(c.data_folder, 99),
                c.data_folder,
            ),
        )
        winners[tid] = best
    return winners


def _verify_outputs(c: Candidate) -> None:
    missing = [rel for rel in REQUIRED_OUTPUT_FILES if not (c.data_path / rel).is_file()]
    if missing:
        raise SystemExit(f"[error] {c.tunnel_id} {c.data_folder} missing: {missing}")


def _verify_params(c: Candidate) -> List[Path]:
    pdir = c.param_dir()
    paths: List[Path] = []
    for name in c.param_filenames():
        p = pdir / name
        if not p.is_file():
            raise SystemExit(f"[error] missing parameter file {p}")
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "r4tun" / "data",
        help="Root containing ablation_* folders",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "r4tun" / "references" / "data",
        help="Output references root",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing --out-root before writing",
    )
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_root: Path = args.out_root

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = _collect_candidates(data_root)
    winners = _pick_winners(candidates)

    manifest_entries: List[Dict] = []

    for tid in sorted(winners):
        w = winners[tid]
        _verify_outputs(w)
        param_paths = _verify_params(w)

        ref_dir = out_root / tid
        outputs_dir = ref_dir / "outputs"
        params_dir = ref_dir / "parameters"

        if ref_dir.exists():
            shutil.rmtree(ref_dir)
        outputs_dir.mkdir(parents=True)
        params_dir.mkdir(parents=True)

        # Full output tree (characteristics, pixel_to_point, etc.)
        for item in w.data_path.iterdir():
            dest = outputs_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        copied_params: Dict[str, str] = {}
        for src in param_paths:
            dst = params_dir / src.name
            shutil.copy2(src, dst)
            copied_params[src.name] = _sha256_file(dst)

        out_checksums, large_outputs = _checksum_selected_under_outputs(outputs_dir, ref_dir)
        param_checksums = _checksum_tree_small(params_dir, ref_dir)

        meta = {
            "tunnel_id": tid,
            "source_data_folder": w.data_folder,
            "source_data_path": str(w.data_path.relative_to(REPO_ROOT)),
            "param_suffix": w.param_suffix,
            "param_source_dir": str(w.param_dir().relative_to(REPO_ROOT)),
            "metrics": {
                "mIoU": w.miou,
                "OA": w.oa,
                "F1": w.f1,
            },
            "candidates_compared": [
                {
                    "data_folder": x.data_folder,
                    "mIoU": x.miou,
                    "OA": x.oa,
                    "F1": x.f1,
                }
                for x in sorted(
                    [c for c in candidates if c.tunnel_id == tid],
                    key=lambda c: (-c.miou, METHOD_PRIORITY.get(c.data_folder, 99)),
                )
            ],
            "parameter_files_copied": list(copied_params.keys()),
            "checksums_sha256": {
                "outputs": out_checksums,
                "parameters": param_checksums,
            },
            "large_output_artifacts": large_outputs,
        }
        (ref_dir / "reference_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

        perf_copy = outputs_dir / "evaluation" / "performance.md"
        _, _, miou_copy = _parse_performance_md(perf_copy.read_text(encoding="utf-8"))
        if miou_copy is None or abs(miou_copy - w.miou) > 1e-9:
            raise SystemExit(
                f"[error] mIoU mismatch for {tid}: winner={w.miou} copy={miou_copy}"
            )

        manifest_entries.append(
            {
                "tunnel_id": tid,
                "winner": w.data_folder,
                "mIoU": w.miou,
                "OA": w.oa,
                "F1": w.f1,
                "param_suffix": w.param_suffix,
                "reference_dir": str(ref_dir.relative_to(REPO_ROOT)),
            }
        )

    manifest = {
        "n_references": len(manifest_entries),
        "expected_subsets": sorted(VALID_TUNNEL_IDS),
        "references": manifest_entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# R4Tun best references (per subset tunnel)",
        "",
        f"Built **{len(manifest_entries)}** references under `{out_root.relative_to(REPO_ROOT)}`.",
        "",
        "| tunnel_id | winner | mIoU | OA | F1 | param_suffix |",
        "|---|---|---:|---:|---:|---|",
    ]
    for e in manifest_entries:
        lines.append(
            f"| {e['tunnel_id']} | {e['winner']} | {e['mIoU']} | "
            f"{e['OA'] if e['OA'] is not None else ''} | "
            f"{e['F1'] if e['F1'] is not None else ''} | {e['param_suffix']} |"
        )
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[references] wrote {len(manifest_entries)} references under {out_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
