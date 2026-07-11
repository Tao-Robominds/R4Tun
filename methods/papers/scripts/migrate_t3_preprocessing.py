#!/usr/bin/env python3
"""Replace broken T3 vendor snapshots under data/ablation_anthropic.

Sources (external drive by default):
  3-1-1 <- ablation_gpt
  3-1-2, 3-1-3 <- ablation_gemini

Usage:
    ./venv/bin/python methods/papers/scripts/migrate_t3_preprocessing.py --dry-run
    ./venv/bin/python methods/papers/scripts/migrate_t3_preprocessing.py --execute
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_SOURCE_ROOT = Path("/media/boringtao/Ezekers/R4Tun/data")
DEST_ROOT = REPO_ROOT / "data" / "ablation_anthropic" / "memory+state+knowledge"
SAMPLE_SRC = DEFAULT_SOURCE_ROOT / "sample" / "characteristics"
SAMPLE_DEST = REPO_ROOT / "data" / "sample" / "characteristics"

MIGRATIONS: dict[str, tuple[str, str]] = {
    "3-1-1": ("ablation_gpt", "memory+state+knowledge/3-1-1"),
    "3-1-2": ("ablation_gemini", "memory+state+knowledge/3-1-2"),
    "3-1-3": ("ablation_gemini", "memory+state+knowledge/3-1-3"),
}

REQUIRED_FILES = (
    "enhanced.csv",
    "depth_map.png",
    "pixel_to_point.pkl",
    "ring_count.txt",
    "unwrapped.csv",
    "denoised.csv",
)


def source_dir(source_root: Path, tunnel: str) -> Path:
    vendor, rel = MIGRATIONS[tunnel]
    return source_root / vendor / rel


def dest_dir(tunnel: str) -> Path:
    return DEST_ROOT / tunnel


def validate_tree(path: Path, tunnel: str) -> list[str]:
    errors: list[str] = []
    if not path.is_dir():
        errors.append(f"{tunnel}: missing directory {path}")
        return errors
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            errors.append(f"{tunnel}: missing {name}")
    dm = path / "depth_map.png"
    if dm.is_file() and dm.stat().st_size < 4_000_000:
        errors.append(
            f"{tunnel}: depth_map.png suspiciously small ({dm.stat().st_size} bytes)"
        )
    enh = path / "enhanced.csv"
    if enh.is_file() and enh.stat().st_size < 100_000_000:
        errors.append(
            f"{tunnel}: enhanced.csv suspiciously small ({enh.stat().st_size} bytes)"
        )
    return errors


def ensure_sample_characteristics(source_root: Path) -> None:
    """Copy SAM4Tun sample characteristics if missing (required by orchestrator)."""
    src = source_root / "sample" / "characteristics"
    if not src.is_dir():
        src = SAMPLE_SRC
    if not src.is_dir():
        return
    SAMPLE_DEST.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.json"):
        dst = SAMPLE_DEST / f.name
        if not dst.is_file():
            shutil.copy2(f, dst)


def ensure_memory_raw_characteristics(tunnel: str) -> None:
    """Ensure data/ablation/memory/{tunnel}/characteristics/raw_characteristics.json exists."""
    mem_dir = REPO_ROOT / "data" / "ablation" / "memory" / tunnel / "characteristics"
    dst = mem_dir / "raw_characteristics.json"
    if dst.is_file():
        return
    for vendor in ("anthropic", "gpt", "gemini"):
        src = REPO_ROOT / "data" / f"ablation_{vendor}" / "memory" / tunnel / "characteristics" / "raw_characteristics.json"
        if src.is_file():
            mem_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
    ext = DEFAULT_SOURCE_ROOT / "ablation_gpt" / "memory" / tunnel / "characteristics" / "raw_characteristics.json"
    if ext.is_file():
        mem_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ext, dst)


def dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def migrate_one(
    tunnel: str,
    source_root: Path,
    execute: bool,
) -> dict:
    src = source_dir(source_root, tunnel)
    dst = dest_dir(tunnel)
    row = {
        "tunnel": tunnel,
        "source": str(src),
        "dest": str(dst),
        "src_bytes": dir_size(src),
        "dst_bytes_before": dir_size(dst) if dst.exists() else 0,
        "status": "pending",
        "errors": validate_tree(src, tunnel),
    }
    if row["errors"]:
        row["status"] = "source_invalid"
        return row

    if not execute:
        row["status"] = "dry_run_ok"
        return row

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    post_errors = validate_tree(dst, tunnel)
    row["dst_bytes_after"] = dir_size(dst)
    row["status"] = "ok" if not post_errors else "post_copy_invalid"
    row["post_errors"] = post_errors
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate T3 preprocessing into ablation_anthropic")
    parser.add_argument("--dry-run", action="store_true", help="Validate only (default)")
    parser.add_argument("--execute", action="store_true", help="Delete dest and copytree from source")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root containing ablation_gpt/ and ablation_gemini/ (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument("--tunnel", choices=list(MIGRATIONS), action="append", dest="tunnels")
    args = parser.parse_args()

    if not args.execute and not args.dry_run:
        args.dry_run = True

    tunnels = args.tunnels or list(MIGRATIONS)
    source_root = args.source_root.expanduser().resolve()

    if not source_root.is_dir():
        print(f"ERROR: source root not found: {source_root}", file=sys.stderr)
        print("Mount external drive or pass --source-root", file=sys.stderr)
        sys.exit(1)

    ensure_sample_characteristics(source_root)

    results = [migrate_one(t, source_root, args.execute) for t in tunnels]
    log_path = REPO_ROOT / "logs" / "t3_hint_loop" / "migrate_preprocessing.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "source_root": str(source_root),
        "results": results,
    }
    log_path.write_text(json.dumps(payload, indent=2) + "\n")

    for r in results:
        print(f"\n{tunnel_label(r['tunnel'])}")
        print(f"  source: {r['source']}")
        print(f"  src_bytes: {r['src_bytes']:,}")
        print(f"  status: {r['status']}")
        for e in r.get("errors", []) + r.get("post_errors", []):
            print(f"  ! {e}")

    failed = [r for r in results if r["status"] not in ("dry_run_ok", "ok")]
    print(f"\nLog: {log_path}")
    if failed:
        sys.exit(1)


def tunnel_label(tunnel: str) -> str:
    return f"=== {tunnel} ==="


if __name__ == "__main__":
    main()
