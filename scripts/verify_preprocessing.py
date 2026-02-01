#!/usr/bin/env python3
"""
Verify 1_preprocessing produces identical enhanced.csv to the original pipeline.

Usage:
  1. Backs up enhanced.csv for each tunnel
  2. Runs 1_preprocessing on each tunnel
  3. Compares new enhanced.csv with backup

Run from project root with: venv/bin/python3 scripts/verify_preprocessing.py
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TUNNELS = ["1-4", "2-2", "3-1", "4-1", "5-1"]
BACKUP_SUFFIX = ".backup_before_preprocessing"
PYTHON = PROJECT_ROOT / "venv" / "bin" / "python3"
PREPROCESSING_SCRIPT = PROJECT_ROOT / "p4tun" / "1_preprocessing.py"


def backup_enhanced(tunnel_id: str) -> bool:
    """Backup enhanced.csv for a tunnel. Returns True if backup was made."""
    tunnel_dir = DATA_DIR / tunnel_id
    enhanced_path = tunnel_dir / "enhanced.csv"
    backup_path = tunnel_dir / f"enhanced.csv{BACKUP_SUFFIX}"
    if not enhanced_path.exists():
        print(f"  [SKIP] No enhanced.csv for {tunnel_id}")
        return False
    shutil.copy2(enhanced_path, backup_path)
    print(f"  [OK] Backed up enhanced.csv -> enhanced.csv{BACKUP_SUFFIX}")
    return True


def run_preprocessing(tunnel_id: str) -> bool:
    """Run 1_preprocessing on a tunnel. Returns True on success."""
    result = subprocess.run(
        [str(PYTHON), str(PREPROCESSING_SCRIPT), tunnel_id],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  [FAIL] 1_preprocessing failed: {result.stderr[:500]}")
        return False
    print(f"  [OK] 1_preprocessing completed")
    return True


def compare_csv(backup_path: Path, new_path: Path, rtol: float = 1e-9, atol: float = 1e-12) -> tuple[bool, str]:
    """
    Compare two CSV files. Returns (match: bool, message: str).
    Uses approximate comparison for float columns.
    """
    df1 = pd.read_csv(backup_path)
    df2 = pd.read_csv(new_path)

    if df1.shape != df2.shape:
        return False, f"Shape mismatch: {df1.shape} vs {df2.shape}"

    if list(df1.columns) != list(df2.columns):
        return False, f"Column mismatch: {set(df1.columns) ^ set(df2.columns)}"

    for col in df1.columns:
        s1, s2 = df1[col], df2[col]
        if pd.api.types.is_numeric_dtype(s1) or pd.api.types.is_numeric_dtype(s2):
            # Numeric: use allclose
            mask1 = pd.isna(s1)
            mask2 = pd.isna(s2)
            if mask1.any() or mask2.any():
                if (mask1 != mask2).any():
                    return False, f"Column {col}: NA mismatch"
                valid = ~mask1 & ~mask2
                if not np.allclose(s1[valid], s2[valid], rtol=rtol, atol=atol, equal_nan=True):
                    diff = np.abs(s1[valid].astype(float) - s2[valid].astype(float))
                    max_diff = diff.max()
                    return False, f"Column {col}: max diff = {max_diff}"
        else:
            # Object/string: exact match
            if not s1.equals(s2):
                return False, f"Column {col}: value mismatch"

    return True, "Identical (within float tolerance)"


def main():
    print("=" * 60)
    print("Verify 1_preprocessing produces same enhanced.csv")
    print("=" * 60)

    if not PYTHON.exists():
        print(f"Error: {PYTHON} not found")
        sys.exit(1)

    results = []

    for tunnel_id in TUNNELS:
        print(f"\n--- Tunnel {tunnel_id} ---")
        tunnel_dir = DATA_DIR / tunnel_id
        raw_path = DATA_DIR / f"{tunnel_id}.txt"

        if not raw_path.exists():
            print(f"  [SKIP] No raw data: {raw_path}")
            results.append((tunnel_id, "SKIP", "No raw data"))
            continue

        # Backup
        if not backup_enhanced(tunnel_id):
            results.append((tunnel_id, "SKIP", "No existing enhanced.csv"))
            continue

        # Run preprocessing
        if not run_preprocessing(tunnel_id):
            results.append((tunnel_id, "FAIL", "Preprocessing failed"))
            continue

        # Compare
        backup_path = tunnel_dir / f"enhanced.csv{BACKUP_SUFFIX}"
        new_path = tunnel_dir / "enhanced.csv"
        match, msg = compare_csv(backup_path, new_path)
        results.append((tunnel_id, "PASS" if match else "FAIL", msg))
        print(f"  [{'PASS' if match else 'FAIL'}] {msg}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for tunnel_id, status, msg in results:
        symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "−"
        print(f"  {symbol} {tunnel_id}: {status} - {msg}")

    failed = sum(1 for _, s, _ in results if s == "FAIL")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\nPassed: {passed}, Failed: {failed}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
