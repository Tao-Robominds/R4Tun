#!/usr/bin/env python3
"""
Extract a ~100MB subsection from each sub-tunnel file in data/points.

For each file (e.g. 1-2, 1-3, 1-5, 2-2, 3-1-1, 4-1, 5-1, ...), write a ~100MB subset
to data/subsets/<same_name>.txt. Run pipeline with tunnel_id "subsets/1-2", etc.

Skips 3-1.txt (the huge combined file); processes 3-1-1, 3-1-2, 3-1-3.
"""

import os
import argparse
import glob

# Default paths
DATA_POINTS = "data/points"
DATA_SUBSETS = "data/subsets"
TARGET_BYTES = 100 * 1024 * 1024  # 100 MiB
BYTES_PER_LINE = 54  # approximate (6 columns scientific + spaces + newline)
SKIP_FILES = {"3-1.txt"}  # single huge combined file, use 3-1-1/3-1-2/3-1-3 instead


def extract_one_file(
    src_path: str,
    out_path: str,
    target_bytes: int = TARGET_BYTES,
    bytes_per_line: int = BYTES_PER_LINE,
) -> None:
    """
    Extract a ~100MB subsection from one point file (subsample if larger than target).
    """
    size = os.path.getsize(src_path)
    n_lines_approx = size // bytes_per_line
    target_lines = target_bytes // bytes_per_line

    if n_lines_approx <= target_lines:
        step = 1
    else:
        step = max(1, n_lines_approx // target_lines)

    n_kept = 0
    with open(src_path, "r") as f_in, open(out_path, "w") as f_out:
        for j, line in enumerate(f_in):
            if j % step == 0:
                f_out.write(line)
                n_kept += 1

    out_size = os.path.getsize(out_path)
    name = os.path.basename(src_path)
    print(f"  {name} -> {os.path.basename(out_path)}  "
          f"({n_kept} points, {out_size / (1024*1024):.1f} MiB, step={step})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ~100MB subsection from each sub-tunnel in data/points into data/subsets."
    )
    parser.add_argument(
        "--points-dir",
        default=DATA_POINTS,
        help=f"Directory containing full tunnel .txt files (default: {DATA_POINTS})",
    )
    parser.add_argument(
        "--subsets-dir",
        default=DATA_SUBSETS,
        help=f"Output directory (default: {DATA_SUBSETS})",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=100,
        help="Target size per subset in MiB (default: 100)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process only this file (e.g. 1-2 or 1-2.txt). Default: all .txt in points-dir.",
    )
    args = parser.parse_args()

    target_bytes = int(args.target_mb * 1024 * 1024)
    points_dir = args.points_dir
    subsets_dir = args.subsets_dir
    os.makedirs(subsets_dir, exist_ok=True)

    if args.file:
        base = args.file if args.file.endswith(".txt") else args.file + ".txt"
        candidates = [os.path.join(points_dir, base)]
    else:
        candidates = sorted(glob.glob(os.path.join(points_dir, "*.txt")))

    to_process = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        if name in SKIP_FILES:
            print(f"Skipping {name} (use 3-1-1, 3-1-2, 3-1-3 instead).")
            continue
        to_process.append(path)

    print(f"Extracting ~{args.target_mb} MiB subsection per file into {subsets_dir}")
    print(f"Processing {len(to_process)} files.\n")

    for src_path in to_process:
        name = os.path.basename(src_path)
        out_path = os.path.join(subsets_dir, name)
        extract_one_file(src_path, out_path, target_bytes=target_bytes)

    print("\nDone. Run pipeline with tunnel_id e.g. 'subsets/1-2' (input: data/subsets/1-2.txt).")


if __name__ == "__main__":
    main()
