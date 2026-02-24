"""
Relabel detection ring indices to match GT order (by Hungarian matching on K positions).
Run after detection so that ring i in all_segments.csv = the physical ring at GT index i.

Usage:
  python -m p4tun.fix_ring_order 4-1 --data-dir data
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from bo.complex_staggered.run_detection_bo import fix_ring_order_to_gt


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fix ring order in all_segments.csv to match GT")
    p.add_argument("tunnel_id", help="e.g. 4-1")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--segments", default="all_segments.csv")
    p.add_argument("--output", default=None, help="default: overwrite segments file")
    args = p.parse_args()
    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    df = fix_ring_order_to_gt(
        tunnel_dir,
        segments_file=args.segments,
        gt_file="all_segments_gt.csv",
        output_file=args.output or args.segments,
    )
    if len(df):
        print(f"Fixed ring order: {os.path.join(tunnel_dir, args.output or args.segments)} ({len(df)} rows)")
    else:
        print("No segments or GT found; nothing written.")
        sys.exit(1)


if __name__ == "__main__":
    main()
