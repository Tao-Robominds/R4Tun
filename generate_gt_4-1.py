#!/usr/bin/env python3
"""Generate detected_gt.csv for tunnel 4-1 from final.csv"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p4tun.generate_detected_from_gt import generate_detected_from_gt

if __name__ == "__main__":
    generate_detected_from_gt("4-1", base_dir="data", n_segments=9, out_name="detected_gt.csv")
    print("Generated detected_gt.csv for tunnel 4-1")
