"""
bo4tun - Bayesian Optimization for Tunnel Segmentation (No-GT)

This module implements a two-layer Bayesian Optimization approach for tunnel
segmentation that can operate without ground truth at runtime:

- Layer A: Hard constraints based on intrinsic metrics (guardrails)
- Layer B: Learned mIoU predictor trained on historical BO data

Submodules:
- training/: Training data extracted from historical BO runs
"""

__version__ = "0.1.0"
