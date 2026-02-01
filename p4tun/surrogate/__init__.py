"""
P4Tun Surrogate Model Pipeline

This module provides a surrogate model approach for fast parameter search
using Gaussian Process models trained on existing BO optimization data.

Pipeline:
1. Data extraction from BO JSON logs
2. GP surrogate training
3. Inverse search for target metrics
4. Validation with full pipeline

Usage:
    from p4tun.surrogate import SurrogatePipeline
    
    pipeline = SurrogatePipeline(stage='detection')
    pipeline.extract_data()
    pipeline.train()
    candidates = pipeline.search(target_miou=0.75)
    validated = pipeline.validate(candidates)
"""

from .data_extractor import DataExtractor
from .gp_surrogate import GPSurrogate
from .inverse_search import InverseSearch
from .validator import Validator
from .pipeline import SurrogatePipeline

__all__ = [
    'DataExtractor',
    'GPSurrogate', 
    'InverseSearch',
    'Validator',
    'SurrogatePipeline',
]
