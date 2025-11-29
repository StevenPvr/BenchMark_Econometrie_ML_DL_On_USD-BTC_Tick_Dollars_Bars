"""Utility modules for feature analysis.

This package contains:
- parallel.py: Parallelization wrappers using joblib
- numba_funcs.py: JIT-compiled functions for performance
- plotting.py: Visualization utilities (Plotly + Matplotlib)
"""

from src.analyse_features.utils.parallel import (
    parallel_apply,
    parallel_map,
    chunked_parallel,
    get_n_jobs,
)

__all__ = [
    "parallel_apply",
    "parallel_map",
    "chunked_parallel",
    "get_n_jobs",
]
