"""Label Meta - Meta-Labeling & Benchmarking Module.

This module handles:
    1. Meta-labeling (De Prado methodology)
    2. Training meta-model ONCE for reuse
    3. Benchmarking primary models with meta-arbitrage

Usage:
    python -m src.label_meta.main

Prerequisites:
    Run 'python -m src.label_primaire.main' first to generate labels.

The workflow:
    1. label_primaire optimizes labeling + LightGBM params
    2. label_primaire generates triple-barrier labels on full dataset
    3. label_meta trains meta-model ONCE using LightGBM params
    4. label_meta benchmarks each primary model with meta-arbitrage
"""

from src.label_meta.benchmark import compute_metrics, run_benchmark_for_primary
from src.label_meta.config import BenchmarkConfig, BenchmarkResult, MetaModelConfig
from src.label_meta.meta_labeling import (
    compute_sharpe_ratio,
    compute_strategy_metrics,
    compute_strategy_returns,
    get_meta_features,
    get_meta_labels,
)
from src.label_meta.meta_model import (
    load_meta_model,
    save_meta_model,
    train_meta_model_once,
)

__all__ = [
    # Meta-labeling
    "get_meta_labels",
    "get_meta_features",
    "compute_strategy_returns",
    "compute_sharpe_ratio",
    "compute_strategy_metrics",
    # Meta-model
    "train_meta_model_once",
    "save_meta_model",
    "load_meta_model",
    # Benchmarking
    "run_benchmark_for_primary",
    "compute_metrics",
    # Configuration
    "MetaModelConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
]
