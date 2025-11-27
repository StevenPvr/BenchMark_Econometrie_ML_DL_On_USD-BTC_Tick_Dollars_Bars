"""Configuration dataclasses for label_meta module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.constants import DEFAULT_RANDOM_STATE


@dataclass
class MetaModelConfig:
    """Configuration for meta-model training.

    Attributes:
        params: LightGBM parameters (loaded from label_primaire optimization).
        random_state: Random seed.
    """

    params: dict[str, Any] = field(default_factory=dict)
    random_state: int = DEFAULT_RANDOM_STATE


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking a primary model.

    Attributes:
        primary_model_name: Name of primary model to benchmark.
        n_trials: Optuna trials for primary model optimization.
        n_splits: Number of CV splits.
        purge_gap: Gap between train/test in CV.
        skip_primary_optim: Skip primary model optimization.
        random_state: Random seed.
        verbose: Enable logging.
    """

    primary_model_name: str = "lightgbm"
    n_trials: int = 50
    n_splits: int = 5
    purge_gap: int = 5
    skip_primary_optim: bool = False
    random_state: int = DEFAULT_RANDOM_STATE
    verbose: bool = True


@dataclass
class BenchmarkResult:
    """Result from benchmarking a primary model.

    Attributes:
        primary_model_name: Name of the primary model.
        primary_params: Optimized primary model parameters.
        meta_params: Meta-model parameters (fixed, reused).
        primary_model: Trained primary model.
        meta_model: Pre-trained meta-model.
        train_predictions: Primary predictions on train.
        test_predictions: Primary predictions on test.
        meta_predictions: Meta-model arbitrage on test (0, 1).
        final_predictions: test_predictions * meta_predictions.
        train_metrics: Metrics on train split.
        test_metrics: Metrics on test split (primary only).
        final_metrics: Metrics on test split (after meta arbitrage).
    """

    primary_model_name: str
    primary_params: dict[str, Any]
    meta_params: dict[str, Any]
    primary_model: Any
    meta_model: Any
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    meta_predictions: np.ndarray
    final_predictions: np.ndarray
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    final_metrics: dict[str, float]
