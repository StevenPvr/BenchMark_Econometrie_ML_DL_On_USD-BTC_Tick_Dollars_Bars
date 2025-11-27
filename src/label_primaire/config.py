"""Configuration dataclasses for label_primaire module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.constants import DEFAULT_RANDOM_STATE


@dataclass
class LabelingHyperparams:
    """Hyperparameter search space for labeling optimization.

    Attributes:
        pt_mult_range: Range for profit-taking multiplier [min, max].
        sl_mult_range: Range for stop-loss multiplier [min, max].
        max_holding_range: Range for max holding period [min, max].
        min_ret_range: Range for minimum return threshold [min, max].
        symmetric_barriers: If True, pt_mult = sl_mult (symmetric barriers).
    """

    pt_mult_range: tuple[float, float] = (0.5, 3.0)
    sl_mult_range: tuple[float, float] = (0.5, 3.0)
    max_holding_range: tuple[int, int] = (5, 50)
    min_ret_range: tuple[float, float] = (0.0, 0.001)
    symmetric_barriers: bool = True


@dataclass
class LabelingOptimizationConfig:
    """Configuration for labeling parameter optimization.

    Attributes:
        n_trials: Number of Optuna trials.
        n_splits: Number of CV splits for walk-forward validation.
        purge_gap: Gap between train/test in CV to prevent leakage.
        hyperparams: Search space configuration.
        metric: Optimization metric ("sharpe", "accuracy", "f1", "mcc").
        random_state: Random seed for reproducibility.
        verbose: Enable logging.
    """

    n_trials: int = 50
    n_splits: int = 5
    purge_gap: int = 5
    hyperparams: LabelingHyperparams = field(default_factory=LabelingHyperparams)
    metric: str = "sharpe"
    random_state: int = DEFAULT_RANDOM_STATE
    verbose: bool = True


@dataclass
class LabelingOptimizationResult:
    """Result from labeling parameter optimization.

    Attributes:
        best_params: Best labeling parameters found.
        best_score: Best optimization score.
        study_stats: Optuna study statistics.
        all_trials: List of all trial results.
    """

    best_params: dict[str, Any]
    best_score: float
    study_stats: dict[str, Any] = field(default_factory=dict)
    all_trials: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LabelGenerationConfig:
    """Configuration for label generation.

    Attributes:
        pt_mult: Profit-taking multiplier.
        sl_mult: Stop-loss multiplier.
        max_holding_period: Maximum holding period in bars.
        min_ret: Minimum return threshold for neutral label.
        train_ratio: Ratio of data for training split.
    """

    pt_mult: float = 1.0
    sl_mult: float = 1.0
    max_holding_period: int = 20
    min_ret: float = 0.0
    train_ratio: float = 0.8
