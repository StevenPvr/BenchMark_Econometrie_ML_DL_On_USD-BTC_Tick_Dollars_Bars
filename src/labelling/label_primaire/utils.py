"""
Utilities for Primary Label module.

Contains:
- Model registry and search spaces
- Data loading functions
- Volatility estimation
- Triple barrier helpers
- Dataclasses for configuration and results
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast, Dict, List, Tuple, Type

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.constants import DEFAULT_RANDOM_STATE
from src.model.base import BaseModel
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    DOLLAR_BARS_PARQUET,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Risk/reward ratio for profit-taking (pt_mult = RISK_REWARD_RATIO * sl_mult)
RISK_REWARD_RATIO: float = 1.5


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Machine Learning - Gradient Boosting
    "xgboost": {
        "class": "src.model.xgboost_model.XGBoostModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("int", [200, 1400, 50]),
            "max_depth": ("int", [3, 12, 1]),
            "learning_rate": ("float", [0.005, 0.2, "log"]),
            "subsample": ("float", [0.5, 1.0]),
            "colsample_bytree": ("float", [0.5, 1.0]),
            "reg_alpha": ("float", [1e-4, 1.0, "log"]),
            "reg_lambda": ("float", [1e-4, 1.0, "log"]),
        },
    },
    "random_forest": {
        "class": "src.model.random_forest_model.RandomForestModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("int", [200, 1600, 100]),
            "max_depth": ("int", [5, 24, 1]),
            "min_samples_split": ("int", [2, 12, 1]),
            "min_samples_leaf": ("int", [1, 8, 1]),
            "max_features": ("float", [0.4, 1.0]),
        },
    },
    # Econometric - Linear
    "ridge": {
        "class": "src.model.ridge_classifier.RidgeClassifierModel",
        "dataset": "linear",
        "search_space": {
            "alpha": ("float", [1e-4, 1e3, "log"]),
            "fit_intercept": ("categorical", [True, False]),
        },
    },
    "logistic": {
        "class": "src.model.logistic_classifier.LogisticClassifierModel",
        "dataset": "linear",
        "search_space": {
            # C: inverse regularization (higher = less regularization)
            # For financial data: test range from strong reg (1e-3) to weak (100.0)
            "C": ("float", [1e-3, 300.0, "log"]),
        },
    },
    # Deep Learning
    "lstm": {
        "class": "src.model.lstm_model.LSTMModel",
        "dataset": "lstm",
        "search_space": {
            "hidden_size": ("int", [32, 256, 16]),
            "num_layers": ("int", [1, 4, 1]),
            "dropout": ("float", [0.0, 0.5]),
            "learning_rate": ("float", [1e-4, 1e-2, "log"]),
            "batch_size": ("int", [32, 128, 16]),
            "sequence_length": ("int", [10, 60, 5]),
        },
    },
}

# Focal loss search space for class imbalance handling
# gamma: focusing parameter (higher = more focus on hard examples)
# use_focal_loss: whether to use focal loss vs standard cross-entropy
FOCAL_LOSS_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = {
    # Focusing parameter: 0=CE, 1=light, 2=standard, 3-5=aggressive
    "focal_gamma": ("categorical", [1.0, 2.0, 3.0]),
    # Whether to use focal loss (False = standard loss with class weights)
    "use_focal_loss": ("categorical", [True]),
}

# Minimum prediction ratio for minority classes (guard against degenerate models)
# If model predicts less than this fraction of long+short, apply penalty
MIN_MINORITY_PREDICTION_RATIO: float = 0.1  # Allow rarer signals (was 10%)

# Hard floor on per-class F1 during CV; below this a fold is pruned
MIN_PER_CLASS_F1_REQUIRED: float = 0.0

# =============================================================================
# COMPOSITE SCORE WEIGHTS
# =============================================================================
# The composite score replaces MCC + F1_weighted with a more interpretable metric:
#   score = α * F1_directional + β * F1_neutral + δ * F1_min - γ * sign_error_rate
#
# Where:
# - F1_directional = mean(F1_-1, F1_+1) : ability to predict direction
# - F1_neutral = F1_0 : ability to identify neutral/no-trade zones
# - F1_min = min(F1_-1, F1_0, F1_+1): forces learning of all 3 classes
# - sign_error_rate = (predict +1 when true -1 OR predict -1 when true +1) / total
#
# Sign errors are the worst: they mean trading in the wrong direction!

COMPOSITE_SCORE_WEIGHTS: Dict[str, float] = {
    "alpha": 0.45,  # Weight for directional F1 (most important: predict direction)
    "beta": 0.20,   # Weight for neutral F1 (avoid false signals)
    "delta": 0.25,  # Weight for the worst class F1 to force learning all classes
    "gamma": 0.20,  # Softer sign error penalty (avoid fleeing to class 0)
}

# Models that support focal loss (gradient boosting with custom objectives)
FOCAL_LOSS_SUPPORTED_MODELS: List[str] = []

# Models that support class_weight parameter directly
# Note: Ridge (sklearn.linear_model.RidgeClassifier) does not support class_weight
CLASS_WEIGHT_SUPPORTED_MODELS: List[str] = [
    "random_forest",
    "logistic",
    "ridge",
]


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for primary model optimization.

    Note: Labels are pre-calculated by relabel_datasets.py. This optimization
    only tunes model hyperparameters, not triple-barrier parameters.
    """

    model_name: str
    n_trials: int = 50
    n_splits: int = 5
    min_train_size: int = 500
    data_fraction: float = 1  # Use full dataset (labels are pre-calculated)
    random_state: int = DEFAULT_RANDOM_STATE
    timeout: int | None = None
    early_stopping_rounds: int = 50
    # Focal loss configuration
    use_focal_loss: bool = False  # Focal loss disabled (no supported models in pipeline)
    focal_gamma: float = 2.0  # Kept for compatibility
    optimize_focal_params: bool = False  # No focal search by default
    # Class weight configuration
    use_class_weights: bool = True  # Use balanced class weights


@dataclass
class CompositeScoreMetrics:
    """Metrics for the composite score computation.

    The composite score is designed for primary model optimization:
    - Rewards correct directional predictions (F1 for +1 and -1)
    - Rewards correct neutral identification (F1 for 0)
    - Rewards the worst-performing class (F1_min) to force learning all 3 classes
    - Penalizes sign errors (predicting +1 when true is -1, or vice versa)
    """

    f1_minus1: float  # F1 for class -1 (Short)
    f1_zero: float    # F1 for class 0 (Neutral)
    f1_plus1: float   # F1 for class +1 (Long)
    f1_min: float     # Worst per-class F1 (enforces learning of all classes)
    sign_error_rate: float  # Rate of sign errors (worst kind of error)
    composite_score: float  # Final composite score
    n_sign_errors: int  # Count of sign errors
    n_total: int  # Total predictions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "f1_minus1": self.f1_minus1,
            "f1_zero": self.f1_zero,
            "f1_plus1": self.f1_plus1,
            "f1_min": self.f1_min,
            "sign_error_rate": self.sign_error_rate,
            "composite_score": self.composite_score,
            "n_sign_errors": self.n_sign_errors,
            "n_total": self.n_total,
        }


def compute_composite_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Dict[str, float] | None = None,
) -> CompositeScoreMetrics:
    """
    Compute composite score for primary model optimization.

    The score combines:
    - F1 for directional classes (-1, +1): ability to predict market direction
    - F1 for neutral class (0): ability to avoid false signals
    - Minimum F1 across classes to prevent collapse of one class
    - Sign error penalty: trading in the wrong direction is the worst error

    Formula:
        score = α * F1_directional + β * F1_neutral + δ * F1_min - γ * sign_error_rate

    Where:
        - F1_directional = (F1_-1 + F1_+1) / 2
        - F1_min = min(F1_-1, F1_0, F1_+1)
        - sign_error_rate = (predict +1 when true -1 + predict -1 when true +1) / total

    Parameters
    ----------
    y_true : np.ndarray
        True labels (-1, 0, +1).
    y_pred : np.ndarray
        Predicted labels (-1, 0, +1).
    weights : Dict[str, float] | None
        Override weights for alpha, beta, delta, gamma. Uses COMPOSITE_SCORE_WEIGHTS if None.

    Returns
    -------
    CompositeScoreMetrics
        Dataclass containing all metrics and the final composite score.
    """
    from sklearn.metrics import f1_score as sklearn_f1_score

    if weights is None:
        weights = COMPOSITE_SCORE_WEIGHTS

    alpha = weights.get("alpha", 0.5)
    beta = weights.get("beta", 0.2)
    delta = weights.get("delta", 0.0)
    gamma = weights.get("gamma", 0.3)

    n_total = len(y_true)

    # Compute F1 per class (with zero_division handling)
    # labels=[-1, 0, 1] ensures consistent ordering even if some classes are missing
    f1_per_class = sklearn_f1_score(
        y_true, y_pred,
        labels=[-1, 0, 1],
        average=None,  # type: ignore[arg-type]  # None is valid for per-class scores
        zero_division=0.0  # type: ignore[arg-type]  # float is valid for zero_division
    )

    # f1_per_class is ndarray when average=None
    f1_minus1 = float(f1_per_class[0])  # type: ignore[index]  # ndarray indexing
    f1_zero = float(f1_per_class[1])    # type: ignore[index]  # ndarray indexing
    f1_plus1 = float(f1_per_class[2])   # type: ignore[index]  # ndarray indexing
    f1_min = min(f1_minus1, f1_zero, f1_plus1)

    # Compute sign errors: predict +1 when true -1, or predict -1 when true +1
    # These are the worst errors: we would trade in the opposite direction!
    sign_errors_mask = (
        ((y_pred == 1) & (y_true == -1)) |  # Predicted Long, was Short
        ((y_pred == -1) & (y_true == 1))    # Predicted Short, was Long
    )
    n_sign_errors = int(np.sum(sign_errors_mask))
    sign_error_rate = n_sign_errors / n_total if n_total > 0 else 0.0

    # Compute composite score
    f1_directional = (f1_minus1 + f1_plus1) / 2.0
    composite_score = (
        alpha * f1_directional +
        beta * f1_zero +
        delta * f1_min -
        gamma * sign_error_rate
    )

    return CompositeScoreMetrics(
        f1_minus1=f1_minus1,
        f1_zero=f1_zero,
        f1_plus1=f1_plus1,
        f1_min=f1_min,
        sign_error_rate=sign_error_rate,
        composite_score=composite_score,
        n_sign_errors=n_sign_errors,
        n_total=n_total,
    )


@dataclass
class OptimizationResult:
    """Result of the optimization process.

    Note: Triple-barrier parameters are no longer optimized here. They are
    pre-computed by relabel_datasets.py. This result contains only model
    hyperparameters and focal loss configuration.
    """

    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    metric: str
    n_trials: int
    cv_scores: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    best_focal_loss_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_focal_loss_params": self.best_focal_loss_params,
            "best_score": self.best_score,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "cv_scores": self.cv_scores,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved to {path}")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_model_class(model_name: str) -> Type[BaseModel]:
    """Dynamically load a model class from the registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    class_path = MODEL_REGISTRY[model_name]["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_dataset_for_model(model_name: str) -> pd.DataFrame:
    """Load the appropriate feature dataset for a model type."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    dataset_type = MODEL_REGISTRY[model_name]["dataset"]

    path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }

    path = path_map.get(dataset_type)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Dataset not found for {dataset_type}: {path}")

    logger.info(f"Loading dataset for {model_name}: {dataset_type} -> {path.name}")
    return pd.read_parquet(path)


def load_dollar_bars() -> pd.DataFrame:
    """Load dollar bars with close prices."""
    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(f"Dollar bars not found: {DOLLAR_BARS_PARQUET}")

    bars = pd.read_parquet(DOLLAR_BARS_PARQUET)

    if "datetime_close" not in bars.columns:
        raise ValueError("Dollar bars must have 'datetime_close' column")

    return bars.set_index("datetime_close").sort_index()


# =============================================================================
# VOLATILITY ESTIMATION (De Prado - getDailyVol)
# =============================================================================


def get_daily_volatility(
    close: pd.Series,
    span: int = 100,
    min_periods: int = 10,
) -> pd.Series:
    """
    Compute daily volatility using exponential weighted moving standard deviation.

    This is the getDailyVol function from De Prado's methodology.

    Parameters
    ----------
    close : pd.Series
        Close prices with datetime index.
    span : int, default=100
        Span for exponential weighting.
    min_periods : int, default=10
        Minimum observations required.

    Returns
    -------
    pd.Series
        Daily volatility estimates.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.ewm(span=span, min_periods=min_periods).std()
