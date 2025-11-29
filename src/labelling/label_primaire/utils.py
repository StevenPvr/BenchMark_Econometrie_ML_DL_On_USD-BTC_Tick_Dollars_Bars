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
    DATASET_FEATURES_PARQUET,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DOLLAR_BARS_PARQUET,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Machine Learning - Gradient Boosting
    "lightgbm": {
        "class": "src.model.lightgbm_model.LightGBMModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("categorical", [100, 200, 300]),
            "max_depth": ("categorical", [4, 6, 8]),
            "num_leaves": ("categorical", [31, 63, 95]),
            "learning_rate": ("categorical", [0.02, 0.05, 0.1]),
            "subsample": ("categorical", [0.7, 0.85, 1.0]),
            "colsample_bytree": ("categorical", [0.7, 0.9]),
            "reg_alpha": ("categorical", [1e-3, 1e-2, 0.1, 1.0]),
            "reg_lambda": ("categorical", [1e-3, 1e-2, 0.1, 1.0]),
        },
    },
    "xgboost": {
        "class": "src.model.xgboost_model.XGBoostModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("categorical", [100, 200, 300]),
            "max_depth": ("categorical", [4, 6, 8]),
            "learning_rate": ("categorical", [0.02, 0.05, 0.1]),
            "subsample": ("categorical", [0.7, 0.85, 1.0]),
            "colsample_bytree": ("categorical", [0.7, 0.9]),
            "reg_alpha": ("categorical", [1e-3, 1e-2, 0.1, 1.0]),
            "reg_lambda": ("categorical", [1e-3, 1e-2, 0.1, 1.0]),
        },
    },
    "catboost": {
        "class": "src.model.catboost_model.CatBoostModel",
        "dataset": "tree",
        "search_space": {
            "iterations": ("categorical", [150, 300]),
            "depth": ("categorical", [4, 6, 8]),
            "learning_rate": ("categorical", [0.02, 0.05, 0.1]),
            "l2_leaf_reg": ("categorical", [0.5, 1.0, 3.0]),
            "random_strength": ("categorical", [0.5, 1.0, 1.5]),
            "bagging_temperature": ("categorical", [0.0, 0.5, 1.0]),
        },
    },
    "random_forest": {
        "class": "src.model.random_forest_model.RandomForestModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("categorical", [100, 200, 400]),
            "max_depth": ("categorical", [5, 8, 12]),
            "min_samples_split": ("categorical", [2, 5, 10]),
            "min_samples_leaf": ("categorical", [1, 2, 4]),
            "max_features": ("categorical", ["sqrt", "log2"]),
        },
    },
    # Econometric - Linear
    "ridge": {
        "class": "src.model.ridge_classifier.RidgeClassifierModel",
        "dataset": "linear",
        "search_space": {
            "alpha": ("categorical", [0.1, 1.0, 10.0]),
        },
    },
    "lasso": {
        "class": "src.model.lasso_classifier.LassoClassifierModel",
        "dataset": "linear",
        "search_space": {
            "C": ("categorical", [0.5, 1.0, 5.0]),
            "max_iter": ("categorical", [500, 1000, 2000]),
            "tol": ("categorical", [1e-3, 1e-2]),
        },
    },
    # Deep Learning
    "lstm": {
        "class": "src.model.lstm_model.LSTMModel",
        "dataset": "lstm",
        "search_space": {
            "hidden_size": ("categorical", [32, 64, 96]),
            "num_layers": ("categorical", [1, 2]),
            "dropout": ("categorical", [0.0, 0.2]),
            "learning_rate": ("categorical", [1e-3, 3e-3, 1e-2]),
            "batch_size": ("categorical", [32, 64]),
            "sequence_length": ("categorical", [10, 20, 30]),
        },
    },
}

# Triple barrier search space for primary model
# Adapted for dollar bars (small log returns ~0.01% to 0.1%)
TRIPLE_BARRIER_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = {
    "pt_mult": ("categorical", [0.5, 1.0, 1.5, 2.0, 3.0]),
    "sl_mult": ("categorical", [0.5, 1.0, 1.5, 2.0, 3.0]),
    "min_return": ("categorical", [0.0, 0.00005, 0.0001, 0.0002]),  # 0 to 0.02%
    "max_holding": ("categorical", [10, 20, 50, 100]),
}


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for primary model optimization."""

    model_name: str
    n_trials: int = 50
    n_splits: int = 5
    min_train_size: int = 500
    vol_span: int = 100
    data_fraction: float = 0.5
    random_state: int = DEFAULT_RANDOM_STATE
    timeout: int | None = None


@dataclass
class OptimizationResult:
    """Result of the optimization process."""

    model_name: str
    best_params: Dict[str, Any]
    best_triple_barrier_params: Dict[str, Any]
    best_score: float
    metric: str
    n_trials: int
    cv_scores: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_triple_barrier_params": self.best_triple_barrier_params,
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
        "tree": DATASET_FEATURES_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_PARQUET,
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


# =============================================================================
# BARRIER HELPERS
# =============================================================================


def compute_barriers(
    events: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """
    Compute profit-taking and stop-loss barrier levels.

    Parameters
    ----------
    events : pd.DataFrame
        Events with 'trgt' (volatility threshold) column.
    pt_mult : float
        Profit-taking multiplier.
    sl_mult : float
        Stop-loss multiplier.

    Returns
    -------
    pd.DataFrame
        Events with 'pt' and 'sl' columns added.
    """
    out = events.copy()

    if pt_mult > 0:
        out["pt"] = pt_mult * out["trgt"]
    else:
        out["pt"] = pd.Series(index=events.index, dtype=float)

    if sl_mult > 0:
        out["sl"] = -sl_mult * out["trgt"]
    else:
        out["sl"] = pd.Series(index=events.index, dtype=float)

    return out


def is_valid_barrier(value: Any) -> bool:
    """Check if a barrier value is valid (not None/NaN)."""
    if value is None:
        return False
    if isinstance(value, (int, float)) and pd.isna(value):
        return False
    return True


def find_barrier_touch(
    path_ret: pd.Series,
    pt_barrier: float | None,
    sl_barrier: float | None,
) -> pd.Timestamp | None:
    """
    Find the first barrier touch in a price path.

    Parameters
    ----------
    path_ret : pd.Series
        Returns relative to entry price.
    pt_barrier : float or None
        Profit-taking barrier level.
    sl_barrier : float or None
        Stop-loss barrier level.

    Returns
    -------
    pd.Timestamp or None
        Timestamp of first barrier touch, or None.
    """
    # Check profit-taking
    if is_valid_barrier(pt_barrier) and pt_barrier is not None and pt_barrier > 0:
        pt_touches = cast(pd.Series, path_ret[path_ret >= pt_barrier])
        if len(pt_touches) > 0:
            first_touch = pt_touches.index[0]
            # Ensure first_touch is a valid timestamp
            try:
                if (first_touch is not None and
                    not (isinstance(first_touch, (int, float)) and pd.isna(first_touch))):
                    try:
                        return cast(pd.Timestamp, pd.to_datetime(first_touch))
                    except (ValueError, TypeError, OSError):
                        pass
            except (ValueError, TypeError):
                pass

    # Check stop-loss
    if is_valid_barrier(sl_barrier) and sl_barrier is not None and sl_barrier < 0:
        sl_touches = cast(pd.Series, path_ret[path_ret <= sl_barrier])
        if len(sl_touches) > 0:
            first_touch = sl_touches.index[0]
            # Ensure first_touch is a valid timestamp
            try:
                if (first_touch is not None and
                    not (isinstance(first_touch, (int, float)) and pd.isna(first_touch))):
                    try:
                        return cast(pd.Timestamp, pd.to_datetime(first_touch))
                    except (ValueError, TypeError, OSError):
                        pass
            except (ValueError, TypeError):
                pass

    return None


def compute_return_and_label(
    close: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    min_return: float,
) -> Tuple[float, int]:
    """
    Compute return and label for a single event.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    t0 : pd.Timestamp
        Entry timestamp.
    t1 : pd.Timestamp
        Exit timestamp.
    min_return : float
        Minimum return threshold.

    Returns
    -------
    Tuple[float, int]
        (return, label) where label is +1, -1, or 0.
    """
    try:
        price_t0 = close.loc[t0]
        price_t1 = close.loc[t1]
        ret = (price_t1 - price_t0) / price_t0

        if abs(ret) < min_return:
            label = 0
        elif ret > 0:
            label = 1
        else:
            label = -1

        return ret, label
    except (KeyError, TypeError):
        return np.nan, 0


def set_vertical_barriers(
    t_events: pd.DatetimeIndex,
    close_idx: pd.Index,
    max_holding: int,
) -> pd.Series:
    """
    Set vertical (time) barriers for all events.

    Parameters
    ----------
    t_events : pd.DatetimeIndex
        Event timestamps.
    close_idx : pd.Index
        Close price index.
    max_holding : int
        Maximum holding period in bars.

    Returns
    -------
    pd.Series
        Series with t1 values for each event.
    """
    t1_series = pd.Series(index=t_events, dtype=object)

    for loc in t_events:
        try:
            t0_pos = close_idx.get_loc(loc)
            if isinstance(t0_pos, int):
                t1_pos = min(t0_pos + max_holding, len(close_idx) - 1)
                t1_series.loc[loc] = close_idx[t1_pos]
            else:
                t1_series.loc[loc] = pd.NaT
        except (KeyError, TypeError):
            t1_series.loc[loc] = pd.NaT

    return t1_series
