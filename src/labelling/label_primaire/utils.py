"""Utilities for the primary labeling pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Type, cast

import numpy as np
import pandas as pd

from src.config_logging import get_logger
from src.constants import (
    CLASS_WEIGHT_SUPPORTED_MODELS_PRIMARY,
    COMPOSITE_SCORE_WEIGHTS_PRIMARY,
    DEFAULT_RANDOM_STATE,
    FOCAL_LOSS_SEARCH_SPACE_PRIMARY,
    FOCAL_LOSS_SUPPORTED_MODELS_PRIMARY,
    LABELING_DAILY_VOL_MIN_PERIODS,
    LABELING_DAILY_VOL_SPAN_DEFAULT,
    LABELING_MIN_MINORITY_PREDICTION_RATIO,
    LABELING_MIN_PER_CLASS_F1_REQUIRED,
    LABELING_RISK_REWARD_RATIO,
    TRIPLE_BARRIER_SEARCH_SPACE_PRIMARY,
)
from src.model.base import BaseModel
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    DOLLAR_BARS_PARQUET,
)

logger = get_logger(__name__)

# =============================================================================
# MODEL REGISTRY AND SEARCH SPACES
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "class": "src.model.xgboost_model.XGBoostModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("int", [50, 800]),
            "max_depth": ("int", [2, 10]),
            "learning_rate": ("float", [0.001, 0.3, "log"]),
            "subsample": ("float", [0.6, 0.95]),
            "colsample_bytree": ("float", [0.5, 0.95]),
            "reg_alpha": ("float", [1e-8, 10.0, "log"]),
            "reg_lambda": ("float", [1e-8, 10.0, "log"]),
            "min_child_weight": ("int", [1, 50]),
            "gamma": ("float", [0.0, 5.0]),
        },
    },
    "lightgbm": {
        "class": "src.model.lightgbm_model.LightGBMModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("int", [50, 800]),
            "max_depth": ("int", [2, 12]),
            "num_leaves": ("int", [8, 128]),
            "learning_rate": ("float", [0.001, 0.3, "log"]),
            "subsample": ("float", [0.6, 0.95]),
            "colsample_bytree": ("float", [0.5, 0.95]),
            "reg_alpha": ("float", [1e-8, 10.0, "log"]),
            "reg_lambda": ("float", [1e-8, 10.0, "log"]),
            "min_child_samples": ("int", [5, 100]),
        },
    },
    "ridge": {
        "class": "src.model.ridge_classifier.RidgeClassifierModel",
        "dataset": "linear",
        "search_space": {
            "alpha": ("float", [1e-6, 1000.0, "log"]),
        },
    },
    "lstm": {
        "class": "src.model.lstm_model.LSTMModel",
        "dataset": "lstm",
        "search_space": {
            "hidden_size": ("int", [16, 256]),
            "num_layers": ("int", [1, 4]),
            "dropout": ("float", [0.0, 0.5]),
            "learning_rate": ("float", [1e-5, 1e-2, "log"]),
            "weight_decay": ("float", [1e-8, 1e-3, "log"]),
            "batch_size": ("categorical", [16, 32, 64, 128]),
        },
    },
}

TRIPLE_BARRIER_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = TRIPLE_BARRIER_SEARCH_SPACE_PRIMARY
FOCAL_LOSS_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = FOCAL_LOSS_SEARCH_SPACE_PRIMARY
FOCAL_LOSS_SUPPORTED_MODELS: List[str] = FOCAL_LOSS_SUPPORTED_MODELS_PRIMARY
CLASS_WEIGHT_SUPPORTED_MODELS: List[str] = CLASS_WEIGHT_SUPPORTED_MODELS_PRIMARY

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for primary model optimization."""

    model_name: str
    n_trials: int = 50
    n_splits: int = 5
    min_train_size: int = 500
    data_fraction: float = 0.8
    random_state: int = DEFAULT_RANDOM_STATE
    timeout: int | None = None
    early_stopping_rounds: int = 50
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    optimize_focal_params: bool = True
    use_class_weights: bool = True
    minority_weight_boost: float = 1.25


@dataclass
class CompositeScoreMetrics:
    """Composite score metrics for multi-class evaluation."""

    f1_minus1: float
    f1_zero: float
    f1_plus1: float
    f1_min: float
    sign_error_rate: float
    composite_score: float
    n_sign_errors: int
    n_total: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation."""
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


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    model_name: str
    best_params: Dict[str, Any]
    best_triple_barrier_params: Dict[str, Any]
    best_score: float
    metric: str
    n_trials: int
    cv_scores: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())
    best_focal_loss_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary suitable for JSON serialization."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_triple_barrier_params": self.best_triple_barrier_params,
            "best_focal_loss_params": self.best_focal_loss_params,
            "best_score": self.best_score,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "cv_scores": self.cv_scores,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Persist the result to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Optimization results saved to %s", path)


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================


def load_model_class(model_name: str) -> Type[BaseModel]:
    """Load a model class from the registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    class_path = MODEL_REGISTRY[model_name]["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return cast(Type[BaseModel], getattr(module, class_name))


def _dataset_path_for_model(model_name: str) -> Path:
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    return path_map[dataset_type]


def get_dataset_for_model(model_name: str) -> pd.DataFrame:
    """Load the features dataset matching the model type."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    path = _dataset_path_for_model(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found for {model_name}: {path}")

    return pd.read_parquet(path)


def load_dollar_bars() -> pd.DataFrame:
    """Load dollar bars, indexed by close time with log returns."""
    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(f"Dollar bars not found: {DOLLAR_BARS_PARQUET}")

    bars = pd.read_parquet(DOLLAR_BARS_PARQUET)
    if "datetime_close" not in bars.columns:
        raise ValueError("Dollar bars must have 'datetime_close' column")

    bars = bars.set_index("datetime_close").sort_index()
    bars["log_return"] = pd.Series(np.log(bars["close"])).diff()
    return bars


# =============================================================================
# VOLATILITY & BARRIERS
# =============================================================================


def get_daily_volatility(
    close: pd.Series,
    span: int = LABELING_DAILY_VOL_SPAN_DEFAULT,
    min_periods: int = LABELING_DAILY_VOL_MIN_PERIODS,
) -> pd.Series:
    """Compute exponentially weighted volatility of close prices."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.ewm(span=span, min_periods=min_periods).std()


def compute_barriers(events: pd.DataFrame, pt_mult: float, sl_mult: float) -> pd.DataFrame:
    """Compute profit-taking and stop-loss barriers."""
    events = events.copy()
    trgt = events["trgt"]
    events["pt"] = trgt * pt_mult if pt_mult > 0 else np.nan
    events["sl"] = -trgt * sl_mult if sl_mult > 0 else np.nan
    return events


def is_valid_barrier(value: Any) -> bool:
    """Return True when the barrier value is numeric."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


def set_vertical_barriers(
    t_events: pd.DatetimeIndex,
    close_idx: pd.Index,
    max_holding: int,
) -> pd.Series:
    """Return time barriers capped by the available close index."""
    t1 = pd.Series(index=t_events, dtype="datetime64[ns]")
    for t0 in t_events:
        try:
            loc = close_idx.get_loc(t0)
        except KeyError:
            continue
        end_loc = min(cast(int, loc) + max_holding, len(close_idx) - 1)
        t1.loc[t0] = close_idx[end_loc]
    return t1


def _first_touch_index(path_returns: pd.Series, pt_barrier: float, sl_barrier: float) -> pd.Timestamp | None:
    for ts, ret in path_returns.items():
        if is_valid_barrier(pt_barrier) and ret >= pt_barrier:
            return cast(pd.Timestamp, ts)
        if is_valid_barrier(sl_barrier) and ret <= sl_barrier:
            return cast(pd.Timestamp, ts)
    return None


def find_barrier_touch(
    path_returns: pd.Series,
    pt_barrier: float | None,
    sl_barrier: float | None,
) -> pd.Timestamp | None:
    """Find the first time a path crosses PT or SL barriers."""
    if path_returns.empty:
        return None
    pt_val = float(pt_barrier) if pt_barrier is not None else np.nan
    sl_val = float(sl_barrier) if sl_barrier is not None else np.nan
    return _first_touch_index(path_returns, pt_val, sl_val)


def compute_return_and_label(
    close: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    min_return: float,
) -> Tuple[float, int]:
    """Compute return between t0 and t1 and convert to {-1,0,1} label."""
    if t0 not in close.index or t1 not in close.index or t0 == t1:
        return np.nan, 0

    ret = (float(close.loc[t1]) - float(close.loc[t0])) / float(close.loc[t0])
    if ret > min_return:
        return ret, 1
    if ret < -min_return:
        return ret, -1
    return ret, 0


def compute_composite_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Mapping[str, float] | None = None,
) -> CompositeScoreMetrics:
    """Compute composite score prioritising directional accuracy."""
    from sklearn.metrics import f1_score

    if weights is None:
        weights = COMPOSITE_SCORE_WEIGHTS_PRIMARY

    f1_values = np.asarray(f1_score(
        y_true,
        y_pred,
        labels=[-1, 0, 1],
        average=None,  # type: ignore[arg-type]
        zero_division=0.0,  # type: ignore[arg-type]
    ))
    f1_minus1 = float(f1_values[0])
    f1_zero = float(f1_values[1])
    f1_plus1 = float(f1_values[2])
    f1_min = min(f1_minus1, f1_zero, f1_plus1)

    sign_errors = np.sum(((y_pred == 1) & (y_true == -1)) | ((y_pred == -1) & (y_true == 1)))
    n_total = len(y_true)
    sign_error_rate = float(sign_errors) / n_total if n_total else 0.0

    alpha = weights.get("alpha", 0.5)
    beta = weights.get("beta", 0.2)
    delta = weights.get("delta", 0.0)
    gamma = weights.get("gamma", 0.3)

    composite = alpha * (f1_minus1 + f1_plus1) / 2 + beta * f1_zero + delta * f1_min - gamma * sign_error_rate

    return CompositeScoreMetrics(
        f1_minus1=f1_minus1,
        f1_zero=f1_zero,
        f1_plus1=f1_plus1,
        f1_min=f1_min,
        sign_error_rate=sign_error_rate,
        composite_score=composite,
        n_sign_errors=int(sign_errors),
        n_total=n_total,
    )


__all__ = [
    "CLASS_WEIGHT_SUPPORTED_MODELS",
    "COMPOSITE_SCORE_WEIGHTS_PRIMARY",
    "FOCAL_LOSS_SEARCH_SPACE",
    "FOCAL_LOSS_SUPPORTED_MODELS",
    "MODEL_REGISTRY",
    "OptimizationConfig",
    "OptimizationResult",
    "RISK_REWARD_RATIO",
    "TRIPLE_BARRIER_SEARCH_SPACE",
    "compute_barriers",
    "compute_composite_score",
    "compute_return_and_label",
    "find_barrier_touch",
    "get_daily_volatility",
    "get_dataset_for_model",
    "is_valid_barrier",
    "load_dollar_bars",
    "load_model_class",
    "set_vertical_barriers",
]

# Backwards compatibility alias
RISK_REWARD_RATIO: float = LABELING_RISK_REWARD_RATIO
