"""
Utilities for Meta-Label module.

Contains:
- Model registry and search spaces
- Data loading functions
- Volatility estimation
- Barrier helpers for meta-labeling
- Dataclasses for configuration and results
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

from src.constants import DEFAULT_RANDOM_STATE
from src.model.base import BaseModel
from src.path import (
    DATASET_FEATURES_PARQUET,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DOLLAR_BARS_PARQUET,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
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
            "reg_alpha": ("categorical", [1e-3, 1e-2, 0.1]),
            "reg_lambda": ("categorical", [1e-3, 1e-2, 0.1]),
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
            "reg_alpha": ("categorical", [1e-3, 1e-2, 0.1]),
            "reg_lambda": ("categorical", [1e-3, 1e-2, 0.1]),
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
        },
    },
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
            "max_iter": ("categorical", [2000, 5000]),
        },
    },
    "logistic": {
        "class": "src.model.logistic_classifier.LogisticClassifierModel",
        "dataset": "linear",
        "search_space": {
            # C: inverse regularization (higher = less regularization)
            "C": ("categorical", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]),
        },
    },
    "elasticnet": {
        "class": "src.model.elasticnet_classifier.ElasticNetClassifierModel",
        "dataset": "linear",
        "search_space": {
            # C: inverse regularization strength
            "C": ("categorical", [0.01, 0.1, 0.5, 1.0, 5.0]),
            # l1_ratio: 0=pure L2 (Ridge), 1=pure L1 (Lasso)
            "l1_ratio": ("categorical", [0.1, 0.3, 0.5, 0.7, 0.9]),
        },
    },
}

# Triple barrier search space for meta model
TRIPLE_BARRIER_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = {
    "pt_mult": ("categorical", [0.8, 1.0, 1.5, 2.0]),
    "sl_mult": ("categorical", [0.8, 1.0, 1.5, 2.0]),
    "max_holding": ("categorical", [5, 10, 20, 30]),
}


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class MetaOptimizationConfig:
    """Configuration for meta model optimization."""

    primary_model_name: str
    meta_model_name: str
    n_trials: int = 50
    n_splits: int = 5
    min_train_size: int = 500
    vol_span: int = 100
    data_fraction: float = 0.3
    random_state: int = DEFAULT_RANDOM_STATE
    timeout: int | None = None


@dataclass
class MetaOptimizationResult:
    """Result of meta model optimization."""

    primary_model_name: str
    meta_model_name: str
    best_params: Dict[str, Any]
    best_triple_barrier_params: Dict[str, Any]
    best_score: float
    metric: str
    n_trials: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_model_name": self.primary_model_name,
            "meta_model_name": self.meta_model_name,
            "best_params": self.best_params,
            "best_triple_barrier_params": self.best_triple_barrier_params,
            "best_score": self.best_score,
            "metric": self.metric,
            "n_trials": self.n_trials,
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

    return pd.read_parquet(path)


def load_dollar_bars() -> pd.DataFrame:
    """Load dollar bars with close prices."""
    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(f"Dollar bars not found: {DOLLAR_BARS_PARQUET}")

    bars = pd.read_parquet(DOLLAR_BARS_PARQUET)

    if "datetime_close" not in bars.columns:
        raise ValueError("Dollar bars must have 'datetime_close' column")

    bars = bars.set_index("datetime_close").sort_index()

    if "log_return" not in bars.columns:
        bars["log_return"] = np.log(bars["close"] / bars["close"].shift(1))
        bars = bars.dropna(subset=["log_return"])

    return bars


def load_primary_model(primary_model_name: str) -> BaseModel:
    """Load a trained primary model."""
    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name
    model_path = model_dir / f"{primary_model_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Primary model not found: {model_path}\n"
            "Train primary model first: python -m src.labelling.label_primaire.train"
        )

    return BaseModel.load(model_path)


def get_available_primary_models() -> List[str]:
    """Get list of trained primary models."""
    if not LABEL_PRIMAIRE_TRAIN_DIR.exists():
        return []

    models = []
    for subdir in LABEL_PRIMAIRE_TRAIN_DIR.iterdir():
        if subdir.is_dir():
            model_file = subdir / f"{subdir.name}_model.joblib"
            if model_file.exists():
                models.append(subdir.name)
    return models


# =============================================================================
# VOLATILITY ESTIMATION
# =============================================================================


def get_daily_volatility(
    close: pd.Series,
    span: int = 100,
    min_periods: int = 10,
) -> pd.Series:
    """
    Compute daily volatility using EWM standard deviation.

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
# META-LABELING BARRIER HELPERS
# =============================================================================


def set_vertical_barriers_meta(
    events: pd.DataFrame,
    close_idx: pd.Index,
    max_holding: int,
) -> pd.DataFrame:
    """Set vertical (time) barriers for meta-labeling events."""
    events["t1"] = pd.NaT

    for loc in events.index:
        try:
            t0_pos = close_idx.get_loc(loc)
            if isinstance(t0_pos, int):
                t1_pos = min(t0_pos + max_holding, len(close_idx) - 1)
                events.loc[loc, "t1"] = close_idx[t1_pos]
        except (KeyError, TypeError):
            pass

    return events


def compute_side_adjusted_barriers(
    events: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """
    Compute barriers adjusted for side (direction).

    For Long (side=+1): PT is positive, SL is negative
    For Short (side=-1): PT is negative, SL is positive
    """
    events["pt"] = pt_mult * events["trgt"] * events["side"]
    events["sl"] = -sl_mult * events["trgt"] * events["side"]
    return events


def get_barrier_touches_for_side(
    path_ret: pd.Series,
    side_val: int,
    pt: float,
    sl: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Get barrier touches based on side direction.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (pt_touches, sl_touches)
    """
    if side_val == 1:  # Long position
        pt_touches = path_ret[path_ret >= pt] if pt > 0 else pd.Series(dtype=float)
        sl_touches = path_ret[path_ret <= sl] if sl < 0 else pd.Series(dtype=float)
    else:  # Short position (side=-1)
        pt_touches = path_ret[path_ret <= pt] if pt < 0 else pd.Series(dtype=float)
        sl_touches = path_ret[path_ret >= sl] if sl > 0 else pd.Series(dtype=float)

    return pt_touches, sl_touches


def find_first_touch_time(
    pt_touches: pd.Series,
    sl_touches: pd.Series,
) -> pd.Timestamp | None:
    """Find the first barrier touch time."""
    pt_time = pt_touches.index[0] if len(pt_touches) > 0 else None
    sl_time = sl_touches.index[0] if len(sl_touches) > 0 else None

    if pt_time is not None and sl_time is not None:
        return min(pt_time, sl_time)
    elif pt_time is not None:
        return pt_time
    elif sl_time is not None:
        return sl_time
    return None


def compute_meta_label(ret: float, side: int) -> int:
    """
    Compute meta-label based on return and side.

    Meta-label is 1 if the trade was profitable in the predicted direction.
    Rule: bin = 1 if (return * side > 0), else 0
    """
    return 1 if ret * side > 0 else 0
