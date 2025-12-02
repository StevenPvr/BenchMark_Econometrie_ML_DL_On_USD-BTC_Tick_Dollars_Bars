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
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.constants import DEFAULT_RANDOM_STATE
from src.model.base import BaseModel
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    DOLLAR_BARS_PARQUET,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "class": "src.model.xgboost_model.XGBoostModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("categorical", [200, 400, 800, 1200]),
            "max_depth": ("categorical", [3, 5, 7, 9]),
            "learning_rate": ("categorical", [0.01, 0.02, 0.05, 0.1]),
            "subsample": ("categorical", [0.6, 0.75, 0.9, 1.0]),
            "colsample_bytree": ("categorical", [0.6, 0.8, 1.0]),
            "reg_alpha": ("categorical", [0.0, 1e-3, 1e-2, 0.1, 1.0]),
            "reg_lambda": ("categorical", [0.0, 1e-3, 1e-2, 0.1, 1.0]),
        },
    },
    "randomforest": {
        "class": "src.model.random_forest_model.RandomForestModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("categorical", [100, 200, 400, 800]),
            "max_depth": ("categorical", [3, 5, 7, 9, 12]),
            "min_samples_split": ("categorical", [2, 5, 10, 20]),
            "min_samples_leaf": ("categorical", [1, 2, 4, 8]),
            "max_features": ("categorical", ["sqrt", "log2", None]),
        },
    },
    "ridge": {
        "class": "src.model.ridge_classifier.RidgeClassifierModel",
        "dataset": "linear",
        "search_space": {
            # alpha: regularization strength (higher = more regularization)
            "alpha": ("categorical", [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]),
        },
    },
    "logistic": {
        "class": "src.model.logistic_classifier.LogisticClassifierModel",
        "dataset": "linear",
        "search_space": {
            # C: inverse regularization (higher = less regularization)
            "C": ("categorical", [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 250.0]),
        },
    },
}

# Triple barrier search space for meta model
TRIPLE_BARRIER_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = {
    # Narrowed ranges to help meet class proportion constraints
    "pt_mult": ("categorical", [0.8, 1.0, 1.5, 2.0]),
    "sl_mult": ("categorical", [0.8, 1.0, 1.5, 2.0]),
    "max_holding": ("categorical", [10, 20, 30, 50]),
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
    data_fraction: float = 1.0
    random_state: int = DEFAULT_RANDOM_STATE
    timeout: int | None = None
    parallelize_labeling: bool = True
    parallel_min_events: int = 10_000
    n_jobs: int | None = None
    filter_neutral_labels: bool = True  # Filter label=0 (neutral) when computing meta labels


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
    """Load the appropriate feature dataset for a model type (base, without labels)."""
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

    return pd.read_parquet(path)


def get_labeled_dataset_for_primary_model(primary_model_name: str) -> pd.DataFrame:
    """
    Load the labeled feature dataset for a specific primary model.

    Each primary model has its own labeled features file created during training,
    containing labels generated with that model's triple barrier parameters.

    Parameters
    ----------
    primary_model_name : str
        Name of the primary model (e.g., 'lightgbm', 'xgboost').

    Returns
    -------
    pd.DataFrame
        DataFrame with features and labels for the specified primary model.

    Raises
    ------
    FileNotFoundError
        If the labeled features file doesn't exist.
    """
    if primary_model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {primary_model_name}")

    dataset_type = MODEL_REGISTRY[primary_model_name]["dataset"]

    base_path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }

    base_path = base_path_map[dataset_type]
    # Model-specific labeled file: dataset_features_final_{model_name}.parquet
    labeled_path = base_path.parent / f"{base_path.stem}_{primary_model_name}.parquet"

    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Labeled features file not found: {labeled_path}\n"
            f"Train the primary model first: python -m src.labelling.label_primaire.train"
        )

    return pd.read_parquet(labeled_path)


def get_meta_labeled_dataset_path(primary_model_name: str, meta_model_name: str) -> Path:
    """
    Get the path for meta-labeled dataset.

    The meta model needs its own dataset type (tree, linear, etc.) with labels.
    File naming: dataset_features_{primary}_meta_{meta}.parquet

    Example: logistic (primary) + lightgbm (meta) -> dataset_features_logistic_meta_lightgbm.parquet
    """
    if meta_model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown meta model: {meta_model_name}")

    # Use the dataset type of the META model (not primary)
    dataset_type = MODEL_REGISTRY[meta_model_name]["dataset"]

    base_path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }

    base_path = base_path_map[dataset_type]
    return base_path.parent / f"{base_path.stem}_{primary_model_name}_meta_{meta_model_name}.parquet"


def get_meta_labeled_dataset(
    primary_model_name: str,
    meta_model_name: str,
) -> Tuple[pd.DataFrame, Path]:
    """
    Load the meta-labeled dataset for a primary/meta model combination.

    Returns the dataset and its path.
    If the file doesn't exist, returns the labeled dataset from the primary model
    which contains the labels (column "label") needed for meta-labeling.
    """
    labeled_path = get_meta_labeled_dataset_path(primary_model_name, meta_model_name)

    if labeled_path.exists():
        return pd.read_parquet(labeled_path), labeled_path

    # File doesn't exist, return primary model's labeled dataset
    # This contains the "label" column with the primary model's triple barrier labels
    return get_labeled_dataset_for_primary_model(primary_model_name), labeled_path


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
    """Load a trained primary model.

    Tries to load from main.py pipeline first (_final_model.joblib),
    then falls back to train.py pipeline (_model.joblib).
    """
    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name

    # Try main.py pipeline first (full K-Fold OOS pipeline)
    model_path_main = model_dir / f"{primary_model_name}_final_model.joblib"
    # Fallback to train.py pipeline (simple one-shot training)
    model_path_train = model_dir / f"{primary_model_name}_model.joblib"

    if model_path_main.exists():
        return BaseModel.load(model_path_main)
    elif model_path_train.exists():
        return BaseModel.load(model_path_train)
    else:
        raise FileNotFoundError(
            f"Primary model not found in:\n"
            f"  - {model_path_main}\n"
            f"  - {model_path_train}\n"
            "Train primary model first: python -m src.labelling.label_primaire.main"
        )


def get_available_primary_models() -> List[str]:
    """Get list of trained primary models.

    Checks for both main.py and train.py pipeline outputs.
    """
    if not LABEL_PRIMAIRE_TRAIN_DIR.exists():
        return []

    models = []
    for subdir in LABEL_PRIMAIRE_TRAIN_DIR.iterdir():
        if subdir.is_dir():
            # Check both file patterns
            model_file_main = subdir / f"{subdir.name}_final_model.joblib"
            model_file_train = subdir / f"{subdir.name}_model.joblib"
            if model_file_main.exists() or model_file_train.exists():
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
    # Initialize t1 column with correct dtype to avoid FutureWarning
    # close_idx may be timezone-aware (datetime64[ns, UTC])
    if isinstance(close_idx, pd.DatetimeIndex) and close_idx.tz is not None:
        # Timezone-aware index
        events["t1"] = pd.Series(pd.NaT, index=events.index, dtype=close_idx.dtype)
    else:
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
        pt_touches = cast(pd.Series, path_ret[path_ret >= pt]) if pt > 0 else pd.Series(dtype=float)
        sl_touches = cast(pd.Series, path_ret[path_ret <= sl]) if sl < 0 else pd.Series(dtype=float)
    else:  # Short position (side=-1)
        pt_touches = cast(pd.Series, path_ret[path_ret <= pt]) if pt < 0 else pd.Series(dtype=float)
        sl_touches = cast(pd.Series, path_ret[path_ret >= sl]) if sl > 0 else pd.Series(dtype=float)

    return pt_touches, sl_touches


def find_first_touch_time(
    pt_touches: pd.Series,
    sl_touches: pd.Series,
) -> pd.Timestamp | None:
    """Find the first barrier touch time."""
    pt_time = cast(pd.Timestamp, pt_touches.index[0]) if len(pt_touches) > 0 else None
    sl_time = cast(pd.Timestamp, sl_touches.index[0]) if len(sl_touches) > 0 else None

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
