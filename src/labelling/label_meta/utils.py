"""Utilities for meta-labeling models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
import pandas as pd

from src.config_logging import get_logger
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LABELING_DAILY_VOL_MIN_PERIODS,
    LABELING_DAILY_VOL_SPAN_DEFAULT,
    TRIPLE_BARRIER_SEARCH_SPACE_META,
)
from src.model.base import BaseModel
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    DOLLAR_BARS_PARQUET,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = get_logger(__name__)

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "lightgbm": {
        "class": "src.model.lightgbm_model.LightGBMModel",
        "dataset": "tree",
        "search_space": {
            "n_estimators": ("int", [50, 500]),
            "max_depth": ("int", [2, 8]),
            "learning_rate": ("float", [0.01, 0.3, "log"]),
            "subsample": ("float", [0.6, 0.95]),
            "colsample_bytree": ("float", [0.5, 0.95]),
            "reg_alpha": ("float", [1e-8, 10.0, "log"]),
            "reg_lambda": ("float", [1e-8, 10.0, "log"]),
        },
    },
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
    "ridge": {
        "class": "src.model.ridge_classifier.RidgeClassifierModel",
        "dataset": "linear",
        "search_space": {
            "alpha": ("float", [1e-6, 1000.0, "log"]),
        },
    },
}

TRIPLE_BARRIER_SEARCH_SPACE: Dict[str, Tuple[str, List[Any]]] = TRIPLE_BARRIER_SEARCH_SPACE_META


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
    filter_neutral_labels: bool = True


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
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Results saved to %s", path)


def load_model_class(model_name: str) -> Type[BaseModel]:
    """Dynamically load a model class from the registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    class_path = MODEL_REGISTRY[model_name]["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return cast(Type[BaseModel], getattr(module, class_name))


def _dataset_path_for_model(model_name: str) -> Path:
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    mapping = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    return mapping[dataset_type]


def get_dataset_for_model(model_name: str) -> pd.DataFrame:
    """Load the base feature dataset (without labels)."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    path = _dataset_path_for_model(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found for {model_name}: {path}")
    return pd.read_parquet(path)


def _primary_labeled_path(primary_model_name: str) -> Path:
    """Get path to labeled dataset for a primary model."""
    if primary_model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown primary model: {primary_model_name}")
    dataset_type = MODEL_REGISTRY[primary_model_name]["dataset"]
    mapping = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    base = mapping[dataset_type]
    return base.parent / f"{base.stem}_{primary_model_name}.parquet"


def get_labeled_dataset_for_primary_model(primary_model_name: str) -> pd.DataFrame:
    """Load labeled dataset produced by a primary model training run."""
    path = _primary_labeled_path(primary_model_name)
    if not path.exists():
        raise FileNotFoundError(f"Labeled dataset for primary model not found: {path}")
    df = pd.read_parquet(path)
    if "datetime_close" in df.columns:
        df = df.set_index("datetime_close")
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df.loc[~df.index.duplicated(keep="first")]
    return df


def load_dollar_bars() -> pd.DataFrame:
    """Load dollar bars with log returns computed."""
    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(f"Dollar bars not found: {DOLLAR_BARS_PARQUET}")
    bars = pd.read_parquet(DOLLAR_BARS_PARQUET)
    if "datetime_close" not in bars.columns:
        raise ValueError("Dollar bars must include 'datetime_close'.")
    bars = bars.set_index("datetime_close").sort_index()
    bars["log_return"] = pd.Series(np.log(bars["close"]), index=bars.index).diff()
    return bars


def get_available_primary_models() -> List[str]:
    """List trained primary models."""
    if not LABEL_PRIMAIRE_TRAIN_DIR.exists():
        return []
    models: List[str] = []
    for subdir in LABEL_PRIMAIRE_TRAIN_DIR.iterdir():
        if subdir.is_dir() and (subdir / f"{subdir.name}_model.joblib").exists():
            models.append(subdir.name)
    return models


def load_primary_model(model_name: str) -> BaseModel:
    """Load a trained primary model for meta-labeling."""
    model_path = LABEL_PRIMAIRE_TRAIN_DIR / model_name / f"{model_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Primary model not found: {model_path}")
    return BaseModel.load(model_path)


def get_daily_volatility(
    close: pd.Series,
    span: int = LABELING_DAILY_VOL_SPAN_DEFAULT,
    min_periods: int = LABELING_DAILY_VOL_MIN_PERIODS,
) -> pd.Series:
    """Compute exponentially weighted volatility."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.ewm(span=span, min_periods=min_periods).std()


def set_vertical_barriers_meta(
    events: pd.DataFrame,
    close_idx: pd.Index,
    max_holding: int,
) -> pd.DataFrame:
    """Set vertical time barriers for meta-labeling events."""
    t1_series = pd.Series(index=events.index, dtype=close_idx.dtype if hasattr(close_idx, "dtype") else "datetime64[ns]")
    for loc in events.index:
        try:
            t0_pos = close_idx.get_loc(loc)  # type: ignore
        except KeyError:
            continue
        end_pos = min(t0_pos + max_holding, len(close_idx) - 1)  # type: ignore
        t1_series.loc[loc] = close_idx[end_pos]
    events["t1"] = t1_series
    return events


def compute_side_adjusted_barriers(
    events: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """Adjust barriers depending on trade side."""
    events = events.copy()
    events["pt"] = pt_mult * events["trgt"] * events["side"]
    events["sl"] = -sl_mult * events["trgt"] * events["side"]
    return events


def get_barrier_touches_for_side(
    path_ret: pd.Series,
    side_val: int,
    pt: float,
    sl: float,
) -> Tuple[pd.Series, pd.Series]:
    """Return profit-taking and stop-loss touch points for a given side."""
    if side_val == 1:
        pt_touches = cast(pd.Series, path_ret[path_ret >= pt]) if pt > 0 else pd.Series(dtype=float)
        sl_touches = cast(pd.Series, path_ret[path_ret <= sl]) if sl < 0 else pd.Series(dtype=float)
    else:
        pt_touches = cast(pd.Series, path_ret[path_ret <= pt]) if pt < 0 else pd.Series(dtype=float)
        sl_touches = cast(pd.Series, path_ret[path_ret >= sl]) if sl > 0 else pd.Series(dtype=float)
    return pt_touches, sl_touches


def find_first_touch_time(
    pt_touches: pd.Series,
    sl_touches: pd.Series,
) -> pd.Timestamp | None:
    """Return the earliest touch among PT and SL."""
    pt_time = cast(pd.Timestamp, pt_touches.index[0]) if len(pt_touches) > 0 else None
    sl_time = cast(pd.Timestamp, sl_touches.index[0]) if len(sl_touches) > 0 else None
    if pt_time is not None and sl_time is not None:
        return min(pt_time, sl_time)
    return pt_time or sl_time


def compute_meta_label(ret: float, side: int) -> int:
    """Compute binary meta-label: 1 when return aligns with side."""
    return 1 if ret * side > 0 else 0


__all__ = [
    "MODEL_REGISTRY",
    "MetaOptimizationConfig",
    "MetaOptimizationResult",
    "TRIPLE_BARRIER_SEARCH_SPACE",
    "compute_meta_label",
    "compute_side_adjusted_barriers",
    "find_first_touch_time",
    "get_available_primary_models",
    "get_barrier_touches_for_side",
    "get_daily_volatility",
    "get_dataset_for_model",
    "get_labeled_dataset_for_primary_model",
    "load_dollar_bars",
    "load_model_class",
    "load_primary_model",
    "set_vertical_barriers_meta",
]
