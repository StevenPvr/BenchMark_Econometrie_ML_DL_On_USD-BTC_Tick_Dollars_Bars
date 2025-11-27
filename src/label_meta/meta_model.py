"""Meta-model training and management.

This module handles training the meta-model ONCE and reusing it
for all primary model benchmarks.

The meta-model is a LightGBM classifier that learns to predict
whether a primary model's prediction is correct (1) or not (0).
"""

from __future__ import annotations

import os
import sys

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_meta.meta_labeling import get_meta_labels

logger = get_logger(__name__)

__all__ = [
    "train_meta_model_once",
    "load_meta_model",
    "save_meta_model",
]


def train_meta_model_once(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    events_train: pd.DataFrame,
    meta_params: dict[str, Any],
    reference_primary_name: str = "lightgbm",
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> Any:
    """Train the meta-model ONCE using a reference primary model.

    This function should be called ONCE before benchmarking multiple models.
    The trained meta-model is then reused for all primary model benchmarks.

    Args:
        X_train: Training features.
        y_train: Training labels (-1, 0, +1).
        events_train: Training events with 'ret' column.
        meta_params: LightGBM params for meta-model (from label_primaire).
        reference_primary_name: Primary model to use as reference.
        random_state: Random seed.
        verbose: Enable logging.

    Returns:
        Trained meta-model (LightGBMModel).
    """
    from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel

    if verbose:
        logger.info("=" * 60)
        logger.info("TRAINING META-MODEL (ONCE)")
        logger.info("=" * 60)
        logger.info("Reference primary: %s", reference_primary_name)
        logger.info("Meta params: %s", meta_params)

    # Get reference primary model class
    model_class = _get_model_class(reference_primary_name)

    # Filter valid labels
    valid_mask = ~y_train.isna()
    valid_mask_arr = np.asarray(valid_mask)
    X_train_valid = np.asarray(X_train)[valid_mask_arr]
    y_train_valid = np.asarray(y_train[valid_mask]).astype(int)

    if verbose:
        logger.info("Training reference primary model on %d samples...", len(y_train_valid))

    # Train reference primary model with default params
    primary_model = model_class(random_state=random_state)
    primary_model.fit(X_train_valid, y_train_valid)

    # Get primary predictions on training data
    train_predictions = primary_model.predict(X_train_valid)

    if verbose:
        logger.info("Generating meta-labels...")

    # Generate meta-labels
    if isinstance(y_train, pd.Series):
        valid_indices = y_train[valid_mask].index.tolist()  # type: ignore
    else:
        valid_indices = list(range(len(y_train_valid)))

    events_train_aligned = events_train[
        events_train["t_start"].isin(valid_indices)
    ].copy()

    if len(events_train_aligned) == 0:
        events_train_aligned = events_train.iloc[:len(train_predictions)].copy()

    primary_signal_train = pd.Series(
        train_predictions,
        index=events_train_aligned.index[:len(train_predictions)],
    )

    events_for_meta = events_train_aligned.iloc[:len(train_predictions)].copy()
    events_for_meta.index = primary_signal_train.index

    meta_labels_train = get_meta_labels(
        events=events_for_meta,
        primary_signal=primary_signal_train,
    )

    # Filter valid meta-labels
    meta_valid_mask = ~meta_labels_train.isna()
    meta_valid_mask_arr = np.asarray(meta_valid_mask)
    X_meta_train = X_train_valid[meta_valid_mask_arr]
    pred_meta_train = train_predictions[meta_valid_mask_arr]
    y_meta_train = np.asarray(meta_labels_train[meta_valid_mask]).astype(int)

    if verbose:
        logger.info("Meta-labels: %d samples", len(y_meta_train))
        logger.info("Training meta-model...")

    # Create meta-features: original features + primary prediction
    X_meta_train_full = np.column_stack([X_meta_train, pred_meta_train])

    # Train meta-model
    meta_params_clean = dict(meta_params)
    meta_params_clean.pop("random_state", None)  # avoid duplicate

    meta_model = LightGBMModel(
        random_state=random_state,
        **meta_params_clean,
    )
    meta_model.fit(X_meta_train_full, y_meta_train)

    if verbose:
        logger.info("Meta-model trained successfully!")
        logger.info("=" * 60)

    return meta_model


def save_meta_model(meta_model: Any, path: Path | str) -> None:
    """Save trained meta-model to file.

    Args:
        meta_model: Trained meta-model.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, path)
    logger.info("Saved meta-model to %s", path)


def load_meta_model(path: Path | str) -> Any:
    """Load trained meta-model from file.

    Args:
        path: Path to saved meta-model.

    Returns:
        Loaded meta-model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Meta-model not found: {path}")

    meta_model = joblib.load(path)
    logger.info("Loaded meta-model from %s", path)
    return meta_model


def _get_model_class(model_name: str) -> type:
    """Get model class by name."""
    model_name = model_name.lower()

    if model_name == "lightgbm":
        from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel
        return LightGBMModel

    elif model_name == "xgboost":
        from src.model.machine_learning.xgboost.xgboost_model import XGBoostModel
        return XGBoostModel

    elif model_name == "catboost":
        from src.model.machine_learning.catboost.catboost_model import CatBoostModel
        return CatBoostModel

    elif model_name in ("rf", "random_forest", "randomforest"):
        from src.model.machine_learning.rf.random_forest_model import RandomForestModel
        return RandomForestModel

    elif model_name == "ridge":
        from src.model.econometrie.ridge.ridge import RidgeModel
        return RidgeModel

    elif model_name == "lasso":
        from src.model.econometrie.lasso.lasso import LassoModel
        return LassoModel

    elif model_name in ("logistic", "logistic_regression"):
        from src.model.econometrie.logistic.logistic import LogisticModel
        return LogisticModel

    else:
        raise ValueError(f"Unknown model: {model_name}")

