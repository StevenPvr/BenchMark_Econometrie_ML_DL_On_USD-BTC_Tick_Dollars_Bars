"""Training module for all model types.

This module provides a unified training interface compatible with:
- Econometric models (Ridge, Lasso)
- Machine Learning models (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning models (LSTM)

Key Features:
- Simple training with optional validation split
- Walk-forward cross-validation with purge gap
- Training callbacks (early stopping, checkpointing, history)
- Automatic model saving

Example Usage:
    >>> from src.model.machine_learning.xgboost_model import XGBoostModel
    >>> from src.training import train_model, train_with_cv
    >>>
    >>> # Simple training with validation
    >>> model = XGBoostModel(n_estimators=100)
    >>> result = train_model(model, X, y, validation_split=0.2)
    >>> print(f"Train MSE: {result.train_score:.4f}")
    >>> print(f"Val MSE: {result.val_score:.4f}")
    >>>
    >>> # Training with walk-forward CV
    >>> model = XGBoostModel(n_estimators=100)
    >>> result = train_with_cv(
    ...     model, X, y,
    ...     n_splits=5,
    ...     purge_gap=10,
    ... )
    >>> print(f"CV MSE: {result.cv_result.mean_score:.4f}")

Using callbacks for deep learning:
    >>> from src.training import EarlyStopping, TrainingHistory, CallbackList
    >>>
    >>> callbacks = CallbackList([
    ...     EarlyStopping(monitor="val_loss", patience=10),
    ...     TrainingHistory(),
    ... ])
    >>> # Use callbacks in custom training loop
"""

from __future__ import annotations

# Trainer
from src.training.trainer import (
    CrossValidationTrainingResult,
    TrainableModel,
    Trainer,
    TrainingConfig,
    TrainingResult,
    save_training_results,
    train_model,
    train_with_cv,
)

# Callbacks
from src.training.callbacks import (
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    TrainingCallback,
    TrainingHistory,
)

__all__ = [
    # Trainer
    "CrossValidationTrainingResult",
    "TrainableModel",
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "save_training_results",
    "train_model",
    "train_with_cv",
    # Callbacks
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "TrainingCallback",
    "TrainingHistory",
]
