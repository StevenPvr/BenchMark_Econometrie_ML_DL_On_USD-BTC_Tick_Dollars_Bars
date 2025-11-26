"""Generic model trainer for all model types.

This module provides a unified training interface compatible with:
- Econometric models (Ridge, Lasso)
- Machine Learning models (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning models (LSTM)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.constants import DEFAULT_RANDOM_STATE
from src.optimisation.walk_forward_cv import (
    FittableModel,
    METRIC_FUNCTIONS,
    WalkForwardConfig,
    WalkForwardResult,
    walk_forward_cv,
)
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


# ============================================================================
# PROTOCOLS
# ============================================================================


class TrainableModel(Protocol):
    """Protocol for models that can be trained."""

    name: str
    is_fitted: bool

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "TrainableModel":
        """Fit the model on training data."""
        ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Get model hyperparameters."""
        ...

    def set_params(self, **params: Any) -> "TrainableModel":
        """Set model hyperparameters."""
        ...

    def save(self, path: Path) -> None:
        """Save model to disk."""
        ...


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        validation_split: Fraction of data for validation (0 = no validation).
        shuffle: Whether to shuffle data before splitting (False for time series).
        random_state: Random seed for reproducibility.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        save_model: Whether to save the trained model.
        output_dir: Directory for saving outputs.
        model_filename: Filename for saved model.
    """

    validation_split: float = 0.0
    shuffle: bool = False  # Default False for time series
    random_state: int = DEFAULT_RANDOM_STATE
    verbose: int = 1
    save_model: bool = True
    output_dir: Path | None = None
    model_filename: str = "model.joblib"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError(f"validation_split must be in [0, 1), got {self.validation_split}")
        if self.verbose not in (0, 1, 2):
            raise ValueError(f"verbose must be 0, 1, or 2, got {self.verbose}")


# ============================================================================
# TRAINING RESULTS
# ============================================================================


@dataclass
class TrainingResult:
    """Result from model training.

    Attributes:
        model: The trained model.
        train_score: Training metric value.
        val_score: Validation metric value (None if no validation).
        train_size: Number of training samples.
        val_size: Number of validation samples.
        metric_name: Name of the metric used.
        training_time: Training duration in seconds.
        model_params: Model hyperparameters.
        model_path: Path where model was saved (if saved).
    """

    model: TrainableModel
    train_score: float
    val_score: float | None
    train_size: int
    val_size: int
    metric_name: str
    training_time: float
    model_params: dict[str, Any]
    model_path: Path | None = None


@dataclass
class CrossValidationTrainingResult:
    """Result from training with cross-validation.

    Attributes:
        model: The final trained model (on full data).
        cv_result: Cross-validation results.
        final_train_score: Score on full training data.
        training_time: Total training duration.
        model_params: Model hyperparameters.
        model_path: Path where model was saved.
    """

    model: TrainableModel
    cv_result: WalkForwardResult
    final_train_score: float
    training_time: float
    model_params: dict[str, Any]
    model_path: Path | None = None


# ============================================================================
# TRAINER CLASS
# ============================================================================


class Trainer:
    """Generic trainer for all model types.

    This trainer provides a unified interface for training any model
    that implements the TrainableModel protocol.

    Example:
        >>> from src.model.machine_learning.xgboost_model import XGBoostModel
        >>> from src.training import Trainer, TrainingConfig
        >>>
        >>> model = XGBoostModel(n_estimators=100)
        >>> trainer = Trainer(config=TrainingConfig(validation_split=0.2))
        >>> result = trainer.train(model, X_train, y_train, metric="mse")
        >>> print(f"Train MSE: {result.train_score:.4f}")
        >>> print(f"Val MSE: {result.val_score:.4f}")
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration.
        """
        self.config = config or TrainingConfig()

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Split data into train and validation sets.

        Args:
            X: Features array.
            y: Target array.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val).
            X_val and y_val are None if validation_split is 0.
        """
        if self.config.validation_split == 0.0:
            return X, y, None, None

        n_samples = len(X)
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val

        if self.config.shuffle:
            rng = np.random.RandomState(self.config.random_state)
            indices = rng.permutation(n_samples)
        else:
            # For time series: validation is the last portion
            indices = np.arange(n_samples)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
    ) -> float:
        """Compute evaluation metric.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.
            metric: Metric name.

        Returns:
            Metric value.
        """
        if metric not in METRIC_FUNCTIONS:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(METRIC_FUNCTIONS.keys())}")
        return METRIC_FUNCTIONS[metric](y_true, y_pred)

    def _save_model(self, model: TrainableModel) -> Path | None:
        """Save model to disk.

        Args:
            model: Trained model.

        Returns:
            Path where model was saved, or None.
        """
        if not self.config.save_model or self.config.output_dir is None:
            return None

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / self.config.model_filename
        model.save(model_path)

        if self.config.verbose >= 1:
            logger.info("Model saved to %s", model_path)

        return model_path

    def train(
        self,
        model: TrainableModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        metric: str = "mse",
        fit_kwargs: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Train a model with optional validation.

        Args:
            model: Model to train.
            X: Training features.
            y: Training target.
            metric: Evaluation metric ("mse", "rmse", "mae", "qlike").
            fit_kwargs: Additional kwargs for model.fit().

        Returns:
            TrainingResult with scores and model info.
        """
        fit_kwargs = fit_kwargs or {}
        start_time = datetime.now()

        # Convert to numpy
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        # Split data
        X_train, y_train, X_val, y_val = self._split_data(X_arr, y_arr)

        if self.config.verbose >= 1:
            logger.info(
                "Training %s: %d samples (train=%d, val=%d)",
                model.name,
                len(X_arr),
                len(X_train),
                len(X_val) if X_val is not None else 0,
            )

        # Train model
        model.fit(X_train, y_train, **fit_kwargs)

        # Compute training score
        train_pred = model.predict(X_train)
        train_score = self._compute_metric(y_train, train_pred, metric)

        # Compute validation score
        val_score = None
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_score = self._compute_metric(y_val, val_pred, metric)

        training_time = (datetime.now() - start_time).total_seconds()

        if self.config.verbose >= 1:
            msg = f"Training complete: train_{metric}={train_score:.6f}"
            if val_score is not None:
                msg += f", val_{metric}={val_score:.6f}"
            msg += f" ({training_time:.2f}s)"
            logger.info(msg)

        # Save model
        model_path = self._save_model(model)

        return TrainingResult(
            model=model,
            train_score=train_score,
            val_score=val_score,
            train_size=len(X_train),
            val_size=len(X_val) if X_val is not None else 0,
            metric_name=metric,
            training_time=training_time,
            model_params=model.get_params(),
            model_path=model_path,
        )

    def train_with_cv(
        self,
        model: TrainableModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cv_config: WalkForwardConfig,
        metric: str = "mse",
        fit_kwargs: dict[str, Any] | None = None,
        retrain_on_full: bool = True,
    ) -> CrossValidationTrainingResult:
        """Train model with walk-forward cross-validation.

        Performs CV to estimate generalization performance, then
        optionally retrains on full data.

        Args:
            model: Model to train.
            X: Training features.
            y: Training target.
            cv_config: Walk-forward CV configuration.
            metric: Evaluation metric.
            fit_kwargs: Additional kwargs for model.fit().
            retrain_on_full: If True, retrain on full data after CV.

        Returns:
            CrossValidationTrainingResult with CV scores and final model.
        """
        fit_kwargs = fit_kwargs or {}
        start_time = datetime.now()

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if self.config.verbose >= 1:
            logger.info(
                "Training %s with %d-fold CV: %d samples",
                model.name,
                cv_config.n_splits,
                len(X_arr),
            )

        # Run cross-validation
        # TrainableModel has all methods required by FittableModel, so cast is safe
        cv_result = walk_forward_cv(
            model=cast(FittableModel, model),
            X=X_arr,
            y=y_arr,
            config=cv_config,
            metric=metric,
            verbose=self.config.verbose >= 2,
            fit_kwargs=fit_kwargs,
        )

        if self.config.verbose >= 1:
            logger.info(
                "CV complete: %s=%.6f (+/- %.6f)",
                metric,
                cv_result.mean_score,
                cv_result.std_score,
            )

        # Retrain on full data
        final_train_score = cv_result.mean_score
        if retrain_on_full:
            if self.config.verbose >= 1:
                logger.info("Retraining on full data...")

            model.fit(X_arr, y_arr, **fit_kwargs)
            train_pred = model.predict(X_arr)
            final_train_score = self._compute_metric(y_arr, train_pred, metric)

            if self.config.verbose >= 1:
                logger.info("Final train_%s=%.6f", metric, final_train_score)

        training_time = (datetime.now() - start_time).total_seconds()

        # Save model
        model_path = self._save_model(model)

        return CrossValidationTrainingResult(
            model=model,
            cv_result=cv_result,
            final_train_score=final_train_score,
            training_time=training_time,
            model_params=model.get_params(),
            model_path=model_path,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def train_model(
    model: TrainableModel,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    validation_split: float = 0.2,
    metric: str = "mse",
    verbose: bool = True,
    **fit_kwargs: Any,
) -> TrainingResult:
    """Convenience function to train a model.

    Args:
        model: Model to train.
        X: Training features.
        y: Training target.
        validation_split: Fraction for validation.
        metric: Evaluation metric.
        verbose: Show progress.
        **fit_kwargs: Additional kwargs for model.fit().

    Returns:
        TrainingResult with scores and model.

    Example:
        >>> from src.model.machine_learning.xgboost_model import XGBoostModel
        >>> model = XGBoostModel(n_estimators=100)
        >>> result = train_model(model, X, y, validation_split=0.2)
    """
    config = TrainingConfig(
        validation_split=validation_split,
        verbose=1 if verbose else 0,
    )
    trainer = Trainer(config)
    return trainer.train(model, X, y, metric=metric, fit_kwargs=fit_kwargs)


def train_with_cv(
    model: TrainableModel,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = 5,
    purge_gap: int = 1,
    min_train_size: int = 100,
    metric: str = "mse",
    verbose: bool = True,
    retrain_on_full: bool = True,
    **fit_kwargs: Any,
) -> CrossValidationTrainingResult:
    """Convenience function to train with cross-validation.

    Args:
        model: Model to train.
        X: Training features.
        y: Training target.
        n_splits: Number of CV splits.
        purge_gap: Samples to purge between train/test.
        min_train_size: Minimum training size.
        metric: Evaluation metric.
        verbose: Show progress.
        retrain_on_full: Retrain on full data after CV.
        **fit_kwargs: Additional kwargs for model.fit().

    Returns:
        CrossValidationTrainingResult with CV scores and model.

    Example:
        >>> from src.model.machine_learning.xgboost_model import XGBoostModel
        >>> model = XGBoostModel(n_estimators=100)
        >>> result = train_with_cv(model, X, y, n_splits=5, purge_gap=10)
        >>> print(f"CV MSE: {result.cv_result.mean_score:.4f}")
    """
    from src.optimisation.walk_forward_cv import create_cv_config

    config = TrainingConfig(verbose=1 if verbose else 0)
    cv_config = create_cv_config(
        n_splits=n_splits,
        purge_gap=purge_gap,
        min_train_size=min_train_size,
    )

    trainer = Trainer(config)
    return trainer.train_with_cv(
        model,
        X,
        y,
        cv_config=cv_config,
        metric=metric,
        fit_kwargs=fit_kwargs,
        retrain_on_full=retrain_on_full,
    )


def save_training_results(
    result: TrainingResult | CrossValidationTrainingResult,
    output_file: str | Path,
) -> None:
    """Save training results to JSON file.

    Args:
        result: Training result to save.
        output_file: Output file path.
    """
    output_path = Path(output_file)

    if isinstance(result, CrossValidationTrainingResult):
        data = {
            "cv_mean_score": result.cv_result.mean_score,
            "cv_std_score": result.cv_result.std_score,
            "cv_metric": result.cv_result.metric_name,
            "cv_n_folds": len(result.cv_result.fold_results),
            "final_train_score": result.final_train_score,
            "training_time": result.training_time,
            "model_params": result.model_params,
            "model_path": str(result.model_path) if result.model_path else None,
        }
    else:
        data = {
            "train_score": result.train_score,
            "val_score": result.val_score,
            "metric": result.metric_name,
            "train_size": result.train_size,
            "val_size": result.val_size,
            "training_time": result.training_time,
            "model_params": result.model_params,
            "model_path": str(result.model_path) if result.model_path else None,
        }

    save_json_pretty(data, output_path)
    logger.info("Saved training results to %s", output_path)
