"""Hyperparameter search spaces for all model types.

This module defines hyperparameter search spaces compatible with Optuna
for all model types in the project:
- Econometric models (Ridge, Lasso)
- Machine Learning models (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning models (LSTM)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import optuna


def _default_max_features() -> list[str | None]:
    """Default max_features options for Random Forest."""
    return ["sqrt", "log2", None]


# ============================================================================
# HYPERPARAMETER SPACE PROTOCOL
# ============================================================================


class HyperparamSpace(Protocol):
    """Protocol for hyperparameter search spaces."""

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of suggested hyperparameters.
        """
        ...


# ============================================================================
# ECONOMETRIC MODELS
# ============================================================================


@dataclass
class RidgeHyperparams:
    """Hyperparameter space for Ridge regression.

    Attributes:
        alpha_log_range: Log range for alpha (regularization).
        fit_intercept: Whether to include intercept option.
        normalize: Whether to include normalization option.
    """

    alpha_log_range: tuple[float, float] = (-5.0, 5.0)
    fit_intercept: bool = True
    normalize: bool = True

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Ridge hyperparameters."""
        params: dict[str, Any] = {
            "alpha": trial.suggest_float("alpha", 10 ** self.alpha_log_range[0], 10 ** self.alpha_log_range[1], log=True),
        }
        if self.fit_intercept:
            params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        if self.normalize:
            params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        return params


@dataclass
class LassoHyperparams:
    """Hyperparameter space for Lasso regression.

    Attributes:
        alpha_log_range: Log range for alpha (regularization).
        fit_intercept: Whether to include intercept option.
        normalize: Whether to include normalization option.
    """

    alpha_log_range: tuple[float, float] = (-5.0, 5.0)
    fit_intercept: bool = True
    normalize: bool = True

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Lasso hyperparameters."""
        params: dict[str, Any] = {
            "alpha": trial.suggest_float("alpha", 10 ** self.alpha_log_range[0], 10 ** self.alpha_log_range[1], log=True),
        }
        if self.fit_intercept:
            params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        if self.normalize:
            params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        return params


# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================


@dataclass
class XGBoostHyperparams:
    """Hyperparameter space for XGBoost.

    Attributes:
        n_estimators_range: Range for number of trees.
        max_depth_range: Range for max tree depth.
        learning_rate_range: Range for learning rate (log scale).
        subsample_range: Range for row subsampling.
        colsample_bytree_range: Range for column subsampling.
        reg_alpha_range: Range for L1 regularization.
        reg_lambda_range: Range for L2 regularization.
        min_child_weight_range: Range for min child weight.
    """

    n_estimators_range: tuple[int, int] = (50, 500)
    max_depth_range: tuple[int, int] = (3, 10)
    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    subsample_range: tuple[float, float] = (0.6, 1.0)
    colsample_bytree_range: tuple[float, float] = (0.6, 1.0)
    reg_alpha_range: tuple[float, float] = (0.0, 10.0)
    reg_lambda_range: tuple[float, float] = (0.0, 10.0)
    min_child_weight_range: tuple[int, int] = (1, 10)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", *self.n_estimators_range),
            "max_depth": trial.suggest_int("max_depth", *self.max_depth_range),
            "learning_rate": trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            "subsample": trial.suggest_float("subsample", *self.subsample_range),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *self.colsample_bytree_range),
            "reg_alpha": trial.suggest_float("reg_alpha", *self.reg_alpha_range),
            "reg_lambda": trial.suggest_float("reg_lambda", *self.reg_lambda_range),
            "min_child_weight": trial.suggest_int("min_child_weight", *self.min_child_weight_range),
        }


@dataclass
class LightGBMHyperparams:
    """Hyperparameter space for LightGBM.

    Attributes:
        n_estimators_range: Range for number of trees.
        max_depth_range: Range for max tree depth (-1 = no limit).
        learning_rate_range: Range for learning rate (log scale).
        num_leaves_range: Range for number of leaves.
        min_child_samples_range: Range for min samples in leaf.
        subsample_range: Range for row subsampling.
        colsample_bytree_range: Range for column subsampling.
        reg_alpha_range: Range for L1 regularization.
        reg_lambda_range: Range for L2 regularization.
    """

    n_estimators_range: tuple[int, int] = (50, 500)
    max_depth_range: tuple[int, int] = (3, 15)
    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    num_leaves_range: tuple[int, int] = (20, 150)
    min_child_samples_range: tuple[int, int] = (5, 100)
    subsample_range: tuple[float, float] = (0.6, 1.0)
    colsample_bytree_range: tuple[float, float] = (0.6, 1.0)
    reg_alpha_range: tuple[float, float] = (0.0, 10.0)
    reg_lambda_range: tuple[float, float] = (0.0, 10.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest LightGBM hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", *self.n_estimators_range),
            "max_depth": trial.suggest_int("max_depth", *self.max_depth_range),
            "learning_rate": trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            "num_leaves": trial.suggest_int("num_leaves", *self.num_leaves_range),
            "min_child_samples": trial.suggest_int("min_child_samples", *self.min_child_samples_range),
            "subsample": trial.suggest_float("subsample", *self.subsample_range),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *self.colsample_bytree_range),
            "reg_alpha": trial.suggest_float("reg_alpha", *self.reg_alpha_range),
            "reg_lambda": trial.suggest_float("reg_lambda", *self.reg_lambda_range),
        }


@dataclass
class CatBoostHyperparams:
    """Hyperparameter space for CatBoost.

    Attributes:
        iterations_range: Range for number of iterations.
        depth_range: Range for tree depth.
        learning_rate_range: Range for learning rate (log scale).
        l2_leaf_reg_range: Range for L2 regularization.
        bagging_temperature_range: Range for bagging temperature.
        random_strength_range: Range for random strength.
    """

    iterations_range: tuple[int, int] = (50, 500)
    depth_range: tuple[int, int] = (4, 10)
    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    l2_leaf_reg_range: tuple[float, float] = (1.0, 10.0)
    bagging_temperature_range: tuple[float, float] = (0.0, 1.0)
    random_strength_range: tuple[float, float] = (0.0, 10.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest CatBoost hyperparameters."""
        return {
            "iterations": trial.suggest_int("iterations", *self.iterations_range),
            "depth": trial.suggest_int("depth", *self.depth_range),
            "learning_rate": trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *self.l2_leaf_reg_range),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *self.bagging_temperature_range),
            "random_strength": trial.suggest_float("random_strength", *self.random_strength_range),
        }


@dataclass
class RandomForestHyperparams:
    """Hyperparameter space for Random Forest.

    Attributes:
        n_estimators_range: Range for number of trees.
        max_depth_range: Range for max tree depth.
        min_samples_split_range: Range for min samples to split.
        min_samples_leaf_range: Range for min samples in leaf.
        max_features: List of max_features options.
    """

    n_estimators_range: tuple[int, int] = (50, 500)
    max_depth_range: tuple[int, int] = (5, 30)
    min_samples_split_range: tuple[int, int] = (2, 20)
    min_samples_leaf_range: tuple[int, int] = (1, 10)
    max_features: list[str | None] = field(default_factory=_default_max_features)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", *self.n_estimators_range),
            "max_depth": trial.suggest_int("max_depth", *self.max_depth_range),
            "min_samples_split": trial.suggest_int("min_samples_split", *self.min_samples_split_range),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *self.min_samples_leaf_range),
            "max_features": trial.suggest_categorical("max_features", self.max_features),
        }


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================


@dataclass
class LSTMHyperparams:
    """Hyperparameter space for LSTM.

    Attributes:
        hidden_size_range: Range for hidden layer size.
        num_layers_range: Range for number of LSTM layers.
        dropout_range: Range for dropout rate.
        sequence_length_range: Range for input sequence length.
        learning_rate_range: Range for learning rate (log scale).
        batch_size_options: List of batch size options.
        epochs_range: Range for number of epochs.
    """

    hidden_size_range: tuple[int, int] = (16, 128)
    num_layers_range: tuple[int, int] = (1, 3)
    dropout_range: tuple[float, float] = (0.0, 0.5)
    sequence_length_range: tuple[int, int] = (5, 30)
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2)
    batch_size_options: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    epochs_range: tuple[int, int] = (50, 200)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest LSTM hyperparameters."""
        num_layers = trial.suggest_int("num_layers", *self.num_layers_range)

        # Dropout only makes sense with num_layers > 1
        dropout = 0.0
        if num_layers > 1:
            dropout = trial.suggest_float("dropout", *self.dropout_range)

        return {
            "hidden_size": trial.suggest_int("hidden_size", *self.hidden_size_range),
            "num_layers": num_layers,
            "dropout": dropout,
            "sequence_length": trial.suggest_int("sequence_length", *self.sequence_length_range),
            "learning_rate": trial.suggest_float("learning_rate", *self.learning_rate_range, log=True),
            "batch_size": trial.suggest_categorical("batch_size", self.batch_size_options),
            "epochs": trial.suggest_int("epochs", *self.epochs_range),
        }


# ============================================================================
# HYPERPARAMETER REGISTRY
# ============================================================================


# Default hyperparameter spaces for each model type
DEFAULT_HYPERPARAMS: dict[str, HyperparamSpace] = {
    # Econometric
    "ridge": RidgeHyperparams(),
    "lasso": LassoHyperparams(),
    # ML
    "xgboost": XGBoostHyperparams(),
    "lightgbm": LightGBMHyperparams(),
    "catboost": CatBoostHyperparams(),
    "randomforest": RandomForestHyperparams(),
    # DL
    "lstm": LSTMHyperparams(),
}


def get_hyperparam_space(model_name: str) -> HyperparamSpace:
    """Get the default hyperparameter space for a model.

    Args:
        model_name: Name of the model (case-insensitive).

    Returns:
        Hyperparameter space for the model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    name_lower = model_name.lower().replace(" ", "").replace("_", "")

    # Handle aliases
    aliases = {
        "ridge": "ridge",
        "ridgeregression": "ridge",
        "lasso": "lasso",
        "lassoregression": "lasso",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "lgb": "lightgbm",
        "catboost": "catboost",
        "cat": "catboost",
        "cb": "catboost",
        "randomforest": "randomforest",
        "rf": "randomforest",
        "lstm": "lstm",
    }

    canonical_name = aliases.get(name_lower)
    if canonical_name is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(DEFAULT_HYPERPARAMS.keys())}"
        )

    return DEFAULT_HYPERPARAMS[canonical_name]


def list_available_models() -> list[str]:
    """List all available model names.

    Returns:
        List of model names with default hyperparameter spaces.
    """
    return list(DEFAULT_HYPERPARAMS.keys())


