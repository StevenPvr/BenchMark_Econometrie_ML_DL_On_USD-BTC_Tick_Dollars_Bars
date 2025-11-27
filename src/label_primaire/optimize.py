"""Joint optimization for labeling parameters and primary model hyperparameters.

This module optimizes:
    - Labeling parameters: pt_mult, sl_mult, max_holding_period, min_ret
    - Primary model hyperparameters (LightGBM, XGBoost, CatBoost, RF, Ridge, Lasso, Logistic)

The primary model parameters will be reused by the meta-model in label_meta module.

Reference:
    De Prado, M. L. (2018). Advances in Financial Machine Learning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_primaire.triple_barrier import get_triple_barrier_events
from src.optimisation.walk_forward_cv import (
    create_cv_config,
    walk_forward_cv,
)
from src.utils import save_json_pretty

logger = get_logger(__name__)

__all__ = [
    "LabelingHyperparams",
    "ModelHyperparams",
    "LightGBMHyperparams",
    "XGBoostHyperparams",
    "CatBoostHyperparams",
    "RandomForestHyperparams",
    "RidgeHyperparams",
    "LassoHyperparams",
    "LogisticHyperparams",
    "JointOptimizationConfig",
    "JointOptimizationResult",
    "JointOptimizer",
    "optimize_joint_params",
    "save_optimization_results",
    "compute_class_weights",
    "get_model_hyperparams",
    "get_model_class",
    "normalize_volatility_scale",
    "AVAILABLE_MODELS",
]


# =============================================================================
# AVAILABLE MODELS
# =============================================================================

AVAILABLE_MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "rf",
    "ridge",
    "lasso",
    "logistic",
]


def get_model_class(model_name: str) -> type:
    """Get model class by name.

    Args:
        model_name: Name of the model (lightgbm, xgboost, catboost, rf, ridge, lasso, logistic).

    Returns:
        Model class.

    Raises:
        ValueError: If model_name is unknown.
    """
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
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")


# =============================================================================
# HYPERPARAMETER SPACES
# =============================================================================


@dataclass
class LabelingHyperparams:
    """Hyperparameter space for triple-barrier labeling.

    Attributes:
        pt_mult_range: Range for profit-taking multiplier.
        sl_mult_range: Range for stop-loss multiplier.
        max_holding_range: Range for maximum holding period (bars).
        min_ret_range: Range for minimum return threshold (neutral zone).
        symmetric_barriers: If True, pt_mult == sl_mult.
    """

    pt_mult_range: tuple[float, float] = (0.1, 1.5)
    sl_mult_range: tuple[float, float] = (0.1, 1.5)
    max_holding_range: tuple[int, int] = (10, 80)
    min_ret_range: tuple[float, float] = (0.00005, 0.002)
    symmetric_barriers: bool = True

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest labeling hyperparameters for a trial."""
        if self.symmetric_barriers:
            mult = trial.suggest_float(
                "barrier_mult", self.pt_mult_range[0], self.pt_mult_range[1]
            )
            pt_mult = mult
            sl_mult = mult
        else:
            pt_mult = trial.suggest_float(
                "pt_mult", self.pt_mult_range[0], self.pt_mult_range[1]
            )
            sl_mult = trial.suggest_float(
                "sl_mult", self.sl_mult_range[0], self.sl_mult_range[1]
            )

        max_holding = trial.suggest_int(
            "max_holding_period",
            self.max_holding_range[0],
            self.max_holding_range[1],
        )

        min_ret = trial.suggest_float(
            "min_ret", self.min_ret_range[0], self.min_ret_range[1]
        )

        return {
            "pt_mult": pt_mult,
            "sl_mult": sl_mult,
            "max_holding_period": max_holding,
            "min_ret": min_ret,
        }


class ModelHyperparams(ABC):
    """Abstract base class for model hyperparameter spaces."""

    @abstractmethod
    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        ...

    @abstractmethod
    def get_param_keys(self) -> set[str]:
        """Return the set of parameter keys for this model."""
        ...


@dataclass
class LightGBMHyperparams(ModelHyperparams):
    """Hyperparameter space for LightGBM model."""

    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    num_leaves_range: tuple[int, int] = (10, 100)
    max_depth_range: tuple[int, int] = (3, 8)
    min_data_in_leaf_range: tuple[int, int] = (10, 100)
    feature_fraction_range: tuple[float, float] = (0.5, 1.0)
    bagging_fraction_range: tuple[float, float] = (0.5, 1.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest LightGBM hyperparameters for a trial."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", self.num_leaves_range[0], self.num_leaves_range[1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.max_depth_range[0], self.max_depth_range[1]
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", self.min_data_in_leaf_range[0], self.min_data_in_leaf_range[1]
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", self.feature_fraction_range[0], self.feature_fraction_range[1]
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", self.bagging_fraction_range[0], self.bagging_fraction_range[1]
            ),
            "bagging_freq": 1,
            "verbosity": -1,
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {
            "learning_rate", "num_leaves", "max_depth",
            "min_data_in_leaf", "feature_fraction", "bagging_fraction"
        }


@dataclass
class XGBoostHyperparams(ModelHyperparams):
    """Hyperparameter space for XGBoost model."""

    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    max_depth_range: tuple[int, int] = (3, 10)
    n_estimators_range: tuple[int, int] = (50, 300)
    min_child_weight_range: tuple[int, int] = (1, 10)
    subsample_range: tuple[float, float] = (0.5, 1.0)
    colsample_bytree_range: tuple[float, float] = (0.5, 1.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest XGBoost hyperparameters for a trial."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.max_depth_range[0], self.max_depth_range[1]
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", self.n_estimators_range[0], self.n_estimators_range[1]
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", self.min_child_weight_range[0], self.min_child_weight_range[1]
            ),
            "subsample": trial.suggest_float(
                "subsample", self.subsample_range[0], self.subsample_range[1]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", self.colsample_bytree_range[0], self.colsample_bytree_range[1]
            ),
            "verbosity": 0,
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {
            "learning_rate", "max_depth", "n_estimators",
            "min_child_weight", "subsample", "colsample_bytree"
        }


@dataclass
class CatBoostHyperparams(ModelHyperparams):
    """Hyperparameter space for CatBoost model."""

    learning_rate_range: tuple[float, float] = (0.01, 0.3)
    depth_range: tuple[int, int] = (4, 10)
    iterations_range: tuple[int, int] = (50, 300)
    l2_leaf_reg_range: tuple[float, float] = (1.0, 10.0)
    bagging_temperature_range: tuple[float, float] = (0.0, 1.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest CatBoost hyperparameters for a trial."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True
            ),
            "depth": trial.suggest_int(
                "depth", self.depth_range[0], self.depth_range[1]
            ),
            "iterations": trial.suggest_int(
                "iterations", self.iterations_range[0], self.iterations_range[1]
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", self.l2_leaf_reg_range[0], self.l2_leaf_reg_range[1]
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", self.bagging_temperature_range[0], self.bagging_temperature_range[1]
            ),
            "verbose": False,
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {
            "learning_rate", "depth", "iterations",
            "l2_leaf_reg", "bagging_temperature"
        }


@dataclass
class RandomForestHyperparams(ModelHyperparams):
    """Hyperparameter space for Random Forest model."""

    n_estimators_range: tuple[int, int] = (50, 300)
    max_depth_range: tuple[int, int] = (5, 20)
    min_samples_split_range: tuple[int, int] = (2, 20)
    min_samples_leaf_range: tuple[int, int] = (1, 10)
    max_features_choices: tuple[str, ...] = ("sqrt", "log2")

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Random Forest hyperparameters for a trial."""
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.n_estimators_range[0], self.n_estimators_range[1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.max_depth_range[0], self.max_depth_range[1]
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", self.min_samples_split_range[0], self.min_samples_split_range[1]
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", self.min_samples_leaf_range[0], self.min_samples_leaf_range[1]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", list(self.max_features_choices)
            ),
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features"
        }


@dataclass
class RidgeHyperparams(ModelHyperparams):
    """Hyperparameter space for Ridge Classifier model."""

    alpha_range: tuple[float, float] = (0.001, 100.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Ridge hyperparameters for a trial."""
        return {
            "alpha": trial.suggest_float(
                "alpha", self.alpha_range[0], self.alpha_range[1], log=True
            ),
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {"alpha"}


@dataclass
class LassoHyperparams(ModelHyperparams):
    """Hyperparameter space for Lasso Classifier (L1 Logistic) model."""

    C_range: tuple[float, float] = (0.001, 100.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Lasso hyperparameters for a trial."""
        return {
            "C": trial.suggest_float(
                "C", self.C_range[0], self.C_range[1], log=True
            ),
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {"C"}


@dataclass
class LogisticHyperparams(ModelHyperparams):
    """Hyperparameter space for Logistic Regression model."""

    C_range: tuple[float, float] = (0.001, 100.0)

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest Logistic hyperparameters for a trial."""
        return {
            "C": trial.suggest_float(
                "C", self.C_range[0], self.C_range[1], log=True
            ),
            "random_state": DEFAULT_RANDOM_STATE,
        }

    def get_param_keys(self) -> set[str]:
        return {"C"}


def get_model_hyperparams(model_name: str) -> ModelHyperparams:
    """Get hyperparameter space for a model.

    Args:
        model_name: Name of the model.

    Returns:
        ModelHyperparams instance.
    """
    model_name = model_name.lower()

    if model_name == "lightgbm":
        return LightGBMHyperparams()
    elif model_name == "xgboost":
        return XGBoostHyperparams()
    elif model_name == "catboost":
        return CatBoostHyperparams()
    elif model_name in ("rf", "random_forest"):
        return RandomForestHyperparams()
    elif model_name == "ridge":
        return RidgeHyperparams()
    elif model_name == "lasso":
        return LassoHyperparams()
    elif model_name in ("logistic", "logistic_regression"):
        return LogisticHyperparams()
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class JointOptimizationConfig:
    """Configuration for joint optimization.

    Attributes:
        model_name: Name of the primary model.
        n_trials: Number of Optuna trials.
        n_splits: Number of CV splits.
        purge_gap: Gap between train/test in CV.
        min_train_size: Minimum training size.
        direction: Optimization direction.
        sampler: Optuna sampler type.
        random_state: Random seed.
    """

    model_name: str = "lightgbm"
    n_trials: int = 100
    n_splits: int = 5
    purge_gap: int = 5
    min_train_size: int = 200
    direction: str = "maximize"
    sampler: str = "tpe"
    random_state: int = DEFAULT_RANDOM_STATE


@dataclass
class JointOptimizationResult:
    """Result from joint optimization.

    Attributes:
        model_name: Name of the primary model.
        best_labeling_params: Best labeling parameters (pt_mult, sl_mult, max_holding, min_ret).
        best_model_params: Best model parameters (reused by meta-model).
        best_score: Best MCC achieved.
        best_trial_number: Trial number of best result.
        n_trials: Total trials.
        n_completed: Completed trials.
        study: Optuna study object.
    """

    model_name: str
    best_labeling_params: dict[str, Any]
    best_model_params: dict[str, Any]
    best_score: float
    best_trial_number: int
    n_trials: int
    n_completed: int
    study: optuna.Study


# =============================================================================
# UTILITIES
# =============================================================================


def compute_class_weights(y: np.ndarray | pd.Series) -> dict[Any, float]:
    """Balanced class weights: n_samples / (n_classes * count_c)."""
    y_arr = np.asarray(y).ravel()
    classes, counts = np.unique(y_arr, return_counts=True)
    n_samples = len(y_arr)

    if len(classes) == 0:
        return {}

    weights = n_samples / (len(classes) * counts.astype(float))
    return {cls: float(w) for cls, w in zip(classes, weights)}


def compute_sharpe_ratio(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from returns."""
    returns = np.asarray(returns)
    valid = returns[~np.isnan(returns)]
    if len(valid) < 2:
        return 0.0
    mean_ret = np.mean(valid)
    std_ret = np.std(valid, ddof=1)
    if std_ret == 0 or not np.isfinite(std_ret):
        return 0.0
    return float(np.sqrt(252) * mean_ret / std_ret)


def normalize_volatility_scale(
    volatility: pd.Series,
    prices: pd.Series,
    scale_threshold: float = 50.0,
    rescale_factor: float = 100.0,
) -> pd.Series:
    """Align volatility magnitude with price log returns.

    Realized volatility is computed on ``log_return`` which is stored as
    percentage points (x100). That makes the volatility roughly two orders of
    magnitude larger than the log returns obtained directly from prices. To
    keep the triple-barrier thresholds meaningful, we downscale the volatility
    when its median magnitude is far above the price-return median.

    Args:
        volatility: Realized volatility series (possibly scaled by 100).
        prices: Close price series used to derive log returns.
        scale_threshold: Ratio above which we consider volatility mis-scaled.
        rescale_factor: Factor to divide by when rescaling.

    Returns:
        Volatility series rescaled when needed (original otherwise).
    """
    if len(volatility) == 0 or len(prices) < 2:
        return volatility

    price_returns = np.log(prices / prices.shift(1))
    ret_median = float(np.nanmedian(np.abs(price_returns)))
    vol_median = float(np.nanmedian(np.abs(volatility)))

    if ret_median == 0 or not np.isfinite(ret_median) or not np.isfinite(vol_median):
        return volatility

    ratio = vol_median / ret_median

    if ratio > scale_threshold:
        logger.info(
            "Volatility median (%.4f) is %.1fx larger than price log-return median (%.6f). "
            "Rescaling volatility by /%d to match log-return units.",
            vol_median,
            ratio,
            ret_median,
            int(rescale_factor),
        )
        return volatility / rescale_factor

    return volatility


# =============================================================================
# JOINT OPTIMIZER
# =============================================================================


class JointOptimizer:
    """Joint optimizer for labeling and primary model parameters.

    This optimizer finds optimal combination of:
    - Triple-barrier labeling parameters
    - Primary model hyperparameters (can be reused by meta-model)

    Example:
        >>> optimizer = JointOptimizer(model_name="xgboost")
        >>> result = optimizer.optimize(prices, X, volatility)
        >>> print(result.best_labeling_params)
        >>> print(result.best_model_params)  # Reuse for meta-model!
    """

    def __init__(
        self,
        model_name: str = "lightgbm",
        labeling_space: LabelingHyperparams | None = None,
        model_space: ModelHyperparams | None = None,
        config: JointOptimizationConfig | None = None,
    ) -> None:
        """Initialize the joint optimizer.

        Args:
            model_name: Name of the primary model.
            labeling_space: Labeling hyperparameter space.
            model_space: Model hyperparameter space (auto-detected if None).
            config: Optimization configuration.
        """
        self.model_name = model_name.lower()
        self.labeling_space = labeling_space or LabelingHyperparams()
        self.model_space = model_space or get_model_hyperparams(self.model_name)
        self.config = config or JointOptimizationConfig(model_name=self.model_name)

        # Get model class
        self.model_class = get_model_class(self.model_name)

        self._study: optuna.Study | None = None
        self._prices: pd.Series | None = None
        self._X: np.ndarray | None = None
        self._volatility: pd.Series | None = None

    def _generate_labels(
        self,
        pt_mult: float,
        sl_mult: float,
        max_holding_period: int,
        min_ret: float = 0.0,
    ) -> pd.Series:
        """Generate triple-barrier labels."""
        if self._prices is None or self._volatility is None:
            raise RuntimeError("Data not set.")

        events = get_triple_barrier_events(
            prices=self._prices,
            pt_sl=[pt_mult, sl_mult],
            target_volatility=self._volatility,
            max_holding_period=max_holding_period,
            min_ret=min_ret,
        )

        labels = pd.Series(index=self._prices.index, dtype=float)
        for _, row in events.iterrows():
            t_start = int(row["t_start"])
            if t_start < len(labels):
                labels.iloc[t_start] = row["label"]

        return labels

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        if self._X is None or self._prices is None:
            raise RuntimeError("Data not set.")

        # Suggest labeling parameters
        labeling_params = self.labeling_space.suggest(trial)

        # Suggest model parameters
        model_params = self.model_space.suggest(trial)

        # Generate labels
        try:
            labels = self._generate_labels(
                pt_mult=labeling_params["pt_mult"],
                sl_mult=labeling_params["sl_mult"],
                max_holding_period=labeling_params["max_holding_period"],
                min_ret=labeling_params.get("min_ret", 0.0),
            )
        except Exception as e:
            logger.debug("Trial %d: Failed to generate labels: %s", trial.number, e)
            raise optuna.TrialPruned() from e

        # Align X and labels
        valid_mask = ~labels.isna()
        X_valid = self._X[valid_mask.values]  # type: ignore
        y_valid = np.asarray(labels[valid_mask]).astype(int)

        if len(y_valid) < self.config.min_train_size:
            raise optuna.TrialPruned()

        # Create CV config
        cv_config = create_cv_config(
            n_splits=self.config.n_splits,
            purge_gap=self.config.purge_gap,
            min_train_size=self.config.min_train_size,
        )

        # Train model with walk-forward CV
        try:
            class_weight = compute_class_weights(y_valid)
            full_model_params = dict(model_params)

            # Add class_weight for models that support it
            if self.model_name in ("lightgbm", "xgboost", "catboost", "rf"):
                full_model_params["class_weight"] = class_weight

            model = self.model_class(**full_model_params)
            cv_result = walk_forward_cv(
                model=model,
                X=X_valid,
                y=y_valid,  # type: ignore
                config=cv_config,
                metric="mcc",
                verbose=False,
            )

            score = cv_result.mean_score

            if not np.isfinite(score):
                raise optuna.TrialPruned()

            return score

        except Exception as e:
            logger.debug("Trial %d: CV failed: %s", trial.number, e)
            raise optuna.TrialPruned() from e

    def optimize(
        self,
        prices: pd.Series,
        X: np.ndarray | pd.DataFrame,
        volatility: pd.Series,
        show_progress_bar: bool = True,
        verbose: bool = True,
    ) -> JointOptimizationResult:
        """Run joint optimization.

        Args:
            prices: Price series (close).
            X: Feature matrix.
            volatility: Volatility series.
            show_progress_bar: Show progress bar.
            verbose: Log progress.

        Returns:
            JointOptimizationResult with best params.
        """
        self._prices = prices
        self._X = np.asarray(X)
        self._volatility = normalize_volatility_scale(volatility, prices)

        np.random.seed(self.config.random_state)

        if self.config.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.random_state)  # type: ignore

        self._study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            study_name=f"joint_optimization_{self.model_name}",
        )

        if verbose:
            logger.info("Starting joint optimization for %s: %d trials",
                       self.model_name.upper(), self.config.n_trials)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            show_progress_bar=show_progress_bar,
        )

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        completed = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])

        if completed == 0:
            raise RuntimeError("No trials completed")

        best_trial = self._study.best_trial

        if best_trial.value is None:
            raise RuntimeError("Best trial has no value")

        best_score: float = best_trial.value

        # Extract best params
        best_params = dict(best_trial.params)

        # Separate labeling and model params
        labeling_keys = {"barrier_mult", "pt_mult", "sl_mult", "max_holding_period", "min_ret"}
        model_keys = self.model_space.get_param_keys()

        labeling_params = {k: v for k, v in best_params.items() if k in labeling_keys}
        model_params = {k: v for k, v in best_params.items() if k in model_keys}

        # Handle symmetric barriers
        if "barrier_mult" in labeling_params:
            mult = labeling_params.pop("barrier_mult")
            labeling_params["pt_mult"] = mult
            labeling_params["sl_mult"] = mult

        # Add fixed model params
        model_params["random_state"] = self.config.random_state

        if verbose:
            logger.info("Optimization complete: best MCC=%.6f", best_score)
            logger.info("Best labeling params: %s", labeling_params)
            logger.info("Best %s params: %s", self.model_name.upper(), model_params)

        return JointOptimizationResult(
            model_name=self.model_name,
            best_labeling_params=labeling_params,
            best_model_params=model_params,
            best_score=best_score,
            best_trial_number=best_trial.number,
            n_trials=len(self._study.trials),
            n_completed=completed,
            study=self._study,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def optimize_joint_params(
    prices: pd.Series,
    X: np.ndarray | pd.DataFrame,
    volatility: pd.Series,
    model_name: str = "lightgbm",
    n_trials: int = 100,
    n_splits: int = 5,
    purge_gap: int = 5,
    symmetric_barriers: bool = True,
    verbose: bool = True,
) -> JointOptimizationResult:
    """Convenience function for joint optimization.

    Args:
        prices: Price series.
        X: Feature matrix.
        volatility: Volatility series.
        model_name: Name of the primary model.
        n_trials: Number of trials.
        n_splits: CV splits.
        purge_gap: Gap in CV.
        symmetric_barriers: If True, pt_mult == sl_mult.
        verbose: Log progress.

    Returns:
        JointOptimizationResult with best params.
    """
    optimizer = JointOptimizer(
        model_name=model_name,
        labeling_space=LabelingHyperparams(symmetric_barriers=symmetric_barriers),
        config=JointOptimizationConfig(
            model_name=model_name,
            n_trials=n_trials,
            n_splits=n_splits,
            purge_gap=purge_gap,
        ),
    )

    return optimizer.optimize(
        prices=prices,
        X=X,
        volatility=volatility,
        verbose=verbose,
    )


def save_optimization_results(
    result: JointOptimizationResult,
    output_path: Path | str,
) -> None:
    """Save optimization results to JSON.

    Args:
        result: Optimization result.
        output_path: Output file path.
    """
    output = {
        "model_name": result.model_name,
        "best_labeling_params": result.best_labeling_params,
        "best_model_params": result.best_model_params,
        "best_score": result.best_score,
        "best_trial_number": result.best_trial_number,
        "n_trials": result.n_trials,
        "n_completed": result.n_completed,
    }
    save_json_pretty(output, output_path)
    logger.info("Saved optimization results to %s", output_path)
