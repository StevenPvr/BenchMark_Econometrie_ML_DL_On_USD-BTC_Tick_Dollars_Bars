"""
Primary Model Optimization - De Prado Methodology.

This module optimizes a PRIMARY model that predicts trade DIRECTION (side).
The primary model learns to predict:
- +1 (Long): Price will increase
- -1 (Short): Price will decrease

This is Step 1 of the Meta-Labeling pipeline:
1. Primary Model (this module) -> Predicts direction (side)
2. Meta Model (label_meta) -> Filters false positives using side information

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado, Chapter 3
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type, cast

import numpy as np
import optuna
import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.utils.class_weight import compute_class_weight  # type: ignore[import-untyped]

from src.constants import TRAIN_SPLIT_LABEL
from src.model.base import BaseModel
from src.path import LABEL_PRIMAIRE_OPTI_DIR

from src.labelling.label_primaire.utils import (
    # Registry and search spaces
    MODEL_REGISTRY,
    FOCAL_LOSS_SEARCH_SPACE,
    FOCAL_LOSS_SUPPORTED_MODELS,
    CLASS_WEIGHT_SUPPORTED_MODELS,
    MIN_MINORITY_PREDICTION_RATIO,
    COMPOSITE_SCORE_WEIGHTS,
    # Dataclasses
    OptimizationConfig,
    OptimizationResult,
    CompositeScoreMetrics,
    # Functions
    compute_composite_score,
    # Data loading
    load_model_class,
    get_dataset_for_model,
)
from src.labelling.label_primaire.focal_loss import (
    create_focal_loss_objective,
    compute_focal_alpha_from_class_weights,
)

# Suppress warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)

logger = logging.getLogger(__name__)

# =============================================================================
# LABEL VALIDATION CONSTANTS
# =============================================================================

# Minimum proportion of labels per class for validation
MIN_LABEL_RATIO_POS_NEG: float = 0.05  # 5% minimum for +1/-1 classes
MIN_LABEL_RATIO_NEUTRAL: float = 0.10  # 10% minimum for neutral (0) class
# Columns that should never be used as model features
NON_FEATURE_COLS = [
    "bar_id",
    "timestamp_open",
    "timestamp_close",
    "datetime_open",
    "datetime_close",
    "threshold_used",
    "log_return",
    "split",
    "label",
    "t1",  # Barrier touch time (datetime, not a feature)
]
# Ensure root logger prints INFO when not already configured (for CLI use)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# =============================================================================
# WALK-FORWARD CROSS-VALIDATION
# =============================================================================


class WalkForwardCV:
    """Walk-Forward Cross-Validation with purging and embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 500,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        events: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/val indices with purging."""
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        base_splitter = TimeSeriesSplit(n_splits=self.n_splits)

        splits = []
        for train_idx, val_idx in base_splitter.split(np.arange(n_samples)):
            split = self._process_split(train_idx, val_idx, X, events, embargo_size)
            if split is not None:
                splits.append(split)

        return splits

    def _process_split(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        events: pd.DataFrame,
        embargo_size: int,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
        """Process a single CV split with purging and embargo."""
        if len(train_idx) < self.min_train_size or len(val_idx) == 0:
            return None

        # Purge: remove overlapping training samples
        if "t1" in events.columns:
            train_idx = self._apply_purging(train_idx, val_idx, X, events)

        # Embargo: gap between train and val
        if embargo_size > 0 and len(train_idx) > embargo_size:
            train_idx = train_idx[:-embargo_size]

        if len(train_idx) >= self.min_train_size:
            return (train_idx, val_idx)
        return None

    def _apply_purging(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        events: pd.DataFrame,
    ) -> np.ndarray:
        """Remove training samples whose labels overlap with validation."""
        val_start = X.index[val_idx[0]]
        t1_series = events.iloc[train_idx]["t1"]
        overlap_mask = t1_series.notna() & (t1_series >= val_start)
        return train_idx[~overlap_mask.to_numpy()]


# =============================================================================
# OPTUNA OBJECTIVE HELPERS
# =============================================================================


def _validate_labels(
    events: pd.DataFrame, 
    config: OptimizationConfig,
    trial_number: int | None = None,
) -> Tuple[bool, str]:
    """Validate that events are suitable for training.
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason) - reason explains why validation failed if not valid.
    """
    trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""
    
    if events.empty:
        reason = f"{trial_prefix}PRUNED: events DataFrame is empty"
        logger.debug(reason)
        return False, reason
    
    if len(events) < config.min_train_size:
        reason = f"{trial_prefix}PRUNED: not enough events"
        logger.debug(reason)
        return False, reason
    
    label_counts = events["label"].value_counts()
    n_classes = len(label_counts)
    
    if n_classes < 2:
        reason = f"{trial_prefix}PRUNED: only {n_classes} class(es)"
        logger.debug(reason)
        return False, reason

    # Require all three classes with minimum proportion
    total_events = len(events)
    required_labels = [-1, 0, 1]
    missing = [lbl for lbl in required_labels if lbl not in label_counts.index]
    if missing:
        reason = f"{trial_prefix}PRUNED: missing labels {missing}"
        logger.debug(reason)
        return False, reason

    proportions = (label_counts / total_events).to_dict()
    low_props = {}
    for lbl, prop in proportions.items():
        if lbl == 0 and prop < MIN_LABEL_RATIO_NEUTRAL:
            low_props[lbl] = prop
        if lbl in (-1, 1) and prop < MIN_LABEL_RATIO_POS_NEG:
            low_props[lbl] = prop
    if low_props:
        reason = (
            f"{trial_prefix}PRUNED: class proportion(s) below thresholds"
        )
        logger.debug(reason)
        return False, reason
    
    return True, "OK"


def _align_features_events(
    features_df: pd.DataFrame,
    events: pd.DataFrame,
    config: OptimizationConfig,
    trial_number: int | None = None,
) -> Tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, str]:
    """Align features with events on common index.
    
    Returns
    -------
    Tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, str]
        (X, y, events_aligned, reason) - reason explains failure if any.
    """
    trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""
    
    common_idx = events.index.intersection(features_df.index)
    
    if len(common_idx) < config.min_train_size:
        reason = (
            f"{trial_prefix}SKIP: not enough common indices after alignment "
            f"({len(common_idx)} < {config.min_train_size} min_train_size). "
            f"Events: {len(events)}, Features: {len(features_df)}"
        )
        logger.debug(reason)
        return None, None, None, reason

    X = features_df.loc[common_idx]
    y = events.loc[common_idx, "label"]
    events_aligned = events.loc[common_idx]
    return X, y, events_aligned, "OK"


def _sample_model_params(
    trial: optuna.Trial,
    search_space: Dict[str, Any],
    config: OptimizationConfig,
) -> Dict[str, Any]:
    """Sample model hyperparameters with support for categorical/int/float spaces."""
    params: Dict[str, Any] = {"random_state": config.random_state}

    for name, (param_type, values) in search_space.items():
        if param_type == "categorical":
            params[name] = trial.suggest_categorical(name, values)
        elif param_type == "int":
            # values: [low, high, step?]
            low, high = values[0], values[1]
            step = values[2] if len(values) > 2 else 1
            params[name] = trial.suggest_int(name, int(low), int(high), step=int(step))
        elif param_type == "float":
            # values: [low, high, log?]
            low, high = values[0], values[1]
            log_flag = False
            if len(values) > 2:
                flag = values[2]
                log_flag = bool(flag == "log" or flag is True)
            params[name] = trial.suggest_float(name, float(low), float(high), log=log_flag)
        else:
            # Fallback to categorical to avoid crashing on unknown types
            params[name] = trial.suggest_categorical(name, values)

    return params


def _evaluate_fold(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    model_name: str | None = None,
    fold_idx: int | None = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_class_weights: bool = True,
    early_stopping_rounds: int | None = None,
) -> Tuple[CompositeScoreMetrics | None, str]:
    """Evaluate a single CV fold with composite score.

    The composite score combines:
    - F1 for each class (-1, 0, +1)
    - Minimum F1 across classes (F1_min) to avoid collapsing a class
    - Penalty for sign errors (predicting +1 when true is -1, or vice versa)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    train_idx : np.ndarray
        Training indices.
    val_idx : np.ndarray
        Validation indices.
    model_class : Type[BaseModel]
        Model class to instantiate.
    model_params : Dict[str, Any]
        Model hyperparameters.
    model_name : str, optional
        Model name for logging.
    fold_idx : int, optional
        Fold index for logging.
    use_focal_loss : bool, default=False
        Whether to use focal loss (for LightGBM only).
    focal_gamma : float, default=2.0
        Focal loss gamma parameter.
    use_class_weights : bool, default=True
        Whether to use balanced class weights.

    Returns
    -------
    Tuple[CompositeScoreMetrics | None, str]
        (metrics, reason) - None metrics mean the fold is invalid.
    """
    model_prefix = f"[{model_name}] " if model_name else ""
    fold_prefix = f"Fold {fold_idx}: " if fold_idx is not None else ""

    try:
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx].to_numpy()
            X_val = X.iloc[val_idx].to_numpy()
        else:
            X_train = cast(np.ndarray, X)[train_idx]
            X_val = cast(np.ndarray, X)[val_idx]

        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx].to_numpy()
            y_val = y.iloc[val_idx].to_numpy()
        else:
            y_train = cast(np.ndarray, y)[train_idx]
            y_val = cast(np.ndarray, y)[val_idx]

        classes = np.unique(y_train)
        if len(classes) < 2:
            reason = (
                f"{model_prefix}{fold_prefix}SKIP: only 1 class in training set "
                f"(classes: {classes.tolist()})"
            )
            logger.debug(reason)
            return None, reason

        # Compute class weights for model training
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        weight_map = dict(zip(classes, cw))
        # Guard: if validation contains labels unseen in training, skip this fold
        unseen_labels = set(np.unique(y_val)) - set(classes.tolist())
        if unseen_labels:
            reason = (
                f"{model_prefix}{fold_prefix}SKIP: val labels unseen in train {sorted(unseen_labels)} "
                f"(train classes: {classes.tolist()})"
            )
            logger.debug(reason)
            return None, reason

        # Make a copy of model params to modify
        final_model_params = model_params.copy()
        es_rounds = early_stopping_rounds if early_stopping_rounds and early_stopping_rounds > 0 else None

        # Apply class weights to model if supported
        if use_class_weights and model_name in CLASS_WEIGHT_SUPPORTED_MODELS:
            if model_name == "lightgbm":
                final_model_params["class_weight"] = weight_map
            elif model_name == "catboost":
                # CatBoost uses class_weights as list in class order
                final_model_params["class_weights"] = [weight_map.get(c, 1.0) for c in sorted(classes)]
            elif model_name == "random_forest":
                final_model_params["class_weight"] = weight_map
            elif model_name in ("logistic", "ridge"):
                final_model_params["class_weight"] = weight_map

        # For XGBoost, early stopping is configured via model params (not fit kwargs)
        if model_name == "xgboost" and es_rounds:
            final_model_params["early_stopping_rounds"] = es_rounds

        # Apply focal loss for LightGBM if enabled
        if use_focal_loss and model_name in FOCAL_LOSS_SUPPORTED_MODELS:
            # Compute alpha weights from class distribution
            alpha = compute_focal_alpha_from_class_weights(weight_map, classes=tuple(sorted(classes)))
            focal_objective = create_focal_loss_objective(
                gamma=focal_gamma,
                alpha=alpha,
                n_classes=len(classes),
            )
            final_model_params["objective"] = focal_objective
            # Remove class_weight when using focal loss (alpha handles it)
            final_model_params.pop("class_weight", None)

        # Train and predict
        model = model_class(**final_model_params)
        fit_kwargs: Dict[str, Any] = {}

        if model_name == "xgboost":
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        elif model_name == "catboost":
            fit_kwargs["eval_set"] = (X_val, y_val)
            if es_rounds:
                fit_kwargs["early_stopping_rounds"] = es_rounds
            fit_kwargs["verbose"] = False
        elif model_name == "lightgbm":
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if es_rounds:
                try:
                    import lightgbm as lgb  # type: ignore
                    fit_kwargs["callbacks"] = [lgb.early_stopping(es_rounds, verbose=False)]
                except Exception:
                    fit_kwargs["early_stopping_rounds"] = es_rounds
            fit_kwargs["verbose"] = -1

        model.fit(X_train, y_train, **fit_kwargs)
        y_pred = model.predict(X_val)

        # Check if model predicts only one class (degenerate case)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) < 2:
            reason = (
                f"{model_prefix}{fold_prefix}SKIP: model predicts only 1 class "
                f"(predicted: {unique_preds.tolist()}, true classes: {np.unique(y_val).tolist()})"
            )
            logger.debug(reason)
            return None, reason

        # Compute composite score with F1 per class and sign error penalty
        metrics = compute_composite_score(
            y_true=np.array(y_val),
            y_pred=np.array(y_pred),
        )

        # Guard against degenerate models that predict too few minority classes
        # If model predicts less than MIN_MINORITY_PREDICTION_RATIO of long/short,
        # apply a penalty to discourage this behavior
        n_minority_preds = np.sum((y_pred == -1) | (y_pred == 1))
        minority_pred_ratio = n_minority_preds / len(y_pred)

        if minority_pred_ratio < MIN_MINORITY_PREDICTION_RATIO:
            # Apply penalty: reduce composite score proportionally
            penalty_factor = minority_pred_ratio / MIN_MINORITY_PREDICTION_RATIO
            penalized_score = metrics.composite_score * penalty_factor
            logger.debug(
                f"{model_prefix}{fold_prefix}PENALTY: minority predictions "
                f"{minority_pred_ratio:.1%} < {MIN_MINORITY_PREDICTION_RATIO:.1%}, "
                f"penalty_factor={penalty_factor:.2f}"
            )
            # Create new metrics with penalized score
            metrics = CompositeScoreMetrics(
                f1_minus1=metrics.f1_minus1,
                f1_zero=metrics.f1_zero,
                f1_plus1=metrics.f1_plus1,
                f1_min=metrics.f1_min,
                sign_error_rate=metrics.sign_error_rate,
                composite_score=penalized_score,
                n_sign_errors=metrics.n_sign_errors,
                n_total=metrics.n_total,
            )

        return metrics, "OK"

    except Exception as e:
        reason = (
            f"{model_prefix}{fold_prefix}FAILED: {type(e).__name__}: {str(e)}"
        )
        logger.debug(reason)
        return None, reason


@dataclass
class CVScoringResult:
    """Result of cross-validation scoring with composite metrics."""

    composite_score: float  # Mean composite score across folds
    f1_minus1: float        # Mean F1 for class -1
    f1_zero: float          # Mean F1 for class 0
    f1_plus1: float         # Mean F1 for class +1
    f1_min: float           # Mean of the worst-class F1 across folds
    sign_error_rate: float  # Mean sign error rate
    n_valid_folds: int      # Number of valid folds
    n_total_folds: int      # Total number of folds


def _run_cv_scoring(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    config: OptimizationConfig,
    trial_number: int | None = None,
    model_name: str | None = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_class_weights: bool = True,
) -> CVScoringResult:
    """Run walk-forward CV and return composite score metrics.

    Early pruning: raises TrialPruned at the first failed fold.

    The composite score combines:
    - F1 for each class (-1, 0, +1)
    - Minimum F1 across classes to enforce coverage of all labels
    - Penalty for sign errors (worst kind of prediction error)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    events : pd.DataFrame
        Events DataFrame for purging.
    model_class : Type[BaseModel]
        Model class to instantiate.
    model_params : Dict[str, Any]
        Model hyperparameters.
    config : OptimizationConfig
        Optimization configuration.
    trial_number : int, optional
        Trial number for logging.
    model_name : str, optional
        Model name for logging.
    use_focal_loss : bool, default=False
        Whether to use focal loss.
    focal_gamma : float, default=2.0
        Focal loss gamma parameter.
    use_class_weights : bool, default=True
        Whether to use balanced class weights.

    Returns
    -------
    CVScoringResult
        Dataclass with composite score and per-class metrics.

    Raises
    ------
    optuna.TrialPruned
        If any fold fails evaluation (early pruning).
    """
    trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""

    cv = WalkForwardCV(
        n_splits=config.n_splits,
        min_train_size=config.min_train_size,
        embargo_pct=0.01,
    )

    splits = cv.split(X, events)
    if len(splits) == 0:
        logger.debug("%sPRUNED: no valid CV splits generated", trial_prefix)
        raise optuna.TrialPruned(f"{trial_prefix}PRUNED")

    # Check we have enough splits
    if len(splits) < config.n_splits:
        logger.debug(
            "%sPRUNED: only %d/%d CV splits generated",
            trial_prefix, len(splits), config.n_splits,
        )
        raise optuna.TrialPruned(f"{trial_prefix}PRUNED")

    # Collect metrics from all folds
    fold_metrics: list[CompositeScoreMetrics] = []
    # Convert once to numpy to avoid repeated pandas slicing
    X_values = X.to_numpy()
    y_values = y.to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        metrics, fold_reason = _evaluate_fold(
            X_values, y_values, train_idx, val_idx, model_class, model_params,
            model_name=model_name,
            fold_idx=fold_idx,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            use_class_weights=use_class_weights,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        # Early pruning: fail fast on first invalid fold
        if metrics is None:
            reason = f"{trial_prefix}PRUNED at fold {fold_idx + 1}: {fold_reason}"
            logger.info(reason)
            raise optuna.TrialPruned(reason)

        fold_metrics.append(metrics)

    # Compute mean metrics across folds
    mean_composite = float(np.mean([m.composite_score for m in fold_metrics]))
    mean_f1_m1 = float(np.mean([m.f1_minus1 for m in fold_metrics]))
    mean_f1_0 = float(np.mean([m.f1_zero for m in fold_metrics]))
    mean_f1_p1 = float(np.mean([m.f1_plus1 for m in fold_metrics]))
    mean_f1_min = float(np.mean([m.f1_min for m in fold_metrics]))
    mean_sign_err = float(np.mean([m.sign_error_rate for m in fold_metrics]))

    logger.debug(
        f"{trial_prefix}COMPOSITE={mean_composite:.4f} "
        f"(F1[-1]={mean_f1_m1:.3f}, F1[0]={mean_f1_0:.3f}, F1[+1]={mean_f1_p1:.3f}, "
        f"F1_min={mean_f1_min:.3f}, SignErr={mean_sign_err:.1%}; "
        f"{len(fold_metrics)}/{len(splits)} folds)"
    )

    return CVScoringResult(
        composite_score=mean_composite,
        f1_minus1=mean_f1_m1,
        f1_zero=mean_f1_0,
        f1_plus1=mean_f1_p1,
        f1_min=mean_f1_min,
        sign_error_rate=mean_sign_err,
        n_valid_folds=len(fold_metrics),
        n_total_folds=len(splits),
    )


def _sample_focal_loss_params(
    trial: optuna.Trial,
    config: OptimizationConfig,
) -> Dict[str, Any]:
    """Sample focal loss parameters from search space.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial.
    config : OptimizationConfig
        Optimization configuration.

    Returns
    -------
    Dict[str, Any]
        Focal loss parameters: use_focal_loss, focal_gamma.
    """
    params: Dict[str, Any] = {}

    # Only sample focal-specific params if model supports it
    if config.optimize_focal_params and config.model_name in FOCAL_LOSS_SUPPORTED_MODELS:
        params["use_focal_loss"] = trial.suggest_categorical(
            "use_focal_loss", FOCAL_LOSS_SEARCH_SPACE["use_focal_loss"][1]
        )
        params["focal_gamma"] = trial.suggest_categorical(
            "focal_gamma", FOCAL_LOSS_SEARCH_SPACE["focal_gamma"][1]
        )
    else:
        # Use config defaults for focal loss
        params["use_focal_loss"] = config.use_focal_loss
        params["focal_gamma"] = config.focal_gamma

    return params


def create_objective(
    config: OptimizationConfig,
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_class: Type[BaseModel],
    model_search_space: Dict[str, Any],
    verbose: bool = True,
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective for model hyperparameter optimization.

    Parameters
    ----------
    config : OptimizationConfig
        Optimization configuration.
    X : pd.DataFrame
        Feature DataFrame (already aligned with labels).
    y : pd.Series
        Pre-calculated labels from relabel_datasets.py.
    events : pd.DataFrame
        Events DataFrame for purging (contains t1 column).
    model_class : Type[BaseModel]
        Model class to optimize.
    model_search_space : Dict[str, Any]
        Model hyperparameter search space.
    verbose : bool, default=True
        If True, log trial failures at INFO level (visible).
        If False, log at DEBUG level only.

    Notes
    -----
    - Labels are PRE-CALCULATED by relabel_datasets.py. This function only
      optimizes model hyperparameters, not triple-barrier parameters.
    - Trials are PRUNED if not all n_splits folds are successfully validated.
    - Focal loss parameters are optimized for supported models (LightGBM).
    - Class weights are applied to supported models for imbalanced classes.
    """
    log_level = logging.INFO if verbose else logging.DEBUG

    # Pre-compute label distribution for logging
    label_counts = y.value_counts()
    n_total = len(y)
    pct_m1 = (label_counts.get(-1, 0) or 0) / n_total * 100 if n_total > 0 else 0.0
    pct_0 = (label_counts.get(0, 0) or 0) / n_total * 100 if n_total > 0 else 0.0
    pct_p1 = (label_counts.get(1, 0) or 0) / n_total * 100 if n_total > 0 else 0.0

    logger.info(
        f"Label distribution: -1={pct_m1:.1f}% 0={pct_0:.1f}% +1={pct_p1:.1f}% "
        f"(n={n_total:,})"
    )

    def objective(trial: optuna.Trial) -> float:
        trial_num = trial.number

        # Sample model params
        model_params = _sample_model_params(trial, model_search_space, config)

        # Sample focal loss params (for supported models)
        focal_params = _sample_focal_loss_params(trial, config)
        use_focal_loss = focal_params.get("use_focal_loss", config.use_focal_loss)
        focal_gamma = focal_params.get("focal_gamma", config.focal_gamma)

        # Run CV with composite score (F1 per class + sign error penalty)
        # Note: _run_cv_scoring raises TrialPruned on first failed fold (early pruning)
        cv_result = _run_cv_scoring(
            X, y, events, model_class, model_params, config, trial_num,
            model_name=config.model_name,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            use_class_weights=config.use_class_weights,
        )

        # Log successful trial (all folds passed)
        focal_info = f", focal={use_focal_loss}, γ={focal_gamma}" if use_focal_loss else ""
        logger.info(
            f"[Trial {trial_num}] OK: SCORE={cv_result.composite_score:.4f} "
            f"(F1[-1]={cv_result.f1_minus1:.3f}, F1[0]={cv_result.f1_zero:.3f}, "
            f"F1[+1]={cv_result.f1_plus1:.3f}, F1_min={cv_result.f1_min:.3f}, "
            f"SignErr={cv_result.sign_error_rate:.1%}{focal_info})"
        )

        return cv_result.composite_score

    return objective


# =============================================================================
# OPTIMIZATION HELPERS
# =============================================================================


def _handle_missing_values(
    features_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Handle missing values by filling with median and logging statistics.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features (may contain NaN values).
    model_name : str
        Name of the model (for logging context).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with NaN values filled with median.
    """
    # Identify feature columns (exclude non-feature columns)
    feature_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
    
    if len(feature_cols) == 0:
        return features_df
    
    # Count missing values before filling
    missing_before = features_df[feature_cols].isna().sum()
    total_missing = missing_before.sum()
    features_with_missing = (missing_before > 0).sum()
    
    if total_missing > 0:
        logger.info(
            f"[{model_name.upper()}] Missing values: {total_missing:,} total "
            f"across {features_with_missing}/{len(feature_cols)} features"
        )
        
        # Fill with median
        features_df = features_df.copy()
        for col in feature_cols:
            col_series = features_df[col]
            n_missing = int(col_series.isna().sum())
            
            if n_missing > 0:
                median_val = float(col_series.median())
                
                # If median is NaN (all values are NaN), use 0 as fallback
                if pd.isna(median_val) or np.isnan(median_val):
                    logger.warning(
                        f"[{model_name.upper()}] Column '{col}' is entirely NaN, "
                        f"filling {n_missing:,} values with 0"
                    )
                    features_df[col] = col_series.fillna(0.0)
                else:
                    features_df[col] = col_series.fillna(median_val)
                    logger.debug(
                        f"[{model_name.upper()}] Filled {n_missing:,} NaN in '{col}' "
                        f"with median={median_val:.6f}"
                    )
        
        # Verify no NaN remaining
        missing_after = features_df[feature_cols].isna().sum().sum()
        if missing_after > 0:
            logger.warning(
                f"[{model_name.upper()}] {missing_after:,} NaN values remain "
                "after median filling (likely all-NaN columns)"
            )
    else:
        logger.info(f"[{model_name.upper()}] No missing values found in {len(feature_cols)} features")
    
    assert isinstance(features_df, pd.DataFrame), "features_df must be a DataFrame"
    return cast(pd.DataFrame, features_df)


def _downcast_feature_columns(features_df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric feature columns to float32/int32 to speed up fit/predict."""
    feature_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
    if not feature_cols:
        return features_df

    df_copy = features_df.copy()
    for col in feature_cols:
        col_series = df_copy[col]
        if pd.api.types.is_float_dtype(col_series):
            df_copy[col] = col_series.astype(np.float32)
        elif pd.api.types.is_integer_dtype(col_series):
            df_copy[col] = col_series.astype(np.int32)
    return df_copy


def _load_and_filter_features(
    model_name: str,
    config: OptimizationConfig,
) -> pd.DataFrame:
    """Load features and apply filtering."""
    features_df = get_dataset_for_model(model_name)

    # Filter to train split
    if "split" in features_df.columns:
        features_df = features_df[features_df["split"] == TRAIN_SPLIT_LABEL].copy()
        features_df = features_df.drop(columns=["split"])

    # Set datetime index
    if "datetime_close" in features_df.columns:
        features_df = features_df.set_index("datetime_close")
    features_df = features_df.sort_index()

    # Remove duplicates
    if features_df.index.has_duplicates:
        features_df = features_df[~features_df.index.duplicated(keep="first")]
    
    assert isinstance(features_df, pd.DataFrame), "features_df must be a DataFrame"
    features_df = cast(pd.DataFrame, features_df)

    # Handle missing values (fill with median)
    features_df = _handle_missing_values(features_df, model_name)
    # Downcast numeric feature columns for faster fit/predict
    features_df = _downcast_feature_columns(features_df)

    return features_df


def _subsample_features(
    features_df: pd.DataFrame,
    config: OptimizationConfig,
) -> pd.DataFrame:
    """Subsample features for faster optimization."""
    if 0 < config.data_fraction < 1.0:
        cutoff = max(
            int(len(features_df) * config.data_fraction),
            config.min_train_size * (config.n_splits + 1)
        )
        features_df = cast(pd.DataFrame, features_df.iloc[:cutoff])
        logger.info(f"Subsampled to {len(features_df)} rows")
    return features_df


def _remove_non_feature_cols(features_df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns from DataFrame."""
    feature_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
    result = features_df[feature_cols]
    assert isinstance(result, pd.DataFrame), "result must be a DataFrame"
    return result


def _prepare_optimization_data(
    config: OptimizationConfig,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and prepare data for optimization.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        (X, y, events) where:
        - X: Feature matrix
        - y: Pre-calculated labels
        - events: DataFrame with t1 column for purging
    """
    logger.info("Loading datasets...")
    features_df = _load_and_filter_features(model_name, config)

    # Verify labels exist
    if "label" not in features_df.columns:
        raise ValueError(
            f"Dataset for {model_name} does not contain 'label' column. "
            "Labels must be pre-calculated by relabel_datasets.py"
        )

    # Filter out rows with NaN labels
    n_before = len(features_df)
    features_df = cast(pd.DataFrame, features_df[features_df["label"].notna()].copy())
    n_after = len(features_df)
    if n_before != n_after:
        logger.info(
            f"Filtered {n_before - n_after:,} rows with NaN labels "
            f"({n_after:,}/{n_before:,} remaining)"
        )

    if len(features_df) == 0:
        raise ValueError(
            f"No valid labels found in dataset for {model_name}. "
            "Run relabel_datasets.py first to generate labels."
        )

    # Extract labels and events for purging
    y = features_df["label"].copy()

    # Create events DataFrame for purging (needs t1 column)
    # t1 represents the barrier touch time, used for purging overlapping samples
    if "t1" in features_df.columns:
        events = features_df[["label", "t1"]].copy()
    else:
        # If no t1 column, create dummy events without purging capability
        logger.warning(
            "Dataset does not contain 't1' column. Purging will be disabled."
        )
        events = features_df[["label"]].copy()

    # Subsample if configured
    features_df = cast(pd.DataFrame, _subsample_features(cast(pd.DataFrame, features_df), config))
    y = y.loc[features_df.index]
    events = events.loc[features_df.index]

    # Remove non-feature columns
    X = _remove_non_feature_cols(features_df)

    logger.info(f"Features shape: {X.shape}, Labels: {len(y)}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    return X, y, events


def _create_study(config: OptimizationConfig) -> optuna.Study:
    """Create Optuna study with configured settings."""
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )


def _build_result(
    study: optuna.Study,
    model_name: str,
    config: OptimizationConfig,
) -> OptimizationResult:
    """Build optimization result from study."""
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise ValueError(
            "No Optuna trials completed successfully; inspect pruning reasons above "
            "or relax constraints (labels, F1 floor, CV splits)."
        ) from exc
    focal_keys = set(FOCAL_LOSS_SEARCH_SPACE.keys())

    best_focal = {k: v for k, v in best_trial.params.items() if k in focal_keys}
    best_model = {
        k: v for k, v in best_trial.params.items()
        if k not in focal_keys
    }

    # If focal loss params were not optimized, use config defaults
    if not best_focal:
        best_focal = {
            "use_focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma,
        }

    # Build metric description with weights
    weights = COMPOSITE_SCORE_WEIGHTS
    delta = weights.get("delta", 0.0)
    metric_desc = (
        f"composite_score(α={weights['alpha']:.2f}*F1_dir + "
        f"β={weights['beta']:.2f}*F1_0 + δ={delta:.2f}*F1_min "
        f"- γ={weights['gamma']:.2f}*sign_err)"
    )

    return OptimizationResult(
        model_name=model_name,
        best_params=best_model,
        best_focal_loss_params=best_focal,
        best_score=best_trial.value if best_trial.value is not None else float("nan"),
        metric=metric_desc,
        n_trials=len(study.trials),
    )


def _log_result(result: OptimizationResult) -> None:
    """Log optimization results."""
    logger.info(f"Best composite score: {result.best_score:.4f}")
    logger.info(f"Metric: {result.metric}")
    logger.info(f"Best focal loss params: {result.best_focal_loss_params}")
    logger.info(f"Best model params: {result.best_params}")


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================


def optimize_model(
    model_name: str,
    config: OptimizationConfig | None = None,
) -> OptimizationResult:
    """Run optimization for a primary model.

    Note: Labels must be pre-calculated by relabel_datasets.py before running
    this optimization. This function only optimizes model hyperparameters.
    """
    if config is None:
        config = OptimizationConfig(model_name=model_name)

    logger.info(f"Starting PRIMARY optimization for {model_name}")
    logger.info("Note: Using pre-calculated labels from dataset")

    # Load model and data
    model_class = load_model_class(model_name)
    model_search_space = MODEL_REGISTRY[model_name]["search_space"]
    X, y, events = _prepare_optimization_data(config, model_name)

    # Validate labels before starting
    is_valid, reason = _validate_labels(events, config)
    if not is_valid:
        raise ValueError(f"Invalid labels in dataset: {reason}")

    # Run Optuna
    study = _create_study(config)
    objective = create_objective(
        config, X, y, events, model_class, model_search_space
    )
    study.optimize(
        objective, n_trials=config.n_trials, timeout=config.timeout, show_progress_bar=True
    )

    # Build and save result
    result = _build_result(study, model_name, config)
    result.save(LABEL_PRIMAIRE_OPTI_DIR / f"{model_name}_optimization.json")

    _log_result(result)
    return result


# =============================================================================
# CLI
# =============================================================================


def select_models_interactive() -> List[str]:
    """Interactive model selection (supports multiple models).
    
    Returns
    -------
    List[str]
        List of selected model names.
    """
    models = list(MODEL_REGISTRY.keys())

    print("\n" + "=" * 60)
    print("PRIMARY MODEL OPTIMIZATION")
    print("=" * 60)
    print("\nPredicts trade DIRECTION (side: +1 Long, -1 Short)")
    print("\nModels disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        info = MODEL_REGISTRY[model]
        print(f"  {i}. {model:<15} ({info['dataset']})")

    print("-" * 40)
    print("  0. ALL (tous les modeles)")
    print("-" * 40)
    print("\nPlusieurs choix possibles: separer par virgule ou espace")
    print("Exemples: '1,2,3' ou '1 2 3' ou 'lightgbm xgboost' ou '0' pour tous")

    while True:
        try:
            choice = input("\nChoisir: ").strip()
            
            # Handle "all" selection
            if choice == "0" or choice.lower() == "all":
                return models.copy()
            
            # Parse multiple selections (comma or space separated)
            # Remove parentheses and other non-alphanumeric separators, then split
            cleaned_choice = choice.replace("(", " ").replace(")", " ").replace(",", " ").replace(";", " ")
            raw_selections = cleaned_choice.split()
            selected_models: List[str] = []
            
            for sel in raw_selections:
                sel = sel.strip()
                if not sel:
                    continue
                
                # Remove any remaining non-digit characters for numeric selection
                numeric_sel = "".join(c for c in sel if c.isdigit())
                    
                if numeric_sel and numeric_sel.isdigit():
                    idx = int(numeric_sel) - 1
                    if 0 <= idx < len(models):
                        model_name = models[idx]
                        if model_name not in selected_models:
                            selected_models.append(model_name)
                    else:
                        print(f"Index invalide: {numeric_sel}")
                elif sel.lower() in models:
                    if sel.lower() not in selected_models:
                        selected_models.append(sel.lower())
                else:
                    print(f"Modele inconnu: {sel}")
            
            if selected_models:
                return selected_models
            print("Aucun modele valide selectionne.")
            
        except KeyboardInterrupt:
            print("\nAnnule.")
            exit(0)


def _get_trials_per_model(
    selected_models: List[str],
    default_trials: int,
) -> Dict[str, int]:
    """Prompt user for number of trials per model.

    Parameters
    ----------
    selected_models : List[str]
        List of selected model names.
    default_trials : int
        Default number of trials for all models.

    Returns
    -------
    Dict[str, int]
        Mapping of model name to number of trials.
    """
    if len(selected_models) == 1:
        return {selected_models[0]: default_trials}

    custom_input = input(
        f"\nNombre de trials different par modele? (o/N) [N]: "
    ).strip().lower()

    if custom_input != "o":
        return {model: default_trials for model in selected_models}

    trials_per_model: Dict[str, int] = {}
    print("\nEntrez le nombre de trials pour chaque modele:")
    print(f"(Appuyez Entree pour utiliser la valeur par defaut: {default_trials})")

    for model in selected_models:
        while True:
            try:
                model_input = input(f"  {model} [{default_trials}]: ").strip()
                if not model_input:
                    trials_per_model[model] = default_trials
                else:
                    trials_per_model[model] = int(model_input)
                break
            except ValueError:
                print("    Veuillez entrer un nombre valide.")

    return trials_per_model


def _run_optimization_worker(
    model_name: str,
    n_trials: int,
    n_splits: int,
) -> OptimizationResult:
    """Worker function for parallel optimization.

    This function is designed to be called in a separate process.
    It configures logging and runs optimization for a single model.

    Parameters
    ----------
    model_name : str
        Name of the model to optimize.
    n_trials : int
        Number of Optuna trials.
    n_splits : int
        Number of CV splits.

    Returns
    -------
    OptimizationResult
        Optimization result for the model.
    """
    # Configure logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [{model_name.upper()}] - %(levelname)s - %(message)s",
    )

    config = OptimizationConfig(
        model_name=model_name,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    return optimize_model(model_name, config)


def _run_sequential(
    selected_models: List[str],
    trials_per_model: Dict[str, int],
    n_splits: int,
) -> List[OptimizationResult]:
    """Run optimization sequentially for all models.

    Parameters
    ----------
    selected_models : List[str]
        List of model names to optimize.
    trials_per_model : Dict[str, int]
        Mapping of model name to number of trials.
    n_splits : int
        Number of CV splits.

    Returns
    -------
    List[OptimizationResult]
        List of optimization results.
    """
    all_results: List[OptimizationResult] = []

    for i, model_name in enumerate(selected_models, 1):
        n_trials = trials_per_model[model_name]
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(selected_models)}] OPTIMISATION: {model_name.upper()} ({n_trials} trials)")
        print(f"{'#'*60}")

        result = _run_optimization_worker(model_name, n_trials, n_splits)
        all_results.append(result)
        print(f"\n[{model_name}] Score (MCC): {result.best_score:.4f}")

    return all_results


def _run_parallel(
    selected_models: List[str],
    trials_per_model: Dict[str, int],
    n_splits: int,
    max_workers: int,
) -> List[OptimizationResult]:
    """Run optimization in parallel for all models.

    Parameters
    ----------
    selected_models : List[str]
        List of model names to optimize.
    trials_per_model : Dict[str, int]
        Mapping of model name to number of trials.
    n_splits : int
        Number of CV splits.
    max_workers : int
        Maximum number of parallel workers.

    Returns
    -------
    List[OptimizationResult]
        List of optimization results.
    """
    all_results: List[OptimizationResult] = []

    print(f"\nLancement de {len(selected_models)} optimisations en parallele "
          f"({max_workers} workers)...")
    print("Note: Les logs seront entremeles entre les modeles.\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_model = {
            executor.submit(
                _run_optimization_worker,
                model_name,
                trials_per_model[model_name],
                n_splits
            ): model_name
            for model_name in selected_models
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"\n[TERMINE] {model_name.upper()} - Score (MCC): {result.best_score:.4f}")
            except Exception as e:
                print(f"\n[ERREUR] {model_name.upper()}: {e}")

    return all_results


def _print_final_summary(all_results: List[OptimizationResult]) -> None:
    """Print final summary of all optimization results."""
    print(f"\n{'='*60}")
    print(f"RESUME FINAL - {len(all_results)} MODELE(S) OPTIMISE(S)")
    print(f"{'='*60}")

    # Sort by score descending
    sorted_results = sorted(all_results, key=lambda r: r.best_score, reverse=True)

    for result in sorted_results:
        print(f"\n{result.model_name.upper()}")
        print(f"  Score: {result.best_score:.4f}")
        if result.best_focal_loss_params:
            print(f"  Focal Loss:")
            for k, v in result.best_focal_loss_params.items():
                print(f"    {k}: {v}")
        print(f"  Model params:")
        for k, v in result.best_params.items():
            print(f"    {k}: {v}")


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    selected_models = select_models_interactive()

    print(f"\nModeles selectionnes ({len(selected_models)}):")
    for m in selected_models:
        print(f"  - {m}")

    default_trials = int(input("\nNombre de trials par defaut [50]: ").strip() or "50")
    n_splits = int(input("Nombre de splits CV [5]: ").strip() or "5")

    # Get trials per model (may be customized per model)
    trials_per_model = _get_trials_per_model(selected_models, default_trials)

    # Ask for parallel execution if multiple models selected
    parallel = False
    max_workers = 1
    cpu_count = os.cpu_count() or 4

    if len(selected_models) > 1:
        parallel_input = input(f"Execution parallele? (O/n) [O]: ").strip().lower()
        parallel = parallel_input != "n"

        if parallel:
            default_workers = min(len(selected_models), cpu_count)
            workers_input = input(
                f"Nombre de workers [{default_workers}] (max CPU: {cpu_count}): "
            ).strip()
            max_workers = int(workers_input) if workers_input else default_workers
            max_workers = min(max_workers, len(selected_models), cpu_count)

    # Build trials summary
    unique_trials = set(trials_per_model.values())
    if len(unique_trials) == 1:
        trials_summary = str(list(unique_trials)[0])
    else:
        trials_summary = ", ".join(f"{m}:{t}" for m, t in trials_per_model.items())

    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"Modeles: {', '.join(selected_models)}")
    print(f"Trials: {trials_summary}")
    print(f"CV splits: {n_splits}")
    print(f"Execution: {'PARALLELE (' + str(max_workers) + ' workers)' if parallel else 'SEQUENTIELLE'}")
    print(f"{'='*60}")

    if input("\nLancer? (O/n): ").strip().lower() == "n":
        print("Annule.")
        return

    # Run optimization
    if parallel and len(selected_models) > 1:
        all_results = _run_parallel(selected_models, trials_per_model, n_splits, max_workers)
    else:
        all_results = _run_sequential(selected_models, trials_per_model, n_splits)

    # Final summary
    _print_final_summary(all_results)


if __name__ == "__main__":
    main()
