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
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple, Type, cast

import numpy as np
import optuna
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import matthews_corrcoef  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.utils.class_weight import compute_class_weight  # type: ignore[import-untyped]

from src.constants import TRAIN_SPLIT_LABEL
from src.model.base import BaseModel
from src.path import LABEL_PRIMAIRE_OPTI_DIR

from src.labelling.label_primaire.utils import (
    # Registry and search spaces
    MODEL_REGISTRY,
    TRIPLE_BARRIER_SEARCH_SPACE,
    # Dataclasses
    OptimizationConfig,
    OptimizationResult,
    # Data loading
    load_model_class,
    get_dataset_for_model,
    load_dollar_bars,
    # Volatility
    get_daily_volatility,
    # Barrier helpers
    compute_barriers,
    find_barrier_touch,
    compute_return_and_label,
    set_vertical_barriers,
)

# Suppress warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# =============================================================================
# TRIPLE BARRIER LABELING
# =============================================================================


def _get_path_returns(
    close: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> pd.Series | None:
    """Get returns relative to entry price for a price path."""
    # Ensure t0 and t1 are timestamps
    if not isinstance(t0, pd.Timestamp) or not isinstance(t1, pd.Timestamp):
        return None
    """Get returns relative to entry price for a price path."""
    try:
        path = close.loc[t0:t1]
        if len(path) < 2:
            return None
        entry_price = path.iloc[0]
        return (path / entry_price) - 1
    except (KeyError, TypeError):
        return None


def _update_barrier_touches(close: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Update t1 based on barrier touches for each event."""
    for loc, row in events.iterrows():
        t1 = row["t1"]
        if t1 is None or (isinstance(t1, (int, float)) and pd.isna(t1)):
            continue

        # Ensure t1 is a timestamp
        if not isinstance(t1, pd.Timestamp):
            continue

        try:
            # loc should be a timestamp from DataFrame index
            path_ret = _get_path_returns(close, loc, t1)  # type: ignore
        except (ValueError, TypeError):
            continue
        if path_ret is None:
            continue

        # Ensure barrier values are numeric
        pt_val = row["pt"]
        sl_val = row["sl"]
        pt_barrier = float(pt_val) if isinstance(pt_val, (int, float)) and not pd.isna(pt_val) else None
        sl_barrier = float(sl_val) if isinstance(sl_val, (int, float)) and not pd.isna(sl_val) else None

        touch_time = find_barrier_touch(path_ret, pt_barrier, sl_barrier)
        if touch_time is not None:
            events.loc[loc, "t1"] = touch_time

    return events


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    """
    Apply profit-taking and stop-loss barriers on events.

    This is applyPtSlOnT1 from De Prado. For PRIMARY model,
    barriers are symmetric (no side consideration).
    """
    out = compute_barriers(events, pt_mult, sl_mult)
    out = _update_barrier_touches(close, out)
    return out


def _filter_valid_events(
    t_events: pd.DatetimeIndex,
    trgt: pd.Series,
) -> pd.DatetimeIndex:
    """Filter events to those with valid target volatility."""
    t_events = t_events[t_events.isin(trgt.index)]
    t_events = t_events[trgt.loc[t_events].notna()]
    return t_events


def _build_events_dataframe(
    t_events: pd.DatetimeIndex,
    trgt: pd.Series,
    close_idx: pd.Index,
    max_holding: int,
) -> pd.DataFrame:
    """Build initial events DataFrame with vertical barriers."""
    events = pd.DataFrame(index=t_events)
    events["trgt"] = trgt.loc[t_events].values
    events["t1"] = set_vertical_barriers(t_events, close_idx, max_holding)

    # Remove invalid events
    valid_mask = events["t1"].notna()
    result = events[valid_mask]
    assert isinstance(result, pd.DataFrame), "result must be a DataFrame"
    return result


def _compute_labels(
    events: pd.DataFrame,
    close: pd.Series,
    min_return: float,
) -> pd.DataFrame:
    """Compute return and label for all events."""
    events["ret"] = np.nan
    events["label"] = 0

    for loc, row in events.iterrows():
        t1 = row["t1"]
        if t1 is None or (isinstance(t1, (int, float)) and pd.isna(t1)):
            continue

        # Ensure t1 is a timestamp
        if isinstance(t1, pd.Timestamp):
            try:
                # loc should be a timestamp from DataFrame index
                ret, label = compute_return_and_label(close, loc, t1, min_return)  # type: ignore
            except (ValueError, TypeError):
                ret, label = np.nan, 0
        else:
            ret, label = np.nan, 0
        events.loc[loc, "ret"] = ret
        events.loc[loc, "label"] = label

    return events


def get_events_primary(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_mult: float,
    sl_mult: float,
    trgt: pd.Series,
    max_holding: int,
    min_return: float = 0.0,
) -> pd.DataFrame:
    """
    Generate events for PRIMARY model training.

    Labels are direction-based: +1 (Long), -1 (Short), 0 (Neutral).
    """
    # Filter to valid events
    t_events = _filter_valid_events(t_events, trgt)
    if len(t_events) == 0:
        return pd.DataFrame()

    # Build events DataFrame
    events = _build_events_dataframe(t_events, trgt, close.index, max_holding)
    if events.empty:
        return pd.DataFrame()

    # Apply horizontal barriers
    events = apply_pt_sl_on_t1(close, events, pt_mult, sl_mult)

    # Compute labels
    events = _compute_labels(events, close, min_return)

    result = events[["t1", "trgt", "ret", "label"]]
    assert isinstance(result, pd.DataFrame), "result must be a DataFrame"
    return result


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


def _sample_barrier_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample triple barrier parameters from search space."""
    params = {}
    for name, (_, choices) in TRIPLE_BARRIER_SEARCH_SPACE.items():
        params[name] = trial.suggest_categorical(name, choices)
    return params


def _generate_trial_events(
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    tb_params: Dict[str, Any],
) -> pd.DataFrame:
    """Generate events with given barrier parameters."""
    return get_events_primary(
        close=close,
        t_events=pd.DatetimeIndex(features_df.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility,
        max_holding=tb_params["max_holding"],
        min_return=tb_params["min_return"],
    )


def _validate_events(
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
        reason = f"{trial_prefix}SKIP: events DataFrame is empty"
        logger.debug(reason)
        return False, reason
    
    if len(events) < config.min_train_size:
        reason = (
            f"{trial_prefix}SKIP: not enough events "
            f"({len(events)} < {config.min_train_size} min_train_size)"
        )
        logger.debug(reason)
        return False, reason
    
    label_counts = events["label"].value_counts()
    n_classes = len(label_counts)
    
    if n_classes < 2:
        reason = (
            f"{trial_prefix}SKIP: only {n_classes} class(es) in labels "
            f"(need 2+). Distribution: {label_counts.to_dict()}"
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
    """Sample model hyperparameters."""
    params = {"random_state": config.random_state}
    for name, (_, choices) in search_space.items():
        params[name] = trial.suggest_categorical(name, choices)
    return params


def _evaluate_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    model_name: str | None = None,
    fold_idx: int | None = None,
) -> Tuple[float | None, str]:
    """Evaluate a single CV fold.
    
    Returns
    -------
    Tuple[float | None, str]
        (score, reason) - score is None if evaluation failed, reason explains why.
    """
    model_prefix = f"[{model_name}] " if model_name else ""
    fold_prefix = f"Fold {fold_idx}: " if fold_idx is not None else ""
    
    try:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if len(y_train.unique()) < 2:
            reason = (
                f"{model_prefix}{fold_prefix}SKIP: only 1 class in training set "
                f"(classes: {y_train.unique().tolist()})"
            )
            logger.debug(reason)
            return None, reason

        # Compute class weights
        classes = np.unique(y_train)
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        weight_map = dict(zip(classes, cw))

        # Train and predict
        model = model_class(**model_params)
        model.fit(X_train, y_train)
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

        # Weighted MCC
        sample_weights = np.array([weight_map.get(c, 1.0) for c in y_val])
        score = matthews_corrcoef(y_val, y_pred, sample_weight=sample_weights)
        
        # Handle NaN MCC (can happen with degenerate predictions)
        if np.isnan(score):
            reason = (
                f"{model_prefix}{fold_prefix}SKIP: MCC is NaN "
                f"(likely degenerate predictions or labels)"
            )
            logger.debug(reason)
            return None, reason
        
        return score, "OK"

    except Exception as e:
        reason = (
            f"{model_prefix}{fold_prefix}FAILED: {type(e).__name__}: {str(e)}"
        )
        logger.debug(reason)
        return None, reason


def _run_cv_scoring(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    config: OptimizationConfig,
    trial_number: int | None = None,
    model_name: str | None = None,
) -> Tuple[float, int, int, str]:
    """Run walk-forward CV and return mean score.

    Early pruning: raises TrialPruned at the first failed fold.

    Returns
    -------
    Tuple[float, int, int, str]
        (score, n_valid_folds, n_total_folds, reason) - reason explains the result.

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
        reason = f"{trial_prefix}SKIP: no valid CV splits generated"
        logger.debug(reason)
        raise optuna.TrialPruned(reason)

    # Check we have enough splits
    if len(splits) < config.n_splits:
        reason = (
            f"{trial_prefix}PRUNED: only {len(splits)}/{config.n_splits} "
            f"CV splits generated"
        )
        logger.debug(reason)
        raise optuna.TrialPruned(reason)

    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        score, fold_reason = _evaluate_fold(
            X, y, train_idx, val_idx, model_class, model_params,
            model_name=model_name, fold_idx=fold_idx
        )

        # Early pruning: fail fast on first invalid fold
        if score is None:
            reason = (
                f"{trial_prefix}PRUNED at fold {fold_idx + 1}/{config.n_splits}: "
                f"{fold_reason}"
            )
            logger.debug(reason)
            raise optuna.TrialPruned(reason)

        cv_scores.append(score)

    mean_score = float(np.mean(cv_scores))
    reason = (
        f"{trial_prefix}MCC={mean_score:.4f} "
        f"({len(cv_scores)}/{len(splits)} folds)"
    )
    logger.debug(reason)
    return mean_score, len(cv_scores), len(splits), reason


def create_objective(
    config: OptimizationConfig,
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    model_class: Type[BaseModel],
    model_search_space: Dict[str, Any],
    verbose: bool = True,
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective for joint optimization.

    Parameters
    ----------
    config : OptimizationConfig
        Optimization configuration.
    features_df : pd.DataFrame
        Feature DataFrame.
    close : pd.Series
        Close prices.
    volatility : pd.Series
        Volatility estimates.
    model_class : Type[BaseModel]
        Model class to optimize.
    model_search_space : Dict[str, Any]
        Model hyperparameter search space.
    verbose : bool, default=True
        If True, log trial failures at INFO level (visible).
        If False, log at DEBUG level only.

    Notes
    -----
    Trials are PRUNED (not just scored -inf) if:
    - Not all n_splits folds are successfully validated
    - This ensures only trials with complete CV are considered
    """
    log_level = logging.INFO if verbose else logging.DEBUG

    def objective(trial: optuna.Trial) -> float:
        trial_num = trial.number

        # Sample and generate events
        tb_params = _sample_barrier_params(trial)
        events = _generate_trial_events(features_df, close, volatility, tb_params)

        # Validate events
        is_valid, reason = _validate_events(events, config, trial_num)
        if not is_valid:
            logger.log(log_level, reason)
            raise optuna.TrialPruned(reason)

        # Prepare data
        X, y, events_aligned, reason = _align_features_events(
            features_df, events, config, trial_num
        )
        if X is None or y is None or events_aligned is None:
            logger.log(log_level, reason)
            raise optuna.TrialPruned(reason)

        # Sample model params and run CV
        # Note: _run_cv_scoring raises TrialPruned on first failed fold (early pruning)
        model_params = _sample_model_params(trial, model_search_space, config)

        score, n_valid_folds, n_total_folds, reason = _run_cv_scoring(
            X, y, events_aligned, model_class, model_params, config, trial_num,
            model_name=config.model_name
        )

        # Log successful trial (all folds passed)
        logger.info(
            f"[Trial {trial_num}] OK: MCC={score:.4f} "
            f"({n_valid_folds}/{n_total_folds} folds)"
        )

        return score

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
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label",
    ]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    
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
        features_df = features_df.iloc[:cutoff]
        logger.info(f"Subsampled to {len(features_df)} rows")
    return features_df


def _remove_non_feature_cols(features_df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns from DataFrame."""
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label",
    ]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    result = features_df[feature_cols]
    assert isinstance(result, pd.DataFrame), "result must be a DataFrame"
    return result


def _align_close_to_features(
    close: pd.Series,
    features_df: pd.DataFrame,
) -> pd.Series:
    """Align close prices to feature window."""
    # Ensure close is a Series
    assert isinstance(close, pd.Series), "close must be a pandas Series"
    """Align close prices to feature window."""
    close = close.loc[
        (close.index >= features_df.index[0]) &
        (close.index <= features_df.index[-1])
    ]
    if close.index.has_duplicates:
        close = close.loc[~close.index.duplicated(keep="first")]
    return close


def _prepare_optimization_data(
    config: OptimizationConfig,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data for optimization."""
    logger.info("Loading datasets...")
    features_df = _load_and_filter_features(model_name, config)
    features_df = _subsample_features(features_df, config)
    features_df = _remove_non_feature_cols(features_df)

    bars = load_dollar_bars()
    close_raw = bars["close"]
    assert isinstance(close_raw, pd.Series), "close must be a pandas Series"
    close_aligned = _align_close_to_features(close_raw, features_df)
    assert isinstance(close_aligned, pd.Series), "aligned close must be a pandas Series"
    close = cast(pd.Series, close_aligned)

    volatility = get_daily_volatility(close, span=config.vol_span)

    logger.info(f"Features shape: {features_df.shape}")
    return features_df, close, volatility


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
    best_trial = study.best_trial
    tb_keys = set(TRIPLE_BARRIER_SEARCH_SPACE.keys())

    best_tb = {k: v for k, v in best_trial.params.items() if k in tb_keys}
    best_model = {k: v for k, v in best_trial.params.items() if k not in tb_keys}

    return OptimizationResult(
        model_name=model_name,
        best_params=best_model,
        best_triple_barrier_params=best_tb,
        best_score=best_trial.value if best_trial.value is not None else float("nan"),
        metric="mcc_weighted",
        n_trials=len(study.trials),
    )


def _log_result(result: OptimizationResult) -> None:
    """Log optimization results."""
    logger.info(f"Best score: {result.best_score:.4f}")
    logger.info(f"Best TB params: {result.best_triple_barrier_params}")
    logger.info(f"Best model params: {result.best_params}")


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================


def optimize_model(
    model_name: str,
    config: OptimizationConfig | None = None,
) -> OptimizationResult:
    """Run optimization for a primary model."""
    if config is None:
        config = OptimizationConfig(model_name=model_name)

    logger.info(f"Starting PRIMARY optimization for {model_name}")

    # Load model and data
    model_class = load_model_class(model_name)
    model_search_space = MODEL_REGISTRY[model_name]["search_space"]
    features_df, close, volatility = _prepare_optimization_data(config, model_name)

    # Run Optuna
    study = _create_study(config)
    objective = create_objective(
        config, features_df, close, volatility, model_class, model_search_space
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
    n_trials: int,
    n_splits: int,
) -> List[OptimizationResult]:
    """Run optimization sequentially for all models."""
    all_results: List[OptimizationResult] = []
    
    for i, model_name in enumerate(selected_models, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(selected_models)}] OPTIMISATION: {model_name.upper()}")
        print(f"{'#'*60}")
        
        result = _run_optimization_worker(model_name, n_trials, n_splits)
        all_results.append(result)
        print(f"\n[{model_name}] Score (MCC): {result.best_score:.4f}")
    
    return all_results


def _run_parallel(
    selected_models: List[str],
    n_trials: int,
    n_splits: int,
    max_workers: int,
) -> List[OptimizationResult]:
    """Run optimization in parallel for all models.
    
    Parameters
    ----------
    selected_models : List[str]
        List of model names to optimize.
    n_trials : int
        Number of Optuna trials per model.
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
                n_trials, 
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
        print(f"  Score (MCC): {result.best_score:.4f}")
        print(f"  Triple Barrier:")
        for k, v in result.best_triple_barrier_params.items():
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

    n_trials = int(input("\nNombre de trials [50]: ").strip() or "50")
    n_splits = int(input("Nombre de splits CV [5]: ").strip() or "5")
    
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

    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Modeles: {', '.join(selected_models)}")
    print(f"Trials: {n_trials}, CV: {n_splits}")
    print(f"Execution: {'PARALLELE (' + str(max_workers) + ' workers)' if parallel else 'SEQUENTIELLE'}")
    print(f"{'='*60}")

    if input("\nLancer? (O/n): ").strip().lower() == "n":
        print("Annule.")
        return

    # Run optimization
    if parallel and len(selected_models) > 1:
        all_results = _run_parallel(selected_models, n_trials, n_splits, max_workers)
    else:
        all_results = _run_sequential(selected_models, n_trials, n_splits)
    
    # Final summary
    _print_final_summary(all_results)


if __name__ == "__main__":
    main()
