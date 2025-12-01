"""
Meta-Labeling Optimization - De Prado Methodology (Chapter 3.6).

This module implements the META model that FILTERS false positives from the primary model.
The meta model learns to predict:
- 1: Take the trade (primary model signal is correct)
- 0: Skip the trade (primary model signal is likely wrong)

This is Step 2 of the Meta-Labeling pipeline:
1. Primary Model (label_primaire) -> Predicts direction (side: +1 Long, -1 Short)
2. Meta Model (this module) -> Filters false positives using side information

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado, Chapter 3.6
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
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Tuple, Type, cast

import numpy as np
import optuna
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import balanced_accuracy_score, f1_score  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.utils.class_weight import compute_class_weight  # type: ignore[import-untyped]

from src.constants import TRAIN_SPLIT_LABEL
from src.model.base import BaseModel
from src.path import LABEL_META_OPTI_DIR

from src.labelling.label_meta.utils import (
    get_meta_labeled_dataset,
    # Registry
    MODEL_REGISTRY,
    TRIPLE_BARRIER_SEARCH_SPACE,
    # Dataclasses
    MetaOptimizationConfig,
    MetaOptimizationResult,
    # Data loading
    load_model_class,
    get_dataset_for_model,
    load_dollar_bars,
    load_primary_model,
    get_available_primary_models,
    # Volatility
    get_daily_volatility,
    # Barrier helpers
    set_vertical_barriers_meta,
    compute_side_adjusted_barriers,
    get_barrier_touches_for_side,
    find_first_touch_time,
    compute_meta_label,
)

# Suppress warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# =============================================================================
# META-LABELING CORE FUNCTIONS
# =============================================================================

MIN_BIN_RATIO = 0.15
PARALLEL_MIN_EVENTS = 10_000


def _resolve_n_jobs(requested: int | None) -> int:
    """Return a valid worker count capped to available CPU cores."""
    max_cores = multiprocessing.cpu_count()
    if requested is None or requested <= 0:
        return max_cores
    return min(requested, max_cores)


def _split_t_events(t_events: pd.DatetimeIndex, n_splits: int) -> List[pd.DatetimeIndex]:
    """Split events into roughly equal chunks while preserving order."""
    chunks = np.array_split(t_events, n_splits)
    return [pd.DatetimeIndex(chunk) for chunk in chunks if len(chunk) > 0]


def _filter_valid_events_meta(
    t_events: pd.DatetimeIndex,
    trgt: pd.Series,
    side: pd.Series,
) -> pd.DatetimeIndex:
    """Filter events to those with valid volatility and side."""
    valid_mask = t_events.isin(trgt.index) & t_events.isin(side.index)
    return t_events[valid_mask]


def _build_events_dataframe_meta(
    t_events: pd.DatetimeIndex,
    trgt: pd.Series,
    side: pd.Series,
) -> pd.DataFrame:
    """Build initial events DataFrame with side information."""
    events = pd.DataFrame(index=t_events)
    events["trgt"] = trgt.loc[t_events].values
    events["side"] = side.loc[t_events].values

    # Remove events with invalid side (must be +1 or -1)
    return events.loc[events["side"].isin([1, -1])]


def _get_path_returns(
    close: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> pd.Series | None:
    """Get returns relative to entry price for a price path."""
    try:
        path = close.loc[t0:t1]
        if len(path) < 2:
            return None
        entry_price = path.iloc[0]
        return (path / entry_price) - 1
    except (KeyError, TypeError):
        return None


def _apply_meta_barriers(close: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Apply barriers and find first touch for meta-labeling."""
    for loc, row in events.iterrows():
        t1 = row["t1"]
        side_val = int(row["side"])
        pt_val = row["pt"]
        sl_val = row["sl"]
        # Handle potential NaN values safely
        try:
            is_pt_na = pd.isna(pt_val)
            pt = 0.0 if (isinstance(is_pt_na, bool) and is_pt_na) else float(pt_val)
        except (TypeError, ValueError):
            pt = 0.0
        try:
            is_sl_na = pd.isna(sl_val)
            sl = 0.0 if (isinstance(is_sl_na, bool) and is_sl_na) else float(sl_val)
        except (TypeError, ValueError):
            sl = 0.0

        # Ensure loc and t1 are Timestamps
        loc_ts = cast(pd.Timestamp, loc)
        t1_ts = cast(pd.Timestamp, t1)
        path_ret = _get_path_returns(close, loc_ts, t1_ts)
        if path_ret is None:
            continue

        pt_touches, sl_touches = get_barrier_touches_for_side(path_ret, side_val, pt, sl)
        touch_time = find_first_touch_time(pt_touches, sl_touches)

        if touch_time is not None:
            events.loc[loc, "t1"] = touch_time

    return events


def get_events_meta(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_mult: float,
    sl_mult: float,
    trgt: pd.Series,
    max_holding: int,
    side: pd.Series,
) -> pd.DataFrame:
    """
    Generate events for META-LABELING with side information.

    Barriers depend on the primary model's prediction (side).
    For Long: PT is above, SL is below.
    For Short: PT is below, SL is above.
    """
    # Filter to valid events
    t_events = _filter_valid_events_meta(t_events, trgt, side)
    if len(t_events) == 0:
        return pd.DataFrame()

    # Build events DataFrame
    events = _build_events_dataframe_meta(t_events, trgt, side)
    if len(events) == 0:
        return pd.DataFrame()

    # Set vertical barriers
    events = set_vertical_barriers_meta(events, close.index, max_holding)
    events = cast(pd.DataFrame, events[events["t1"].notna()].copy())  # Ensure we have a DataFrame
    if len(events) == 0:
        return pd.DataFrame()

    # Compute side-adjusted barriers
    events = cast(pd.DataFrame, compute_side_adjusted_barriers(events, pt_mult, sl_mult))

    # Apply barriers and find first touch
    events = cast(pd.DataFrame, _apply_meta_barriers(close, events))

    result: pd.DataFrame = cast(pd.DataFrame, events[["t1", "trgt", "side", "pt", "sl"]].copy())
    return result


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Generate meta-labels (getBins) for meta-labeling.

    The meta-label is 1 if the PRIMARY model's prediction was CORRECT.
    Rule: bin = 1 if (return * side > 0), else 0
    """
    out = events.copy()
    out["ret"] = np.nan
    out["bin"] = 0

    for loc, row in out.iterrows():
        t1_val = row["t1"]
        side = int(row["side"])

        if t1_val is None or cast(bool, pd.isna(t1_val)):
            continue

        try:
            price_t0 = close.loc[loc]
            price_t1 = close.loc[t1_val]
            ret = (price_t1 - price_t0) / price_t0
            out.loc[loc, "ret"] = ret
            out.loc[loc, "bin"] = compute_meta_label(ret, side)
        except (KeyError, TypeError):
            continue

    return out


# =============================================================================
# WALK-FORWARD CV
# =============================================================================


class WalkForwardCV:
    """Walk-Forward CV with purging for meta-labeling."""

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

        # Purge overlapping labels
        if "t1" in events.columns:
            train_idx = self._apply_purging(train_idx, val_idx, X, events)

        # Embargo
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
        overlap = t1_series.notna() & (t1_series >= val_start)
        return train_idx[~overlap.to_numpy()]


# =============================================================================
# OPTUNA OBJECTIVE HELPERS
# =============================================================================


def _sample_barrier_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample triple barrier parameters from search space."""
    params = {}
    for name, (_, choices) in TRIPLE_BARRIER_SEARCH_SPACE.items():
        params[name] = trial.suggest_categorical(name, choices)
    return params


def _process_chunk_events_meta(
    chunk: pd.DatetimeIndex,
    close: pd.Series,
    volatility: pd.Series,
    side_predictions: pd.Series,
    tb_params: Dict[str, Any],
) -> pd.DataFrame:
    """Process a chunk of events for meta-labeling (module-level for pickling)."""
    ev = get_events_meta(
        close=close,
        t_events=chunk,
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility,
        max_holding=tb_params["max_holding"],
        side=side_predictions,
    )
    if not ev.empty:
        ev = get_bins(ev, close)
    return ev


def _generate_trial_events_meta(
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    side_predictions: pd.Series,
    tb_params: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> pd.DataFrame:
    """Generate meta events with given barrier parameters (optionally in parallel)."""
    t_events = pd.DatetimeIndex(features_df.index)

    if (not config.parallelize_labeling) or len(t_events) < max(config.parallel_min_events, 1):
        return _process_chunk_events_meta(t_events, close, volatility, side_predictions, tb_params)

    n_jobs = _resolve_n_jobs(config.n_jobs)
    event_chunks = _split_t_events(t_events, n_jobs)
    if len(event_chunks) <= 1:
        return _process_chunk_events_meta(t_events, close, volatility, side_predictions, tb_params)

    with ProcessPoolExecutor(max_workers=len(event_chunks)) as executor:
        futures = [
            executor.submit(_process_chunk_events_meta, chunk, close, volatility, side_predictions, tb_params)
            for chunk in event_chunks
        ]
        results = [f.result() for f in futures]

    non_empty = [df for df in results if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()

    return pd.concat(non_empty).sort_index()


def _validate_meta_events(
    events: pd.DataFrame,
    config: MetaOptimizationConfig,
) -> Tuple[bool, str]:
    """Validate that events are suitable for training."""
    if events.empty:
        return False, "PRUNED: empty events"
    if len(events) < config.min_train_size:
        return False, "PRUNED: not enough events"
    if len(events["bin"].value_counts()) < 2:
        return False, "PRUNED: only one class"
    return True, "OK"


def _align_features_events_meta(
    features_df: pd.DataFrame,
    events: pd.DataFrame,
    config: MetaOptimizationConfig,
) -> Tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None]:
    """Align features with events on common index."""
    common_idx = events.index.intersection(features_df.index)
    if len(common_idx) < config.min_train_size:
        return None, None, None

    X = features_df.loc[common_idx]
    y = events.loc[common_idx, "bin"]
    events_aligned = events.loc[common_idx]
    return X, y, events_aligned


def _sample_model_params(
    trial: optuna.Trial,
    search_space: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> Dict[str, Any]:
    """Sample model hyperparameters."""
    params = {"random_state": config.random_state}
    for name, (_, choices) in search_space.items():
        params[name] = trial.suggest_categorical(name, choices)
    return params


def _evaluate_fold_meta(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
) -> Tuple[float | None, float | None, float | None]:
    """Evaluate a single CV fold for meta-labeling."""
    try:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if len(y_train.unique()) < 2:
            return None, None, None

        # Train meta model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Combined metrics for robust binary classification
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division="warn")

        # Mean of both metrics
        score = (bal_acc + f1_weighted) / 2
        return score, bal_acc, f1_weighted

    except Exception:
        return None, None, None


def _run_cv_scoring_meta(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> Tuple[float, float, float]:
    """Run walk-forward CV and return mean score and component metrics."""
    cv = WalkForwardCV(
        n_splits=config.n_splits,
        min_train_size=config.min_train_size,
    )

    cv_scores: list[float] = []
    cv_bal_acc: list[float] = []
    cv_f1_weighted: list[float] = []
    for train_idx, val_idx in cv.split(X, events):
        score, bal_acc, f1_weighted = _evaluate_fold_meta(
            X, y, train_idx, val_idx, model_class, model_params
        )
        if score is None or bal_acc is None or f1_weighted is None:
            raise optuna.TrialPruned("PRUNED: invalid fold")
        cv_scores.append(score)
        cv_bal_acc.append(bal_acc)
        cv_f1_weighted.append(f1_weighted)

    if len(cv_scores) == 0:
        raise optuna.TrialPruned("PRUNED: no valid CV folds")
    mean_score = float(np.mean(cv_scores))
    mean_bal_acc = float(np.mean(cv_bal_acc))
    mean_f1_weighted = float(np.mean(cv_f1_weighted))

    logger.info(
        "CV metrics - score: %.4f | balanced_acc: %.4f | f1_weighted: %.4f | folds: %d",
        mean_score,
        mean_bal_acc,
        mean_f1_weighted,
        len(cv_scores),
    )

    return mean_score, mean_bal_acc, mean_f1_weighted


def create_objective(
    config: MetaOptimizationConfig,
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    side_predictions: pd.Series,
    meta_model_class: Type[BaseModel],
    meta_search_space: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective for meta model optimization."""

    def objective(trial: optuna.Trial) -> float:
        # Sample and generate events
        tb_params = _sample_barrier_params(trial)
        events = _generate_trial_events_meta(
            features_df, close, volatility, side_predictions, tb_params, config
        )

        # Validate events
        is_valid, reason = _validate_meta_events(events, config)
        if not is_valid:
            logger.debug(reason)
            raise optuna.TrialPruned(reason)

        # Prepare data
        X, y, events_aligned = _align_features_events_meta(features_df, events, config)
        if X is None or y is None or events_aligned is None:
            raise optuna.TrialPruned("PRUNED: alignment failed")

        # Sample model params and run CV
        model_params = _sample_model_params(trial, meta_search_space, config)
        score, mean_bal_acc, mean_f1w = _run_cv_scoring_meta(
            X, y, events_aligned, meta_model_class, model_params, config
        )

        logger.info(
            "[Trial %s] OBJ=%.4f | balanced_acc=%.4f | f1_weighted=%.4f",
            trial.number,
            score,
            mean_bal_acc,
            mean_f1w,
        )

        return score

    return objective


# =============================================================================
# OPTIMIZATION HELPERS
# =============================================================================


def _load_and_filter_features(
    config: MetaOptimizationConfig,
) -> pd.DataFrame:
    """Load features for meta model and apply filtering."""
    # Load meta model's dataset type (tree for lightgbm, etc.)
    features_df, _ = get_meta_labeled_dataset(
        config.primary_model_name,
        config.meta_model_name,
    )

    # Filter to train split
    if "split" in features_df.columns:
        features_df = cast(pd.DataFrame, features_df[features_df["split"] == TRAIN_SPLIT_LABEL].copy())
        features_df = features_df.drop(columns=["split"])

    # Set datetime index
    if "datetime_close" in features_df.columns:
        features_df = cast(pd.DataFrame, features_df.set_index("datetime_close"))
    features_df = cast(pd.DataFrame, features_df.sort_index())

    # Remove duplicates
    if features_df.index.has_duplicates:
        features_df = cast(pd.DataFrame, features_df[~features_df.index.duplicated(keep="first")].copy())

    return features_df


def _subsample_features(
    features_df: pd.DataFrame,
    config: MetaOptimizationConfig,
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


def _get_feature_columns(features_df: pd.DataFrame) -> List[str]:
    """Get list of feature columns."""
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label",
    ]
    return [c for c in features_df.columns if c not in non_feature_cols]


def _align_close_to_features(
    close: pd.Series,
    features_df: pd.DataFrame,
) -> pd.Series:
    """Align close prices to feature window."""
    close = close.loc[
        (close.index >= features_df.index[0]) &
        (close.index <= features_df.index[-1])
    ]
    if close.index.has_duplicates:
        close = cast(pd.Series, close[~close.index.duplicated(keep="first")].copy())
    return close


def _create_study(config: MetaOptimizationConfig) -> optuna.Study:
    """Create Optuna study with configured settings."""
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )


def _build_result(
    study: optuna.Study,
    config: MetaOptimizationConfig,
) -> MetaOptimizationResult:
    """Build optimization result from study."""
    best_trial = study.best_trial
    tb_keys = set(TRIPLE_BARRIER_SEARCH_SPACE.keys())

    best_tb = {k: v for k, v in best_trial.params.items() if k in tb_keys}
    best_model = {k: v for k, v in best_trial.params.items() if k not in tb_keys}

    return MetaOptimizationResult(
        primary_model_name=config.primary_model_name,
        meta_model_name=config.meta_model_name,
        best_params=best_model,
        best_triple_barrier_params=best_tb,
        best_score=best_trial.value if best_trial.value is not None else float("nan"),
        metric="mean_balanced_accuracy_f1_weighted",
        n_trials=len(study.trials),
    )


def _log_result(result: MetaOptimizationResult) -> None:
    """Log optimization results."""
    logger.info(f"Best score (balanced_acc + f1_weighted)/2: {result.best_score:.4f}")
    logger.info(f"Best TB params: {result.best_triple_barrier_params}")
    logger.info(f"Best meta model params: {result.best_params}")


def _generate_primary_side_predictions(config: MetaOptimizationConfig) -> pd.Series:
    """
    Generate primary model side predictions on its own feature set (train split).

    This avoids using ground-truth labels as side inputs for meta optimization.
    """
    from src.labelling.label_meta.utils import get_labeled_dataset_for_primary_model

    primary_df = cast(pd.DataFrame, get_labeled_dataset_for_primary_model(config.primary_model_name))

    if "split" in primary_df.columns:
        primary_df = cast(pd.DataFrame, primary_df[primary_df["split"] == TRAIN_SPLIT_LABEL].copy())

    if "datetime_close" in primary_df.columns:
        primary_df = cast(pd.DataFrame, primary_df.set_index("datetime_close"))
    primary_df = cast(pd.DataFrame, primary_df.sort_index())

    if primary_df.index.has_duplicates:
        primary_df = cast(pd.DataFrame, primary_df[~primary_df.index.duplicated(keep="first")])

    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label",
    ]
    feature_cols = [c for c in primary_df.columns if c not in non_feature_cols]
    X_primary = cast(pd.DataFrame, primary_df[feature_cols])

    primary_model = load_primary_model(config.primary_model_name)
    side_predictions = pd.Series(
        primary_model.predict(X_primary),
        index=X_primary.index,
        name="side",
    )

    return side_predictions.sort_index()


def _prepare_meta_data(
    config: MetaOptimizationConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Prepare features, close, volatility and side predictions."""
    # Load meta model's features (train split only)
    meta_features_df = _load_and_filter_features(config)
    meta_features_df = _subsample_features(meta_features_df, config)

    # Generate side predictions from the trained primary model (no ground-truth labels)
    side_predictions_primary = _generate_primary_side_predictions(config)

    # Align indices between meta features and primary side predictions
    common_idx = meta_features_df.index.intersection(side_predictions_primary.index)
    meta_features_df = meta_features_df.loc[common_idx]
    side_predictions = side_predictions_primary.loc[common_idx]

    # Filter to valid sides (+1/-1) and keep aligned features
    valid_sides_mask = side_predictions.isin([1, -1])
    meta_features_df = meta_features_df.loc[valid_sides_mask]
    side_predictions = side_predictions.loc[valid_sides_mask]

    # Extract feature columns for the meta model
    feature_cols = _get_feature_columns(meta_features_df)
    X_meta_features = cast(pd.DataFrame, meta_features_df[feature_cols])

    # Load market data
    bars = load_dollar_bars()
    close_prices = cast(pd.Series, bars["close"])
    close = _align_close_to_features(close_prices, meta_features_df)
    volatility = get_daily_volatility(close, span=config.vol_span)

    logger.info(f"Side distribution: {side_predictions.value_counts().to_dict()}")

    return X_meta_features, close, volatility, side_predictions


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================


def optimize_meta_model(
    primary_model_name: str,
    meta_model_name: str,
    config: MetaOptimizationConfig | None = None,
) -> MetaOptimizationResult:
    """Optimize a meta model that filters false positives."""
    if config is None:
        config = MetaOptimizationConfig(
            primary_model_name=primary_model_name,
            meta_model_name=meta_model_name,
        )

    logger.info(f"Starting META optimization: {primary_model_name} -> {meta_model_name}")

    meta_model_class = load_model_class(meta_model_name)
    meta_search_space = MODEL_REGISTRY[meta_model_name]["search_space"]

    # Load features and side labels (from primary model's labeled dataset)
    X_features, close, volatility, side = _prepare_meta_data(config)

    study = _create_study(config)
    objective = create_objective(
        config, X_features, close, volatility, side, meta_model_class, meta_search_space
    )
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout, show_progress_bar=True)

    result = _build_result(study, config)
    result.save(LABEL_META_OPTI_DIR / f"{primary_model_name}_{meta_model_name}_optimization.json")
    _log_result(result)
    return result


# =============================================================================
# CLI
# =============================================================================


def _select_primary_model() -> str | None:
    """Select a primary model interactively."""
    available_primary = get_available_primary_models()
    if not available_primary:
        print("\nNo trained primary models found!")
        print("Train primary model first:")
        print("  python -m src.labelling.label_primaire.train")
        return None

    print("\nPrimary models disponibles:")
    for i, name in enumerate(available_primary, 1):
        print(f"  {i}. {name}")

    while True:
        choice = input("\nChoisir primary model (numero): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_primary):
                return available_primary[idx]
        print("Choix invalide.")


def _select_meta_model() -> str:
    """Select a meta model interactively."""
    meta_models = list(MODEL_REGISTRY.keys())
    print("\nMeta models disponibles:")
    for i, name in enumerate(meta_models, 1):
        print(f"  {i}. {name}")

    while True:
        choice = input("\nChoisir meta model (numero): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(meta_models):
                return meta_models[idx]
        print("Choix invalide.")


def _print_result(result: MetaOptimizationResult) -> None:
    """Print optimization result summary."""
    print(f"\n{'='*60}")
    print("META OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Primary: {result.primary_model_name}")
    print(f"Meta: {result.meta_model_name}")
    print(f"F1 Score: {result.best_score:.4f}")
    print(f"\nTriple Barrier:")
    for k, v in result.best_triple_barrier_params.items():
        print(f"  {k}: {v}")
    print(f"\nMeta model params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")


def main() -> None:
    """Main entry point with interactive prompts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("\n" + "=" * 60)
    print("META-LABELING OPTIMIZATION")
    print("=" * 60)
    print("\nFilters false positives from primary model predictions.")

    primary_model_name = _select_primary_model()
    if primary_model_name is None:
        return
    meta_model_name = _select_meta_model()

    n_trials = int(input("Nombre de trials [50]: ").strip() or "50")
    n_splits = int(input("Nombre de splits CV [5]: ").strip() or "5")

    print(f"\n{'='*40}")
    print(f"Primary: {primary_model_name}, Meta: {meta_model_name}")
    print(f"Trials: {n_trials}, CV: {n_splits}")
    print(f"{'='*40}")

    if input("\nLancer? (O/n): ").strip().lower() == "n":
        return

    config = MetaOptimizationConfig(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        n_trials=n_trials,
        n_splits=n_splits,
    )
    result = optimize_meta_model(primary_model_name, meta_model_name, config)
    _print_result(result)


if __name__ == "__main__":
    main()
