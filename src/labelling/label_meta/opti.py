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
import warnings
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from src.constants import TRAIN_SPLIT_LABEL
from src.model.base import BaseModel
from src.path import LABEL_META_OPTI_DIR

from src.labelling.label_meta.utils import (
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
        pt = float(row["pt"]) if pd.notna(row["pt"]) else 0.0
        sl = float(row["sl"]) if pd.notna(row["sl"]) else 0.0

        path_ret = _get_path_returns(close, loc, t1)
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
    events = events[events["t1"].notna()]
    if len(events) == 0:
        return pd.DataFrame()

    # Compute side-adjusted barriers
    events = compute_side_adjusted_barriers(events, pt_mult, sl_mult)

    # Apply barriers and find first touch
    events = _apply_meta_barriers(close, events)

    return events[["t1", "trgt", "side", "pt", "sl"]]


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
        t1 = row["t1"]
        side = int(row["side"])

        if t1 is None or pd.isna(t1):
            continue

        try:
            price_t0 = close.loc[loc]
            price_t1 = close.loc[t1]
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


def _generate_trial_events_meta(
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    side_predictions: pd.Series,
    tb_params: Dict[str, Any],
) -> pd.DataFrame:
    """Generate meta events with given barrier parameters."""
    events = get_events_meta(
        close=close,
        t_events=pd.DatetimeIndex(features_df.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility,
        max_holding=tb_params["max_holding"],
        side=side_predictions,
    )
    if not events.empty:
        events = get_bins(events, close)
    return events


def _validate_meta_events(
    events: pd.DataFrame,
    config: MetaOptimizationConfig,
) -> bool:
    """Validate that events are suitable for training."""
    if events.empty or len(events) < config.min_train_size:
        return False
    if len(events["bin"].value_counts()) < 2:
        return False
    return True


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
) -> float | None:
    """Evaluate a single CV fold for meta-labeling."""
    try:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if len(y_train.unique()) < 2:
            return None

        # Train meta model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # F1 score for binary classification
        return f1_score(y_val, y_pred, zero_division=0.0)

    except Exception:
        return None


def _run_cv_scoring_meta(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> float:
    """Run walk-forward CV and return mean score."""
    cv = WalkForwardCV(
        n_splits=config.n_splits,
        min_train_size=config.min_train_size,
    )

    cv_scores = []
    for train_idx, val_idx in cv.split(X, events):
        score = _evaluate_fold_meta(X, y, train_idx, val_idx, model_class, model_params)
        if score is not None:
            cv_scores.append(score)

    if len(cv_scores) == 0:
        return float("-inf")
    return float(np.mean(cv_scores))


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
            features_df, close, volatility, side_predictions, tb_params
        )

        # Validate events
        if not _validate_meta_events(events, config):
            return float("-inf")

        # Prepare data
        X, y, events_aligned = _align_features_events_meta(features_df, events, config)
        if X is None:
            return float("-inf")

        # Sample model params and run CV
        model_params = _sample_model_params(trial, meta_search_space, config)
        return _run_cv_scoring_meta(
            X, y, events_aligned, meta_model_class, model_params, config
        )

    return objective


# =============================================================================
# OPTIMIZATION HELPERS
# =============================================================================


def _load_and_filter_features(
    model_name: str,
    config: MetaOptimizationConfig,
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
        close = close[~close.index.duplicated(keep="first")]
    return close


def _generate_side_predictions(
    primary_model: BaseModel,
    X_features: pd.DataFrame,
) -> pd.Series:
    """Generate primary model predictions (side)."""
    side_predictions = pd.Series(
        primary_model.predict(X_features),
        index=X_features.index,
        name="side",
    )
    # Filter to valid sides (+1 or -1)
    return side_predictions.loc[side_predictions.isin([1, -1])]


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
        metric="f1_score",
        n_trials=len(study.trials),
    )


def _log_result(result: MetaOptimizationResult) -> None:
    """Log optimization results."""
    logger.info(f"Best F1 score: {result.best_score:.4f}")
    logger.info(f"Best TB params: {result.best_triple_barrier_params}")
    logger.info(f"Best meta model params: {result.best_params}")


def _prepare_meta_data(
    config: MetaOptimizationConfig,
    primary_model: BaseModel,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Prepare features, close, volatility and side predictions."""
    features_df = _load_and_filter_features(config.meta_model_name, config)
    features_df = _subsample_features(features_df, config)
    X_features = features_df[_get_feature_columns(features_df)]

    bars = load_dollar_bars()
    close = _align_close_to_features(bars["close"], features_df)
    volatility = get_daily_volatility(close, span=config.vol_span)

    side_predictions = _generate_side_predictions(primary_model, X_features)
    X_features = X_features.loc[side_predictions.index]
    logger.info(f"Side distribution: {side_predictions.value_counts().to_dict()}")

    return X_features, close, volatility, side_predictions


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

    primary_model = load_primary_model(primary_model_name)
    meta_model_class = load_model_class(meta_model_name)
    meta_search_space = MODEL_REGISTRY[meta_model_name]["search_space"]

    X_features, close, volatility, side = _prepare_meta_data(config, primary_model)

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
