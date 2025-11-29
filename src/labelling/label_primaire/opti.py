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
import warnings
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


def _validate_events(events: pd.DataFrame, config: OptimizationConfig) -> bool:
    """Validate that events are suitable for training."""
    if events.empty or len(events) < config.min_train_size:
        return False
    if len(events["label"].value_counts()) < 2:
        return False
    return True


def _align_features_events(
    features_df: pd.DataFrame,
    events: pd.DataFrame,
    config: OptimizationConfig,
) -> Tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None]:
    """Align features with events on common index."""
    common_idx = events.index.intersection(features_df.index)
    if len(common_idx) < config.min_train_size:
        return None, None, None

    X = features_df.loc[common_idx]
    y = events.loc[common_idx, "label"]
    events_aligned = events.loc[common_idx]
    return X, y, events_aligned


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
) -> float | None:
    """Evaluate a single CV fold."""
    try:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if len(y_train.unique()) < 2:
            return None

        # Compute class weights
        classes = np.unique(y_train)
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        weight_map = dict(zip(classes, cw))

        # Train and predict
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Weighted MCC
        sample_weights = np.array([weight_map.get(c, 1.0) for c in y_val])
        return matthews_corrcoef(y_val, y_pred, sample_weight=sample_weights)

    except Exception:
        return None


def _run_cv_scoring(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    config: OptimizationConfig,
) -> float:
    """Run walk-forward CV and return mean score."""
    cv = WalkForwardCV(
        n_splits=config.n_splits,
        min_train_size=config.min_train_size,
        embargo_pct=0.01,
    )

    cv_scores = []
    for train_idx, val_idx in cv.split(X, events):
        score = _evaluate_fold(X, y, train_idx, val_idx, model_class, model_params)
        if score is not None:
            cv_scores.append(score)

    if len(cv_scores) == 0:
        return float("-inf")
    return float(np.mean(cv_scores))


def create_objective(
    config: OptimizationConfig,
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    model_class: Type[BaseModel],
    model_search_space: Dict[str, Any],
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective for joint optimization."""

    def objective(trial: optuna.Trial) -> float:
        # Sample and generate events
        tb_params = _sample_barrier_params(trial)
        events = _generate_trial_events(features_df, close, volatility, tb_params)

        # Validate events
        if not _validate_events(events, config):
            return float("-inf")

        # Prepare data
        X, y, events_aligned = _align_features_events(features_df, events, config)
        if X is None:
            return float("-inf")

        # Sample model params and run CV
        model_params = _sample_model_params(trial, model_search_space, config)
        if X is None or y is None or events_aligned is None:
            return float("-inf")
        return _run_cv_scoring(X, y, events_aligned, model_class, model_params, config)

    return objective


# =============================================================================
# OPTIMIZATION HELPERS
# =============================================================================


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


def select_model_interactive() -> str:
    """Interactive model selection."""
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

    while True:
        try:
            choice = input("\nChoisir (numero ou nom): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx]
            elif choice.lower() in models:
                return choice.lower()
            print("Choix invalide.")
        except KeyboardInterrupt:
            print("\nAnnule.")
            exit(0)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model_name = select_model_interactive()
    print(f"\nModele: {model_name}")

    n_trials = int(input("Nombre de trials [50]: ").strip() or "50")
    n_splits = int(input("Nombre de splits CV [5]: ").strip() or "5")

    print(f"\n{'='*40}")
    print(f"Modele: {model_name}")
    print(f"Trials: {n_trials}, CV: {n_splits}")
    print(f"{'='*40}")

    if input("\nLancer? (O/n): ").strip().lower() == "n":
        print("Annule.")
        return

    config = OptimizationConfig(
        model_name=model_name,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    result = optimize_model(model_name, config)

    print(f"\n{'='*60}")
    print(f"OPTIMISATION TERMINEE: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Score (MCC): {result.best_score:.4f}")
    print(f"\nTriple Barrier:")
    for k, v in result.best_triple_barrier_params.items():
        print(f"  {k}: {v}")
    print(f"\nModel params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
