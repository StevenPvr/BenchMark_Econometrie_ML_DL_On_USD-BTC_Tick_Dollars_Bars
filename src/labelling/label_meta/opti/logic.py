"""Hyperparameter optimization helpers for meta-labeling."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from src.config_logging import get_logger
from src.labelling.label_meta.utils import (
    MODEL_REGISTRY,
    MetaOptimizationConfig,
    MetaOptimizationResult,
    TRIPLE_BARRIER_SEARCH_SPACE,
    compute_meta_label,
    compute_side_adjusted_barriers,
    find_first_touch_time,
    get_barrier_touches_for_side,
    get_daily_volatility,
    get_dataset_for_model,
    get_labeled_dataset_for_primary_model,
    load_dollar_bars,
    load_model_class,
    load_primary_model,
    set_vertical_barriers_meta,
)
from src.path import LABEL_META_OPTI_DIR

logger = get_logger(__name__)


class WalkForwardCV:
    """Walk-forward splitter for time series."""

    def __init__(self, n_splits: int = 5, min_train_size: int = 500, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, events: pd.DataFrame) -> List[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        fold_size = max(1, n_samples // self.n_splits)
        splits: List[tuple[np.ndarray, np.ndarray]] = []
        embargo_size = int(fold_size * self.embargo_pct)
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = n_samples if fold == self.n_splits - 1 else (fold + 1) * fold_size
            train_idx = np.arange(0, start)
            val_idx = np.arange(start, end)
            processed = self._process_split(train_idx, val_idx, X, events, embargo_size)
            if processed is not None:
                splits.append(processed)
        return splits

    def _process_split(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        events: pd.DataFrame,
        embargo_size: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(train_idx) < self.min_train_size or len(val_idx) == 0:
            return None
        if embargo_size > 0 and len(train_idx) > embargo_size:
            train_idx = train_idx[:-embargo_size]
        train_idx = self._apply_purging(train_idx, val_idx, X, events)
        if len(train_idx) < self.min_train_size:
            return None
        return train_idx, val_idx

    def _apply_purging(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        events: pd.DataFrame,
    ) -> np.ndarray:
        if "t1" not in events.columns or len(train_idx) == 0 or len(val_idx) == 0:
            return train_idx
        val_start = X.index[val_idx[0]]
        t1_series = events.reindex(X.index).iloc[train_idx]["t1"]
        mask = ~(t1_series.notna() & (t1_series >= val_start))
        return train_idx[mask.to_numpy()]


def _resolve_n_jobs(requested: int | None) -> int:
    import multiprocessing

    max_cores = multiprocessing.cpu_count()
    if requested is None or requested <= 0:
        return max_cores
    return min(requested, max_cores)


def _split_t_events(t_events: pd.DatetimeIndex, n_splits: int) -> List[pd.DatetimeIndex]:
    return [pd.DatetimeIndex(chunk) for chunk in np.array_split(t_events, n_splits) if len(chunk) > 0]


def _filter_valid_events_meta(
    t_events: pd.DatetimeIndex,
    volatility: pd.Series,
    side: pd.Series,
) -> pd.DatetimeIndex:
    valid = [
        t for t in t_events
        if t in volatility.index and t in side.index and not pd.isna(volatility.loc[t])
    ]
    return pd.DatetimeIndex(valid)


def _build_events_dataframe_meta(
    t_events: pd.DatetimeIndex,
    volatility: pd.Series,
    side: pd.Series,
) -> pd.DataFrame:
    events = pd.DataFrame(index=t_events)
    events["trgt"] = volatility.reindex(t_events)
    events["side"] = side.reindex(t_events)
    return events


def _get_path_returns(close: pd.Series, t0: Any, t1: Any) -> pd.Series | None:
    if t0 not in close.index or t1 not in close.index or t0 == t1:
        return None
    path = close.loc[t0:t1]
    if len(path) < 2:
        return None
    base = float(path.iloc[0])
    return cast(pd.Series, (path / base) - 1.0)


def get_events_meta(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_mult: float,
    sl_mult: float,
    trgt: pd.Series,
    max_holding: int,
    side: pd.Series,
    min_return: float = 0.0,
) -> pd.DataFrame:
    """Generate meta-labeling events."""
    if len(t_events) == 0:
        return pd.DataFrame()
    valid_events = _filter_valid_events_meta(t_events, trgt, side)
    if len(valid_events) == 0:
        return pd.DataFrame()
    events = _build_events_dataframe_meta(valid_events, trgt, side)
    events = set_vertical_barriers_meta(events, close.index, max_holding)
    events = compute_side_adjusted_barriers(events, pt_mult, sl_mult)
    return _compute_meta_labels(events, close, min_return)


def _compute_meta_labels(events: pd.DataFrame, close: pd.Series, min_return: float) -> pd.DataFrame:
    returns: List[float] = []
    bins: List[int] = []
    for t0, row in events.iterrows():
        path = _get_path_returns(close, t0, row["t1"])
        if path is None:
            returns.append(np.nan)
            bins.append(0)
            continue
        pt_touches, sl_touches = get_barrier_touches_for_side(path, int(row["side"]), float(row["pt"]), float(row["sl"]))
        touch_time = find_first_touch_time(pt_touches, sl_touches)
        exit_time = touch_time if touch_time is not None else row["t1"]
        ret = float(path.loc[exit_time]) if exit_time in path.index else float(path.iloc[-1])
        returns.append(ret)
        bins.append(compute_meta_label(ret, int(row["side"])) if abs(ret) >= min_return else 0)
    events = events.copy()
    events["ret"] = returns
    events["bin"] = bins
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Compute returns and bins for provided events."""
    if events.empty:
        return events
    return _compute_meta_labels(events, close, min_return=0.0)


def _sample_barrier_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, (kind, values) in TRIPLE_BARRIER_SEARCH_SPACE.items():
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, values)  # type: ignore[arg-type]
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(values[0]), float(values[1]))
        else:
            params[name] = trial.suggest_int(name, int(values[0]), int(values[1]))
    return params


def _validate_meta_events(events: pd.DataFrame, config: MetaOptimizationConfig) -> tuple[bool, str]:
    if events.empty:
        return False, "events empty"
    if len(events) < config.min_train_size:
        return False, "not enough events"
    if "bin" in events.columns and events["bin"].nunique() <= 1:
        return False, "only one class"
    return True, "OK"


def _align_features_events_meta(
    features: pd.DataFrame,
    events: pd.DataFrame,
    config: MetaOptimizationConfig,
) -> tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None]:
    aligned = pd.DataFrame(events.reindex(features.index).dropna(subset=["bin"]))
    if len(aligned) < config.min_train_size:
        return None, None, None
    y = pd.Series(aligned["bin"].astype(int))
    X = pd.DataFrame(features.loc[aligned.index])
    return X, y, aligned


def _sample_model_params(
    trial: optuna.trial.Trial,
    search_space: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"random_state": config.random_state}
    for name, (kind, values) in search_space.items():
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, values)  # type: ignore[arg-type]
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(values[0]), float(values[1]))
        else:
            params[name] = trial.suggest_int(name, int(values[0]), int(values[1]))
    return params


def _evaluate_fold_meta(
    features: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_cls: Callable[..., Any],
    params: Dict[str, Any],
    fold_idx: int = 0,
) -> tuple[float | None, float | None, float | None]:
    X_train = features.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = features.iloc[val_idx]
    y_val = y.iloc[val_idx]

    if y_val.nunique() <= 1 or y_train.nunique() <= 1:
        return None, None, None

    try:
        model = model_cls(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mcc_raw = float(matthews_corrcoef(y_val, preds))
        mcc = max(0.0, mcc_raw)
        bal_acc = float(balanced_accuracy_score(y_val, preds))
        f1 = float(f1_score(y_val, preds, average="binary"))
        return mcc, bal_acc, f1
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Fold %s failed: %s", fold_idx, exc)
        return None, None, None


def _run_cv_scoring_meta(
    features: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    model_cls: Callable[..., Any],
    params: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> tuple[float, float, float]:
    cv = WalkForwardCV(config.n_splits, config.min_train_size)
    splits = cv.split(features, events)
    mcc_scores: List[float] = []
    bal_acc_scores: List[float] = []
    f1_scores: List[float] = []
    for idx, (train_idx, val_idx) in enumerate(splits):
        mcc, bal_acc, f1_val = _evaluate_fold_meta(features, y, train_idx, val_idx, model_cls, params, idx)
        if mcc is not None and bal_acc is not None and f1_val is not None:
            mcc_scores.append(mcc)
            bal_acc_scores.append(bal_acc)
            f1_scores.append(f1_val)
    if not mcc_scores:
        raise optuna.TrialPruned("No valid CV folds")
    return float(np.mean(mcc_scores)), float(np.mean(bal_acc_scores)), float(np.mean(f1_scores))


def _process_chunk_events_meta(
    t_events_chunk: pd.DatetimeIndex,
    close: pd.Series,
    volatility: pd.Series,
    side: pd.Series,
    tb_params: Dict[str, Any],
) -> pd.DataFrame:
    return get_events_meta(
        close=close,
        t_events=t_events_chunk,
        pt_mult=float(tb_params["pt_mult"]),
        sl_mult=float(tb_params["sl_mult"]),
        trgt=volatility,
        max_holding=int(tb_params["max_holding"]),
        side=side,
    )


def _generate_trial_events_meta(
    features: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    side: pd.Series,
    tb_params: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> pd.DataFrame:
    t_events = pd.DatetimeIndex(features.index)
    return _process_chunk_events_meta(t_events, close, volatility, side, tb_params)


def _subsample_features(features: pd.DataFrame, config: MetaOptimizationConfig) -> pd.DataFrame:
    if config.data_fraction >= 1.0:
        return features
    return features.sample(frac=config.data_fraction, random_state=config.random_state).sort_index()


def create_objective(
    config: MetaOptimizationConfig,
    features: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    side: pd.Series,
    model_cls: Callable[..., Any],
    search_space: Dict[str, Any],
) -> Callable[[optuna.trial.Trial], float]:
    """Create Optuna objective for meta-label optimization."""
    cv = WalkForwardCV(config.n_splits, config.min_train_size)
    events_cache: Dict[tuple[Any, ...], pd.DataFrame] = {}

    def objective(trial: optuna.trial.Trial) -> float:
        tb_params = _sample_barrier_params(trial)
        cache_key = tuple(tb_params.values())
        events = events_cache.get(cache_key)
        if events is None:
            events = _generate_trial_events_meta(features, close, volatility, side, tb_params, config)
            events_cache[cache_key] = events

        valid, _ = _validate_meta_events(events, config)
        if not valid:
            return 0.0

        X, y, aligned_events = _align_features_events_meta(features, events, config)
        if X is None or y is None or aligned_events is None:
            return 0.0

        model_params = _sample_model_params(trial, search_space, config)
        mean_mcc, bal_acc, f1w = _run_cv_scoring_meta(X, y, aligned_events, model_cls, model_params, config)
        trial.set_user_attr("balanced_accuracy", bal_acc)
        trial.set_user_attr("f1_weighted", f1w)
        return mean_mcc

    return objective


# -----------------------------------------------------------------------------
# Convenience helpers (not fully exercised in tests, kept lightweight)
# -----------------------------------------------------------------------------


def _prepare_optimization_data(primary_model_name: str, data_fraction: float) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    dummy_config = MetaOptimizationConfig(primary_model_name, "meta", data_fraction=data_fraction)
    features = _subsample_features(get_dataset_for_model("logistic"), dummy_config)
    primary_dataset = get_labeled_dataset_for_primary_model(primary_model_name)
    if "prediction" in primary_dataset.columns:
        features["prediction"] = primary_dataset["prediction"]
    bars = load_dollar_bars()
    close = pd.Series(bars["close"])
    volatility = get_daily_volatility(close)
    side = pd.Series(primary_dataset.get("side", pd.Series(index=primary_dataset.index, data=1)))
    return features, close, volatility, side


def _create_study(meta_model_name: str) -> optuna.Study:
    LABEL_META_OPTI_DIR.mkdir(parents=True, exist_ok=True)
    storage = LABEL_META_OPTI_DIR / f"{meta_model_name}_study.db"
    return optuna.create_study(direction="maximize", study_name=meta_model_name, storage=f"sqlite:///{storage}", load_if_exists=True)


def optimize_model(config: MetaOptimizationConfig) -> MetaOptimizationResult:
    model_cls = load_model_class(config.meta_model_name)
    features, close, volatility, side = _prepare_optimization_data(config.primary_model_name, config.data_fraction)
    objective = create_objective(
        config,
        features,
        close,
        volatility,
        side,
        model_cls,
        MODEL_REGISTRY[config.meta_model_name]["search_space"],
    )
    study = _create_study(config.meta_model_name)
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout)
    result = MetaOptimizationResult(
        primary_model_name=config.primary_model_name,
        meta_model_name=config.meta_model_name,
        best_params=study.best_params,
        best_triple_barrier_params=study.best_trial.user_attrs.get("tb_params", {}),
        best_score=float(study.best_value),
        metric="mcc",
        n_trials=len(study.trials),
    )
    result.save(LABEL_META_OPTI_DIR / f"{config.primary_model_name}_{config.meta_model_name}_optimization.json")
    return result


__all__ = [
    "MODEL_REGISTRY",
    "TRIPLE_BARRIER_SEARCH_SPACE",
    "WalkForwardCV",
    "create_objective",
    "get_bins",
    "get_events_meta",
    "optimize_model",
    "_resolve_n_jobs",
    "_split_t_events",
    "_filter_valid_events_meta",
    "_build_events_dataframe_meta",
    "_get_path_returns",
    "_sample_barrier_params",
    "_validate_meta_events",
    "_align_features_events_meta",
    "_sample_model_params",
    "_evaluate_fold_meta",
    "_run_cv_scoring_meta",
    "_process_chunk_events_meta",
    "_generate_trial_events_meta",
    "_subsample_features",
]
