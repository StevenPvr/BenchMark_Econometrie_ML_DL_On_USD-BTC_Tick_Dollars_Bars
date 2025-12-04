"""Hyperparameter optimization for meta-labeling models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_score

from src.config_logging import get_logger
from src.labelling.label_meta.utils import (
    MODEL_REGISTRY,
    MetaOptimizationConfig,
    MetaOptimizationResult,
    get_labeled_dataset_for_primary_model,
    load_model_class,
)
from src.path import LABEL_META_OPTI_DIR

logger = get_logger(__name__)

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)


def _optuna_logging_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Callback to log Optuna trial results."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if trial.state == optuna.trial.TrialState.COMPLETE:
        attrs = trial.user_attrs
        precision = trial.value or 0.0
        std_precision = attrs.get("std_precision", 0)

        is_best = study.best_trial and study.best_trial.number == trial.number
        best_precision = study.best_value if study.best_trial else precision

        print(f"\n{CYAN}{'─' * 60}{RESET}")
        print(f"{YELLOW}[Trial {trial.number}] COMPLETED{RESET}")
        print(f"{CYAN}{'─' * 60}{RESET}")

        print(f"  {BOLD}Precision:{RESET} {GREEN}{precision:.4f}{RESET} (±{std_precision:.4f})")

        if is_best:
            print(f"  {GREEN}{BOLD}★ NEW BEST ★{RESET}")
        else:
            gap = precision - best_precision
            print(f"  Best: {best_precision:.4f} (#{study.best_trial.number if study.best_trial else '?'}) | gap: {gap:+.4f}")

        params_str = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in trial.params.items()
        )
        print(f"  {BOLD}Params:{RESET} {params_str}")
        print(f"{CYAN}{'─' * 60}{RESET}", flush=True)

    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"{YELLOW}[Trial {trial.number}] PRUNED{RESET}")


class WalkForwardCV:
    """Walk-forward splitter for time series."""

    def __init__(self, n_splits: int = 5, min_train_size: int = 500, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame) -> List[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        fold_size = max(1, n_samples // self.n_splits)
        splits: List[tuple[np.ndarray, np.ndarray]] = []
        embargo_size = int(fold_size * self.embargo_pct)
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = n_samples if fold == self.n_splits - 1 else (fold + 1) * fold_size
            train_idx = np.arange(0, max(0, start - embargo_size))
            val_idx = np.arange(start, end)
            if len(train_idx) >= self.min_train_size and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits


def _prepare_meta_dataset(primary_model_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare dataset for meta-labeling.

    Uses only TRAIN split with valid OOF predictions.
    Filters to directional predictions (!=0) with valid OOF coverage.
    Creates binary meta-label: 1 if primary prediction was correct, 0 otherwise.
    """
    dataset = get_labeled_dataset_for_primary_model(primary_model_name)

    # Filter: TRAIN split only (OOF predictions), directional, with valid coverage
    train_mask = dataset["split"] == "train"
    dataset = dataset.loc[train_mask].copy()
    logger.info("Using TRAIN split only: %d samples", len(dataset))

    mask = (dataset["prediction"] != 0) & (dataset["coverage"] == 1)
    filtered = dataset.loc[mask].copy()
    logger.info("Meta dataset: %d/%d samples after filtering (prediction!=0, coverage==1)",
                len(filtered), len(dataset))

    # Create meta label: 1 if primary was correct, 0 otherwise
    meta_label = (filtered["prediction"] == filtered["label"]).astype(int)

    # Features: all columns except meta-specific ones
    non_feature_cols = {"label", "prediction", "coverage", "split", "t1", "bar_id",
                        "datetime_close", "datetime_open", "timestamp_open", "timestamp_close",
                        "log_return", "threshold_used"}
    feature_cols = [c for c in filtered.columns if c not in non_feature_cols]
    features = filtered[feature_cols].copy()

    # Handle missing values
    for col in features.columns:
        if features[col].isna().any():
            median = features[col].median()
            features[col] = features[col].fillna(0.0 if pd.isna(median) else median)

    logger.info("Meta label distribution: correct=%d (%.1f%%), incorrect=%d (%.1f%%)",
                meta_label.sum(), 100 * meta_label.mean(),
                len(meta_label) - meta_label.sum(), 100 * (1 - meta_label.mean()))

    return features, meta_label


def _sample_model_params(
    trial: optuna.trial.Trial,
    search_space: Dict[str, Any],
    config: MetaOptimizationConfig,
) -> Dict[str, Any]:
    """Sample hyperparameters from search space."""
    params: Dict[str, Any] = {"random_state": config.random_state}
    for name, (kind, values) in search_space.items():
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, values)
        elif kind == "float":
            log_scale = len(values) > 2 and values[2] == "log"
            params[name] = trial.suggest_float(name, float(values[0]), float(values[1]), log=log_scale)
        else:
            params[name] = trial.suggest_int(name, int(values[0]), int(values[1]))
    return params


def _evaluate_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_cls: Callable[..., Any],
    params: Dict[str, Any],
) -> float:
    """Evaluate a single CV fold, return precision.

    Meta-model goal: maximize precision (filter bad trades from primary).
    De Prado approach: primary maximizes recall, meta maximizes precision.
    """
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = model_cls(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Precision: among predictions of "correct", how many are actually correct
    precision = float(precision_score(y_val, y_pred, average="binary", zero_division=0))
    return precision


def create_objective(
    config: MetaOptimizationConfig,
    features: pd.DataFrame,
    meta_label: pd.Series,
    model_cls: Callable[..., Any],
    search_space: Dict[str, Any],
) -> Callable[[optuna.trial.Trial], float]:
    """Create Optuna objective function for meta-model optimization."""

    cv = WalkForwardCV(n_splits=config.n_splits, min_train_size=config.min_train_size)
    splits = cv.split(features)

    if len(splits) == 0:
        raise ValueError("No valid CV splits generated")

    def objective(trial: optuna.trial.Trial) -> float:
        params = _sample_model_params(trial, search_space, config)

        precision_scores: List[float] = []

        for train_idx, val_idx in splits:
            try:
                precision = _evaluate_fold(features, meta_label, train_idx, val_idx, model_cls, params)
                precision_scores.append(precision)
            except Exception as e:
                logger.warning("Fold failed: %s", e)
                continue

        if not precision_scores:
            return float("-inf")

        mean_precision = float(np.mean(precision_scores))
        trial.set_user_attr("std_precision", float(np.std(precision_scores)))

        return mean_precision

    return objective


def _create_study(primary_model_name: str, meta_model_name: str, fresh: bool = True) -> optuna.Study:
    """Create or load Optuna study."""
    LABEL_META_OPTI_DIR.mkdir(parents=True, exist_ok=True)
    study_name = f"{primary_model_name}_{meta_model_name}"
    storage = LABEL_META_OPTI_DIR / f"{study_name}_study.db"

    # Delete existing study if fresh start requested
    if fresh and storage.exists():
        storage.unlink()
        logger.info("Deleted existing study: %s", storage)

    return optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{storage}",
        load_if_exists=not fresh,
    )


def optimize_model(config: MetaOptimizationConfig) -> MetaOptimizationResult:
    """Run hyperparameter optimization for meta-labeling model."""
    logger.info("Preparing meta dataset for primary model: %s", config.primary_model_name)
    features, meta_label = _prepare_meta_dataset(config.primary_model_name)

    model_cls = load_model_class(config.meta_model_name)
    search_space = MODEL_REGISTRY[config.meta_model_name]["search_space"]

    objective = create_objective(config, features, meta_label, model_cls, search_space)

    study = _create_study(config.primary_model_name, config.meta_model_name)
    logger.info("Starting optimization: %d trials", config.n_trials)
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        callbacks=[_optuna_logging_callback],
    )

    result = MetaOptimizationResult(
        primary_model_name=config.primary_model_name,
        meta_model_name=config.meta_model_name,
        best_params=study.best_params,
        best_triple_barrier_params={},  # Not used anymore
        best_score=float(study.best_value),
        metric="precision",
        n_trials=len(study.trials),
    )
    output_path = LABEL_META_OPTI_DIR / f"{config.primary_model_name}_{config.meta_model_name}_optimization.json"
    result.save(output_path)
    logger.info("Best Precision: %.4f, saved to %s", result.best_score, output_path)

    return result


__all__ = [
    "WalkForwardCV",
    "create_objective",
    "optimize_model",
]
