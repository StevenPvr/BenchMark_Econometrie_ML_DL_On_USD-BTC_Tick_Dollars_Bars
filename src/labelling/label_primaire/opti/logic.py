"""Hyperparameter optimization utilities for primary labeling models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, cast

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE, TRAIN_SPLIT_LABEL

# =============================================================================
# OPTUNA NATIVE LOGGING
# =============================================================================
# Configure Optuna logging to show trial progress
optuna.logging.set_verbosity(optuna.logging.INFO)


def _optuna_logging_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Callback to log Optuna trial results with detailed metrics and best score tracking."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if trial.state == optuna.trial.TrialState.COMPLETE:
        # Extract detailed metrics from user_attrs
        attrs = trial.user_attrs
        f1_per_class = attrs.get("f1_per_class", {})
        f1_macro = attrs.get("mean_f1_macro", 0)
        sign_err = attrs.get("mean_sign_error_rate", 0)
        mcc = attrs.get("mean_mcc", 0)
        folds = f"{attrs.get('n_valid_folds', '?')}/{attrs.get('n_total_folds', '?')}"
        composite = trial.value or 0.0

        # Check if this is the new best
        is_best = study.best_trial and study.best_trial.number == trial.number
        best_composite = study.best_value if study.best_trial else composite

        # Build per-class F1 string
        f1_class_str = " | ".join(
            f"F1[{k.split('_')[-1]}]={v:.3f}"
            for k, v in sorted(f1_per_class.items())
        ) or "no F1 data"

        # Compute sign penalty contribution
        sign_penalty_contrib = SIGN_ERROR_PENALTY * sign_err

        # Log trial result with clear structure
        print(f"\n{CYAN}{'─' * 70}{RESET}")
        print(f"{YELLOW}[Trial {trial.number}] COMPLETED{RESET}")
        print(f"{CYAN}{'─' * 70}{RESET}")

        # Composite score breakdown
        print(f"  {BOLD}Composite Score:{RESET} {GREEN}{composite:.4f}{RESET}")
        print(f"    ├─ F1 macro:       {f1_macro:.4f}")
        print(f"    └─ Sign penalty:  -{sign_penalty_contrib:.4f} (err={sign_err*100:.1f}% × {SIGN_ERROR_PENALTY})")

        # Per-class F1
        print(f"  {BOLD}F1 per class:{RESET} {f1_class_str}")

        # Additional metrics
        print(f"  {BOLD}Other metrics:{RESET} MCC={mcc:.4f} | folds={folds}")

        # Best score tracking
        if is_best:
            print(f"  {GREEN}{BOLD}★ NEW BEST SCORE ★{RESET}")
        else:
            improvement_needed = composite - best_composite
            print(f"  Best so far: {best_composite:.4f} (trial #{study.best_trial.number if study.best_trial else '?'}) | gap: {improvement_needed:+.4f}")

        # Show params (collapsed format for readability)
        params_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in trial.params.items())
        print(f"  {BOLD}Params:{RESET} {params_str}")

        print(f"{CYAN}{'─' * 70}{RESET}", flush=True)

    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"{YELLOW}[Trial {trial.number}] PRUNED{RESET}")

from src.labelling.label_primaire.utils import (
    CLASS_WEIGHT_SUPPORTED_MODELS,
    FOCAL_LOSS_SEARCH_SPACE,
    FOCAL_LOSS_SUPPORTED_MODELS,
    MODEL_REGISTRY,
    OptimizationConfig,
    OptimizationResult,
    get_dataset_for_model,
    load_model_class,
)
from src.path import LABEL_PRIMAIRE_OPTI_DIR

logger = get_logger(__name__)

# Sign error penalty weight for composite score
SIGN_ERROR_PENALTY = 0.3


# =============================================================================
# CROSS-VALIDATION
# =============================================================================


class WalkForwardCV:
    """Simple walk-forward splitter with optional purging and embargo."""

    def __init__(self, n_splits: int = 5, min_train_size: int = 500, embargo_pct: float = 0.01) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, events: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        fold_size = max(1, n_samples // self.n_splits)
        splits: List[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(self.n_splits):
            start = fold * fold_size
            end = n_samples if fold == self.n_splits - 1 else (fold + 1) * fold_size
            val_idx = np.arange(start, end)
            train_idx = np.arange(0, start)
            if self.embargo_pct > 0:
                embargo = int(len(val_idx) * self.embargo_pct)
                train_idx = train_idx[:-embargo] if embargo < len(train_idx) else np.array([], dtype=int)
            train_idx = self._apply_purging(train_idx, val_idx, X, events)
            if len(train_idx) >= self.min_train_size and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

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
        t1 = events.reindex(X.index).iloc[train_idx]["t1"]
        mask = ~(t1.notna() & (t1 >= val_start))
        return train_idx[mask.to_numpy()]


# =============================================================================
# SCORING METRICS
# =============================================================================


def _compute_sign_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute rate of sign errors (predicting opposite direction).

    Sign error occurs when:
    - True = -1 and Pred = +1 (or vice versa)

    This is worse than predicting neutral (0) when wrong.
    """
    # Only consider non-neutral predictions and non-neutral true labels
    mask = (y_true != 0) & (y_pred != 0)
    if not mask.any():
        return 0.0
    # Sign error = opposite signs
    sign_errors = (np.sign(y_true[mask]) != np.sign(y_pred[mask])).sum()
    return float(sign_errors / mask.sum())


def _compute_composite_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sign_penalty: float = SIGN_ERROR_PENALTY,
) -> Tuple[float, Dict[str, float]]:
    """Compute composite score: weighted F1 per class with sign error penalty.

    Returns:
        composite_score: The final score to optimize
        details: Dictionary with per-class F1, sign_error_rate, etc.
    """
    # F1 per class
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_per_class = {}
    for cls in sorted(classes):
        cls_mask_true = y_true == cls
        cls_mask_pred = y_pred == cls
        if cls_mask_true.sum() > 0:
            # Binary F1 for this class
            tp = (cls_mask_true & cls_mask_pred).sum()
            fp = (~cls_mask_true & cls_mask_pred).sum()
            fn = (cls_mask_true & ~cls_mask_pred).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_per_class[int(cls)] = float(f1)

    # Weighted F1 (macro average)
    f1_macro = float(np.mean(list(f1_per_class.values()))) if f1_per_class else 0.0

    # Sign error rate
    sign_error_rate = _compute_sign_error_rate(y_true, y_pred)

    # Composite score: F1 macro - penalty * sign_error_rate
    composite = f1_macro - sign_penalty * sign_error_rate

    details = {
        "f1_macro": f1_macro,
        "sign_error_rate": sign_error_rate,
        **{f"f1_class_{k}": v for k, v in f1_per_class.items()},
    }

    return composite, details


# =============================================================================
# DATA PREPARATION
# =============================================================================


def _align_features_events(
    features: pd.DataFrame,
    events: pd.DataFrame,
    config: OptimizationConfig,
) -> Tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, str]:
    """Align feature matrix with event labels."""
    aligned = events.reindex(features.index).dropna(subset=["label"])
    if len(aligned) < config.min_train_size:
        return None, None, None, "not enough aligned samples"
    y = aligned["label"].astype(int)
    X = features.loc[aligned.index]
    return X, cast(pd.Series, y), aligned, "OK"


def _compute_class_weight(y: pd.Series) -> Dict[int, float]:
    """Compute simple balanced class weights."""
    counts = y.value_counts()
    total = len(y)
    return {int(cast(Any, cls)): total / (len(counts) * count) for cls, count in counts.items()}


def _evaluate_fold(
    features: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_cls: Callable[..., Any],
    params: Dict[str, Any],
    model_name: str,
    fold_idx: int,
    use_class_weights: bool = False,
) -> Tuple[float | None, Dict[str, float] | None, str]:
    """Train and evaluate a single CV fold.

    Returns:
        composite_score: The composite F1 score with sign penalty
        details: Per-class F1 scores and sign error rate
        status: "OK", "SKIP", or "FAILED"
    """
    X_train = features.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = features.iloc[val_idx]
    y_val = y.iloc[val_idx]

    if y_val.nunique() <= 1:
        return None, None, "SKIP - only 1 class in validation"

    fit_params = params.copy()
    if use_class_weights and model_name in CLASS_WEIGHT_SUPPORTED_MODELS:
        fit_params["class_weight"] = _compute_class_weight(y_train)

    try:
        model = model_cls(**fit_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Compute composite score with sign penalty
        composite, details = _compute_composite_score(y_val.to_numpy(), y_pred)

        # Also compute MCC for reference
        details["mcc"] = float(matthews_corrcoef(y_val, y_pred))

        return composite, details, "OK"
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Fold %s failed: %s", fold_idx, exc)
        return None, None, "FAILED"


def _run_cv_scoring(
    features: pd.DataFrame,
    events: pd.DataFrame,
    cv: WalkForwardCV,
    model_cls: Callable[..., Any],
    params: Dict[str, Any],
    model_name: str,
    focal_params: Dict[str, Any],
) -> Tuple[float, int, int, str, List[float], List[Dict[str, float]]]:
    """Run CV scoring and aggregate metrics.

    Returns:
        mean_composite: Mean composite score across folds
        valid_folds: Number of valid folds
        total_folds: Total number of folds
        reason: Status message
        composite_scores: Per-fold composite scores
        all_details: Per-fold detailed metrics
    """
    splits = cv.split(features, events)
    composite_scores: List[float] = []
    all_details: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        composite, details, _ = _evaluate_fold(
            features,
            cast(pd.Series, events["label"]),
            train_idx,
            val_idx,
            model_cls,
            params,
            model_name,
            fold_idx,
            use_class_weights=focal_params.get("use_class_weights", True),
        )
        if composite is not None and details is not None:
            composite_scores.append(composite)
            all_details.append(details)
            logger.debug(
                "  Fold %d: composite=%.4f, F1_macro=%.4f, sign_err=%.2f%%, MCC=%.4f",
                fold_idx,
                composite,
                details.get("f1_macro", 0),
                details.get("sign_error_rate", 0) * 100,
                details.get("mcc", 0),
            )

    if not composite_scores:
        return 0.0, 0, len(splits), "no valid folds", [], []

    mean_composite = float(np.mean(composite_scores))
    return mean_composite, len(composite_scores), len(splits), "OK", composite_scores, all_details


# =============================================================================
# SAMPLING HELPERS
# =============================================================================


def _subsample_features(features: pd.DataFrame, data_fraction: float) -> pd.DataFrame:
    """Optionally subsample features for faster experimentation."""
    if data_fraction >= 1.0:
        return features
    return features.sample(frac=data_fraction, random_state=DEFAULT_RANDOM_STATE).sort_index()


def _sample_focal_loss_params(trial: optuna.trial.Trial, config: OptimizationConfig) -> Dict[str, Any]:
    """Sample focal loss parameters depending on model support and config."""
    params: Dict[str, Any] = {"use_class_weights": True}
    if not config.optimize_focal_params:
        return params

    if config.model_name in FOCAL_LOSS_SUPPORTED_MODELS:
        params["use_focal_loss"] = trial.suggest_categorical(
            "use_focal_loss",
            cast(List[Any], FOCAL_LOSS_SEARCH_SPACE["use_focal_loss"][1]),
        )
        params["focal_gamma"] = trial.suggest_categorical(
            "focal_gamma",
            cast(List[Any], FOCAL_LOSS_SEARCH_SPACE["focal_gamma"][1]),
        )

    params["minority_weight_boost"] = trial.suggest_categorical(
        "minority_weight_boost",
        cast(List[Any], FOCAL_LOSS_SEARCH_SPACE["minority_weight_boost"][1]),
    )
    return params


def _sample_model_params(trial: optuna.trial.Trial, model_name: str) -> Dict[str, Any]:
    """Sample model hyperparameters from MODEL_REGISTRY search space."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    search_space = MODEL_REGISTRY[model_name]["search_space"]
    params: Dict[str, Any] = {}

    for param_name, (param_type, param_values) in search_space.items():
        if param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_values)
        elif param_type == "int":
            low, high = param_values[0], param_values[1]
            log_scale = len(param_values) > 2 and param_values[2] == "log"
            params[param_name] = trial.suggest_int(param_name, low, high, log=log_scale)
        elif param_type == "float":
            low, high = param_values[0], param_values[1]
            log_scale = len(param_values) > 2 and param_values[2] == "log"
            params[param_name] = trial.suggest_float(param_name, low, high, log=log_scale)
        else:
            raise ValueError(f"Unknown param type '{param_type}' for {param_name}")

    return params


# =============================================================================
# OBJECTIVE FACTORY
# =============================================================================


def create_objective(
    config: OptimizationConfig,
    features: pd.DataFrame,
    events: pd.DataFrame,
    model_cls: Callable[..., Any],
) -> Callable[[optuna.trial.Trial], float]:
    """Create an Optuna objective function using pre-computed labels.

    The objective maximizes a composite score:
        composite = F1_macro - SIGN_ERROR_PENALTY * sign_error_rate

    This penalizes models that predict the wrong direction (e.g., +1 when true is -1).
    """
    cv = WalkForwardCV(config.n_splits, config.min_train_size)

    # Pre-align features and events once (labels are fixed)
    aligned_X, y, aligned_events, align_reason = _align_features_events(features, events, config)
    if aligned_X is None or y is None or aligned_events is None:
        raise ValueError(f"Cannot align features with events: {align_reason}")

    logger.info("Aligned %d samples for CV scoring", len(aligned_X))

    def objective(trial: optuna.trial.Trial) -> float:
        # Sample model hyperparameters from MODEL_REGISTRY
        model_params = _sample_model_params(trial, config.model_name)
        focal_params = _sample_focal_loss_params(trial, config)

        # Log trial start (brief, details shown in callback after completion)
        logger.info("[Trial %d] Starting...", trial.number)

        mean_composite, valid_folds, total_folds, cv_reason, composite_scores, all_details = _run_cv_scoring(
            aligned_X,
            aligned_events,
            cv,
            model_cls,
            model_params,
            config.model_name,
            focal_params,
        )

        # Aggregate per-fold details
        if all_details:
            mean_f1_macro = float(np.mean([d.get("f1_macro", 0) for d in all_details]))
            mean_sign_err = float(np.mean([d.get("sign_error_rate", 0) for d in all_details]))
            mean_mcc = float(np.mean([d.get("mcc", 0) for d in all_details]))

            # Per-class F1 averages
            f1_by_class: Dict[str, List[float]] = {}
            for d in all_details:
                for k, v in d.items():
                    if k.startswith("f1_class_"):
                        f1_by_class.setdefault(k, []).append(v)
            mean_f1_per_class = {k: float(np.mean(v)) for k, v in f1_by_class.items()}
        else:
            mean_f1_macro = 0.0
            mean_sign_err = 0.0
            mean_mcc = 0.0
            mean_f1_per_class = {}

        # Store in trial attributes
        trial.set_user_attr("n_valid_folds", valid_folds)
        trial.set_user_attr("n_total_folds", total_folds)
        trial.set_user_attr("cv_reason", cv_reason)
        trial.set_user_attr("mean_f1_macro", mean_f1_macro)
        trial.set_user_attr("mean_sign_error_rate", mean_sign_err)
        trial.set_user_attr("mean_mcc", mean_mcc)
        trial.set_user_attr("composite_per_fold", composite_scores)
        trial.set_user_attr("f1_per_class", mean_f1_per_class)

        return mean_composite

    return objective


# =============================================================================
# OPTIMIZATION ORCHESTRATION
# =============================================================================


def _create_study(model_name: str) -> optuna.Study:
    """Create an Optuna study aimed at maximizing composite F1 with sign penalty."""
    storage = LABEL_PRIMAIRE_OPTI_DIR / f"{model_name}_study.db"
    LABEL_PRIMAIRE_OPTI_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Optuna study storage: %s", storage)
    return optuna.create_study(direction="maximize", study_name=model_name, storage=f"sqlite:///{storage}", load_if_exists=True)


def _prepare_optimization_data(model_name: str, data_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and extract existing labels/t1 from dataset.

    Only uses TRAIN split data for optimization to prevent data leakage.
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA FOR MODEL: %s", model_name)
    logger.info("=" * 70)

    # Load full dataset
    full_dataset = get_dataset_for_model(model_name)
    logger.info("Full dataset loaded: %d samples, %d features", len(full_dataset), len(full_dataset.columns))

    if "datetime_close" in full_dataset.columns:
        full_dataset = full_dataset.set_index("datetime_close")
        logger.info("Index set to datetime_close")

    # Filter TRAIN split only
    if "split" in full_dataset.columns:
        train_mask = full_dataset["split"] == TRAIN_SPLIT_LABEL
        dataset = full_dataset[train_mask].copy()
        logger.info("Filtered to TRAIN split only: %d samples (%.1f%% of total)",
                   len(dataset), 100 * len(dataset) / len(full_dataset))
    else:
        dataset = full_dataset.copy()
        logger.warning("No 'split' column found - using all data (risk of data leakage!)")

    # Apply subsampling if requested
    if data_fraction < 1.0:
        dataset = _subsample_features(cast(pd.DataFrame, dataset), data_fraction)
        logger.info("Subsampled to %.1f%%: %d samples", data_fraction * 100, len(dataset))

    # Check for label column
    if "label" not in dataset.columns:
        raise ValueError(f"Dataset for model '{model_name}' does not contain 'label' column. "
                        "Run triple_barriere labeling first.")

    # Build events DataFrame from existing labels
    events = pd.DataFrame(index=dataset.index)
    events["label"] = dataset["label"]
    if "t1" in dataset.columns:
        events["t1"] = dataset["t1"]
        logger.info("t1 column found - purging will be applied in CV")
    else:
        logger.warning("No 't1' column found - CV purging disabled")

    # Log label distribution
    label_dist = events["label"].value_counts().sort_index()
    logger.info("Label distribution:")
    for label, count in label_dist.items():
        pct = 100 * count / len(events)
        logger.info("  Class %+d: %d samples (%.1f%%)", label, count, pct)

    # Remove non-feature columns (target, metadata, and leaky features)
    excluded_cols = {"label", "t1", "split", "log_return", "datetime_close"}
    feature_cols = [c for c in dataset.columns if c not in excluded_cols]
    features = cast(pd.DataFrame, dataset[feature_cols])
    logger.info("Feature matrix: %d samples x %d features", len(features), len(feature_cols))

    # Check for NaN values
    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        logger.warning("Features with NaN values: %s", nan_cols[:10])

    logger.info("=" * 70)

    return features, events


def _build_result(
    model_name: str,
    study: optuna.Study,
    best_focal_params: Dict[str, Any],
    metric: str = "composite_f1",
) -> OptimizationResult:
    """Convert study results into OptimizationResult."""
    best_score = float(study.best_value) if study.best_trials else 0.0

    # Get per-fold composite scores from best trial
    cv_scores: List[float] = []
    if study.best_trial:
        cv_scores = study.best_trial.user_attrs.get("composite_per_fold", [])

    return OptimizationResult(
        model_name=model_name,
        best_params=study.best_params,
        best_triple_barrier_params={},  # No longer optimized here
        best_focal_loss_params=best_focal_params,
        best_score=best_score,
        metric=metric,
        n_trials=len(study.trials),
        cv_scores=cv_scores,
    )


def _log_result(result: OptimizationResult, study: optuna.Study) -> None:
    """Log detailed summary of the optimization."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("OPTIMIZATION COMPLETE: %s", result.model_name)
    logger.info("=" * 70)
    logger.info("Best composite score: %.4f after %d trials", result.best_score, result.n_trials)
    logger.info("Composite = F1_macro - %.1f%% * sign_error_rate", SIGN_ERROR_PENALTY * 100)

    if study.best_trial:
        logger.info("-" * 70)
        logger.info("Best trial: #%d", study.best_trial.number)
        logger.info("Best params: %s", result.best_params)
        logger.info("-" * 70)

        # Detailed metrics from best trial
        attrs = study.best_trial.user_attrs
        mean_f1_macro = attrs.get("mean_f1_macro", 0)
        mean_sign_err = attrs.get("mean_sign_error_rate", 0)
        mean_mcc = attrs.get("mean_mcc", 0)

        logger.info("F1 macro:        %.4f", mean_f1_macro)
        logger.info("Sign error rate: %.1f%%", mean_sign_err * 100)
        logger.info("MCC:             %.4f", mean_mcc)

        # Per-class F1
        f1_per_class = attrs.get("f1_per_class", {})
        if f1_per_class:
            logger.info("-" * 70)
            logger.info("F1 per class:")
            for cls_key, f1_val in sorted(f1_per_class.items()):
                cls_label = cls_key.split("_")[-1]
                logger.info("  Class %s: %.4f", cls_label, f1_val)

        # Per-fold composite
        composite_per_fold = attrs.get("composite_per_fold", [])
        if composite_per_fold:
            logger.info("-" * 70)
            logger.info("Composite per fold: %s", [f"{x:.4f}" for x in composite_per_fold])

        n_folds = attrs.get("n_valid_folds", 0)
        n_total = attrs.get("n_total_folds", 0)
        logger.info("Valid folds: %d/%d", n_folds, n_total)

    logger.info("=" * 70)


def optimize_model(model_name: str, config: OptimizationConfig | None = None) -> OptimizationResult:
    """Run Optuna optimization for a given model.

    Uses composite score: F1_macro - SIGN_ERROR_PENALTY * sign_error_rate
    """
    config = config or OptimizationConfig(model_name=model_name)

    logger.info("")
    logger.info("*" * 70)
    logger.info("STARTING OPTIMIZATION: %s", model_name)
    logger.info("*" * 70)
    logger.info("Config: n_trials=%d, n_splits=%d, data_fraction=%.2f",
               config.n_trials, config.n_splits, config.data_fraction)

    model_cls = load_model_class(model_name)
    logger.info("Model class loaded: %s", model_cls.__name__)

    features, events = _prepare_optimization_data(model_name, config.data_fraction)

    objective = create_objective(config, features, events, model_cls)
    study = _create_study(model_name)

    logger.info("")
    logger.info("Starting %d optimization trials...", config.n_trials)
    logger.info("-" * 70)

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        callbacks=[_optuna_logging_callback],
    )

    # Collect all details from best trial
    best_focal: Dict[str, Any] = {}
    if study.best_trial:
        attrs = study.best_trial.user_attrs
        best_focal = {
            "use_class_weights": attrs.get("use_class_weights", True),
            # Detailed metrics for reference
            "mean_f1_macro": attrs.get("mean_f1_macro", 0),
            "mean_sign_error_rate": attrs.get("mean_sign_error_rate", 0),
            "mean_mcc": attrs.get("mean_mcc", 0),
            "f1_per_class": attrs.get("f1_per_class", {}),
            "sign_penalty_used": SIGN_ERROR_PENALTY,
        }

    result = _build_result(model_name, study, best_focal)

    # Save results to JSON
    output_path = LABEL_PRIMAIRE_OPTI_DIR / f"{model_name}_optimization.json"
    result.save(output_path)

    _log_result(result, study)
    return result


# =============================================================================
# CLI HELPERS
# =============================================================================


def select_models_interactive() -> List[str]:
    """Prompt user to select models from the registry."""
    model_names = list(MODEL_REGISTRY.keys())
    prompt = ["Select model(s):", "0 - All models"]
    prompt.extend([f"{i + 1} - {name}" for i, name in enumerate(model_names)])
    logger.info("\n".join(prompt))
    choice = input("Enter choice (name or number): ").strip()
    if choice == "0":
        return model_names
    if choice.isdigit():
        idx = int(choice) - 1
        return [model_names[idx]] if 0 <= idx < len(model_names) else []
    return [choice] if choice in model_names else []


def _run_optimization_worker(model_name: str, trials: int, n_splits: int) -> OptimizationResult:
    """Run optimization for a single model (sequential worker)."""
    config = OptimizationConfig(model_name=model_name, n_trials=trials, n_splits=n_splits)
    return optimize_model(model_name, config)


def run_sequential(models: List[str], trials_per_model: Dict[str, int], n_splits: int) -> List[OptimizationResult]:
    """Run optimizations sequentially over selected models."""
    results: List[OptimizationResult] = []
    for name in models:
        trials = trials_per_model.get(name, 10)
        results.append(_run_optimization_worker(name, trials, n_splits))
    return results


def run_parallel(models: List[str], trials_per_model: Dict[str, int], n_splits: int) -> List[OptimizationResult]:
    """Placeholder parallel runner using sequential execution."""
    return run_sequential(models, trials_per_model, n_splits)


def print_final_summary(results: List[OptimizationResult]) -> None:
    """Print optimization summary."""
    for res in results:
        logger.info("%s -> best %.4f (%s trials)", res.model_name, res.best_score, res.n_trials)
