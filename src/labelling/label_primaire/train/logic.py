"""Training utilities for primary-label models with OOF predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE, TRAIN_SPLIT_LABEL
from src.labelling.label_primaire.utils import (
    MODEL_REGISTRY,
    get_dataset_for_model,
    load_model_class,
)
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    LABEL_PRIMAIRE_OPTI_DIR,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = get_logger(__name__)


class WalkForwardKFold:
    """Time-series aware CV with optional purging."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01, min_train_size: int = 50) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series | None = None,
    ) -> List[tuple[np.ndarray, np.ndarray]]:
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
                if embargo > 0:
                    train_idx = train_idx[:-embargo] if embargo < len(train_idx) else np.array([], dtype=int)
            if t1 is not None:
                train_idx = self._apply_purge(train_idx, val_idx, X, t1)
            if len(train_idx) >= self.min_train_size and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

    def _apply_purge(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        t1: pd.Series | None,
    ) -> np.ndarray:
        if t1 is None or len(train_idx) == 0 or len(val_idx) == 0:
            return train_idx
        val_start = X.index[val_idx[0]]
        t1_series = t1.iloc[train_idx]
        # Ensure timezone compatibility: convert both to naive if needed
        if isinstance(val_start, pd.Timestamp) and hasattr(val_start, 'tz') and val_start.tz is not None:
            val_start = val_start.tz_localize(None)
        if hasattr(t1_series, 'dt') and hasattr(t1_series.dtype, 'tz') and t1_series.dt.tz is not None:
            t1_series = t1_series.dt.tz_localize(None)
        mask = ~(t1_series.notna() & (t1_series >= val_start))
        purged = train_idx[mask.to_numpy()]
        return purged if len(purged) > 0 else train_idx


@dataclass
class PrimaryEvaluationMetrics:
    """Multiclass evaluation metrics for primary model."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    mcc: float
    auc_roc: float | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "mcc": self.mcc,
            "auc_roc": self.auc_roc,
        }


@dataclass
class TrainingConfig:
    """Configuration used for primary model training."""

    model_name: str
    random_state: int = DEFAULT_RANDOM_STATE
    use_class_weight: bool = True


@dataclass
class TrainingResult:
    """Training result summary for primary model with OOF."""

    model_name: str
    model_params: Dict[str, Any]
    triple_barrier_params: Dict[str, Any]
    train_samples: int
    test_samples: int
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    label_distribution_train: Dict[str, Any]
    label_distribution_test: Dict[str, Any]
    n_folds: int
    model_path: str
    oof_predictions_path: str
    test_predictions_path: str
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "triple_barrier_params": self.triple_barrier_params,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "label_distribution_train": self.label_distribution_train,
            "label_distribution_test": self.label_distribution_test,
            "n_folds": self.n_folds,
            "model_path": self.model_path,
            "oof_predictions_path": self.oof_predictions_path,
            "test_predictions_path": self.test_predictions_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def load_optimized_params(model_name: str, opti_dir: Path | None = None) -> Dict[str, Any]:
    """Load optimization parameters for a primary model."""
    directory = opti_dir or LABEL_PRIMAIRE_OPTI_DIR
    opti_file = directory / f"{model_name}_optimization.json"
    if not opti_file.exists():
        raise FileNotFoundError(f"Primary optimization results not found: {opti_file}")
    with open(opti_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "model_params": data.get("best_params", {}),
        "triple_barrier_params": data.get("best_triple_barrier_params", {}),
        "best_score": data.get("best_score", 0.0),
        "metric": data.get("metric", "mcc"),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> PrimaryEvaluationMetrics:
    """Compute multiclass classification metrics."""
    auc_roc = None
    if y_proba is not None:
        try:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y_true)
            if len(classes) > 2:
                y_true_bin = label_binarize(y_true, classes=classes)
                auc_roc = float(roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro"))
            else:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                auc_roc = float(roc_auc_score(y_true, proba_pos))
        except ValueError:
            auc_roc = None
    return PrimaryEvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division="warn")),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division="warn")),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division="warn")),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        auc_roc=auc_roc,
    )


def _remove_non_feature_cols(features_df: pd.DataFrame) -> pd.DataFrame:
    non_feature_cols = {"bar_id", "split", "datetime_close", "label", "prediction", "t1",
                        "timestamp_open", "timestamp_close", "datetime_open", "threshold_used", "log_return"}
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    return pd.DataFrame(features_df[feature_cols])


def _handle_missing_values(features_df: pd.DataFrame) -> pd.DataFrame:
    filled = features_df.copy()
    for col in filled.columns:
        if bool(filled[col].isna().any()):
            median = filled[col].median()
            filled[col] = filled[col].fillna(0.0 if bool(pd.isna(median)) else median)
    return filled


def _compute_class_weights(y: pd.Series) -> Dict[int, float]:
    counts = y.value_counts()
    total = len(y)
    return {int(cast(Any, cls)): total / (len(counts) * count) for cls, count in counts.items()}


def get_available_optimized_models() -> List[str]:
    """List available optimization results."""
    if not LABEL_PRIMAIRE_OPTI_DIR.exists():
        return []
    return [p.stem.replace("_optimization", "") for p in LABEL_PRIMAIRE_OPTI_DIR.glob("*_optimization.json")]


def _get_labeled_output_path(model_name: str) -> Path:
    """Get output path for labeled dataset with OOF predictions."""
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    mapping = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    base = mapping[dataset_type]
    return base.parent / f"{base.stem}_{model_name}.parquet"


def generate_oof_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    t1: pd.Series | None,
    model_class: Any,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    min_train_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions using walk-forward CV."""
    cv = WalkForwardKFold(n_splits=n_splits, min_train_size=min_train_size)
    splits = cv.split(features, labels, t1=t1)

    n_classes = len(labels.unique())
    if len(splits) == 0:
        n = len(labels)
        return np.full(n, np.nan), np.full((n, n_classes), np.nan), np.zeros(n, dtype=int)

    oof_preds = np.full(len(labels), np.nan)
    oof_proba = np.full((len(labels), n_classes), np.nan)
    coverage = np.zeros(len(labels), dtype=int)

    for train_idx, val_idx in splits:
        X_train, y_train = features.iloc[train_idx], labels.iloc[train_idx]
        X_val = features.iloc[val_idx]
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        coverage[val_idx] = 1
        if hasattr(model, "predict_proba"):
            try:
                oof_proba[val_idx] = model.predict_proba(X_val)
            except Exception:
                pass

    return oof_preds, oof_proba, coverage


def _prepare_dataset(model_name: str) -> pd.DataFrame:
    dataset = get_dataset_for_model(model_name)
    if "datetime_close" in dataset.columns:
        dataset = dataset.set_index("datetime_close")
    return dataset.sort_index()


def _ensure_labels(dataset: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataset has labels. Labels must already exist from triple_barriere."""
    if "label" not in dataset.columns:
        raise ValueError(
            "Dataset does not contain 'label' column. "
            "Run triple_barriere labeling first."
        )
    return dataset


def _get_train_dataframe(dataset: pd.DataFrame) -> pd.DataFrame:
    if "split" not in dataset.columns:
        return dataset
    return dataset.loc[dataset["split"] == TRAIN_SPLIT_LABEL]


def train_model(
    model_name: str,
    config: TrainingConfig | None = None,
    opti_dir: Path | None = None,
    output_dir: Path | None = None,
    n_splits: int = 10,
) -> TrainingResult:
    """Train a primary model using optimized hyperparameters and generate OOF predictions."""
    cfg = config or TrainingConfig(model_name=model_name)
    optimized = load_optimized_params(model_name, opti_dir)
    model_params = optimized.get("model_params", {})
    tb_params = optimized.get("triple_barrier_params", {})

    dataset = _ensure_labels(_prepare_dataset(model_name))
    train_df = _get_train_dataframe(dataset)

    # Filter rows with missing labels before processing
    valid_label_mask = train_df["label"].notna()
    train_df = train_df.loc[valid_label_mask]
    logger.info("Dropped %d rows with missing labels", (~valid_label_mask).sum())

    features = _remove_non_feature_cols(train_df)
    features = _handle_missing_values(features)
    labels = cast(pd.Series, train_df["label"].astype(int))
    t1 = train_df.get("t1") if "t1" in train_df.columns else None

    if cfg.use_class_weight:
        model_params["class_weight"] = _compute_class_weights(labels)

    model_cls = load_model_class(model_name)
    oof_preds, oof_proba, coverage = generate_oof_predictions(
        features,
        labels,
        t1,
        model_cls,
        model_params,
        n_splits=n_splits,
    )

    valid_mask = coverage == 1
    labels_valid = cast(pd.Series, labels[valid_mask])
    train_metrics = compute_metrics(
        labels_valid.to_numpy(),
        oof_preds[valid_mask],
        oof_proba[valid_mask] if oof_proba is not None else None,
    )

    # Train final model on all training data
    model = model_cls(**model_params)
    model.fit(features, labels.to_numpy())

    model_dir = (output_dir or LABEL_PRIMAIRE_TRAIN_DIR) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}_model.joblib"
    model.save(model_path)

    # Save FULL dataset with OOF predictions for meta-labelling
    # Train set has OOF predictions, test set has NaN (will be predicted at eval time)
    n_classes = len(labels.unique())

    # Start with full dataset
    full_labeled_df = dataset.copy()

    # Initialize columns with NaN
    full_labeled_df["prediction"] = np.nan
    for i in range(n_classes):
        full_labeled_df[f"proba_{i}"] = np.nan
    full_labeled_df["coverage"] = 0

    # Fill in OOF values for train set
    train_idx = train_df.index
    full_labeled_df.loc[train_idx, "prediction"] = oof_preds
    for i in range(n_classes):
        if oof_proba is not None and oof_proba.ndim == 2:
            full_labeled_df.loc[train_idx, f"proba_{i}"] = oof_proba[:, i]
    full_labeled_df.loc[train_idx, "coverage"] = coverage

    labeled_output_path = _get_labeled_output_path(model_name)
    full_labeled_df.to_parquet(labeled_output_path)
    logger.info("Full labeled dataset saved to %s (train: %d, test: %d)",
                labeled_output_path, len(train_df), len(dataset) - len(train_df))

    label_dist_train = labels.value_counts().to_dict()

    result = TrainingResult(
        model_name=model_name,
        model_params=model_params,
        triple_barrier_params=tb_params,
        train_samples=len(features),
        test_samples=0,
        train_metrics=train_metrics.to_dict(),
        test_metrics={},
        label_distribution_train={str(k): int(v) for k, v in label_dist_train.items()},
        label_distribution_test={},
        n_folds=n_splits,
        model_path=str(model_path),
        oof_predictions_path=str(labeled_output_path),
        test_predictions_path="",
    )

    results_path = model_dir / "training_results.json"
    result.save(results_path)
    logger.info("Primary model trained and saved to %s", model_path)

    return result


def evaluate_model(
    model_name: str,
    train_dir: Path | None = None,
) -> PrimaryEvaluationMetrics:
    """Load and evaluate a trained primary model from OOF predictions."""
    model_dir = (train_dir or LABEL_PRIMAIRE_TRAIN_DIR) / model_name
    oof_path = model_dir / "oof_predictions.parquet"

    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

    oof_df = pd.read_parquet(oof_path)
    valid = oof_df["coverage"] == 1
    y_true = oof_df.loc[valid, "label"].to_numpy()
    y_pred = oof_df.loc[valid, "prediction"].to_numpy()

    proba_cols = [c for c in oof_df.columns if c.startswith("proba_")]
    y_proba = None
    if proba_cols:
        proba_arr = oof_df.loc[valid, proba_cols].to_numpy()
        if not np.isnan(proba_arr).all():
            y_proba = proba_arr

    return compute_metrics(y_true, y_pred, y_proba)


def select_model() -> str | None:
    """Prompt user for a single model selection."""
    available = get_available_optimized_models() or list(MODEL_REGISTRY.keys())
    for idx, name in enumerate(available, start=1):
        logger.info("%s - %s", idx, name)
    choice = input("Select model (number or name): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        return available[idx] if 0 <= idx < len(available) else None
    return choice if choice in available else None


def get_yes_no_input(prompt: str, default: bool | None = None) -> bool:
    """Return True/False based on user input with optional default."""
    suffix = "[y/n]"
    if default is True:
        suffix = "[Y/n]"
    elif default is False:
        suffix = "[y/N]"
    value = input(f"{prompt} {suffix}: ").strip().lower()
    if not value and default is not None:
        return default
    return value.startswith("y")


__all__ = [
    "PrimaryEvaluationMetrics",
    "TrainingConfig",
    "TrainingResult",
    "WalkForwardKFold",
    "_handle_missing_values",
    "_remove_non_feature_cols",
    "compute_metrics",
    "evaluate_model",
    "generate_oof_predictions",
    "get_available_optimized_models",
    "get_yes_no_input",
    "load_optimized_params",
    "select_model",
    "train_model",
]
