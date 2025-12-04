"""Training pipeline for meta-labeling models."""


from __future__ import annotations


import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.labelling.label_meta.utils import (
    get_labeled_dataset_for_primary_model,
    get_available_primary_models,
    load_model_class,
)
from src.path import LABEL_META_OPTI_DIR, LABEL_META_TRAIN_DIR

logger = get_logger(__name__)


@dataclass
class MetaTrainingConfig:
    """Configuration used for meta model training."""

    primary_model_name: str
    meta_model_name: str
    random_state: int = DEFAULT_RANDOM_STATE
    use_class_weight: bool = True


@dataclass
class MetaEvaluationMetrics:
    """Binary evaluation metrics for meta model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
        }


@dataclass
class MetaTrainingResult:
    """Summary of a meta model training run."""

    primary_model_name: str
    meta_model_name: str
    meta_model_params: Dict[str, Any]
    triple_barrier_params: Dict[str, Any]
    train_samples: int
    label_distribution: Dict[str, Any]
    train_metrics: Dict[str, Any]
    model_path: str
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_model_name": self.primary_model_name,
            "meta_model_name": self.meta_model_name,
            "meta_model_params": self.meta_model_params,
            "triple_barrier_params": self.triple_barrier_params,
            "train_samples": self.train_samples,
            "label_distribution": self.label_distribution,
            "train_metrics": self.train_metrics,
            "model_path": self.model_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Training results saved to %s", path)


def load_optimized_params(
    primary_model_name: str,
    meta_model_name: str,
    opti_dir: Path | None = None,
) -> Dict[str, Any]:
    """Load optimized parameters produced by the optimization step."""
    directory = opti_dir or LABEL_META_OPTI_DIR
    opti_file = directory / f"{primary_model_name}_{meta_model_name}_optimization.json"
    if not opti_file.exists():
        raise FileNotFoundError(f"Optimization results not found: {opti_file}")
    with open(opti_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "meta_model_params": data.get("best_params", {}),
        "triple_barrier_params": data.get("best_triple_barrier_params", {}),
        "best_score": data.get("best_score", 0.0),
        "metric": data.get("metric", "f1_score"),
    }


def _compute_class_weights(y: pd.Series) -> Dict[int, float]:
    counts = y.value_counts()
    total = len(y)
    return {int(cast(Any, cls)): total / (len(counts) * count) for cls, count in counts.items()}


def _remove_non_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    non_feature_cols = {
        "bar_id", "split", "datetime_close", "label", "prediction", "t1",
        "timestamp_open", "timestamp_close", "datetime_open", "threshold_used",
        "log_return", "coverage",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    return cast(pd.DataFrame, df[feature_cols])


def _handle_missing_values(features_df: pd.DataFrame) -> pd.DataFrame:
    filled = features_df.copy()
    for col in filled.columns:
        if bool(filled[col].isna().any()):
            median = filled[col].median()
            filled[col] = filled[col].fillna(0.0 if bool(pd.isna(median)) else median)
    return filled


def build_meta_features(
    primary_model_name: str,
    filter_neutral_labels: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build meta-label features using OOF predictions from primary model.

    The labeled dataset contains prediction, coverage, and label columns from
    the primary model training.
    """
    # Load labeled dataset (contains features + OOF predictions + labels)
    dataset = get_labeled_dataset_for_primary_model(primary_model_name)

    # Filter: valid OOF predictions only
    if "coverage" not in dataset.columns or "prediction" not in dataset.columns:
        raise ValueError(
            f"Dataset for '{primary_model_name}' missing 'coverage' or 'prediction'. "
            "Re-train the primary model to generate OOF predictions."
        )

    valid_mask = dataset["coverage"] == 1
    dataset = dataset.loc[valid_mask].copy()
    logger.info("Using %d samples with valid OOF predictions", len(dataset))

    # Filter neutral predictions if requested (keep only directional trades)
    if filter_neutral_labels:
        directional_mask = dataset["prediction"] != 0
        dataset = dataset.loc[directional_mask].copy()
        logger.info("After filtering neutral predictions: %d samples", len(dataset))

    # Create meta-label: 1 if primary prediction was correct, 0 otherwise
    meta_labels = (dataset["prediction"] == dataset["label"]).astype(int)
    logger.info("Meta label distribution: correct=%d (%.1f%%), incorrect=%d (%.1f%%)",
                meta_labels.sum(), 100 * meta_labels.mean(),
                len(meta_labels) - meta_labels.sum(), 100 * (1 - meta_labels.mean()))

    # Prepare features
    features = _remove_non_feature_cols(dataset)
    features = _handle_missing_values(features)

    return features, pd.Series(meta_labels, index=features.index)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> MetaEvaluationMetrics:
    """Compute binary classification metrics."""
    auc_roc = None
    if y_proba is not None:
        try:
            proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            auc_roc = float(roc_auc_score(y_true, proba_pos))
        except ValueError:
            auc_roc = None
    return MetaEvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division="warn")),
        recall=float(recall_score(y_true, y_pred, zero_division="warn")),
        f1=float(f1_score(y_true, y_pred, zero_division="warn")),
        auc_roc=auc_roc,
    )


def get_available_meta_optimizations() -> List[tuple[str, str]]:
    """List available (primary, meta) optimization results."""
    if not LABEL_META_OPTI_DIR.exists():
        return []
    optimizations: List[tuple[str, str]] = []
    for file in LABEL_META_OPTI_DIR.glob("*_optimization.json"):
        stem = file.stem.replace("_optimization", "")
        parts = stem.split("_")
        if len(parts) >= 2:
            primary = parts[0]
            meta = "_".join(parts[1:])
            optimizations.append((primary, meta))
    return optimizations


def train_meta_model(
    primary_model_name: str,
    meta_model_name: str,
    config: MetaTrainingConfig | None = None,
    opti_dir: Path | None = None,
    output_dir: Path | None = None,
) -> MetaTrainingResult:
    """Train a meta model using optimized hyperparameters."""
    cfg = config or MetaTrainingConfig(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
    )

    optimized = load_optimized_params(primary_model_name, meta_model_name, opti_dir)
    meta_params = optimized.get("meta_model_params", {})
    tb_params = optimized.get("triple_barrier_params", {})

    features, meta_labels = build_meta_features(primary_model_name, filter_neutral_labels=True)

    if cfg.use_class_weight:
        meta_params["class_weight"] = _compute_class_weights(meta_labels)

    model_cls = load_model_class(meta_model_name)
    model = model_cls(**meta_params)
    model.fit(features, meta_labels)

    # Compute training metrics
    train_preds = model.predict(features)
    train_proba = None
    predict_proba_fn = getattr(model, "predict_proba", None)
    if predict_proba_fn is not None:
        try:
            train_proba = predict_proba_fn(features)
        except Exception:
            pass
    train_metrics = compute_metrics(meta_labels.to_numpy(), train_preds, train_proba)

    # Save model
    model_dir = (output_dir or LABEL_META_TRAIN_DIR) / f"{primary_model_name}_{meta_model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{meta_model_name}_model.joblib"
    model.save(model_path)

    label_dist = meta_labels.value_counts().to_dict()

    result = MetaTrainingResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        meta_model_params=meta_params,
        triple_barrier_params=tb_params,
        train_samples=len(features),
        label_distribution={str(k): int(v) for k, v in label_dist.items()},
        train_metrics=train_metrics.to_dict(),
        model_path=str(model_path),
    )

    results_path = model_dir / "training_results.json"
    result.save(results_path)

    return result


def select_primary_model() -> str | None:
    """Prompt user for a primary model selection."""
    available = get_available_primary_models()
    if not available:
        logger.info("No trained primary models found.")
        return None
    logger.info("Select primary model:")
    for idx, name in enumerate(available, start=1):
        logger.info("%s - %s", idx, name)
    choice = input("Enter choice (number or name): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        return available[idx] if 0 <= idx < len(available) else None
    return choice if choice in available else None


def select_meta_model(primary_model_name: str) -> str | None:
    """Prompt user for a meta model type selection for given primary model."""
    available = get_available_meta_optimizations()
    matching = [(p, m) for p, m in available if p == primary_model_name]
    if not matching:
        logger.info("No meta optimizations found for primary model %s.", primary_model_name)
        return None
    meta_models = list(set(m for _, m in matching))
    logger.info("Select meta model type:")
    for idx, name in enumerate(meta_models, start=1):
        logger.info("%s - %s", idx, name)
    choice = input("Enter choice (number or name): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        return meta_models[idx] if 0 <= idx < len(meta_models) else None
    return choice if choice in meta_models else None


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
    "MetaEvaluationMetrics",
    "MetaTrainingConfig",
    "MetaTrainingResult",
    "build_meta_features",
    "compute_metrics",
    "get_available_meta_optimizations",
    "get_yes_no_input",
    "load_optimized_params",
    "select_meta_model",
    "select_primary_model",
    "train_meta_model",
]
