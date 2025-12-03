"""Evaluation utilities for primary labeling models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config_logging import get_logger
from src.labelling.label_primaire.utils import MODEL_REGISTRY
from src.model.base import BaseModel
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for standard classification metrics."""

    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    f1_per_class: Dict[str, float]
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    mcc: float
    cohen_kappa: float
    auc_roc_macro: float | None
    log_loss_value: float | None
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "f1_per_class": self.f1_per_class,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "mcc": self.mcc,
            "cohen_kappa": self.cohen_kappa,
            "auc_roc_macro": self.auc_roc_macro,
            "log_loss": self.log_loss_value,
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
            "n_samples": self.n_samples,
        }


@dataclass
class EvaluationResult:
    """Evaluation summary across train and test splits."""

    model_name: str
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    train_samples: int
    test_samples: int
    label_distribution_train: Dict[str, Any]
    label_distribution_test: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "label_distribution_train": self.label_distribution_train,
            "label_distribution_test": self.label_distribution_test,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Evaluation results saved to %s", path)


# =============================================================================
# DATA HELPERS
# =============================================================================


def _dataset_path_for_model(model_name: str) -> Path:
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    mapping = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    return mapping[dataset_type]


def get_features_path(model_name: str) -> Path:
    """Return the base features path for the given model type."""
    return _dataset_path_for_model(model_name)


def get_labeled_features_path(model_name: str) -> Path:
    """Return the model-specific labeled features path."""
    base = _dataset_path_for_model(model_name)
    return base.parent / f"{base.stem}_{model_name}.parquet"


def get_trained_models() -> List[str]:
    """List models that have been trained."""
    if not LABEL_PRIMAIRE_TRAIN_DIR.exists():
        return []
    trained: List[str] = []
    for subdir in LABEL_PRIMAIRE_TRAIN_DIR.iterdir():
        if subdir.is_dir() and (subdir / f"{subdir.name}_model.joblib").exists():
            trained.append(subdir.name)
    return trained


def load_model(model_name: str) -> BaseModel:
    """Load a trained model from disk."""
    model_path = LABEL_PRIMAIRE_TRAIN_DIR / model_name / f"{model_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return BaseModel.load(model_path)


def load_data(model_name: str) -> pd.DataFrame:
    """Load labeled dataset for evaluation."""
    path = get_labeled_features_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Labeled features not found: {path}")
    df = pd.read_parquet(path)
    if "label" not in df.columns:
        raise ValueError("No 'label' column present in dataset.")
    if "datetime_close" in df.columns:
        df = df.set_index("datetime_close")
    return df.sort_index()


# =============================================================================
# METRICS
# =============================================================================


def _safe_confusion_matrix(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Tuple[List[List[int]], List[str]]:
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.tolist(), [str(label) for label in labels]


def compute_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute a collection of classification metrics."""
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    average_mode = "binary" if len(labels) == 2 else "macro"
    f1_per_class_values = np.asarray(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0))  # type: ignore
    f1_per_class = {str(label): float(score) for label, score in zip(labels, f1_per_class_values)}

    cm, cm_labels = _safe_confusion_matrix(y_true, y_pred)
    auc_roc = None
    log_loss_val = None
    if y_proba is not None:
        try:
            if len(labels) == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                auc_roc = float(roc_auc_score(y_true, proba_pos))
            else:
                auc_roc = float(roc_auc_score(y_true, y_proba, multi_class="ovo"))
            log_loss_val = float(log_loss(y_true, y_proba))
        except ValueError:
            pass

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),  # type: ignore
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0.0)),  # type: ignore
        f1_per_class=f1_per_class,
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0.0)),  # type: ignore
        precision_weighted=float(precision_score(y_true, y_pred, average="weighted", zero_division=0.0)),  # type: ignore
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0.0)),  # type: ignore
        recall_weighted=float(recall_score(y_true, y_pred, average="weighted", zero_division=0.0)),  # type: ignore
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        auc_roc_macro=auc_roc,
        log_loss_value=log_loss_val,
        confusion_matrix=cm,
        class_labels=cm_labels,
        n_samples=len(y_true),
    )


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================


def _split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # type: ignore
    if "split" not in df.columns:
        return df, df.iloc[0:0]
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] != "train"]
    return train_df, test_df  # type: ignore


def _prepare_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    features = df.drop(columns=["label", "t1"], errors="ignore")
    labels = df["label"]
    return features, labels  # type: ignore


def evaluate_model(model_name: str) -> EvaluationResult:
    """Evaluate a trained model on train and test splits."""
    model = load_model(model_name)
    data = load_data(model_name)
    train_df, test_df = _split_dataset(data)

    X_train, y_train = _prepare_features_and_labels(train_df)
    X_test, y_test = _prepare_features_and_labels(test_df if not test_df.empty else train_df)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None  # type: ignore
    y_proba_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None  # type: ignore

    train_metrics = compute_metrics(y_train, y_pred_train, y_proba_train).to_dict()  # type: ignore
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test).to_dict()  # type: ignore

    label_distribution_train = {"counts": y_train.value_counts().to_dict(), "percentages": (y_train.value_counts(normalize=True) * 100).to_dict()}
    label_distribution_test = {"counts": y_test.value_counts().to_dict(), "percentages": (y_test.value_counts(normalize=True) * 100).to_dict()}

    return EvaluationResult(
        model_name=model_name,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_samples=len(y_train),
        test_samples=len(y_test),
        label_distribution_train=label_distribution_train,
        label_distribution_test=label_distribution_test,
    )


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def print_metrics(metrics: Dict[str, Any], title: str) -> None:
    """Pretty print a metrics dictionary."""
    lines = [title]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key.replace('_', ' ').title()}: {value:.4f}")
    print("\n".join(lines))


def print_confusion_matrix(cm: List[List[int]], labels: List[str]) -> None:
    """Print a confusion matrix."""
    print("Confusion Matrix:")
    print("Labels:", labels)
    for row in cm:
        print(" ".join(str(int(v)) for v in row))


def print_comparison(result: EvaluationResult) -> None:
    """Print train vs test comparison."""
    print("TRAIN vs TEST")
    for metric in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc"):
        train_val = result.train_metrics.get(metric, 0.0)
        test_val = result.test_metrics.get(metric, 0.0)
        print(f"{metric}: train={train_val:.4f} | test={test_val:.4f} | Delta={train_val - test_val:+.4f}")


def print_results(result: EvaluationResult) -> None:
    """Print a complete evaluation report."""
    print("EVALUATION RESULTS -", result.model_name.upper())
    print("\nTRAIN METRICS")
    print_metrics(result.train_metrics, "Train")
    if "confusion_matrix" in result.train_metrics:
        print_confusion_matrix(
            result.train_metrics["confusion_matrix"],
            result.train_metrics.get("class_labels", []),
        )
    print("\nTEST METRICS")
    print_metrics(result.test_metrics, "Test")
    if "confusion_matrix" in result.test_metrics:
        print_confusion_matrix(
            result.test_metrics["confusion_matrix"],
            result.test_metrics.get("class_labels", []),
        )
    print("\nTRAIN vs TEST")
    print_comparison(result)


__all__ = [
    "ClassificationMetrics",
    "EvaluationResult",
    "compute_metrics",
    "evaluate_model",
    "get_features_path",
    "get_labeled_features_path",
    "get_trained_models",
    "load_data",
    "load_model",
    "print_comparison",
    "print_confusion_matrix",
    "print_metrics",
    "print_results",
]
