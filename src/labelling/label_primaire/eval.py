"""
Primary Model Evaluation Module.

Evaluates trained primary models with comprehensive metrics.
Loads the model, makes predictions on train/test data, and displays metrics.

Metrics computed:
- Accuracy, Balanced Accuracy
- F1-Score (macro, weighted, per-class)
- Precision, Recall
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- AUC-ROC, Log Loss
- Confusion Matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
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

from src.constants import TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL
from src.labelling.label_primaire.utils import MODEL_REGISTRY
from src.model.base import BaseModel
from src.path import (
    LABEL_PRIMAIRE_TRAIN_DIR,
    LABEL_PRIMAIRE_EVAL_DIR,
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
)

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class ClassificationMetrics:
    """Classification metrics for model evaluation."""

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
        """Convert to dictionary."""
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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_labels = [str(c) for c in sorted(classes)]

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division="warn"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division="warn"))
    f1_per_class_arr = cast(np.ndarray, f1_score(y_true, y_pred, average=None, zero_division="warn", labels=sorted(classes)))  # type: ignore[call-overload]
    f1_per_class = {str(c): float(f) for c, f in zip(sorted(classes), f1_per_class_arr)}

    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division="warn"))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division="warn"))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division="warn"))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division="warn"))

    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    auc_roc_macro = None
    log_loss_val = None

    if y_proba is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                auc_roc_macro = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                auc_roc_macro = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                ))
            log_loss_val = float(log_loss(y_true, y_proba, labels=sorted(classes)))
        except (ValueError, IndexError):
            pass

    cm = confusion_matrix(y_true, y_pred, labels=sorted(classes))

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        f1_per_class=f1_per_class,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        mcc=mcc,
        cohen_kappa=kappa,
        auc_roc_macro=auc_roc_macro,
        log_loss_value=log_loss_val,
        confusion_matrix=cm.tolist(),
        class_labels=class_labels,
        n_samples=len(y_true),
    )


# =============================================================================
# DATA LOADING
# =============================================================================


def get_features_path(model_name: str) -> Path:
    """Get the base features file path for a model type (without labels)."""
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    return path_map[dataset_type]


def get_labeled_features_path(model_name: str) -> Path:
    """
    Get the labeled features file path for a specific model.

    Each model has its own labeled features file to avoid conflicts
    when training multiple models with different triple barrier parameters.

    Example: lightgbm -> data/features/dataset_features_final_lightgbm.parquet
    """
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    base_path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    base_path = base_path_map[dataset_type]
    # Insert model name before .parquet extension
    return base_path.parent / f"{base_path.stem}_{model_name}.parquet"


def get_trained_models() -> List[str]:
    """Get list of models that have been trained."""
    trained = []
    for model_name in MODEL_REGISTRY.keys():
        model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
        model_file = model_dir / f"{model_name}_model.joblib"
        if model_file.exists():
            trained.append(model_name)
    return trained


def load_model(model_name: str) -> BaseModel:
    """Load a trained model."""
    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
    model_path = model_dir / f"{model_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run training first: python -m src.labelling.label_primaire.train"
        )

    return BaseModel.load(model_path)


def load_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data with labels.

    Loads from the model-specific labeled features file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with features and labels
    """
    features_path = get_labeled_features_path(model_name)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Labeled features file not found: {features_path}\n"
            f"Run training first: python -m src.labelling.label_primaire.train"
        )

    df = pd.read_parquet(features_path)

    if "label" not in df.columns:
        raise ValueError(
            f"No 'label' column in {features_path}.\n"
            f"Run training first to generate labels: python -m src.labelling.label_primaire.train"
        )

    if "split" not in df.columns:
        raise ValueError(f"No 'split' column in {features_path}")

    train_df = cast(pd.DataFrame, df[df["split"] == TRAIN_SPLIT_LABEL].copy())
    test_df = cast(pd.DataFrame, df[df["split"] == TEST_SPLIT_LABEL].copy())

    return train_df, test_df


# =============================================================================
# EVALUATION
# =============================================================================


@dataclass
class EvaluationResult:
    """Evaluation result."""

    model_name: str
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    train_samples: int
    test_samples: int
    label_distribution_train: Dict[str, Any]
    label_distribution_test: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {path}")


def evaluate_model(model_name: str) -> EvaluationResult:
    """
    Evaluate a trained primary model.

    Loads the model, makes predictions on train/test data,
    and computes comprehensive metrics.
    """
    logger.info(f"{'='*60}")
    logger.info(f"EVALUATION: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Load model
    logger.info("Loading model...")
    model = load_model(model_name)

    # Load data
    logger.info("Loading data...")
    train_df, test_df = load_data(model_name)

    # Prepare features
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close", "threshold_used",
        "log_return", "split", "label",
    ]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

    # Filter valid labels
    train_valid = train_df[~train_df["label"].isna()].copy()
    test_valid = test_df[~test_df["label"].isna()].copy()

    X_train = cast(pd.DataFrame, train_valid[feature_cols])
    y_train = cast(np.ndarray, cast(pd.Series, train_valid["label"]).astype(int).values)
    X_test = cast(pd.DataFrame, test_valid[feature_cols])
    y_test = cast(np.ndarray, cast(pd.Series, test_valid["label"]).astype(int).values)

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Make predictions
    logger.info("Making predictions...")
    y_train_pred = cast(np.ndarray, model.predict(X_train))
    y_test_pred = cast(np.ndarray, model.predict(X_test))

    # Get probabilities if available
    y_train_proba = None
    y_test_proba = None
    try:
        y_train_proba = cast(Any, model).predict_proba(X_train)
        y_test_proba = cast(Any, model).predict_proba(X_test)
    except Exception:
        pass

    # Compute metrics
    logger.info("Computing metrics...")
    train_metrics = compute_metrics(cast(np.ndarray, y_train), y_train_pred, y_train_proba)
    test_metrics = compute_metrics(cast(np.ndarray, y_test), y_test_pred, y_test_proba)

    # Label distributions
    train_counts = pd.Series(y_train).value_counts().to_dict()
    test_counts = pd.Series(y_test).value_counts().to_dict()

    train_dist = {
        "total": len(y_train),
        "counts": {str(k): int(v) for k, v in train_counts.items()},
        "percentages": {str(k): v / len(y_train) * 100 for k, v in train_counts.items()},
    }
    test_dist = {
        "total": len(y_test),
        "counts": {str(k): int(v) for k, v in test_counts.items()},
        "percentages": {str(k): v / len(y_test) * 100 for k, v in test_counts.items()},
    }

    return EvaluationResult(
        model_name=model_name,
        train_metrics=train_metrics.to_dict(),
        test_metrics=test_metrics.to_dict(),
        train_samples=len(y_train),
        test_samples=len(y_test),
        label_distribution_train=train_dist,
        label_distribution_test=test_dist,
    )


# =============================================================================
# DISPLAY
# =============================================================================


def print_metrics(metrics: Dict[str, Any], title: str) -> None:
    """Print metrics table."""
    print(f"\n{title}")
    print("=" * 50)

    key_metrics = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("F1 Macro", "f1_macro"),
        ("F1 Weighted", "f1_weighted"),
        ("Precision Macro", "precision_macro"),
        ("Recall Macro", "recall_macro"),
        ("MCC", "mcc"),
        ("Cohen's Kappa", "cohen_kappa"),
        ("AUC-ROC Macro", "auc_roc_macro"),
        ("Log Loss", "log_loss"),
    ]

    for name, key in key_metrics:
        value = metrics.get(key)
        if value is not None:
            print(f"  {name:<22}: {value:>8.4f}")
        else:
            print(f"  {name:<22}: {'N/A':>8}")

    if "f1_per_class" in metrics:
        print("\n  Per-Class F1:")
        for cls, f1 in metrics["f1_per_class"].items():
            print(f"    Class {cls}: {f1:.4f}")


def print_confusion_matrix(cm: List[List[int]], labels: List[str]) -> None:
    """Print confusion matrix."""
    print("\nConfusion Matrix:")
    print("-" * 40)

    header = "True\\Pred"
    for label in labels:
        header += f"  {label:>6}"
    print(header)

    for i, label in enumerate(labels):
        row = f"  {label:>6}"
        for j in range(len(labels)):
            row += f"  {cm[i][j]:>6}"
        print(row)


def print_comparison(result: EvaluationResult) -> None:
    """Print train vs test comparison."""
    print("\n" + "=" * 60)
    print("TRAIN vs TEST COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Train':>12} {'Test':>12} {'Delta':>10}")
    print("-" * 60)

    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc"]:
        train_val = result.train_metrics.get(metric, 0)
        test_val = result.test_metrics.get(metric, 0)
        delta = test_val - train_val
        print(f"{metric:<25} {train_val:>12.4f} {test_val:>12.4f} {delta:>+10.4f}")


def print_results(result: EvaluationResult) -> None:
    """Print complete evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {result.model_name.upper()}")
    print("=" * 70)

    print(f"\nSamples: Train={result.train_samples}, Test={result.test_samples}")

    print("\n--- Label Distribution ---")
    print("Train:")
    for label, count in result.label_distribution_train["counts"].items():
        pct = result.label_distribution_train["percentages"][label]
        print(f"  Class {label}: {count:>6} ({pct:>5.1f}%)")
    print("Test:")
    for label, count in result.label_distribution_test["counts"].items():
        pct = result.label_distribution_test["percentages"][label]
        print(f"  Class {label}: {count:>6} ({pct:>5.1f}%)")

    print_metrics(result.train_metrics, "TRAIN METRICS")
    print_confusion_matrix(
        result.train_metrics["confusion_matrix"],
        result.train_metrics["class_labels"]
    )

    print_metrics(result.test_metrics, "TEST METRICS")
    print_confusion_matrix(
        result.test_metrics["confusion_matrix"],
        result.test_metrics["class_labels"]
    )

    print_comparison(result)
    print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================


def select_model() -> str:
    """Interactive model selection."""
    models = list(MODEL_REGISTRY.keys())
    trained = get_trained_models()

    print("\n" + "=" * 60)
    print("LABEL PRIMAIRE - EVALUATION")
    print("=" * 60)
    print("\nEvalue le modele primaire seul (avant meta-model).")
    print("\nModeles disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        status = "[entraine]" if model in trained else "[non entraine]"
        info = MODEL_REGISTRY[model]
        dataset_type = info["dataset"]
        print(f"  {i}. {model:<15} ({dataset_type}) {status}")

    print("-" * 40)

    if not trained:
        print("\nAucun modele entraine. Lancer d'abord:")
        print("  python -m src.labelling.label_primaire.train")
        sys.exit(1)

    while True:
        try:
            choice = input("\nChoisir le modele (numero ou nom): ").strip()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                else:
                    print(f"Numero invalide. Choisir entre 1 et {len(models)}")
                    continue
            elif choice.lower() in models:
                selected = choice.lower()
            else:
                print(f"Modele inconnu: {choice}")
                continue

            if selected not in trained:
                print(f"Le modele '{selected}' n'a pas ete entraine.")
                print("Lancer d'abord: python -m src.labelling.label_primaire.train")
                continue

            return selected

        except KeyboardInterrupt:
            print("\nAnnule.")
            sys.exit(0)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model_name = select_model()
    print(f"\nModele selectionne: {model_name}")

    confirm = input("\nLancer l'evaluation? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    result = evaluate_model(model_name)
    print_results(result)

    # Save results
    output_dir = LABEL_PRIMAIRE_EVAL_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{model_name}_evaluation.json"
    result.save(results_path)

    print(f"\nResultats sauvegardes: {results_path}")


if __name__ == "__main__":
    main()
