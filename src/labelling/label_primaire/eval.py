"""
Primary Model Evaluation Module.

Comprehensive evaluation of trained primary models with metrics suitable
for imbalanced classification (triple-barrier labeling).

Metrics computed:
- Accuracy
- Balanced Accuracy (macro-averaged recall)
- F1-Score (macro, weighted, per-class)
- Precision (macro, weighted, per-class)
- Recall (macro, weighted, per-class)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- AUC-ROC (One-vs-Rest, macro)
- Log Loss
- Confusion Matrix
- Classification Report

This module allows comparison between primary and meta models.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.labelling.label_primaire.utils import MODEL_REGISTRY
from src.path import LABEL_PRIMAIRE_TRAIN_DIR, LABEL_PRIMAIRE_EVAL_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE METRICS
# =============================================================================


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics for model evaluation."""

    # Basic metrics
    accuracy: float
    balanced_accuracy: float

    # F1 scores
    f1_macro: float
    f1_weighted: float
    f1_per_class: Dict[str, float]

    # Precision
    precision_macro: float
    precision_weighted: float
    precision_per_class: Dict[str, float]

    # Recall
    recall_macro: float
    recall_weighted: float
    recall_per_class: Dict[str, float]

    # Correlation metrics
    mcc: float  # Matthews Correlation Coefficient
    cohen_kappa: float

    # Probabilistic metrics (optional)
    auc_roc_macro: float | None
    auc_roc_weighted: float | None
    log_loss_value: float | None

    # Confusion matrix
    confusion_matrix: List[List[int]]
    class_labels: List[str]

    # Sample counts
    n_samples: int
    n_classes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "f1_per_class": self.f1_per_class,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "precision_per_class": self.precision_per_class,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "recall_per_class": self.recall_per_class,
            "mcc": self.mcc,
            "cohen_kappa": self.cohen_kappa,
            "auc_roc_macro": self.auc_roc_macro,
            "auc_roc_weighted": self.auc_roc_weighted,
            "log_loss": self.log_loss_value,
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
        }

    def summary(self) -> Dict[str, float]:
        """Return key metrics for quick comparison."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "mcc": self.mcc,
            "cohen_kappa": self.cohen_kappa,
            "auc_roc_macro": self.auc_roc_macro,
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_labels: List[str] | None = None,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities (n_samples, n_classes).
    class_labels : List[str], optional
        Names for each class. If None, inferred from y_true.

    Returns
    -------
    ClassificationMetrics
        All computed metrics.
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    if class_labels is None:
        class_labels = [str(c) for c in sorted(classes)]

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class_arr = f1_score(y_true, y_pred, average=None, zero_division=0, labels=sorted(classes))
    f1_per_class = {str(c): float(f) for c, f in zip(sorted(classes), f1_per_class_arr)}

    # Precision
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_per_class_arr = precision_score(y_true, y_pred, average=None, zero_division=0, labels=sorted(classes))
    precision_per_class = {str(c): float(p) for c, p in zip(sorted(classes), precision_per_class_arr)}

    # Recall
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_per_class_arr = recall_score(y_true, y_pred, average=None, zero_division=0, labels=sorted(classes))
    recall_per_class = {str(c): float(r) for c, r in zip(sorted(classes), recall_per_class_arr)}

    # Correlation metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Probabilistic metrics
    auc_roc_macro = None
    auc_roc_weighted = None
    log_loss_val = None

    if y_proba is not None and n_classes >= 2:
        try:
            # Ensure probability matrix has correct shape
            if y_proba.shape[1] == n_classes:
                auc_roc_macro = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                auc_roc_weighted = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
                log_loss_val = log_loss(y_true, y_proba, labels=sorted(classes))
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not compute probabilistic metrics: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted(classes))
    cm_list = cm.tolist()

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        f1_per_class=f1_per_class,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        precision_per_class=precision_per_class,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        recall_per_class=recall_per_class,
        mcc=mcc,
        cohen_kappa=kappa,
        auc_roc_macro=auc_roc_macro,
        auc_roc_weighted=auc_roc_weighted,
        log_loss_value=log_loss_val,
        confusion_matrix=cm_list,
        class_labels=class_labels,
        n_samples=len(y_true),
        n_classes=n_classes,
    )


# =============================================================================
# EVALUATION RESULT
# =============================================================================


@dataclass
class EvaluationResult:
    """Complete evaluation result for a trained model."""

    model_name: str
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    train_samples: int
    test_samples: int
    label_distribution_train: Dict[str, Any]
    label_distribution_test: Dict[str, Any]
    model_path: str
    oof_predictions_path: str
    test_predictions_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "label_distribution_train": self.label_distribution_train,
            "label_distribution_test": self.label_distribution_test,
            "model_path": self.model_path,
            "oof_predictions_path": self.oof_predictions_path,
            "test_predictions_path": self.test_predictions_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {path}")

    def comparison_summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary for train/test comparison."""
        return {
            "train": {
                "accuracy": self.train_metrics.get("accuracy", 0),
                "balanced_accuracy": self.train_metrics.get("balanced_accuracy", 0),
                "f1_macro": self.train_metrics.get("f1_macro", 0),
                "f1_weighted": self.train_metrics.get("f1_weighted", 0),
                "mcc": self.train_metrics.get("mcc", 0),
            },
            "test": {
                "accuracy": self.test_metrics.get("accuracy", 0),
                "balanced_accuracy": self.test_metrics.get("balanced_accuracy", 0),
                "f1_macro": self.test_metrics.get("f1_macro", 0),
                "f1_weighted": self.test_metrics.get("f1_weighted", 0),
                "mcc": self.test_metrics.get("mcc", 0),
            },
        }


# =============================================================================
# DATA LOADING
# =============================================================================


def get_trained_models() -> List[str]:
    """Get list of models that have been trained."""
    trained = []
    for model_name in MODEL_REGISTRY.keys():
        model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
        model_file = model_dir / f"{model_name}_final_model.joblib"
        if model_file.exists():
            trained.append(model_name)
    return trained


def load_predictions(model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OOF and test predictions for a trained model.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (oof_predictions, test_predictions)
    """
    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name

    oof_path = model_dir / f"{model_name}_oof_predictions.parquet"
    test_path = model_dir / f"{model_name}_test_predictions.parquet"

    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test predictions not found: {test_path}")

    oof_df = pd.read_parquet(oof_path)
    test_df = pd.read_parquet(test_path)

    return oof_df, test_df


def load_trained_model(model_name: str):
    """Load a trained model from disk."""
    from src.model.base import BaseModel

    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
    model_path = model_dir / f"{model_name}_final_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run training first: python -m src.labelling.label_primaire.main"
        )

    return BaseModel.load(model_path)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def extract_probabilities(df: pd.DataFrame) -> np.ndarray | None:
    """Extract probability columns from predictions DataFrame."""
    proba_cols = [c for c in df.columns if c.startswith("proba_class_")]

    if not proba_cols:
        return None

    # Sort by class index
    proba_cols = sorted(proba_cols, key=lambda x: int(x.split("_")[-1]))
    return df[proba_cols].values


def evaluate_model(model_name: str) -> EvaluationResult:
    """
    Evaluate a trained primary model.

    Parameters
    ----------
    model_name : str
        Name of the model to evaluate.

    Returns
    -------
    EvaluationResult
        Complete evaluation results.
    """
    logger.info(f"{'='*60}")
    logger.info(f"EVALUATION: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Load predictions
    logger.info("Loading predictions...")
    oof_df, test_df = load_predictions(model_name)

    # Get model path
    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
    model_path = model_dir / f"{model_name}_final_model.joblib"
    oof_path = model_dir / f"{model_name}_oof_predictions.parquet"
    test_path = model_dir / f"{model_name}_test_predictions.parquet"

    # Extract data
    # OOF (train) - filter out NaN predictions
    oof_valid_mask = ~oof_df["y_pred"].isna()
    y_train_true = oof_df.loc[oof_valid_mask, "y_true"].values.astype(int)
    y_train_pred = oof_df.loc[oof_valid_mask, "y_pred"].values.astype(int)
    y_train_proba = extract_probabilities(oof_df.loc[oof_valid_mask])

    # Test
    y_test_true = test_df["y_true"].values.astype(int)
    y_test_pred = test_df["y_pred"].values.astype(int)
    y_test_proba = extract_probabilities(test_df)

    # Compute metrics
    logger.info("Computing train (OOF) metrics...")
    train_metrics = compute_classification_metrics(
        y_train_true, y_train_pred, y_train_proba
    )

    logger.info("Computing test metrics...")
    test_metrics = compute_classification_metrics(
        y_test_true, y_test_pred, y_test_proba
    )

    # Label distributions
    train_counts = pd.Series(y_train_true).value_counts().to_dict()
    test_counts = pd.Series(y_test_true).value_counts().to_dict()

    train_dist = {
        "total": len(y_train_true),
        "counts": {str(k): int(v) for k, v in train_counts.items()},
        "percentages": {str(k): v / len(y_train_true) * 100 for k, v in train_counts.items()},
    }
    test_dist = {
        "total": len(y_test_true),
        "counts": {str(k): int(v) for k, v in test_counts.items()},
        "percentages": {str(k): v / len(y_test_true) * 100 for k, v in test_counts.items()},
    }

    # Build result
    result = EvaluationResult(
        model_name=model_name,
        train_metrics=train_metrics.to_dict(),
        test_metrics=test_metrics.to_dict(),
        train_samples=len(y_train_true),
        test_samples=len(y_test_true),
        label_distribution_train=train_dist,
        label_distribution_test=test_dist,
        model_path=str(model_path),
        oof_predictions_path=str(oof_path),
        test_predictions_path=str(test_path),
    )

    return result


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def print_confusion_matrix(cm: List[List[int]], labels: List[str]) -> None:
    """Print formatted confusion matrix."""
    print("\nConfusion Matrix:")
    print("-" * 40)

    # Header
    header = "True\\Pred"
    for label in labels:
        header += f"  {label:>6}"
    print(header)

    # Rows
    for i, label in enumerate(labels):
        row = f"  {label:>6}"
        for j in range(len(labels)):
            row += f"  {cm[i][j]:>6}"
        print(row)


def print_metrics_table(metrics: Dict[str, Any], title: str) -> None:
    """Print metrics in a formatted table."""
    print(f"\n{title}")
    print("=" * 50)

    # Key metrics
    key_metrics = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("F1 Macro", "f1_macro"),
        ("F1 Weighted", "f1_weighted"),
        ("Precision Macro", "precision_macro"),
        ("Precision Weighted", "precision_weighted"),
        ("Recall Macro", "recall_macro"),
        ("Recall Weighted", "recall_weighted"),
        ("MCC", "mcc"),
        ("Cohen's Kappa", "cohen_kappa"),
        ("AUC-ROC Macro", "auc_roc_macro"),
        ("AUC-ROC Weighted", "auc_roc_weighted"),
        ("Log Loss", "log_loss"),
    ]

    for name, key in key_metrics:
        value = metrics.get(key)
        if value is not None:
            print(f"  {name:<22}: {value:>8.4f}")
        else:
            print(f"  {name:<22}: {'N/A':>8}")

    # Per-class metrics
    if "f1_per_class" in metrics:
        print("\n  Per-Class F1:")
        for cls, f1 in metrics["f1_per_class"].items():
            print(f"    Class {cls}: {f1:.4f}")


def print_comparison_table(result: EvaluationResult) -> None:
    """Print train vs test comparison."""
    print("\n" + "=" * 60)
    print("TRAIN vs TEST COMPARISON")
    print("=" * 60)

    summary = result.comparison_summary()

    print(f"\n{'Metric':<25} {'Train (OOF)':>12} {'Test':>12} {'Delta':>10}")
    print("-" * 60)

    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc"]:
        train_val = summary["train"].get(metric, 0)
        test_val = summary["test"].get(metric, 0)
        delta = test_val - train_val

        # Color coding (positive delta = test better)
        delta_str = f"{delta:+.4f}"

        print(f"{metric:<25} {train_val:>12.4f} {test_val:>12.4f} {delta_str:>10}")


def print_evaluation_results(result: EvaluationResult) -> None:
    """Print complete evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {result.model_name.upper()}")
    print("=" * 70)

    print(f"\nSamples: Train={result.train_samples}, Test={result.test_samples}")

    # Label distributions
    print("\n--- Label Distribution ---")
    print("Train:")
    for label, count in result.label_distribution_train["counts"].items():
        pct = result.label_distribution_train["percentages"][label]
        print(f"  Class {label}: {count:>6} ({pct:>5.1f}%)")
    print("Test:")
    for label, count in result.label_distribution_test["counts"].items():
        pct = result.label_distribution_test["percentages"][label]
        print(f"  Class {label}: {count:>6} ({pct:>5.1f}%)")

    # Train metrics
    print_metrics_table(result.train_metrics, "TRAIN (Out-of-Fold) METRICS")
    print_confusion_matrix(
        result.train_metrics["confusion_matrix"],
        result.train_metrics["class_labels"]
    )

    # Test metrics
    print_metrics_table(result.test_metrics, "TEST METRICS")
    print_confusion_matrix(
        result.test_metrics["confusion_matrix"],
        result.test_metrics["class_labels"]
    )

    # Comparison
    print_comparison_table(result)

    print("\n" + "=" * 70)


# =============================================================================
# INTERACTIVE CLI
# =============================================================================


def select_model() -> str:
    """Interactive model selection for evaluation."""
    models = list(MODEL_REGISTRY.keys())
    trained = get_trained_models()

    print("\n" + "=" * 60)
    print("LABEL PRIMAIRE - EVALUATION")
    print("=" * 60)
    print("\nThis module evaluates trained primary models with comprehensive metrics:")
    print("  - F1 (macro, weighted, per-class)")
    print("  - MCC (Matthews Correlation Coefficient)")
    print("  - Balanced Accuracy, Cohen's Kappa")
    print("  - AUC-ROC, Log Loss")
    print("  - Confusion Matrix")
    print("\nModels disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        status = "[entraine]" if model in trained else "[non entraine]"
        info = MODEL_REGISTRY[model]
        dataset_type = info["dataset"]
        print(f"  {i}. {model:<15} ({dataset_type}) {status}")

    print("-" * 40)

    if not trained:
        print("\nAucun modele entraine. Lancer d'abord l'entrainement:")
        print("  python -m src.labelling.label_primaire.main")
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

            # Check if trained
            if selected not in trained:
                print(f"Le modele '{selected}' n'a pas ete entraine.")
                print("Lancer d'abord: python -m src.labelling.label_primaire.main")
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

    # Interactive model selection
    model_name = select_model()
    print(f"\nModele selectionne: {model_name}")

    confirm = input("\nLancer l'evaluation? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    # Run evaluation
    result = evaluate_model(model_name)

    # Print results
    print_evaluation_results(result)

    # Save results
    output_dir = LABEL_PRIMAIRE_EVAL_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{model_name}_evaluation.json"
    result.save(results_path)

    print(f"\nResultats sauvegardes: {results_path}")


if __name__ == "__main__":
    main()
