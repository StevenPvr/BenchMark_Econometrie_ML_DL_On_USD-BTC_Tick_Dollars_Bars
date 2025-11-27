"""Main script for Baseline models evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
from typing import cast


import numpy as np
import pandas as pd # type: ignore

from src.evaluation.metrics import (
    accuracy,
    balanced_accuracy,
    f1_macro,
    f1_weighted,
    precision_macro,
    recall_macro,
)
from src.model.baseline.ar1_baseline import AR1Baseline
from src.model.baseline.persistence_baseline import PersistenceBaseline
from src.model.baseline.random_baseline import RandomBaseline
from src.path import DATASET_FEATURES_PARQUET, RESULTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)

# Output directory for baseline results
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge le dataset features et extrait X et y.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features X et target y (labels: -1, 0, 1).
    """
    logger.info("Loading dataset from %s", DATASET_FEATURES_PARQUET)
    df = pd.read_parquet(DATASET_FEATURES_PARQUET)

    # Target column (De Prado triple-barrier labels: -1, 0, 1)
    target_col = "label"
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            "Please ensure labels have been generated using triple-barrier labeling."
        )

    # Features: all columns except target and non-feature columns
    non_feature_cols = ["bar_id", "datetime_close", "log_return", target_col]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = cast(pd.DataFrame, df.loc[:, feature_cols])
    y = cast(pd.Series, df[target_col])

    logger.info("Dataset loaded: %d samples, %d features", len(X), len(feature_cols))
    logger.info("Label distribution: %s", dict(y.value_counts().sort_index()))
    return X, y


def train_test_split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporel (pas de shuffle pour eviter le leakage).

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    test_ratio : float, default=0.2
        Ratio du test set.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    n = len(X)
    split_idx = int(n * (1 - test_ratio))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def evaluate_random_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evalue le modele Random Baseline.

    Returns
    -------
    dict[str, float]
        Metriques d'evaluation (classification).
    """
    model = RandomBaseline(random_state=42)
    model.fit(X_train, y_train)

    params = model.get_distribution_params()
    logger.info("Random Baseline - classes=%s, probs=%s", params["classes"], params["probabilities"])

    y_pred = model.predict(X_test)

    y_true_np = np.asarray(y_test.values)
    metrics = {
        "accuracy": accuracy(y_true_np, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true_np, y_pred),
        "f1_macro": f1_macro(y_true_np, y_pred),
        "f1_weighted": f1_weighted(y_true_np, y_pred),
        "precision_macro": precision_macro(y_true_np, y_pred),
        "recall_macro": recall_macro(y_true_np, y_pred),
    }

    results = {
        "metrics": metrics,
        "distribution": params,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    output_file = BASELINE_RESULTS_DIR / "random_baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Random Baseline results saved to %s", output_file)

    return metrics


def evaluate_persistence_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evalue le modele Persistence Baseline (naive forecast).

    Returns
    -------
    dict[str, float]
        Metriques d'evaluation (classification).
    """
    model = PersistenceBaseline()
    model.fit(X_train, y_train)

    last_val = model.get_last_value()
    logger.info("Persistence Baseline - last_value=%d", last_val)

    # Prediction one-step ahead (utilise les vraies valeurs precedentes)
    y_pred = model.predict_with_actuals(X_test, y_test)

    y_true_np = np.asarray(y_test.values)
    metrics = {
        "accuracy": accuracy(y_true_np, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true_np, y_pred),
        "f1_macro": f1_macro(y_true_np, y_pred),
        "f1_weighted": f1_weighted(y_true_np, y_pred),
        "precision_macro": precision_macro(y_true_np, y_pred),
        "recall_macro": recall_macro(y_true_np, y_pred),
    }

    results = {
        "metrics": metrics,
        "last_train_value": int(last_val),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    output_file = BASELINE_RESULTS_DIR / "persistence_baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Persistence Baseline results saved to %s", output_file)

    return metrics


def evaluate_ar1_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evalue le modele Markov(1) Baseline (classification).

    Returns
    -------
    dict[str, float]
        Metriques d'evaluation (classification).
    """
    model = AR1Baseline()
    model.fit(X_train, y_train)

    transition_matrix = model.get_transition_matrix()
    logger.info("Markov(1) Baseline - Transition matrix:\n%s", transition_matrix)

    # Prediction one-step ahead (utilise les vraies valeurs precedentes)
    y_pred = model.predict_with_actuals(X_test, y_test)

    y_true_np = np.asarray(y_test.values)
    metrics = {
        "accuracy": accuracy(y_true_np, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true_np, y_pred),
        "f1_macro": f1_macro(y_true_np, y_pred),
        "f1_weighted": f1_weighted(y_true_np, y_pred),
        "precision_macro": precision_macro(y_true_np, y_pred),
        "recall_macro": recall_macro(y_true_np, y_pred),
    }

    results = {
        "metrics": metrics,
        "transition_matrix": transition_matrix.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    output_file = BASELINE_RESULTS_DIR / "markov_baseline_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Markov(1) Baseline results saved to %s", output_file)

    return metrics


def main() -> None:
    """Point d'entree principal."""
    logger.info("=" * 60)
    logger.info("Baseline Models Evaluation")
    logger.info("=" * 60)

    # Load data
    X, y = load_data()

    # Temporal split
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_ratio=0.2)
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    # Create output directory
    BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate Random Baseline
    logger.info("-" * 40)
    logger.info("Evaluating Random Baseline...")
    random_metrics = evaluate_random_baseline(X_train, X_test, y_train, y_test)

    # Evaluate Persistence Baseline
    logger.info("-" * 40)
    logger.info("Evaluating Persistence Baseline...")
    persistence_metrics = evaluate_persistence_baseline(X_train, X_test, y_train, y_test)

    # Evaluate Markov(1) Baseline
    logger.info("-" * 40)
    logger.info("Evaluating Markov(1) Baseline...")
    ar1_metrics = evaluate_ar1_baseline(X_train, X_test, y_train, y_test)

    # Display results
    print("\n" + "=" * 60)
    print("BASELINE MODELS - CLASSIFICATION METRICS COMPARISON")
    print("=" * 60)

    def print_metrics(name: str, metrics: dict[str, float]) -> None:
        """Print classification metrics in a formatted way."""
        print(f"\n{name}")
        print("-" * 40)
        print(f"   Accuracy:          {metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"   F1 (macro):        {metrics['f1_macro']:.4f}")
        print(f"   F1 (weighted):     {metrics['f1_weighted']:.4f}")
        print(f"   Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"   Recall (macro):    {metrics['recall_macro']:.4f}")

    print_metrics("1. Random Baseline (stratified random by class distribution)", random_metrics)
    print_metrics("2. Persistence Baseline (y_pred[t] = y[t-1])", persistence_metrics)
    print_metrics("3. Markov(1) Baseline (y_pred[t] = argmax P(y|y[t-1]))", ar1_metrics)

    print("\n" + "=" * 60)

    # Save combined results
    combined_results = {
        "random_baseline": random_metrics,
        "persistence_baseline": persistence_metrics,
        "markov_baseline": ar1_metrics,
    }
    combined_file = BASELINE_RESULTS_DIR / "all_baselines_metrics.json"
    with open(combined_file, "w") as f:
        json.dump(combined_results, f, indent=2)
    logger.info("Combined results saved to %s", combined_file)


if __name__ == "__main__":
    main()
