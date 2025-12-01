"""
Tests for eval.py (Evaluation Module)
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.labelling.label_primaire.eval import (
    evaluate_model,
    compute_metrics,
    ClassificationMetrics,
    EvaluationResult,
)

# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def evaluation_data():
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples, 2)  # Probabilities for class 0 and 1

    # Normalize probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    return y_true, y_pred, y_prob

# =============================================================================
# EVALUATION METRICS TESTS
# =============================================================================

def test_compute_metrics(evaluation_data):
    y_true, y_pred, y_prob = evaluation_data

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert isinstance(metrics, ClassificationMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.f1_macro <= 1
    assert -1 <= metrics.mcc <= 1
    assert metrics.auc_roc_macro is not None
    assert metrics.log_loss_value is not None

    d = metrics.to_dict()
    assert isinstance(d, dict)
    assert d["accuracy"] == metrics.accuracy


def test_compute_metrics_no_proba(evaluation_data):
    y_true, y_pred, _ = evaluation_data

    metrics = compute_metrics(y_true, y_pred, None)
    assert metrics.auc_roc_macro is None
    assert metrics.log_loss_value is None


def test_compute_metrics_multiclass():
    """Test compute_metrics with 3 classes (triple-barrier labels)."""
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.choice([-1, 0, 1], n_samples)
    y_pred = np.random.choice([-1, 0, 1], n_samples)
    y_prob = np.random.rand(n_samples, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert isinstance(metrics, ClassificationMetrics)
    assert len(metrics.class_labels) == 3
    assert len(metrics.f1_per_class) == 3
    assert metrics.n_samples == n_samples


# =============================================================================
# EVALUATION RESULT CLASS
# =============================================================================

def test_evaluation_result_serialization(tmp_path):
    res = EvaluationResult(
        model_name="test",
        train_metrics={"acc": 0.8},
        test_metrics={"acc": 0.7},
        train_samples=100,
        test_samples=50,
        label_distribution_train={"counts": {"0": 50, "1": 50}},
        label_distribution_test={"counts": {"0": 25, "1": 25}},
    )

    d = res.to_dict()
    assert d["model_name"] == "test"
    assert d["train_samples"] == 100
    assert d["test_samples"] == 50
    assert "timestamp" in d

    p = tmp_path / "result.json"
    res.save(p)
    assert p.exists()


def test_classification_metrics_to_dict(evaluation_data):
    """Test that ClassificationMetrics.to_dict() returns all fields."""
    y_true, y_pred, y_prob = evaluation_data
    metrics = compute_metrics(y_true, y_pred, y_prob)

    d = metrics.to_dict()

    expected_keys = [
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "f1_per_class", "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted", "mcc", "cohen_kappa",
        "auc_roc_macro", "log_loss", "confusion_matrix", "class_labels", "n_samples"
    ]

    for key in expected_keys:
        assert key in d, f"Missing key: {key}"
