"""
Tests for eval.py (Evaluation Module)
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.labelling.label_primaire.eval.logic import (
    evaluate_model,
    compute_metrics,
    ClassificationMetrics,
    EvaluationResult,
    get_features_path,
    get_labeled_features_path,
    get_trained_models,
    load_model,
    load_data,
    print_metrics,
    print_confusion_matrix,
    print_comparison,
    print_results,
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


# =============================================================================
# DATA LOADING TESTS
# =============================================================================


def test_get_features_path(mocker):
    """Test getting features path for different model types."""
    # Mock the path constants
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.DATASET_FEATURES_FINAL_PARQUET",
        Path("/path/to/tree.parquet")
    )
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.DATASET_FEATURES_LINEAR_FINAL_PARQUET",
        Path("/path/to/linear.parquet")
    )
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.DATASET_FEATURES_LSTM_FINAL_PARQUET",
        Path("/path/to/lstm.parquet")
    )

    # Test tree model
    path = get_features_path("lightgbm")
    assert "tree" in str(path)

    # Test linear model
    path = get_features_path("ridge")
    assert "linear" in str(path)

    # Test lstm model
    path = get_features_path("lstm")
    assert "lstm" in str(path)


def test_get_labeled_features_path(mocker):
    """Test getting labeled features path."""
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.DATASET_FEATURES_FINAL_PARQUET",
        Path("/path/to/features.parquet")
    )

    path = get_labeled_features_path("lightgbm")
    assert "lightgbm" in str(path)
    assert path.suffix == ".parquet"


def test_get_trained_models_empty(tmp_path, mocker):
    """Test when no models are trained."""
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.LABEL_PRIMAIRE_TRAIN_DIR",
        tmp_path
    )

    trained = get_trained_models()
    assert trained == []


def test_get_trained_models(tmp_path, mocker):
    """Test listing trained models."""
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.LABEL_PRIMAIRE_TRAIN_DIR",
        tmp_path
    )

    # Create mock trained model
    model_dir = tmp_path / "lightgbm"
    model_dir.mkdir()
    (model_dir / "lightgbm_model.joblib").touch()

    # Create dir without model file
    (tmp_path / "xgboost").mkdir()

    trained = get_trained_models()
    assert "lightgbm" in trained
    assert "xgboost" not in trained


def test_load_model_not_found(tmp_path, mocker):
    """Test loading when model not found."""
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.LABEL_PRIMAIRE_TRAIN_DIR",
        tmp_path
    )

    with pytest.raises(FileNotFoundError):
        load_model("nonexistent")


def test_load_data_not_found(tmp_path, mocker):
    """Test loading data when file not found."""
    mocker.patch(
        "src.labelling.label_primaire.eval.logic.DATASET_FEATURES_FINAL_PARQUET",
        tmp_path / "nonexistent.parquet"
    )

    with pytest.raises(FileNotFoundError):
        load_data("lightgbm")


def test_load_data_no_label_column(tmp_path, mocker):
    """Test loading data when no label column."""
    # Create parquet without label column
    df = pd.DataFrame({"f1": [1, 2, 3], "split": ["train"] * 3})
    parquet_path = tmp_path / "features_lightgbm.parquet"
    df.to_parquet(parquet_path)

    mocker.patch(
        "src.labelling.label_primaire.eval.logic.DATASET_FEATURES_FINAL_PARQUET",
        tmp_path / "features.parquet"
    )

    # No label column raises ValueError
    with pytest.raises(ValueError, match="No 'label' column"):
        load_data("lightgbm")


# =============================================================================
# DISPLAY TESTS
# =============================================================================


def test_print_metrics(capsys):
    """Test printing metrics."""
    metrics = {
        "accuracy": 0.85,
        "balanced_accuracy": 0.82,
        "f1_macro": 0.80,
        "f1_weighted": 0.81,
        "precision_macro": 0.78,
        "recall_macro": 0.79,
        "mcc": 0.65,
        "cohen_kappa": 0.60,
        "auc_roc_macro": 0.88,
        "log_loss": 0.45,
        "f1_per_class": {"0": 0.75, "1": 0.85},
    }

    print_metrics(metrics, "Test Title")
    captured = capsys.readouterr()

    assert "Test Title" in captured.out
    assert "Accuracy" in captured.out
    assert "0.8500" in captured.out


def test_print_confusion_matrix(capsys):
    """Test printing confusion matrix."""
    cm = [[10, 5], [3, 12]]
    labels = ["0", "1"]

    print_confusion_matrix(cm, labels)
    captured = capsys.readouterr()

    assert "Confusion Matrix" in captured.out
    assert "10" in captured.out
    assert "12" in captured.out


def test_print_comparison(capsys):
    """Test printing comparison."""
    result = EvaluationResult(
        model_name="test",
        train_metrics={
            "accuracy": 0.90,
            "balanced_accuracy": 0.88,
            "f1_macro": 0.85,
            "f1_weighted": 0.86,
            "mcc": 0.70,
        },
        test_metrics={
            "accuracy": 0.85,
            "balanced_accuracy": 0.82,
            "f1_macro": 0.80,
            "f1_weighted": 0.81,
            "mcc": 0.65,
        },
        train_samples=1000,
        test_samples=200,
        label_distribution_train={"counts": {"0": 500, "1": 500}, "percentages": {"0": 50.0, "1": 50.0}},
        label_distribution_test={"counts": {"0": 100, "1": 100}, "percentages": {"0": 50.0, "1": 50.0}},
    )

    print_comparison(result)
    captured = capsys.readouterr()

    assert "TRAIN vs TEST" in captured.out
    assert "Delta" in captured.out


def test_print_results(capsys):
    """Test printing full results."""
    result = EvaluationResult(
        model_name="test_model",
        train_metrics={
            "accuracy": 0.90,
            "balanced_accuracy": 0.88,
            "f1_macro": 0.85,
            "f1_weighted": 0.86,
            "mcc": 0.70,
            "confusion_matrix": [[45, 5], [10, 40]],
            "class_labels": ["0", "1"],
            "f1_per_class": {"0": 0.82, "1": 0.88},
        },
        test_metrics={
            "accuracy": 0.85,
            "balanced_accuracy": 0.82,
            "f1_macro": 0.80,
            "f1_weighted": 0.81,
            "mcc": 0.65,
            "confusion_matrix": [[18, 2], [5, 15]],
            "class_labels": ["0", "1"],
            "f1_per_class": {"0": 0.78, "1": 0.82},
        },
        train_samples=100,
        test_samples=40,
        label_distribution_train={"counts": {"0": 50, "1": 50}, "percentages": {"0": 50.0, "1": 50.0}},
        label_distribution_test={"counts": {"0": 20, "1": 20}, "percentages": {"0": 50.0, "1": 50.0}},
    )

    print_results(result)
    captured = capsys.readouterr()

    assert "EVALUATION RESULTS" in captured.out
    assert "TEST_MODEL" in captured.out
    assert "TRAIN METRICS" in captured.out
    assert "TEST METRICS" in captured.out
