"""
Unit tests for label_primaire/eval.py

Tests the evaluation module with synthetic data.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.labelling.label_primaire.eval import (
    ClassificationMetrics,
    EvaluationResult,
    compute_classification_metrics,
    extract_probabilities,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Perfect predictions for testing."""
    y_true = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def random_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Random predictions for testing."""
    np.random.seed(42)
    y_true = np.random.choice([-1, 0, 1], size=100)
    y_pred = np.random.choice([-1, 0, 1], size=100)
    return y_true, y_pred


@pytest.fixture
def imbalanced_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Imbalanced class distribution for testing."""
    # Majority class 0, minority class -1 and 1
    y_true = np.array([0] * 80 + [-1] * 10 + [1] * 10)
    # Model predicts mostly 0
    y_pred = np.array([0] * 85 + [-1] * 8 + [1] * 7)
    return y_true, y_pred


@pytest.fixture
def predictions_with_proba() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions with probability estimates."""
    np.random.seed(42)
    n = 50
    y_true = np.random.choice([-1, 0, 1], size=n)

    # Generate probabilities that are somewhat aligned with true labels
    y_proba = np.random.dirichlet([1, 1, 1], size=n)

    # Make probabilities more aligned with true labels
    for i, label in enumerate(y_true):
        class_idx = label + 1  # -1 -> 0, 0 -> 1, 1 -> 2
        y_proba[i, class_idx] += 0.3
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_pred = np.array([-1, 0, 1])[y_proba.argmax(axis=1)]

    return y_true, y_pred, y_proba


# =============================================================================
# TESTS - compute_classification_metrics
# =============================================================================


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics function."""

    def test_perfect_predictions(self, perfect_predictions):
        """Test metrics with perfect predictions."""
        y_true, y_pred = perfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.balanced_accuracy == 1.0
        assert metrics.f1_macro == 1.0
        assert metrics.f1_weighted == 1.0
        assert metrics.mcc == 1.0
        assert metrics.cohen_kappa == 1.0

    def test_random_predictions(self, random_predictions):
        """Test metrics with random predictions."""
        y_true, y_pred = random_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        # Random predictions should have low metrics
        assert 0.2 <= metrics.accuracy <= 0.5
        assert 0.2 <= metrics.balanced_accuracy <= 0.5
        assert metrics.mcc < 0.3  # MCC should be near 0 for random

    def test_imbalanced_predictions(self, imbalanced_predictions):
        """Test metrics handle imbalanced classes correctly."""
        y_true, y_pred = imbalanced_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        # Accuracy can be high due to class imbalance
        assert metrics.accuracy > 0.7

        # But balanced accuracy should be lower
        assert metrics.balanced_accuracy < metrics.accuracy

        # MCC should account for imbalance
        assert -1 <= metrics.mcc <= 1

    def test_with_probabilities(self, predictions_with_proba):
        """Test probabilistic metrics are computed."""
        y_true, y_pred, y_proba = predictions_with_proba
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        # AUC-ROC should be computed
        assert metrics.auc_roc_macro is not None
        assert 0 <= metrics.auc_roc_macro <= 1

        assert metrics.auc_roc_weighted is not None
        assert 0 <= metrics.auc_roc_weighted <= 1

        # Log loss should be computed
        assert metrics.log_loss_value is not None
        assert metrics.log_loss_value > 0

    def test_per_class_metrics(self, perfect_predictions):
        """Test per-class metrics are computed."""
        y_true, y_pred = perfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        # Should have 3 classes
        assert len(metrics.f1_per_class) == 3
        assert len(metrics.precision_per_class) == 3
        assert len(metrics.recall_per_class) == 3

        # All per-class F1 should be 1.0 for perfect predictions
        for f1 in metrics.f1_per_class.values():
            assert f1 == 1.0

    def test_confusion_matrix(self, perfect_predictions):
        """Test confusion matrix is computed correctly."""
        y_true, y_pred = perfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        # Should be a 3x3 matrix
        assert len(metrics.confusion_matrix) == 3
        assert all(len(row) == 3 for row in metrics.confusion_matrix)

        # For perfect predictions, diagonal should be non-zero, off-diagonal should be 0
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert metrics.confusion_matrix[i][j] > 0
                else:
                    assert metrics.confusion_matrix[i][j] == 0

    def test_sample_count(self, random_predictions):
        """Test sample count is correct."""
        y_true, y_pred = random_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics.n_samples == len(y_true)
        assert metrics.n_classes == 3


# =============================================================================
# TESTS - ClassificationMetrics
# =============================================================================


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""

    def test_to_dict(self, perfect_predictions):
        """Test conversion to dictionary."""
        y_true, y_pred = perfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert "accuracy" in d
        assert "f1_macro" in d
        assert "mcc" in d
        assert "confusion_matrix" in d

    def test_summary(self, perfect_predictions):
        """Test summary method returns key metrics."""
        y_true, y_pred = perfect_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        summary = metrics.summary()

        assert isinstance(summary, dict)
        assert "accuracy" in summary
        assert "balanced_accuracy" in summary
        assert "f1_macro" in summary
        assert "mcc" in summary

        # Summary should be smaller than full dict
        assert len(summary) < len(metrics.to_dict())


# =============================================================================
# TESTS - EvaluationResult
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            model_name="test_model",
            train_metrics={"accuracy": 0.8, "f1_macro": 0.75},
            test_metrics={"accuracy": 0.75, "f1_macro": 0.70},
            train_samples=1000,
            test_samples=200,
            label_distribution_train={"counts": {"-1": 300, "0": 400, "1": 300}},
            label_distribution_test={"counts": {"-1": 60, "0": 80, "1": 60}},
            model_path="/path/to/model.joblib",
            oof_predictions_path="/path/to/oof.parquet",
            test_predictions_path="/path/to/test.parquet",
        )

        d = result.to_dict()

        assert d["model_name"] == "test_model"
        assert d["train_samples"] == 1000
        assert d["train_metrics"]["accuracy"] == 0.8

    def test_comparison_summary(self):
        """Test comparison summary method."""
        result = EvaluationResult(
            model_name="test_model",
            train_metrics={
                "accuracy": 0.8,
                "balanced_accuracy": 0.78,
                "f1_macro": 0.75,
                "f1_weighted": 0.77,
                "mcc": 0.65,
            },
            test_metrics={
                "accuracy": 0.75,
                "balanced_accuracy": 0.73,
                "f1_macro": 0.70,
                "f1_weighted": 0.72,
                "mcc": 0.55,
            },
            train_samples=1000,
            test_samples=200,
            label_distribution_train={},
            label_distribution_test={},
            model_path="",
            oof_predictions_path="",
            test_predictions_path="",
        )

        summary = result.comparison_summary()

        assert "train" in summary
        assert "test" in summary
        assert summary["train"]["accuracy"] == 0.8
        assert summary["test"]["accuracy"] == 0.75

    def test_save(self, tmp_path):
        """Test saving results to JSON."""
        result = EvaluationResult(
            model_name="test_model",
            train_metrics={"accuracy": 0.8},
            test_metrics={"accuracy": 0.75},
            train_samples=1000,
            test_samples=200,
            label_distribution_train={},
            label_distribution_test={},
            model_path="",
            oof_predictions_path="",
            test_predictions_path="",
        )

        output_path = tmp_path / "eval_results.json"
        result.save(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["model_name"] == "test_model"
        assert loaded["train_samples"] == 1000


# =============================================================================
# TESTS - extract_probabilities
# =============================================================================


class TestExtractProbabilities:
    """Tests for extract_probabilities function."""

    def test_with_proba_columns(self):
        """Test extraction with probability columns."""
        df = pd.DataFrame({
            "y_true": [0, 1, -1],
            "y_pred": [0, 1, -1],
            "proba_class_0": [0.8, 0.1, 0.1],
            "proba_class_1": [0.1, 0.8, 0.1],
            "proba_class_2": [0.1, 0.1, 0.8],
        })

        proba = extract_probabilities(df)

        assert proba is not None
        assert proba.shape == (3, 3)
        # Columns should be sorted by class index
        np.testing.assert_array_almost_equal(proba[0], [0.8, 0.1, 0.1])

    def test_without_proba_columns(self):
        """Test extraction without probability columns."""
        df = pd.DataFrame({
            "y_true": [0, 1, -1],
            "y_pred": [0, 1, -1],
        })

        proba = extract_probabilities(df)

        assert proba is None


# =============================================================================
# TESTS - Metric Properties
# =============================================================================


class TestMetricProperties:
    """Tests for mathematical properties of metrics."""

    def test_mcc_bounds(self):
        """Test MCC is always in [-1, 1]."""
        np.random.seed(42)
        for _ in range(10):
            y_true = np.random.choice([-1, 0, 1], size=100)
            y_pred = np.random.choice([-1, 0, 1], size=100)
            metrics = compute_classification_metrics(y_true, y_pred)
            assert -1 <= metrics.mcc <= 1

    def test_kappa_bounds(self):
        """Test Cohen's Kappa is always in [-1, 1]."""
        np.random.seed(42)
        for _ in range(10):
            y_true = np.random.choice([-1, 0, 1], size=100)
            y_pred = np.random.choice([-1, 0, 1], size=100)
            metrics = compute_classification_metrics(y_true, y_pred)
            assert -1 <= metrics.cohen_kappa <= 1

    def test_f1_bounds(self):
        """Test F1 scores are always in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            y_true = np.random.choice([-1, 0, 1], size=100)
            y_pred = np.random.choice([-1, 0, 1], size=100)
            metrics = compute_classification_metrics(y_true, y_pred)
            assert 0 <= metrics.f1_macro <= 1
            assert 0 <= metrics.f1_weighted <= 1

    def test_auc_bounds(self):
        """Test AUC-ROC is always in [0, 1]."""
        np.random.seed(42)
        y_true = np.random.choice([-1, 0, 1], size=100)
        y_proba = np.random.dirichlet([1, 1, 1], size=100)
        y_pred = np.array([-1, 0, 1])[y_proba.argmax(axis=1)]

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        if metrics.auc_roc_macro is not None:
            assert 0 <= metrics.auc_roc_macro <= 1
        if metrics.auc_roc_weighted is not None:
            assert 0 <= metrics.auc_roc_weighted <= 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests for the evaluation module."""

    def test_full_evaluation_flow(self, tmp_path):
        """Test complete evaluation flow with synthetic data."""
        np.random.seed(42)

        # Create synthetic OOF predictions
        n_train = 1000
        y_train_true = np.random.choice([-1, 0, 1], size=n_train)
        y_train_pred = y_train_true.copy()
        # Add some noise to predictions
        noise_idx = np.random.choice(n_train, size=100, replace=False)
        y_train_pred[noise_idx] = np.random.choice([-1, 0, 1], size=100)

        # Create synthetic test predictions
        n_test = 200
        y_test_true = np.random.choice([-1, 0, 1], size=n_test)
        y_test_pred = y_test_true.copy()
        noise_idx = np.random.choice(n_test, size=30, replace=False)
        y_test_pred[noise_idx] = np.random.choice([-1, 0, 1], size=30)

        # Compute metrics
        train_metrics = compute_classification_metrics(y_train_true, y_train_pred)
        test_metrics = compute_classification_metrics(y_test_true, y_test_pred)

        # Create result
        result = EvaluationResult(
            model_name="synthetic_test",
            train_metrics=train_metrics.to_dict(),
            test_metrics=test_metrics.to_dict(),
            train_samples=n_train,
            test_samples=n_test,
            label_distribution_train={
                "counts": pd.Series(y_train_true).value_counts().to_dict()
            },
            label_distribution_test={
                "counts": pd.Series(y_test_true).value_counts().to_dict()
            },
            model_path=str(tmp_path / "model.joblib"),
            oof_predictions_path=str(tmp_path / "oof.parquet"),
            test_predictions_path=str(tmp_path / "test.parquet"),
        )

        # Save and reload
        output_path = tmp_path / "eval_results.json"
        result.save(output_path)

        with open(output_path) as f:
            loaded = json.load(f)

        # Verify key metrics are present
        assert loaded["train_metrics"]["accuracy"] > 0.8
        assert loaded["test_metrics"]["accuracy"] > 0.8
        assert "mcc" in loaded["train_metrics"]
        assert "f1_macro" in loaded["test_metrics"]
