"""Tests for src/evaluation/evaluator.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    evaluate_model,
    quick_evaluate,
    save_evaluation_results,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, name: str = "MockModel", prediction_value: float = 0.0):
        self.name = name
        self.is_fitted = True
        self.prediction_value = prediction_value

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return constant predictions."""
        return np.full(len(X), self.prediction_value)

    def get_params(self) -> dict[str, Any]:
        """Return mock params."""
        return {"prediction_value": self.prediction_value}


class TestEvaluationConfig:
    """Test cases for EvaluationConfig class."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = EvaluationConfig()

        assert config.metrics is None
        assert config.compute_residual_diagnostics is True
        assert config.compute_mz_regression is False
        assert config.verbose == 1
        assert config.output_dir is None

    def test_custom_metrics(self):
        """Should accept custom metrics."""
        config = EvaluationConfig(metrics=["mse", "rmse"])

        assert config.metrics == ["mse", "rmse"]

    def test_raises_for_invalid_metric(self):
        """Should raise ValueError for invalid metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            EvaluationConfig(metrics=["invalid_metric"])


class TestEvaluationResult:
    """Test cases for EvaluationResult class."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = EvaluationResult(
            model_name="TestModel",
            metrics={"mse": 0.1, "rmse": 0.316},
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1, 2, 3]),
            residuals=np.array([0, 0, 0]),
            residual_diagnostics=None,
            mz_r2=None,
            mz_alpha=None,
            mz_beta=None,
            evaluation_time=1.0,
            n_samples=3,
            model_params={"param": 1},
        )

        d = result.to_dict()

        assert d["model_name"] == "TestModel"
        assert d["metrics"]["mse"] == 0.1
        assert d["n_samples"] == 3

    def test_summary(self):
        """Should generate text summary."""
        result = EvaluationResult(
            model_name="TestModel",
            metrics={"mse": 0.1, "rmse": 0.316},
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1, 2, 3]),
            residuals=np.array([0, 0, 0]),
            residual_diagnostics=None,
            mz_r2=None,
            mz_alpha=None,
            mz_beta=None,
            evaluation_time=1.0,
            n_samples=3,
            model_params={},
        )

        summary = result.summary()

        assert "TestModel" in summary
        assert "mse" in summary
        assert "Samples: 3" in summary


class TestEvaluator:
    """Test cases for Evaluator class."""

    def test_evaluate_model(self):
        """Should evaluate model and return result."""
        model = MockModel(prediction_value=2.0)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        evaluator = Evaluator(EvaluationConfig(verbose=0))
        result = evaluator.evaluate(model, X, y)

        assert result.model_name == "MockModel"
        assert result.n_samples == 5
        assert "mse" in result.metrics

    def test_computes_residuals(self):
        """Should compute residuals correctly."""
        model = MockModel(prediction_value=0.0)
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])

        evaluator = Evaluator(EvaluationConfig(verbose=0))
        result = evaluator.evaluate(model, X, y)

        # Residuals = actuals - predictions = [1,2,3] - [0,0,0] = [1,2,3]
        np.testing.assert_array_equal(result.residuals, [1, 2, 3])

    def test_computes_specified_metrics(self):
        """Should compute only specified metrics."""
        model = MockModel(prediction_value=2.0)
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])

        config = EvaluationConfig(metrics=["mse", "mae"], verbose=0)
        evaluator = Evaluator(config)
        result = evaluator.evaluate(model, X, y)

        assert "mse" in result.metrics
        assert "mae" in result.metrics

    def test_computes_mz_regression(self):
        """Should compute Mincer-Zarnowitz regression when requested."""
        model = MockModel(prediction_value=2.5)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        config = EvaluationConfig(compute_mz_regression=True, verbose=0)
        evaluator = Evaluator(config)
        result = evaluator.evaluate(model, X, y)

        assert result.mz_r2 is not None
        assert result.mz_alpha is not None
        assert result.mz_beta is not None

    def test_computes_residual_diagnostics(self):
        """Should compute residual diagnostics by default."""
        model = MockModel(prediction_value=2.0)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        config = EvaluationConfig(verbose=0)
        evaluator = Evaluator(config)
        result = evaluator.evaluate(model, X, y)

        assert result.residual_diagnostics is not None
        assert hasattr(result.residual_diagnostics, "mean")

    def test_accepts_dataframe(self):
        """Should accept pandas DataFrame."""
        model = MockModel(prediction_value=2.0)
        X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
        y = pd.Series([1, 2, 3, 4, 5])

        evaluator = Evaluator(EvaluationConfig(verbose=0))
        result = evaluator.evaluate(model, X, y)

        assert result.n_samples == 5


class TestEvaluateModel:
    """Test cases for evaluate_model convenience function."""

    def test_evaluates_model(self):
        """Should evaluate model correctly."""
        model = MockModel(prediction_value=2.0)
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])

        result = evaluate_model(model, X, y, verbose=False)

        assert result.model_name == "MockModel"
        assert "mse" in result.metrics

    def test_custom_metrics(self):
        """Should use custom metrics."""
        model = MockModel(prediction_value=2.0)
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])

        result = evaluate_model(model, X, y, metrics=["mae"], verbose=False)

        assert "mae" in result.metrics


class TestQuickEvaluate:
    """Test cases for quick_evaluate convenience function."""

    def test_returns_metrics_dict(self):
        """Should return only metrics dictionary."""
        model = MockModel(prediction_value=2.0)
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])

        result = quick_evaluate(model, X, y)

        assert isinstance(result, dict)
        assert "mse" in result


class TestSaveEvaluationResults:
    """Test cases for save_evaluation_results function."""

    def test_saves_to_file(self):
        """Should save results to JSON file."""
        result = EvaluationResult(
            model_name="TestModel",
            metrics={"mse": 0.1},
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1, 2, 3]),
            residuals=np.array([0, 0, 0]),
            residual_diagnostics=None,
            mz_r2=None,
            mz_alpha=None,
            mz_beta=None,
            evaluation_time=1.0,
            n_samples=3,
            model_params={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            save_evaluation_results(result, path)

            assert path.exists()

    def test_includes_predictions_when_requested(self):
        """Should include predictions when requested."""
        import json

        result = EvaluationResult(
            model_name="TestModel",
            metrics={"mse": 0.1},
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1, 2, 3]),
            residuals=np.array([0, 0, 0]),
            residual_diagnostics=None,
            mz_r2=None,
            mz_alpha=None,
            mz_beta=None,
            evaluation_time=1.0,
            n_samples=3,
            model_params={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            save_evaluation_results(result, path, include_predictions=True)

            with open(path) as f:
                data = json.load(f)

            assert "predictions" in data
            assert "actuals" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
