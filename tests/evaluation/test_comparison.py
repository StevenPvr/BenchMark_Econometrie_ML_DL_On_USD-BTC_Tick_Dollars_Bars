"""Tests for src/evaluation/comparison.py module."""

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

from src.evaluation.comparison import (
    ModelComparisonResult,
    diebold_mariano_test,
    pairwise_dm_tests,
    PairwiseTestResult,
    ModelComparator,
    compare_models,
    rank_models,
    save_comparison_results,
)
from src.evaluation.evaluator import EvaluationResult


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


class TestDieboldMarianoTest:
    """Test cases for diebold_mariano_test function."""

    def test_equal_errors_returns_zero(self):
        """Equal errors should give DM stat near 0."""
        errors1 = np.array([0.1, -0.1, 0.2, -0.2])
        errors2 = np.array([0.1, -0.1, 0.2, -0.2])

        dm_stat, p_value = diebold_mariano_test(errors1, errors2)

        assert pytest.approx(dm_stat, abs=1e-10) == 0.0
        assert p_value == 1.0  # No difference

    def test_smaller_errors_gives_negative_dm(self):
        """Model 1 with smaller errors should give negative DM."""
        # Longer series needed for meaningful test
        np.random.seed(42)
        errors1 = np.random.randn(100) * 0.1
        errors2 = np.random.randn(100) * 1.0

        dm_stat, p_value = diebold_mariano_test(errors1, errors2)

        assert dm_stat < 0  # Model 1 is better

    def test_larger_errors_gives_positive_dm(self):
        """Model 1 with larger errors should give positive DM."""
        # Longer series needed for meaningful test
        np.random.seed(42)
        errors1 = np.random.randn(100) * 1.0
        errors2 = np.random.randn(100) * 0.1

        dm_stat, p_value = diebold_mariano_test(errors1, errors2)

        assert dm_stat > 0  # Model 2 is better

    def test_handles_short_series(self):
        """Should handle very short error series."""
        errors1 = np.array([0.1])
        errors2 = np.array([0.2])

        dm_stat, p_value = diebold_mariano_test(errors1, errors2)

        assert dm_stat == 0.0
        assert p_value == 1.0

    def test_mae_loss(self):
        """Should work with MAE loss (power=1)."""
        errors1 = np.array([0.1, -0.1, 0.2, -0.2, 0.1])
        errors2 = np.array([0.5, -0.5, 0.6, -0.6, 0.5])

        dm_stat, p_value = diebold_mariano_test(errors1, errors2, power=1)

        assert dm_stat < 0  # Model 1 is better


class TestPairwiseDMTests:
    """Test cases for pairwise_dm_tests function."""

    def test_returns_results_for_all_pairs(self):
        """Should return results for all model pairs."""
        results = {
            "model1": EvaluationResult(
                model_name="model1",
                metrics={"mse": 0.1},
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1, 2, 3]),
                residuals=np.array([0.1, 0.1, 0.1]),
                residual_diagnostics=None,
                mz_r2=None,
                mz_alpha=None,
                mz_beta=None,
                evaluation_time=1.0,
                n_samples=3,
                model_params={},
            ),
            "model2": EvaluationResult(
                model_name="model2",
                metrics={"mse": 0.2},
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1, 2, 3]),
                residuals=np.array([0.2, 0.2, 0.2]),
                residual_diagnostics=None,
                mz_r2=None,
                mz_alpha=None,
                mz_beta=None,
                evaluation_time=1.0,
                n_samples=3,
                model_params={},
            ),
            "model3": EvaluationResult(
                model_name="model3",
                metrics={"mse": 0.3},
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1, 2, 3]),
                residuals=np.array([0.3, 0.3, 0.3]),
                residual_diagnostics=None,
                mz_r2=None,
                mz_alpha=None,
                mz_beta=None,
                evaluation_time=1.0,
                n_samples=3,
                model_params={},
            ),
        }

        test_results = pairwise_dm_tests(results)

        # 3 models -> 3 pairs
        assert len(test_results) == 3

    def test_returns_pairwise_test_results(self):
        """Should return PairwiseTestResult objects."""
        results = {
            "model1": EvaluationResult(
                model_name="model1",
                metrics={"mse": 0.1},
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1, 2, 3]),
                residuals=np.array([0.1, 0.1, 0.1]),
                residual_diagnostics=None,
                mz_r2=None,
                mz_alpha=None,
                mz_beta=None,
                evaluation_time=1.0,
                n_samples=3,
                model_params={},
            ),
            "model2": EvaluationResult(
                model_name="model2",
                metrics={"mse": 0.2},
                predictions=np.array([1, 2, 3]),
                actuals=np.array([1, 2, 3]),
                residuals=np.array([0.5, 0.5, 0.5]),
                residual_diagnostics=None,
                mz_r2=None,
                mz_alpha=None,
                mz_beta=None,
                evaluation_time=1.0,
                n_samples=3,
                model_params={},
            ),
        }

        test_results = pairwise_dm_tests(results)

        assert len(test_results) == 1
        result = test_results[0]
        assert isinstance(result, PairwiseTestResult)
        assert result.model1 == "model1"
        assert result.model2 == "model2"


class TestModelComparisonResult:
    """Test cases for ModelComparisonResult class."""

    def test_summary(self):
        """Should generate text summary."""
        metrics_df = pd.DataFrame({
            "model1": {"mse": 0.1, "rmse": 0.316},
            "model2": {"mse": 0.2, "rmse": 0.447},
        }).T
        metrics_df.index.name = "model"

        result = ModelComparisonResult(
            model_names=["model1", "model2"],
            metrics_df=metrics_df,
            rankings={"mse": ["model1", "model2"]},
            best_model="model1",
            primary_metric="mse",
            evaluation_results={},
        )

        summary = result.summary()

        assert "model1" in summary
        assert "model2" in summary
        assert "Best model: model1" in summary

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics_df = pd.DataFrame({
            "model1": {"mse": 0.1},
        }).T

        result = ModelComparisonResult(
            model_names=["model1"],
            metrics_df=metrics_df,
            rankings={"mse": ["model1"]},
            best_model="model1",
            primary_metric="mse",
            evaluation_results={},
        )

        d = result.to_dict()

        assert d["best_model"] == "model1"
        assert d["primary_metric"] == "mse"


class TestModelComparator:
    """Test cases for ModelComparator class."""

    def test_compare_models(self):
        """Should compare multiple models."""
        models = {
            "good": MockModel("good", prediction_value=2.5),
            "bad": MockModel("bad", prediction_value=10.0),
        }
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        comparator = ModelComparator(metrics=["mse", "mae"], verbose=False)
        result = comparator.compare(models, X, y)

        assert result.best_model == "good"
        assert "good" in result.model_names
        assert "bad" in result.model_names

    def test_rankings(self):
        """Should rank models correctly."""
        models = {
            "model_a": MockModel("model_a", prediction_value=3.0),
            "model_b": MockModel("model_b", prediction_value=5.0),
            "model_c": MockModel("model_c", prediction_value=10.0),
        }
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        comparator = ModelComparator(primary_metric="mse", verbose=False)
        result = comparator.compare(models, X, y)

        # model_a (pred=3.0) should be closest to mean=3, so best
        rankings = result.rankings["mse"]
        assert rankings[0] == "model_a"

    def test_compare_with_significance(self):
        """Should return DM test results."""
        models = {
            "good": MockModel("good", prediction_value=3.0),
            "bad": MockModel("bad", prediction_value=10.0),
        }
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        comparator = ModelComparator(verbose=False)
        result, dm_tests = comparator.compare_with_significance(models, X, y)

        assert isinstance(result, ModelComparisonResult)
        assert isinstance(dm_tests, list)


class TestCompareModels:
    """Test cases for compare_models convenience function."""

    def test_compares_models(self):
        """Should compare models correctly."""
        models = {
            "a": MockModel("a", prediction_value=2.5),
            "b": MockModel("b", prediction_value=5.0),
        }
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        result = compare_models(models, X, y, verbose=False)

        assert isinstance(result, ModelComparisonResult)
        assert result.best_model in ["a", "b"]


class TestRankModels:
    """Test cases for rank_models function."""

    def test_returns_ranked_list(self):
        """Should return ranked list of (name, score) tuples."""
        models = {
            "a": MockModel("a", prediction_value=3.0),
            "b": MockModel("b", prediction_value=5.0),
        }
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])

        ranking = rank_models(models, X, y, metric="mse")

        assert len(ranking) == 2
        assert isinstance(ranking[0], tuple)
        assert ranking[0][0] in ["a", "b"]


class TestSaveComparisonResults:
    """Test cases for save_comparison_results function."""

    def test_saves_to_file(self):
        """Should save results to JSON file."""
        metrics_df = pd.DataFrame({
            "model1": {"mse": 0.1},
        }).T

        result = ModelComparisonResult(
            model_names=["model1"],
            metrics_df=metrics_df,
            rankings={"mse": ["model1"]},
            best_model="model1",
            primary_metric="mse",
            evaluation_results={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "comparison.json"
            save_comparison_results(result, path)

            assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
