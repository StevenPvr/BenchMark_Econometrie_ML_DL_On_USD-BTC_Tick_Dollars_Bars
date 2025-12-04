"""Tests for src.labelling.label_meta.train.logic."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.labelling.label_meta.train.logic import (
    MetaEvaluationMetrics,
    MetaTrainingConfig,
    MetaTrainingResult,
    load_optimized_params,
    compute_metrics,
    _remove_non_feature_cols,
    _handle_missing_values,
    get_available_meta_optimizations,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Generate sample features DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "bar_id": range(100),
        "split": ["train"] * 80 + ["test"] * 20,
        "datetime_close": dates,
        "label": np.random.choice([0, 1], 100),
        "prediction": np.random.choice([0, 1], 100),
        "coverage": [1] * 100,
    })


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestMetaEvaluationMetrics:
    """Tests for MetaEvaluationMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = MetaEvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            auc_roc=0.90,
        )

        d = metrics.to_dict()

        assert d["accuracy"] == 0.85
        assert d["precision"] == 0.80
        assert d["recall"] == 0.75
        assert d["f1"] == 0.77
        assert d["auc_roc"] == 0.90

    def test_to_dict_none_auc(self) -> None:
        """Test conversion when AUC is None."""
        metrics = MetaEvaluationMetrics(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            auc_roc=None,
        )

        d = metrics.to_dict()
        assert d["auc_roc"] is None


class TestMetaTrainingConfig:
    """Tests for MetaTrainingConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = MetaTrainingConfig(
            primary_model_name="lightgbm",
            meta_model_name="xgboost",
        )

        assert config.primary_model_name == "lightgbm"
        assert config.meta_model_name == "xgboost"
        assert config.random_state == 42
        assert config.use_class_weight is True


class TestMetaTrainingResult:
    """Tests for MetaTrainingResult dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = MetaTrainingResult(
            primary_model_name="primary",
            meta_model_name="meta",
            meta_model_params={"n_estimators": 100},
            triple_barrier_params={"pt_mult": 1.0},
            train_samples=1000,
            label_distribution={"0": 500, "1": 500},
            train_metrics={"accuracy": 0.85},
            model_path="/path/to/model",
        )

        d = result.to_dict()

        assert d["primary_model_name"] == "primary"
        assert d["meta_model_name"] == "meta"
        assert d["train_samples"] == 1000
        assert "timestamp" in d

    def test_save(self, tmp_path: Path) -> None:
        """Test saving to JSON."""
        result = MetaTrainingResult(
            primary_model_name="primary",
            meta_model_name="meta",
            meta_model_params={"n_estimators": 100},
            triple_barrier_params={"pt_mult": 1.0},
            train_samples=1000,
            label_distribution={"0": 500, "1": 500},
            train_metrics={"accuracy": 0.85},
            model_path="/path/to/model",
        )

        path = tmp_path / "result.json"
        result.save(path)

        assert path.exists()

        with open(path) as f:
            data = json.load(f)
            assert data["primary_model_name"] == "primary"


# =============================================================================
# PARAMETER LOADING TESTS
# =============================================================================


class TestParameterLoading:
    """Tests for parameter loading functions."""

    def test_load_optimized_params(self, tmp_path: Path) -> None:
        """Test loading optimized parameters."""
        opti_data = {
            "best_params": {"n_estimators": 100},
            "best_triple_barrier_params": {"pt_mult": 1.0},
            "best_score": 0.85,
            "metric": "precision",
        }

        opti_file = tmp_path / "primary_meta_optimization.json"
        with open(opti_file, "w") as f:
            json.dump(opti_data, f)

        result = load_optimized_params("primary", "meta", opti_dir=tmp_path)

        assert result["meta_model_params"] == {"n_estimators": 100}
        assert result["best_score"] == 0.85

    def test_load_optimized_params_not_found(self, tmp_path: Path) -> None:
        """Test loading when file not found."""
        with pytest.raises(FileNotFoundError):
            load_optimized_params("nonexistent", "model", opti_dir=tmp_path)


# =============================================================================
# METRICS TESTS
# =============================================================================


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_basic(self) -> None:
        """Test basic metrics computation."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1

    def test_compute_metrics_with_proba(self) -> None:
        """Test metrics with probabilities."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4],
            [0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]
        ])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics.auc_roc is not None

    def test_compute_metrics_1d_proba(self) -> None:
        """Test metrics with 1D probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics.auc_roc is not None

    def test_compute_metrics_perfect_predictions(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_compute_metrics_all_wrong(self) -> None:
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.0


# =============================================================================
# DATA PREPARATION TESTS
# =============================================================================


class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_remove_non_feature_cols(self, sample_features: pd.DataFrame) -> None:
        """Test removing non-feature columns."""
        result = _remove_non_feature_cols(sample_features)

        assert "bar_id" not in result.columns
        assert "split" not in result.columns
        assert "datetime_close" not in result.columns
        assert "label" not in result.columns
        assert "prediction" not in result.columns
        assert "coverage" not in result.columns
        assert "f1" in result.columns
        assert "f2" in result.columns

    def test_remove_non_feature_cols_empty(self) -> None:
        """Test with no columns to remove."""
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
        })

        result = _remove_non_feature_cols(df)
        assert "f1" in result.columns
        assert "f2" in result.columns

    def test_handle_missing_values(self) -> None:
        """Test handling missing values."""
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [np.nan, np.nan, np.nan, np.nan],
            "f3": [1.0, 2.0, 3.0, 4.0],
        })

        result = _handle_missing_values(df.copy())

        assert isinstance(result, pd.DataFrame)
        # f3 should remain unchanged
        assert result["f3"].tolist() == [1.0, 2.0, 3.0, 4.0]
        # f1 should have NaN filled
        assert not result["f1"].isna().any()

    def test_handle_missing_values_no_nan(self) -> None:
        """Test with no missing values."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        })

        result = _handle_missing_values(df)

        pd.testing.assert_frame_equal(result, df)


# =============================================================================
# OPTIMIZATION AVAILABILITY TESTS
# =============================================================================


class TestOptimizationAvailability:
    """Tests for optimization availability functions."""

    def test_get_available_meta_optimizations_empty(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test when no optimizations exist."""
        mocker.patch(
            "src.labelling.label_meta.train.logic.LABEL_META_OPTI_DIR",
            tmp_path / "nonexistent",
        )

        result = get_available_meta_optimizations()
        assert result == []

    def test_get_available_meta_optimizations(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test listing available optimizations."""
        mocker.patch(
            "src.labelling.label_meta.train.logic.LABEL_META_OPTI_DIR",
            tmp_path,
        )

        # Create mock optimization files
        (tmp_path / "primary1_meta1_optimization.json").touch()
        (tmp_path / "primary2_meta2_optimization.json").touch()

        result = get_available_meta_optimizations()

        assert len(result) == 2
        assert ("primary1", "meta1") in result
        assert ("primary2", "meta2") in result
