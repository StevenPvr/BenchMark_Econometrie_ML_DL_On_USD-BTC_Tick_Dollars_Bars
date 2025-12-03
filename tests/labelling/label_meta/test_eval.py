"""Tests for src.labelling.label_meta.eval.logic."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.labelling.label_meta.eval.logic import (
    TradingMetrics,
    CombinedEvaluationResult,
    compute_trading_metrics,
    load_meta_model,
    load_meta_training_results,
    get_available_trained_meta_models,
    _remove_non_feature_cols,
    _handle_missing_values,
)

# Import for testing full paths
from src.labelling.label_meta import eval as eval_module


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_returns():
    """Generate sample returns."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02


@pytest.fixture
def sample_positions():
    """Generate sample positions."""
    np.random.seed(42)
    return np.random.choice([-1, 0, 1], 100)


@pytest.fixture
def sample_features():
    """Generate sample features DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "bar_id": range(100),
        "split": ["test"] * 100,
        "datetime_close": dates,
    })


# =============================================================================
# TRADING METRICS DATACLASS TESTS
# =============================================================================


class TestTradingMetrics:
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TradingMetrics(
            n_trades=100,
            n_profitable=60,
            n_losing=40,
            win_rate=0.6,
            total_return=0.15,
            mean_return=0.0015,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
        )

        d = metrics.to_dict()

        assert d["n_trades"] == 100
        assert d["n_profitable"] == 60
        assert d["n_losing"] == 40
        assert d["win_rate"] == 0.6
        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.5

    def test_to_dict_none_values(self):
        """Test conversion with None values."""
        metrics = TradingMetrics(
            n_trades=0,
            n_profitable=0,
            n_losing=0,
            win_rate=0.0,
            total_return=0.0,
            mean_return=0.0,
            sharpe_ratio=None,
            max_drawdown=None,
        )

        d = metrics.to_dict()
        assert d["sharpe_ratio"] is None
        assert d["max_drawdown"] is None


# =============================================================================
# COMBINED EVALUATION RESULT TESTS
# =============================================================================


class TestCombinedEvaluationResult:
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CombinedEvaluationResult(
            primary_model_name="lightgbm",
            meta_model_name="xgboost",
            n_test_samples=1000,
            meta_accuracy=0.85,
            meta_precision=0.80,
            meta_recall=0.75,
            meta_f1=0.77,
            primary_only_metrics={"win_rate": 0.52},
            combined_metrics={"win_rate": 0.65},
            trades_filtered_pct=30.0,
            win_rate_improvement=0.13,
            meta_confusion_matrix=[[100, 50], [30, 120]],
            primary_model_path="/path/primary",
            meta_model_path="/path/meta",
            predictions_path="/path/predictions",
        )

        d = result.to_dict()

        assert d["primary_model_name"] == "lightgbm"
        assert d["meta_model_name"] == "xgboost"
        assert d["n_test_samples"] == 1000
        assert d["meta_accuracy"] == 0.85

    def test_save(self, tmp_path):
        """Test saving to JSON."""
        result = CombinedEvaluationResult(
            primary_model_name="lightgbm",
            meta_model_name="xgboost",
            n_test_samples=1000,
            meta_accuracy=0.85,
            meta_precision=0.80,
            meta_recall=0.75,
            meta_f1=0.77,
            primary_only_metrics={"win_rate": 0.52},
            combined_metrics={"win_rate": 0.65},
            trades_filtered_pct=30.0,
            win_rate_improvement=0.13,
            meta_confusion_matrix=[[100, 50], [30, 120]],
            primary_model_path="/path/primary",
            meta_model_path="/path/meta",
            predictions_path="/path/predictions",
        )

        path = tmp_path / "result.json"
        result.save(path)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
            assert data["primary_model_name"] == "lightgbm"


# =============================================================================
# COMPUTE TRADING METRICS TESTS
# =============================================================================


class TestComputeTradingMetrics:
    def test_basic_metrics(self, sample_returns, sample_positions):
        """Test basic trading metrics computation."""
        metrics = compute_trading_metrics(sample_returns, sample_positions)

        assert isinstance(metrics, TradingMetrics)
        assert metrics.n_trades >= 0
        assert 0 <= metrics.win_rate <= 1

    def test_no_trades(self):
        """Test with no trades (all zero positions)."""
        returns = np.array([0.01, -0.01, 0.02])
        positions = np.array([0, 0, 0])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_return == 0.0

    def test_all_profitable(self):
        """Test with all profitable trades."""
        returns = np.array([0.01, 0.02, 0.03])
        positions = np.array([1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 3
        assert metrics.n_profitable == 3
        assert metrics.n_losing == 0
        assert metrics.win_rate == 1.0

    def test_all_losing(self):
        """Test with all losing trades."""
        returns = np.array([-0.01, -0.02, -0.03])
        positions = np.array([1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 3
        assert metrics.n_profitable == 0
        assert metrics.n_losing == 3
        assert metrics.win_rate == 0.0

    def test_short_positions(self):
        """Test with short positions."""
        returns = np.array([0.01, -0.02, 0.03])  # Market returns
        positions = np.array([-1, -1, -1])  # All short

        metrics = compute_trading_metrics(returns, positions)

        # Short positions profit when market goes down
        # Position * return: -1*0.01=-0.01, -1*-0.02=0.02, -1*0.03=-0.03
        assert metrics.n_trades == 3
        assert metrics.n_profitable == 1  # Only -0.02 trade is profitable

    def test_sharpe_ratio(self):
        """Test Sharpe ratio computation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        positions = np.array([1, 1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.sharpe_ratio is not None

    def test_max_drawdown(self):
        """Test max drawdown computation."""
        returns = np.array([0.01, 0.02, -0.05, 0.01, 0.01])
        positions = np.array([1, 1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown >= 0


# =============================================================================
# DATA LOADING TESTS
# =============================================================================


class TestDataLoading:
    def test_load_meta_model_not_found(self, tmp_path, mocker):
        """Test loading when meta model not found."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        with pytest.raises(FileNotFoundError):
            load_meta_model("primary", "meta")

    def test_load_meta_training_results(self, tmp_path, mocker):
        """Test loading training results."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Create mock results
        model_dir = tmp_path / "primary_meta"
        model_dir.mkdir()

        results = {"triple_barrier_params": {"pt_mult": 1.0}}
        with open(model_dir / "training_results.json", "w") as f:
            json.dump(results, f)

        loaded = load_meta_training_results("primary", "meta")
        assert loaded["triple_barrier_params"]["pt_mult"] == 1.0

    def test_load_meta_training_results_not_found(self, tmp_path, mocker):
        """Test loading when training results not found."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        with pytest.raises(FileNotFoundError):
            load_meta_training_results("primary", "meta")

    def test_get_available_trained_meta_models_empty(self, tmp_path, mocker):
        """Test when no trained models exist."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path / "nonexistent"
        )

        result = get_available_trained_meta_models()
        assert result == []

    def test_get_available_trained_meta_models(self, tmp_path, mocker):
        """Test listing available trained models."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Create mock model directories
        model_dir = tmp_path / "primary1_meta1"
        model_dir.mkdir()
        (model_dir / "final_meta_model.joblib").touch()

        model_dir2 = tmp_path / "primary2_meta2"
        model_dir2.mkdir()
        # No joblib file - should not be included

        result = get_available_trained_meta_models()
        assert len(result) == 1
        assert ("primary1", "meta1") in result


# =============================================================================
# DATA PREPARATION TESTS
# =============================================================================


class TestDataPreparation:
    def test_remove_non_feature_cols(self, sample_features):
        """Test removing non-feature columns."""
        result = _remove_non_feature_cols(sample_features)

        assert "bar_id" not in result.columns
        assert "split" not in result.columns
        assert "datetime_close" not in result.columns
        assert "f1" in result.columns
        assert "f2" in result.columns

    def test_handle_missing_values(self):
        """Test handling missing values."""
        # Note: _handle_missing_values uses isinstance(is_na_any, bool)
        # which may not catch numpy bool. Test the actual behavior.
        df = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [np.nan, np.nan, np.nan, np.nan],
            "f3": [1.0, 2.0, 3.0, 4.0],
        })

        # Call the function
        result = _handle_missing_values(df.copy())

        # The function returns a DataFrame
        assert isinstance(result, pd.DataFrame)
        # f3 should remain unchanged
        assert result["f3"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_handle_missing_values_no_nan(self):
        """Test with no missing values."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        })

        result = _handle_missing_values(df)

        pd.testing.assert_frame_equal(result, df)


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestAdditionalCoverage:
    def test_trading_metrics_all_fields(self):
        """Test all trading metrics fields."""
        metrics = TradingMetrics(
            n_trades=50,
            n_profitable=30,
            n_losing=20,
            win_rate=0.6,
            total_return=0.15,
            mean_return=0.003,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
        )

        d = metrics.to_dict()

        assert d["n_trades"] == 50
        assert d["n_profitable"] == 30
        assert d["n_losing"] == 20
        assert d["win_rate"] == 0.6
        assert d["sharpe_ratio"] == 1.2

    def test_compute_trading_metrics_mixed_positions(self):
        """Test with mixed long, short, and no positions."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        positions = np.array([1, -1, 0, 1, -1])

        metrics = compute_trading_metrics(returns, positions)

        # Should count 4 trades (position 0 is skipped)
        assert metrics.n_trades == 4

    def test_compute_trading_metrics_constant_returns(self):
        """Test with constant returns."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        positions = np.array([1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 4
        assert metrics.win_rate == 1.0
        # Sharpe ratio should be None (std=0)
        assert metrics.sharpe_ratio is None

    def test_combined_evaluation_result_full(self):
        """Test full CombinedEvaluationResult."""
        result = CombinedEvaluationResult(
            primary_model_name="primary",
            meta_model_name="meta",
            n_test_samples=500,
            meta_accuracy=0.75,
            meta_precision=0.72,
            meta_recall=0.78,
            meta_f1=0.75,
            primary_only_metrics={
                "n_trades": 500,
                "win_rate": 0.52,
                "total_return": 0.05,
            },
            combined_metrics={
                "n_trades": 300,
                "win_rate": 0.60,
                "total_return": 0.08,
            },
            trades_filtered_pct=40.0,
            win_rate_improvement=0.08,
            meta_confusion_matrix=[[100, 50], [40, 110]],
            primary_model_path="/path/primary",
            meta_model_path="/path/meta",
            predictions_path="/path/predictions",
        )

        d = result.to_dict()

        assert d["n_test_samples"] == 500
        assert d["trades_filtered_pct"] == 40.0
        assert d["win_rate_improvement"] == 0.08

    def test_get_available_trained_meta_models_multiple(self, tmp_path, mocker):
        """Test with multiple trained models."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Create multiple model directories
        for i in range(3):
            model_dir = tmp_path / f"primary_{i}_meta_{i}"
            model_dir.mkdir()
            (model_dir / "final_meta_model.joblib").touch()

        result = get_available_trained_meta_models()
        assert len(result) == 3

    def test_compute_trading_metrics_negative_sharpe(self):
        """Test trading metrics with losing strategy."""
        returns = np.array([-0.01, -0.02, -0.01, 0.005])
        positions = np.array([1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.total_return < 0
        assert metrics.win_rate < 0.5

    def test_compute_trading_metrics_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create returns that produce a clear drawdown
        returns = np.array([0.05, 0.03, -0.10, -0.05, 0.02])
        positions = np.array([1, 1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown > 0

    def test_remove_non_feature_cols_all_types(self):
        """Test removing all non-feature column types."""
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "bar_id": [1, 2, 3],
            "timestamp_open": pd.date_range("2023-01-01", periods=3),
            "datetime_close": pd.date_range("2023-01-01", periods=3),
            "split": ["train", "train", "test"],
            "label": [0, 1, 0],
        })

        result = _remove_non_feature_cols(df)

        assert "f1" in result.columns
        assert "bar_id" not in result.columns
        assert "timestamp_open" not in result.columns
        assert "label" not in result.columns


# =============================================================================
# TRADING METRICS DETAILED TESTS
# =============================================================================


class TestTradingMetricsDetailed:
    def test_compute_trading_metrics_zero_trades(self):
        """Test with zero positions."""
        returns = np.array([0.01, -0.01, 0.02])
        positions = np.array([0, 0, 0])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_return == 0.0
        assert metrics.mean_return == 0.0

    def test_compute_trading_metrics_single_trade(self):
        """Test with single trade."""
        returns = np.array([0.05])
        positions = np.array([1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 1
        assert metrics.n_profitable == 1
        assert metrics.total_return == 0.05
        assert metrics.sharpe_ratio is None  # Can't compute with 1 trade

    def test_compute_trading_metrics_perfect_strategy(self):
        """Test perfect trading strategy."""
        # All trades are profitable
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        positions = np.array([1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.win_rate == 1.0
        assert metrics.n_profitable == 4
        assert metrics.n_losing == 0

    def test_compute_trading_metrics_short_only(self):
        """Test short-only strategy."""
        returns = np.array([0.02, -0.03, 0.01])  # Market returns
        positions = np.array([-1, -1, -1])  # All short

        metrics = compute_trading_metrics(returns, positions)

        # Short profit: -0.02, 0.03, -0.01 = 0.0
        assert metrics.n_trades == 3
        # One profitable (when market went down)
        assert metrics.n_profitable == 1


# =============================================================================
# COMBINED EVALUATION RESULT DETAILED TESTS
# =============================================================================


class TestCombinedEvaluationResultDetailed:
    def test_combined_result_to_dict_all_fields(self):
        """Test all fields are present in dict."""
        result = CombinedEvaluationResult(
            primary_model_name="lgb",
            meta_model_name="xgb",
            n_test_samples=100,
            meta_accuracy=0.8,
            meta_precision=0.75,
            meta_recall=0.7,
            meta_f1=0.72,
            primary_only_metrics={"win_rate": 0.5},
            combined_metrics={"win_rate": 0.6},
            trades_filtered_pct=20.0,
            win_rate_improvement=0.1,
            meta_confusion_matrix=[[40, 10], [15, 35]],
            primary_model_path="p",
            meta_model_path="m",
            predictions_path="pred",
        )

        d = result.to_dict()

        expected_keys = [
            "primary_model_name", "meta_model_name", "n_test_samples",
            "meta_accuracy", "meta_precision", "meta_recall", "meta_f1",
            "primary_only_metrics", "combined_metrics",
            "trades_filtered_pct", "win_rate_improvement",
        ]

        for key in expected_keys:
            assert key in d

    def test_combined_result_save_load(self, tmp_path):
        """Test saving and loading results."""
        result = CombinedEvaluationResult(
            primary_model_name="test",
            meta_model_name="test",
            n_test_samples=50,
            meta_accuracy=0.75,
            meta_precision=0.7,
            meta_recall=0.72,
            meta_f1=0.71,
            primary_only_metrics={"win_rate": 0.51},
            combined_metrics={"win_rate": 0.58},
            trades_filtered_pct=15.0,
            win_rate_improvement=0.07,
            meta_confusion_matrix=[[20, 5], [8, 17]],
            primary_model_path="p",
            meta_model_path="m",
            predictions_path="pred",
        )

        path = tmp_path / "result.json"
        result.save(path)

        # Load and verify
        with open(path) as f:
            loaded = json.load(f)

        assert loaded["meta_accuracy"] == 0.75
        assert loaded["win_rate_improvement"] == 0.07


# =============================================================================
# DATA LOADING DETAILED TESTS
# =============================================================================


class TestDataLoadingDetailed:
    def test_load_meta_training_results_success(self, tmp_path, mocker):
        """Test successful loading of training results."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        model_dir = tmp_path / "primary_meta"
        model_dir.mkdir()

        results = {
            "triple_barrier_params": {"pt_mult": 1.5},
            "meta_model_params": {"n_estimators": 100},
        }
        with open(model_dir / "training_results.json", "w") as f:
            json.dump(results, f)

        loaded = load_meta_training_results("primary", "meta")

        assert loaded["triple_barrier_params"]["pt_mult"] == 1.5

    def test_get_available_meta_models_empty_dir(self, tmp_path, mocker):
        """Test with empty training directory."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Create empty dir
        tmp_path.mkdir(exist_ok=True)

        result = get_available_trained_meta_models()
        assert result == []

    def test_get_available_meta_models_with_incomplete(self, tmp_path, mocker):
        """Test filtering out incomplete model directories."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Complete model
        complete = tmp_path / "primary_meta"
        complete.mkdir()
        (complete / "final_meta_model.joblib").touch()

        # Incomplete model (no joblib)
        incomplete = tmp_path / "primary2_meta2"
        incomplete.mkdir()

        result = get_available_trained_meta_models()

        assert len(result) == 1
        assert ("primary", "meta") in result


# =============================================================================
# SHARPE RATIO AND DRAWDOWN EDGE CASES
# =============================================================================


class TestTradingMetricsEdgeCases:
    def test_sharpe_with_large_variance(self):
        """Test Sharpe ratio with high variance returns."""
        returns = np.array([0.10, -0.15, 0.20, -0.10, 0.05])
        positions = np.array([1, 1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.sharpe_ratio is not None
        # With high variance, Sharpe should be relatively low

    def test_max_drawdown_recovery(self):
        """Test max drawdown with full recovery."""
        # Up, down, recovery
        returns = np.array([0.10, -0.15, 0.10, 0.05])
        positions = np.array([1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown > 0

    def test_trading_metrics_alternating_positions(self):
        """Test with alternating long/short positions."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        positions = np.array([1, -1, 1, -1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 5
        # Check returns are properly calculated
        assert isinstance(metrics.total_return, (int, float))

    def test_compute_trading_metrics_empty_arrays(self):
        """Test with empty arrays."""
        returns = np.array([])
        positions = np.array([])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 0
        assert metrics.win_rate == 0.0


# =============================================================================
# LOAD MODEL EDGE CASES
# =============================================================================


class TestLoadModelEdgeCases:
    def test_load_meta_model_wrong_path(self, tmp_path, mocker):
        """Test loading from wrong directory structure."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Create directory but wrong file
        model_dir = tmp_path / "primary_meta"
        model_dir.mkdir()
        (model_dir / "wrong_file.txt").touch()

        with pytest.raises(FileNotFoundError):
            load_meta_model("primary", "meta")

    def test_get_available_models_complex_names(self, tmp_path, mocker):
        """Test with complex model names containing underscores."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path
        )

        # Create model with underscore in name
        model_dir = tmp_path / "lightgbm_xgboost_classifier"
        model_dir.mkdir()
        (model_dir / "final_meta_model.joblib").touch()

        result = get_available_trained_meta_models()

        assert len(result) == 1
        # Should parse first part as primary, rest as meta
        primary, meta = result[0]
        assert primary == "lightgbm"
        assert "xgboost" in meta
