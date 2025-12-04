"""Tests for src.labelling.label_meta.eval.logic."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.labelling.label_meta.eval.logic import (
    BARS_PER_YEAR,
    INITIAL_CAPITAL,
    TOTAL_COST_PCT,
    TradingMetrics,
    CombinedEvaluationResult,
    compute_trading_metrics,
    load_meta_model,
    load_meta_training_results,
    get_available_trained_meta_models,
    _remove_non_feature_cols,
    _handle_missing_values,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample returns."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02


@pytest.fixture
def sample_positions() -> np.ndarray:
    """Generate sample positions."""
    np.random.seed(42)
    return np.random.choice([-1, 0, 1], 100)


@pytest.fixture
def sample_features() -> pd.DataFrame:
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
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_initial_capital(self) -> None:
        """Test INITIAL_CAPITAL is set correctly."""
        assert INITIAL_CAPITAL == 10_000.0

    def test_total_cost_pct(self) -> None:
        """Test TOTAL_COST_PCT is set correctly."""
        assert TOTAL_COST_PCT == 0.01

    def test_bars_per_year(self) -> None:
        """Test BARS_PER_YEAR is set correctly."""
        assert BARS_PER_YEAR == 12500


# =============================================================================
# TRADING METRICS DATACLASS TESTS
# =============================================================================


class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = TradingMetrics(
            n_trades=100,
            n_profitable=60,
            n_losing=40,
            win_rate=0.6,
            gross_pnl=1500.0,
            net_pnl=1200.0,
            total_costs=300.0,
            final_capital=11200.0,
            gross_return_pct=15.0,
            net_return_pct=12.0,
            sharpe_ratio=1.5,
            max_drawdown=500.0,
            max_drawdown_pct=5.0,
            profit_factor=2.0,
            avg_win=50.0,
            avg_loss=-30.0,
        )

        d = metrics.to_dict()

        assert d["n_trades"] == 100
        assert d["n_profitable"] == 60
        assert d["n_losing"] == 40
        assert d["win_rate"] == 0.6
        assert d["gross_pnl"] == 1500.0
        assert d["net_pnl"] == 1200.0
        assert d["sharpe_ratio"] == 1.5

    def test_to_dict_none_values(self) -> None:
        """Test conversion with None values."""
        metrics = TradingMetrics(
            n_trades=0,
            n_profitable=0,
            n_losing=0,
            win_rate=0.0,
            gross_pnl=0.0,
            net_pnl=0.0,
            total_costs=0.0,
            final_capital=10000.0,
            gross_return_pct=0.0,
            net_return_pct=0.0,
            sharpe_ratio=None,
            max_drawdown=None,
            max_drawdown_pct=None,
            profit_factor=None,
            avg_win=None,
            avg_loss=None,
        )

        d = metrics.to_dict()
        assert d["sharpe_ratio"] is None
        assert d["max_drawdown"] is None


# =============================================================================
# COMBINED EVALUATION RESULT TESTS
# =============================================================================


class TestCombinedEvaluationResult:
    """Tests for CombinedEvaluationResult dataclass."""

    def test_to_dict(self) -> None:
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

    def test_save(self, tmp_path: Path) -> None:
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
    """Tests for compute_trading_metrics function."""

    def test_basic_metrics(
        self, sample_returns: np.ndarray, sample_positions: np.ndarray
    ) -> None:
        """Test basic trading metrics computation."""
        metrics = compute_trading_metrics(sample_returns, sample_positions)

        assert isinstance(metrics, TradingMetrics)
        assert metrics.n_trades >= 0
        assert 0 <= metrics.win_rate <= 1

    def test_no_trades(self) -> None:
        """Test with no trades (all zero positions)."""
        returns = np.array([0.01, -0.01, 0.02])
        positions = np.array([0, 0, 0])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.gross_pnl == 0.0
        assert metrics.net_pnl == 0.0
        assert metrics.final_capital == INITIAL_CAPITAL

    def test_all_profitable(self) -> None:
        """Test with all profitable trades."""
        returns = np.array([0.01, 0.02, 0.03])
        positions = np.array([1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 3
        assert metrics.n_profitable == 3
        assert metrics.n_losing == 0
        assert metrics.win_rate == 1.0
        assert metrics.gross_pnl > 0

    def test_all_losing(self) -> None:
        """Test with all losing trades."""
        returns = np.array([-0.01, -0.02, -0.03])
        positions = np.array([1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 3
        assert metrics.n_profitable == 0
        assert metrics.n_losing == 3
        assert metrics.win_rate == 0.0

    def test_short_positions(self) -> None:
        """Test with short positions."""
        returns = np.array([0.01, -0.02, 0.03])  # Market returns
        positions = np.array([-1, -1, -1])  # All short

        metrics = compute_trading_metrics(returns, positions)

        # Short positions profit when market goes down
        assert metrics.n_trades == 3
        assert metrics.n_profitable == 1  # Only -0.02 trade is profitable

    def test_sharpe_ratio(self) -> None:
        """Test Sharpe ratio computation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
        positions = np.array([1, 1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.sharpe_ratio is not None

    def test_max_drawdown(self) -> None:
        """Test max drawdown computation."""
        returns = np.array([0.01, 0.02, -0.05, 0.01, 0.01])
        positions = np.array([1, 1, 1, 1, 1])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown >= 0

    def test_empty_arrays(self) -> None:
        """Test with empty arrays."""
        returns = np.array([])
        positions = np.array([])

        metrics = compute_trading_metrics(returns, positions)

        assert metrics.n_trades == 0
        assert metrics.win_rate == 0.0

    def test_mixed_positions(self) -> None:
        """Test with mixed long, short, and no positions."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        positions = np.array([1, -1, 0, 1, -1])

        metrics = compute_trading_metrics(returns, positions)

        # Should count 4 trades (position 0 is skipped)
        assert metrics.n_trades == 4


# =============================================================================
# DATA LOADING TESTS
# =============================================================================


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_meta_model_not_found(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test loading when meta model not found."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            load_meta_model("primary", "meta")

    def test_load_meta_training_results(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test loading training results."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path,
        )

        # Create mock results
        model_dir = tmp_path / "primary_meta"
        model_dir.mkdir()

        results = {"triple_barrier_params": {"pt_mult": 1.0}}
        with open(model_dir / "training_results.json", "w") as f:
            json.dump(results, f)

        loaded = load_meta_training_results("primary", "meta")
        assert loaded["triple_barrier_params"]["pt_mult"] == 1.0

    def test_load_meta_training_results_not_found(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test loading when training results not found."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            load_meta_training_results("primary", "meta")

    def test_get_available_trained_meta_models_empty(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test when no trained models exist."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path / "nonexistent",
        )

        result = get_available_trained_meta_models()
        assert result == []

    def test_get_available_trained_meta_models(
        self, tmp_path: Path, mocker: MagicMock
    ) -> None:
        """Test listing available trained models."""
        mocker.patch(
            "src.labelling.label_meta.eval.logic.LABEL_META_TRAIN_DIR",
            tmp_path,
        )

        # Create mock model directories
        model_dir = tmp_path / "primary1_meta1"
        model_dir.mkdir()
        (model_dir / "meta1_model.joblib").touch()

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
    """Tests for data preparation functions."""

    def test_remove_non_feature_cols(self, sample_features: pd.DataFrame) -> None:
        """Test removing non-feature columns."""
        result = _remove_non_feature_cols(sample_features)

        assert "bar_id" not in result.columns
        assert "split" not in result.columns
        assert "datetime_close" not in result.columns
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
        assert result["f3"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_handle_missing_values_no_nan(self) -> None:
        """Test with no missing values."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        })

        result = _handle_missing_values(df)

        pd.testing.assert_frame_equal(result, df)
