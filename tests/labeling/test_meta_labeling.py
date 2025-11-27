"""Tests for the meta-labeling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.label_meta.meta_labeling import (
    compute_sharpe_ratio,
    compute_strategy_metrics,
    compute_strategy_returns,
    get_meta_features,
    get_meta_labels,
)


class TestGetMetaLabels:
    """Tests for get_meta_labels function."""

    @pytest.fixture
    def sample_events(self) -> pd.DataFrame:
        """Create sample events DataFrame."""
        return pd.DataFrame({
            "t_start": [0, 1, 2, 3, 4],
            "t_end": [5, 6, 7, 8, 9],
            "ret": [0.05, -0.03, 0.02, -0.04, 0.01],
            "barrier": [1, -1, 1, -1, 0],
            "label": [1, -1, 1, -1, 1],
        })

    def test_long_signal_correct(self, sample_events: pd.DataFrame) -> None:
        """Test meta-label for correct long signal."""
        # Primary signal = 1 (long), return > 0 -> meta = 1
        primary_signal = pd.Series([1, 0, 1, 0, 1])  # Long on positive returns

        meta_labels = get_meta_labels(sample_events, primary_signal)

        # Check long signals
        assert meta_labels.iloc[0] == 1  # Long, ret=0.05 > 0 -> correct
        assert meta_labels.iloc[2] == 1  # Long, ret=0.02 > 0 -> correct
        assert meta_labels.iloc[4] == 1  # Long, ret=0.01 > 0 -> correct

    def test_long_signal_wrong(self) -> None:
        """Test meta-label for wrong long signal."""
        events = pd.DataFrame({
            "t_start": [0],
            "t_end": [5],
            "ret": [-0.03],  # Negative return
            "barrier": [-1],
            "label": [-1],
        })
        primary_signal = pd.Series([1])  # Long signal

        meta_labels = get_meta_labels(events, primary_signal)

        # Long signal but negative return -> wrong -> meta = 0
        assert meta_labels.iloc[0] == 0

    def test_short_signal_correct(self) -> None:
        """Test meta-label for correct short signal."""
        events = pd.DataFrame({
            "t_start": [0],
            "t_end": [5],
            "ret": [-0.03],  # Negative return
            "barrier": [-1],
            "label": [-1],
        })
        primary_signal = pd.Series([-1])  # Short signal

        meta_labels = get_meta_labels(events, primary_signal)

        # Short signal and negative return -> correct -> meta = 1
        assert meta_labels.iloc[0] == 1

    def test_short_signal_wrong(self) -> None:
        """Test meta-label for wrong short signal."""
        events = pd.DataFrame({
            "t_start": [0],
            "t_end": [5],
            "ret": [0.05],  # Positive return
            "barrier": [1],
            "label": [1],
        })
        primary_signal = pd.Series([-1])  # Short signal

        meta_labels = get_meta_labels(events, primary_signal)

        # Short signal but positive return -> wrong -> meta = 0
        assert meta_labels.iloc[0] == 0

    def test_neutral_signal_nan(self, sample_events: pd.DataFrame) -> None:
        """Test that neutral signals result in NaN meta-labels."""
        primary_signal = pd.Series([0, 0, 0, 0, 0])  # All neutral

        meta_labels = get_meta_labels(sample_events, primary_signal)

        # All neutral -> all NaN
        assert meta_labels.isna().all()

    def test_mixed_signals(self, sample_events: pd.DataFrame) -> None:
        """Test mixed signal scenario."""
        # Events: ret = [0.05, -0.03, 0.02, -0.04, 0.01]
        primary_signal = pd.Series([1, -1, 0, 1, -1])

        meta_labels = get_meta_labels(sample_events, primary_signal)

        # idx 0: Long, ret=0.05 > 0 -> correct -> 1
        assert meta_labels.iloc[0] == 1

        # idx 1: Short, ret=-0.03 < 0 -> correct -> 1
        assert meta_labels.iloc[1] == 1

        # idx 2: Neutral -> NaN
        assert pd.isna(meta_labels.iloc[2])

        # idx 3: Long, ret=-0.04 < 0 -> wrong -> 0
        assert meta_labels.iloc[3] == 0

        # idx 4: Short, ret=0.01 > 0 -> wrong -> 0
        assert meta_labels.iloc[4] == 0


class TestGetMetaFeatures:
    """Tests for get_meta_features function."""

    def test_add_primary_signal(self) -> None:
        """Test adding primary signal to features."""
        X = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
        })
        primary_signal = pd.Series([1, -1, 0])

        meta_features = get_meta_features(X, primary_signal)

        assert "primary_signal" in meta_features.columns
        assert list(meta_features["primary_signal"]) == [1, -1, 0]
        assert "feature1" in meta_features.columns
        assert "feature2" in meta_features.columns

    def test_add_primary_proba(self) -> None:
        """Test adding primary probabilities to features."""
        X = pd.DataFrame({
            "feature1": [1, 2, 3],
        })
        primary_signal = pd.Series([1, -1, 0])
        primary_proba = pd.DataFrame({
            "class_-1": [0.2, 0.7, 0.3],
            "class_0": [0.3, 0.2, 0.4],
            "class_1": [0.5, 0.1, 0.3],
        })

        meta_features = get_meta_features(X, primary_signal, primary_proba)

        # Check probability columns
        assert "primary_proba_class_-1" in meta_features.columns
        assert "primary_proba_class_0" in meta_features.columns
        assert "primary_proba_class_1" in meta_features.columns
        assert "primary_confidence" in meta_features.columns

        # Check confidence (max probability)
        assert meta_features["primary_confidence"].iloc[0] == 0.5
        assert meta_features["primary_confidence"].iloc[1] == 0.7


class TestComputeStrategyReturns:
    """Tests for compute_strategy_returns function."""

    def test_primary_only_returns(self) -> None:
        """Test strategy returns with primary signal only."""
        events = pd.DataFrame({
            "ret": [0.05, -0.03, 0.02],
        })
        primary_signal = pd.Series([1, 1, -1])  # Long, Long, Short

        result = compute_strategy_returns(events, primary_signal)

        # Primary returns = ret * signal
        # [0.05 * 1, -0.03 * 1, 0.02 * -1] = [0.05, -0.03, -0.02]
        assert np.isclose(result["primary_return"].iloc[0], 0.05)
        assert np.isclose(result["primary_return"].iloc[1], -0.03)
        assert np.isclose(result["primary_return"].iloc[2], -0.02)

    def test_meta_filtered_returns(self) -> None:
        """Test strategy returns with meta filtering."""
        events = pd.DataFrame({
            "ret": [0.05, -0.03, 0.02],
        })
        primary_signal = pd.Series([1, 1, -1])
        meta_signal = pd.Series([1, 0, 1])  # Trade, No trade, Trade

        result = compute_strategy_returns(events, primary_signal, meta_signal)

        # Meta returns = ret * signal * meta
        # [0.05 * 1 * 1, -0.03 * 1 * 0, 0.02 * -1 * 1] = [0.05, 0, -0.02]
        assert np.isclose(result["meta_return"].iloc[0], 0.05)
        assert np.isclose(result["meta_return"].iloc[1], 0.0)
        assert np.isclose(result["meta_return"].iloc[2], -0.02)


class TestComputeSharpeRatio:
    """Tests for compute_sharpe_ratio function."""

    def test_positive_sharpe(self) -> None:
        """Test Sharpe ratio with positive returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        sharpe = compute_sharpe_ratio(returns)

        # Positive mean with small std -> positive Sharpe
        assert sharpe > 0

    def test_negative_sharpe(self) -> None:
        """Test Sharpe ratio with negative returns."""
        returns = np.array([-0.01, -0.02, -0.03, -0.01, -0.02])
        sharpe = compute_sharpe_ratio(returns)

        # Negative mean -> negative Sharpe
        assert sharpe < 0

    def test_zero_std(self) -> None:
        """Test Sharpe ratio with zero standard deviation."""
        returns = np.array([0.01, 0.01, 0.01])  # Constant returns
        sharpe = compute_sharpe_ratio(returns)

        # Zero std -> return 0
        assert sharpe == 0.0

    def test_nan_handling(self) -> None:
        """Test Sharpe ratio with NaN values."""
        returns = np.array([0.01, np.nan, 0.02, np.nan, 0.03])
        sharpe = compute_sharpe_ratio(returns)

        # Should handle NaN gracefully
        assert np.isfinite(sharpe)

    def test_insufficient_data(self) -> None:
        """Test Sharpe ratio with insufficient data."""
        returns = np.array([0.01])  # Only one return
        sharpe = compute_sharpe_ratio(returns)

        # Insufficient data -> return 0
        assert sharpe == 0.0

    def test_annualization(self) -> None:
        """Test Sharpe ratio with annualization factor."""
        returns = np.array([0.01, 0.02, 0.01, 0.02])
        sharpe_daily = compute_sharpe_ratio(returns, annualization_factor=1)
        sharpe_annual = compute_sharpe_ratio(returns, annualization_factor=252)

        # Annualized Sharpe should be sqrt(252) times daily
        assert np.isclose(sharpe_annual, sharpe_daily * np.sqrt(252))


class TestComputeStrategyMetrics:
    """Tests for compute_strategy_metrics function."""

    def test_compute_metrics(self) -> None:
        """Test computing strategy metrics."""
        strategy_returns = pd.DataFrame({
            "return": [0.05, -0.03, 0.02],
            "signal": [1, 1, -1],
            "primary_return": [0.05, -0.03, -0.02],
            "meta_return": [0.05, 0.0, -0.02],
        })

        metrics = compute_strategy_metrics(strategy_returns)

        # Check that metrics are computed
        assert "primary_total_return" in metrics
        assert "primary_sharpe" in metrics
        assert "primary_n_trades" in metrics
        assert "primary_win_rate" in metrics
        assert "primary_mean_return" in metrics
