"""Unit tests for triple_barriere.relabel_datasets module.

Tests cover:
- Label computation
- Class proportion validation
- Economic metrics computation
- No look-ahead bias in optimization
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.labelling.triple_barriere.relabel_datasets import (
    check_class_proportions,
    compute_economic_metrics,
    compute_label_stats,
    compute_labels,
    get_feature_columns,
    load_features,
    MIN_CLASS_RATIO,
    MAX_CLASS_RATIO,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create sample features DataFrame."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
            "split": ["train"] * 150 + ["test"] * 50,
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_close() -> pd.Series:
    """Create sample close price series."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sample_volatility(sample_close: pd.Series) -> pd.Series:
    """Compute EWM volatility from close prices."""
    log_returns = np.log(sample_close / sample_close.shift(1))
    return log_returns.ewm(span=20, min_periods=5).std()


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Create sample events DataFrame with labels and returns."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t1_dates = dates + pd.Timedelta(hours=10)
    labels = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    returns = rng.standard_normal(n) * 0.02
    return pd.DataFrame(
        {
            "t1": t1_dates,
            "label": labels,
            "ret": returns,
            "trgt": np.abs(rng.standard_normal(n)) * 0.01,
        },
        index=dates,
    )


# =============================================================================
# TESTS: get_feature_columns
# =============================================================================


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_excludes_metadata_columns(self) -> None:
        """Metadata columns are excluded from feature list."""
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
                "bar_id": [1, 2, 3],
                "split": ["train", "train", "test"],
                "label": [0, 1, -1],
                "datetime_close": pd.date_range("2024-01-01", periods=3),
            }
        )

        features = get_feature_columns(df)

        assert "feature_1" in features
        assert "feature_2" in features
        assert "bar_id" not in features
        assert "split" not in features
        assert "label" not in features
        assert "datetime_close" not in features

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame()
        features = get_feature_columns(df)
        assert features == []


# =============================================================================
# TESTS: check_class_proportions
# =============================================================================


class TestCheckClassProportions:
    """Tests for check_class_proportions function."""

    def test_balanced_proportions_pass(self) -> None:
        """Balanced class proportions pass validation."""
        n = 100
        labels = (
            [-1] * 30 + [0] * 40 + [1] * 30
        )  # 30%, 40%, 30%
        df = pd.DataFrame({"label": labels})

        result = check_class_proportions(df)

        assert result is True

    def test_imbalanced_proportions_fail(self) -> None:
        """Imbalanced class proportions fail validation."""
        n = 100
        labels = (
            [-1] * 10 + [0] * 80 + [1] * 10
        )  # 10%, 80%, 10%
        df = pd.DataFrame({"label": labels})

        result = check_class_proportions(df)

        assert result is False

    def test_missing_class_fails(self) -> None:
        """Missing class fails validation."""
        df = pd.DataFrame({"label": [0, 1, 1, 0, 1]})  # No -1 class

        result = check_class_proportions(df)

        assert result is False

    def test_empty_labels_fail(self) -> None:
        """Empty labels fail validation."""
        df = pd.DataFrame({"label": []})

        result = check_class_proportions(df)

        assert result is False

    def test_no_label_column_fails(self) -> None:
        """Missing label column fails validation."""
        df = pd.DataFrame({"other": [1, 2, 3]})

        result = check_class_proportions(df)

        assert result is False


# =============================================================================
# TESTS: compute_label_stats
# =============================================================================


class TestComputeLabelStats:
    """Tests for compute_label_stats function."""

    def test_computes_correct_stats(self) -> None:
        """Compute correct statistics for label distribution."""
        labels = pd.Series([-1, -1, 0, 0, 0, 1, 1, 1, 1, np.nan])

        stats = compute_label_stats(labels)

        assert stats["total"] == 9  # NaN excluded
        assert stats["count_-1"] == 2
        assert stats["count_0"] == 3
        assert stats["count_1"] == 4
        assert stats["pct_-1"] == pytest.approx(22.22, rel=0.01)
        assert stats["pct_0"] == pytest.approx(33.33, rel=0.01)
        assert stats["pct_1"] == pytest.approx(44.44, rel=0.01)

    def test_empty_series(self) -> None:
        """Empty series returns zero counts."""
        labels = pd.Series([], dtype=float)

        stats = compute_label_stats(labels)

        assert stats["total"] == 0


# =============================================================================
# TESTS: compute_economic_metrics
# =============================================================================


class TestComputeEconomicMetrics:
    """Tests for compute_economic_metrics function."""

    def test_computes_metrics_correctly(self, sample_events_df: pd.DataFrame) -> None:
        """Compute economic metrics from events."""
        close = pd.Series(100.0, index=sample_events_df.index)

        metrics = compute_economic_metrics(sample_events_df, close)

        assert "sharpe_per_trade" in metrics
        assert "total_return_pct" in metrics
        assert "net_pnl" in metrics
        assert "n_trades" in metrics
        assert "win_rate" in metrics
        assert 0 <= metrics["win_rate"] <= 1

    def test_no_trades_returns_zeros(self) -> None:
        """Zero trades returns zero metrics."""
        # All labels are 0 (no trade)
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        events = pd.DataFrame(
            {
                "t1": dates + pd.Timedelta(hours=5),
                "label": [0] * 20,
                "ret": np.zeros(20),
            },
            index=dates,
        )
        close = pd.Series(100.0, index=dates)

        metrics = compute_economic_metrics(events, close)

        assert metrics["n_trades"] == 0
        assert metrics["total_return_pct"] == 0.0

    def test_too_few_trades_returns_zeros(self) -> None:
        """Fewer than 10 trades returns zero metrics."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        events = pd.DataFrame(
            {
                "t1": dates + pd.Timedelta(hours=5),
                "label": [1, -1, 1, -1, 1],
                "ret": [0.01] * 5,
            },
            index=dates,
        )
        close = pd.Series(100.0, index=dates)

        metrics = compute_economic_metrics(events, close)

        assert metrics["n_trades"] == 0
        assert metrics["sharpe_per_trade"] == 0.0

    def test_sharpe_ratio_positive_for_positive_returns(self) -> None:
        """Sharpe ratio is positive for consistently positive returns."""
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        # Create positive returns with some variance
        rng = np.random.default_rng(42)
        returns = 0.02 + rng.standard_normal(20) * 0.005  # Mean ~2%, small variance
        events = pd.DataFrame(
            {
                "t1": dates + pd.Timedelta(hours=5),
                "label": [1] * 20,  # All long trades
                "ret": returns,
            },
            index=dates,
        )
        close = pd.Series(100.0, index=dates)

        metrics = compute_economic_metrics(events, close, cost_pct=0.0)

        # With positive mean returns, Sharpe should be positive
        assert metrics["sharpe_per_trade"] > 0

    def test_sharpe_ratio_with_variance(self) -> None:
        """Sharpe ratio accounts for return variance."""
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        # Create alternating returns
        returns = [0.03, 0.01] * 10  # Mean = 0.02, has variance
        events = pd.DataFrame(
            {
                "t1": dates + pd.Timedelta(hours=5),
                "label": [1] * 20,
                "ret": returns,
            },
            index=dates,
        )
        close = pd.Series(100.0, index=dates)

        metrics = compute_economic_metrics(events, close, cost_pct=0.0)

        # Sharpe = mean / std * sqrt(n)
        # With variance, Sharpe should be positive and non-zero
        assert metrics["sharpe_per_trade"] > 0
        assert metrics["n_trades"] == 20


# =============================================================================
# TESTS: compute_labels
# =============================================================================


class TestComputeLabels:
    """Tests for compute_labels function."""

    def test_computes_labels(
        self,
        sample_features_df: pd.DataFrame,
        sample_close: pd.Series,
        sample_volatility: pd.Series,
    ) -> None:
        """Compute labels from features and close prices."""
        labeled_df, events = compute_labels(
            features_df=sample_features_df,
            close=sample_close,
            volatility=sample_volatility,
            sl_mult=1.0,
            min_return=0.001,
            max_holding=15,
        )

        assert "label" in labeled_df.columns
        assert isinstance(events, pd.DataFrame)

    def test_labels_in_valid_range(
        self,
        sample_features_df: pd.DataFrame,
        sample_close: pd.Series,
        sample_volatility: pd.Series,
    ) -> None:
        """All labels must be in {-1, 0, 1, NaN}."""
        labeled_df, _ = compute_labels(
            features_df=sample_features_df,
            close=sample_close,
            volatility=sample_volatility,
            sl_mult=1.0,
            min_return=0.001,
            max_holding=15,
        )

        valid_labels = {-1, 0, 1}
        non_nan_labels = labeled_df["label"].dropna()
        assert all(label in valid_labels for label in non_nan_labels)

    def test_no_look_ahead_bias(
        self,
        sample_features_df: pd.DataFrame,
        sample_close: pd.Series,
        sample_volatility: pd.Series,
    ) -> None:
        """Verify no look-ahead bias in labels.

        For each event, the exit time (t1) must be >= entry time (event index).
        """
        _, events = compute_labels(
            features_df=sample_features_df,
            close=sample_close,
            volatility=sample_volatility,
            sl_mult=1.0,
            min_return=0.001,
            max_holding=15,
        )

        for t0, row in events.iterrows():
            t1 = row["t1"]
            if pd.notna(t1):
                assert t1 >= t0, f"Look-ahead bias: t1={t1} < t0={t0}"


# =============================================================================
# TESTS: load_features
# =============================================================================


class TestLoadFeatures:
    """Tests for load_features function."""

    def test_file_not_found_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        """Non-existent file raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.parquet"  # type: ignore

        with pytest.raises(FileNotFoundError):
            load_features(fake_path)

    def test_sets_datetime_index(self, tmp_path: pytest.TempPathFactory) -> None:
        """Sets datetime_close as index if present."""
        path = tmp_path / "test.parquet"  # type: ignore
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "datetime_close": dates,
                "feature_1": rng.standard_normal(10),
            }
        )
        df.to_parquet(path)

        result = load_features(path)

        # Index values should match, ignore name attribute
        np.testing.assert_array_equal(result.index.values, dates.values)
        assert "datetime_close" not in result.columns


# =============================================================================
# TESTS: Integration - No Look-Ahead Bias
# =============================================================================


class TestNoLookAheadBias:
    """Integration tests verifying no look-ahead bias."""

    def test_volatility_is_causal(self, sample_close: pd.Series) -> None:
        """Volatility estimation uses only past data (EWM is causal)."""
        # EWM with span=20 uses exponential weights on past data only
        log_returns = np.log(sample_close / sample_close.shift(1))
        volatility = log_returns.ewm(span=20, min_periods=5).std()

        # Verify that volatility at time t only depends on data up to t
        # by checking that modifying future data doesn't change past volatility
        modified_close = sample_close.copy()
        modified_close.iloc[100:] = modified_close.iloc[100:] * 2  # Modify future

        modified_log_returns = np.log(modified_close / modified_close.shift(1))
        modified_vol = modified_log_returns.ewm(span=20, min_periods=5).std()

        # Volatility before modification point should be identical
        np.testing.assert_array_almost_equal(
            volatility.iloc[:100].values,
            modified_vol.iloc[:100].values,
            decimal=10,
        )

    def test_barrier_labels_are_causal(
        self,
        sample_features_df: pd.DataFrame,
        sample_close: pd.Series,
        sample_volatility: pd.Series,
    ) -> None:
        """Barrier labels only use information available at event time."""
        _, events = compute_labels(
            features_df=sample_features_df,
            close=sample_close,
            volatility=sample_volatility,
            sl_mult=1.0,
            min_return=0.001,
            max_holding=15,
        )

        # Check that t1 is always in the future relative to event time
        for t0, row in events.iterrows():
            t1 = row["t1"]
            if pd.notna(t1):
                # t1 must be strictly after t0 (or equal for edge cases)
                assert t1 >= t0

                # The return is computed between t0 and t1
                # which is a forward-looking but non-leaking operation
                # (we're labeling based on what happens AFTER the event)
                ret = row["ret"]
                if pd.notna(ret):
                    # Return should be consistent with price movement
                    if t0 in sample_close.index and t1 in sample_close.index:
                        expected_ret = (
                            sample_close.loc[t1] - sample_close.loc[t0]
                        ) / sample_close.loc[t0]
                        assert abs(ret - expected_ret) < 1e-6
