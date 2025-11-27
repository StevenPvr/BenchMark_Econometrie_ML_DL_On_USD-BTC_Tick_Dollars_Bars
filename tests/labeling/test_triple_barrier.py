"""Tests for the triple-barrier labeling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.label_primaire.optimize import normalize_volatility_scale
from src.label_primaire.triple_barrier import (
    get_triple_barrier_events,
    get_vertical_barriers,
    apply_triple_barrier_labels,
)


class TestGetVerticalBarriers:
    """Tests for get_vertical_barriers function."""

    def test_basic_vertical_barriers(self) -> None:
        """Test basic vertical barrier calculation."""
        t_events = pd.RangeIndex(0, 10)
        max_holding = 5
        n_bars = 20

        barriers = get_vertical_barriers(t_events, max_holding, n_bars)

        assert len(barriers) == 10
        assert barriers.iloc[0] == 5  # 0 + 5
        assert barriers.iloc[9] == 14  # 9 + 5

    def test_vertical_barriers_clipping(self) -> None:
        """Test that vertical barriers are clipped to max index."""
        t_events = pd.RangeIndex(0, 10)
        max_holding = 15
        n_bars = 12

        barriers = get_vertical_barriers(t_events, max_holding, n_bars)

        # Last barrier should be clipped to n_bars - 1 = 11
        assert barriers.iloc[0] == 11  # min(0 + 15, 11)
        assert barriers.iloc[9] == 11  # min(9 + 15, 11)


class TestGetTripleBarrierEvents:
    """Tests for get_triple_barrier_events function."""

    @pytest.fixture
    def sample_prices(self) -> pd.Series:
        """Create sample price series."""
        # Simulate a trending then mean-reverting price
        np.random.seed(42)
        n = 100
        returns = np.random.normal(0, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, name="close")

    @pytest.fixture
    def sample_volatility(self, sample_prices: pd.Series) -> pd.Series:
        """Create sample volatility series."""
        # Simple rolling std as volatility
        log_returns = np.log(sample_prices / sample_prices.shift(1))
        vol = log_returns.rolling(10).std()
        return vol

    def test_basic_triple_barrier(
        self, sample_prices: pd.Series, sample_volatility: pd.Series
    ) -> None:
        """Test basic triple-barrier event generation."""
        events = get_triple_barrier_events(
            prices=sample_prices,
            pt_sl=[1.0, 1.0],
            target_volatility=sample_volatility,
            max_holding_period=20,
        )

        # Check output structure
        assert "t_start" in events.columns
        assert "t_end" in events.columns
        assert "ret" in events.columns
        assert "barrier" in events.columns
        assert "label" in events.columns

        # Check barrier values
        assert set(events["barrier"].unique()).issubset({-1, 0, 1})

        # Check label values
        assert set(events["label"].unique()).issubset({-1, 0, 1})

        # Check that t_end >= t_start
        assert (events["t_end"] >= events["t_start"]).all()

    def test_symmetric_barriers(
        self, sample_prices: pd.Series, sample_volatility: pd.Series
    ) -> None:
        """Test symmetric PT/SL barriers."""
        events = get_triple_barrier_events(
            prices=sample_prices,
            pt_sl=[1.0, 1.0],
            target_volatility=sample_volatility,
            max_holding_period=50,
        )

        # With symmetric barriers and random walk, expect roughly equal PT/SL
        n_pt = (events["barrier"] == 1).sum()
        n_sl = (events["barrier"] == -1).sum()

        # Allow for some variance, but should be in same order of magnitude
        assert n_pt > 0 or n_sl > 0  # At least some barrier touches

    def test_asymmetric_barriers(
        self, sample_prices: pd.Series, sample_volatility: pd.Series
    ) -> None:
        """Test asymmetric PT/SL barriers."""
        # Wide PT, tight SL
        events_tight_sl = get_triple_barrier_events(
            prices=sample_prices,
            pt_sl=[2.0, 0.5],  # Wide PT, tight SL
            target_volatility=sample_volatility,
            max_holding_period=50,
        )

        # Tight SL should trigger more often
        n_sl = (events_tight_sl["barrier"] == -1).sum()
        assert n_sl > 0  # Should have some SL hits

    def test_max_holding_period(
        self, sample_prices: pd.Series, sample_volatility: pd.Series
    ) -> None:
        """Test max holding period constraint."""
        max_hold = 10

        events = get_triple_barrier_events(
            prices=sample_prices,
            pt_sl=[5.0, 5.0],  # Very wide barriers
            target_volatility=sample_volatility,
            max_holding_period=max_hold,
        )

        # With very wide barriers, most should hit vertical barrier
        durations = events["t_end"] - events["t_start"]
        assert (durations <= max_hold).all()

    def test_no_volatility_provided(self, sample_prices: pd.Series) -> None:
        """Test automatic volatility computation."""
        events = get_triple_barrier_events(
            prices=sample_prices,
            pt_sl=[1.0, 1.0],
            target_volatility=None,  # Should compute automatically
            max_holding_period=20,
        )

        # Should still generate events
        assert len(events) > 0
        assert "label" in events.columns

    def test_label_consistency(
        self, sample_prices: pd.Series, sample_volatility: pd.Series
    ) -> None:
        """Test that labels match return sign."""
        events = get_triple_barrier_events(
            prices=sample_prices,
            pt_sl=[1.0, 1.0],
            target_volatility=sample_volatility,
            max_holding_period=20,
        )

        # Label should match sign of return
        for _, row in events.iterrows():
            if row["ret"] > 0:
                assert row["label"] == 1
            elif row["ret"] < 0:
                assert row["label"] == -1
            else:
                assert row["label"] == 0


class TestApplyTripleBarrierLabels:
    """Tests for apply_triple_barrier_labels function."""

    def test_apply_labels_to_dataframe(self) -> None:
        """Test applying labels to a DataFrame."""
        # Create sample DataFrame
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "close": 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n))),
            "volume": np.random.uniform(100, 1000, n),
        })

        # Apply labels
        df_labeled = apply_triple_barrier_labels(
            df=df,
            price_col="close",
            pt_sl=[1.0, 1.0],
            max_holding_period=10,
        )

        # Check that label column was added
        assert "label" in df_labeled.columns

        # Check that labels are valid
        valid_labels = df_labeled["label"].dropna()
        if len(valid_labels) > 0:
            assert set(valid_labels.unique()).issubset({-1, 0, 1})


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_price_series(self) -> None:
        """Test with very short price series."""
        prices = pd.Series([100, 101, 102])
        vol = pd.Series([0.02, 0.02, 0.02])

        events = get_triple_barrier_events(
            prices=prices,
            pt_sl=[1.0, 1.0],
            target_volatility=vol,
            max_holding_period=5,
        )

        # Should handle gracefully
        assert isinstance(events, pd.DataFrame)

    def test_constant_prices(self) -> None:
        """Test with constant prices."""
        prices = pd.Series([100] * 50)
        vol = pd.Series([0.02] * 50)

        events = get_triple_barrier_events(
            prices=prices,
            pt_sl=[1.0, 1.0],
            target_volatility=vol,
            max_holding_period=10,
        )

        # All should hit vertical barrier (no price movement)
        if len(events) > 0:
            # Most should be vertical barrier or zero return
            assert (events["ret"].abs() < 1e-10).all() or (events["barrier"] == 0).all()

    def test_nan_in_volatility(self) -> None:
        """Test handling of NaN in volatility."""
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.02, 50))))

        vol = pd.Series([np.nan] * 10 + [0.02] * 40)

        events = get_triple_barrier_events(
            prices=prices,
            pt_sl=[1.0, 1.0],
            target_volatility=vol,
            max_holding_period=10,
        )

        # Should skip events where volatility is NaN
        assert isinstance(events, pd.DataFrame)


class TestNormalizeVolatilityScale:
    """Tests for normalize_volatility_scale helper."""

    def test_rescales_percent_volatility(self) -> None:
        """Volatility much larger than price returns should be downscaled."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0])
        volatility = pd.Series([50.0, 60.0, 55.0, 65.0])

        adjusted = normalize_volatility_scale(volatility, prices)

        expected = volatility / 100.0
        pd.testing.assert_series_equal(adjusted.reset_index(drop=True), expected.reset_index(drop=True))

    def test_keeps_aligned_scale(self) -> None:
        """Volatility already aligned with returns should stay unchanged."""
        prices = pd.Series([100.0, 100.5, 101.0, 100.8, 101.5])
        volatility = pd.Series([0.01, 0.02, 0.015, 0.018, 0.02])

        adjusted = normalize_volatility_scale(volatility, prices)

        pd.testing.assert_series_equal(adjusted.reset_index(drop=True), volatility.reset_index(drop=True))
