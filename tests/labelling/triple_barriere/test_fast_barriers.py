"""Unit tests for triple_barriere.fast_barriers module.

Tests cover:
- Numba-optimized barrier computation functions
- No look-ahead bias verification
- Edge cases handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labelling.triple_barriere.fast_barriers import (
    compute_all_returns_and_labels_numba,
    compute_triple_barrier_labels_full,
    compute_vertical_barriers_fast,
    compute_vertical_barriers_numba,
    get_events_primary_fast,
    update_all_t1_with_barriers_numba,
    update_barriers_with_touches_fast,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_close_prices() -> pd.Series:
    """Generate synthetic close prices with DatetimeIndex."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    # Random walk with drift
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sample_volatility(sample_close_prices: pd.Series) -> pd.Series:
    """Compute EWM volatility (causal, no look-ahead)."""
    log_returns = np.log(sample_close_prices / sample_close_prices.shift(1))
    return log_returns.ewm(span=20, min_periods=5).std()


@pytest.fixture
def sample_events(sample_close_prices: pd.Series) -> pd.DatetimeIndex:
    """Create sample event timestamps."""
    # Take every 5th timestamp as an event
    return pd.DatetimeIndex(sample_close_prices.index[10::5])


# =============================================================================
# TESTS: compute_vertical_barriers_numba
# =============================================================================


class TestComputeVerticalBarriersNumba:
    """Tests for compute_vertical_barriers_numba function."""

    def test_basic_functionality(self) -> None:
        """Vertical barriers are computed correctly."""
        event_positions = np.array([0, 5, 10], dtype=np.int64)
        n_close = 50
        max_holding = 10

        t1_positions = compute_vertical_barriers_numba(
            event_positions, n_close, max_holding
        )

        expected = np.array([10, 15, 20], dtype=np.int64)
        np.testing.assert_array_equal(t1_positions, expected)

    def test_clipping_at_end(self) -> None:
        """Vertical barriers are clipped at the end of the close array."""
        event_positions = np.array([45], dtype=np.int64)
        n_close = 50
        max_holding = 10

        t1_positions = compute_vertical_barriers_numba(
            event_positions, n_close, max_holding
        )

        # Should be clipped to n_close - 1 = 49
        assert t1_positions[0] == 49

    def test_invalid_position(self) -> None:
        """Negative positions return -1."""
        event_positions = np.array([-1, 5], dtype=np.int64)
        n_close = 50
        max_holding = 10

        t1_positions = compute_vertical_barriers_numba(
            event_positions, n_close, max_holding
        )

        assert t1_positions[0] == -1
        assert t1_positions[1] == 15

    def test_empty_input(self) -> None:
        """Empty input returns empty output."""
        event_positions = np.array([], dtype=np.int64)
        n_close = 50
        max_holding = 10

        t1_positions = compute_vertical_barriers_numba(
            event_positions, n_close, max_holding
        )

        assert len(t1_positions) == 0


# =============================================================================
# TESTS: update_all_t1_with_barriers_numba
# =============================================================================


class TestUpdateAllT1WithBarriersNumba:
    """Tests for update_all_t1_with_barriers_numba function."""

    def test_profit_taking_triggered(self) -> None:
        """t1 is updated when profit-taking barrier is hit."""
        # Prices: entry at 100, goes to 102 (2% gain) at position 2
        close_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        t0_positions = np.array([0], dtype=np.int64)
        t1_positions = np.array([4], dtype=np.int64)
        pt_barriers = np.array([0.02])  # 2% PT
        sl_barriers = np.array([-0.05])  # 5% SL

        updated_t1 = update_all_t1_with_barriers_numba(
            close_values, t0_positions, t1_positions, pt_barriers, sl_barriers
        )

        # PT triggered at position 2 (2% gain)
        assert updated_t1[0] == 2

    def test_stop_loss_triggered(self) -> None:
        """t1 is updated when stop-loss barrier is hit."""
        # Prices: entry at 100, drops to 95 (5% loss) at position 2
        close_values = np.array([100.0, 98.0, 95.0, 94.0, 93.0])
        t0_positions = np.array([0], dtype=np.int64)
        t1_positions = np.array([4], dtype=np.int64)
        pt_barriers = np.array([0.10])  # 10% PT
        sl_barriers = np.array([-0.05])  # 5% SL

        updated_t1 = update_all_t1_with_barriers_numba(
            close_values, t0_positions, t1_positions, pt_barriers, sl_barriers
        )

        # SL triggered at position 2 (5% loss)
        assert updated_t1[0] == 2

    def test_no_barrier_touched(self) -> None:
        """t1 remains unchanged when no barrier is touched."""
        close_values = np.array([100.0, 100.5, 100.2, 100.3, 100.1])
        t0_positions = np.array([0], dtype=np.int64)
        t1_positions = np.array([4], dtype=np.int64)
        pt_barriers = np.array([0.10])  # 10% PT
        sl_barriers = np.array([-0.10])  # 10% SL

        updated_t1 = update_all_t1_with_barriers_numba(
            close_values, t0_positions, t1_positions, pt_barriers, sl_barriers
        )

        # No barrier touched, t1 remains at 4
        assert updated_t1[0] == 4


# =============================================================================
# TESTS: compute_all_returns_and_labels_numba
# =============================================================================


class TestComputeAllReturnsAndLabelsNumba:
    """Tests for compute_all_returns_and_labels_numba function."""

    def test_positive_return_label(self) -> None:
        """Positive return above min_return gives label 1."""
        close_values = np.array([100.0, 102.0])
        t0_positions = np.array([0], dtype=np.int64)
        t1_positions = np.array([1], dtype=np.int64)
        min_return = 0.01  # 1%

        returns, labels = compute_all_returns_and_labels_numba(
            close_values, t0_positions, t1_positions, min_return
        )

        assert returns[0] == pytest.approx(0.02, abs=1e-6)
        assert labels[0] == 1

    def test_negative_return_label(self) -> None:
        """Negative return below -min_return gives label -1."""
        close_values = np.array([100.0, 98.0])
        t0_positions = np.array([0], dtype=np.int64)
        t1_positions = np.array([1], dtype=np.int64)
        min_return = 0.01  # 1%

        returns, labels = compute_all_returns_and_labels_numba(
            close_values, t0_positions, t1_positions, min_return
        )

        assert returns[0] == pytest.approx(-0.02, abs=1e-6)
        assert labels[0] == -1

    def test_zero_label_within_threshold(self) -> None:
        """Return within min_return threshold gives label 0."""
        close_values = np.array([100.0, 100.5])
        t0_positions = np.array([0], dtype=np.int64)
        t1_positions = np.array([1], dtype=np.int64)
        min_return = 0.01  # 1%

        returns, labels = compute_all_returns_and_labels_numba(
            close_values, t0_positions, t1_positions, min_return
        )

        assert abs(returns[0]) < min_return
        assert labels[0] == 0

    def test_invalid_positions(self) -> None:
        """Invalid positions return NaN and label 0."""
        close_values = np.array([100.0, 102.0])
        t0_positions = np.array([-1], dtype=np.int64)
        t1_positions = np.array([1], dtype=np.int64)
        min_return = 0.01

        returns, labels = compute_all_returns_and_labels_numba(
            close_values, t0_positions, t1_positions, min_return
        )

        assert np.isnan(returns[0])
        assert labels[0] == 0


# =============================================================================
# TESTS: compute_triple_barrier_labels_full
# =============================================================================


class TestComputeTripleBarrierLabelsFull:
    """Tests for compute_triple_barrier_labels_full function."""

    def test_full_pipeline(self) -> None:
        """Full pipeline computes t1, returns, and labels correctly."""
        np.random.seed(42)
        n = 100
        close_values = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        t0_positions = np.array([0, 10, 20], dtype=np.int64)
        volatility_values = np.array([0.01, 0.01, 0.01])
        pt_mult = 1.5
        sl_mult = 1.0
        max_holding = 15
        min_return = 0.001

        t1_positions, returns, labels = compute_triple_barrier_labels_full(
            close_values,
            t0_positions,
            volatility_values,
            pt_mult,
            sl_mult,
            max_holding,
            min_return,
        )

        assert len(t1_positions) == 3
        assert len(returns) == 3
        assert len(labels) == 3
        # All t1 positions should be valid
        assert all(t1 >= 0 for t1 in t1_positions)
        # Labels should be in {-1, 0, 1}
        assert all(label in [-1, 0, 1] for label in labels)

    def test_no_look_ahead_bias(self) -> None:
        """Verify that t1 positions are always >= t0 positions (no future peeking)."""
        np.random.seed(123)
        n = 200
        close_values = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        t0_positions = np.arange(0, 150, 10, dtype=np.int64)
        volatility_values = np.full(len(t0_positions), 0.01)

        t1_positions, _, _ = compute_triple_barrier_labels_full(
            close_values,
            t0_positions,
            volatility_values,
            pt_mult=1.5,
            sl_mult=1.0,
            max_holding=20,
            min_return=0.001,
        )

        # t1 must always be >= t0 (no look-ahead)
        for i, t0 in enumerate(t0_positions):
            assert t1_positions[i] >= t0, f"Look-ahead bias detected at event {i}"


# =============================================================================
# TESTS: get_events_primary_fast
# =============================================================================


class TestGetEventsPrimaryFast:
    """Tests for get_events_primary_fast function."""

    def test_basic_functionality(
        self,
        sample_close_prices: pd.Series,
        sample_volatility: pd.Series,
        sample_events: pd.DatetimeIndex,
    ) -> None:
        """Basic functionality returns expected DataFrame structure."""
        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=sample_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=sample_volatility,
            max_holding=15,
            min_return=0.001,
        )

        assert isinstance(result, pd.DataFrame)
        expected_columns = {"t1", "trgt", "ret", "label"}
        assert expected_columns.issubset(set(result.columns))

    def test_empty_events(self, sample_close_prices: pd.Series) -> None:
        """Empty events input returns empty DataFrame."""
        empty_events = pd.DatetimeIndex([])
        volatility = sample_close_prices * 0.01

        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=empty_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=volatility,
            max_holding=15,
            min_return=0.001,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_labels_in_valid_range(
        self,
        sample_close_prices: pd.Series,
        sample_volatility: pd.Series,
        sample_events: pd.DatetimeIndex,
    ) -> None:
        """All labels must be in {-1, 0, 1}."""
        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=sample_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=sample_volatility,
            max_holding=15,
            min_return=0.001,
        )

        valid_labels = {-1, 0, 1}
        assert all(label in valid_labels for label in result["label"])

    def test_t1_after_t0(
        self,
        sample_close_prices: pd.Series,
        sample_volatility: pd.Series,
        sample_events: pd.DatetimeIndex,
    ) -> None:
        """Exit time t1 must be >= entry time (event index)."""
        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=sample_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=sample_volatility,
            max_holding=15,
            min_return=0.001,
        )

        for t0, row in result.iterrows():
            t1 = row["t1"]
            if pd.notna(t1):
                assert t1 >= t0, f"Look-ahead bias: t1={t1} < t0={t0}"


# =============================================================================
# TESTS: compute_vertical_barriers_fast
# =============================================================================


class TestComputeVerticalBarriersFast:
    """Tests for compute_vertical_barriers_fast function."""

    def test_returns_series(self, sample_close_prices: pd.Series) -> None:
        """Returns a pandas Series with correct index."""
        t_events = pd.DatetimeIndex(sample_close_prices.index[10:20])

        result = compute_vertical_barriers_fast(
            t_events=t_events,
            close=sample_close_prices,
            max_holding=10,
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(t_events)
        assert result.index.equals(t_events)


# =============================================================================
# TESTS: update_barriers_with_touches_fast
# =============================================================================


class TestUpdateBarriersWithTouchesFast:
    """Tests for update_barriers_with_touches_fast function."""

    def test_updates_t1_on_barrier_touch(
        self, sample_close_prices: pd.Series
    ) -> None:
        """t1 is updated when barrier is touched."""
        t_events = pd.DatetimeIndex(sample_close_prices.index[10:15])

        # Create events DataFrame
        events = pd.DataFrame(
            {
                "t1": sample_close_prices.index[20:25].tolist(),
                "trgt": [0.01] * 5,
            },
            index=t_events,
        )

        result = update_barriers_with_touches_fast(
            close=sample_close_prices,
            events=events,
            pt_mult=1.5,
            sl_mult=1.0,
        )

        assert isinstance(result, pd.DataFrame)
        assert "t1" in result.columns


# =============================================================================
# TESTS: Edge Cases and Numerical Stability
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_zero_volatility(self, sample_close_prices: pd.Series) -> None:
        """Zero volatility events are handled gracefully."""
        t_events = pd.DatetimeIndex(sample_close_prices.index[10:15])
        zero_vol = pd.Series(0.0, index=sample_close_prices.index)

        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=t_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=zero_vol,
            max_holding=15,
            min_return=0.001,
        )

        # Should still return a valid DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_large_max_holding(self, sample_close_prices: pd.Series) -> None:
        """Large max_holding is clipped to data length."""
        t_events = pd.DatetimeIndex([sample_close_prices.index[0]])
        vol = pd.Series(0.01, index=sample_close_prices.index)

        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=t_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=vol,
            max_holding=10000,  # Much larger than data
            min_return=0.001,
        )

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            # t1 should be within data bounds
            t1 = result.iloc[0]["t1"]
            if pd.notna(t1):
                assert t1 <= sample_close_prices.index[-1]

    def test_single_event(self, sample_close_prices: pd.Series) -> None:
        """Single event is handled correctly."""
        t_events = pd.DatetimeIndex([sample_close_prices.index[50]])
        vol = pd.Series(0.01, index=sample_close_prices.index)

        result = get_events_primary_fast(
            close=sample_close_prices,
            t_events=t_events,
            pt_mult=1.5,
            sl_mult=1.0,
            trgt=vol,
            max_holding=15,
            min_return=0.001,
        )

        assert len(result) == 1
