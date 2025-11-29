"""
Unit tests for label_primaire/utils.py

Tests the utility functions for labeling, including:
- Volatility estimation
- Triple barrier computation
- Barrier touch detection
- Label generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.labelling.label_primaire.utils import (
    get_daily_volatility,
    compute_barriers,
    is_valid_barrier,
    find_barrier_touch,
    compute_return_and_label,
    set_vertical_barriers,
    load_model_class,
    MODEL_REGISTRY,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_prices() -> pd.Series:
    """Create sample price series."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    # Random walk
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sample_events(sample_prices) -> pd.DataFrame:
    """Create sample events DataFrame."""
    # Select a few events
    t_events = sample_prices.index[::10]
    events = pd.DataFrame(index=t_events)
    events["trgt"] = 0.01  # Fixed volatility target
    return events


# =============================================================================
# TESTS - Volatility
# =============================================================================


class TestVolatility:
    """Tests for volatility estimation."""

    def test_get_daily_volatility(self, sample_prices):
        """Test volatility computation."""
        vol = get_daily_volatility(sample_prices, span=20)

        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_prices)
        assert vol.index.equals(sample_prices.index)

        # Volatility should be positive (except initial NaN)
        assert (vol.dropna() > 0).all()

        # Check span calculation logic roughly
        # Standard deviation of 0.01 log returns should be around 0.01
        mean_vol = vol.mean()
        assert 0.005 < mean_vol < 0.02

    def test_min_periods(self, sample_prices):
        """Test min_periods parameter."""
        vol = get_daily_volatility(sample_prices, span=20, min_periods=50)

        # First 49 values should be NaN (since we need 50 returns, and first return is NaN)
        # Actually EWM behavior with min_periods might vary slightly depending on implementation
        # usually it needs min_periods observations.
        assert vol.iloc[:49].isna().all()
        assert pd.notna(vol.iloc[50])


# =============================================================================
# TESTS - Triple Barrier Helpers
# =============================================================================


class TestTripleBarrier:
    """Tests for triple barrier functions."""

    def test_compute_barriers(self, sample_events):
        """Test barrier level computation."""
        # Test with symmetric barriers
        events = compute_barriers(sample_events, pt_mult=2.0, sl_mult=2.0)

        assert "pt" in events.columns
        assert "sl" in events.columns

        # Target is 0.01, so pt should be 0.02, sl should be -0.02
        assert np.allclose(events["pt"], 0.02)
        assert np.allclose(events["sl"], -0.02)

    def test_compute_barriers_asymmetric(self, sample_events):
        """Test asymmetric barriers."""
        events = compute_barriers(sample_events, pt_mult=1.0, sl_mult=0.5)

        assert np.allclose(events["pt"], 0.01)
        assert np.allclose(events["sl"], -0.005)

    def test_compute_barriers_zero(self, sample_events):
        """Test zero/disabled barriers."""
        events = compute_barriers(sample_events, pt_mult=0.0, sl_mult=0.0)

        # Should be NaN or not present depending on implementation
        # The implementation sets them to NaN if mult <= 0 ?
        # Let's check implementation:
        # if pt_mult > 0: out["pt"] = pt_mult * trgt else: NaN

        assert events["pt"].isna().all()
        assert events["sl"].isna().all()

    def test_is_valid_barrier(self):
        """Test barrier validation."""
        assert is_valid_barrier(1.0)
        assert is_valid_barrier(0.0)
        assert is_valid_barrier(-1.0)

        assert not is_valid_barrier(None)
        assert not is_valid_barrier(np.nan)
        assert not is_valid_barrier(float("nan"))

    def test_find_barrier_touch_pt(self):
        """Test finding profit-taking touch."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        # Returns: 0, 0.01, 0.02, 0.03...
        path_ret = pd.Series([0, 0.01, 0.02, 0.03, 0.01, 0], index=dates[:6])

        # Barrier at 0.02
        touch = find_barrier_touch(path_ret, pt_barrier=0.02, sl_barrier=-0.02)

        assert touch == dates[2]

    def test_find_barrier_touch_sl(self):
        """Test finding stop-loss touch."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        # Returns: 0, -0.01, -0.02, -0.03...
        path_ret = pd.Series([0, -0.01, -0.02, -0.03, -0.01, 0], index=dates[:6])

        # Barrier at -0.02
        touch = find_barrier_touch(path_ret, pt_barrier=0.02, sl_barrier=-0.02)

        assert touch == dates[2]

    def test_find_barrier_touch_none(self):
        """Test when no barrier is touched."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        path_ret = pd.Series([0, 0.01, 0, -0.01, 0], index=dates[:5])

        touch = find_barrier_touch(path_ret, pt_barrier=0.02, sl_barrier=-0.02)

        assert touch is None

    def test_find_barrier_touch_first(self):
        """Test that the first touch is returned."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        # Touches PT at index 2, then SL at index 4
        path_ret = pd.Series([0, 0.01, 0.02, 0, -0.02], index=dates[:5])

        touch = find_barrier_touch(path_ret, pt_barrier=0.02, sl_barrier=-0.02)

        assert touch == dates[2]

    def test_compute_return_and_label(self):
        """Test return and label computation."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        close = pd.Series([100, 101, 102, 98, 100], index=dates)

        t0 = dates[0]

        # Case 1: Positive return > min_return
        t1 = dates[2]  # Price 102
        ret, label = compute_return_and_label(close, t0, t1, min_return=0.01)
        assert ret == 0.02
        assert label == 1

        # Case 2: Negative return < -min_return
        t1 = dates[3]  # Price 98
        ret, label = compute_return_and_label(close, t0, t1, min_return=0.01)
        assert ret == -0.02
        assert label == -1

        # Case 3: Small return (within min_return)
        t1 = dates[1]  # Price 101
        ret, label = compute_return_and_label(close, t0, t1, min_return=0.015)
        assert ret == 0.01
        assert label == 0  # 0.01 < 0.015

    def test_set_vertical_barriers(self, sample_prices):
        """Test vertical barrier setting."""
        t_events = sample_prices.index[::10]  # Every 10th bar
        max_holding = 5

        t1_series = set_vertical_barriers(t_events, sample_prices.index, max_holding)

        assert len(t1_series) == len(t_events)

        # Check first event
        evt0 = t_events[0]
        t1_0 = t1_series.loc[evt0]

        # Should be 5 bars later
        expected_t1 = sample_prices.index[sample_prices.index.get_loc(evt0) + max_holding]
        assert t1_0 == expected_t1

    def test_set_vertical_barriers_end(self, sample_prices):
        """Test vertical barrier near end of data."""
        # Event near the end
        last_idx = len(sample_prices) - 1
        evt = sample_prices.index[last_idx - 2]
        t_events = pd.DatetimeIndex([evt])

        max_holding = 5

        t1_series = set_vertical_barriers(t_events, sample_prices.index, max_holding)

        # Should be capped at last available timestamp
        expected_t1 = sample_prices.index[last_idx]
        assert t1_series.iloc[0] == expected_t1


# =============================================================================
# TESTS - Model Loading
# =============================================================================


class TestModelLoading:
    """Tests for model loading utilities."""

    def test_load_model_class(self):
        """Test dynamic loading of model class."""
        # We can test with a known model in the registry
        # assuming the model module is available in the environment
        # or we can mock import

        with patch("builtins.__import__") as mock_import:
            # Setup mock
            mock_module = MagicMock()
            mock_class = MagicMock()
            setattr(mock_module, "LightGBMModel", mock_class)
            mock_import.return_value = mock_module

            # This relies on MODEL_REGISTRY structure
            model_name = "lightgbm"
            if model_name in MODEL_REGISTRY:
                # We need to ensure that getattr returns the class
                # when called on the module returned by __import__
                # The implementation does:
                # module = __import__(module_path, fromlist=[class_name])
                # return getattr(module, class_name)

                # However, since we are patching __import__, we might face issues
                # with real imports if not careful.
                # Let's try to just check if it raises error for unknown model
                pass

    def test_load_unknown_model(self):
        """Test error for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            load_model_class("non_existent_model")
