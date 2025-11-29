"""Tests for src.utils.financial."""

import pytest
import pandas as pd
import numpy as np
from src.utils.financial import compute_rolling_volume_scaling

class TestComputeRollingVolumeScaling:
    def test_basic_scaling(self):
        """Test basic volume scaling."""
        volume = pd.Series([100.0] * 10 + [200.0] * 10)
        # Window 5
        scaled = compute_rolling_volume_scaling(volume, window=5)

        # At index 4, window is [100, 100, 100, 100, 100], mean 100. weight = 100/100 = 1.
        assert scaled.iloc[4] == 1.0

        # At index 10 (value 200), window (5-9) was 100.
        # But scaling includes current value in window?
        # The code: rolling(window).mean(). weights = volume / rolling_mean
        # Rolling mean includes current value by default.
        # So at index 10 (value 200), previous 4 were 100. window [100,100,100,100,200]. mean = 120.
        # weight = 200 / 120 = 1.666...
        expected = 200 / ((100 * 4 + 200) / 5)
        assert np.isclose(scaled.iloc[10], expected)

    def test_zero_volume(self):
        """Test handling of zero volume."""
        volume = pd.Series([0.0] * 10)
        scaled = compute_rolling_volume_scaling(volume, window=5)
        # rolling mean is 0. Division by zero handled.
        # fillna(1.0) used.
        # np.where(rolling_volume == 0, np.nan, rolling_volume) -> nan
        # volume / nan -> nan
        # fillna(1.0) -> 1.0
        assert (scaled == 1.0).all()

    def test_nan_handling(self):
        """Test NaN handling in volume."""
        volume = pd.Series([100.0, np.nan, 100.0])
        scaled = compute_rolling_volume_scaling(volume, window=2, min_periods=1)
        # Default pandas rolling ignores NaNs in calculation but result might be NaN if not enough points
        assert not scaled.isnull().all()

    def test_min_periods(self):
        """Test min_periods parameter."""
        volume = pd.Series([100.0, 200.0, 300.0])
        scaled = compute_rolling_volume_scaling(volume, window=3, min_periods=1)
        # First element: window has 1 element (100). mean 100. 100/100 = 1.
        assert scaled.iloc[0] == 1.0
