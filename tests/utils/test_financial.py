"""Tests for src/utils/financial.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.financial import compute_rolling_volume_scaling


class TestComputeRollingVolumeScaling:
    """Test cases for compute_rolling_volume_scaling function."""

    def test_returns_series(self):
        """Should return a pandas Series."""
        volume = pd.Series([100, 200, 150, 180, 220])

        result = compute_rolling_volume_scaling(volume, window=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(volume)

    def test_default_window(self):
        """Should use default window of 21."""
        volume = pd.Series(np.random.randint(100, 200, size=50))

        # Should not raise
        result = compute_rolling_volume_scaling(volume)

        assert len(result) == len(volume)

    def test_values_relative_to_mean(self):
        """Values should be relative to rolling mean."""
        # Constant volume should give weights of 1.0 after warm-up
        volume = pd.Series([100.0] * 20)

        result = compute_rolling_volume_scaling(volume, window=5)

        # After warm-up, all weights should be ~1.0
        np.testing.assert_allclose(result[5:], 1.0, rtol=1e-10)

    def test_high_volume_gives_high_weight(self):
        """Higher than average volume should give weight > 1."""
        volume = pd.Series([100.0] * 10 + [200.0])

        result = compute_rolling_volume_scaling(volume, window=5)

        # Last value (200) is higher than rolling mean (~100), so weight > 1
        assert result.iloc[-1] > 1.0

    def test_low_volume_gives_low_weight(self):
        """Lower than average volume should give weight < 1."""
        volume = pd.Series([100.0] * 10 + [50.0])

        result = compute_rolling_volume_scaling(volume, window=5)

        # Last value (50) is lower than rolling mean (~100), so weight < 1
        assert result.iloc[-1] < 1.0

    def test_handles_zero_volume(self):
        """Should handle zero volume without division by zero."""
        volume = pd.Series([100.0, 0.0, 0.0, 0.0, 0.0, 100.0])

        result = compute_rolling_volume_scaling(volume, window=3)

        # Should not have inf values
        assert not np.any(np.isinf(result))
        # NaN values should be filled with 1.0
        assert not np.any(np.isnan(result))

    def test_fills_nan_with_one(self):
        """NaN values should be filled with 1.0."""
        volume = pd.Series([100.0, np.nan, 100.0])

        result = compute_rolling_volume_scaling(volume, window=2)

        assert not np.any(np.isnan(result))

    def test_custom_min_periods(self):
        """Should use custom min_periods."""
        volume = pd.Series([100.0, 200.0, 150.0])

        result = compute_rolling_volume_scaling(volume, window=10, min_periods=1)

        # Should have valid values from the start
        assert np.isfinite(result.iloc[0])

    def test_min_periods_default(self):
        """Default min_periods should be window // 3."""
        volume = pd.Series([100.0] * 30)

        result = compute_rolling_volume_scaling(volume, window=21)

        # min_periods = 21 // 3 = 7
        # First 6 values should still be valid due to fillna
        assert all(np.isfinite(result))


class TestRollingVolumeScalingProperties:
    """Property-based tests for rolling volume scaling."""

    def test_weights_are_positive(self):
        """All weights should be positive."""
        np.random.seed(42)
        volume = pd.Series(np.random.uniform(10, 100, size=100))

        result = compute_rolling_volume_scaling(volume, window=10)

        assert all(result > 0)

    def test_weights_are_finite(self):
        """All weights should be finite."""
        np.random.seed(42)
        volume = pd.Series(np.random.uniform(0.1, 100, size=100))

        result = compute_rolling_volume_scaling(volume, window=10)

        assert all(np.isfinite(result))

    def test_average_weight_near_one(self):
        """Average weight should be approximately 1 for stable volume."""
        # Generate stable volume data
        volume = pd.Series([100.0] * 100)

        result = compute_rolling_volume_scaling(volume, window=10)

        # Skip warm-up period
        assert pytest.approx(result[10:].mean(), rel=0.01) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
