"""Tests for src/garch/garch_diagnostic/statistics.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_diagnostic.statistics import (
    autocorr,
    compute_autocorr_denominator,
    compute_autocorr_lag,
    pacf_init_first_lag,
    pacf_compute_lag,
    pacf_from_autocorr,
    compute_ljung_box_statistics,
)


class TestComputeAutocorrDenominator:
    """Test cases for compute_autocorr_denominator function."""

    def test_computes_sum_of_squares(self):
        """Should compute sum of squares."""
        x = np.array([1.0, 2.0, 3.0])

        result = compute_autocorr_denominator(x)

        assert result == 14.0  # 1 + 4 + 9 = 14

    def test_handles_negative_values(self):
        """Should handle negative values."""
        x = np.array([-1.0, 2.0, -3.0])

        result = compute_autocorr_denominator(x)

        assert result == 14.0  # 1 + 4 + 9 = 14


class TestComputeAutocorrLag:
    """Test cases for compute_autocorr_lag function."""

    def test_returns_zero_for_zero_denom(self):
        """Should return 0 when denominator is 0."""
        x = np.array([1.0, 2.0, 3.0])

        result = compute_autocorr_lag(x, k=1, denom=0.0)

        assert result == 0.0

    def test_computes_lag_autocorrelation(self):
        """Should compute autocorrelation for lag k."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        denom = compute_autocorr_denominator(x)

        result = compute_autocorr_lag(x, k=1, denom=denom)

        # For constant series, autocorrelation is high
        assert result > 0.5


class TestAutocorr:
    """Test cases for autocorr function."""

    def test_lag_zero_is_one(self):
        """Autocorrelation at lag 0 should be 1."""
        x = np.random.randn(100)

        result = autocorr(x, nlags=10)

        assert result[0] == 1.0

    def test_returns_correct_length(self):
        """Should return nlags+1 values."""
        x = np.random.randn(100)

        result = autocorr(x, nlags=5)

        assert len(result) == 6  # lags 0, 1, 2, 3, 4, 5

    def test_handles_empty_array(self):
        """Should handle empty array."""
        x = np.array([])

        result = autocorr(x, nlags=3)

        assert len(result) == 4
        assert all(r == 0.0 for r in result)

    def test_white_noise_low_autocorr(self):
        """White noise should have low autocorrelation at lags > 0."""
        np.random.seed(42)
        x = np.random.randn(1000)

        result = autocorr(x, nlags=10)

        # Autocorrelations at lags 1-10 should be small (< 0.1)
        for lag in range(1, 11):
            assert abs(result[lag]) < 0.1

    def test_highly_correlated_series(self):
        """Highly correlated series should show high autocorrelation."""
        np.random.seed(42)
        # AR(1) process with high persistence
        x = np.zeros(1000)
        x[0] = np.random.randn()
        for i in range(1, 1000):
            x[i] = 0.9 * x[i - 1] + np.random.randn() * 0.1

        result = autocorr(x, nlags=5)

        # First few lags should have high positive autocorrelation
        assert result[1] > 0.5


class TestPacfInitFirstLag:
    """Test cases for pacf_init_first_lag function."""

    def test_initializes_pacf(self):
        """Should initialize PACF for first lag."""
        r = np.array([1.0, 0.8, 0.5, 0.3])
        phi_prev = np.zeros(4)

        result = pacf_init_first_lag(r, phi_prev)

        assert result == 0.8
        assert phi_prev[0] == 0.8


class TestPacfComputeLag:
    """Test cases for pacf_compute_lag function."""

    def test_computes_pacf_for_lag(self):
        """Should compute PACF for given lag."""
        r = np.array([1.0, 0.8, 0.5, 0.3])
        phi_prev = np.array([0.8, 0.0, 0.0, 0.0])
        den_prev = 1.0 - 0.8 * 0.8

        phi_kk, den_new = pacf_compute_lag(r, k=2, phi_prev=phi_prev, den_prev=den_prev)

        assert np.isfinite(phi_kk)
        assert np.isfinite(den_new)


class TestPacfFromAutocorr:
    """Test cases for pacf_from_autocorr function."""

    def test_returns_correct_length(self):
        """Should return nlags values."""
        r = autocorr(np.random.randn(100), nlags=10)

        result = pacf_from_autocorr(r, nlags=5)

        assert len(result) == 5

    def test_returns_empty_for_zero_nlags(self):
        """Should return empty array for nlags=0."""
        r = np.array([1.0, 0.5])

        result = pacf_from_autocorr(r, nlags=0)

        assert len(result) == 0

    def test_values_in_range(self):
        """PACF values should be in [-1, 1]."""
        np.random.seed(42)
        r = autocorr(np.random.randn(100), nlags=10)

        result = pacf_from_autocorr(r, nlags=10)

        assert all(-1 <= v <= 1 for v in result)


class TestComputeLjungBoxStatistics:
    """Test cases for compute_ljung_box_statistics function."""

    def test_returns_dict_with_expected_keys(self):
        """Should return dict with lags, lb_stat, lb_pvalue."""
        np.random.seed(42)
        series = np.random.randn(100)

        result = compute_ljung_box_statistics(series, lags=5)

        assert "lags" in result
        assert "lb_stat" in result
        assert "lb_pvalue" in result

    def test_correct_number_of_lags(self):
        """Should compute stats for correct number of lags."""
        np.random.seed(42)
        series = np.random.randn(100)

        result = compute_ljung_box_statistics(series, lags=10)

        assert len(result["lags"]) == 10
        assert len(result["lb_stat"]) == 10
        assert len(result["lb_pvalue"]) == 10

    def test_lb_stat_increasing(self):
        """Ljung-Box statistic should be non-decreasing with lags."""
        np.random.seed(42)
        series = np.random.randn(100)

        result = compute_ljung_box_statistics(series, lags=10)

        lb_stats = result["lb_stat"]
        for i in range(1, len(lb_stats)):
            assert lb_stats[i] >= lb_stats[i - 1]

    def test_pvalues_in_range(self):
        """P-values should be in [0, 1]."""
        np.random.seed(42)
        series = np.random.randn(100)

        result = compute_ljung_box_statistics(series, lags=10)

        for p in result["lb_pvalue"]:
            if np.isfinite(p):
                assert 0 <= p <= 1

    def test_white_noise_high_pvalues(self):
        """White noise should have high p-values (fail to reject H0)."""
        np.random.seed(42)
        series = np.random.randn(1000)

        result = compute_ljung_box_statistics(series, lags=10)

        # At least some p-values should be > 0.05
        assert any(p > 0.05 for p in result["lb_pvalue"] if np.isfinite(p))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
