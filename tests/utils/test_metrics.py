"""Tests for src.utils.metrics."""

import pytest
import pandas as pd
import numpy as np
from src.utils.metrics import chi2_sf, compute_log_returns, compute_residuals

class TestMetrics:
    def test_chi2_sf(self):
        """Test chi-squared survival function."""
        # Check against a known value or property
        # chi2.sf(0, df) = 1
        assert chi2_sf(0, 1) == 1.0
        # Value should be between 0 and 1
        val = chi2_sf(1.0, 1)
        assert 0 <= val <= 1

    def test_compute_log_returns(self):
        """Test log returns calculation."""
        prices = pd.Series([100.0, 110.0, 100.0])
        returns = compute_log_returns(prices)

        # log(110/100) approx 0.0953
        assert np.isclose(returns.iloc[1], np.log(110/100))
        # log(100/110) approx -0.0953
        assert np.isclose(returns.iloc[2], np.log(100/110))
        # First element should be NaN or undefined? numpy log of NaN is NaN.
        # shifts 1: first element is NaN. log(val/NaN) -> NaN.
        assert np.isnan(returns.iloc[0])

    def test_compute_residuals(self):
        """Test residuals calculation."""
        y_true = pd.Series([10.0, 20.0])
        y_pred = pd.Series([9.0, 22.0])
        residuals = compute_residuals(y_true, y_pred)

        assert residuals.iloc[0] == 1.0
        assert residuals.iloc[1] == -2.0
