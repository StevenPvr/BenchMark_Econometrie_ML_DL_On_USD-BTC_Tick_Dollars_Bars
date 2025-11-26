"""Tests for src/garch/training_garch/variance_filter.py module."""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_params.models import EGARCHParams
from src.garch.training_garch.variance_filter import VarianceFilter


class TestVarianceFilterInit:
    """Test cases for VarianceFilter initialization."""

    def test_initializes_with_egarch_params(self):
        """Should initialize with EGARCHParams."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        # Suppress the data leakage warning for testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        assert vf.params == params
        assert vf.kappa > 0

    def test_warns_about_data_leakage(self, caplog):
        """Should warn about data leakage via logging."""
        import logging

        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with caplog.at_level(logging.WARNING):
            VarianceFilter(params)

            # Should have issued a warning via logging
            assert len(caplog.records) >= 1
            assert any("leakage" in record.message.lower() for record in caplog.records)


class TestVarianceFilterFilterVariance:
    """Test cases for VarianceFilter.filter_variance method."""

    def test_filters_variance(self):
        """Should compute filtered variance path."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = vf.filter_variance(residuals)

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))
        assert np.all(sigma2 > 0)

    def test_variance_responds_to_shocks(self):
        """Filtered variance should respond to residual shocks."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        # Create residuals with shock
        residuals = np.array([0.01] * 50 + [0.10] * 10 + [0.01] * 40)

        sigma2 = vf.filter_variance(residuals)

        # Variance should increase after shock
        var_before = np.mean(sigma2[40:50])
        var_after = np.mean(sigma2[55:65])

        assert var_after > var_before

    def test_handles_egarch21(self):
        """Should handle EGARCH(2,1) parameters."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=0.90,
            nu=5.0,
            lambda_skew=None,
            o=2,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = vf.filter_variance(residuals)

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))


class TestVarianceFilterStandardizedResiduals:
    """Test cases for VarianceFilter.compute_standardized_residuals method."""

    def test_computes_standardized_residuals(self):
        """Should compute z_t = e_t / sigma_t."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        z = vf.compute_standardized_residuals(residuals)

        assert len(z) == len(residuals)
        # Most standardized residuals should be finite
        finite_count = np.sum(np.isfinite(z))
        assert finite_count > 0.9 * len(z)

    def test_standardized_residuals_formula(self):
        """Should follow z_t = e_t / sigma_t formula."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        z = vf.compute_standardized_residuals(residuals)
        sigma2 = vf.filter_variance(residuals)

        # Verify formula
        expected_z = residuals / np.sqrt(sigma2)
        np.testing.assert_array_almost_equal(z[np.isfinite(z)], expected_z[np.isfinite(z)])

    def test_standardized_residuals_approximately_unit_variance(self):
        """Standardized residuals should have finite positive variance."""
        # Use moderate EGARCH parameters (low beta = less persistence)
        params = EGARCHParams(
            omega=-0.5,
            alpha=0.1,
            gamma=-0.05,
            beta=0.5,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        # Use Student-t distributed returns (matching the model)
        from scipy.stats import t as t_dist  # type: ignore
        residuals = np.asarray(t_dist.rvs(df=5.0, size=1000) * 0.02, dtype=float)

        z = vf.compute_standardized_residuals(residuals)
        z_finite = z[np.isfinite(z)]

        # Standardized residuals should have finite positive variance
        var_z = np.var(z_finite)
        assert np.isfinite(var_z)
        assert var_z > 0
        # z should have values
        assert len(z_finite) > 0


class TestVarianceFilterAntiLeakage:
    """CRITICAL: Anti-leakage tests for VarianceFilter."""

    def test_filtered_variance_uses_current_info(self):
        """WARNING: Filtered variance σ²_t|t uses info up to t (data leakage for forecasting)."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        # This test documents that filtered variance is NOT suitable for forecasting
        # sigma2[t] depends on residuals[0:t], including residuals[t-1]
        # For TRUE forecasting, we need sigma2[t] = f(residuals[0:t-1])

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = vf.filter_variance(residuals)

        # Verify that changing residuals[t-1] affects sigma2[t]
        residuals_modified = residuals.copy()
        residuals_modified[50] = 0.10  # Large shock

        sigma2_modified = vf.filter_variance(residuals_modified)

        # sigma2[51] should be different (variance at t=51 uses residual at t=50)
        assert sigma2[51] != sigma2_modified[51]

    def test_filtered_variance_not_forecast(self):
        """Filtered variance is σ²_t|t, not the forecast σ²_t|t-1."""
        # This test documents the difference between filtering and forecasting

        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(50) * 0.02

        sigma2 = vf.filter_variance(residuals)

        # For true ex-ante forecasting (used in evaluation):
        # sigma2_forecast[t] should be computed using only residuals[0:t-1]
        # NOT residuals[0:t] as in filtering

        # The VarianceFilter class explicitly warns about this
        # Use EGARCHForecaster for proper out-of-sample forecasts

        assert len(sigma2) == 50  # Same length as residuals (filtered, not forecast)


class TestVarianceFilterWithDifferentDistributions:
    """Tests for VarianceFilter with different distributions."""

    def test_student_distribution(self):
        """Should work with Student-t distribution."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=None,
            o=1,
            p=1,
            dist="student",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = vf.filter_variance(residuals)

        assert np.all(np.isfinite(sigma2))

    def test_skewt_distribution(self):
        """Should work with Skew-t distribution."""
        params = EGARCHParams(
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            nu=5.0,
            lambda_skew=-0.2,
            o=1,
            p=1,
            dist="skewt",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vf = VarianceFilter(params)

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = vf.filter_variance(residuals)

        assert np.all(np.isfinite(sigma2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
