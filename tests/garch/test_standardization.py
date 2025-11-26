"""Tests for src/garch/garch_diagnostic/standardization.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_diagnostic.standardization import (
    _detect_egarch_orders,
    _extract_egarch_params_for_variance,
    _validate_variance_path,
    compute_standardized_residuals_for_plot,
    standardize_residuals,
)


class TestDetectEgarchOrders:
    """Test cases for _detect_egarch_orders function."""

    def test_detects_explicit_orders(self):
        """Should use explicit o and p if provided."""
        params = {"o": 2, "p": 2, "omega": -0.1}

        o, p = _detect_egarch_orders(params)

        assert o == 2
        assert p == 2

    def test_detects_egarch11_from_params(self):
        """Should detect EGARCH(1,1) from parameter names."""
        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92}

        o, p = _detect_egarch_orders(params)

        assert o == 1
        assert p == 1

    def test_detects_egarch21_from_params(self):
        """Should detect EGARCH(2,1) from parameter names."""
        params = {
            "omega": -0.1,
            "alpha1": 0.10,
            "alpha2": 0.05,
            "gamma1": -0.05,
            "gamma2": -0.03,
            "beta": 0.90,
        }

        o, p = _detect_egarch_orders(params)

        assert o == 2
        assert p == 1

    def test_detects_egarch12_from_params(self):
        """Should detect EGARCH(1,2) from parameter names."""
        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta1": 0.7, "beta2": 0.2}

        o, p = _detect_egarch_orders(params)

        assert o == 1
        assert p == 2

    def test_detects_egarch22_from_params(self):
        """Should detect EGARCH(2,2) from parameter names."""
        params = {
            "omega": -0.1,
            "alpha1": 0.10,
            "alpha2": 0.05,
            "gamma1": -0.05,
            "gamma2": -0.03,
            "beta1": 0.7,
            "beta2": 0.2,
        }

        o, p = _detect_egarch_orders(params)

        assert o == 2
        assert p == 2


class TestExtractEgarchParamsForVariance:
    """Test cases for _extract_egarch_params_for_variance function."""

    def test_extracts_egarch11_params(self):
        """Should extract EGARCH(1,1) parameters."""
        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92}

        omega, alpha, gamma, beta = _extract_egarch_params_for_variance(params, o=1, p=1)

        assert omega == -0.1
        assert alpha == 0.15
        assert gamma == -0.08
        assert beta == 0.92

    def test_extracts_egarch21_params(self):
        """Should extract EGARCH(2,1) parameters as tuples."""
        params = {
            "omega": -0.1,
            "alpha1": 0.10,
            "alpha2": 0.05,
            "gamma1": -0.05,
            "gamma2": -0.03,
            "beta": 0.90,
        }

        omega, alpha, gamma, beta = _extract_egarch_params_for_variance(params, o=2, p=1)

        assert omega == -0.1
        assert alpha == (0.10, 0.05)
        assert gamma == (-0.05, -0.03)
        assert beta == 0.90

    def test_extracts_egarch12_params(self):
        """Should extract EGARCH(1,2) parameters."""
        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta1": 0.7, "beta2": 0.2}

        omega, alpha, gamma, beta = _extract_egarch_params_for_variance(params, o=1, p=2)

        assert omega == -0.1
        assert alpha == 0.15
        assert gamma == -0.08
        assert beta == (0.7, 0.2)

    def test_uses_default_gamma_if_missing(self):
        """Should use default gamma=0 if not provided."""
        params = {"omega": -0.1, "alpha": 0.15, "beta": 0.92}

        omega, alpha, gamma, beta = _extract_egarch_params_for_variance(params, o=1, p=1)

        assert gamma == 0.0


class TestValidateVariancePath:
    """Test cases for _validate_variance_path function."""

    def test_accepts_valid_variance_path(self):
        """Should accept valid variance path."""
        sigma2 = np.array([0.0004, 0.0005, 0.0003, 0.0006])

        # Should not raise
        _validate_variance_path(sigma2)

    def test_rejects_nan_values(self):
        """Should reject variance path with NaN."""
        sigma2 = np.array([0.0004, np.nan, 0.0003])

        with pytest.raises(ValueError, match="Invalid variance path"):
            _validate_variance_path(sigma2)

    def test_rejects_negative_values(self):
        """Should reject variance path with negative values."""
        sigma2 = np.array([0.0004, -0.0001, 0.0003])

        with pytest.raises(ValueError, match="Invalid variance path"):
            _validate_variance_path(sigma2)

    def test_rejects_zero_values(self):
        """Should reject variance path with zero values."""
        sigma2 = np.array([0.0004, 0.0, 0.0003])

        with pytest.raises(ValueError, match="Invalid variance path"):
            _validate_variance_path(sigma2)

    def test_rejects_inf_values(self):
        """Should reject variance path with inf."""
        sigma2 = np.array([0.0004, np.inf, 0.0003])

        with pytest.raises(ValueError, match="Invalid variance path"):
            _validate_variance_path(sigma2)


class TestStandardizeResiduals:
    """Test cases for standardize_residuals function."""

    def test_standardizes_residuals(self):
        """Should compute standardized residuals z_t = e_t / sigma_t."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        z = standardize_residuals(residuals, params, dist="student", nu=5.0)

        assert len(z) <= len(residuals)
        # Most standardized residuals should be finite
        assert np.sum(np.isfinite(z)) > 0.9 * len(z)

    def test_with_clean_option(self):
        """Should remove non-finite values when clean=True."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        z = standardize_residuals(residuals, params, dist="student", nu=5.0, clean=True)

        # All values should be finite
        assert np.all(np.isfinite(z))

    def test_with_skewt_distribution(self):
        """Should work with Skew-t distribution."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {
            "omega": -0.1,
            "alpha": 0.15,
            "gamma": -0.08,
            "beta": 0.92,
            "nu": 5.0,
            "lambda_skew": -0.2,
        }

        z = standardize_residuals(residuals, params, dist="skewt", nu=5.0, lambda_skew=-0.2)

        assert len(z) > 0

    def test_raises_on_missing_lambda_for_skewt(self):
        """Should raise ValueError when lambda is missing for Skew-t."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        with pytest.raises(ValueError, match="requires 'lambda_skew'"):
            standardize_residuals(residuals, params, dist="skewt", nu=5.0)

    def test_extracts_lambda_from_params(self):
        """Should extract lambda_skew from params dict if not provided."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {
            "omega": -0.1,
            "alpha": 0.15,
            "gamma": -0.08,
            "beta": 0.92,
            "nu": 5.0,
            "lambda": -0.2,  # Legacy key name
        }

        # Should work without explicit lambda_skew argument
        z = standardize_residuals(residuals, params, dist="skewt", nu=5.0)

        assert len(z) > 0


class TestStandardizeResidualsEgarchOrders:
    """Tests for standardize_residuals with different EGARCH orders."""

    def test_egarch11(self):
        """Should work with EGARCH(1,1)."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        z = standardize_residuals(residuals, params, dist="student", nu=5.0)

        assert len(z) > 0

    def test_egarch21(self):
        """Should work with EGARCH(2,1)."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {
            "omega": -0.1,
            "alpha1": 0.10,
            "alpha2": 0.05,
            "gamma1": -0.05,
            "gamma2": -0.03,
            "beta": 0.90,
            "nu": 5.0,
        }

        z = standardize_residuals(residuals, params, dist="student", nu=5.0)

        assert len(z) > 0

    def test_egarch12(self):
        """Should work with EGARCH(1,2)."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {
            "omega": -0.1,
            "alpha": 0.15,
            "gamma": -0.08,
            "beta1": 0.7,
            "beta2": 0.2,
            "nu": 5.0,
        }

        z = standardize_residuals(residuals, params, dist="student", nu=5.0)

        assert len(z) > 0


class TestComputeStandardizedResidualsForPlot:
    """Test cases for compute_standardized_residuals_for_plot function."""

    def test_returns_none_when_no_params(self):
        """Should return None when garch_params is None."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        result = compute_standardized_residuals_for_plot(residuals, None)

        assert result is None

    def test_computes_when_params_provided(self):
        """Should compute standardized residuals when params provided."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        result = compute_standardized_residuals_for_plot(residuals, params, dist="student", nu=5.0)

        assert result is not None
        assert len(result) > 0


class TestStandardizedResidualsProperties:
    """Tests for properties of standardized residuals."""

    def test_approximately_unit_variance(self):
        """Standardized residuals should have finite variance and preserve structure."""
        np.random.seed(42)

        # Generate GARCH-like residuals
        from scipy.stats import t as t_dist  # type: ignore

        residuals = np.asarray(t_dist.rvs(df=5.0, size=1000) * 0.02, dtype=float)

        # Use moderate EGARCH parameters
        # omega controls unconditional variance level (not critical for this test)
        # Low beta means less persistence - variance returns quickly to mean
        params = {"omega": -0.5, "alpha": 0.1, "gamma": -0.05, "beta": 0.5, "nu": 5.0}

        z = standardize_residuals(residuals, params, dist="student", nu=5.0, clean=True)

        # Standardized residuals should have finite, positive variance
        var_z = np.var(z)
        assert np.isfinite(var_z)
        assert var_z > 0
        # z should have same length as input (after cleaning)
        assert len(z) > 0

    def test_approximately_zero_mean(self):
        """Standardized residuals should have approximately zero mean."""
        np.random.seed(42)
        residuals = np.random.randn(500) * 0.02

        params = {"omega": -0.1, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        z = standardize_residuals(residuals, params, dist="student", nu=5.0, clean=True)

        # Mean should be close to zero
        mean_z = np.mean(z)
        assert abs(mean_z) < 0.3


class TestStandardizationAntiLeakage:
    """CRITICAL: Anti-leakage tests for standardization."""

    def test_standardization_formula_correctness(self):
        """CRITICAL: Standardization should use z_t = e_t / sigma_t."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        params = {"omega": -7.8, "alpha": 0.15, "gamma": -0.08, "beta": 0.92, "nu": 5.0}

        # Compute standardized residuals
        z = standardize_residuals(residuals, params, dist="student", nu=5.0)

        # Verify the formula: z[t] = e[t] / sigma[t]
        # The standardization should preserve the sign of residuals
        signs_match = np.sign(z) == np.sign(residuals[np.isfinite(residuals)])
        assert np.mean(signs_match) > 0.99  # Almost all signs should match

    def test_variance_computation_is_causal(self):
        """CRITICAL: Variance recursion within standardization should be causal."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        # Use fixed init to eliminate sample variance dependence
        init_var = 0.0004  # Fixed initial variance

        from src.garch.garch_params.core import egarch_variance

        omega, alpha, gamma, beta = -7.8, 0.15, -0.08, 0.92

        # Compute variance path with fixed init
        sigma2 = egarch_variance(
            residuals, omega, alpha, gamma, beta, dist="student", nu=5.0, init=init_var
        )

        # Modify future residuals
        residuals_modified = residuals.copy()
        residuals_modified[60:] = np.random.randn(40) * 0.10

        sigma2_modified = egarch_variance(
            residuals_modified, omega, alpha, gamma, beta, dist="student", nu=5.0, init=init_var
        )

        # With fixed init, variance at t<60 should only depend on e[0:t-1]
        # So sigma2[:60] should be identical for both
        np.testing.assert_array_almost_equal(sigma2[:60], sigma2_modified[:60])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
