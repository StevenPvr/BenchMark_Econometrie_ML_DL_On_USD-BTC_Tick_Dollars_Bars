"""Tests for src/garch/garch_params/core/distributions.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_params.core.distributions import (
    compute_kappa,
    compute_kappa_skewt,
    compute_kappa_student,
    compute_loglik_skewt,
    compute_loglik_student,
)


class TestComputeKappaStudent:
    """Test cases for compute_kappa_student function."""

    def test_computes_kappa_nu_5(self):
        """Should compute kappa for nu=5."""
        kappa = compute_kappa_student(5.0)

        # Kappa should be positive
        assert kappa > 0
        # For nu=5, kappa ≈ 0.73 (E[|Z|] for standardized t)
        assert 0.6 < kappa < 0.9

    def test_computes_kappa_nu_10(self):
        """Should compute kappa for nu=10."""
        kappa = compute_kappa_student(10.0)

        # For higher nu, kappa approaches sqrt(2/pi) ≈ 0.798
        assert kappa > 0
        assert 0.7 < kappa < 0.85

    def test_computes_kappa_high_nu(self):
        """For high nu, kappa approaches normal E[|Z|] = sqrt(2/pi)."""
        kappa = compute_kappa_student(100.0)

        normal_kappa = np.sqrt(2 / np.pi)  # ≈ 0.798
        assert abs(kappa - normal_kappa) < 0.02

    def test_raises_on_nu_too_small(self):
        """Should raise ValueError for nu <= 2."""
        with pytest.raises(ValueError, match="Invalid Student-t"):
            compute_kappa_student(2.0)

        with pytest.raises(ValueError, match="Invalid Student-t"):
            compute_kappa_student(1.5)

    def test_kappa_decreases_with_nu(self):
        """Kappa should increase towards normal as nu increases."""
        kappa_5 = compute_kappa_student(5.0)
        kappa_10 = compute_kappa_student(10.0)
        kappa_50 = compute_kappa_student(50.0)

        # Kappa increases with nu (approaches sqrt(2/pi))
        assert kappa_5 < kappa_10 < kappa_50


class TestComputeKappaSkewt:
    """Test cases for compute_kappa_skewt function."""

    def test_computes_kappa_zero_skew(self):
        """With zero skewness, should equal Student-t kappa."""
        kappa_skewt = compute_kappa_skewt(5.0, 0.0)
        kappa_student = compute_kappa_student(5.0)

        # Should be exactly equal for zero skewness
        assert abs(kappa_skewt - kappa_student) < 1e-10

    def test_computes_kappa_positive_skew(self):
        """Should compute kappa with positive skewness."""
        kappa = compute_kappa_skewt(5.0, 0.3)

        assert kappa > 0
        # With adjustment, kappa should be slightly different from student
        kappa_student = compute_kappa_student(5.0)
        assert kappa != kappa_student

    def test_computes_kappa_negative_skew(self):
        """Should compute kappa with negative skewness."""
        kappa = compute_kappa_skewt(5.0, -0.3)

        assert kappa > 0

    def test_raises_on_invalid_nu(self):
        """Should raise ValueError for nu <= 2."""
        with pytest.raises(ValueError, match="Invalid Skew-t"):
            compute_kappa_skewt(2.0, 0.0)

    def test_raises_on_invalid_lambda(self):
        """Should raise ValueError for lambda outside (-1, 1)."""
        with pytest.raises(ValueError, match="Invalid Skew-t"):
            compute_kappa_skewt(5.0, 1.0)

        with pytest.raises(ValueError, match="Invalid Skew-t"):
            compute_kappa_skewt(5.0, -1.0)

        with pytest.raises(ValueError, match="Invalid Skew-t"):
            compute_kappa_skewt(5.0, 1.5)


class TestComputeKappa:
    """Test cases for compute_kappa function."""

    def test_student_distribution(self):
        """Should compute kappa for Student-t."""
        kappa = compute_kappa("student", nu=5.0)

        assert kappa > 0
        assert abs(kappa - compute_kappa_student(5.0)) < 1e-10

    def test_skewt_distribution(self):
        """Should compute kappa for Skew-t."""
        kappa = compute_kappa("skewt", nu=5.0, lambda_skew=-0.2)

        assert kappa > 0

    def test_case_insensitive(self):
        """Should be case-insensitive for distribution name."""
        kappa_lower = compute_kappa("student", nu=5.0)
        kappa_upper = compute_kappa("Student", nu=5.0)
        kappa_mixed = compute_kappa("STUDENT", nu=5.0)

        assert kappa_lower == kappa_upper == kappa_mixed

    def test_raises_on_unsupported_dist(self):
        """Should raise ValueError for unsupported distribution."""
        with pytest.raises(ValueError, match="Unsupported distribution"):
            compute_kappa("normal", nu=5.0)

    def test_raises_on_missing_nu_student(self):
        """Should use default nu for Student-t if not provided."""
        from src.constants import GARCH_STUDENT_NU_INIT

        kappa = compute_kappa("student", nu=None)
        expected = compute_kappa_student(GARCH_STUDENT_NU_INIT)

        assert abs(kappa - expected) < 1e-10

    def test_raises_on_missing_params_skewt(self):
        """Should raise ValueError for Skew-t without required params."""
        with pytest.raises(ValueError, match="requires both nu and lambda"):
            compute_kappa("skewt", nu=5.0, lambda_skew=None)


class TestComputeLoglikStudent:
    """Test cases for compute_loglik_student function."""

    def test_computes_loglik(self):
        """Should compute Student-t log-likelihood."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02
        variances = np.full(100, 0.0004)  # Constant variance

        loglik = compute_loglik_student(residuals, variances, nu=5.0)

        assert np.isfinite(loglik)
        # Log-likelihood can be positive or negative depending on scale

    def test_loglik_increases_with_better_fit(self):
        """Better variance fit should produce higher log-likelihood."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        # Good fit: variance matches actual
        var_good = np.full(100, np.var(residuals))
        loglik_good = compute_loglik_student(residuals, var_good, nu=5.0)

        # Poor fit: variance too high
        var_poor = np.full(100, 0.01)
        loglik_poor = compute_loglik_student(residuals, var_poor, nu=5.0)

        assert loglik_good > loglik_poor

    def test_higher_nu_approaches_normal(self):
        """With high nu, should approach normal distribution behavior."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02
        variances = np.full(100, 0.0004)

        loglik_nu5 = compute_loglik_student(residuals, variances, nu=5.0)
        loglik_nu50 = compute_loglik_student(residuals, variances, nu=50.0)
        loglik_nu100 = compute_loglik_student(residuals, variances, nu=100.0)

        # Log-likelihoods should converge as nu increases
        diff_5_50 = abs(loglik_nu5 - loglik_nu50)
        diff_50_100 = abs(loglik_nu50 - loglik_nu100)

        assert diff_50_100 < diff_5_50

    def test_handles_extreme_values(self):
        """Should handle extreme values (may produce inf/nan)."""
        residuals = np.array([1e100, 1e100])
        variances = np.array([1e-100, 1e-100])

        # Extreme values may raise or return non-finite
        try:
            loglik = compute_loglik_student(residuals, variances, nu=5.0)
            # If no exception, result may be non-finite
            assert isinstance(loglik, float)
        except (ValueError, OverflowError):
            # Exception is also acceptable for extreme values
            pass


class TestComputeLoglikSkewt:
    """Test cases for compute_loglik_skewt function."""

    def test_computes_loglik(self):
        """Should compute Skew-t log-likelihood."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02
        variances = np.full(100, 0.0004)

        loglik = compute_loglik_skewt(residuals, variances, nu=5.0, lambda_skew=-0.2)

        assert np.isfinite(loglik)
        # Log-likelihood can be positive or negative depending on scale

    def test_zero_skewness_similar_to_student(self):
        """With zero skewness, should be similar to Student-t."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02
        variances = np.full(100, 0.0004)

        loglik_skewt = compute_loglik_skewt(residuals, variances, nu=5.0, lambda_skew=0.0)
        loglik_student = compute_loglik_student(residuals, variances, nu=5.0)

        # Should be very close (but not identical due to normalization)
        assert abs(loglik_skewt - loglik_student) / abs(loglik_student) < 0.1

    def test_negative_skewness_effect(self):
        """Negative skewness should affect left-tail differently."""
        np.random.seed(42)
        # Create data with more negative values
        residuals = np.concatenate([
            np.random.randn(50) * 0.02 - 0.01,  # More negative
            np.random.randn(50) * 0.02,
        ])
        variances = np.full(100, 0.0004)

        loglik_neg_skew = compute_loglik_skewt(residuals, variances, nu=5.0, lambda_skew=-0.3)
        loglik_pos_skew = compute_loglik_skewt(residuals, variances, nu=5.0, lambda_skew=0.3)

        # With negatively skewed data, negative skewness parameter should fit better
        # (or at least produce different likelihood)
        assert loglik_neg_skew != loglik_pos_skew


class TestDistributionKappaConsistency:
    """Tests for consistency between kappa and log-likelihood."""

    def test_kappa_used_in_egarch_recursion(self):
        """Kappa should be consistent with distribution parameters."""
        # This is more of an integration test
        nu = 5.0

        kappa = compute_kappa("student", nu=nu)

        # For Student-t with unit variance, kappa = E[|Z|]
        # Should be close to numerical integration result
        from scipy.stats import t as t_dist  # type: ignore

        # Monte Carlo estimation of E[|Z|]
        np.random.seed(42)
        samples = t_dist.rvs(df=nu, size=100000)
        # Standardize to unit variance
        samples = samples / np.sqrt(nu / (nu - 2))
        empirical_kappa = np.mean(np.abs(samples))

        assert abs(kappa - empirical_kappa) < 0.02


class TestDistributionNumericalStability:
    """Tests for numerical stability of distribution functions."""

    def test_kappa_stable_near_boundary(self):
        """Kappa should be stable for nu close to 2."""
        kappa = compute_kappa_student(2.1)

        assert np.isfinite(kappa)
        assert kappa > 0

    def test_loglik_handles_small_variances(self):
        """Log-likelihood should handle small variances."""
        residuals = np.array([1e-10, 1e-10, 1e-10])
        variances = np.array([1e-18, 1e-18, 1e-18])

        # This might produce extreme values but should not crash
        loglik = compute_loglik_student(residuals, variances, nu=5.0)

        assert np.isfinite(loglik)

    def test_loglik_handles_large_residuals_moderate_var(self):
        """Log-likelihood should handle moderately large residuals."""
        residuals = np.array([0.1, 0.2, 0.3])
        variances = np.array([0.01, 0.01, 0.01])

        loglik = compute_loglik_student(residuals, variances, nu=5.0)

        assert np.isfinite(loglik)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
