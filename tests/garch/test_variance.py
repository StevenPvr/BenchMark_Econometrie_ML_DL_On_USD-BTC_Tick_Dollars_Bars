"""Tests for src/garch/garch_params/core/variance.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_params.core.variance import (
    _validate_and_extract_alpha,
    _validate_and_extract_beta,
    _validate_and_extract_gamma,
    clip_and_exp_log_variance,
    compute_variance_path,
    compute_variance_path_egarch11,
    compute_variance_step_egarch11,
    initialize_variance,
    safe_variance,
    validate_param_types,
)


class TestInitializeVariance:
    """Test cases for initialize_variance function."""

    def test_uses_provided_init(self):
        """Should use provided init value if positive."""
        residuals = np.array([0.01, 0.02, -0.01])
        init = 0.0001

        result = initialize_variance(residuals, init)

        assert result == 0.0001

    def test_uses_sample_variance_if_init_none(self):
        """Should use sample variance when init is None."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        result = initialize_variance(residuals, None)

        # Should be close to sample variance
        sample_var = np.var(residuals, ddof=0)
        assert abs(result - sample_var) < 1e-10

    def test_clamps_to_minimum(self):
        """Should clamp variance to minimum threshold."""
        from src.constants import GARCH_MIN_INIT_VAR

        # Very small residuals leading to tiny variance
        residuals = np.array([1e-20, 1e-20, 1e-20])

        result = initialize_variance(residuals, None)

        assert result >= GARCH_MIN_INIT_VAR

    def test_handles_empty_residuals(self):
        """Should return minimum variance for empty residuals."""
        from src.constants import GARCH_MIN_INIT_VAR

        residuals = np.array([])

        result = initialize_variance(residuals, None)

        assert result == GARCH_MIN_INIT_VAR

    def test_ignores_zero_init(self):
        """Should ignore zero init and use sample variance."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        result = initialize_variance(residuals, 0.0)

        sample_var = np.var(residuals, ddof=0)
        assert abs(result - sample_var) < 1e-10


class TestSafeVariance:
    """Test cases for safe_variance function."""

    def test_returns_input_if_above_minimum(self):
        """Should return input if above minimum threshold."""
        from src.constants import GARCH_MIN_INIT_VAR

        variance = GARCH_MIN_INIT_VAR * 10

        result = safe_variance(variance)

        assert result == variance

    def test_clamps_to_minimum(self):
        """Should clamp to minimum if below threshold."""
        from src.constants import GARCH_MIN_INIT_VAR

        result = safe_variance(1e-30)

        assert result == GARCH_MIN_INIT_VAR

    def test_clamps_negative(self):
        """Should clamp negative variance to minimum."""
        from src.constants import GARCH_MIN_INIT_VAR

        result = safe_variance(-0.001)

        assert result == GARCH_MIN_INIT_VAR


class TestClipAndExpLogVariance:
    """Test cases for clip_and_exp_log_variance function."""

    def test_normal_log_variance(self):
        """Should compute exp of normal log-variance."""
        log_var = np.log(0.0004)  # sigma^2 = 0.0004

        result = clip_and_exp_log_variance(log_var)

        assert abs(result - 0.0004) < 1e-10

    def test_clips_high_log_variance(self):
        """Should clip very high log-variance to prevent overflow."""
        log_var = 100.0  # Extremely high

        result = clip_and_exp_log_variance(log_var)

        # Result should be finite and positive (clipped)
        assert np.isfinite(result)
        assert result > 0

    def test_clips_low_log_variance(self):
        """Should clip very low log-variance."""
        from src.constants import GARCH_LOG_VAR_MIN

        log_var = -100.0  # Extremely low

        result = clip_and_exp_log_variance(log_var)

        assert result == pytest.approx(np.exp(GARCH_LOG_VAR_MIN), rel=1e-6)

    def test_returns_nan_for_inf(self):
        """Should return NaN for infinite log-variance."""
        result = clip_and_exp_log_variance(float("inf"))

        assert np.isnan(result)

    def test_returns_nan_for_nan(self):
        """Should return NaN for NaN input."""
        result = clip_and_exp_log_variance(float("nan"))

        assert np.isnan(result)


class TestComputeVarianceStepEGARCH11:
    """Test cases for compute_variance_step_egarch11 function."""

    def test_basic_variance_step(self):
        """Should compute next variance step."""
        residual = 0.02  # Previous residual
        variance_prev = 0.0004  # Previous variance (sigma^2)
        omega = -0.1
        alpha = 0.15
        gamma = -0.08
        beta = 0.92
        kappa = 0.8  # E[|Z|] for Student-t

        result = compute_variance_step_egarch11(
            residual, variance_prev, omega, alpha, gamma, beta, kappa
        )

        # Result should be positive and finite
        assert result > 0
        assert np.isfinite(result)

    def test_leverage_effect(self):
        """Negative residual should increase variance (leverage effect)."""
        variance_prev = 0.0004
        omega = -0.1
        alpha = 0.15
        gamma = -0.08  # Negative gamma = leverage
        beta = 0.92
        kappa = 0.8

        # Positive shock
        var_pos = compute_variance_step_egarch11(
            0.02, variance_prev, omega, alpha, gamma, beta, kappa
        )

        # Negative shock (same magnitude)
        var_neg = compute_variance_step_egarch11(
            -0.02, variance_prev, omega, alpha, gamma, beta, kappa
        )

        # Negative shock should produce higher variance (leverage effect)
        assert var_neg > var_pos

    def test_handles_small_variance(self):
        """Should handle very small previous variance."""
        from src.constants import GARCH_MIN_INIT_VAR

        result = compute_variance_step_egarch11(
            0.01, 1e-20, -0.1, 0.15, -0.08, 0.92, 0.8
        )

        # Should still produce valid result (uses safe_variance internally)
        assert result > 0 or np.isnan(result)


class TestValidateAndExtractParams:
    """Test cases for parameter extraction validation functions."""

    def test_extract_alpha_order_1(self):
        """Should extract alpha for o=1."""
        alpha1, alpha2 = _validate_and_extract_alpha(0.15, o=1)

        assert alpha1 == 0.15
        assert alpha2 == 0.0

    def test_extract_alpha_order_2(self):
        """Should extract alpha tuple for o=2."""
        alpha1, alpha2 = _validate_and_extract_alpha((0.10, 0.05), o=2)

        assert alpha1 == 0.10
        assert alpha2 == 0.05

    def test_raises_on_alpha_type_mismatch_o1(self):
        """Should raise ValueError for tuple alpha with o=1."""
        with pytest.raises(ValueError, match="o=1"):
            _validate_and_extract_alpha((0.1, 0.05), o=1)

    def test_raises_on_alpha_type_mismatch_o2(self):
        """Should raise ValueError for float alpha with o=2."""
        with pytest.raises(ValueError, match="o=2"):
            _validate_and_extract_alpha(0.15, o=2)

    def test_extract_gamma_order_1(self):
        """Should extract gamma for o=1."""
        gamma1, gamma2 = _validate_and_extract_gamma(-0.08, o=1)

        assert gamma1 == -0.08
        assert gamma2 == 0.0

    def test_extract_beta_order_1(self):
        """Should extract beta for p=1."""
        beta1, beta2, beta3 = _validate_and_extract_beta(0.92, p=1)

        assert beta1 == 0.92
        assert beta2 == 0.0
        assert beta3 == 0.0

    def test_extract_beta_order_2(self):
        """Should extract beta for p=2."""
        beta1, beta2, beta3 = _validate_and_extract_beta((0.7, 0.2), p=2)

        assert beta1 == 0.7
        assert beta2 == 0.2
        assert beta3 == 0.0

    def test_extract_beta_order_3(self):
        """Should extract beta for p=3."""
        beta1, beta2, beta3 = _validate_and_extract_beta((0.5, 0.3, 0.1), p=3)

        assert beta1 == 0.5
        assert beta2 == 0.3
        assert beta3 == 0.1

    def test_raises_on_beta_type_mismatch(self):
        """Should raise ValueError for wrong beta type."""
        with pytest.raises(ValueError, match="p=1"):
            _validate_and_extract_beta((0.7, 0.2), p=1)


class TestValidateParamTypes:
    """Test cases for validate_param_types function."""

    def test_egarch11_params(self):
        """Should validate EGARCH(1,1) params."""
        result = validate_param_types(
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            o=1,
            p=1,
        )

        alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3 = result
        assert alpha1 == 0.15
        assert alpha2 == 0.0
        assert gamma1 == -0.08
        assert gamma2 == 0.0
        assert beta1 == 0.92
        assert beta2 == 0.0
        assert beta3 == 0.0

    def test_egarch22_params(self):
        """Should validate EGARCH(2,2) params."""
        result = validate_param_types(
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=(0.7, 0.2),
            o=2,
            p=2,
        )

        alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3 = result
        assert alpha1 == 0.10
        assert alpha2 == 0.05
        assert gamma1 == -0.05
        assert gamma2 == -0.03
        assert beta1 == 0.7
        assert beta2 == 0.2


class TestComputeVariancePathEGARCH11:
    """Test cases for compute_variance_path_egarch11 function."""

    def test_computes_variance_path(self):
        """Should compute full variance path."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = compute_variance_path_egarch11(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
        )

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))
        assert np.all(sigma2 > 0)

    def test_variance_responds_to_shocks(self):
        """Variance should increase after large residuals."""
        # Construct residuals with a large shock
        residuals = np.array([0.01] * 50 + [0.10] * 10 + [0.01] * 40)

        sigma2 = compute_variance_path_egarch11(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
        )

        # Variance should be higher after the shock
        var_before_shock = np.mean(sigma2[40:50])
        var_after_shock = np.mean(sigma2[55:65])

        assert var_after_shock > var_before_shock

    def test_with_custom_init(self):
        """Should use custom initial variance."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        init_var = 0.001
        sigma2 = compute_variance_path_egarch11(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            init=init_var,
        )

        assert sigma2[0] == init_var


class TestComputeVariancePath:
    """Test cases for compute_variance_path function."""

    def test_egarch11(self):
        """Should compute EGARCH(1,1) variance path."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
        )

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))
        assert np.all(sigma2 > 0)

    def test_egarch12(self):
        """Should compute EGARCH(1,2) variance path."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=(0.7, 0.2),
            kappa=0.8,
            o=1,
            p=2,
        )

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))

    def test_egarch21(self):
        """Should compute EGARCH(2,1) variance path."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=0.90,
            kappa=0.8,
            o=2,
            p=1,
        )

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))

    def test_egarch22(self):
        """Should compute EGARCH(2,2) variance path."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=(0.10, 0.05),
            gamma=(-0.05, -0.03),
            beta=(0.7, 0.2),
            kappa=0.8,
            o=2,
            p=2,
        )

        assert len(sigma2) == len(residuals)
        assert np.all(np.isfinite(sigma2))

    def test_raises_on_invalid_arch_order(self):
        """Should raise ValueError for invalid ARCH order."""
        residuals = np.random.randn(50) * 0.02

        with pytest.raises(ValueError, match="o=3"):
            compute_variance_path(
                residuals,
                omega=-0.1,
                alpha=0.15,
                gamma=-0.08,
                beta=0.92,
                kappa=0.8,
                o=3,
                p=1,
            )

    def test_raises_on_invalid_garch_order(self):
        """Should raise ValueError for invalid GARCH order."""
        residuals = np.random.randn(50) * 0.02

        with pytest.raises(ValueError, match="p=4"):
            compute_variance_path(
                residuals,
                omega=-0.1,
                alpha=0.15,
                gamma=-0.08,
                beta=0.92,
                kappa=0.8,
                o=1,
                p=4,
            )


class TestVariancePathAntiLeakage:
    """CRITICAL: Anti-leakage tests for variance computation."""

    def test_variance_uses_only_past_residuals(self):
        """CRITICAL: Variance at t uses residuals up to t-1 (causal recursion)."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        # Use fixed init to avoid sample variance dependency
        init_var = 0.0004

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
            init=init_var,
        )

        # Verify causality: sigma2[t] depends only on residuals[0:t-1]
        # Change residual at t=50, only sigma2[51:] should change
        residuals_modified = residuals.copy()
        residuals_modified[50] = 0.10  # Large shock at t=50

        sigma2_modified = compute_variance_path(
            residuals_modified,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
            init=init_var,
        )

        # sigma2[0:51] should be identical (shock at t=50 affects sigma2[51])
        np.testing.assert_array_almost_equal(sigma2[:51], sigma2_modified[:51])
        # sigma2[51] should be different
        assert sigma2[51] != sigma2_modified[51]

    def test_changing_future_doesnt_affect_past_variance(self):
        """CRITICAL: Changing future residuals must not affect past variance."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        # Use fixed init to ensure identical starting point
        init_var = 0.0004

        # Compute variance
        sigma2_original = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
            init=init_var,
        )

        # Modify future residuals (starting at t=60)
        residuals_modified = residuals.copy()
        residuals_modified[60:] = np.random.randn(40) * 0.10

        sigma2_modified = compute_variance_path(
            residuals_modified,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
            init=init_var,
        )

        # Variance up to t=60 should be identical (causality)
        # Note: sigma2[t] depends on residuals[t-1], so sigma2[0:61] should be same
        np.testing.assert_array_almost_equal(sigma2_original[:61], sigma2_modified[:61])

    def test_variance_is_causal(self):
        """CRITICAL: Variance filtering is causal (no look-ahead)."""
        # Variance at time t depends only on:
        # - Previous variance(s)
        # - Previous residual(s)
        # - Model parameters

        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
        )

        # Manually verify variance at t=10 using only past information
        # log(σ²₁₀) = ω + β·log(σ²₉) + α·(|z₉| - κ) + γ·z₉
        z_9 = residuals[9] / np.sqrt(sigma2[9])
        log_var_10_manual = (
            -0.1 + 0.92 * np.log(sigma2[9]) + 0.15 * (abs(z_9) - 0.8) + (-0.08) * z_9
        )
        var_10_manual = np.exp(np.clip(log_var_10_manual, -30, 30))

        assert abs(sigma2[10] - var_10_manual) < 1e-8


class TestVariancePathNumericalStability:
    """Tests for numerical stability of variance computation."""

    def test_handles_large_residuals(self):
        """Should handle large residuals without overflow."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02
        residuals[50] = 0.5  # Large shock

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
        )

        # Should still produce valid results
        assert np.all(np.isfinite(sigma2))

    def test_handles_small_residuals(self):
        """Should handle very small residuals."""
        residuals = np.full(100, 1e-8)

        sigma2 = compute_variance_path(
            residuals,
            omega=-0.1,
            alpha=0.15,
            gamma=-0.08,
            beta=0.92,
            kappa=0.8,
            o=1,
            p=1,
        )

        assert np.all(np.isfinite(sigma2))
        assert np.all(sigma2 > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
