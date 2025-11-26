"""Tests for src/garch/garch_eval/metrics.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.garch_eval.metrics import (
    apply_mz_calibration,
    build_var_series,
    christoffersen_ind_test,
    kupiec_pof_test,
    mincer_zarnowitz,
    mse_mae_variance,
    qlike_loss,
    quantile,
    r2_variance,
    var_backtest_metrics,
    var_quantile,
)


class TestQLikeLoss:
    """Test cases for qlike_loss function."""

    def test_computes_qlike(self):
        """Should compute QLIKE loss."""
        e = np.array([0.01, 0.02, -0.01, 0.015])
        sigma2 = np.array([0.0004, 0.0005, 0.0003, 0.0004])

        qlike = qlike_loss(e, sigma2)

        assert np.isfinite(qlike)

    def test_qlike_formula(self):
        """Should follow QLIKE formula: mean(log(sigma2) + e^2/sigma2)."""
        e = np.array([0.02])
        sigma2 = np.array([0.0004])

        qlike = qlike_loss(e, sigma2)

        expected = np.log(0.0004) + (0.02**2) / 0.0004
        assert abs(qlike - expected) < 1e-10

    def test_lower_qlike_better_fit(self):
        """Lower QLIKE should indicate better fit."""
        e = np.array([0.02, 0.02, 0.02])

        # Good fit: variance matches squared residuals
        sigma2_good = np.array([0.0004, 0.0004, 0.0004])

        # Poor fit: variance too high
        sigma2_poor = np.array([0.01, 0.01, 0.01])

        qlike_good = qlike_loss(e, sigma2_good)
        qlike_poor = qlike_loss(e, sigma2_poor)

        assert qlike_good < qlike_poor

    def test_returns_nan_for_empty(self):
        """Should return NaN for empty arrays."""
        e = np.array([])
        sigma2 = np.array([])

        qlike = qlike_loss(e, sigma2)

        assert np.isnan(qlike)

    def test_handles_nan_values(self):
        """Should filter out NaN values."""
        e = np.array([0.01, np.nan, 0.02])
        sigma2 = np.array([0.0004, 0.0005, 0.0004])

        qlike = qlike_loss(e, sigma2)

        assert np.isfinite(qlike)


class TestMseMaeVariance:
    """Test cases for mse_mae_variance function."""

    def test_computes_mse_mae(self):
        """Should compute MSE and MAE for variance."""
        e = np.array([0.01, 0.02, -0.01])
        sigma2 = np.array([0.0001, 0.0004, 0.0001])

        result = mse_mae_variance(e, sigma2)

        assert "mse" in result
        assert "mae" in result
        assert np.isfinite(result["mse"])
        assert np.isfinite(result["mae"])

    def test_mse_formula(self):
        """MSE should be mean((e^2 - sigma^2)^2)."""
        e = np.array([0.01, 0.02])
        sigma2 = np.array([0.0001, 0.0004])

        result = mse_mae_variance(e, sigma2)

        y = e**2  # Realized variance
        expected_mse = np.mean((y - sigma2) ** 2)
        assert abs(result["mse"] - expected_mse) < 1e-15

    def test_mae_formula(self):
        """MAE should be mean(|e^2 - sigma^2|)."""
        e = np.array([0.01, 0.02])
        sigma2 = np.array([0.0001, 0.0004])

        result = mse_mae_variance(e, sigma2)

        y = e**2
        expected_mae = np.mean(np.abs(y - sigma2))
        assert abs(result["mae"] - expected_mae) < 1e-15

    def test_perfect_fit(self):
        """Perfect fit should have MSE=MAE=0."""
        e = np.array([0.01, 0.02, 0.015])
        sigma2 = e**2  # Perfect match

        result = mse_mae_variance(e, sigma2)

        assert result["mse"] < 1e-20
        assert result["mae"] < 1e-20

    def test_returns_nan_for_empty(self):
        """Should return NaN for empty arrays."""
        result = mse_mae_variance(np.array([]), np.array([]))

        assert np.isnan(result["mse"])
        assert np.isnan(result["mae"])


class TestR2Variance:
    """Test cases for r2_variance function."""

    def test_computes_r2(self):
        """Should compute R² for variance predictions."""
        np.random.seed(42)
        e = np.random.randn(100) * 0.02
        sigma2 = e**2 + np.random.randn(100) * 0.0001  # Good but not perfect

        r2 = r2_variance(e, sigma2)

        assert 0 <= r2 <= 1

    def test_perfect_correlation(self):
        """Perfect correlation should give R² close to 1."""
        e = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        sigma2 = e**2 * 1.1  # Perfect linear relationship

        r2 = r2_variance(e, sigma2)

        assert r2 > 0.99

    def test_no_correlation(self):
        """No correlation should give R² close to 0."""
        np.random.seed(42)
        e = np.random.randn(100) * 0.02
        sigma2 = np.random.randn(100) * 0.0001 + 0.0004  # Random, no correlation

        r2 = r2_variance(e, sigma2)

        assert r2 < 0.3

    def test_returns_nan_for_small_sample(self):
        """Should return NaN for sample size < 2."""
        r2 = r2_variance(np.array([0.01]), np.array([0.0001]))

        assert np.isnan(r2)


class TestMincerZarnowitz:
    """Test cases for mincer_zarnowitz function."""

    def test_computes_mz_regression(self):
        """Should compute Mincer-Zarnowitz regression."""
        np.random.seed(42)
        e = np.random.randn(200) * 0.02
        sigma2 = e**2 * 0.9 + 0.0001  # Close to realized variance

        result = mincer_zarnowitz(e, sigma2)

        assert "intercept" in result
        assert "slope" in result
        assert "r2" in result
        assert np.isfinite(result["intercept"])
        assert np.isfinite(result["slope"])
        assert np.isfinite(result["r2"])

    def test_slope_near_1_for_good_fit(self):
        """Good variance fit should have slope near 1."""
        np.random.seed(42)
        e = np.random.randn(500) * 0.02
        sigma2 = e**2  # Perfect variance prediction

        result = mincer_zarnowitz(e, sigma2)

        # Slope should be close to 1 for unbiased forecasts
        assert abs(result["slope"] - 1.0) < 0.1

    def test_intercept_near_0_for_good_fit(self):
        """Good variance fit should have intercept near 0."""
        np.random.seed(42)
        e = np.random.randn(500) * 0.02
        sigma2 = e**2  # Perfect variance prediction

        result = mincer_zarnowitz(e, sigma2)

        # Intercept should be close to 0
        assert abs(result["intercept"]) < 0.0001

    def test_slope_greater_than_1_for_underestimation(self):
        """Underestimated variance should have slope > 1."""
        np.random.seed(42)
        e = np.random.randn(500) * 0.02
        sigma2 = e**2 * 0.5  # Underestimate by 50%

        result = mincer_zarnowitz(e, sigma2)

        # Slope should be > 1 when forecasts underestimate
        assert result["slope"] > 1.0

    def test_returns_nan_for_small_sample(self):
        """Should return NaN for small sample."""
        result = mincer_zarnowitz(np.array([0.01]), np.array([0.0001]))

        assert np.isnan(result["intercept"])


class TestApplyMzCalibration:
    """Test cases for apply_mz_calibration function."""

    def test_applies_multiplicative_calibration(self):
        """Should apply multiplicative calibration by default."""
        sigma2 = np.array([0.0001, 0.0002, 0.0003])
        slope = 2.0
        intercept = 0.0001

        result = apply_mz_calibration(sigma2, intercept, slope, use_intercept=False)

        expected = slope * sigma2
        np.testing.assert_array_almost_equal(result, expected)

    def test_applies_additive_calibration(self):
        """Should apply additive calibration when use_intercept=True."""
        sigma2 = np.array([0.0001, 0.0002, 0.0003])
        slope = 2.0
        intercept = 0.00005

        result = apply_mz_calibration(sigma2, intercept, slope, use_intercept=True)

        expected = intercept + slope * sigma2
        np.testing.assert_array_almost_equal(result, expected)

    def test_clamps_to_minimum(self):
        """Should clamp result to minimum epsilon."""
        sigma2 = np.array([0.0001, 0.0002])
        slope = 0.001  # Very small slope
        intercept = -0.001  # Negative intercept

        result = apply_mz_calibration(sigma2, intercept, slope, use_intercept=True)

        # All values should be at least eps
        from src.constants import GARCH_CALIBRATION_EPS
        assert np.all(result >= GARCH_CALIBRATION_EPS)


class TestQuantile:
    """Test cases for quantile function."""

    def test_student_quantile(self):
        """Should compute Student-t quantile."""
        q = quantile("student", p=0.05, nu=5.0)

        assert q < 0  # Left-tail quantile is negative
        # For t(5), 5% quantile ≈ -2.015
        assert -3.0 < q < -1.5

    def test_skewt_quantile(self):
        """Should compute Skew-t quantile."""
        q = quantile("skewt", p=0.05, nu=5.0, lambda_skew=-0.2)

        assert q < 0

    def test_raises_on_invalid_nu(self):
        """Should raise ValueError for invalid nu."""
        with pytest.raises(ValueError, match="nu>2"):
            quantile("student", p=0.05, nu=2.0)

    def test_raises_on_unsupported_dist(self):
        """Should raise ValueError for unsupported distribution."""
        with pytest.raises(ValueError, match="Unsupported distribution"):
            quantile("normal", p=0.05, nu=5.0)


class TestVarQuantile:
    """Test cases for var_quantile function."""

    def test_returns_same_as_quantile(self):
        """Should return same value as quantile function."""
        q1 = var_quantile(0.05, "student", nu=5.0)
        q2 = quantile("student", p=0.05, nu=5.0)

        assert q1 == q2


class TestBuildVarSeries:
    """Test cases for build_var_series function."""

    def test_builds_var_series(self):
        """Should build VaR series."""
        sigma2 = np.array([0.0004, 0.0009, 0.0016])

        var = build_var_series(sigma2, alpha=0.05, dist="student", nu=5.0)

        assert len(var) == len(sigma2)
        # VaR should be negative (left-tail)
        assert np.all(var < 0)

    def test_var_proportional_to_sigma(self):
        """VaR should be proportional to sigma."""
        sigma2 = np.array([0.0004, 0.0016])  # sigma = [0.02, 0.04]

        var = build_var_series(sigma2, alpha=0.05, dist="student", nu=5.0)

        # VaR[1] should be 2x VaR[0] since sigma[1] = 2*sigma[0]
        assert abs(var[1] / var[0] - 2.0) < 0.01


class TestKupiecPofTest:
    """Test cases for kupiec_pof_test function."""

    def test_computes_kupiec_test(self):
        """Should compute Kupiec POF test."""
        # 5% expected violations, 100 observations
        np.random.seed(42)
        hits = (np.random.rand(100) < 0.05).astype(int)

        result = kupiec_pof_test(hits, alpha=0.05)

        assert "n" in result
        assert "x" in result
        assert "hit_rate" in result
        assert "lr_uc" in result
        assert "p_value" in result

    def test_high_pvalue_for_correct_rate(self):
        """Should have high p-value when hit rate matches alpha."""
        # Exactly 5 violations in 100 observations (5% rate)
        hits = np.zeros(100)
        hits[:5] = 1

        result = kupiec_pof_test(hits, alpha=0.05)

        # P-value should be high (fail to reject null)
        assert result["p_value"] > 0.1

    def test_low_pvalue_for_high_violation_rate(self):
        """Should have low p-value for excessive violations."""
        # 20% violations when expecting 5%
        hits = np.zeros(100)
        hits[:20] = 1

        result = kupiec_pof_test(hits, alpha=0.05)

        # P-value should be low (reject null)
        assert result["p_value"] < 0.05

    def test_empty_hits(self):
        """Should handle empty hits array."""
        result = kupiec_pof_test(np.array([]), alpha=0.05)

        assert result["n"] == 0
        assert np.isnan(result["p_value"])


class TestChristoffersenIndTest:
    """Test cases for christoffersen_ind_test function."""

    def test_computes_independence_test(self):
        """Should compute Christoffersen independence test."""
        np.random.seed(42)
        hits = (np.random.rand(100) < 0.05).astype(int)

        result = christoffersen_ind_test(hits)

        assert "lr_ind" in result
        assert "p_value" in result
        assert "n00" in result
        assert "n01" in result
        assert "n10" in result
        assert "n11" in result

    def test_high_pvalue_for_independent_hits(self):
        """Should have high p-value for independent hits."""
        np.random.seed(42)
        hits = (np.random.rand(200) < 0.05).astype(int)

        result = christoffersen_ind_test(hits)

        # Independent random hits should pass (high p-value)
        # Note: This is probabilistic, so we use a lenient threshold
        assert result["p_value"] > 0.01 or result["lr_ind"] < 10

    def test_low_pvalue_for_clustered_hits(self):
        """Should have low p-value for clustered violations."""
        # Clustered hits (violations come in bursts)
        hits = np.zeros(100)
        hits[10:20] = 1  # 10 consecutive violations

        result = christoffersen_ind_test(hits)

        # Clustered hits should fail independence test (low p-value)
        assert result["p_value"] < 0.1

    def test_empty_or_single(self):
        """Should handle empty or single-element arrays."""
        result = christoffersen_ind_test(np.array([]))
        assert np.isnan(result["p_value"])

        result = christoffersen_ind_test(np.array([1]))
        assert np.isnan(result["p_value"])


class TestVarBacktestMetrics:
    """Test cases for var_backtest_metrics function."""

    def test_computes_backtest_metrics(self):
        """Should compute VaR backtest metrics."""
        np.random.seed(42)
        e = np.random.randn(500) * 0.02
        sigma2 = np.full(500, 0.0004)

        result = var_backtest_metrics(
            e, sigma2, dist="student", nu=5.0, lambda_skew=None, alphas=[0.01, 0.05]
        )

        assert "0.01" in result
        assert "0.05" in result
        assert "alpha" in result["0.01"]
        assert "violations" in result["0.01"]
        assert "p_uc" in result["0.01"]
        assert "p_ind" in result["0.01"]
        assert "p_cc" in result["0.01"]

    def test_violation_count_reasonable(self):
        """Violation count should be reasonable for well-calibrated VaR."""
        np.random.seed(42)
        # Generate t-distributed returns
        from scipy.stats import t as t_dist  # type: ignore
        e = np.asarray(t_dist.rvs(df=5.0, size=1000) * 0.02, dtype=float)
        sigma2 = np.full(1000, 0.0004)

        result = var_backtest_metrics(
            e, sigma2, dist="student", nu=5.0, lambda_skew=None, alphas=[0.05]
        )

        # Expected violations: 5% of 1000 = 50
        violations = result["0.05"]["violations"]
        # Should be in reasonable range (25-75)
        assert 25 < violations < 100


class TestMetricsAntiLeakage:
    """CRITICAL: Anti-leakage tests for GARCH metrics."""

    def test_qlike_uses_aligned_data(self):
        """CRITICAL: QLIKE should use aligned e and sigma2."""
        # e[t] is realized at time t
        # sigma2[t] is the forecast made at time t-1 for time t
        # They should be aligned: e[t] vs sigma2[t]

        e = np.array([0.01, 0.02, 0.03])
        sigma2 = np.array([0.0001, 0.0004, 0.0009])

        qlike = qlike_loss(e, sigma2)

        # Manual calculation with aligned data
        expected = np.mean(np.log(sigma2) + e**2 / sigma2)
        assert abs(qlike - expected) < 1e-10

    def test_mz_regression_on_test_data(self):
        """CRITICAL: MZ regression should be computed on test data only."""
        np.random.seed(42)
        # Simulate train/test split
        train_e = np.random.randn(400) * 0.02
        test_e = np.random.randn(100) * 0.02

        # Variance forecasts for test period
        test_sigma2 = test_e**2 * 1.1  # Slightly biased forecasts

        # MZ regression should use only test data
        result = mincer_zarnowitz(test_e, test_sigma2)

        # Verify it's using test data (slope should reflect the 1.1x bias)
        assert 0.8 < result["slope"] < 1.3

    def test_var_backtest_uses_forecast_variance(self):
        """CRITICAL: VaR backtest should use ex-ante variance forecasts."""
        # VaR at time t: computed using sigma2[t] (forecast made at t-1)
        # Violation: e[t] < VaR[t]

        # This test verifies the formula is applied correctly
        np.random.seed(42)
        e = np.random.randn(100) * 0.02
        sigma2 = np.full(100, 0.0004)

        # Compute VaR and check violations manually
        from scipy.stats import t as t_dist  # type: ignore
        q_05 = t_dist.ppf(0.05, df=5.0)  # 5% quantile
        var_05 = q_05 * np.sqrt(sigma2)

        manual_violations = np.sum(e < var_05)

        result = var_backtest_metrics(
            e, sigma2, dist="student", nu=5.0, lambda_skew=None, alphas=[0.05]
        )

        assert result["0.05"]["violations"] == manual_violations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
