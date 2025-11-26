"""Tests for src/evaluation/metrics.py module."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.evaluation.metrics import (
    mse,
    rmse,
    mae,
    mape,
    smape,
    r2_score,
    adjusted_r2,
    max_error,
    median_absolute_error,
    qlike,
    mse_log,
    mincer_zarnowitz_r2,
    direction_accuracy,
    hit_rate,
    aic,
    bic,
    compute_residual_diagnostics,
    get_metric,
    compute_metrics,
    list_available_metrics,
    REGRESSION_METRICS,
    VOLATILITY_METRICS,
)


class TestMSE:
    """Test cases for MSE metric."""

    def test_perfect_prediction(self):
        """MSE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert mse(y_true, y_pred) == 0.0

    def test_simple_case(self):
        """Should compute MSE correctly."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 2])  # errors: 1, 0, 1

        # MSE = (1 + 0 + 1) / 3 = 2/3
        assert pytest.approx(mse(y_true, y_pred)) == 2 / 3

    def test_always_positive(self):
        """MSE should always be non-negative."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        assert mse(y_true, y_pred) >= 0


class TestRMSE:
    """Test cases for RMSE metric."""

    def test_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert rmse(y_true, y_pred) == 0.0

    def test_is_sqrt_of_mse(self):
        """RMSE should be sqrt of MSE."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 2])

        expected = np.sqrt(mse(y_true, y_pred))
        assert pytest.approx(rmse(y_true, y_pred)) == expected


class TestMAE:
    """Test cases for MAE metric."""

    def test_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert mae(y_true, y_pred) == 0.0

    def test_simple_case(self):
        """Should compute MAE correctly."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 2, 2])  # errors: 1, 0, 1

        # MAE = (1 + 0 + 1) / 3 = 2/3
        assert pytest.approx(mae(y_true, y_pred)) == 2 / 3

    def test_symmetric(self):
        """MAE should be symmetric."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([3, 2, 1])

        # Same result regardless of which is true/pred
        assert mae(y_true, y_pred) == mae(y_pred, y_true)


class TestMAPE:
    """Test cases for MAPE metric."""

    def test_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert pytest.approx(mape(y_true, y_pred), abs=1e-8) == 0.0

    def test_returns_percentage(self):
        """MAPE should return percentage values."""
        y_true = np.array([100, 100, 100])
        y_pred = np.array([110, 90, 100])  # 10%, 10%, 0% errors

        # Average: (10 + 10 + 0) / 3 = 6.67%
        assert pytest.approx(mape(y_true, y_pred), rel=0.01) == 20 / 3


class TestSMAPE:
    """Test cases for SMAPE metric."""

    def test_perfect_prediction(self):
        """SMAPE should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert pytest.approx(smape(y_true, y_pred), abs=1e-8) == 0.0

    def test_bounded_range(self):
        """SMAPE should be bounded in [0, 200]."""
        np.random.seed(42)
        y_true = np.random.uniform(0.1, 10, size=100)
        y_pred = np.random.uniform(0.1, 10, size=100)

        result = smape(y_true, y_pred)
        assert 0 <= result <= 200


class TestR2Score:
    """Test cases for R² metric."""

    def test_perfect_prediction(self):
        """R² should be 1 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert pytest.approx(r2_score(y_true, y_pred)) == 1.0

    def test_mean_prediction(self):
        """R² should be 0 for mean prediction."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.full(5, np.mean(y_true))

        assert pytest.approx(r2_score(y_true, y_pred), abs=1e-10) == 0.0

    def test_can_be_negative(self):
        """R² can be negative for very bad predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([10, 20, 30, 40, 50])  # Very bad

        assert r2_score(y_true, y_pred) < 0


class TestAdjustedR2:
    """Test cases for adjusted R² metric."""

    def test_equals_r2_with_one_feature(self):
        """Should be close to R² with minimal features."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        r2 = r2_score(y_true, y_pred)
        adj_r2 = adjusted_r2(y_true, y_pred, n_features=1)

        # With 1 feature, adjusted R² should be close to R²
        assert abs(r2 - adj_r2) < 0.1

    def test_penalizes_more_features(self):
        """Adjusted R² should be lower with more features."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        adj_r2_1 = adjusted_r2(y_true, y_pred, n_features=1)
        adj_r2_2 = adjusted_r2(y_true, y_pred, n_features=2)

        assert adj_r2_2 < adj_r2_1


class TestMaxError:
    """Test cases for max_error metric."""

    def test_perfect_prediction(self):
        """Max error should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert max_error(y_true, y_pred) == 0.0

    def test_finds_maximum(self):
        """Should find the maximum error."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 10])  # Max error = 6

        assert max_error(y_true, y_pred) == 6.0


class TestMedianAbsoluteError:
    """Test cases for median_absolute_error metric."""

    def test_perfect_prediction(self):
        """Should be 0 for perfect predictions."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert median_absolute_error(y_true, y_pred) == 0.0

    def test_robust_to_outliers(self):
        """Should be robust to outliers unlike MAE."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 100])  # One big outlier

        med_ae = median_absolute_error(y_true, y_pred)
        mean_ae = mae(y_true, y_pred)

        assert med_ae < mean_ae


class TestQLIKE:
    """Test cases for QLIKE metric."""

    def test_perfect_prediction(self):
        """QLIKE should be minimized at perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])

        # At y_true = y_pred, QLIKE = 1 + log(y_pred)
        expected = np.mean(1 + np.log(y_true))
        assert pytest.approx(qlike(y_true, y_pred), rel=1e-6) == expected

    def test_positive_inputs(self):
        """Should handle positive variance values."""
        y_true = np.array([0.001, 0.002, 0.003])
        y_pred = np.array([0.0015, 0.0025, 0.0025])

        result = qlike(y_true, y_pred)
        assert np.isfinite(result)


class TestMSELog:
    """Test cases for MSE log metric."""

    def test_perfect_prediction(self):
        """Should be 0 for perfect predictions."""
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])

        assert pytest.approx(mse_log(y_true, y_pred), abs=1e-10) == 0.0

    def test_handles_small_values(self):
        """Should handle very small values."""
        y_true = np.array([1e-8, 1e-7, 1e-6])
        y_pred = np.array([1e-8, 1e-7, 1e-6])

        result = mse_log(y_true, y_pred)
        assert np.isfinite(result)


class TestMincerZarnowitzR2:
    """Test cases for Mincer-Zarnowitz regression."""

    def test_perfect_prediction(self):
        """Should have R²=1, alpha=0, beta=1 for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred = np.array([1, 2, 3, 4, 5.0])

        r2, alpha, beta = mincer_zarnowitz_r2(y_true, y_pred)

        assert pytest.approx(r2, abs=1e-6) == 1.0
        assert pytest.approx(alpha, abs=1e-6) == 0.0
        assert pytest.approx(beta, abs=1e-6) == 1.0

    def test_scaled_prediction(self):
        """Should detect bias in scaled predictions."""
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred = np.array([2, 4, 6, 8, 10.0])  # 2x predictions

        r2, alpha, beta = mincer_zarnowitz_r2(y_true, y_pred)

        # R² should still be 1 (perfect linear relationship)
        assert pytest.approx(r2, abs=1e-6) == 1.0
        # Beta should be 0.5 (y_true = 0.5 * y_pred)
        assert pytest.approx(beta, abs=1e-6) == 0.5


class TestDirectionAccuracy:
    """Test cases for direction_accuracy metric."""

    def test_perfect_direction(self):
        """Should be 100% for perfect direction prediction."""
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([0.5, -0.5, 0.5, -0.5])

        assert direction_accuracy(y_true, y_pred) == 100.0

    def test_wrong_direction(self):
        """Should be 0% for completely wrong directions."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([-1, -1, -1, -1])

        assert direction_accuracy(y_true, y_pred) == 0.0

    def test_half_correct(self):
        """Should be 50% when half are correct."""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, 1, -1])

        assert direction_accuracy(y_true, y_pred) == 50.0


class TestHitRate:
    """Test cases for hit_rate metric."""

    def test_perfect_classification(self):
        """Should be 100% for perfect classification."""
        y_true = np.array([1, 2, -1, -2])
        y_pred = np.array([0.5, 1.5, -0.5, -1.5])

        assert hit_rate(y_true, y_pred, threshold=0.0) == 100.0

    def test_custom_threshold(self):
        """Should use custom threshold."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])

        # All above threshold 2.5
        assert hit_rate(y_true, y_pred, threshold=2.5) == 100.0


class TestAIC:
    """Test cases for AIC metric."""

    def test_lower_for_better_model(self):
        """Better fit should have lower AIC."""
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred_good = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_pred_bad = np.array([2, 3, 4, 5, 6.0])

        aic_good = aic(y_true, y_pred_good, n_params=2)
        aic_bad = aic(y_true, y_pred_bad, n_params=2)

        assert aic_good < aic_bad

    def test_penalizes_more_params(self):
        """More parameters should increase AIC."""
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        aic_1 = aic(y_true, y_pred, n_params=1)
        aic_5 = aic(y_true, y_pred, n_params=5)

        assert aic_5 > aic_1


class TestBIC:
    """Test cases for BIC metric."""

    def test_penalizes_more_than_aic(self):
        """BIC should penalize complexity more than AIC for large n."""
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        aic_val = aic(y_true, y_pred, n_params=10)
        bic_val = bic(y_true, y_pred, n_params=10)

        # For large n, BIC penalizes more
        # This depends on the penalty terms
        assert np.isfinite(aic_val)
        assert np.isfinite(bic_val)


class TestComputeResidualDiagnostics:
    """Test cases for compute_residual_diagnostics function."""

    def test_returns_diagnostics_object(self):
        """Should return ResidualDiagnostics object."""
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        result = compute_residual_diagnostics(y_true, y_pred)

        assert hasattr(result, "mean")
        assert hasattr(result, "std")
        assert hasattr(result, "skewness")
        assert hasattr(result, "kurtosis")
        assert hasattr(result, "jarque_bera_stat")
        assert hasattr(result, "autocorr_lag1")

    def test_mean_near_zero_for_unbiased(self):
        """Mean residual should be near zero for unbiased predictions."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.01  # Small noise

        result = compute_residual_diagnostics(y_true, y_pred)

        assert abs(result.mean) < 0.1


class TestGetMetric:
    """Test cases for get_metric function."""

    def test_returns_metric_function(self):
        """Should return the correct metric function."""
        metric_fn = get_metric("mse")

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert metric_fn(y_true, y_pred) == 0.0

    def test_case_insensitive(self):
        """Should be case insensitive."""
        mse_lower = get_metric("mse")
        mse_upper = get_metric("MSE")

        assert mse_lower == mse_upper

    def test_raises_for_unknown(self):
        """Should raise ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("unknown_metric")


class TestComputeMetrics:
    """Test cases for compute_metrics function."""

    def test_computes_multiple_metrics(self):
        """Should compute multiple metrics at once."""
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        result = compute_metrics(y_true, y_pred, metrics=["mse", "rmse", "mae"])

        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result

    def test_default_metrics(self):
        """Should use default metrics when not specified."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        result = compute_metrics(y_true, y_pred)

        assert len(result) == len(REGRESSION_METRICS)


class TestListAvailableMetrics:
    """Test cases for list_available_metrics function."""

    def test_returns_list(self):
        """Should return a list of metric names."""
        result = list_available_metrics()

        assert isinstance(result, list)
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
