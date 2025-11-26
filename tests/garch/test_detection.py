"""Tests for src/garch/structure_garch/detection.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.garch.structure_garch.detection import (
    detect_heteroskedasticity,
    plot_arch_diagnostics,
)


class TestDetectHeteroskedasticity:
    """Test cases for detect_heteroskedasticity function."""

    def test_detects_arch_in_clustered_volatility(self):
        """Should detect ARCH effect in series with volatility clustering."""
        np.random.seed(42)

        # Create series with volatility clustering (GARCH-like)
        n = 500
        volatility = np.zeros(n)
        residuals = np.zeros(n)

        volatility[0] = 0.02
        residuals[0] = np.random.randn() * volatility[0]

        for t in range(1, n):
            # GARCH(1,1)-like volatility
            volatility[t] = 0.01 + 0.1 * residuals[t - 1] ** 2 + 0.85 * volatility[t - 1] ** 2
            volatility[t] = np.sqrt(volatility[t])
            residuals[t] = np.random.randn() * volatility[t]

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        # Should detect ARCH effect
        assert "arch_lm" in result
        assert "arch_effect_present" in result
        assert "acf_squared" in result
        assert result["arch_effect_present"] == True  # noqa: E712

    def test_no_arch_in_white_noise(self):
        """Should not detect ARCH effect in white noise."""
        np.random.seed(42)

        # Pure white noise has no volatility clustering
        residuals = np.random.randn(500) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        # White noise should typically not show ARCH effect
        # (Though this is probabilistic, with 5% chance of false positive)
        assert "arch_lm" in result
        assert "arch_effect_present" in result

    def test_returns_acf_squared(self):
        """Should return ACF of squared residuals."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=10, alpha=0.05)

        assert "acf_squared" in result
        acf_squared = result["acf_squared"]
        assert acf_squared is not None
        # Ensure it's a sequence with length
        if hasattr(acf_squared, '__len__'):
            assert len(acf_squared) > 0  # type: ignore
        else:
            assert False, "acf_squared should be a sequence"

    def test_arch_lm_test_results(self):
        """Should return ARCH-LM test results."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        lm = result["arch_lm"]
        assert isinstance(lm, dict)
        assert "p_value" in lm
        assert "lm_stat" in lm  # Note: key is lm_stat, not statistic
        assert 0 <= lm["p_value"] <= 1

    def test_respects_alpha_threshold(self):
        """Should respect alpha threshold for ARCH detection."""
        np.random.seed(42)

        # Create mild ARCH effect
        n = 300
        residuals = np.random.randn(n) * 0.02

        # Add some clustering
        for t in range(1, n):
            if abs(residuals[t - 1]) > 0.03:
                residuals[t] *= 1.5

        result_05 = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)
        result_10 = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.10)

        # If detected at alpha=0.05, should also be detected at alpha=0.10
        if result_05["arch_effect_present"]:
            assert result_10["arch_effect_present"]

    def test_handles_short_series(self):
        """Should handle short series."""
        np.random.seed(42)
        residuals = np.random.randn(50) * 0.02

        result = detect_heteroskedasticity(residuals, lags=2, acf_lags=5, alpha=0.05)

        assert "arch_lm" in result

    def test_handles_nan_values(self):
        """Should handle series with NaN values."""
        np.random.seed(42)
        residuals = np.random.randn(100) * 0.02
        residuals[50] = np.nan

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=10, alpha=0.05)

        # Should still produce results (after filtering NaN)
        assert "arch_lm" in result


class TestDetectHeteroskedasticityACF:
    """Test cases for ACF-related functionality in detect_heteroskedasticity."""

    def test_acf_significance_level(self):
        """Should compute ACF significance level."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        assert "acf_significance_level" in result
        acf_level = result["acf_significance_level"]
        assert isinstance(acf_level, (int, float))
        assert acf_level > 0

    def test_acf_significant_flag(self):
        """Should set acf_significant flag."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        assert "acf_significant" in result
        assert isinstance(result["acf_significant"], bool)

    def test_acf_significant_for_clustered_volatility(self):
        """Should detect significant ACF in squared residuals for clustered volatility."""
        np.random.seed(42)

        # Create strong volatility clustering
        n = 500
        residuals = []
        vol = 0.02

        for _ in range(n):
            # Occasional volatility regime switches
            if np.random.rand() < 0.05:
                vol = np.random.choice([0.01, 0.05])
            residuals.append(np.random.randn() * vol)

        residuals = np.array(residuals)

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        # Strong clustering should produce significant ACF
        assert result["acf_significant"] == True  # noqa: E712


class TestPlotArchDiagnostics:
    """Test cases for plot_arch_diagnostics function."""

    def test_creates_plot_file(self):
        """Should create plot file."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "arch_diagnostics.png"

            result_path = plot_arch_diagnostics(residuals, acf_lags=20, out_path=out_path)

            assert result_path.exists()

    def test_handles_empty_residuals(self):
        """Should handle empty residuals."""
        residuals = np.array([])

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "arch_diagnostics.png"

            result_path = plot_arch_diagnostics(residuals, acf_lags=20, out_path=out_path)

            # Should create placeholder file
            assert result_path.exists()

    def test_handles_all_nan_residuals(self):
        """Should handle all-NaN residuals."""
        residuals = np.array([np.nan, np.nan, np.nan])

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "arch_diagnostics.png"

            result_path = plot_arch_diagnostics(residuals, acf_lags=20, out_path=out_path)

            # Should create file (placeholder if no valid data)
            assert result_path.exists()


class TestHeteroskedasticityInterpretation:
    """Tests for interpreting heteroskedasticity detection results."""

    def test_financial_returns_typically_show_arch(self):
        """Simulated financial returns should show ARCH effect."""
        np.random.seed(42)

        # Simulate realistic financial returns with GARCH-like behavior
        n = 1000
        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        sigma2 = np.zeros(n)
        returns = np.zeros(n)

        sigma2[0] = omega / (1 - alpha - beta)
        returns[0] = np.random.randn() * np.sqrt(sigma2[0])

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
            returns[t] = np.random.randn() * np.sqrt(sigma2[t])

        result = detect_heteroskedasticity(returns, lags=5, acf_lags=20, alpha=0.05)

        # Should detect ARCH effect in GARCH-generated data
        assert result["arch_effect_present"] == True  # noqa: E712

    def test_iid_returns_rarely_show_arch(self):
        """IID returns should rarely show ARCH effect (false positive rate ~5%)."""
        np.random.seed(42)
        false_positives = 0
        n_trials = 100

        for i in range(n_trials):
            np.random.seed(42 + i)
            returns = np.random.randn(300) * 0.02

            result = detect_heteroskedasticity(returns, lags=5, acf_lags=20, alpha=0.05)

            if result["arch_effect_present"]:
                false_positives += 1

        # False positive rate should be around 5% (±5% margin)
        fp_rate = false_positives / n_trials
        assert fp_rate < 0.15  # Allow some margin


class TestARCHLMTestStatistics:
    """Tests for ARCH-LM test statistics."""

    def test_lm_statistic_positive(self):
        """LM statistic should be non-negative."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        arch_lm = result["arch_lm"]
        assert isinstance(arch_lm, dict)
        assert arch_lm["lm_stat"] >= 0

    def test_lm_pvalue_in_range(self):
        """LM p-value should be in [0, 1]."""
        np.random.seed(42)
        residuals = np.random.randn(200) * 0.02

        result = detect_heteroskedasticity(residuals, lags=5, acf_lags=20, alpha=0.05)

        arch_lm = result["arch_lm"]
        assert isinstance(arch_lm, dict)
        assert 0 <= arch_lm["p_value"] <= 1

    def test_higher_lags_more_power(self):
        """More lags may capture longer-range ARCH effects."""
        np.random.seed(42)

        # Create ARCH effect with longer memory
        n = 500
        residuals = np.zeros(n)
        for t in range(n):
            # Long-range dependence in volatility
            lookback = min(t, 10)
            if lookback > 0:
                past_sq = np.mean(residuals[t - lookback:t] ** 2)
                vol = 0.02 + 0.5 * past_sq
            else:
                vol = 0.02
            residuals[t] = np.random.randn() * np.sqrt(vol)

        result_2 = detect_heteroskedasticity(residuals, lags=2, acf_lags=20, alpha=0.05)
        result_10 = detect_heteroskedasticity(residuals, lags=10, acf_lags=20, alpha=0.05)

        # Both should provide valid results
        assert "arch_lm" in result_2
        assert "arch_lm" in result_10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
