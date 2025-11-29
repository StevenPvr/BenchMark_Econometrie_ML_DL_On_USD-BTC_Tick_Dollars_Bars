from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
# Find project root by looking for .git, pyproject.toml, or setup.py
_project_root = _script_dir
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from unittest.mock import patch, MagicMock
from src.analyse_features.temporal import (
    compute_acf_single,
    compute_acf_all_features,
    compute_rolling_target_correlation,
    compute_temporal_stability,
    compute_correlation_over_time,
    run_temporal_analysis
)
from src.analyse_features.config import TARGET_COLUMN

class TestTemporal:

    def test_compute_acf_single(self):
        # Sine wave -> periodic ACF
        t = np.linspace(0, 10*np.pi, 100)
        series = np.sin(t)

        result = compute_acf_single(series, nlags=10)

        assert "acf" in result
        assert "pacf" in result
        assert len(result["acf"]) == 11 # 0 to 10
        # Lag 0 is 1
        assert result["acf"][0] == 1.0

    def test_compute_acf_all_features(self, non_stationary_df):
        cols = ["stationary", "non_stationary"]
        result = compute_acf_all_features(non_stationary_df, cols, nlags=10, n_jobs=1)

        assert isinstance(result, pd.DataFrame)
        assert "persistence" in result.columns

        # Non-stationary should have higher persistence
        row_stat = result[result["feature"] == "stationary"].iloc[0]
        row_nonstat = result[result["feature"] == "non_stationary"].iloc[0]

        assert row_nonstat["persistence"] > row_stat["persistence"]

    def test_compute_rolling_target_correlation(self):
        # Create data where correlation flips halfway
        n = 200
        x = np.random.normal(0, 1, n)
        y = np.concatenate([x[:100], -x[100:]]) # Pos corr then Neg corr

        df = pd.DataFrame({"x": x, TARGET_COLUMN: y})

        # Window smaller than flip
        window = 50
        result = compute_rolling_target_correlation(df, "x", TARGET_COLUMN, windows=[window])

        col_name = f"corr_w{window}"
        assert col_name in result.columns

        # Check start (pos corr)
        assert np.nanmean(result[col_name].iloc[window:90]) > 0.8
        # Check end (neg corr)
        # Relaxed threshold due to potential smoothing/noise
        assert np.nanmean(result[col_name].iloc[110:]) < -0.6

    def test_compute_temporal_stability(self):
        # Stable vs Unstable
        n = 1000
        stable = np.random.normal(0, 1, n)
        unstable = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(10, 5, 500)])

        df = pd.DataFrame({"stable": stable, "unstable": unstable})

        result = compute_temporal_stability(df, ["stable", "unstable"], n_periods=10)

        row_stable = result[result["feature"] == "stable"].iloc[0]
        row_unstable = result[result["feature"] == "unstable"].iloc[0]

        assert row_stable["overall_stability"] > row_unstable["overall_stability"]

    def test_compute_correlation_over_time(self):
        # Consistent vs Inconsistent
        n = 100
        y = np.random.normal(0, 1, n)
        consistent = y + np.random.normal(0, 0.1, n)
        inconsistent = np.concatenate([y[:50], np.random.normal(0, 1, 50)]) # Correlated then random

        df = pd.DataFrame({"consistent": consistent, "inconsistent": inconsistent, TARGET_COLUMN: y})

        result = compute_correlation_over_time(df, ["consistent", "inconsistent"], TARGET_COLUMN, n_periods=2)

        row_const = result[result["feature"] == "consistent"].iloc[0]
        row_inconst = result[result["feature"] == "inconsistent"].iloc[0]

        assert row_const["corr_consistency"] > row_inconst["corr_consistency"]

    @patch("src.analyse_features.temporal.save_json")
    @patch("src.analyse_features.temporal.ensure_directories")
    def test_run_temporal_analysis(self, mock_ensure, mock_save, non_stationary_df):
        results = run_temporal_analysis(non_stationary_df, target_column=TARGET_COLUMN, save_results=True)

        assert "acf_summary" in results
        assert "stability" in results
        assert "rolling_correlations" in results

        mock_ensure.assert_called()
        mock_save.assert_called()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
