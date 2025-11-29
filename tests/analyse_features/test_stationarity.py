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
# Import module to avoid pytest collecting the imported functions as tests
from src.analyse_features import stationarity

class TestStationarity:

    @patch("src.analyse_features.stationarity.adfuller")
    def test_test_adf(self, mock_adfuller):
        # Mock adfuller return: (stat, pvalue, lags, nobs, crit, icbest)
        mock_adfuller.return_value = (-3.5, 0.01, 1, 100, {}, 100.0)

        series = np.random.normal(0, 1, 100)
        result = stationarity._test_adf(series)

        assert result["adf_pvalue"] == 0.01
        assert result["adf_reject_h0"] is True # 0.01 < 0.05

    @patch("src.analyse_features.stationarity.kpss")
    def test_test_kpss(self, mock_kpss):
        # Mock kpss return: (stat, pvalue, lags, crit)
        mock_kpss.return_value = (0.1, 0.1, 1, {})

        series = np.random.normal(0, 1, 100)
        result = stationarity._test_kpss(series)

        assert result["kpss_pvalue"] == 0.1
        assert result["kpss_reject_h0"] is False # 0.1 > 0.05

    @patch("src.analyse_features.stationarity._test_adf")
    @patch("src.analyse_features.stationarity._test_kpss")
    def test_test_stationarity_single_stationary(self, mock_kpss, mock_adf):
        # Stationary: ADF rejects (p < 0.05), KPSS fails to reject (p > 0.05)
        mock_adf.return_value = {"adf_reject_h0": True}
        mock_kpss.return_value = {"kpss_reject_h0": False}

        series = np.zeros(10)
        result = stationarity.test_stationarity_single(series, "feat1")

        assert result["stationarity_conclusion"] == "stationary"

    @patch("src.analyse_features.stationarity._test_adf")
    @patch("src.analyse_features.stationarity._test_kpss")
    def test_test_stationarity_single_non_stationary(self, mock_kpss, mock_adf):
        # Non-Stationary: ADF fails to reject (p > 0.05), KPSS rejects (p < 0.05)
        mock_adf.return_value = {"adf_reject_h0": False}
        mock_kpss.return_value = {"kpss_reject_h0": True}

        series = np.zeros(10)
        result = stationarity.test_stationarity_single(series, "feat1")

        assert result["stationarity_conclusion"] == "non_stationary"

    @patch("src.analyse_features.stationarity._process_feature_batch")
    @patch("src.analyse_features.stationarity._load_cache")
    @patch("src.analyse_features.stationarity._save_cache")
    @patch("src.analyse_features.stationarity._clear_cache")
    def test_test_stationarity_all(self, mock_clear, mock_save, mock_load, mock_batch, sample_df):
        # Mock batch processing to return dummy results
        mock_load.return_value = {}

        def side_effect(df, cols, n_jobs):
            return [{"feature": c, "stationarity_conclusion": "stationary"} for c in cols]

        mock_batch.side_effect = side_effect

        result_df = stationarity.test_stationarity_all(sample_df, n_jobs=1)

        assert len(result_df) >= 3
        assert all(result_df["stationarity_conclusion"] == "stationary")

        # Ensure cache handling called
        mock_load.assert_called()
        mock_save.assert_called()

    @patch("src.analyse_features.stationarity.test_stationarity_all")
    @patch("src.analyse_features.stationarity.save_json")
    @patch("src.analyse_features.stationarity.plot_stationarity_summary")
    @patch("src.analyse_features.stationarity.ensure_directories")
    def test_run_stationarity_analysis(self, mock_ensure, mock_plot, mock_save, mock_test_all, sample_df):
        # Mock test_stationarity_all return
        mock_test_all.return_value = pd.DataFrame([
            {"feature": "f1", "stationarity_conclusion": "stationary"},
            {"feature": "f2", "stationarity_conclusion": "non_stationary"}
        ])

        results = stationarity.run_stationarity_analysis(sample_df)

        assert isinstance(results, pd.DataFrame)
        mock_ensure.assert_called()
        mock_save.assert_called()
        mock_plot.assert_called()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
