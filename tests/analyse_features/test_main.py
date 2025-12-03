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
import sys
import pandas as pd  # type: ignore
from unittest.mock import patch, MagicMock
from src.analyse_features.main import main, run_all_analyses

class TestMain:

    @patch("src.analyse_features.main.run_correlation_analysis")
    @patch("src.analyse_features.main.run_stationarity_analysis")
    @patch("src.analyse_features.main.run_multicollinearity_analysis")
    @patch("src.analyse_features.main.run_target_analysis")
    @patch("src.analyse_features.main.run_clustering_analysis")
    @patch("src.analyse_features.main.run_temporal_analysis")
    @patch("src.analyse_features.main._save_full_summary")
    @patch("src.analyse_features.main.ensure_directories")
    def test_run_all_analyses(self, mock_ensure, mock_save, mock_temporal, mock_cluster,
                             mock_target, mock_multi, mock_stat, mock_corr, sample_df):

        feature_cols = ["feature_1", "feature_2"]

        # Setup mocks to return empty dicts/dfs as needed
        mock_corr.return_value = {"spearman": pd.DataFrame()}
        mock_stat.return_value = pd.DataFrame({"stationarity_conclusion": [], "feature": []})
        mock_multi.return_value = {"vif": pd.DataFrame({"vif": [], "feature": []})}
        mock_target.return_value = {"summary": {}}
        mock_cluster.return_value = {"clusters": pd.DataFrame({"cluster": [], "feature": []})}
        mock_temporal.return_value = {}

        results = run_all_analyses(sample_df, feature_cols, generate_plots=False)

        assert isinstance(results, dict)
        assert "correlation" in results
        assert "stationarity" in results

        mock_corr.assert_called()
        mock_stat.assert_called()
        mock_save.assert_called()

    @patch("src.analyse_features.main.pd.read_parquet")
    @patch("src.analyse_features.main.run_all_analyses")
    def test_main_cli_all(self, mock_run_all, mock_read, sample_df):
        mock_read.return_value = sample_df
        mock_run_all.return_value = {}

        test_args = ["main.py", "--analysis", "all", "--no-plots", "--input", "dummy.parquet"]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_all.assert_called_once()

    @patch("src.analyse_features.main.pd.read_parquet")
    @patch("src.analyse_features.main.run_correlation_analysis")
    def test_main_cli_specific(self, mock_run_corr, mock_read, sample_df):
        mock_read.return_value = sample_df
        mock_run_corr.return_value = {}

        test_args = ["main.py", "--analysis", "correlation", "--no-plots", "--input", "dummy.parquet"]
        with patch.object(sys, "argv", test_args):
            main()

        mock_run_corr.assert_called_once()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
