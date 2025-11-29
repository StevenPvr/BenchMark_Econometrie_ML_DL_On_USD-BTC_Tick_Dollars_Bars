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
from src.analyse_features.correlation import (
    compute_spearman_matrix,
    compute_mutual_information,
    compute_feature_feature_mi,
    run_correlation_analysis
)
from src.analyse_features.config import TARGET_COLUMN

class TestCorrelation:
    def test_compute_spearman_matrix(self, correlated_df):
        result = compute_spearman_matrix(correlated_df)

        assert isinstance(result, pd.DataFrame)
        assert TARGET_COLUMN not in result.columns
        assert "feat_a" in result.columns
        assert "feat_b" in result.columns

        corr_ab = result.loc["feat_a", "feat_b"]
        assert abs(corr_ab) > 0.8
        assert result.loc["feat_b", "feat_a"] == corr_ab
        assert result.loc["feat_a", "feat_a"] == pytest.approx(1.0)

    def test_compute_mutual_information(self, correlated_df):
        result = compute_mutual_information(correlated_df, target_column=TARGET_COLUMN)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "mutual_information" in result.columns

        expected_features = [c for c in correlated_df.columns if c != TARGET_COLUMN]
        assert set(result["feature"]) == set(expected_features)
        assert (result["mutual_information"] >= 0).all()

    @patch("src.analyse_features.correlation.save_json")
    @patch("src.analyse_features.correlation.plot_correlation_heatmap")
    @patch("src.analyse_features.correlation.ensure_directories")
    def test_run_correlation_analysis(self, mock_ensure, mock_plot, mock_save, correlated_df):
        results = run_correlation_analysis(correlated_df, save_results=True)

        assert isinstance(results, dict)
        assert "spearman" in results
        assert "mutual_information" in results

        mock_ensure.assert_called_once()
        mock_save.assert_called()
        mock_plot.assert_called()

    def test_compute_spearman_single_feature(self):
        df = pd.DataFrame({"feat_1": [1, 2, 3]})
        result = compute_spearman_matrix(df)
        assert result.shape == (1, 1)
        assert result.iloc[0, 0] == pytest.approx(1.0)

    def test_compute_mutual_information_missing_target(self, sample_df):
        with pytest.raises(ValueError):
            compute_mutual_information(sample_df, target_column="non_existent")

    def test_compute_feature_feature_mi(self, correlated_df):
        # Relaxed test: check structure and non-empty result
        # n_jobs=1 to avoid multiprocessing issues in test
        result = compute_feature_feature_mi(correlated_df, n_jobs=1, top_n_pairs=5)

        assert isinstance(result, pd.DataFrame)
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "mutual_information" in result.columns

        # We expect at least one pair (feat_a - feat_b) to have some MI
        assert not result.empty
        assert (result["mutual_information"] >= 0).all()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
