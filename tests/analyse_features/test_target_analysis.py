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
from src.analyse_features.target_analysis import (
    compute_target_correlations,
    compute_target_mutual_information,
    compute_f_scores,
    compute_combined_target_metrics,
    analyze_nonlinearity,
    run_target_analysis
)
from src.analyse_features.config import TARGET_COLUMN

class TestTargetAnalysis:

    def test_compute_target_correlations(self, correlated_df):
        # correlated_df: target is correlated with feat_a (0.5 + noise)
        result = compute_target_correlations(correlated_df, target_column=TARGET_COLUMN)

        assert isinstance(result, pd.DataFrame)
        assert "pearson_corr" in result.columns
        assert "spearman_corr" in result.columns

        # Check feat_a correlation
        row_a = result[result["feature"] == "feat_a"].iloc[0]
        assert abs(row_a["pearson_corr"]) > 0.3

    def test_compute_target_mutual_information(self, correlated_df):
        result = compute_target_mutual_information(correlated_df, target_column=TARGET_COLUMN)

        assert "mutual_information" in result.columns
        # MI should be positive
        assert (result["mutual_information"] >= 0).all()

    def test_compute_f_scores(self, correlated_df):
        result = compute_f_scores(correlated_df, target_column=TARGET_COLUMN)

        assert "f_score" in result.columns
        assert "f_pvalue" in result.columns

    def test_compute_combined_target_metrics(self, correlated_df):
        result = compute_combined_target_metrics(correlated_df, target_column=TARGET_COLUMN, n_jobs=1)

        # Should have all columns
        expected_cols = [
            "feature", "pearson_corr", "spearman_corr",
            "mutual_information", "f_score", "avg_rank"
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_analyze_nonlinearity(self):
        # Create non-linear relationship: y = x^2
        x = np.random.uniform(-1, 1, 100)
        y = x**2 + np.random.normal(0, 0.01, 100)
        df = pd.DataFrame({"x": x, TARGET_COLUMN: y})

        result = analyze_nonlinearity(df, target_column=TARGET_COLUMN)

        assert "nonlinearity_score" in result.columns
        score = result.iloc[0]["nonlinearity_score"]
        assert score > 0

    @patch("src.analyse_features.target_analysis.save_json")
    @patch("src.analyse_features.target_analysis.plot_target_correlations")
    @patch("src.analyse_features.target_analysis.ensure_directories")
    def test_run_target_analysis(self, mock_ensure, mock_plot, mock_save, correlated_df):
        results = run_target_analysis(correlated_df, target_column=TARGET_COLUMN, save_results=True)

        assert "combined_metrics" in results
        assert "nonlinearity" in results
        assert "summary" in results

        mock_ensure.assert_called()
        mock_save.assert_called()
        mock_plot.assert_called()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
