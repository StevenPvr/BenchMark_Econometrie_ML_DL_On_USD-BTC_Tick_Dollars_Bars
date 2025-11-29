import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.analyse_features.multicollinearity import (
    compute_vif_correlation_method,
    compute_vif_regression_method,
    compute_condition_number,
    identify_collinear_pairs,
    suggest_features_to_drop,
    run_multicollinearity_analysis
)

class TestMulticollinearity:

    def test_compute_vif_correlation_method(self, correlated_df):
        # correlated_df: feat_a and feat_b are highly correlated
        cols = ["feat_a", "feat_b", "feat_c"]
        result = compute_vif_correlation_method(correlated_df, cols)

        assert isinstance(result, pd.DataFrame)
        assert "vif" in result.columns

        # High VIF for correlated features
        vif_a = result[result["feature"] == "feat_a"]["vif"].values[0]
        vif_b = result[result["feature"] == "feat_b"]["vif"].values[0]
        vif_c = result[result["feature"] == "feat_c"]["vif"].values[0]

        assert vif_a > 5
        assert vif_b > 5
        assert vif_c < 5 # Uncorrelated

    def test_compute_vif_regression_method(self, correlated_df):
        # Test regression method (slower but same logic)
        cols = ["feat_a", "feat_b", "feat_c"]
        result = compute_vif_regression_method(correlated_df, cols, n_jobs=1)

        vif_a = result[result["feature"] == "feat_a"]["vif"].values[0]
        vif_b = result[result["feature"] == "feat_b"]["vif"].values[0]
        vif_c = result[result["feature"] == "feat_c"]["vif"].values[0]

        assert vif_a > 5
        assert vif_b > 5
        assert vif_c < 5

    def test_compute_condition_number(self, correlated_df):
        cols = ["feat_a", "feat_b", "feat_c"]
        result = compute_condition_number(correlated_df, cols)

        assert "condition_number" in result
        # High condition number due to correlation
        assert result["condition_number"] > 10

    def test_identify_collinear_pairs(self, correlated_df):
        cols = ["feat_a", "feat_b", "feat_c"]
        result = identify_collinear_pairs(correlated_df, cols, threshold=0.8)

        assert not result.empty
        # Should find feat_a - feat_b pair
        pair = result.iloc[0]
        features = {pair["feature_1"], pair["feature_2"]}
        assert "feat_a" in features
        assert "feat_b" in features
        assert pair["abs_correlation"] > 0.8

    def test_suggest_features_to_drop(self):
        vif_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "vif": [100.0, 90.0, 1.0]
        })
        corr_pairs = pd.DataFrame({
            "feature_1": ["f1"],
            "feature_2": ["f2"],
            "abs_correlation": [0.99]
        })

        # Should suggest dropping f1 (higher VIF in pair)
        to_drop = suggest_features_to_drop(vif_df, corr_pairs, vif_threshold=10)

        assert "f1" in to_drop
        # f2 might also be dropped if it exceeds threshold individually,
        # but logic prioritizes pairs. Here f2 is 90 > 10, so it is also added.
        assert "f2" in to_drop
        assert "f3" not in to_drop

    @patch("src.analyse_features.multicollinearity.save_json")
    @patch("src.analyse_features.multicollinearity.plot_vif_scores")
    @patch("src.analyse_features.multicollinearity.ensure_directories")
    def test_run_multicollinearity_analysis(self, mock_ensure, mock_plot, mock_save, correlated_df):
        results = run_multicollinearity_analysis(correlated_df, save_results=True)

        assert "vif" in results
        assert "condition_number" in results
        assert "collinear_pairs" in results
        assert "drop_suggestions" in results

        mock_ensure.assert_called()
        mock_save.assert_called()
        mock_plot.assert_called()
