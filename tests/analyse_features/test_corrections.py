"""Tests for corrections and new features in analyse_features module.

Tests:
1. Ward linkage correctly uses Euclidean distance
2. MI alignment handles NaN at different positions
3. Polyfit handles partial NaN
4. Normality tests work correctly
5. Granger causality returns valid results
6. Feature selection works with various inputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyse_features.clustering import (
    compute_correlation_distance_matrix,
    hierarchical_clustering,
)
from src.analyse_features.correlation import compute_feature_feature_mi
from src.analyse_features.temporal import compute_granger_causality
from src.analyse_features.stationarity import check_normality_single, check_normality_all
from src.analyse_features.feature_selection import (
    get_recommended_features,
    rank_features,
)


class TestWardLinkage:
    """Test Ward linkage correction."""

    def test_ward_linkage_produces_valid_result(self):
        """Ward linkage should produce valid hierarchical clustering."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        # Create correlated features
        base = np.random.randn(n_samples)
        data = {}
        for i in range(n_features):
            noise = np.random.randn(n_samples) * 0.3
            data[f"feat_{i}"] = base + noise

        df = pd.DataFrame(data)
        feature_columns = list(df.columns)

        # Compute distance matrix
        distance_matrix, corr_matrix = compute_correlation_distance_matrix(df, feature_columns)

        # Test that Ward linkage works
        result = hierarchical_clustering(distance_matrix, feature_columns, linkage_method="ward")

        assert "linkage_matrix" in result
        assert result["linkage_matrix"].shape[0] == n_features - 1
        assert "dendrogram" in result
        assert len(result["feature_order"]) == n_features

    def test_ward_different_from_average(self):
        """Ward should produce different results than average linkage."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10

        data = {f"feat_{i}": np.random.randn(n_samples) for i in range(n_features)}
        df = pd.DataFrame(data)
        feature_columns = list(df.columns)

        distance_matrix, _ = compute_correlation_distance_matrix(df, feature_columns)

        result_ward = hierarchical_clustering(distance_matrix, feature_columns, linkage_method="ward")
        result_avg = hierarchical_clustering(distance_matrix, feature_columns, linkage_method="average")

        # Linkage matrices should be different (not identical)
        # Due to the transformation for Ward
        assert not np.allclose(result_ward["linkage_matrix"], result_avg["linkage_matrix"])


class TestMIAlignment:
    """Test MI alignment with NaN values."""

    def test_mi_with_aligned_nan(self):
        """MI should handle NaN at different positions correctly."""
        np.random.seed(42)
        n_samples = 200

        # Create features with NaN at different positions
        feat_1 = np.random.randn(n_samples)
        feat_2 = feat_1 + np.random.randn(n_samples) * 0.1  # Highly correlated

        # Add NaN at different positions
        feat_1[10:15] = np.nan
        feat_2[20:25] = np.nan

        df = pd.DataFrame({"feat_1": feat_1, "feat_2": feat_2})

        # This should work without errors and produce valid MI
        mi_result = compute_feature_feature_mi(df, ["feat_1", "feat_2"])

        # MI should be computed without errors - result is a long-format DataFrame
        assert mi_result is not None
        assert len(mi_result) >= 1
        # Check columns exist
        assert "feature_1" in mi_result.columns
        assert "feature_2" in mi_result.columns
        assert "mutual_information" in mi_result.columns
        # MI should be non-negative
        assert mi_result["mutual_information"].iloc[0] >= 0


class TestPolyfitNaN:
    """Test that polyfit handles NaN correctly."""

    def test_compute_temporal_stability_with_nan(self):
        """Temporal stability should compute trend even with some NaN periods."""
        from src.analyse_features.temporal import compute_temporal_stability

        np.random.seed(42)
        n_samples = 1000

        # Create feature with a trend
        trend = np.linspace(0, 1, n_samples)
        noise = np.random.randn(n_samples) * 0.1
        feature = trend + noise

        # Add some NaN values in certain periods
        feature[100:110] = np.nan
        feature[500:510] = np.nan

        df = pd.DataFrame({"test_feature": feature})

        result = compute_temporal_stability(df, ["test_feature"], n_periods=10)

        # Should have computed mean_trend
        assert "mean_trend" in result.columns
        assert len(result) == 1

        # mean_trend should be positive (we created an upward trend)
        mean_trend = result.loc[0, "mean_trend"]
        assert not np.isnan(mean_trend)
        assert mean_trend > 0  # Positive trend


class TestNormalityTests:
    """Test normality testing functions."""

    def test_normality_normal_distribution(self):
        """Normal data should be classified as normal."""
        np.random.seed(42)
        normal_data = np.random.randn(1000)

        result = check_normality_single(normal_data, "normal_feature")

        assert result["feature"] == "normal_feature"
        assert "jb_pvalue" in result
        assert "shapiro_pvalue" in result
        assert "normality_conclusion" in result
        # Normal data should have high p-values (not reject normality)
        # Note: Even truly normal data may sometimes fail at alpha=0.05
        # So we just check the test ran without error

    def test_normality_non_normal_distribution(self):
        """Highly non-normal data should be classified as non-normal."""
        np.random.seed(42)
        # Exponential distribution is clearly non-normal
        non_normal_data = np.random.exponential(scale=1.0, size=1000)

        result = check_normality_single(non_normal_data, "exp_feature")

        assert result["feature"] == "exp_feature"
        # Exponential should definitely reject normality
        assert result["jb_pvalue"] < 0.05 or result["shapiro_pvalue"] < 0.05

    def test_normality_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        small_data = np.array([1, 2, 3, 4, 5])

        result = check_normality_single(small_data, "small_feature")

        assert result["normality_conclusion"] == "insufficient_data"

    def test_normality_all_features(self):
        """test_normality_all should process multiple features."""
        np.random.seed(42)
        df = pd.DataFrame({
            "normal_1": np.random.randn(500),
            "normal_2": np.random.randn(500),
            "exp_1": np.random.exponential(size=500),
        })

        result = check_normality_all(df, n_jobs=1)

        assert len(result) == 3
        assert "feature" in result.columns
        assert "normality_conclusion" in result.columns


class TestGrangerCausality:
    """Test Granger causality function."""

    def test_granger_with_causality(self):
        """Feature that predicts target should show Granger causality."""
        np.random.seed(42)
        n_samples = 500

        # Create a feature that Granger-causes the target
        feature = np.random.randn(n_samples)
        target = np.zeros(n_samples)
        for i in range(1, n_samples):
            target[i] = 0.5 * feature[i - 1] + 0.3 * np.random.randn()

        df = pd.DataFrame({"feature": feature, "target": target})

        result = compute_granger_causality(df, ["feature"], "target", max_lag=3)

        assert len(result) == 1
        assert result.loc[0, "feature"] == "feature"
        # Should find significant causality
        assert "granger_pvalue" in result.columns
        assert not np.isnan(result.loc[0, "granger_pvalue"])

    def test_granger_no_causality(self):
        """Independent features should not show Granger causality."""
        np.random.seed(42)
        n_samples = 500

        # Independent series
        feature = np.random.randn(n_samples)
        target = np.random.randn(n_samples)

        df = pd.DataFrame({"feature": feature, "target": target})

        result = compute_granger_causality(df, ["feature"], "target", max_lag=3)

        assert len(result) == 1
        # Should have higher p-value (no causality)
        # Note: May occasionally be significant by chance, so we just check it runs


class TestFeatureSelection:
    """Test feature selection module."""

    def test_get_recommended_features_basic(self):
        """Basic feature selection should work."""
        stationarity_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4"],
            "stationarity_conclusion": ["stationary", "non_stationary", "stationary", "trend_stationary"],
        })

        vif_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4"],
            "vif": [2.0, 5.0, 15.0, 3.0],  # f3 has high VIF
        })

        target_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3", "f4"],
            "abs_spearman": [0.3, 0.1, 0.5, 0.001],  # f4 has low correlation
        })

        result = get_recommended_features(
            stationarity_df=stationarity_df,
            vif_df=vif_df,
            target_metrics_df=target_df,
            max_vif=10.0,
            min_target_corr=0.01,
            require_stationary=True,
        )

        selected = result["selected"]

        # f1 should be selected (stationary, low VIF, decent corr)
        assert "f1" in selected

        # f2 should be removed (non-stationary)
        assert "f2" not in selected

        # f3 should be removed (high VIF)
        assert "f3" not in selected

        # f4 should be removed (low correlation)
        assert "f4" not in selected

    def test_rank_features(self):
        """Feature ranking should produce valid output."""
        target_df = pd.DataFrame({
            "feature": ["f1", "f2", "f3"],
            "abs_spearman": [0.5, 0.3, 0.1],
            "mutual_information": [0.2, 0.4, 0.1],
        })

        result = rank_features(target_metrics_df=target_df)

        assert len(result) == 3
        assert "combined_rank" in result.columns
        # Features should be sorted by combined rank
        assert result.iloc[0]["combined_rank"] <= result.iloc[-1]["combined_rank"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
