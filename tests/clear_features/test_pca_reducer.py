import pytest
import pandas as pd
import numpy as np
from src.clear_features.pca_reducer import WeightedPCAReducer, ClusterPCAResult

@pytest.fixture
def pca_df():
    np.random.seed(42)
    n = 100
    # Cluster 1: Correlated for PCA
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.1
    x3 = x1 + np.random.randn(n) * 0.1

    # Cluster 2: Perfect correlation (Identical)
    # To trigger the "weighted_average" path, the code checks for low variance
    # So we use constant features
    y1 = np.ones(n)
    y2 = np.ones(n)

    # Independent
    z1 = np.random.randn(n)

    df = pd.DataFrame({
        "c1_1": x1, "c1_2": x2, "c1_3": x3,
        "c2_1": y1, "c2_2": y2,
        "indep": z1
    })
    return df

@pytest.fixture
def clusters_config():
    return {
        "1": ["c1_1", "c1_2", "c1_3"],
        "2": ["c2_1", "c2_2"]
    }

def test_fit_pca_reducer(pca_df, clusters_config):
    reducer = WeightedPCAReducer(clusters=clusters_config)
    summary = reducer.fit(pca_df)

    assert summary.original_n_features == 6
    assert summary.final_n_features < 6

    # Check that cluster 1 was processed via PCA
    c1_result = next(c for c in summary.clusters_processed if c.cluster_id == "1")
    assert c1_result.reduction_type == "pca"
    assert c1_result.n_components >= 1

    # Check that cluster 2 was processed via weighted average (perfect correlation)
    c2_result = next(c for c in summary.clusters_processed if c.cluster_id == "2")
    assert c2_result.reduction_type == "weighted_average"
    assert c2_result.n_components == 1

def test_transform_pca_reducer(pca_df, clusters_config):
    reducer = WeightedPCAReducer(clusters=clusters_config)
    reducer.fit(pca_df)
    transformed = reducer.transform(pca_df)

    # Original columns should be gone
    for col in ["c1_1", "c1_2", "c1_3", "c2_1", "c2_2"]:
        assert col not in transformed.columns

    # New columns should exist
    assert any(c.startswith("pca_cluster1_") for c in transformed.columns)
    assert "avg_cluster2" in transformed.columns
    assert "indep" in transformed.columns

def test_fit_transform(pca_df, clusters_config):
    reducer = WeightedPCAReducer(clusters=clusters_config)
    transformed, summary = reducer.fit_transform(pca_df)

    assert isinstance(transformed, pd.DataFrame)
    assert summary is not None

def test_save_load_artifacts(pca_df, clusters_config, tmp_path):
    reducer = WeightedPCAReducer(clusters=clusters_config)
    reducer.fit(pca_df)

    # Save
    reducer.save_artifacts(output_dir=tmp_path)

    assert (tmp_path / "pca_models.joblib").exists()
    assert (tmp_path / "pca_summary.json").exists()

    # Load
    new_reducer = WeightedPCAReducer()
    new_reducer.load_artifacts(input_dir=tmp_path)

    # Check if loaded state matches
    assert new_reducer._summary is not None
    assert len(new_reducer._pca_models) == 1 # Cluster 1
    assert "2" in new_reducer._avg_clusters # Cluster 2

def test_missing_features_in_df(pca_df, clusters_config):
    # Remove a feature from DF that is in config
    df_missing = pca_df.drop(columns=["c1_3"])

    reducer = WeightedPCAReducer(clusters=clusters_config)
    reducer.fit(df_missing)

    # Should still work, just using available features
    c1_result = next(c for c in reducer._summary.clusters_processed if c.cluster_id == "1")
    assert "c1_3" not in c1_result.original_features
    assert "c1_1" in c1_result.original_features

def test_nan_handling(pca_df, clusters_config):
    # Introduce NaNs
    df_nan = pca_df.copy()
    df_nan.loc[0, "c1_1"] = np.nan

    reducer = WeightedPCAReducer(clusters=clusters_config)
    # Should not raise error (uses mean imputation)
    reducer.fit(df_nan)
    transformed = reducer.transform(df_nan)
    assert not transformed.isna().all().all() # Check we don't have all NaNs
