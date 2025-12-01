from __future__ import annotations

import sys
from pathlib import Path
import json
import tempfile

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
from src.clear_features.pca_reducer import GroupPCAReducer, GroupPCAResult, PCAReductionSummary


@pytest.fixture
def feature_categories_file(tmp_path):
    """Create a temporary feature_categories.json for testing."""
    categories = {
        "total_columns": 10,
        "groups": [
            {
                "name": "momentum_lag1",
                "category": "momentum",
                "lag": "lag1",
                "description": "Momentum features",
                "count": 3,
                "features": ["feat_a", "feat_b", "feat_c"]
            },
            {
                "name": "volatility_lag5",
                "category": "volatility",
                "lag": "lag5",
                "description": "Volatility features",
                "count": 2,
                "features": ["vol_1", "vol_2"]
            },
            {
                "name": "singular_feature",
                "category": "singular",
                "lag": "no_lag",
                "description": "Single feature",
                "count": 1,
                "features": ["single_feat"]
            },
            {
                "name": "pca_clusters_no_lag",
                "category": "pca_clusters",
                "lag": "no_lag",
                "description": "Old PCA features to remove",
                "count": 2,
                "features": ["pca_cluster1_c0", "pca_cluster2_c0"]
            }
        ]
    }

    file_path = tmp_path / "feature_categories.json"
    with open(file_path, "w") as f:
        json.dump(categories, f)

    return file_path


@pytest.fixture
def sample_df():
    """Create sample DataFrame with train/test split."""
    np.random.seed(42)
    n_train = 80
    n_test = 20
    n_total = n_train + n_test

    # Create correlated features for momentum group
    base_momentum = np.random.randn(n_total)
    feat_a = base_momentum + np.random.randn(n_total) * 0.1
    feat_b = base_momentum + np.random.randn(n_total) * 0.2
    feat_c = base_momentum + np.random.randn(n_total) * 0.15

    # Create correlated features for volatility group
    base_vol = np.abs(np.random.randn(n_total))
    vol_1 = base_vol + np.random.randn(n_total) * 0.1
    vol_2 = base_vol + np.random.randn(n_total) * 0.15

    # Single feature
    single_feat = np.random.randn(n_total)

    # Old PCA cluster features (to be removed)
    pca_cluster1_c0 = np.random.randn(n_total)
    pca_cluster2_c0 = np.random.randn(n_total)

    # Independent feature (not in any group)
    independent = np.random.randn(n_total)

    # Target
    log_return = np.random.randn(n_total) * 0.01

    # Split column
    split = ["train"] * n_train + ["test"] * n_test

    df = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
        "vol_1": vol_1,
        "vol_2": vol_2,
        "single_feat": single_feat,
        "pca_cluster1_c0": pca_cluster1_c0,
        "pca_cluster2_c0": pca_cluster2_c0,
        "independent": independent,
        "log_return": log_return,
        "split": split,
    })

    return df


def test_fit_on_train_only(sample_df, feature_categories_file):
    """Test that PCA is fitted on training data only."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    summary = reducer.fit(sample_df)

    # Should have processed momentum and volatility groups
    assert len(summary.groups_processed) == 2

    # Singular and pca_clusters should be skipped
    assert "singular_feature" in summary.groups_skipped
    assert "pca_clusters_no_lag" in summary.groups_skipped


def test_transform_replaces_features(sample_df, feature_categories_file):
    """Test that transform replaces original features with PCA components."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)
    transformed = reducer.transform(sample_df)

    # Original features should be removed
    assert "feat_a" not in transformed.columns
    assert "feat_b" not in transformed.columns
    assert "feat_c" not in transformed.columns
    assert "vol_1" not in transformed.columns
    assert "vol_2" not in transformed.columns

    # Old pca_cluster features should be removed
    assert "pca_cluster1_c0" not in transformed.columns
    assert "pca_cluster2_c0" not in transformed.columns

    # New PCA components should exist
    assert any(c.startswith("pca_momentum_lag1_") for c in transformed.columns)
    assert any(c.startswith("pca_volatility_lag5_") for c in transformed.columns)

    # Single feature should be kept (not in any processed group)
    assert "single_feat" in transformed.columns

    # Independent feature should be kept
    assert "independent" in transformed.columns

    # Metadata columns preserved
    assert "split" in transformed.columns
    assert "log_return" in transformed.columns


def test_90_percent_variance_threshold(sample_df, feature_categories_file):
    """Test that PCA keeps components explaining 90% variance."""
    config = {"variance_explained_threshold": 0.90}
    reducer = GroupPCAReducer(categories_file=feature_categories_file, config=config)
    summary = reducer.fit(sample_df)

    for group_result in summary.groups_processed:
        # Cumulative variance of retained components should be >= 90%
        assert group_result.cumulative_variance[-1] >= 0.90


def test_fit_transform(sample_df, feature_categories_file):
    """Test fit_transform convenience method."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    transformed, summary = reducer.fit_transform(sample_df)

    assert isinstance(transformed, pd.DataFrame)
    assert isinstance(summary, PCAReductionSummary)
    assert len(transformed) == len(sample_df)


def test_save_load_artifacts(sample_df, feature_categories_file, tmp_path):
    """Test saving and loading PCA artifacts."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)

    # Save
    artifacts_dir = tmp_path / "pca_artifacts"
    reducer.save_artifacts(output_dir=artifacts_dir)

    assert (artifacts_dir / "pca_models.joblib").exists()
    assert (artifacts_dir / "pca_summary.json").exists()

    # Load into new reducer
    new_reducer = GroupPCAReducer(categories_file=feature_categories_file)
    new_reducer.load_artifacts(input_dir=artifacts_dir)

    # Transform should work identically
    original_transformed = reducer.transform(sample_df)
    loaded_transformed = new_reducer.transform(sample_df)

    # Check same columns
    assert set(original_transformed.columns) == set(loaded_transformed.columns)


def test_no_data_leakage(sample_df, feature_categories_file):
    """Test that test data doesn't influence PCA fitting."""
    # Modify test data significantly
    sample_df_modified = sample_df.copy()
    test_mask = sample_df_modified["split"] == "test"
    sample_df_modified.loc[test_mask, "feat_a"] = sample_df_modified.loc[test_mask, "feat_a"] * 1000

    reducer1 = GroupPCAReducer(categories_file=feature_categories_file)
    reducer2 = GroupPCAReducer(categories_file=feature_categories_file)

    # Fit on original and modified dataframes
    summary1 = reducer1.fit(sample_df)
    summary2 = reducer2.fit(sample_df_modified)

    # PCA should be identical since only train data is used
    for g1, g2 in zip(summary1.groups_processed, summary2.groups_processed):
        assert g1.n_components == g2.n_components
        np.testing.assert_array_almost_equal(
            g1.explained_variance_ratio,
            g2.explained_variance_ratio
        )


def test_nan_handling(sample_df, feature_categories_file):
    """Test that NaN values are handled properly."""
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "feat_a"] = np.nan
    df_with_nan.loc[5, "vol_1"] = np.nan

    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(df_with_nan)
    transformed = reducer.transform(df_with_nan)

    # Should not have any NaN in PCA components (median imputation)
    pca_cols = [c for c in transformed.columns if c.startswith("pca_")]
    for col in pca_cols:
        assert not transformed[col].isna().any()


def test_missing_features(sample_df, feature_categories_file):
    """Test handling when some features from config are missing in dataframe."""
    # Remove one feature
    df_missing = sample_df.drop(columns=["feat_c"])

    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(df_missing)

    # Should still process momentum group with 2 features
    momentum_result = next(
        g for g in reducer._summary.groups_processed
        if g.group_name == "momentum_lag1"
    )
    assert len(momentum_result.original_features) == 2
    assert "feat_c" not in momentum_result.original_features


def test_requires_split_column(sample_df, feature_categories_file):
    """Test that fit raises error without split column."""
    df_no_split = sample_df.drop(columns=["split"])

    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    with pytest.raises(ValueError, match="split"):
        reducer.fit(df_no_split)


def test_transform_before_fit_raises(sample_df, feature_categories_file):
    """Test that transform raises error if fit wasn't called."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    with pytest.raises(RuntimeError, match="fit"):
        reducer.transform(sample_df)


def test_standardization_before_pca(sample_df, feature_categories_file):
    """Test that data is standardized before PCA fitting."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)

    # Check standardization params exist
    assert "momentum_lag1" in reducer._standardization_params
    assert "volatility_lag5" in reducer._standardization_params

    # Check params structure
    for group_name, params in reducer._standardization_params.items():
        assert "mean" in params
        assert "std" in params
        # Std should never be zero
        assert all(params["std"] > 0)


def test_standardization_params_from_train_only(sample_df, feature_categories_file):
    """Test that standardization uses train data only."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)

    # Get train data
    train_data = sample_df[sample_df["split"] == "train"]
    momentum_cols = ["feat_a", "feat_b", "feat_c"]

    # Compute expected params from train only
    expected_means = train_data[momentum_cols].mean().values
    expected_stds = train_data[momentum_cols].std(ddof=1).values

    # Compare with stored params
    stored_means = reducer._standardization_params["momentum_lag1"]["mean"]
    stored_stds = reducer._standardization_params["momentum_lag1"]["std"]

    np.testing.assert_array_almost_equal(expected_means, stored_means)
    np.testing.assert_array_almost_equal(expected_stds, stored_stds)


def test_standardization_handles_constant_features(feature_categories_file):
    """Test that constant features don't cause division by zero."""
    np.random.seed(42)
    n_train = 80
    n_test = 20
    n_total = n_train + n_test

    df = pd.DataFrame({
        "feat_a": np.ones(n_total),  # Constant feature
        "feat_b": np.random.randn(n_total),
        "feat_c": np.random.randn(n_total),
        "vol_1": np.random.randn(n_total),
        "vol_2": np.random.randn(n_total),
        "single_feat": np.random.randn(n_total),
        "pca_cluster1_c0": np.random.randn(n_total),
        "pca_cluster2_c0": np.random.randn(n_total),
        "independent": np.random.randn(n_total),
        "log_return": np.random.randn(n_total),
        "split": ["train"] * n_train + ["test"] * n_test,
    })

    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    # Should not raise division by zero error
    reducer.fit(df)

    # Std for constant feature should be 1.0 (safety fallback)
    assert reducer._standardization_params["momentum_lag1"]["std"][0] == 1.0


def test_transform_raises_if_standardization_params_missing(sample_df, feature_categories_file):
    """Test that transform raises error if standardization params missing."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)

    # Manually remove standardization params
    del reducer._standardization_params["momentum_lag1"]

    with pytest.raises(RuntimeError, match="standardization parameters"):
        reducer.transform(sample_df)


def test_transform_raises_if_medians_missing_with_nan(sample_df, feature_categories_file):
    """Test that transform raises error if medians missing and NaN present."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)

    # Manually remove medians
    del reducer._imputation_medians["momentum_lag1"]

    # Add NaN to trigger median lookup
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "feat_a"] = np.nan

    with pytest.raises(RuntimeError, match="No stored medians"):
        reducer.transform(df_with_nan)


def test_save_load_preserves_standardization_params(sample_df, feature_categories_file, tmp_path):
    """Test that save/load preserves standardization parameters."""
    reducer = GroupPCAReducer(categories_file=feature_categories_file)
    reducer.fit(sample_df)

    artifacts_dir = tmp_path / "artifacts"
    reducer.save_artifacts(output_dir=artifacts_dir)

    # Load into new reducer
    new_reducer = GroupPCAReducer(categories_file=feature_categories_file)
    new_reducer.load_artifacts(input_dir=artifacts_dir)

    # Verify standardization params preserved
    for group_name in reducer._standardization_params:
        np.testing.assert_array_equal(
            reducer._standardization_params[group_name]["mean"],
            new_reducer._standardization_params[group_name]["mean"]
        )
        np.testing.assert_array_equal(
            reducer._standardization_params[group_name]["std"],
            new_reducer._standardization_params[group_name]["std"]
        )


if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
