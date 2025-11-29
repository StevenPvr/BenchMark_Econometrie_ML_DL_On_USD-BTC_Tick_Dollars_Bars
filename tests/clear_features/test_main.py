from __future__ import annotations

import sys
from pathlib import Path
import json

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
from unittest.mock import MagicMock, patch
from src.clear_features.main import (
    clean_nan_values,
    _to_scalar_int,
    load_dataset,
    save_dataset,
    run_full_pipeline
)

def test_to_scalar_int():
    assert _to_scalar_int(5) == 5
    assert _to_scalar_int(5.7) == 5
    assert _to_scalar_int(np.int64(10)) == 10
    assert _to_scalar_int(np.float64(10.5)) == 10
    assert _to_scalar_int(pd.Series([1, 2])) == 3 # Sums series
    assert _to_scalar_int(np.array([1, 2])) == 3

def test_clean_nan_values():
    df = pd.DataFrame({
        "feat_1": [1.0, np.nan, 3.0],
        "target": [1.0, 1.0, np.nan], # Last row should be removed
        "meta": [1, 2, 3]
    })

    cleaned, stats = clean_nan_values(df, meta_columns=["meta"], target_column="target")

    # Check median fill for feat_1 (median of 1, 3 is 2)
    assert cleaned["feat_1"].iloc[1] == 2.0

    # Check row removal
    assert len(cleaned) == 2
    assert 2 not in cleaned.index # Index 2 removed

    assert stats["rows_removed"] == 1
    assert "feat_1" in stats["nan_cols"]

def test_load_save_dataset(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.parquet"

    save_dataset(df, path)
    assert path.exists()

    loaded = load_dataset(path)
    pd.testing.assert_frame_equal(df, loaded)


@patch("src.clear_features.main.load_dataset")
@patch("src.clear_features.main.save_dataset")
@patch("src.clear_features.main.GroupPCAReducer")
@patch("src.clear_features.main.ScalerApplier")
def test_run_full_pipeline(
    mock_scaler, mock_pca,
    mock_save, mock_load, tmp_path
):
    # Setup mocks
    mock_df = pd.DataFrame({
        "feat1": np.random.rand(10),
        "split": ["train"] * 10,
        "log_return": np.random.rand(10)
    })
    mock_load.return_value = mock_df

    # Mock Scaler (applied BEFORE PCA now)
    mock_scaler_instance = mock_scaler.return_value
    mock_scaler_instance.apply_zscore.return_value = mock_df
    mock_scaler_instance.apply_minmax.return_value = mock_df

    # Mock PCA
    mock_pca_instance = mock_pca.return_value
    mock_pca_summary = mock_pca_instance.fit.return_value
    mock_pca_summary.original_n_features = 10
    mock_pca_summary.final_n_features = 5
    mock_pca_summary.groups_processed = []
    mock_pca_summary.groups_skipped = []
    mock_pca_summary.features_removed = []
    mock_pca_summary.features_added = []
    mock_pca_instance.transform.return_value = mock_df

    # Run pipeline
    results = run_full_pipeline(dry_run=True)

    assert results is not None
    assert "steps" in results
    assert "datasets" in results

    # Verify method calls - scalers loaded first
    assert mock_scaler_instance.load_scalers.called
    # PCA fit called after scaling
    assert mock_pca_instance.fit.called

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
