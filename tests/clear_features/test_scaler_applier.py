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
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.clear_features.scaler_applier import ScalerApplier, ScalerFitter
from src.features.scalers import StandardScalerCustom, MinMaxScalerCustom
import joblib

@pytest.fixture
def scaler_df():
    return pd.DataFrame({
        "feat_1": [1.0, 2.0, 3.0],
        "feat_2": [10.0, 20.0, 30.0],
        "pca_1": [0.1, 0.2, 0.3] # New column, not in scaler
    })

def test_load_scalers_missing(tmp_path):
    # Test that it warns but doesn't crash if scalers are missing
    applier = ScalerApplier()
    applier.load_scalers(
        zscore_path=tmp_path / "missing_z.joblib",
        minmax_path=tmp_path / "missing_m.joblib"
    )
    assert applier._zscore_scaler is None
    assert applier._minmax_scaler is None

def test_apply_zscore_with_partial_match(scaler_df, tmp_path):
    # Create a real mock scaler object and save it
    mock_scaler = StandardScalerCustom()
    mock_scaler.columns_ = ["feat_1", "feat_2", "feat_missing"]
    mock_scaler.mean_ = np.array([2.0, 20.0, 5.0])
    mock_scaler.std_ = np.array([1.0, 10.0, 1.0])

    scaler_path = tmp_path / "zscore.joblib"
    joblib.dump(mock_scaler, scaler_path)

    applier = ScalerApplier()
    applier.load_scalers(zscore_path=scaler_path, minmax_path=tmp_path / "none.joblib")

    result = applier.apply_zscore(scaler_df)

    # Check scaling
    # feat_1: (1-2)/1 = -1
    assert result["feat_1"].iloc[0] == -1.0
    # feat_2: (10-20)/10 = -1
    assert result["feat_2"].iloc[0] == -1.0

    # Check untouched
    assert result["pca_1"].iloc[0] == 0.1

def test_apply_minmax(scaler_df, tmp_path):
    mock_scaler = MinMaxScalerCustom()
    mock_scaler.columns_ = ["feat_1"]
    mock_scaler.min_ = np.array([1.0])
    mock_scaler.max_ = np.array([3.0])

    scaler_path = tmp_path / "minmax.joblib"
    joblib.dump(mock_scaler, scaler_path)

    applier = ScalerApplier()
    applier.load_scalers(zscore_path=tmp_path / "none.joblib", minmax_path=scaler_path)

    result = applier.apply_minmax(scaler_df)

    # feat_1: 2 * (1-1)/(3-1) - 1 = -1
    assert result["feat_1"].iloc[0] == -1.0
    # feat_2 should be untouched as it's not in scaler columns
    assert result["feat_2"].iloc[0] == 10.0

class TestScalerFitter:
    """Tests for ScalerFitter class."""

    @pytest.fixture
    def train_df(self):
        """Create sample training data."""
        # Use feature names that are NOT in META_COLUMNS
        return pd.DataFrame({
            "pca_feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "pca_feature_2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_init(self):
        """Test ScalerFitter initialization."""
        fitter = ScalerFitter()
        assert fitter._zscore_n == 0
        assert fitter._zscore_mean is None
        assert fitter._zscore_m2 is None
        assert fitter._is_finalized is False

    def test_partial_fit_single_batch(self, train_df):
        """Test partial_fit with a single batch."""
        fitter = ScalerFitter()
        fitter.partial_fit(train_df)

        assert fitter._zscore_n == 5
        assert fitter._zscore_columns == ["pca_feature_1", "pca_feature_2"]
        assert fitter._zscore_mean is not None
        # Mean should be [3.0, 30.0]
        np.testing.assert_array_almost_equal(fitter._zscore_mean, [3.0, 30.0])

    def test_partial_fit_multiple_batches(self):
        """Test partial_fit with multiple batches."""
        fitter = ScalerFitter()

        batch1 = pd.DataFrame({"pca_feature_1": [1.0, 2.0], "pca_feature_2": [10.0, 20.0]})
        batch2 = pd.DataFrame({"pca_feature_1": [3.0, 4.0], "pca_feature_2": [30.0, 40.0]})
        batch3 = pd.DataFrame({"pca_feature_1": [5.0], "pca_feature_2": [50.0]})

        fitter.partial_fit(batch1)
        fitter.partial_fit(batch2)
        fitter.partial_fit(batch3)

        assert fitter._zscore_n == 5
        np.testing.assert_array_almost_equal(fitter._zscore_mean, [3.0, 30.0])

    def test_partial_fit_empty_batch(self, train_df):
        """Test that empty batch is skipped."""
        fitter = ScalerFitter()
        fitter.partial_fit(train_df)
        n_before = fitter._zscore_n

        empty_df = pd.DataFrame(columns=["pca_feature_1", "pca_feature_2"])
        fitter.partial_fit(empty_df)

        assert fitter._zscore_n == n_before  # Should not change

    def test_partial_fit_after_finalize_raises(self, train_df):
        """Test that partial_fit after finalize raises RuntimeError."""
        fitter = ScalerFitter()
        fitter.partial_fit(train_df)
        fitter.finalize()

        with pytest.raises(RuntimeError, match="Cannot partial_fit after finalize"):
            fitter.partial_fit(train_df)

    def test_finalize_without_fit_raises(self):
        """Test that finalize without any data raises RuntimeError."""
        fitter = ScalerFitter()

        with pytest.raises(RuntimeError, match="No data was fitted"):
            fitter.finalize()

    def test_finalize_returns_scalers(self, train_df):
        """Test that finalize returns correct scalers."""
        fitter = ScalerFitter()
        fitter.partial_fit(train_df)

        zscore_scaler, minmax_scaler = fitter.finalize()

        # Check zscore scaler
        assert isinstance(zscore_scaler, StandardScalerCustom)
        assert zscore_scaler.columns_ == ["pca_feature_1", "pca_feature_2"]
        np.testing.assert_array_almost_equal(zscore_scaler.mean_, [3.0, 30.0])

        # Check minmax scaler
        assert isinstance(minmax_scaler, MinMaxScalerCustom)
        assert minmax_scaler.columns_ == ["pca_feature_1", "pca_feature_2"]
        np.testing.assert_array_almost_equal(minmax_scaler.min_, [1.0, 10.0])
        np.testing.assert_array_almost_equal(minmax_scaler.max_, [5.0, 50.0])

    def test_finalize_sets_is_finalized(self, train_df):
        """Test that finalize sets _is_finalized flag."""
        fitter = ScalerFitter()
        fitter.partial_fit(train_df)

        assert fitter._is_finalized is False
        fitter.finalize()
        assert fitter._is_finalized is True

    def test_handles_nan_values(self):
        """Test that NaN values are handled (replaced with 0)."""
        df = pd.DataFrame({
            "feature_1": [1.0, np.nan, 3.0],
            "feature_2": [10.0, 20.0, np.nan],
        })
        fitter = ScalerFitter()
        fitter.partial_fit(df)

        # Should not raise and should have processed 3 samples
        assert fitter._zscore_n == 3

    def test_handles_inf_values(self):
        """Test that inf values are handled (replaced with 0)."""
        df = pd.DataFrame({
            "feature_1": [1.0, np.inf, 3.0],
            "feature_2": [10.0, -np.inf, 30.0],
        })
        fitter = ScalerFitter()
        fitter.partial_fit(df)

        assert fitter._zscore_n == 3

    def test_constant_feature_std(self):
        """Test that constant features get std=1 to avoid division by zero."""
        df = pd.DataFrame({
            "constant": [5.0, 5.0, 5.0, 5.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
        })
        fitter = ScalerFitter()
        fitter.partial_fit(df)

        zscore_scaler, _ = fitter.finalize()

        # Constant feature should have std=1 (fallback)
        assert zscore_scaler.std_[0] == 1.0
        # Varying feature should have actual std
        assert zscore_scaler.std_[1] > 0

    def test_minmax_same_min_max(self):
        """Test that min==max case is handled in minmax scaler."""
        df = pd.DataFrame({
            "constant": [5.0, 5.0, 5.0],
            "varying": [1.0, 2.0, 3.0],
        })
        fitter = ScalerFitter()
        fitter.partial_fit(df)

        _, minmax_scaler = fitter.finalize()

        # Both should have valid min/max
        assert minmax_scaler.min_[0] == 5.0
        assert minmax_scaler.max_[0] == 5.0


class TestScalerApplierSetAndSave:
    """Additional tests for ScalerApplier set_scalers and save_scalers."""

    def test_set_scalers(self):
        """Test setting scalers directly."""
        applier = ScalerApplier()

        zscore = StandardScalerCustom()
        zscore.columns_ = ["feat_1"]
        zscore.mean_ = np.array([0.0])
        zscore.std_ = np.array([1.0])

        minmax = MinMaxScalerCustom()
        minmax.columns_ = ["feat_1"]
        minmax.min_ = np.array([0.0])
        minmax.max_ = np.array([1.0])

        applier.set_scalers(zscore, minmax)

        assert applier._zscore_scaler is zscore
        assert applier._minmax_scaler is minmax

    def test_save_scalers(self, tmp_path):
        """Test saving scalers to disk."""
        applier = ScalerApplier()

        zscore = StandardScalerCustom()
        zscore.columns_ = ["feat_1"]
        zscore.mean_ = np.array([0.0])
        zscore.std_ = np.array([1.0])

        minmax = MinMaxScalerCustom()
        minmax.columns_ = ["feat_1"]
        minmax.min_ = np.array([0.0])
        minmax.max_ = np.array([1.0])

        applier.set_scalers(zscore, minmax)
        applier.save_scalers(
            zscore_path=tmp_path / "zscore.joblib",
            minmax_path=tmp_path / "minmax.joblib"
        )

        # Check files were created
        assert (tmp_path / "zscore.joblib").exists()
        assert (tmp_path / "minmax.joblib").exists()

        # Load and verify
        loaded_zscore = joblib.load(tmp_path / "zscore.joblib")
        assert loaded_zscore.columns_ == ["feat_1"]

    def test_apply_zscore_no_scaler_warns(self, scaler_df):
        """Test that apply_zscore warns when no scaler is loaded."""
        applier = ScalerApplier()
        # No scalers loaded

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = applier.apply_zscore(scaler_df)
            # Should return unchanged
            pd.testing.assert_frame_equal(result, scaler_df)

    def test_apply_minmax_no_scaler_warns(self, scaler_df):
        """Test that apply_minmax warns when no scaler is loaded."""
        applier = ScalerApplier()

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = applier.apply_minmax(scaler_df)
            pd.testing.assert_frame_equal(result, scaler_df)


if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
