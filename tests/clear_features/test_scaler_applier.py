import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.clear_features.scaler_applier import ScalerApplier
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
    applier = ScalerApplier(
        zscore_scaler_path=tmp_path / "missing_z.joblib",
        minmax_scaler_path=tmp_path / "missing_m.joblib"
    )
    applier.load_scalers()
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

    applier = ScalerApplier(zscore_scaler_path=scaler_path)
    applier.load_scalers()

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

    applier = ScalerApplier(minmax_scaler_path=scaler_path)
    applier.load_scalers()

    result = applier.apply_minmax(scaler_df)

    # feat_1: 2 * (1-1)/(3-1) - 1 = -1
    assert result["feat_1"].iloc[0] == -1.0
    # feat_2 should be untouched as it's not in scaler columns
    assert result["feat_2"].iloc[0] == 10.0
