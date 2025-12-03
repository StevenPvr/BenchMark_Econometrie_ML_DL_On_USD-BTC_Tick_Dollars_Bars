from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.features.main import main
from src.features.compute import apply_lags, compute_all_features
from src.features.io import fit_and_save_scalers, load_input_data, save_outputs
from src.features.pipeline import (
    TRAIN_RATIO,
    compute_timestamp_features,
    drop_initial_nan_rows,
    drop_timestamp_columns,
    get_columns_to_scale,
    interpolate_sporadic_nan,
    shift_target_to_future_return,
    split_train_test,
)

# Mock constants where they are IMPORTED in the modules
@pytest.fixture
def mock_paths(mocker):
    mocker.patch("src.features.io.DOLLAR_BARS_PARQUET", MagicMock(exists=MagicMock(return_value=True), name="DOLLAR_BARS_PARQUET"))
    mocker.patch("src.features.io.DATASET_FEATURES_PARQUET", MagicMock(name="DATASET_FEATURES_PARQUET"))
    mocker.patch("src.features.io.DATASET_FEATURES_LINEAR_PARQUET", MagicMock(name="DATASET_FEATURES_LINEAR_PARQUET"))
    mocker.patch("src.features.io.DATASET_FEATURES_LSTM_PARQUET", MagicMock(name="DATASET_FEATURES_LSTM_PARQUET"))
    mocker.patch("src.features.io.SCALERS_DIR", MagicMock(name="SCALERS_DIR"))
    mocker.patch("src.features.io.ZSCORE_SCALER_FILE", MagicMock(name="ZSCORE_SCALER_FILE"))
    mocker.patch("src.features.io.MINMAX_SCALER_FILE", MagicMock(name="MINMAX_SCALER_FILE"))
    mocker.patch("src.features.io.FEATURES_DIR", MagicMock(name="FEATURES_DIR"))

def test_load_input_data(mocker, mock_paths):
    # Mock pd.read_parquet
    mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mocker.patch("pandas.read_parquet", return_value=mock_df)

    df = load_input_data()
    assert df.equals(mock_df)

    # Test file not found
    from src.features.io import DOLLAR_BARS_PARQUET
    DOLLAR_BARS_PARQUET.exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_input_data()

def test_compute_all_features(sample_bars, mocker):
    # Mock all sub-feature computation functions to avoid heavy computation
    # We just need to check if they are called and if results are combined

    # Mock return values as DataFrames with specific columns
    mocker.patch("src.features.compute.compute_cumulative_returns", return_value=pd.DataFrame({"cum_ret": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_recent_extremes", return_value=pd.DataFrame({"extremes": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_realized_volatility", return_value=pd.DataFrame({"vol": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_return_volatility_ratio", return_value=pd.DataFrame({"rvr": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_realized_skewness", return_value=pd.DataFrame({"skew": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_realized_kurtosis", return_value=pd.DataFrame({"kurt": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_all_jump_features", return_value=pd.DataFrame({"jump": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_moving_averages", return_value=pd.DataFrame({"ma": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_price_zscore", return_value=pd.DataFrame({"zscore": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_cross_ma", return_value=pd.DataFrame({"cross": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_return_streak", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="streak"))

    mocker.patch("src.features.compute.compute_parkinson_volatility", return_value=pd.DataFrame({"park": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_garman_klass_volatility", return_value=pd.DataFrame({"gk": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_rogers_satchell_volatility", return_value=pd.DataFrame({"rs": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_yang_zhang_volatility", return_value=pd.DataFrame({"yz": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_range_ratios", return_value=pd.DataFrame({"ratio": [1]*len(sample_bars)}, index=sample_bars.index))

    # Add dummy columns for conditional checks
    df_bars = sample_bars.copy()
    df_bars["duration_sec"] = 100
    df_bars["buy_volume"] = 100
    df_bars["sell_volume"] = 100

    mocker.patch("src.features.compute.compute_temporal_acceleration", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="accel"))
    mocker.patch("src.features.compute.compute_temporal_acceleration_smoothed", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="accel_smooth"))
    mocker.patch("src.features.compute.compute_temporal_jerk", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="jerk"))

    mocker.patch("src.features.compute.compute_vpin", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="vpin"))
    mocker.patch("src.features.compute.compute_kyle_lambda", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="kyle"))

    mocker.patch("src.features.compute.compute_shannon_entropy", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="shannon"))
    mocker.patch("src.features.compute.compute_approximate_entropy", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="apen"))
    mocker.patch("src.features.compute.compute_sample_entropy", return_value=pd.Series([1]*len(sample_bars), index=sample_bars.index, name="sampen"))

    mocker.patch("src.features.compute.compute_all_temporal_features", return_value=pd.DataFrame({"temporal": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_frac_diff_features", return_value=pd.DataFrame({"frac": [1]*len(sample_bars)}, index=sample_bars.index))
    mocker.patch("src.features.compute.compute_all_technical_indicators", return_value=pd.DataFrame({"ta": [1]*len(sample_bars)}, index=sample_bars.index))

    df_all = compute_all_features(df_bars)

    # Check if original columns are preserved
    assert all(col in df_all.columns for col in df_bars.columns)
    # Check if new columns are added
    assert "cum_ret" in df_all.columns
    assert "extremes" in df_all.columns
    assert "ta" in df_all.columns

    # Check volume imbalance manual calculation
    assert "volume_imbalance" in df_all.columns

def test_apply_lags(mocker):
    df_in = pd.DataFrame({"a": [1, 2, 3], "timestamp": [1, 2, 3]})
    df_out = pd.DataFrame({"a": [1, 2, 3], "a_lag1": [np.nan, 1, 2]})

    mocker.patch("src.features.compute.generate_all_lags", return_value=df_out)

    res = apply_lags(df_in)
    assert res.equals(df_out)

def test_interpolate_sporadic_nan():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0],
        "b": [np.nan, 2.0, 3.0, np.nan]
    })

    res = interpolate_sporadic_nan(df)

    # Check no NaNs
    assert not res.isna().any().any()
    # Check values (ffill then bfill)
    assert res["a"].tolist() == [1.0, 1.0, 3.0, 4.0]
    assert res["b"].tolist() == [2.0, 2.0, 3.0, 3.0]

def test_drop_initial_nan_rows():
    df = pd.DataFrame({
        "a": [np.nan, np.nan, 3.0, 4.0],
        "b": [np.nan, 2.0, 3.0, 4.0],
        "c": ["x", "y", "z", "w"] # Non-numeric, should be ignored for check
    })

    # Both numeric cols have values starting from index 2
    res = drop_initial_nan_rows(df)
    assert len(res) == 2
    assert res.index[0] == 2
    assert res.iloc[0]["a"] == 3.0

def test_shift_target_to_future_return():
    df = pd.DataFrame({
        "log_return": [1.0, 2.0, 3.0, 4.0],
        "feature": [10, 20, 30, 40]
    })

    res = shift_target_to_future_return(df, target_col="log_return")

    # Should drop last row
    assert len(res) == 3
    # Check shift: index 0 should have log_return from index 1 (which is 2.0)
    assert res.iloc[0]["log_return"] == 2.0
    assert res.iloc[1]["log_return"] == 3.0
    assert res.iloc[2]["log_return"] == 4.0
    # Features stay same
    assert res.iloc[0]["feature"] == 10

def test_split_train_test():
    df = pd.DataFrame({"a": range(10)})
    train, test, split_idx = split_train_test(df, train_ratio=0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert split_idx == 8
    assert train.iloc[-1]["a"] == 7
    assert test.iloc[0]["a"] == 8

def test_get_columns_to_scale():
    df = pd.DataFrame({
        "feature_1": [1],
        "feature_2_sin": [1], # excluded suffix
        "timestamp_open": [1], # excluded pattern
        "log_return": [1], # target
        "log_return_lag1": [1], # lag of target
    })

    # 1. Default: exclude target, exclude log_return lags
    cols = get_columns_to_scale(df, exclude_target=True, include_log_return_lags=False)
    assert "feature_1" in cols
    assert "feature_2_sin" not in cols
    assert "timestamp_open" not in cols
    assert "log_return" not in cols
    assert "log_return_lag1" not in cols

    # 2. Include log_return lags
    cols = get_columns_to_scale(df, exclude_target=True, include_log_return_lags=True)
    assert "log_return_lag1" in cols

    # 3. Include target
    cols = get_columns_to_scale(df, exclude_target=False)
    assert "log_return" in cols

def test_fit_and_save_scalers(mocker, mock_paths):
    df_train = pd.DataFrame({"a": [1, 2, 3]})

    mock_std_scaler = MagicMock()
    mock_minmax_scaler = MagicMock()

    mocker.patch("src.features.io.StandardScalerCustom", return_value=mock_std_scaler)
    mocker.patch("src.features.io.MinMaxScalerCustom", return_value=mock_minmax_scaler)

    fit_and_save_scalers(df_train)

    assert mock_std_scaler.fit.called
    assert mock_std_scaler.save.called
    assert mock_minmax_scaler.fit.called
    assert mock_minmax_scaler.save.called

def test_compute_timestamp_features(sample_bars):
    # Ensure needed columns
    df = sample_bars.copy()
    # sample_bars index is datetime, but we need timestamp_open/close columns for this function
    df["timestamp_open"] = df.index.astype(np.int64) // 10**6 # ms
    df["timestamp_close"] = df["timestamp_open"] + 60*1000 # 1 min later

    res = compute_timestamp_features(df)

    assert "bar_duration_ts" in res.columns
    assert "time_gap_bars" in res.columns
    assert res["bar_duration_ts"].iloc[0] == 60.0

def test_drop_timestamp_columns(sample_bars):
    df = sample_bars.copy()
    df["timestamp_open"] = 123
    df["timestamp_close"] = 456
    df["datetime_open"] = pd.Timestamp.now()

    res = drop_timestamp_columns(df)

    assert "timestamp_open" not in res.columns
    assert "timestamp_close" not in res.columns
    assert "datetime_open" not in res.columns

def test_save_outputs(mocker, mock_paths):
    df = pd.DataFrame({"a": [1], "timestamp_open": [123]})

    mock_to_parquet = mocker.patch.object(pd.DataFrame, "to_parquet")

    save_outputs(df)

    # Should be called 3 times (features, linear, lstm)
    assert mock_to_parquet.call_count == 3

def test_main(mocker, mock_paths):
    # Mock sys.argv with --no-batch to use non-batch mode where our mocks apply
    mocker.patch("sys.argv", ["main.py", "--no-batch"])

    # Mock all the steps (these apply to non-batch mode)
    mocker.patch("src.features.main.setup_logging")
    mocker.patch("src.features.main.load_input_data", return_value=pd.DataFrame({"close": [1, 2], "log_return": [0.1, 0.2]}))
    mocker.patch("src.features.main.compute_timestamp_features", side_effect=lambda x: x)
    mocker.patch("src.features.main.compute_all_features", side_effect=lambda x: x)
    mocker.patch("src.features.main.apply_lags", side_effect=lambda x: x)
    mocker.patch("src.features.main.drop_initial_nan_rows", side_effect=lambda x: x)
    mocker.patch("src.features.main.interpolate_sporadic_nan", side_effect=lambda x: x)
    mocker.patch("src.features.main.shift_target_to_future_return", side_effect=lambda x, target_col: x)
    mocker.patch("src.features.main.split_train_test", return_value=(pd.DataFrame(), pd.DataFrame(), 0))
    mocker.patch("src.features.main.save_outputs")

    main()

    # If no exception, it passed the flow
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--color=yes"])
