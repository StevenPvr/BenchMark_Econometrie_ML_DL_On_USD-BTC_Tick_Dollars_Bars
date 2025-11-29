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
from src.features.temporal_calendar import (
    compute_cyclical_time_features,
    compute_time_since_shock,
    compute_volatility_regime,
    compute_drawdown_features,
    compute_all_temporal_features,
    _time_since_shock,
    _time_since_positive_shock,
    _time_since_negative_shock,
    _rolling_std,
    _expanding_quantile,
    _expanding_mean,
    _expanding_std,
    _compute_drawdown,
    _compute_drawup,
)

# =============================================================================
# NUMBA UNIT TESTS
# =============================================================================

def test_time_since_shock_numba():
    returns = np.array([0.005, 0.02, 0.001, 0.002, 0.03, 0.001], dtype=np.float64)
    threshold = 0.01

    # 0: 0.005 <= 0.01. Bars=0.
    # 1: 0.02 > 0.01. Reset. Bars=0.
    # 2: 0.001 <= 0.01. Bars=1.
    # 3: 0.002 <= 0.01. Bars=2.
    # 4: 0.03 > 0.01. Reset. Bars=0.
    # 5: 0.001 <= 0.01. Bars=1.

    expected = np.array([0., 0., 1., 2., 0., 1.], dtype=np.float64)
    res = _time_since_shock(returns, threshold)
    np.testing.assert_array_equal(res, expected)

    # NaNs
    returns_nan = np.array([np.nan, 0.02, np.nan, 0.001], dtype=np.float64)
    # 0: NaN -> Bars=0. Next=1.
    # 1: 0.02 > 0.01 -> Bars=0. Next=1.
    # 2: NaN -> Bars=1. Next=2.
    # 3: 0.001 <= 0.01 -> Bars=2.
    res_nan = _time_since_shock(returns_nan, threshold)
    np.testing.assert_array_equal(res_nan, np.array([0., 0., 1., 2.]))

def test_time_since_positive_shock_numba():
    returns = np.array([0.005, 0.02, -0.02, 0.001], dtype=np.float64)
    threshold = 0.01
    # 0: 0.005 <= 0.01 -> 0
    # 1: 0.02 > 0.01 -> 0 (reset)
    # 2: -0.02 <= 0.01 -> 1
    # 3: 0.001 <= 0.01 -> 2
    res = _time_since_positive_shock(returns, threshold)
    np.testing.assert_array_equal(res, np.array([0., 0., 1., 2.]))

def test_time_since_negative_shock_numba():
    returns = np.array([-0.005, -0.02, 0.02, -0.001], dtype=np.float64)
    threshold = 0.01 # Checks < -0.01
    # 0: -0.005 >= -0.01 -> 0
    # 1: -0.02 < -0.01 -> 0 (reset)
    # 2: 0.02 >= -0.01 -> 1
    # 3: -0.001 >= -0.01 -> 2
    res = _time_since_negative_shock(returns, threshold)
    np.testing.assert_array_equal(res, np.array([0., 0., 1., 2.]))

def test_expanding_functions():
    values = np.array([1., 2., 3., 4., 5.], dtype=np.float64)
    min_periods = 2

    # Mean:
    # 0: 1. (count=1 < 2) -> NaN
    # 1: (1+2)/2 = 1.5
    # 2: (1+2+3)/3 = 2.
    res_mean = _expanding_mean(values, min_periods)
    assert np.isnan(res_mean[0])
    assert res_mean[1] == 1.5
    assert res_mean[4] == 3.0

    # Std:
    # 0: NaN
    # 1: std([1,2]) = sqrt(0.5) ~= 0.707
    res_std = _expanding_std(values, min_periods)
    assert np.isnan(res_std[0])
    assert np.isclose(res_std[1], np.std([1,2], ddof=1))

    # NaNs in input
    val_nan = np.array([1., np.nan, 2.], dtype=np.float64)
    # 0: 1. (count=1) -> NaN
    # 1: NaN (count=1) -> NaN (if min_periods=2)
    # 2: 2. (count=2) -> mean=1.5
    res_mean_nan = _expanding_mean(val_nan, min_periods=2)
    assert np.isnan(res_mean_nan[1])
    assert res_mean_nan[2] == 1.5

def test_expanding_std_edge_cases():
    # Only 1 value valid
    values = np.array([1., np.nan, np.nan], dtype=np.float64)
    res = _expanding_std(values, min_periods=1)
    # Count=1 -> std=0 per logic? Or NaN?
    # Logic: if count >= min_periods and count > 1: sqrt(...) else 0.0 if count >= min
    # At i=0: count=1. min=1. count > 1 False -> 0.0
    assert np.isnan(res[0])

    # All NaNs
    values_all_nan = np.array([np.nan, np.nan], dtype=np.float64)
    res_all_nan = _expanding_std(values_all_nan, min_periods=1)
    assert np.all(np.isnan(res_all_nan))

def test_compute_drawdown_numba():
    prices = np.array([100., 110., 99., 120., 120., 108.], dtype=np.float64)

    # 0: 100. Max=100. DD=0.
    # 1: 110. Max=110. DD=0.
    # 2: 99. Max=110. DD=(99-110)/110 = -0.1
    # 3: 120. Max=120. DD=0.
    # 4: 120. Max=120. DD=0.
    # 5: 108. Max=120. DD=(108-120)/120 = -0.1

    dd, rmax, bars = _compute_drawdown(prices)
    np.testing.assert_allclose(dd, np.array([0., 0., -0.1, 0., 0., -0.1]))
    np.testing.assert_array_equal(rmax, np.array([100., 110., 110., 120., 120., 120.]))

    # Bars since high
    # 0: 0
    # 1: 0 (new high)
    # 2: 1
    # 3: 0 (new high)
    # 4: 1 (equal high is not > current_max? logic: if p > current_max. 120 > 120 False. else bars+=1)
    # So index 4 (120) is same as max. Logic: else bars+=1. So bars=1. Correct.
    # 5: 2
    # Note: at index 0, code increments bars_since because it goes to 'else' branch (100 > 100 False).
    np.testing.assert_array_equal(bars, np.array([1., 0., 1., 0., 1., 2.]))

def test_compute_drawup_numba():
    prices = np.array([100., 90., 99., 80., 80., 88.], dtype=np.float64)
    # Min logic similar to max
    du, rmin, bars = _compute_drawup(prices)

    # 0: 100. Min=100. DU=0.
    # 1: 90. Min=90. DU=0.
    # 2: 99. Min=90. DU=(99-90)/90 = 0.1

    assert du[2] == 0.1
    assert rmin[1] == 90.

def test_drawdown_nan_empty():
    # Empty
    dd, _, _ = _compute_drawdown(np.array([], dtype=np.float64))
    assert len(dd) == 0

    # NaN
    prices = np.array([100., np.nan, 90.], dtype=np.float64)
    dd, _, _ = _compute_drawdown(prices)
    assert np.isnan(dd[1])
    assert dd[2] == -0.1

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def sample_bars_temporal(sample_bars):
    # Ensure timestamp_close is datetime
    sample_bars['timestamp_close'] = pd.date_range(start='2023-01-01', periods=len(sample_bars), freq='h')
    return sample_bars

def test_compute_cyclical_time_features(sample_bars_temporal):
    df = compute_cyclical_time_features(sample_bars_temporal)
    expected_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
                     "hour_of_day", "day_of_week", "day_of_month", "week_of_year", "month_of_year", "quarter"]
    for col in expected_cols:
        assert col in df.columns

    assert not df["hour_sin"].isna().any()

def test_compute_time_since_shock(sample_bars_temporal):
    # Create some shocks
    # Use iloc to ensure we modify the array at the correct position
    log_ret_idx = sample_bars_temporal.columns.get_loc("log_return")
    sample_bars_temporal.iloc[10, log_ret_idx] = 0.1
    sample_bars_temporal.iloc[20, log_ret_idx] = -0.1

    df = compute_time_since_shock(sample_bars_temporal, thresholds=[0.05])
    assert "bars_since_shock_0.05" in df.columns
    assert "bars_since_pos_shock_0.05" in df.columns
    assert "bars_since_neg_shock_0.05" in df.columns

    # Check logic
    # At index 10, shock happens. So value should be 0.
    assert df["bars_since_pos_shock_0.05"].iloc[10] == 0.0
    # At index 11, should be 1.
    assert df["bars_since_pos_shock_0.05"].iloc[11] == 1.0

def test_compute_volatility_regime(sample_bars_temporal):
    df = compute_volatility_regime(sample_bars_temporal, vol_window=5)
    assert "volatility_5" in df.columns
    assert "vol_regime_q90" in df.columns
    assert "vol_zscore" in df.columns

    # Check that regime is 0 or 1 (or NaN)
    valid = df["vol_regime_q90"].dropna()
    assert valid.isin([0.0, 1.0]).all()

def test_compute_drawdown_features(sample_bars_temporal):
    df = compute_drawdown_features(sample_bars_temporal)
    assert "drawdown" in df.columns
    assert "drawup" in df.columns
    assert "crash_20pct" in df.columns # default threshold -0.20

    assert (df["drawdown"].dropna() <= 0).all()
    assert (df["drawup"].dropna() >= 0).all()

def test_compute_all_temporal_features(sample_bars_temporal):
    df = compute_all_temporal_features(sample_bars_temporal)
    # Should contain cols from all
    assert "hour_sin" in df.columns
    assert "drawdown" in df.columns
    assert "volatility_20" in df.columns # default window

    # Test with empty df
    empty_df = pd.DataFrame()
    res = compute_all_temporal_features(empty_df)
    assert res.empty

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
