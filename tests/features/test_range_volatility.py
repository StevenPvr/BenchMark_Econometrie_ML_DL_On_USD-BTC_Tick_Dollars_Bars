from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from src.features.range_volatility import (
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_rogers_satchell_volatility,
    compute_yang_zhang_volatility,
    compute_range_ratios,
)
from src.features.range_volatility_core import (
    _parkinson_single,
    _garman_klass_single,
    _rogers_satchell_single,
    _rolling_parkinson,
    _rolling_garman_klass,
    _rolling_rogers_satchell,
    _rolling_yang_zhang,
    _compute_range_ratios,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def edge_case_data():
    """Create data with NaNs, zeros, and negative values to test robustness."""
    data = {
        'high': [10, np.nan, 10, 10, -5],
        'low': [5, 5, np.nan, 0, -10],
        'open': [6, 6, 6, 6, -8],
        'close': [9, 9, 9, 9, -6]
    }
    return pd.DataFrame(data)

@pytest.fixture
def small_data():
    """Create data smaller than the window size."""
    data = {
        'high': [10, 11, 12],
        'low': [5, 6, 7],
        'open': [6, 7, 8],
        'close': [9, 10, 11]
    }
    return pd.DataFrame(data)

@pytest.fixture
def constant_data():
    """Create data with constant prices (zero volatility)."""
    return pd.DataFrame({
        'high': [10.0] * 20,
        'low': [10.0] * 20,
        'open': [10.0] * 20,
        'close': [10.0] * 20
    })

# =============================================================================
# UNIT TESTS FOR NUMBA FUNCTIONS
# =============================================================================

def test_parkinson_single_edge_cases():
    # Normal case
    assert _parkinson_single(10.0, 5.0) > 0
    # NaNs
    assert np.isnan(_parkinson_single(np.nan, 5.0))
    assert np.isnan(_parkinson_single(10.0, np.nan))
    # Zero or Negative
    assert np.isnan(_parkinson_single(10.0, 0.0))
    assert np.isnan(_parkinson_single(10.0, -5.0))
    assert np.isnan(_parkinson_single(0.0, 5.0))

def test_garman_klass_single_edge_cases():
    # Normal case
    assert _garman_klass_single(10.0, 5.0, 6.0, 9.0) > 0
    # NaNs
    assert np.isnan(_garman_klass_single(np.nan, 5.0, 6.0, 9.0))
    assert np.isnan(_garman_klass_single(10.0, np.nan, 6.0, 9.0))
    assert np.isnan(_garman_klass_single(10.0, 5.0, np.nan, 9.0))
    assert np.isnan(_garman_klass_single(10.0, 5.0, 6.0, np.nan))
    # Zero or Negative
    assert np.isnan(_garman_klass_single(10.0, 0.0, 6.0, 9.0))
    assert np.isnan(_garman_klass_single(10.0, 5.0, 0.0, 9.0))
    assert np.isnan(_garman_klass_single(10.0, 5.0, 6.0, 0.0))

def test_rogers_satchell_single_edge_cases():
    # Normal case
    assert isinstance(_rogers_satchell_single(10.0, 5.0, 6.0, 9.0), float)
    # NaNs
    assert np.isnan(_rogers_satchell_single(np.nan, 5.0, 6.0, 9.0))
    # Zero or Negative
    assert np.isnan(_rogers_satchell_single(10.0, 0.0, 6.0, 9.0))

def test_rolling_functions_short_input():
    """Test rolling functions with input length < window."""
    h = np.array([10., 11.], dtype=np.float64)
    l = np.array([5., 6.], dtype=np.float64)
    o = np.array([6., 7.], dtype=np.float64)
    c = np.array([9., 10.], dtype=np.float64)
    window = 5

    res_park = _rolling_parkinson(h, l, window)
    assert np.all(np.isnan(res_park))
    assert len(res_park) == 2

    res_gk = _rolling_garman_klass(h, l, o, c, window)
    assert np.all(np.isnan(res_gk))

    res_rs = _rolling_rogers_satchell(h, l, o, c, window)
    assert np.all(np.isnan(res_rs))

    res_yz = _rolling_yang_zhang(h, l, o, c, window)
    assert np.all(np.isnan(res_yz))

def test_rolling_functions_all_nan_window():
    """Test rolling functions where a window contains only invalid data."""
    h = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    l = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    window = 2

    # Parkinson
    res = _rolling_parkinson(h, l, window)
    assert np.isnan(res[1]) # Window at index 1 is all NaN

def test_compute_range_ratios_edge_cases():
    h = np.array([10., 10., 10.], dtype=np.float64)
    l = np.array([5., 10., np.nan], dtype=np.float64) # 2nd element: High=Low -> range=0
    o = np.array([6., 10., 6.], dtype=np.float64)
    c = np.array([9., 10., 9.], dtype=np.float64)

    r_ratio, b_ratio = _compute_range_ratios(h, l, o, c)

    # Index 0: Normal
    assert not np.isnan(r_ratio[0])
    assert not np.isnan(b_ratio[0])

    # Index 1: High=Low -> range=0. Body ratio should be 0.0
    assert r_ratio[1] == 0.0
    assert b_ratio[1] == 0.0

    # Index 2: NaN input
    assert np.isnan(r_ratio[2])
    assert np.isnan(b_ratio[2])

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_compute_parkinson_volatility(sample_bars):
    df = compute_parkinson_volatility(sample_bars)
    # Check for one of the windows
    assert "parkinson_vol_20" in df.columns
    assert not df["parkinson_vol_20"].iloc[20:].isna().all()

def test_compute_parkinson_volatility_edge_cases(edge_case_data):
    df = compute_parkinson_volatility(edge_case_data, windows=[2])
    assert "parkinson_vol_2" in df.columns
    # Should handle NaNs gracefully (result in NaNs) without crashing
    assert len(df) == len(edge_case_data)

def test_compute_garman_klass_volatility(sample_bars):
    df = compute_garman_klass_volatility(sample_bars)
    assert "garman_klass_vol_20" in df.columns
    assert not df["garman_klass_vol_20"].iloc[20:].isna().all()

def test_compute_garman_klass_volatility_edge_cases(edge_case_data):
    df = compute_garman_klass_volatility(edge_case_data, windows=[2])
    assert "garman_klass_vol_2" in df.columns

def test_compute_rogers_satchell_volatility(sample_bars):
    df = compute_rogers_satchell_volatility(sample_bars)
    assert "rogers_satchell_vol_20" in df.columns
    assert not df["rogers_satchell_vol_20"].iloc[20:].isna().all()

def test_compute_rogers_satchell_volatility_edge_cases(edge_case_data):
    df = compute_rogers_satchell_volatility(edge_case_data, windows=[2])
    assert "rogers_satchell_vol_2" in df.columns

def test_compute_yang_zhang_volatility(sample_bars):
    df = compute_yang_zhang_volatility(sample_bars)
    assert "yang_zhang_vol_20" in df.columns
    assert not df["yang_zhang_vol_20"].iloc[20:].isna().all()

def test_compute_yang_zhang_volatility_edge_cases(edge_case_data):
    # Yang Zhang needs previous close, so index 0 is always NaN,
    # and it needs a window.
    df = compute_yang_zhang_volatility(edge_case_data, windows=[2])
    assert "yang_zhang_vol_2" in df.columns

def test_compute_range_ratios(sample_bars):
    df = compute_range_ratios(sample_bars)
    assert "body_ratio" in df.columns
    assert "range_ratio" in df.columns

    # Check bounds
    assert (df["body_ratio"].dropna() >= 0).all()

def test_compute_range_ratios_edge_cases(edge_case_data):
    df = compute_range_ratios(edge_case_data)
    assert "body_ratio" in df.columns
    assert len(df) == len(edge_case_data)

def test_yang_zhang_insufficient_data_in_window():
    """Test Yang-Zhang when a window has too few valid data points."""
    # Create data where valid count < 2 inside the loop
    # We need a window where we have mostly NaNs
    h = np.array([10., 10., 10., np.nan, np.nan], dtype=np.float64)
    l = h
    o = h
    c = h
    window = 3

    # We call the rolling function directly to test the specific branch
    res = _rolling_yang_zhang(h, l, o, c, window)
    # The last few points might be NaN because of the NaNs in input
    assert np.isnan(res[-1])

def test_zero_volatility(constant_data):
    """Test that constant prices result in 0 or NaN volatility."""
    # Parkinson
    df_park = compute_parkinson_volatility(constant_data, windows=[5])
    # log(10/10) = 0 -> vol should be 0
    assert df_park["parkinson_vol_5"].iloc[10] == 0.0

    # Garman Klass
    df_gk = compute_garman_klass_volatility(constant_data, windows=[5])
    assert df_gk["garman_klass_vol_5"].iloc[10] == 0.0

    # Rogers Satchell
    df_rs = compute_rogers_satchell_volatility(constant_data, windows=[5])
    assert df_rs["rogers_satchell_vol_5"].iloc[10] == 0.0

    # Yang Zhang
    df_yz = compute_yang_zhang_volatility(constant_data, windows=[5])
    assert df_yz["yang_zhang_vol_5"].iloc[10] == 0.0
