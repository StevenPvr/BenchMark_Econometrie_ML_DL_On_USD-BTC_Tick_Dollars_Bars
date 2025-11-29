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
from src.features.microstructure_volatility import (
    compute_intrabar_volatility,
    compute_microstructure_features,
    _compute_tick_stats,
    _aggregate_tick_stats_by_bar,
    _rolling_std_numba,
)

# =============================================================================
# NUMBA UNIT TESTS
# =============================================================================

def test_compute_tick_stats_numba():
    # Prices: [100, 101, 100, 102]
    # Returns: ln(101/100)~=0.01, ln(100/101)~=-0.01, ln(102/100)~=0.02
    prices = np.array([100., 101., 100., 102.], dtype=np.float64)
    start_idx = 0
    end_idx = 4

    var, rng, rv, mean_ret, count = _compute_tick_stats(prices, start_idx, end_idx)

    assert count == 4
    assert not np.isnan(var)
    assert not np.isnan(rng)
    assert not np.isnan(rv)

    # Check range ratio: (102 - 100) / 102 = 2/102 ~= 0.0196
    assert np.isclose(rng, 2.0/102.0)

def test_compute_tick_stats_insufficient_ticks():
    # Only 1 tick -> cant compute returns
    prices = np.array([100.], dtype=np.float64)
    var, rng, rv, mean_ret, count = _compute_tick_stats(prices, 0, 1)

    assert count == 1
    assert np.isnan(var)
    assert np.isnan(rng)

    # 2 ticks but one is nan or zero
    prices_bad = np.array([100., np.nan], dtype=np.float64)
    var, rng, rv, mean_ret, count = _compute_tick_stats(prices_bad, 0, 2)
    # returns will have 0 valid entries
    assert count == 2
    assert np.isnan(var)

def test_aggregate_tick_stats_by_bar_numba():
    # Ticks:
    # 100 at t=10
    # 101 at t=20
    # -- Bar 1 ends at 25 --
    # 102 at t=30
    # 100 at t=40
    # -- Bar 2 ends at 45 --

    tick_prices = np.array([100., 101., 102., 100.], dtype=np.float64)
    tick_ts = np.array([10, 20, 30, 40], dtype=np.int64)

    bar_open = np.array([0, 26], dtype=np.int64)
    bar_close = np.array([25, 45], dtype=np.int64)
    n_bars = 2

    ivar, trng, rv, mret, tcnt = _aggregate_tick_stats_by_bar(
        tick_prices, tick_ts, bar_open, bar_close, n_bars
    )

    # Bar 1: ticks at 10, 20. Prices 100, 101.
    # Return: ln(101/100). One return. Variance requires >=2 valid returns?
    # Logic in _compute_tick_stats: if valid_returns < 2: return nan
    # So Bar 1 should be NaN variance. Count = 2.
    assert np.isnan(ivar[0])
    assert tcnt[0] == 2

    # Bar 2: ticks at 30, 40. Prices 102, 100.
    # Same, only 2 ticks -> 1 return -> NaN variance.
    assert np.isnan(ivar[1])
    assert tcnt[1] == 2

def test_aggregate_tick_stats_complex():
    # Need 3 ticks to get 2 returns for variance
    tick_prices = np.array([100., 101., 102.], dtype=np.float64)
    tick_ts = np.array([10, 20, 30], dtype=np.int64)

    bar_open = np.array([0], dtype=np.int64)
    bar_close = np.array([50], dtype=np.int64)

    ivar, trng, rv, mret, tcnt = _aggregate_tick_stats_by_bar(
        tick_prices, tick_ts, bar_open, bar_close, 1
    )

    assert not np.isnan(ivar[0])
    assert tcnt[0] == 3

def test_rolling_std_numba():
    values = np.array([1., 2., 3., 4.], dtype=np.float64)
    window = 3
    # 0: NaN
    # 1: NaN
    # 2: std(1,2,3) = 1.0
    # 3: std(2,3,4) = 1.0
    res = _rolling_std_numba(values, window)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isclose(res[2], 1.0)
    assert np.isclose(res[3], 1.0)

    # With NaNs
    vals_nan = np.array([1., np.nan, 3., 4.], dtype=np.float64)
    # window=3 at idx 2: [1, nan, 3]. Valid: 1, 3. Count=2. Std exists.
    res_nan = _rolling_std_numba(vals_nan, window)
    assert not np.isnan(res_nan[2])

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def sample_ticks():
    timestamps = pd.date_range("2023-01-01 09:00", periods=100, freq="1s")
    prices = np.linspace(100, 110, 100)
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices
    })

@pytest.fixture
def sample_bars_micro(sample_ticks):
    # One bar covering all ticks
    return pd.DataFrame({
        "timestamp_open": [sample_ticks["timestamp"].iloc[0]],
        "timestamp_close": [sample_ticks["timestamp"].iloc[-1]],
        "high": [110.0],
        "low": [100.0],
        "close": [110.0]
    })

def test_compute_intrabar_volatility(sample_ticks, sample_bars_micro):
    df = compute_intrabar_volatility(
        sample_ticks, sample_bars_micro,
        price_col="price", timestamp_col="timestamp",
        bar_timestamp_open="timestamp_open", bar_timestamp_close="timestamp_close"
    )

    assert "intrabar_variance" in df.columns
    assert "tick_count" in df.columns
    assert df["tick_count"].iloc[0] == 100
    assert not np.isnan(df["intrabar_variance"].iloc[0])

def test_compute_microstructure_features(sample_ticks, sample_bars_micro):
    df = compute_microstructure_features(
        sample_ticks, sample_bars_micro,
        price_col="price", timestamp_col="timestamp",
        bar_timestamp_open="timestamp_open", bar_timestamp_close="timestamp_close",
        high_col="high", low_col="low"
    )

    assert "range_efficiency" in df.columns
    assert "vol_of_vol_20" in df.columns

    # Range efficiency: tick range should be close to OHLC range since we constructed it linearly
    # Tick range: (110-100)/110. OHLC range: (110-100)/110. Ratio should be ~1.
    assert np.isclose(df["range_efficiency"].iloc[0], 1.0)

def test_compute_microstructure_features_with_duration(sample_ticks, sample_bars_micro):
    sample_bars_micro["duration_sec"] = 100.0
    df = compute_microstructure_features(
        sample_ticks, sample_bars_micro,
        price_col="price", timestamp_col="timestamp",
        bar_timestamp_open="timestamp_open", bar_timestamp_close="timestamp_close"
    )
    assert "tick_intensity" in df.columns
    # 100 ticks / 100 sec = 1.0
    assert np.isclose(df["tick_intensity"].iloc[0], 1.0)

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
