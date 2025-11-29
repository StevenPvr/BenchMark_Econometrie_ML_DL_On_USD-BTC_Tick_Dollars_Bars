
import pytest
import pandas as pd
import numpy as np
from src.features.temporal_calendar import (
    compute_cyclical_time_features,
    compute_time_since_shock,
    compute_volatility_regime,
    compute_drawdown_features,
    compute_all_temporal_features,
)

def test_compute_cyclical_time_features(sample_bars):
    df = compute_cyclical_time_features(sample_bars, timestamp_col="datetime")

    assert "hour_sin" in df.columns
    assert "hour_cos" in df.columns
    assert "day_of_week" in df.columns

    # Check values range
    assert df["hour_sin"].between(-1, 1).all()

def test_compute_time_since_shock(sample_bars):
    df = compute_time_since_shock(sample_bars, return_col="log_return", thresholds=[0.01])

    assert "bars_since_shock_0.01" in df.columns
    assert "bars_since_pos_shock_0.01" in df.columns
    assert "bars_since_neg_shock_0.01" in df.columns

    # Check that it increments
    # If no shock, should increment by 1
    # We can't guarantee shock locations in random data, but we can check it's monotonic increasing until reset
    valid = df["bars_since_shock_0.01"].dropna()
    if len(valid) > 1:
        diffs = valid.diff().dropna()
        # diff is either 1 (no shock) or -(prev_value) (shock reset)
        # So diff <= 1
        assert (diffs <= 1).all()

def test_compute_volatility_regime(sample_bars):
    df = compute_volatility_regime(sample_bars, return_col="log_return", vol_window=10)

    assert "volatility_10" in df.columns
    # Quantiles default: 0.75, 0.90, 0.95 -> 75, 90, 95
    assert "vol_regime_q75" in df.columns
    assert "vol_zscore" in df.columns

def test_compute_drawdown_features(sample_bars):
    df = compute_drawdown_features(sample_bars, price_col="close")

    assert "drawdown" in df.columns
    assert "rolling_max" in df.columns
    assert "bars_since_high" in df.columns

    # Drawdown should be <= 0
    assert (df["drawdown"].dropna() <= 0).all()

def test_compute_all_temporal_features(sample_bars):
    df = compute_all_temporal_features(sample_bars, timestamp_col="datetime", return_col="log_return", price_col="close")
    assert "hour_sin" in df.columns
    assert "drawdown" in df.columns
