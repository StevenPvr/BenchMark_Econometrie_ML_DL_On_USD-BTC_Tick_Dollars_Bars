
import pytest
import pandas as pd
import numpy as np
from src.features.trend import (
    compute_moving_averages,
    compute_price_zscore,
    compute_cross_ma,
    compute_return_streak,
)

def test_compute_moving_averages(sample_bars):
    df = compute_moving_averages(sample_bars)
    # Default windows: 5, 10, 20, 50, 100
    assert "ma_5" in df.columns
    assert "ma_100" in df.columns

    # Check values
    expected = sample_bars["close"].rolling(5).mean()
    pd.testing.assert_series_equal(
        df["ma_5"],
        expected.rename("ma_5"),
        check_names=True
    )

def test_compute_price_zscore(sample_bars):
    df = compute_price_zscore(sample_bars)
    # Default windows: 10, 20, 50
    assert "price_zscore_10" in df.columns
    assert "price_zscore_50" in df.columns

def test_compute_cross_ma(sample_bars):
    df = compute_cross_ma(sample_bars)
    # Default pairs: (5,20), (10,50), (20,100)
    assert "cross_ma_5_20" in df.columns
    assert "cross_ma_20_100" in df.columns

    # Values should be -1, 0, 1 (or NaN)
    valid = df["cross_ma_5_20"].dropna()
    assert valid.isin([-1, 0, 1]).all()

def test_compute_return_streak(sample_bars):
    s = compute_return_streak(sample_bars)
    assert s.name == "return_streak"
    assert len(s) == len(sample_bars)

    # Check logic manually
    # + + - -> 1, 2, -1
    returns = pd.Series([0.01, 0.01, -0.01, -0.01, -0.01, 0], name="log_return")
    df_mini = pd.DataFrame({"log_return": returns})
    streak = compute_return_streak(df_mini)

    assert streak.iloc[0] == 1.0
    assert streak.iloc[1] == 2.0
    assert streak.iloc[2] == -1.0
    assert streak.iloc[3] == -2.0
    assert streak.iloc[4] == -3.0
    assert streak.iloc[5] == 0.0
