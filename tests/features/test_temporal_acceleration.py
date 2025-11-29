
import pytest
import pandas as pd
import numpy as np
from src.features.temporal_acceleration import (
    compute_temporal_acceleration,
    compute_temporal_acceleration_smoothed,
    compute_temporal_jerk,
)

def test_compute_temporal_acceleration(sample_bars):
    s = compute_temporal_acceleration(sample_bars, duration_col="duration_sec")
    assert s.name == "temporal_acceleration"
    assert len(s) == len(sample_bars)
    # First value should be NaN (diff)
    assert np.isnan(s.iloc[0])

def test_compute_temporal_acceleration_smoothed(sample_bars):
    s = compute_temporal_acceleration_smoothed(sample_bars, duration_col="duration_sec", span=5)
    assert s.name == "temporal_acceleration_ema5"
    assert len(s) == len(sample_bars)

def test_compute_temporal_jerk(sample_bars):
    s = compute_temporal_jerk(sample_bars, duration_col="duration_sec", smoothed=False)
    assert s.name == "temporal_jerk"
    # First 2 values NaN
    assert np.isnan(s.iloc[0])
    assert np.isnan(s.iloc[1])

    s_smooth = compute_temporal_jerk(sample_bars, duration_col="duration_sec", smoothed=True, span=5)
    assert s_smooth.name == "temporal_jerk_ema5"
