
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

def test_compute_parkinson_volatility(sample_bars):
    df = compute_parkinson_volatility(sample_bars)
    # Check for one of the windows
    assert "parkinson_vol_20" in df.columns
    assert not df["parkinson_vol_20"].iloc[20:].isna().all()

def test_compute_garman_klass_volatility(sample_bars):
    df = compute_garman_klass_volatility(sample_bars)
    assert "garman_klass_vol_20" in df.columns
    assert not df["garman_klass_vol_20"].iloc[20:].isna().all()

def test_compute_rogers_satchell_volatility(sample_bars):
    df = compute_rogers_satchell_volatility(sample_bars)
    assert "rogers_satchell_vol_20" in df.columns
    assert not df["rogers_satchell_vol_20"].iloc[20:].isna().all()

def test_compute_yang_zhang_volatility(sample_bars):
    df = compute_yang_zhang_volatility(sample_bars)
    assert "yang_zhang_vol_20" in df.columns
    assert not df["yang_zhang_vol_20"].iloc[20:].isna().all()

def test_compute_range_ratios(sample_bars):
    df = compute_range_ratios(sample_bars)
    assert "body_ratio" in df.columns
    assert "range_ratio" in df.columns

    # Check bounds
    assert (df["body_ratio"].dropna() >= 0).all()
