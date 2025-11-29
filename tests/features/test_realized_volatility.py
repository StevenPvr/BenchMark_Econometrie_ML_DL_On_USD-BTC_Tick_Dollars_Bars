
import pytest
import pandas as pd
import numpy as np
from src.features.realized_volatility import (
    compute_realized_volatility,
    compute_local_sharpe,
)

def test_compute_realized_volatility(sample_bars):
    df = compute_realized_volatility(sample_bars, return_col="log_return")
    assert "realized_vol_20" in df.columns

    # Check that vol is non-negative
    assert (df["realized_vol_20"].dropna() >= 0).all()

def test_compute_local_sharpe(sample_bars):
    df = compute_local_sharpe(sample_bars, return_col="log_return")
    assert "local_sharpe_20" in df.columns

    # Sharpe can be negative, so just check it's numeric
    assert pd.api.types.is_float_dtype(df["local_sharpe_20"])
