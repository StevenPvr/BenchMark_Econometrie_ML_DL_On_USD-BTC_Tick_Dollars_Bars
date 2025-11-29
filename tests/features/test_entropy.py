
import pytest
import pandas as pd
import numpy as np
from src.features.entropy import (
    compute_shannon_entropy,
    compute_approximate_entropy,
    compute_sample_entropy,
)

def test_compute_shannon_entropy(sample_bars):
    df = compute_shannon_entropy(sample_bars, return_col="log_return")
    # Verify return type is Series or DataFrame
    assert isinstance(df, (pd.Series, pd.DataFrame))
    # Check name if it's a series
    if isinstance(df, pd.Series):
        assert "shannon" in df.name
    else:
        assert "shannon_entropy_100" in df.columns # window default might vary

def test_compute_approximate_entropy(sample_bars):
    s = compute_approximate_entropy(sample_bars, return_col="log_return", window=50)
    assert isinstance(s, pd.Series)
    assert s.name == "apen_50"
    assert not s.iloc[50:].isna().all()

def test_compute_sample_entropy(sample_bars):
    s = compute_sample_entropy(sample_bars, return_col="log_return", window=50)
    assert isinstance(s, pd.Series)
    assert s.name == "sampen_50"
    assert not s.iloc[50:].isna().all()
