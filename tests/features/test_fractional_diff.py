
import pytest
import pandas as pd
import numpy as np
from src.features.fractional_diff import compute_frac_diff_features

def test_compute_frac_diff_features(sample_bars):
    df = compute_frac_diff_features(sample_bars, price_col="close", d_values=[0.3, 0.5])

    assert "frac_diff_d03" in df.columns
    assert "frac_diff_d05" in df.columns
    assert "log_price" in df.columns

    # Check that it returns numeric values
    assert not df["frac_diff_d03"].iloc[50:].isna().all()
