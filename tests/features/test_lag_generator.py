
import pytest
import pandas as pd
import numpy as np
from src.features.lag_generator import generate_all_lags

def test_generate_all_lags(sample_bars):
    n = len(sample_bars)
    # Create a simple df with features to lag
    df = pd.DataFrame({
        "feature_a": np.arange(n),
        "feature_b": np.random.randn(n),
        "log_return": np.random.randn(n)
    }, index=sample_bars.index)

    df_lagged = generate_all_lags(df, include_original=True)

    # Check for lag columns
    # Depends on default config in lag_generator, but typically lags 1..3
    assert "feature_a" in df_lagged.columns
    assert "feature_a_lag1" in df_lagged.columns
    assert "log_return_lag1" in df_lagged.columns

    # Check shift logic
    pd.testing.assert_series_equal(
        df_lagged["feature_a_lag1"],
        df["feature_a"].shift(1).rename("feature_a_lag1"),
        check_names=True
    )
