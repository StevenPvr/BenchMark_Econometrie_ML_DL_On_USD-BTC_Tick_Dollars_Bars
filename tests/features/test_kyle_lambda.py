
import pytest
import pandas as pd
import numpy as np
from src.features.kyle_lambda import compute_kyle_lambda

def test_compute_kyle_lambda(sample_bars):
    # Use small window to ensure we get results with 100 rows
    s = compute_kyle_lambda(
        sample_bars,
        window=20,
        price_col="close",
        v_buy_col="buy_volume",
        v_sell_col="sell_volume"
    )

    assert isinstance(s, pd.Series)
    assert s.name == "kyle_lambda_20"
    assert len(s) == len(sample_bars)

    # First 19 rows should be NaN
    assert s.iloc[:19].isna().all()
    # Others should have values (unless variance is 0)
    assert not s.iloc[20:].isna().all()
