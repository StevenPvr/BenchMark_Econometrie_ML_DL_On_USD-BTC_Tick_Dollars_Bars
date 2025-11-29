
import pytest
import pandas as pd
import numpy as np
from src.features.vpin import compute_vpin

def test_compute_vpin(sample_bars):
    s = compute_vpin(
        sample_bars,
        window=20,
        v_buy_col="buy_volume",
        v_sell_col="sell_volume"
    )

    assert isinstance(s, pd.Series)
    assert s.name == "vpin_20"
    assert len(s) == len(sample_bars)

    # Check range 0 to 1
    valid = s.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()

    # First 19 rows NaN
    assert s.iloc[:19].isna().all()
    assert not s.iloc[20:].isna().all()
