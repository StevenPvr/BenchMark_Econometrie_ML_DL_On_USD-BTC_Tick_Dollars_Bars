
import pytest
import pandas as pd
import numpy as np
from src.features.trade_classification import (
    classify_trades_tick_rule,
    classify_trades_direct,
)

def test_classify_trades_tick_rule(sample_trades):
    s = classify_trades_tick_rule(sample_trades, price_col="price")
    assert isinstance(s, pd.Series)
    assert s.name == "trade_sign"
    assert s.isin([1, -1, 0]).all()

def test_classify_trades_direct():
    # Create specific data for direct classification
    df = pd.DataFrame({
        "side": ["buy", "sell", "buy", None],
        "price": [100, 101, 102, 103]
    })
    s = classify_trades_direct(df, side_col="side")
    assert s.iloc[0] == 1
    assert s.iloc[1] == -1
    assert s.iloc[2] == 1
    assert s.iloc[3] == 0 # fillna(0)
