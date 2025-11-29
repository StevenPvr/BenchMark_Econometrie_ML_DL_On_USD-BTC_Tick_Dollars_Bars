
import pytest
import pandas as pd
import numpy as np
from src.features.volume_imbalance import (
    compute_volume_imbalance,
    compute_volume_imbalance_bars,
)

def test_compute_volume_imbalance(aligned_data):
    df_ticks, _ = aligned_data
    # We use "quantity" as volume_col
    result = compute_volume_imbalance(
        df_ticks,
        volume_col="quantity",
        price_col="price",
        use_tick_rule=True
    )

    assert "volume_imbalance" in result
    assert "v_buy" in result
    assert "v_sell" in result
    assert -1.0 <= result["volume_imbalance"] <= 1.0

def test_compute_volume_imbalance_bars(aligned_data):
    df_ticks, df_bars = aligned_data

    df_result = compute_volume_imbalance_bars(
        df_ticks,
        df_bars,
        volume_col="quantity",
        price_col="price",
        timestamp_col="timestamp",
        bar_timestamp_open="timestamp_open",
        bar_timestamp_close="timestamp_close",
        use_tick_rule=True
    )

    assert "volume_imbalance" in df_result.columns
    assert "v_buy" in df_result.columns
    assert "v_sell" in df_result.columns
    assert len(df_result) == len(df_bars)

    # Check values for first bar
    # Ticks:
    # 1. 100, qty 10 (first trade, assume buy) -> +10
    # 2. 105, qty 20 (up) -> +20
    # 3. 102, qty 10 (down) -> -10
    # Total buy = 30, sell = 10. VI = (30-10)/(30+10) = 20/40 = 0.5

    assert np.isclose(df_result["volume_imbalance"].iloc[0], 0.5)
