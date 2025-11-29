
import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import (
    compute_all_technical_indicators,
    compute_momentum_indicators,
    compute_volatility_indicators,
    compute_trend_indicators,
    compute_volume_indicators,
)

def test_compute_all_technical_indicators(sample_bars):
    df = compute_all_technical_indicators(sample_bars, fillna=True)

    # Check prefixes
    assert all(col.startswith("ta_") for col in df.columns)

    # Check for some expected columns (ta library uses prefixes like momentum_, volatility_)
    # And we add ta_ prefix, so ta_momentum_rsi
    assert "ta_momentum_rsi" in df.columns
    assert "ta_volatility_bbh" in df.columns # bb high
    assert "ta_trend_sma_fast" in df.columns # defaults might differ but these are standard ta names
    assert "ta_volume_obv" in df.columns

def test_compute_momentum_indicators(sample_bars):
    df = compute_momentum_indicators(sample_bars, fillna=True)
    assert "ta_rsi_14" in df.columns
    assert "ta_macd" in df.columns

def test_compute_volatility_indicators(sample_bars):
    df = compute_volatility_indicators(sample_bars, fillna=True)
    assert "ta_bb_upper" in df.columns
    assert "ta_atr_14" in df.columns

def test_compute_trend_indicators(sample_bars):
    df = compute_trend_indicators(sample_bars, fillna=True)
    assert "ta_sma_20" in df.columns
    assert "ta_adx" in df.columns

def test_compute_volume_indicators(sample_bars):
    df = compute_volume_indicators(sample_bars, fillna=True)
    assert "ta_obv" in df.columns
    assert "ta_vwap" in df.columns
