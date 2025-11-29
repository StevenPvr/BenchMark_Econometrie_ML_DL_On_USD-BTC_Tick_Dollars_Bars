from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
# Find project root by looking for .git, pyproject.toml, or setup.py
_project_root = _script_dir
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest  # type: ignore
import pandas as pd  # type: ignore
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

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
