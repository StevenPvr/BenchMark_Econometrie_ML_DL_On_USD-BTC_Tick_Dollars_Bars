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
from src.features.microstructure_volatility import (
    compute_intrabar_volatility,
    compute_microstructure_features,
)

def test_compute_intrabar_volatility(aligned_data):
    df_ticks, df_bars = aligned_data

    df_result = compute_intrabar_volatility(
        df_ticks,
        df_bars,
        price_col="price",
        timestamp_col="timestamp",
        bar_timestamp_open="timestamp_open",
        bar_timestamp_close="timestamp_close"
    )

    assert len(df_result) == 2
    assert "intrabar_variance" in df_result.columns
    assert "tick_count" in df_result.columns

    # Check tick count
    assert df_result["tick_count"].iloc[0] == 3
    assert df_result["tick_count"].iloc[1] == 3

def test_compute_microstructure_features(aligned_data):
    df_ticks, df_bars = aligned_data

    df_result = compute_microstructure_features(
        df_ticks,
        df_bars,
        price_col="price",
        timestamp_col="timestamp",
        bar_timestamp_open="timestamp_open",
        bar_timestamp_close="timestamp_close",
        high_col="high",
        low_col="low"
    )

    assert "range_efficiency" in df_result.columns
    assert "vol_of_vol_20" in df_result.columns
    assert "tick_intensity" in df_result.columns

    # Test values
    # tick intensity = count / duration = 3 / 100 = 0.03
    assert np.isclose(df_result["tick_intensity"].iloc[0], 0.03)

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
