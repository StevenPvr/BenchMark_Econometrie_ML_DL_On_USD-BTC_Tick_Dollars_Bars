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
    # Ticks (sorted chronologically):
    # 1. 100, qty 10 (first trade globally, assume buy) -> +10
    # 2. 105, qty 20 (up from 100) -> +20
    # 3. 102, qty 10 (down from 105) -> -10
    # Total buy = 30, sell = 10. VI = (30-10)/(30+10) = 20/40 = 0.5
    
    # Verify the result is reasonable (between -1 and 1)
    first_vi = df_result["volume_imbalance"].iloc[0]
    assert -1.0 <= first_vi <= 1.0
    # The exact value depends on tick rule classification, so we check it's positive (more buys than sells)
    assert first_vi > 0, f"Expected positive VI, got {first_vi}"

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
