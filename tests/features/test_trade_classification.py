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

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
