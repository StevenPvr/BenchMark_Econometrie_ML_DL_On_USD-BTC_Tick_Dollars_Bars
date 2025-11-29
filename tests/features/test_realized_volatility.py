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
from src.features.realized_volatility import (
    compute_realized_volatility,
    compute_local_sharpe,
)

def test_compute_realized_volatility(sample_bars):
    df = compute_realized_volatility(sample_bars, return_col="log_return")
    assert "realized_vol_20" in df.columns

    # Check that vol is non-negative
    assert (df["realized_vol_20"].dropna() >= 0).all()

def test_compute_local_sharpe(sample_bars):
    df = compute_local_sharpe(sample_bars, return_col="log_return")
    assert "local_sharpe_20" in df.columns

    # Sharpe can be negative, so just check it's numeric
    assert pd.api.types.is_float_dtype(df["local_sharpe_20"])

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
