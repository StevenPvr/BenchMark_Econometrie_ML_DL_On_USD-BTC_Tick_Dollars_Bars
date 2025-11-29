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
from src.features.range_volatility import (
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_rogers_satchell_volatility,
    compute_yang_zhang_volatility,
    compute_range_ratios,
)

def test_compute_parkinson_volatility(sample_bars):
    df = compute_parkinson_volatility(sample_bars)
    # Check for one of the windows
    assert "parkinson_vol_20" in df.columns
    assert not df["parkinson_vol_20"].iloc[20:].isna().all()

def test_compute_garman_klass_volatility(sample_bars):
    df = compute_garman_klass_volatility(sample_bars)
    assert "garman_klass_vol_20" in df.columns
    assert not df["garman_klass_vol_20"].iloc[20:].isna().all()

def test_compute_rogers_satchell_volatility(sample_bars):
    df = compute_rogers_satchell_volatility(sample_bars)
    assert "rogers_satchell_vol_20" in df.columns
    assert not df["rogers_satchell_vol_20"].iloc[20:].isna().all()

def test_compute_yang_zhang_volatility(sample_bars):
    df = compute_yang_zhang_volatility(sample_bars)
    assert "yang_zhang_vol_20" in df.columns
    assert not df["yang_zhang_vol_20"].iloc[20:].isna().all()

def test_compute_range_ratios(sample_bars):
    df = compute_range_ratios(sample_bars)
    assert "body_ratio" in df.columns
    assert "range_ratio" in df.columns

    # Check bounds
    assert (df["body_ratio"].dropna() >= 0).all()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
