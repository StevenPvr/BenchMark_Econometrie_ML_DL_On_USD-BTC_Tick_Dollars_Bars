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
from src.features.momentum import (
    compute_cumulative_returns,
    compute_recent_extremes,
)

def test_compute_cumulative_returns(sample_bars):
    """Test cumulative returns computation."""
    df = compute_cumulative_returns(sample_bars, return_col="log_return")

    # Based on error output, windows are 1, 3, 5, 10, 20
    assert "cum_return_20" in df.columns
    assert len(df) == len(sample_bars)

    # Check values
    expected_20 = sample_bars["log_return"].rolling(window=20).sum()
    pd.testing.assert_series_equal(
        df["cum_return_20"],
        expected_20,
        check_names=False
    )

def test_compute_recent_extremes(sample_bars):
    """Test recent extremes computation."""
    df = compute_recent_extremes(sample_bars, return_col="log_return")

    # Based on error output, windows are 5, 10, 20
    assert "max_return_20" in df.columns
    assert "min_return_20" in df.columns

    # Check max logic
    expected_max = sample_bars["log_return"].rolling(window=20).max()
    pd.testing.assert_series_equal(
        df["max_return_20"],
        expected_max,
        check_names=False
    )

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
