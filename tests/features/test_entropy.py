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
from src.features.entropy import (
    compute_shannon_entropy,
    compute_approximate_entropy,
    compute_sample_entropy,
)

def test_compute_shannon_entropy(sample_bars):
    df = compute_shannon_entropy(sample_bars, return_col="log_return")
    # Verify return type is Series or DataFrame
    assert isinstance(df, (pd.Series, pd.DataFrame))
    # Check name if it's a series
    if isinstance(df, pd.Series):
        assert "shannon" in df.name
    else:
        assert "shannon_entropy_100" in df.columns # window default might vary

def test_compute_approximate_entropy(sample_bars):
    s = compute_approximate_entropy(sample_bars, return_col="log_return", window=50)
    assert isinstance(s, pd.Series)
    assert s.name == "apen_50"
    assert not s.iloc[50:].isna().all()

def test_compute_sample_entropy(sample_bars):
    s = compute_sample_entropy(sample_bars, return_col="log_return", window=50)
    assert isinstance(s, pd.Series)
    assert s.name == "sampen_50"
    assert not s.iloc[50:].isna().all()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
