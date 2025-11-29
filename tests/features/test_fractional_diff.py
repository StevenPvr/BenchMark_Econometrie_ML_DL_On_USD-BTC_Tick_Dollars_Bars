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
from src.features.fractional_diff import compute_frac_diff_features

def test_compute_frac_diff_features(sample_bars):
    df = compute_frac_diff_features(sample_bars, price_col="close", d_values=[0.3, 0.5])

    assert "frac_diff_d03" in df.columns
    assert "frac_diff_d05" in df.columns
    assert "log_price" in df.columns

    # Check that it returns numeric values
    assert not df["frac_diff_d03"].iloc[50:].isna().all()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
