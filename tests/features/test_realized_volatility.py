from __future__ import annotations

import sys
from pathlib import Path
import warnings

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
    compute_return_volatility_ratio,
    compute_local_sharpe,  # Deprecated alias
    compute_realized_skewness,
    compute_realized_kurtosis,
)


def test_compute_realized_volatility(sample_bars):
    df = compute_realized_volatility(sample_bars, return_col="log_return")
    assert "realized_vol_20" in df.columns

    # Check that vol is non-negative
    assert (df["realized_vol_20"].dropna() >= 0).all()


def test_compute_return_volatility_ratio(sample_bars):
    """Test the new return-volatility ratio function."""
    df = compute_return_volatility_ratio(sample_bars, return_col="log_return")
    assert "return_vol_ratio_20" in df.columns

    # RVR can be negative, so just check it's numeric
    assert pd.api.types.is_float_dtype(df["return_vol_ratio_20"])


def test_compute_local_sharpe_deprecated(sample_bars):
    """Test that the deprecated alias still works but raises a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = compute_local_sharpe(sample_bars, return_col="log_return")

        # Should have triggered a deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Should still return valid data (column name changed in new function)
    assert "return_vol_ratio_20" in df.columns


def test_compute_realized_skewness(sample_bars):
    """Test realized skewness computation."""
    df = compute_realized_skewness(sample_bars, return_col="log_return")
    assert "realized_skew_20" in df.columns

    # Skewness can be any real number
    assert pd.api.types.is_float_dtype(df["realized_skew_20"])


def test_compute_realized_kurtosis(sample_bars):
    """Test realized kurtosis computation."""
    df = compute_realized_kurtosis(sample_bars, return_col="log_return")
    assert "realized_kurt_20" in df.columns

    # Excess kurtosis can be any real number (negative for platykurtic)
    assert pd.api.types.is_float_dtype(df["realized_kurt_20"])


if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
