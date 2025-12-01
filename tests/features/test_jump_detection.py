"""Tests for jump detection features (bipower variation, jump component)."""
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
from src.features.jump_detection import (
    compute_bipower_variation,
    compute_jump_component,
    compute_all_jump_features,
)


def test_compute_bipower_variation(sample_bars):
    """Test bipower variation computation."""
    df = compute_bipower_variation(sample_bars, return_col="log_return")
    assert "bipower_var_20" in df.columns

    # Bipower variation should be non-negative
    valid_bv = df["bipower_var_20"].dropna()
    assert (valid_bv >= 0).all(), "Bipower variation should be non-negative"


def test_compute_jump_component(sample_bars):
    """Test jump component computation."""
    df = compute_jump_component(sample_bars, return_col="log_return")

    assert "jump_20" in df.columns
    assert "jump_ratio_20" in df.columns

    # Jump should be non-negative (by construction: max(RV - BV, 0))
    valid_jump = df["jump_20"].dropna()
    assert (valid_jump >= 0).all(), "Jump component should be non-negative"

    # Jump ratio should be between 0 and 1
    valid_ratio = df["jump_ratio_20"].dropna()
    assert (valid_ratio >= 0).all(), "Jump ratio should be >= 0"
    assert (valid_ratio <= 1).all(), "Jump ratio should be <= 1"


def test_compute_all_jump_features(sample_bars):
    """Test all jump features are computed."""
    df = compute_all_jump_features(sample_bars, return_col="log_return")

    # Should have bipower, jump, and ratio columns
    expected_cols = ["bipower_var_20", "jump_20", "jump_ratio_20"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_bipower_less_than_realized_variance(sample_bars):
    """Test that bipower variation <= realized variance (theoretically).

    In the presence of jumps, RV captures total variance while BV captures
    only the continuous component, so BV <= RV.
    """
    # Compute both
    returns = sample_bars["log_return"].values.astype(np.float64)
    n = len(returns)
    window = 20

    # Simple realized variance
    rv = pd.Series(returns).rolling(window=window).apply(lambda x: np.sum(x**2), raw=True)

    # Bipower variation
    df_bv = compute_bipower_variation(sample_bars, return_col="log_return", horizons=[window])
    bv = df_bv[f"bipower_var_{window}"]

    # For most indices, BV should be <= RV (allowing some numerical tolerance)
    # Note: Due to finite sample effects, this may not hold exactly for all windows
    valid_idx = (~rv.isna()) & (~bv.isna())
    if valid_idx.sum() > 0:
        # Allow small numerical tolerance
        tolerance = 1e-10
        comparison = (bv[valid_idx] <= rv[valid_idx] + tolerance)
        # At least 95% should satisfy the inequality
        assert comparison.mean() >= 0.95, \
            "Bipower variation should be <= realized variance for most observations"


def test_jump_with_artificial_jump(sample_bars):
    """Test that jump detection finds an artificial jump."""
    # Create data with an artificial large return (jump)
    df = sample_bars.copy()
    # Insert a large return (jump) in the middle
    mid_idx = len(df) // 2
    df.iloc[mid_idx, df.columns.get_loc("log_return")] = 0.1  # 10% return = jump

    # Compute jump features
    df_jump = compute_jump_component(df, return_col="log_return", horizons=[10])

    # The jump component should be elevated around the jump
    # (not exact due to rolling window)
    jump_values = df_jump["jump_10"].dropna()
    if len(jump_values) > 0:
        max_jump = jump_values.max()
        assert max_jump > 0, "Should detect some jump component"


if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
