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
from src.features.lag_generator import generate_all_lags

def test_generate_all_lags(sample_bars):
    n = len(sample_bars)
    # Create a simple df with features to lag
    df = pd.DataFrame({
        "feature_a": np.arange(n),
        "feature_b": np.random.randn(n),
        "log_return": np.random.randn(n)
    }, index=sample_bars.index)

    df_lagged = generate_all_lags(df, include_original=True)

    # Check for lag columns
    # Depends on default config in lag_generator, but typically lags 1..3
    assert "feature_a" in df_lagged.columns
    assert "feature_a_lag1" in df_lagged.columns
    assert "log_return_lag1" in df_lagged.columns

    # Check shift logic
    pd.testing.assert_series_equal(
        df_lagged["feature_a_lag1"],
        df["feature_a"].shift(1).rename("feature_a_lag1"),
        check_names=True
    )

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
