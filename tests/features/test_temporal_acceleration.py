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
from src.features.temporal_acceleration import (
    compute_temporal_acceleration,
    compute_temporal_acceleration_smoothed,
    compute_temporal_jerk,
)

@pytest.fixture
def sample_bars_with_duration(sample_bars):
    df = sample_bars.copy()
    # Add duration_sec column
    df["duration_sec"] = np.random.uniform(10, 100, len(df))
    return df

def test_compute_temporal_acceleration(sample_bars_with_duration):
    s = compute_temporal_acceleration(sample_bars_with_duration, duration_col="duration_sec")
    assert s.name == "temporal_acceleration"
    assert len(s) == len(sample_bars_with_duration)
    # First value should be NaN (diff)
    assert np.isnan(s.iloc[0])

def test_compute_temporal_acceleration_smoothed(sample_bars_with_duration):
    s = compute_temporal_acceleration_smoothed(sample_bars_with_duration, duration_col="duration_sec", span=5)
    assert s.name == "temporal_acceleration_ema5"
    assert len(s) == len(sample_bars_with_duration)

def test_compute_temporal_jerk(sample_bars_with_duration):
    s = compute_temporal_jerk(sample_bars_with_duration, duration_col="duration_sec", smoothed=False)
    assert s.name == "temporal_jerk"
    # First 2 values NaN
    assert np.isnan(s.iloc[0])
    assert np.isnan(s.iloc[1])

    s_smooth = compute_temporal_jerk(sample_bars_with_duration, duration_col="duration_sec", smoothed=True, span=5)
    assert s_smooth.name == "temporal_jerk_ema5"

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
