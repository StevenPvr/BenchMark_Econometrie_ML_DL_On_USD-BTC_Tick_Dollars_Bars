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
from src.features.zscore_normalizer import (
    compute_rolling_zscore,
    compute_all_features_zscore,
    save_zscore_features,
    _rolling_zscore,
    _rolling_zscore_expanding,
)

# =============================================================================
# NUMBA UNIT TESTS
# =============================================================================

def test_rolling_zscore_numba():
    # Mean of [1, 2, 3] = 2. Std = 1.
    # 3rd point (3): (3 - 2)/1 = 1.
    values = np.array([1., 2., 3., 4., 5.], dtype=np.float64)
    window = 3
    min_periods = 3

    # 0: NaN
    # 1: NaN
    # 2: Mean 2, std 1. z = (3-2)/1 = 1.
    # 3: [2,3,4] -> Mean 3, std 1. z = (4-3)/1 = 1.
    res = _rolling_zscore(values, window, min_periods)
    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isclose(res[2], 1.0)
    assert np.isclose(res[3], 1.0)

    # Constant input -> std=0 -> z=0
    vals_const = np.array([1., 1., 1.], dtype=np.float64)
    res_const = _rolling_zscore(vals_const, 3, 3)
    assert res_const[2] == 0.0

def test_rolling_zscore_numba_with_nan():
    # [1, NaN, 3, 5]
    values = np.array([1., np.nan, 3., 5.], dtype=np.float64)
    window = 3
    min_periods = 2

    # 0: 1. Count=1 < 2 -> NaN
    # 1: [1, NaN]. Count=1 < 2 -> NaN
    # 2: [1, NaN, 3]. Valid=[1, 3]. Mean=2. Std=sqrt(((1-2)^2 + (3-2)^2)/1) = sqrt(2) ~= 1.414.
    # z = (3 - 2)/1.414 = 0.707
    res = _rolling_zscore(values, window, min_periods)
    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isclose(res[2], 1.0/np.sqrt(2))

def test_rolling_zscore_expanding_numba():
    values = np.array([1., 2., 3.], dtype=np.float64)
    min_periods = 2

    # 0: Count=1 < 2 -> NaN
    # 1: [1, 2]. Mean=1.5. Std=0.707. z=(2-1.5)/0.707 = 0.5/0.707 = 0.707
    # 2: [1, 2, 3]. Mean=2. Std=1. z=(3-2)/1 = 1.
    res = _rolling_zscore_expanding(values, min_periods)
    assert np.isnan(res[0])
    assert np.isclose(res[1], 1.0/np.sqrt(2))
    assert np.isclose(res[2], 1.0)

    # Expanding NaNs
    vals_nan = np.array([1., np.nan, 3.], dtype=np.float64)
    res_nan = _rolling_zscore_expanding(vals_nan, 2)
    # 0: NaN
    # 1: NaN (count=1)
    # 2: [1, 3]. Count=2. Mean=2. Std=1.414. z=(3-2)/1.414 = 0.707
    assert np.isclose(res_nan[2], 1.0/np.sqrt(2))

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def sample_features(sample_bars):
    df = sample_bars.copy()
    df["feature1"] = np.linspace(0, 10, len(df))
    df["feature2"] = np.random.randn(len(df))
    return df

def test_compute_rolling_zscore(sample_features):
    df = compute_rolling_zscore(sample_features, columns=["feature1", "feature2"], window=10)
    assert "feature1_zscore" in df.columns
    assert "feature2_zscore" in df.columns

    # Check values exist
    valid = df.dropna()
    assert len(valid) > 0

def test_compute_rolling_zscore_expanding(sample_features):
    df = compute_rolling_zscore(sample_features, columns=["feature1"], expanding=True)
    assert "feature1_zscore" in df.columns
    # With expanding, later values should be valid
    assert not np.isnan(df["feature1_zscore"].iloc[-1])

def test_compute_all_features_zscore(sample_features):
    # Should automatically pick feature1, feature2 and exclude timestamp/OHLC if configured logic used?
    # Logic: selects all numeric. Excludes default list.
    # OHLC are numeric, so they will be z-scored unless excluded explicitly or in default list.
    # default list has "index", "bar_id", etc. but not "close".

    df = compute_all_features_zscore(sample_features, window=10)

    assert "feature1_zscore" in df.columns
    assert "feature1" in df.columns # Keeps original
    assert "close_zscore" in df.columns # OHLC usually zscored too unless excluded

def test_save_zscore_features(sample_features, tmp_path):
    output_path = tmp_path / "features_zscore"

    df = save_zscore_features(sample_features, output_path, window=10, save_original=False)

    # Check parquet file created
    assert (tmp_path / "features_zscore.parquet").exists()

    # Check return df has zscores
    assert "feature1_zscore" in df.columns
    # And maybe timestamp if available

    # Test save_original=True
    df_orig = save_zscore_features(sample_features, output_path, window=10, save_original=True)
    assert "feature1" in df_orig.columns
    assert "feature1_zscore" in df_orig.columns

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
