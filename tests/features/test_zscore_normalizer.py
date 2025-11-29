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
from src.features.zscore_normalizer import compute_rolling_zscore

def test_compute_rolling_zscore(sample_bars):
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    # window=4
    # 1,2,3,4 -> mean=2.5, std=1.29 -> 4 -> (4-2.5)/1.29 = 1.16
    df_z = compute_rolling_zscore(df, window=4, min_periods=4)

    assert "A_zscore" in df_z.columns
    # First 3 should be NaN
    assert df_z["A_zscore"].iloc[:3].isna().all()
    # 4th should be valid
    assert not np.isnan(df_z["A_zscore"].iloc[3])

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
