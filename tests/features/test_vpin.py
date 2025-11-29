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
from src.features.vpin import compute_vpin

def test_compute_vpin(sample_bars):
    s = compute_vpin(
        sample_bars,
        window=20,
        v_buy_col="buy_volume",
        v_sell_col="sell_volume"
    )

    assert isinstance(s, pd.Series)
    assert s.name == "vpin_20"
    assert len(s) == len(sample_bars)

    # Check range 0 to 1
    valid = s.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()

    # First 19 rows NaN
    assert s.iloc[:19].isna().all()
    assert not s.iloc[20:].isna().all()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
