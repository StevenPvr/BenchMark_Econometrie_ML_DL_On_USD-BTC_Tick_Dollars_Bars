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
from src.features.scalers import (
    StandardScalerCustom,
    MinMaxScalerCustom,
    fit_and_transform_features,
)
import joblib

def test_StandardScalerCustom():
    # Create simple data
    data = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50]
    })

    scaler = StandardScalerCustom()
    scaler.fit(data, ["A", "B"])

    transformed = scaler.transform(data)

    # Check A: mean=3, std=1.58
    # 1 -> (1-3)/1.58 = -1.26
    # 5 -> (5-3)/1.58 = 1.26
    assert np.isclose(transformed["A"].mean(), 0)
    assert np.isclose(transformed["A"].std(), 1)

    # Inverse transform
    inversed = scaler.inverse_transform(transformed)
    pd.testing.assert_frame_equal(data.astype(float), inversed)

def test_MinMaxScalerCustom():
    data = pd.DataFrame({
        "A": [1, 2, 3, 4, 5], # min 1, max 5 -> range 4
    })

    scaler = MinMaxScalerCustom()
    scaler.fit(data, ["A"])

    transformed = scaler.transform(data)

    # Check ranges [-1, 1]
    assert transformed["A"].min() == -1.0
    assert transformed["A"].max() == 1.0

    # 3 should be 0 (midpoint)
    assert transformed["A"].iloc[2] == 0.0

def test_fit_and_transform_features(tmp_path):
    df_train = pd.DataFrame({"A": [1, 2, 3]})
    df_test = pd.DataFrame({"A": [4, 5]})

    train_scaled, test_scaled, scaler = fit_and_transform_features(
        df_train, df_test, ["A"], scaler_type="minmax", scaler_path=tmp_path / "scaler.joblib"
    )

    assert (train_scaled["A"].min() == -1.0)
    assert (train_scaled["A"].max() == 1.0)

    # Check save
    assert (tmp_path / "scaler.joblib").exists()

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
