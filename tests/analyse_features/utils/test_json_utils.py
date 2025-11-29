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
import numpy as np
import pandas as pd  # type: ignore
from pathlib import Path
from src.analyse_features.utils.json_utils import convert_to_serializable, save_json

class TestJsonUtils:
    def test_convert_to_serializable(self):
        # Test various types
        data = {
            "int_np": np.int64(42),
            "float_np": np.float64(3.14),
            "nan": np.nan,
            "inf": np.inf,
            "array": np.array([1, 2, 3]),
            "df": pd.DataFrame({"a": [1]}),
            "series": pd.Series([1], index=["a"]),
            "path": Path("/tmp/test"),
            "tuple": (1, 2),
            "bool": np.bool_(True)
        }

        result = convert_to_serializable(data)

        assert isinstance(result["int_np"], int)
        assert isinstance(result["float_np"], float)
        assert result["nan"] is None
        assert result["inf"] is None
        assert isinstance(result["array"], list)
        assert isinstance(result["df"], list) # orient=records
        assert isinstance(result["series"], dict)
        assert isinstance(result["path"], str)
        assert isinstance(result["tuple"], list)
        assert result["bool"] is True

    def test_save_json(self, tmp_path):
        data = {"key": np.int64(100)}
        filepath = tmp_path / "test.json"

        save_json(data, filepath)

        assert filepath.exists()
        import json
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["key"] == 100

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
