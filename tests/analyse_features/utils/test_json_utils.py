import pytest
import numpy as np
import pandas as pd
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
