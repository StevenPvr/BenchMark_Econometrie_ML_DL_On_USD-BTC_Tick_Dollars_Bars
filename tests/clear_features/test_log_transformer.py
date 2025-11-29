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
from unittest.mock import MagicMock, patch
from src.clear_features.log_transformer import LogTransformer

@pytest.fixture
def log_df():
    return pd.DataFrame({
        "feat_pos": [1.0, 10.0, 100.0],
        "feat_zero": [0.0, 1.0, 10.0],
        "feat_neg": [-10.0, 0.0, 10.0],
        "feat_stationary": [1, 2, 3] # Should be skipped
    })

@pytest.fixture
def mock_stationarity_data():
    return {
        "all_results": [
            {"feature": "feat_pos", "stationarity_conclusion": "non_stationary"},
            {"feature": "feat_zero", "stationarity_conclusion": "trend_stationary"},
            {"feature": "feat_neg", "stationarity_conclusion": "non_stationary"},
            {"feature": "feat_stationary", "stationarity_conclusion": "stationary"},
        ]
    }

def test_identify_features(log_df, mock_stationarity_data):
    transformer = LogTransformer()
    transformer._stationarity_data = mock_stationarity_data

    features = transformer.identify_non_stationary_features(log_df)

    assert "feat_pos" in features
    assert "feat_zero" in features
    assert "feat_neg" in features
    assert "feat_stationary" not in features

def test_fit_transform_types(log_df, mock_stationarity_data):
    transformer = LogTransformer()
    transformer._stationarity_data = mock_stationarity_data

    transformer.fit(log_df)

    params = transformer._transform_params

    # Positive feature -> log or log1p
    assert params["feat_pos"]["transform_type"] in ["log", "log1p"]

    # Zero containing -> log1p (with possible shift if 0 is min)
    assert params["feat_zero"]["transform_type"] == "log1p"

    # Negative containing -> signed_log1p
    assert params["feat_neg"]["transform_type"] == "signed_log1p"

def test_transform_values(log_df, mock_stationarity_data):
    transformer = LogTransformer()
    transformer._stationarity_data = mock_stationarity_data
    transformer.fit(log_df)

    transformed = transformer.transform(log_df)

    # Check feat_pos (log1p by default)
    # log1p(1) = 0.693, log1p(100) = 4.615
    assert np.allclose(transformed["feat_pos"], np.log1p(log_df["feat_pos"]))

    # Check feat_neg (signed log)
    # sign(-10) * log1p(10) = -2.39
    expected_neg = np.sign(log_df["feat_neg"]) * np.log1p(np.abs(log_df["feat_neg"]))
    assert np.allclose(transformed["feat_neg"], expected_neg)

def test_save_load(log_df, mock_stationarity_data, tmp_path):
    transformer = LogTransformer()
    transformer._stationarity_data = mock_stationarity_data
    transformer.fit(log_df)

    transformer.save_artifacts(output_dir=tmp_path)

    assert (tmp_path / "log_transform_params.joblib").exists()

    new_transformer = LogTransformer()
    new_transformer.load_artifacts(input_dir=tmp_path)

    assert new_transformer._transform_params == transformer._transform_params

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
