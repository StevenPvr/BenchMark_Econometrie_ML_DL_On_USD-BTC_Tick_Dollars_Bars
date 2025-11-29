import pytest
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Import constant to ensure consistency
from src.analyse_features.config import TARGET_COLUMN

@pytest.fixture
def sample_df():
    """Create a simple random DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, n_rows),
        "feature_2": np.random.normal(0, 1, n_rows),
        "feature_3": np.random.uniform(0, 1, n_rows),
        TARGET_COLUMN: np.random.normal(0, 1, n_rows)
    })
    return df

@pytest.fixture
def correlated_df():
    """Create a DataFrame with correlated features."""
    np.random.seed(42)
    n_rows = 100
    x = np.random.normal(0, 1, n_rows)
    df = pd.DataFrame({
        "feat_a": x,
        "feat_b": x * 0.9 + np.random.normal(0, 0.1, n_rows), # Highly correlated
        "feat_c": np.random.normal(0, 1, n_rows),             # Uncorrelated
        TARGET_COLUMN: x * 0.5 + np.random.normal(0, 0.5, n_rows)
    })
    return df

@pytest.fixture
def non_stationary_df():
    """Create a DataFrame with non-stationary features."""
    np.random.seed(42)
    n_rows = 200
    # Random walk
    rw = np.cumsum(np.random.normal(0, 1, n_rows))
    df = pd.DataFrame({
        "stationary": np.random.normal(0, 1, n_rows),
        "non_stationary": rw,
        TARGET_COLUMN: np.random.normal(0, 1, n_rows)
    })
    return df

@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to provide a temporary directory."""
    return tmp_path
