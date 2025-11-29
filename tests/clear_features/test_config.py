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

from src.clear_features.config import (
    CORRELATION_CONFIG,
    PCA_CONFIG,
    LOG_TRANSFORM_CONFIG,
    META_COLUMNS,
    TARGET_COLUMN,
)

def test_config_constants():
    """Test that configuration constants are loaded correctly."""
    assert isinstance(META_COLUMNS, list)
    assert "bar_id" in META_COLUMNS
    assert isinstance(TARGET_COLUMN, str)
    assert TARGET_COLUMN == "log_return"

def test_correlation_config():
    """Test correlation configuration."""
    assert "method" in CORRELATION_CONFIG
    assert "threshold" in CORRELATION_CONFIG
    assert isinstance(CORRELATION_CONFIG["threshold"], float)
    assert 0 < CORRELATION_CONFIG["threshold"] < 1

def test_pca_config():
    """Test PCA configuration."""
    assert "variance_explained_threshold" in PCA_CONFIG
    assert isinstance(PCA_CONFIG["variance_explained_threshold"], float)

def test_log_transform_config():
    """Test log transform configuration."""
    assert "non_stationary_conclusions" in LOG_TRANSFORM_CONFIG
    assert isinstance(LOG_TRANSFORM_CONFIG["non_stationary_conclusions"], list)


if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
