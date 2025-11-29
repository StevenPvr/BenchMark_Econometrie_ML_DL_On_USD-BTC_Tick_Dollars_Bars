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
