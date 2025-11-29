"""
Unit tests for label_primaire/train.py

Tests the training module, mocking external dependencies.
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path

from src.labelling.label_primaire.train import (
    train_model,
    TrainingConfig,
    TrainingResult,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Create a mock feature dataset."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "feature_1": range(100),
        "feature_2": range(100, 200),
        "split": ["train"] * 80 + ["test"] * 20,
    }, index=dates)
    # Add datetime_close column for alignment
    df["datetime_close"] = dates
    return df


@pytest.fixture
def mock_dollar_bars() -> pd.DataFrame:
    """Create mock dollar bars."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "close": [100.0 + i for i in range(100)],
        "datetime_close": dates,
    })
    return df


@pytest.fixture
def mock_events() -> pd.DataFrame:
    """Create mock events DataFrame."""
    dates = pd.date_range("2024-01-01", periods=50, freq="2h") # subset of timestamps
    df = pd.DataFrame({
        "label": [1, -1, 0, 1, 0] * 10,
        "t1": dates + pd.Timedelta(hours=1),
        "trgt": 0.01,
        "ret": 0.02,
    }, index=dates)
    return df


@pytest.fixture
def mock_optimized_params():
    """Create mock optimized parameters."""
    return {
        "model_params": {"n_estimators": 10},
        "triple_barrier_params": {
            "pt_mult": 1.0,
            "sl_mult": 1.0,
            "max_holding": 10
        },
        "best_score": 0.5,
        "metric": "mcc",
    }


# =============================================================================
# TESTS
# =============================================================================


@patch("src.labelling.label_primaire.train.load_optimized_params")
@patch("src.labelling.label_primaire.train.load_model_class")
@patch("src.labelling.label_primaire.train.get_dataset_for_model")
@patch("src.labelling.label_primaire.train.pd.read_parquet")
@patch("src.labelling.label_primaire.train.get_daily_volatility")
@patch("src.labelling.label_primaire.train.get_events_primary")
def test_train_model(
    mock_get_events,
    mock_get_vol,
    mock_read_parquet,
    mock_get_dataset,
    mock_load_model_class,
    mock_load_params,
    mock_dataset,
    mock_dollar_bars,
    mock_events,
    mock_optimized_params,
    tmp_path,
):
    """Test train_model function with mocked dependencies."""

    # Setup mocks
    mock_load_params.return_value = mock_optimized_params

    # Mock model class
    mock_model_instance = MagicMock()
    mock_model_class = MagicMock(return_value=mock_model_instance)
    mock_load_model_class.return_value = mock_model_class

    # Mock data loading
    mock_get_dataset.return_value = mock_dataset

    # Mock pd.read_parquet for dollar bars (using side_effect to distinguish calls if needed,
    # but here we just need it to return dollar bars when called)
    # Note: train.py checks DOLLAR_BARS_PARQUET.exists()
    # We should also patch Path.exists but let's assume the code under test handles mocks gracefully
    # Actually, we need to patch DOLLAR_BARS_PARQUET in train.py to avoid FileNotFoundError

    mock_read_parquet.return_value = mock_dollar_bars

    # Mock volatility
    mock_get_vol.return_value = pd.Series(0.01, index=mock_dollar_bars.index)

    # Mock events generation
    mock_get_events.return_value = mock_events

    # Run training
    with patch("src.labelling.label_primaire.train.DOLLAR_BARS_PARQUET") as mock_path_const:
        mock_path_const.exists.return_value = True

        result = train_model(
            model_name="lightgbm",
            output_dir=tmp_path / "output",
        )

    # Verifications
    assert isinstance(result, TrainingResult)
    assert result.model_name == "lightgbm"
    assert result.train_samples > 0

    # Check that model was initialized with params
    mock_model_class.assert_called_once()
    call_kwargs = mock_model_class.call_args[1]
    assert call_kwargs["n_estimators"] == 10

    # Check that fit was called
    mock_model_instance.fit.assert_called_once()

    # Check that save was called
    mock_model_instance.save.assert_called_once()

    # Check results file
    assert (tmp_path / "output" / "lightgbm_training_results.json").exists()
    assert (tmp_path / "output" / "lightgbm_events_train.parquet").exists()


@patch("src.labelling.label_primaire.train.load_optimized_params")
def test_train_model_config(mock_load_params, tmp_path):
    """Test TrainingConfig usage."""

    config = TrainingConfig(
        model_name="xgboost",
        random_state=123,
        vol_window=50,
        use_class_weight=False,
    )

    assert config.model_name == "xgboost"
    assert config.random_state == 123
    assert config.vol_window == 50
    assert not config.use_class_weight


@patch("src.labelling.label_primaire.train.load_optimized_params")
@patch("src.labelling.label_primaire.train.load_model_class")
@patch("src.labelling.label_primaire.train.get_dataset_for_model")
@patch("src.labelling.label_primaire.train.pd.read_parquet")
@patch("src.labelling.label_primaire.train.get_daily_volatility")
@patch("src.labelling.label_primaire.train.get_events_primary")
def test_train_model_class_weights(
    mock_get_events,
    mock_get_vol,
    mock_read_parquet,
    mock_get_dataset,
    mock_load_model_class,
    mock_load_params,
    mock_dataset,
    mock_dollar_bars,
    mock_events,
    mock_optimized_params,
    tmp_path,
):
    """Test class weight computation."""

    # Setup mocks
    mock_load_params.return_value = mock_optimized_params
    mock_model_class = MagicMock()
    mock_load_model_class.return_value = mock_model_class
    mock_get_dataset.return_value = mock_dataset
    mock_read_parquet.return_value = mock_dollar_bars
    mock_get_vol.return_value = pd.Series(0.01, index=mock_dollar_bars.index)
    mock_get_events.return_value = mock_events

    # Configure to use class weights
    config = TrainingConfig(model_name="lightgbm", use_class_weight=True)

    with patch("src.labelling.label_primaire.train.DOLLAR_BARS_PARQUET") as mock_path_const:
        mock_path_const.exists.return_value = True

        train_model(
            model_name="lightgbm",
            config=config,
            output_dir=tmp_path / "output",
        )

    # Check that class_weight was passed to model
    call_kwargs = mock_model_class.call_args[1]
    assert "class_weight" in call_kwargs
    weights = call_kwargs["class_weight"]
    assert isinstance(weights, dict)
    # We have 3 classes in mock_events: -1, 0, 1
    assert len(weights) == 3
