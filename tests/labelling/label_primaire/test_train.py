"""
Tests for training module in label_primaire.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.labelling.label_primaire.train import (
    train_model,
    TrainingConfig,
    TrainingResult,
    load_optimized_params,
    get_available_optimized_models,
    select_model,
    get_yes_no_input,
    main,
)

# =============================================================================
# DATA HELPERS
# =============================================================================

@pytest.fixture
def mock_dataset():
    dates = pd.date_range("2021-01-01", periods=10)
    features = pd.DataFrame(
        np.random.randn(10, 5),
        index=dates,
        columns=["f1", "f2", "f3", "f4", "f5"]
    )
    features["split"] = "train"
    features["datetime_close"] = dates
    return features

@pytest.fixture
def mock_bars():
    dates = pd.date_range("2021-01-01", periods=10)
    bars = pd.DataFrame({
        "datetime_close": dates,
        "close": np.linspace(100, 110, 10)
    })
    return bars

# =============================================================================
# TRAIN FUNCTION TESTS
# =============================================================================

def test_train_model(mocker: MockerFixture, mock_dataset, mock_bars, tmp_path):
    # Mock MODEL_REGISTRY to include test_model
    mocker.patch.dict(
        "src.labelling.label_primaire.train.MODEL_REGISTRY",
        {"test_model": {"class": "test.Model", "dataset": "tree", "search_space": {}}}
    )

    # Mock external dependencies
    mocker.patch(
        "src.labelling.label_primaire.train.load_optimized_params",
        return_value={
            "model_params": {"n_estimators": 10},
            "triple_barrier_params": {"pt_mult": 1, "sl_mult": 1, "max_holding": 5},
            "best_score": 0.8,
            "metric": "mcc"
        }
    )

    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_cls.return_value = mock_model_instance
    mocker.patch(
        "src.labelling.label_primaire.train.load_model_class",
        return_value=mock_model_cls
    )

    mocker.patch(
        "src.labelling.label_primaire.train.get_dataset_for_model",
        return_value=mock_dataset
    )

    mocker.patch(
        "pandas.read_parquet",
        return_value=mock_bars
    )
    mocker.patch("src.labelling.label_primaire.train.DOLLAR_BARS_PARQUET", MagicMock(exists=lambda: True))

    mocker.patch(
        "src.labelling.label_primaire.train.get_daily_volatility",
        return_value=pd.Series(np.ones(10)*0.01, index=mock_dataset.index)
    )

    # Mock events generation
    mock_events = pd.DataFrame({
        "label": [1, 0] * 5
    }, index=mock_dataset.index)
    mocker.patch(
        "src.labelling.label_primaire.train.get_events_primary",
        return_value=mock_events
    )

    # Run training
    config = TrainingConfig(model_name="test_model")
    result = train_model("test_model", config, output_dir=tmp_path)

    assert isinstance(result, TrainingResult)
    assert result.model_name == "test_model"
    assert result.train_samples == 10

    # Verify fit called
    mock_model_instance.fit.assert_called()
    mock_model_instance.save.assert_called()

# =============================================================================
# PARAMETER LOADING TESTS
# =============================================================================

def test_load_optimized_params(tmp_path):
    import json

    # Setup mock file
    p = tmp_path / "test_model_optimization.json"
    data = {
        "best_params": {"a": 1},
        "best_triple_barrier_params": {"b": 2},
        "best_score": 0.5,
        "metric": "accuracy"
    }
    p.write_text(json.dumps(data))

    params = load_optimized_params("test_model", opti_dir=tmp_path)
    assert params["model_params"] == {"a": 1}
    assert params["best_score"] == 0.5

    # Test not found
    with pytest.raises(FileNotFoundError):
        load_optimized_params("non_existent", opti_dir=tmp_path)

# =============================================================================
# CLI HELPER TESTS
# =============================================================================

def test_get_available_optimized_models(mocker, tmp_path):
    # Mock MODEL_REGISTRY and path
    mocker.patch("src.labelling.label_primaire.train.MODEL_REGISTRY", {"m1": {}, "m2": {}})
    mocker.patch("src.labelling.label_primaire.train.LABEL_PRIMAIRE_OPTI_DIR", tmp_path)

    # Create one optimization file
    (tmp_path / "m1_optimization.json").touch()

    available = get_available_optimized_models()
    assert "m1" in available
    assert "m2" not in available

def test_select_model(mocker):
    mocker.patch("src.labelling.label_primaire.train.MODEL_REGISTRY", {"m1": {"dataset": "d"}})
    mocker.patch("src.labelling.label_primaire.train.get_available_optimized_models", return_value=["m1"])

    mocker.patch("builtins.input", return_value="1")
    assert select_model() == "m1"

def test_get_yes_no_input(mocker):
    mocker.patch("builtins.input", return_value="y")
    assert get_yes_no_input("test")

    mocker.patch("builtins.input", return_value="n")
    assert not get_yes_no_input("test")

    mocker.patch("builtins.input", return_value="")
    assert get_yes_no_input("test", default=True)

# =============================================================================
# MAIN CLI TEST
# =============================================================================

def test_main(mocker):
    mocker.patch("src.labelling.label_primaire.train.select_model", return_value="m1")
    mocker.patch("src.labelling.label_primaire.train.load_optimized_params", return_value={
        "metric": "score", "best_score": 0.9, "triple_barrier_params": {}
    })
    mocker.patch("src.labelling.label_primaire.train.get_yes_no_input", return_value=True)
    mocker.patch("builtins.input", return_value="o") # confirm

    mock_train = mocker.patch("src.labelling.label_primaire.train.train_model", return_value=MagicMock(
        train_samples=100,
        label_distribution={"label_counts": {}, "label_percentages": {}},
        triple_barrier_params={},
        model_path="path",
        events_path="events"
    ))

    main()
    mock_train.assert_called_once()
