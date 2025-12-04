"""
Tests for utilities in label_primaire.
"""

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.labelling.label_primaire.utils import (
    MODEL_REGISTRY,
    TRIPLE_BARRIER_SEARCH_SPACE,
    FOCAL_LOSS_SEARCH_SPACE,
    FOCAL_LOSS_SUPPORTED_MODELS,
    CLASS_WEIGHT_SUPPORTED_MODELS,
    OptimizationConfig,
    OptimizationResult,
    compute_barriers,
    compute_return_and_label,
    find_barrier_touch,
    get_daily_volatility,
    get_dataset_for_model,
    is_valid_barrier,
    load_dollar_bars,
    load_model_class,
    set_vertical_barriers,
)


# =============================================================================
# MODEL REGISTRY TESTS
# =============================================================================


def test_registry_contains_expected_models():
    """Verify registry has the expected keys."""
    expected = [
        "lightgbm",
        "xgboost",
        "ridge",
        "lstm",
    ]
    for model in expected:
        assert model in MODEL_REGISTRY
        assert "class" in MODEL_REGISTRY[model]
        assert "dataset" in MODEL_REGISTRY[model]
        assert "search_space" in MODEL_REGISTRY[model]


def test_triple_barrier_search_space():
    """Verify triple barrier search space."""
    keys = ["pt_mult", "sl_mult", "min_return", "max_holding"]
    for key in keys:
        assert key in TRIPLE_BARRIER_SEARCH_SPACE


# =============================================================================
# DATACLASS TESTS
# =============================================================================


def test_optimization_config():
    """Test config dataclass defaults."""
    config = OptimizationConfig(model_name="test")
    assert config.model_name == "test"
    assert config.n_trials == 50
    assert config.n_splits == 5


def test_optimization_result_serialization(tmp_path: Path):
    """Test result serialization and saving."""
    res = OptimizationResult(
        model_name="test",
        best_params={"a": 1},
        best_triple_barrier_params={"b": 2},
        best_score=0.8,
        metric="f1",
        n_trials=10,
        cv_scores=[0.7, 0.9],
    )

    # Test to_dict
    d = res.to_dict()
    assert d["model_name"] == "test"
    assert d["best_score"] == 0.8
    assert d["cv_scores"] == [0.7, 0.9]

    # Test save
    save_path = tmp_path / "result.json"
    res.save(save_path)
    assert save_path.exists()

    # Verify content
    import json
    with open(save_path) as f:
        loaded = json.load(f)
    assert loaded["model_name"] == "test"
    assert loaded["best_params"] == {"a": 1}


# =============================================================================
# LOADING TESTS
# =============================================================================


def test_load_model_class_success(mocker: MockerFixture):
    """Test loading a valid model class."""
    # Mock dynamic import
    mock_module = mocker.Mock()
    mock_class = mocker.Mock()
    setattr(mock_module, "LightGBMModel", mock_class)

    mocker.patch("builtins.__import__", return_value=mock_module)
    mocker.patch.dict(MODEL_REGISTRY, {
        "test_model": {
            "class": "src.model.lightgbm_model.LightGBMModel",
            "dataset": "tree",
            "search_space": {}
        }
    })

    cls = load_model_class("test_model")
    assert cls == mock_class


def test_load_model_class_unknown():
    """Test loading an unknown model raises error."""
    with pytest.raises(ValueError, match="Unknown model"):
        load_model_class("non_existent_model")


def test_get_dataset_for_model(mocker: MockerFixture, tmp_path: Path):
    """Test dataset loading logic."""
    # Create dummy parquet
    df = pd.DataFrame({"a": [1, 2, 3]})
    p = tmp_path / "test.parquet"
    df.to_parquet(p)

    mocker.patch("src.labelling.label_primaire.utils.DATASET_FEATURES_FINAL_PARQUET", p)
    mocker.patch("src.labelling.label_primaire.utils.DATASET_FEATURES_LINEAR_FINAL_PARQUET", p)
    mocker.patch("src.labelling.label_primaire.utils.DATASET_FEATURES_LSTM_FINAL_PARQUET", p)

    # Valid model - tree dataset (lightgbm)
    loaded_df = get_dataset_for_model("lightgbm")
    pd.testing.assert_frame_equal(df, loaded_df)

    # Unknown model
    with pytest.raises(ValueError):
        get_dataset_for_model("unknown")

    # File not found
    mocker.patch("src.labelling.label_primaire.utils.DATASET_FEATURES_FINAL_PARQUET", Path("nonexistent"))
    with pytest.raises(FileNotFoundError):
        get_dataset_for_model("lightgbm")


def test_load_dollar_bars(mocker: MockerFixture, tmp_path: Path):
    """Test dollar bars loading."""
    # Success case
    df = pd.DataFrame({
        "datetime_close": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "close": [100, 101]
    })
    p = tmp_path / "bars.parquet"
    df.to_parquet(p)

    mocker.patch("src.labelling.label_primaire.utils.DOLLAR_BARS_PARQUET", p)
    loaded = load_dollar_bars()
    assert loaded.index.name == "datetime_close"
    assert len(loaded) == 2

    # Missing column
    df_bad = pd.DataFrame({"close": [100]})
    df_bad.to_parquet(p)
    with pytest.raises(ValueError, match="must have 'datetime_close'"):
        load_dollar_bars()

    # File not found
    mocker.patch("src.labelling.label_primaire.utils.DOLLAR_BARS_PARQUET", Path("nonexistent"))
    with pytest.raises(FileNotFoundError):
        load_dollar_bars()


# =============================================================================
# VOLATILITY & BARRIER TESTS
# =============================================================================


def test_get_daily_volatility():
    """Test volatility calculation."""
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    close = pd.Series(np.random.randn(100) + 100, index=dates)

    vol = get_daily_volatility(close, span=10, min_periods=5)
    assert isinstance(vol, pd.Series)
    assert len(vol) == 100
    assert pd.isna(vol.iloc[0])  # First values should be NaN due to min_periods


def test_compute_barriers():
    """Test barrier level computation."""
    events = pd.DataFrame({"trgt": [0.01, 0.02]}, index=[0, 1])

    # Both active
    res = compute_barriers(events, pt_mult=2.0, sl_mult=1.0)
    assert res["pt"].iloc[0] == 0.02
    assert res["sl"].iloc[0] == -0.01

    # One disabled
    res = compute_barriers(events, pt_mult=0, sl_mult=1.0)
    assert pd.isna(res["pt"].iloc[0])
    assert res["sl"].iloc[0] == -0.01


def test_is_valid_barrier():
    assert is_valid_barrier(1.0)
    assert is_valid_barrier(0.0)
    assert not is_valid_barrier(None)
    assert not is_valid_barrier(np.nan)


def test_find_barrier_touch():
    """Test finding first touch."""
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    ret = pd.Series([0.0, 0.01, 0.02, -0.01, -0.02], index=dates)

    # Hit PT
    touch = find_barrier_touch(ret, pt_barrier=0.015, sl_barrier=-0.015)
    assert touch == dates[2]  # 0.02 > 0.015

    # Hit SL
    touch = find_barrier_touch(ret, pt_barrier=0.03, sl_barrier=-0.015)
    assert touch == dates[4]  # -0.02 < -0.015

    # Hit Neither
    touch = find_barrier_touch(ret, pt_barrier=0.03, sl_barrier=-0.03)
    assert touch is None


def test_compute_return_and_label():
    """Test return and labeling logic."""
    dates = pd.date_range("2021-01-01", periods=3, freq="D")
    close = pd.Series([100, 101, 99], index=dates)

    # Positive
    ret, label = compute_return_and_label(close, dates[0], dates[1], min_return=0.005)
    assert ret == 0.01
    assert label == 1

    # Negative
    ret, label = compute_return_and_label(close, dates[1], dates[2], min_return=0.005)
    assert ret < 0
    assert label == -1

    # Neutral (small move)
    ret, label = compute_return_and_label(close, dates[0], dates[1], min_return=0.02)
    assert label == 0

    # Error case
    ret, label = compute_return_and_label(close, dates[0], pd.Timestamp("2022-01-01"), 0.01)
    assert pd.isna(ret)
    assert label == 0


def test_set_vertical_barriers():
    """Test vertical barrier setting."""
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    close_idx = dates
    t_events = dates[:5]

    # Normal case
    t1 = set_vertical_barriers(t_events, close_idx, max_holding=2)
    assert t1.iloc[0] == dates[2]

    # Cap at end
    t1 = set_vertical_barriers(t_events, close_idx, max_holding=20)
    assert t1.iloc[0] == dates[-1]


# =============================================================================
# FOCAL LOSS CONFIG TESTS
# =============================================================================


def test_focal_loss_search_space():
    """Test focal loss search space is properly defined."""
    assert "focal_gamma" in FOCAL_LOSS_SEARCH_SPACE
    assert "use_focal_loss" in FOCAL_LOSS_SEARCH_SPACE

    # Check gamma values include expected range
    _, gamma_values = FOCAL_LOSS_SEARCH_SPACE["focal_gamma"]
    assert 0.0 in gamma_values  # CE baseline
    assert 2.0 in gamma_values  # Standard focal loss


def test_focal_loss_supported_models():
    """Test focal loss supported models list."""
    # LightGBM supports custom objectives
    assert "lightgbm" in FOCAL_LOSS_SUPPORTED_MODELS
    # These don't support custom objectives
    assert "ridge" not in FOCAL_LOSS_SUPPORTED_MODELS
    assert "logistic" not in FOCAL_LOSS_SUPPORTED_MODELS


def test_class_weight_supported_models():
    """Test class weight supported models list."""
    expected_supported = ["lightgbm", "xgboost"]
    for model in expected_supported:
        assert model in CLASS_WEIGHT_SUPPORTED_MODELS

    # Ridge doesn't support class_weight in sklearn
    assert "ridge" not in CLASS_WEIGHT_SUPPORTED_MODELS


def test_optimization_config_focal_params():
    """Test OptimizationConfig includes focal loss parameters."""
    config = OptimizationConfig(model_name="lightgbm")

    # Check focal loss defaults
    assert hasattr(config, "use_focal_loss")
    assert hasattr(config, "focal_gamma")
    assert hasattr(config, "optimize_focal_params")
    assert hasattr(config, "use_class_weights")

    # Check default values
    assert config.use_focal_loss == True
    assert config.focal_gamma == 2.0
    assert config.optimize_focal_params == True
    assert config.use_class_weights == True


def test_optimization_config_custom_focal_params():
    """Test OptimizationConfig with custom focal loss parameters."""
    config = OptimizationConfig(
        model_name="lightgbm",
        use_focal_loss=False,
        focal_gamma=5.0,
        optimize_focal_params=False,
        use_class_weights=False,
    )

    assert config.use_focal_loss == False
    assert config.focal_gamma == 5.0
    assert config.optimize_focal_params == False
    assert config.use_class_weights == False


def test_optimization_result_focal_params(tmp_path: Path):
    """Test OptimizationResult includes focal loss params."""
    result = OptimizationResult(
        model_name="lightgbm",
        best_params={"n_estimators": 100},
        best_triple_barrier_params={"pt_mult": 1.0, "sl_mult": 1.0},
        best_focal_loss_params={"use_focal_loss": True, "focal_gamma": 2.0},
        best_score=0.75,
        metric="mcc",
        n_trials=50,
    )

    # Test to_dict includes focal params
    d = result.to_dict()
    assert "best_focal_loss_params" in d
    assert d["best_focal_loss_params"]["use_focal_loss"] == True
    assert d["best_focal_loss_params"]["focal_gamma"] == 2.0

    # Test save includes focal params
    save_path = tmp_path / "result_focal.json"
    result.save(save_path)
    assert save_path.exists()

    import json
    with open(save_path) as f:
        loaded = json.load(f)
    assert "best_focal_loss_params" in loaded
    assert loaded["best_focal_loss_params"]["focal_gamma"] == 2.0
