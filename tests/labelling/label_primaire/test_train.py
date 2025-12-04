"""Tests for training module in label_primaire."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.labelling.label_primaire.train.logic import (
    PrimaryEvaluationMetrics,
    TrainingConfig,
    TrainingResult,
    WalkForwardKFold,
    _compute_class_weights,
    _handle_missing_values,
    _remove_non_feature_cols,
    compute_metrics,
    generate_oof_predictions,
    get_available_optimized_models,
    get_yes_no_input,
    load_optimized_params,
    select_model,
    train_model,
)
from src.labelling.label_primaire.train.main import main


# =============================================================================
# DATA FIXTURES
# =============================================================================


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """Create a mock dataset for testing."""
    dates = pd.date_range("2021-01-01", periods=100)
    features = pd.DataFrame(
        np.random.randn(100, 5), index=dates, columns=["f1", "f2", "f3", "f4", "f5"]
    )
    features["split"] = "train"
    features["label"] = np.random.choice([-1, 0, 1], 100)
    features["t1"] = dates[5:].tolist() + dates[-5:].tolist()
    return features


@pytest.fixture
def mock_bars() -> pd.DataFrame:
    """Create mock dollar bars."""
    dates = pd.date_range("2021-01-01", periods=100)
    return pd.DataFrame({"datetime_close": dates, "close": np.linspace(100, 110, 100)})


# =============================================================================
# WALK-FORWARD CV TESTS
# =============================================================================


def test_walk_forward_kfold_basic() -> None:
    """Test basic WalkForwardKFold functionality."""
    dates = pd.date_range("2021-01-01", periods=100)
    X = pd.DataFrame(np.random.randn(100, 5), index=dates)
    y = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)

    cv = WalkForwardKFold(n_splits=5, min_train_size=10)
    splits = cv.split(X, y)

    assert len(splits) > 0
    for train_idx, val_idx in splits:
        assert len(train_idx) >= 10
        assert len(val_idx) > 0
        # Walk-forward: train comes before val
        if len(train_idx) > 0 and len(val_idx) > 0:
            assert train_idx[-1] < val_idx[0]


def test_walk_forward_kfold_with_t1() -> None:
    """Test WalkForwardKFold with purging (t1)."""
    dates = pd.date_range("2021-01-01", periods=100)
    X = pd.DataFrame(np.random.randn(100, 5), index=dates)
    y = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)
    t1 = pd.Series(dates[5:].tolist() + dates[-5:].tolist(), index=dates)

    cv = WalkForwardKFold(n_splits=5, min_train_size=10)
    splits = cv.split(X, y, t1=t1)

    assert len(splits) > 0


def test_walk_forward_kfold_with_embargo() -> None:
    """Test WalkForwardKFold with embargo."""
    dates = pd.date_range("2021-01-01", periods=100)
    X = pd.DataFrame(np.random.randn(100, 5), index=dates)
    y = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)

    cv = WalkForwardKFold(n_splits=5, embargo_pct=0.05, min_train_size=10)
    splits = cv.split(X, y)

    assert len(splits) > 0


# =============================================================================
# DATA PROCESSING TESTS
# =============================================================================


def test_remove_non_feature_cols() -> None:
    """Test removal of non-feature columns."""
    df = pd.DataFrame(
        {
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "label": [0, 1, 0],
            "split": ["train", "train", "test"],
            "bar_id": [1, 2, 3],
            "datetime_close": pd.date_range("2021-01-01", periods=3),
        }
    )

    result = _remove_non_feature_cols(df)

    assert "f1" in result.columns
    assert "f2" in result.columns
    assert "label" not in result.columns
    assert "split" not in result.columns
    assert "bar_id" not in result.columns


def test_handle_missing_values() -> None:
    """Test missing value handling."""
    df = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0],
            "f2": [np.nan, np.nan, np.nan],
        }
    )

    result = _handle_missing_values(df)

    assert result["f1"].isna().sum() == 0
    assert result["f1"].iloc[1] == 2.0  # Median of 1, 3
    assert result["f2"].isna().sum() == 0
    assert (result["f2"] == 0).all()  # All NaN -> 0


def test_compute_class_weights() -> None:
    """Test class weight computation."""
    y = pd.Series([-1, -1, 0, 0, 0, 0, 1, 1, 1, 1])
    weights = _compute_class_weights(y)

    assert -1 in weights
    assert 0 in weights
    assert 1 in weights
    # Minority class (-1) should have higher weight
    assert weights[-1] > weights[0]


# =============================================================================
# METRICS TESTS
# =============================================================================


def test_compute_metrics() -> None:
    """Test metrics computation."""
    np.random.seed(42)
    y_true = np.random.choice([-1, 0, 1], 100)
    y_pred = np.random.choice([-1, 0, 1], 100)

    metrics = compute_metrics(y_true, y_pred)

    assert isinstance(metrics, PrimaryEvaluationMetrics)
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.f1_macro <= 1
    assert -1 <= metrics.mcc <= 1


def test_compute_metrics_with_proba() -> None:
    """Test metrics computation with probabilities."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2], 100)
    y_pred = np.random.choice([0, 1, 2], 100)
    y_proba = np.random.rand(100, 3)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics.auc_roc is not None


def test_primary_evaluation_metrics_to_dict() -> None:
    """Test PrimaryEvaluationMetrics serialization."""
    metrics = PrimaryEvaluationMetrics(
        accuracy=0.85,
        precision_macro=0.80,
        recall_macro=0.78,
        f1_macro=0.79,
        mcc=0.65,
        auc_roc=0.88,
    )

    d = metrics.to_dict()

    assert d["accuracy"] == 0.85
    assert d["mcc"] == 0.65
    assert "auc_roc" in d


# =============================================================================
# OOF PREDICTION TESTS
# =============================================================================


def test_generate_oof_predictions() -> None:
    """Test out-of-fold prediction generation."""
    dates = pd.date_range("2021-01-01", periods=100)
    features = pd.DataFrame(np.random.randn(100, 5), index=dates)
    labels = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)

    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_instance.predict.return_value = np.random.choice([-1, 0, 1], 20)
    mock_instance.predict_proba.return_value = np.random.rand(20, 3)

    preds, proba, coverage = generate_oof_predictions(
        features, labels, None, mock_model_cls, {}, n_splits=5, min_train_size=10
    )

    assert len(preds) == 100
    assert len(coverage) == 100
    assert coverage.sum() > 0  # Some samples should be covered


# =============================================================================
# PARAMETER LOADING TESTS
# =============================================================================


def test_load_optimized_params(tmp_path: Path) -> None:
    """Test loading optimization parameters."""
    p = tmp_path / "test_model_optimization.json"
    data = {
        "best_params": {"n_estimators": 100},
        "best_triple_barrier_params": {"pt_mult": 1.0},
        "best_score": 0.8,
        "metric": "mcc",
    }
    p.write_text(json.dumps(data))

    params = load_optimized_params("test_model", opti_dir=tmp_path)

    assert params["model_params"] == {"n_estimators": 100}
    assert params["best_score"] == 0.8


def test_load_optimized_params_not_found(tmp_path: Path) -> None:
    """Test loading non-existent optimization file."""
    with pytest.raises(FileNotFoundError):
        load_optimized_params("non_existent", opti_dir=tmp_path)


# =============================================================================
# TRAINING CONFIG/RESULT TESTS
# =============================================================================


def test_training_config_defaults() -> None:
    """Test TrainingConfig defaults."""
    config = TrainingConfig(model_name="test")

    assert config.model_name == "test"
    assert config.use_class_weight is True


def test_training_result_serialization(tmp_path: Path) -> None:
    """Test TrainingResult serialization."""
    result = TrainingResult(
        model_name="test",
        model_params={"a": 1},
        triple_barrier_params={"b": 2},
        train_samples=1000,
        test_samples=200,
        train_metrics={"accuracy": 0.85},
        test_metrics={"accuracy": 0.80},
        label_distribution_train={"0": 500, "1": 500},
        label_distribution_test={"0": 100, "1": 100},
        n_folds=5,
        model_path="/path/to/model.joblib",
        oof_predictions_path="/path/to/oof.parquet",
        test_predictions_path="/path/to/test.parquet",
    )

    d = result.to_dict()
    assert d["model_name"] == "test"
    assert d["train_samples"] == 1000

    save_path = tmp_path / "result.json"
    result.save(save_path)
    assert save_path.exists()


# =============================================================================
# TRAIN MODEL TESTS
# =============================================================================


def test_train_model(
    mocker: MockerFixture, mock_dataset: pd.DataFrame, tmp_path: Path
) -> None:
    """Test train_model function."""
    mocker.patch.dict(
        "src.labelling.label_primaire.train.logic.MODEL_REGISTRY",
        {"test_model": {"class": "test.Model", "dataset": "tree", "search_space": {}}},
    )
    mocker.patch(
        "src.labelling.label_primaire.train.logic.load_optimized_params",
        return_value={
            "model_params": {"n_estimators": 10},
            "triple_barrier_params": {},
            "best_score": 0.8,
            "metric": "mcc",
        },
    )

    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_cls.return_value = mock_model_instance

    # Return predictions matching the input size dynamically
    def dynamic_predict(X: pd.DataFrame) -> np.ndarray:
        return np.random.choice([-1, 0, 1], len(X))

    def dynamic_predict_proba(X: pd.DataFrame) -> np.ndarray:
        return np.random.rand(len(X), 3)

    mock_model_instance.predict.side_effect = dynamic_predict
    mock_model_instance.predict_proba.side_effect = dynamic_predict_proba

    mocker.patch(
        "src.labelling.label_primaire.train.logic.load_model_class",
        return_value=mock_model_cls,
    )
    mocker.patch(
        "src.labelling.label_primaire.train.logic.get_dataset_for_model",
        return_value=mock_dataset,
    )

    config = TrainingConfig(model_name="test_model")
    result = train_model("test_model", config, output_dir=tmp_path)

    assert isinstance(result, TrainingResult)
    assert result.model_name == "test_model"
    mock_model_instance.fit.assert_called()
    mock_model_instance.save.assert_called()


# =============================================================================
# CLI HELPER TESTS
# =============================================================================


def test_get_available_optimized_models(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test listing available optimized models."""
    mocker.patch(
        "src.labelling.label_primaire.train.logic.LABEL_PRIMAIRE_OPTI_DIR", tmp_path
    )

    (tmp_path / "m1_optimization.json").touch()

    available = get_available_optimized_models()

    assert "m1" in available


def test_get_available_optimized_models_empty(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test when no optimized models exist."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    mocker.patch(
        "src.labelling.label_primaire.train.logic.LABEL_PRIMAIRE_OPTI_DIR", empty_dir
    )

    available = get_available_optimized_models()

    assert available == []


def test_select_model(mocker: MockerFixture) -> None:
    """Test model selection."""
    mocker.patch(
        "src.labelling.label_primaire.train.logic.get_available_optimized_models",
        return_value=["m1"],
    )
    mocker.patch("builtins.input", return_value="1")

    result = select_model()

    assert result == "m1"


def test_select_model_by_name(mocker: MockerFixture) -> None:
    """Test model selection by name."""
    mocker.patch(
        "src.labelling.label_primaire.train.logic.get_available_optimized_models",
        return_value=["lightgbm", "xgboost"],
    )
    mocker.patch("builtins.input", return_value="lightgbm")

    result = select_model()

    assert result == "lightgbm"


def test_get_yes_no_input_yes(mocker: MockerFixture) -> None:
    """Test yes input."""
    mocker.patch("builtins.input", return_value="y")
    assert get_yes_no_input("test") is True


def test_get_yes_no_input_no(mocker: MockerFixture) -> None:
    """Test no input."""
    mocker.patch("builtins.input", return_value="n")
    assert get_yes_no_input("test") is False


def test_get_yes_no_input_default(mocker: MockerFixture) -> None:
    """Test default input."""
    mocker.patch("builtins.input", return_value="")
    assert get_yes_no_input("test", default=True) is True
    assert get_yes_no_input("test", default=False) is False


# =============================================================================
# MAIN CLI TEST
# =============================================================================
# Note: The main() function imports directly from logic.py via "from ... import",
# making it hard to mock. We test the helper functions individually instead,
# and trust that the main() function correctly wires them together.
# The logic functions (select_model, load_optimized_params, get_yes_no_input,
# train_model) are all tested above.
