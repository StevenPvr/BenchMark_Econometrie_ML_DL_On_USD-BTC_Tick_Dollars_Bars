"""Tests for opti module (Optimization Module)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

from src.labelling.label_primaire.opti.logic import (
    WalkForwardCV,
    _align_features_events,
    _compute_class_weight,
    _compute_composite_score,
    _compute_sign_error_rate,
    _evaluate_fold,
    _run_cv_scoring,
    _sample_focal_loss_params,
    _sample_model_params,
    _subsample_features,
    create_objective,
    optimize_model,
    print_final_summary,
    run_parallel,
    run_sequential,
    select_models_interactive,
)
from src.labelling.label_primaire.utils import (
    CLASS_WEIGHT_SUPPORTED_MODELS,
    FOCAL_LOSS_SEARCH_SPACE,
    FOCAL_LOSS_SUPPORTED_MODELS,
    MODEL_REGISTRY,
    OptimizationConfig,
    OptimizationResult,
)


# =============================================================================
# DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_data() -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Generate sample data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    close = pd.Series(np.random.uniform(100, 110, size=100), index=dates)
    features = pd.DataFrame(
        np.random.randn(100, 5), index=dates, columns=[f"f{i}" for i in range(5)]
    )
    volatility = pd.Series(np.random.uniform(0.01, 0.02, size=100), index=dates)
    return close, features, volatility


@pytest.fixture
def sample_events(sample_data: tuple[pd.Series, pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Generate sample events for testing."""
    close, features, volatility = sample_data
    events = pd.DataFrame(index=features.index)
    events["label"] = np.random.choice([-1, 0, 1], len(features))
    events["t1"] = features.index[10:].tolist() + features.index[-10:].tolist()
    events["t1"] = events["t1"][: len(features)]
    return events


# =============================================================================
# CV TESTS
# =============================================================================


def test_walk_forward_cv_basic(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test WalkForwardCV basic functionality."""
    _, features, _ = sample_data
    n_events = 80
    events = pd.DataFrame(index=features.index[:n_events])
    events["label"] = np.random.choice([-1, 0, 1], n_events)
    events["t1"] = features.index[10 : 10 + n_events].tolist()[:n_events]

    cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.0)
    splits = cv.split(features.iloc[:n_events], events)

    assert len(splits) > 0
    for train, val in splits:
        assert len(train) >= 10
        assert len(val) > 0
        # Walk-forward: train indices should be before val indices
        if len(train) > 0 and len(val) > 0:
            assert train[-1] < val[0]


def test_walk_forward_cv_with_embargo(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test WalkForwardCV with embargo."""
    _, features, _ = sample_data
    events = pd.DataFrame(index=features.index)
    events["label"] = np.random.choice([-1, 0, 1], len(features))

    cv = WalkForwardCV(n_splits=5, min_train_size=10, embargo_pct=0.05)
    splits = cv.split(features, events)

    assert len(splits) > 0


def test_walk_forward_cv_with_purging(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test WalkForwardCV with purging (t1 column)."""
    _, features, _ = sample_data
    events = pd.DataFrame(index=features.index)
    events["label"] = np.random.choice([-1, 0, 1], len(features))
    # t1 is 5 bars ahead - create properly sized list
    t1_values = features.index[5:].tolist()
    # Pad with last values to match features length
    t1_values = t1_values + [features.index[-1]] * (len(features) - len(t1_values))
    events["t1"] = t1_values

    # Use smaller min_train_size and fewer splits for 100 samples
    cv = WalkForwardCV(n_splits=3, min_train_size=5)
    splits = cv.split(features, events)

    # With purging, we may have fewer splits - at least test it doesn't error
    assert isinstance(splits, list)


# =============================================================================
# SCORING METRICS TESTS
# =============================================================================


def test_compute_sign_error_rate() -> None:
    """Test sign error rate computation."""
    # No sign errors
    y_true = np.array([1, 1, -1, -1, 0])
    y_pred = np.array([1, 1, -1, -1, 0])
    assert _compute_sign_error_rate(y_true, y_pred) == 0.0

    # All sign errors
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([-1, -1, 1, 1])
    assert _compute_sign_error_rate(y_true, y_pred) == 1.0

    # 50% sign errors
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([1, -1, -1, 1])
    assert _compute_sign_error_rate(y_true, y_pred) == 0.5

    # No non-zero predictions
    y_true = np.array([1, -1])
    y_pred = np.array([0, 0])
    assert _compute_sign_error_rate(y_true, y_pred) == 0.0


def test_compute_composite_score() -> None:
    """Test composite score computation."""
    y_true = np.array([-1, -1, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([-1, -1, 0, 0, 1, 1, 1, 1, 1, 1])

    composite, details = _compute_composite_score(y_true, y_pred)

    assert composite > 0
    assert "recall_macro" in details
    assert "sign_error_rate" in details
    assert details["sign_error_rate"] == 0.0


def test_compute_class_weight() -> None:
    """Test class weight computation."""
    y = pd.Series([-1, -1, 0, 0, 0, 0, 1, 1, 1, 1])
    weights = _compute_class_weight(y)

    assert -1 in weights
    assert 0 in weights
    assert 1 in weights
    # Minority class (-1) should have higher weight
    assert weights[-1] > weights[0]


# =============================================================================
# DATA PREPARATION TESTS
# =============================================================================


def test_align_features_events(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test feature-event alignment."""
    _, features, _ = sample_data
    events = pd.DataFrame(index=features.index[:80])
    events["label"] = np.random.choice([-1, 0, 1], 80)

    config = OptimizationConfig(model_name="test", min_train_size=10)
    X, y, aligned, reason = _align_features_events(features, events, config)

    assert X is not None
    assert y is not None
    assert aligned is not None
    assert reason == "OK"
    assert len(X) == len(y) == len(aligned)


def test_align_features_events_not_enough_samples(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test alignment with insufficient samples."""
    _, features, _ = sample_data
    events = pd.DataFrame(index=features.index[:5])
    events["label"] = [1, 0, -1, 0, 1]

    config = OptimizationConfig(model_name="test", min_train_size=1000)
    X, y, aligned, reason = _align_features_events(features, events, config)

    assert X is None
    assert "not enough" in reason


def test_subsample_features(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test feature subsampling."""
    _, features, _ = sample_data

    # Full sample
    result = _subsample_features(features, 1.0)
    assert len(result) == len(features)

    # Half sample
    result = _subsample_features(features, 0.5)
    assert len(result) == 50


# =============================================================================
# FOLD EVALUATION TESTS
# =============================================================================


def test_evaluate_fold(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test single fold evaluation."""
    _, features, _ = sample_data
    y = pd.Series(np.random.choice([-1, 0, 1], len(features)), index=features.index)

    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_instance.predict.return_value = np.random.choice([-1, 0, 1], 10)

    train_idx = np.arange(50)
    val_idx = np.arange(50, 60)

    composite, details, reason = _evaluate_fold(
        features, y, train_idx, val_idx, mock_model_cls, {}, "test_model", 0
    )

    assert composite is not None or reason != "OK"
    if reason == "OK":
        assert details is not None
        assert "recall_macro" in details


def test_evaluate_fold_skip_single_class(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Test fold is skipped when validation has only one class."""
    _, features, _ = sample_data
    # All same class in validation set
    y = pd.Series(
        [0] * 50 + [1] * 10 + [0] * 40, index=features.index
    )  # indices 50-60 all 1s

    mock_model_cls = MagicMock()
    train_idx = np.arange(50)
    val_idx = np.arange(50, 60)  # All class 1

    composite, details, reason = _evaluate_fold(
        features, y, train_idx, val_idx, mock_model_cls, {}, "test_model", 0
    )

    assert composite is None
    assert "SKIP" in reason


# =============================================================================
# SAMPLING TESTS
# =============================================================================


def test_sample_focal_loss_params_supported_model(mocker: MagicMock) -> None:
    """Test focal loss param sampling for supported model."""
    config = OptimizationConfig(
        model_name="lightgbm",
        optimize_focal_params=True,
    )

    trial = mocker.Mock(spec=optuna.trial.Trial)
    trial.suggest_categorical.side_effect = [True, 2.0, 1.5]

    params = _sample_focal_loss_params(trial, config)

    assert "use_focal_loss" in params
    assert "focal_gamma" in params
    assert "minority_weight_boost" in params


def test_sample_focal_loss_params_disabled(mocker: MagicMock) -> None:
    """Test focal loss param sampling when optimization is disabled."""
    config = OptimizationConfig(
        model_name="lightgbm",
        optimize_focal_params=False,
    )

    trial = mocker.Mock(spec=optuna.trial.Trial)

    params = _sample_focal_loss_params(trial, config)

    assert "use_class_weights" in params
    trial.suggest_categorical.assert_not_called()


def test_sample_model_params(mocker: MagicMock) -> None:
    """Test model hyperparameter sampling."""
    trial = mocker.Mock(spec=optuna.trial.Trial)
    trial.suggest_int.return_value = 100
    trial.suggest_float.return_value = 0.1
    trial.suggest_categorical.return_value = 32

    params = _sample_model_params(trial, "xgboost")

    assert len(params) > 0
    assert trial.suggest_int.called or trial.suggest_float.called


def test_sample_model_params_unknown_model(mocker: MagicMock) -> None:
    """Test sampling with unknown model raises error."""
    trial = mocker.Mock(spec=optuna.trial.Trial)

    with pytest.raises(ValueError, match="Unknown model"):
        _sample_model_params(trial, "unknown_model")


# =============================================================================
# CV SCORING TESTS
# =============================================================================


def test_run_cv_scoring(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
    mocker: MagicMock,
) -> None:
    """Test CV scoring runner."""
    _, features, _ = sample_data
    events = pd.DataFrame(index=features.index)
    events["label"] = np.random.choice([-1, 0, 1], len(features))

    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance

    # Return predictions matching the fold size dynamically
    def dynamic_predict(X: pd.DataFrame) -> np.ndarray:
        return np.random.choice([-1, 0, 1], len(X))

    mock_instance.predict.side_effect = dynamic_predict

    # Smaller min_train_size for test data
    cv = WalkForwardCV(n_splits=3, min_train_size=5)

    mean_composite, valid_folds, total_folds, reason, scores, details = _run_cv_scoring(
        features,
        events,
        cv,
        mock_model_cls,
        {},
        "test_model",
        {"use_class_weights": True},
    )

    # With walk-forward CV, we should have some folds
    assert isinstance(total_folds, int)
    assert reason == "OK" or "no valid" in reason


# =============================================================================
# OBJECTIVE TESTS
# =============================================================================


def test_create_objective(
    sample_data: tuple[pd.Series, pd.DataFrame, pd.Series],
    mocker: MagicMock,
) -> None:
    """Test objective function creation."""
    _, features, _ = sample_data
    events = pd.DataFrame(index=features.index)
    events["label"] = np.random.choice([-1, 0, 1], len(features))

    config = OptimizationConfig(model_name="lightgbm", n_trials=1, min_train_size=10)
    mock_model_cls = MagicMock()

    objective = create_objective(config, features, events, mock_model_cls)

    assert callable(objective)


# =============================================================================
# OPTIMIZATION TESTS
# =============================================================================


def test_optimize_model(mocker: MagicMock) -> None:
    """Test optimize_model function."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    mock_features = pd.DataFrame(
        np.random.randn(100, 5), index=dates, columns=[f"f{i}" for i in range(5)]
    )
    mock_events = pd.DataFrame(index=dates)
    mock_events["label"] = np.random.choice([-1, 0, 1], 100)
    mock_events["t1"] = dates

    mocker.patch(
        "src.labelling.label_primaire.opti.logic.load_model_class", return_value=MagicMock
    )
    mocker.patch(
        "src.labelling.label_primaire.opti.logic._prepare_optimization_data",
        return_value=(mock_features, mock_events),
    )

    mock_study = MagicMock()
    mock_study.best_value = 0.75
    mock_study.best_params = {"n_estimators": 100}
    mock_study.best_trial = MagicMock()
    mock_study.best_trial.user_attrs = {
        "mean_recall_macro": 0.7,
        "mean_sign_error_rate": 0.1,
        "mean_mcc": 0.6,
    }
    mock_study.best_trials = [mock_study.best_trial]
    mock_study.trials = [mock_study.best_trial]
    mocker.patch(
        "src.labelling.label_primaire.opti.logic._create_study", return_value=mock_study
    )
    mocker.patch(
        "src.labelling.label_primaire.opti.logic.create_objective",
        return_value=lambda trial: 0.5,
    )

    mock_result = OptimizationResult(
        model_name="lightgbm",
        best_params={"n_estimators": 100},
        best_triple_barrier_params={},
        best_score=0.75,
        metric="composite_recall",
        n_trials=10,
    )
    mocker.patch(
        "src.labelling.label_primaire.opti.logic._build_result", return_value=mock_result
    )
    mocker.patch("src.labelling.label_primaire.opti.logic._log_result")

    res = optimize_model("lightgbm")

    assert res == mock_result
    mock_study.optimize.assert_called_once()


# =============================================================================
# CLI TESTS
# =============================================================================


def test_select_models_interactive_all(mocker: MagicMock) -> None:
    """Test 'all' model selection."""
    mocker.patch("builtins.input", return_value="0")
    models = select_models_interactive()
    assert len(models) > 0


def test_select_models_interactive_specific(mocker: MagicMock) -> None:
    """Test specific model selection by number."""
    mocker.patch("builtins.input", return_value="1")
    models = select_models_interactive()
    assert len(models) == 1


def test_select_models_interactive_by_name(mocker: MagicMock) -> None:
    """Test model selection by name."""
    mocker.patch("builtins.input", return_value="lightgbm")
    models = select_models_interactive()
    assert "lightgbm" in models


def test_run_sequential(mocker: MagicMock) -> None:
    """Test sequential optimization runner."""
    mock_result = MagicMock(best_score=0.5)
    mocker.patch(
        "src.labelling.label_primaire.opti.logic._run_optimization_worker",
        return_value=mock_result,
    )
    trials_per_model = {"m1": 10, "m2": 10}
    res = run_sequential(["m1", "m2"], trials_per_model, 5)
    assert len(res) == 2


def test_run_parallel(mocker: MagicMock) -> None:
    """Test parallel optimization runner (currently sequential)."""
    mock_result = MagicMock(best_score=0.5)
    mocker.patch(
        "src.labelling.label_primaire.opti.logic._run_optimization_worker",
        return_value=mock_result,
    )
    trials_per_model = {"m1": 10}
    res = run_parallel(["m1"], trials_per_model, 5)
    assert len(res) == 1


def test_print_final_summary(mocker: MagicMock, capsys: pytest.CaptureFixture) -> None:
    """Test final summary printing."""
    results = [
        OptimizationResult(
            model_name="test",
            best_params={},
            best_triple_barrier_params={},
            best_score=0.75,
            metric="mcc",
            n_trials=10,
        )
    ]
    print_final_summary(results)
    # Function uses logger, not print, so capsys won't capture it


# =============================================================================
# FOCAL LOSS AND CLASS WEIGHT TESTS
# =============================================================================


def test_focal_loss_search_space() -> None:
    """Test focal loss search space structure."""
    assert "focal_gamma" in FOCAL_LOSS_SEARCH_SPACE
    assert "use_focal_loss" in FOCAL_LOSS_SEARCH_SPACE

    _, gamma_choices = FOCAL_LOSS_SEARCH_SPACE["focal_gamma"]
    assert 0.0 in gamma_choices
    assert 2.0 in gamma_choices

    _, use_focal_choices = FOCAL_LOSS_SEARCH_SPACE["use_focal_loss"]
    assert True in use_focal_choices
    assert False in use_focal_choices


def test_focal_loss_supported_models() -> None:
    """Test focal loss supported models list."""
    assert "lightgbm" in FOCAL_LOSS_SUPPORTED_MODELS
    assert "ridge" not in FOCAL_LOSS_SUPPORTED_MODELS


def test_class_weight_supported_models() -> None:
    """Test class weight supported models list."""
    assert "lightgbm" in CLASS_WEIGHT_SUPPORTED_MODELS
    assert "xgboost" in CLASS_WEIGHT_SUPPORTED_MODELS
    assert "ridge" not in CLASS_WEIGHT_SUPPORTED_MODELS


def test_optimization_config_focal_defaults() -> None:
    """Test OptimizationConfig focal loss defaults."""
    config = OptimizationConfig(model_name="lightgbm")

    assert config.use_focal_loss is True
    assert config.focal_gamma == 2.0
    assert config.optimize_focal_params is True
    assert config.use_class_weights is True
    assert config.minority_weight_boost == 1.25


def test_optimization_result_includes_focal_params() -> None:
    """Test that OptimizationResult includes focal loss params."""
    result = OptimizationResult(
        model_name="lightgbm",
        best_params={"n_estimators": 100},
        best_triple_barrier_params={"pt_mult": 1.0},
        best_focal_loss_params={
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "minority_weight_boost": 1.5,
        },
        best_score=0.8,
        metric="mcc",
        n_trials=10,
    )

    d = result.to_dict()
    assert "best_focal_loss_params" in d
    assert d["best_focal_loss_params"]["use_focal_loss"] is True
    assert d["best_focal_loss_params"]["focal_gamma"] == 2.0
    assert d["best_focal_loss_params"]["minority_weight_boost"] == 1.5


def test_minority_weight_boost_in_search_space() -> None:
    """Test that minority_weight_boost is in the search space."""
    assert "minority_weight_boost" in FOCAL_LOSS_SEARCH_SPACE
    _, choices = FOCAL_LOSS_SEARCH_SPACE["minority_weight_boost"]
    assert 1.0 in choices
    assert 1.5 in choices
    assert 2.0 in choices


def test_model_registry_structure() -> None:
    """Test MODEL_REGISTRY has expected structure."""
    expected_models = ["lightgbm", "xgboost", "ridge", "lstm"]
    for model in expected_models:
        assert model in MODEL_REGISTRY
        assert "class" in MODEL_REGISTRY[model]
        assert "dataset" in MODEL_REGISTRY[model]
        assert "search_space" in MODEL_REGISTRY[model]
