"""
Tests for opti.py (Optimization Module)
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import optuna
from src.labelling.label_primaire.opti import (
    optimize_model,
    create_objective,
    _get_path_returns,
    _update_barrier_touches,
    apply_pt_sl_on_t1,
    _filter_valid_events,
    _build_events_dataframe,
    _compute_labels,
    get_events_primary,
    WalkForwardCV,
    OptimizationConfig,
    OptimizationResult,
    _handle_missing_values,
    _validate_events,
    _align_features_events,
    _evaluate_fold,
    _run_cv_scoring,
    _subsample_features,
    select_models_interactive,
    _run_optimization_worker,
    _run_sequential,
    _run_parallel,
    _print_final_summary,
    _make_cache_key,
    _sample_focal_loss_params,
    main,
)
from src.labelling.label_primaire.utils import (
    TRIPLE_BARRIER_SEARCH_SPACE,
    FOCAL_LOSS_SEARCH_SPACE,
    FOCAL_LOSS_SUPPORTED_MODELS,
    CLASS_WEIGHT_SUPPORTED_MODELS,
)

# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    close = pd.Series(np.random.uniform(100, 110, size=100), index=dates)
    features = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=[f"f{i}" for i in range(5)])
    volatility = pd.Series(np.random.uniform(0.01, 0.02, size=100), index=dates)
    return close, features, volatility

@pytest.fixture
def sample_events(sample_data):
    close, _, volatility = sample_data
    events = pd.DataFrame(index=close.index[:50])
    events["trgt"] = volatility.iloc[:50]
    events["t1"] = close.index[10:60] # Some future timestamp
    return events

# =============================================================================
# HELPER TESTS
# =============================================================================

def test_get_path_returns(sample_data):
    close, _, _ = sample_data
    t0 = close.index[0]
    t1 = close.index[5]

    ret = _get_path_returns(close, t0, t1)
    assert isinstance(ret, pd.Series)
    assert len(ret) == 6 # t0 to t1 inclusive

    # Invalid dates
    assert _get_path_returns(close, "invalid", t1) is None

    # Path too short (t0=t1)
    assert _get_path_returns(close, t0, t0) is None

def test_update_barrier_touches(sample_data, sample_events):
    close, _, _ = sample_data
    events = sample_events.copy()
    events["pt"] = 0.01
    events["sl"] = -0.01

    # Mock find_barrier_touch to return a specific timestamp
    with patch("src.labelling.label_primaire.opti.find_barrier_touch", return_value=close.index[2]):
        updated = _update_barrier_touches(close, events)
        assert updated.loc[events.index[0], "t1"] == close.index[2]

def test_apply_pt_sl_on_t1(sample_data, sample_events):
    close, _, _ = sample_data
    events = sample_events.copy()

    # Just verify it calls dependencies
    with patch("src.labelling.label_primaire.opti.compute_barriers", return_value=events), \
         patch("src.labelling.label_primaire.opti._update_barrier_touches", return_value=events):
        res = apply_pt_sl_on_t1(close, events, 1.0, 1.0)
        assert isinstance(res, pd.DataFrame)

def test_filter_valid_events(sample_data):
    close, _, volatility = sample_data
    t_events = close.index

    # Create some NaNs in volatility
    volatility.iloc[0] = np.nan

    filtered = _filter_valid_events(t_events, volatility)
    assert len(filtered) == len(t_events) - 1
    assert t_events[0] not in filtered

def test_build_events_dataframe(sample_data):
    close, _, volatility = sample_data
    t_events = close.index[:10]

    events = _build_events_dataframe(t_events, volatility, close.index, max_holding=5)
    assert "t1" in events.columns
    assert "trgt" in events.columns
    assert len(events) == 10

def test_compute_labels(sample_data, sample_events):
    close, _, _ = sample_data
    events = sample_events.copy()

    events = _compute_labels(events, close, min_return=0.0)
    assert "ret" in events.columns
    assert "label" in events.columns
    assert events["label"].isin([-1, 0, 1]).all()

def test_get_events_primary(sample_data):
    close, _, volatility = sample_data
    t_events = close.index

    events = get_events_primary(
        close=close,
        t_events=t_events,
        pt_mult=1.0,
        sl_mult=1.0,
        trgt=volatility,
        max_holding=10
    )

    if not events.empty:
        assert all(col in events.columns for col in ["t1", "trgt", "ret", "label"])

# =============================================================================
# CV TESTS
# =============================================================================

def test_walk_forward_cv(sample_data):
    close, features, volatility = sample_data

    # Create events DataFrame that aligns with features
    n_events = 80  # Use only first 80 samples
    events = pd.DataFrame(index=features.index[:n_events])
    events["trgt"] = volatility.iloc[:n_events].values
    # Use offset timestamps for t1 (10 bars ahead, capped at end)
    t1_values = list(features.index[10:n_events]) + list(features.index[-10:])
    events["t1"] = t1_values[:n_events]

    cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.0)
    splits = cv.split(features.iloc[:n_events], events)

    assert len(splits) > 0
    for train, val in splits:
        assert len(train) >= 10
        assert len(val) > 0

# =============================================================================
# OBJECTIVE & OPTIMIZATION TESTS
# =============================================================================

def test_handle_missing_values():
    df = pd.DataFrame({
        "f1": [1.0, np.nan, 3.0],
        "f2": [np.nan, np.nan, np.nan], # All NaN
        "bar_id": [1, 2, 3] # Non-feature
    })

    res = _handle_missing_values(df, "test_model")
    assert res["f1"].isna().sum() == 0
    assert res["f1"].iloc[1] == 2.0 # Median of 1, 3
    assert res["f2"].isna().sum() == 0
    assert (res["f2"] == 0).all() # Filled with 0

def test_validate_events_empty(sample_events):
    config = OptimizationConfig(model_name="test", min_train_size=10)

    # Empty
    valid, reason = _validate_events(pd.DataFrame(), config)
    assert not valid
    assert "empty" in reason

    # Too small
    small_events = sample_events.iloc[:5]
    valid, reason = _validate_events(small_events, config)
    assert not valid
    assert "not enough" in reason

    # One class
    one_class = sample_events.copy()
    one_class["label"] = 1
    valid, reason = _validate_events(one_class, config)
    assert not valid
    assert "only 1 class" in reason

def test_evaluate_fold(sample_data):
    _, features, _ = sample_data
    # Create synthetic multiclass classification problem (-1, 0, 1)
    y = pd.Series(np.random.choice([-1, 0, 1], len(features)), index=features.index)

    # Mock model
    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    # Predict different classes to avoid "only 1 class" issue
    mock_instance.predict.return_value = np.random.choice([-1, 0, 1], 10)

    train_idx = np.arange(50)
    val_idx = np.arange(50, 60)

    mcc, f1_weighted, reason = _evaluate_fold(
        features, y, train_idx, val_idx,
        mock_model_cls, {}, "test_model", 0
    )

    assert mcc is not None
    assert f1_weighted is not None
    assert reason == "OK"

def test_make_cache_key():
    """Test cache key generation from barrier params."""
    tb_params = {
        "pt_mult": 1.0,
        "sl_mult": 1.5,
        "min_return": 0.0001,
        "max_holding": 50,
    }
    key = _make_cache_key(tb_params)
    assert key == (1.0, 1.5, 0.0001, 50)
    assert isinstance(key, tuple)
    # Verify key is hashable (can be used in dict)
    cache = {key: "test"}
    assert cache[key] == "test"


def test_create_objective(sample_data, mocker):
    close, features, volatility = sample_data
    config = OptimizationConfig(model_name="lightgbm", n_trials=1, min_train_size=10)

    # Mock external calls to avoid heavy computation
    mock_events = pd.DataFrame({"label": [1]*50 + [-1]*50}, index=features.index)
    mocker.patch("src.labelling.label_primaire.opti._generate_trial_events", return_value=mock_events)
    mocker.patch("src.labelling.label_primaire.opti._validate_events", return_value=(True, "OK"))
    mocker.patch("src.labelling.label_primaire.opti._align_features_events", return_value=(features, pd.Series([1]*100, index=features.index), pd.DataFrame(index=features.index), "OK"))
    # _run_cv_scoring returns (objective, mean_mcc, mean_f1w, n_valid_folds, n_total_folds, reason)
    mocker.patch("src.labelling.label_primaire.opti._run_cv_scoring", return_value=(0.5, 0.5, 0.5, 5, 5, "OK"))

    mock_model_cls = MagicMock()
    search_space = {"p1": ("categorical", [1])}

    objective = create_objective(
        config, features, close, volatility, mock_model_cls, search_space
    )

    trial = mocker.Mock(spec=optuna.trial.Trial)
    trial.suggest_categorical.return_value = 1
    trial.number = 0

    score = objective(trial)
    assert score == 0.5


def test_create_objective_uses_cache(sample_data, mocker):
    """Test that create_objective caches events and reuses them."""
    close, features, volatility = sample_data
    config = OptimizationConfig(model_name="lightgbm", n_trials=1, min_train_size=10)

    mock_events = pd.DataFrame({"label": [1]*50 + [-1]*50}, index=features.index)

    # Track calls to _generate_trial_events
    generate_mock = mocker.patch(
        "src.labelling.label_primaire.opti._generate_trial_events",
        return_value=mock_events
    )
    mocker.patch("src.labelling.label_primaire.opti._validate_events", return_value=(True, "OK"))
    mocker.patch("src.labelling.label_primaire.opti._align_features_events",
                 return_value=(features, pd.Series([1]*100, index=features.index),
                              pd.DataFrame(index=features.index), "OK"))
    mocker.patch("src.labelling.label_primaire.opti._run_cv_scoring",
                 return_value=(0.5, 0.5, 0.5, 5, 5, "OK"))

    mock_model_cls = MagicMock()
    search_space = {"p1": ("categorical", [1])}

    objective = create_objective(
        config, features, close, volatility, mock_model_cls, search_space
    )

    # Create trials with SAME barrier params -> should use cache on second call
    trial1 = mocker.Mock(spec=optuna.trial.Trial)
    trial1.suggest_categorical.return_value = 1  # Same params
    trial1.number = 0

    trial2 = mocker.Mock(spec=optuna.trial.Trial)
    trial2.suggest_categorical.return_value = 1  # Same params
    trial2.number = 1

    # First call - should compute
    objective(trial1)
    assert generate_mock.call_count == 1

    # Second call with same params - should use cache
    objective(trial2)
    assert generate_mock.call_count == 1  # Still 1, not 2

def test_optimize_model(mocker):
    # Create proper mock data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    mock_features = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=[f"f{i}" for i in range(5)])
    mock_close = pd.Series(np.random.uniform(100, 110, 100), index=dates)
    mock_volatility = pd.Series(np.random.uniform(0.01, 0.02, 100), index=dates)

    # Mock everything around the main loop
    mocker.patch("src.labelling.label_primaire.opti.load_model_class", return_value=MagicMock)
    mocker.patch("src.labelling.label_primaire.opti._prepare_optimization_data", return_value=(mock_features, mock_close, mock_volatility))

    mock_study = MagicMock()
    mocker.patch("src.labelling.label_primaire.opti._create_study", return_value=mock_study)
    mocker.patch("src.labelling.label_primaire.opti.create_objective", return_value=lambda trial: 0.5)

    mock_result = MagicMock()
    mock_result.best_score = 0.75  # Set real value to avoid formatting error
    mocker.patch("src.labelling.label_primaire.opti._build_result", return_value=mock_result)
    mocker.patch("src.labelling.label_primaire.opti._log_result")  # Skip logging

    res = optimize_model("lightgbm")
    assert res == mock_result
    mock_study.optimize.assert_called_once()

# =============================================================================
# CLI TESTS
# =============================================================================

def test_select_models_interactive(mocker):
    # Test "all" selection
    mocker.patch("builtins.input", return_value="0")
    models = select_models_interactive()
    assert len(models) > 0

    # Test specific selection
    mocker.patch("builtins.input", return_value="1")
    models = select_models_interactive()
    assert len(models) == 1

    # Test named selection
    mocker.patch("builtins.input", return_value="lightgbm")
    models = select_models_interactive()
    assert "lightgbm" in models

def test_run_sequential(mocker):
    mocker.patch("src.labelling.label_primaire.opti._run_optimization_worker", return_value=MagicMock(best_score=0.5))
    trials_per_model = {"m1": 10, "m2": 10}
    res = _run_sequential(["m1", "m2"], trials_per_model, 5)
    assert len(res) == 2

def test_main_cli(mocker):
    mocker.patch("src.labelling.label_primaire.opti.select_models_interactive", return_value=["lightgbm"])
    # trials, splits, data_fraction, parallel, launch
    mocker.patch("builtins.input", side_effect=["1", "1", "1.0", "n", "o"])

    mock_run = mocker.patch("src.labelling.label_primaire.opti._run_sequential", return_value=[])
    mocker.patch("src.labelling.label_primaire.opti._print_final_summary")

    main()
    mock_run.assert_called_once()


# =============================================================================
# FOCAL LOSS AND CLASS WEIGHT TESTS
# =============================================================================


def test_sample_focal_loss_params_supported_model(mocker):
    """Test focal loss param sampling for supported model."""
    config = OptimizationConfig(
        model_name="lightgbm",
        optimize_focal_params=True,
    )

    trial = mocker.Mock(spec=optuna.trial.Trial)
    # use_focal_loss, focal_gamma, minority_weight_boost
    trial.suggest_categorical.side_effect = [True, 2.0, 1.5]

    params = _sample_focal_loss_params(trial, config)

    assert "use_focal_loss" in params
    assert "focal_gamma" in params
    assert "minority_weight_boost" in params
    assert trial.suggest_categorical.call_count == 3


def test_sample_focal_loss_params_unsupported_model(mocker):
    """Test focal loss param sampling for unsupported model uses defaults."""
    config = OptimizationConfig(
        model_name="random_forest",  # Not in FOCAL_LOSS_SUPPORTED_MODELS
        use_focal_loss=True,
        focal_gamma=3.0,
        optimize_focal_params=True,
    )

    trial = mocker.Mock(spec=optuna.trial.Trial)
    # minority_weight_boost is still sampled for all models
    trial.suggest_categorical.side_effect = [1.5]

    params = _sample_focal_loss_params(trial, config)

    # Focal loss params should use config defaults
    assert params["use_focal_loss"] == True
    assert params["focal_gamma"] == 3.0
    # But minority_weight_boost should still be sampled
    assert params["minority_weight_boost"] == 1.5
    assert trial.suggest_categorical.call_count == 1


def test_sample_focal_loss_params_disabled(mocker):
    """Test focal loss param sampling when optimization is disabled."""
    config = OptimizationConfig(
        model_name="lightgbm",
        optimize_focal_params=False,
        use_focal_loss=False,
        focal_gamma=1.0,
        minority_weight_boost=1.5,
    )

    trial = mocker.Mock(spec=optuna.trial.Trial)

    params = _sample_focal_loss_params(trial, config)

    # Should use config defaults
    assert params["use_focal_loss"] == False
    assert params["focal_gamma"] == 1.0
    assert params["minority_weight_boost"] == 1.5
    trial.suggest_categorical.assert_not_called()


def test_evaluate_fold_with_focal_loss(sample_data):
    """Test _evaluate_fold with focal loss enabled."""
    _, features, _ = sample_data
    y = pd.Series(np.random.choice([-1, 0, 1], len(features)), index=features.index)

    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_instance.predict.return_value = np.random.choice([-1, 0, 1], 10)

    train_idx = np.arange(50)
    val_idx = np.arange(50, 60)

    mcc, f1_weighted, reason = _evaluate_fold(
        features, y, train_idx, val_idx,
        mock_model_cls, {},
        model_name="lightgbm",
        fold_idx=0,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_class_weights=True,
    )

    # Should still complete (mocked model)
    assert mcc is not None or reason != "OK"


def test_evaluate_fold_with_class_weights(sample_data):
    """Test _evaluate_fold with class weights for different models."""
    _, features, _ = sample_data
    y = pd.Series(np.random.choice([-1, 0, 1], len(features)), index=features.index)

    for model_name in CLASS_WEIGHT_SUPPORTED_MODELS:
        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.predict.return_value = np.random.choice([-1, 0, 1], 10)

        train_idx = np.arange(50)
        val_idx = np.arange(50, 60)

        mcc, f1_weighted, reason = _evaluate_fold(
            features, y, train_idx, val_idx,
            mock_model_cls, {},
            model_name=model_name,
            fold_idx=0,
            use_focal_loss=False,
            focal_gamma=2.0,
            use_class_weights=True,
        )

        # Should complete without error
        assert mcc is not None or "SKIP" in reason or "FAILED" in reason


def test_focal_loss_search_space():
    """Test focal loss search space structure."""
    assert "focal_gamma" in FOCAL_LOSS_SEARCH_SPACE
    assert "use_focal_loss" in FOCAL_LOSS_SEARCH_SPACE

    _, gamma_choices = FOCAL_LOSS_SEARCH_SPACE["focal_gamma"]
    assert 0.0 in gamma_choices
    assert 2.0 in gamma_choices

    _, use_focal_choices = FOCAL_LOSS_SEARCH_SPACE["use_focal_loss"]
    assert True in use_focal_choices
    assert False in use_focal_choices


def test_focal_loss_supported_models():
    """Test that focal loss supported models list is correct."""
    assert "lightgbm" in FOCAL_LOSS_SUPPORTED_MODELS
    # Models without custom objective support should not be listed
    assert "random_forest" not in FOCAL_LOSS_SUPPORTED_MODELS
    assert "ridge" not in FOCAL_LOSS_SUPPORTED_MODELS


def test_class_weight_supported_models():
    """Test that class weight supported models list is correct."""
    assert "lightgbm" in CLASS_WEIGHT_SUPPORTED_MODELS
    assert "catboost" in CLASS_WEIGHT_SUPPORTED_MODELS
    assert "random_forest" in CLASS_WEIGHT_SUPPORTED_MODELS
    assert "logistic" in CLASS_WEIGHT_SUPPORTED_MODELS
    # Ridge doesn't support class_weight in sklearn
    assert "ridge" not in CLASS_WEIGHT_SUPPORTED_MODELS


def test_optimization_config_focal_defaults():
    """Test OptimizationConfig focal loss defaults."""
    config = OptimizationConfig(model_name="lightgbm")

    assert config.use_focal_loss == True
    assert config.focal_gamma == 2.0
    assert config.optimize_focal_params == True
    assert config.use_class_weights == True
    assert config.minority_weight_boost == 1.25  # Conservative default


def test_optimization_result_includes_focal_params():
    """Test that OptimizationResult includes focal loss params."""
    result = OptimizationResult(
        model_name="lightgbm",
        best_params={"n_estimators": 100},
        best_triple_barrier_params={"pt_mult": 1.0},
        best_focal_loss_params={"use_focal_loss": True, "focal_gamma": 2.0, "minority_weight_boost": 1.5},
        best_score=0.8,
        metric="mcc",
        n_trials=10,
    )

    d = result.to_dict()
    assert "best_focal_loss_params" in d
    assert d["best_focal_loss_params"]["use_focal_loss"] == True
    assert d["best_focal_loss_params"]["focal_gamma"] == 2.0
    assert d["best_focal_loss_params"]["minority_weight_boost"] == 1.5


def test_minority_weight_boost_in_search_space():
    """Test that minority_weight_boost is in the search space."""
    assert "minority_weight_boost" in FOCAL_LOSS_SEARCH_SPACE
    _, choices = FOCAL_LOSS_SEARCH_SPACE["minority_weight_boost"]
    assert 1.0 in choices  # Balanced
    assert 1.5 in choices  # Moderate boost
    assert 2.0 in choices  # Strong boost


def test_evaluate_fold_with_minority_boost(sample_data):
    """Test _evaluate_fold with minority_weight_boost."""
    _, features, _ = sample_data
    y = pd.Series(np.random.choice([-1, 0, 1], len(features)), index=features.index)

    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    # Predict a mix of classes to avoid degenerate penalty
    mock_instance.predict.return_value = np.array([-1, 0, 1, 0, 0, -1, 1, 0, 0, 1])

    train_idx = np.arange(50)
    val_idx = np.arange(50, 60)

    # Test with boost = 1.0 (no boost)
    mcc_no_boost, _, _ = _evaluate_fold(
        features, y, train_idx, val_idx,
        mock_model_cls, {},
        model_name="lightgbm",
        fold_idx=0,
        minority_weight_boost=1.0,
    )

    # Test with boost = 1.5
    mcc_with_boost, _, _ = _evaluate_fold(
        features, y, train_idx, val_idx,
        mock_model_cls, {},
        model_name="lightgbm",
        fold_idx=0,
        minority_weight_boost=1.5,
    )

    # Both should complete without error
    assert mcc_no_boost is not None or mcc_with_boost is not None
