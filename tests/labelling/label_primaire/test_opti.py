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
    main,
)
from src.labelling.label_primaire.utils import TRIPLE_BARRIER_SEARCH_SPACE

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
    res = _run_sequential(["m1", "m2"], 1, 1)
    assert len(res) == 2

def test_main_cli(mocker):
    mocker.patch("src.labelling.label_primaire.opti.select_models_interactive", return_value=["lightgbm"])
    # trials, splits, data_fraction, parallel, launch
    mocker.patch("builtins.input", side_effect=["1", "1", "1.0", "n", "o"])

    mock_run = mocker.patch("src.labelling.label_primaire.opti._run_sequential", return_value=[])
    mocker.patch("src.labelling.label_primaire.opti._print_final_summary")

    main()
    mock_run.assert_called_once()
