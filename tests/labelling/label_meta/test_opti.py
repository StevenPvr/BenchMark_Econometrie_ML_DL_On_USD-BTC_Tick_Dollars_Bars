"""Tests for src.labelling.label_meta.opti."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import optuna

from src.labelling.label_meta.opti import (
    get_events_meta,
    get_bins,
    WalkForwardCV,
    _resolve_n_jobs,
    _split_t_events,
    _filter_valid_events_meta,
    _build_events_dataframe_meta,
    _get_path_returns,
    _sample_barrier_params,
    _validate_meta_events,
    _align_features_events_meta,
    _sample_model_params,
    _evaluate_fold_meta,
    _run_cv_scoring_meta,
    _process_chunk_events_meta,
    _generate_trial_events_meta,
    _subsample_features,
    create_objective,
    TRIPLE_BARRIER_SEARCH_SPACE,
)
from src.labelling.label_meta.utils import MetaOptimizationConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    close = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
    volatility = pd.Series(np.abs(np.random.randn(100)) * 0.01 + 0.01, index=dates)
    side = pd.Series(np.random.choice([1, -1], 100), index=dates)
    return close, volatility, side


@pytest.fixture
def sample_features():
    """Generate sample features for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    features = pd.DataFrame(
        np.random.randn(100, 5),
        index=dates,
        columns=[f"f{i}" for i in range(5)]
    )
    return features


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    def test_resolve_n_jobs_default(self):
        """Test resolving n_jobs with default value."""
        import multiprocessing
        result = _resolve_n_jobs(None)
        assert result == multiprocessing.cpu_count()

    def test_resolve_n_jobs_zero(self):
        """Test resolving n_jobs with zero."""
        import multiprocessing
        result = _resolve_n_jobs(0)
        assert result == multiprocessing.cpu_count()

    def test_resolve_n_jobs_specific(self):
        """Test resolving n_jobs with specific value."""
        result = _resolve_n_jobs(2)
        assert result == 2

    def test_split_t_events(self):
        """Test splitting events into chunks."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        t_events = pd.DatetimeIndex(dates)

        chunks = _split_t_events(t_events, 4)

        assert len(chunks) == 4
        total = sum(len(c) for c in chunks)
        assert total == 100

    def test_filter_valid_events_meta(self, sample_data):
        """Test filtering valid events."""
        close, volatility, side = sample_data

        # Create t_events that includes some indices not in volatility
        extra_dates = pd.date_range("2020-01-01", periods=5, freq="h")
        t_events = pd.DatetimeIndex(list(extra_dates) + list(close.index))

        filtered = _filter_valid_events_meta(t_events, volatility, side)

        # Should filter out the 5 extra dates not in volatility
        assert len(filtered) == len(close.index)
        assert not any(d in filtered for d in extra_dates)

    def test_build_events_dataframe_meta(self, sample_data):
        """Test building events dataframe."""
        close, volatility, side = sample_data
        t_events = close.index[:50]

        events = _build_events_dataframe_meta(t_events, volatility, side)

        assert "trgt" in events.columns
        assert "side" in events.columns
        assert len(events) == 50

    def test_get_path_returns_success(self, sample_data):
        """Test getting path returns."""
        close, _, _ = sample_data
        t0 = close.index[0]
        t1 = close.index[10]

        ret = _get_path_returns(close, t0, t1)

        assert isinstance(ret, pd.Series)
        assert len(ret) == 11

    def test_get_path_returns_invalid(self, sample_data):
        """Test getting path returns with invalid dates."""
        close, _, _ = sample_data

        # Invalid dates
        result = _get_path_returns(close, "invalid", close.index[5])
        assert result is None

    def test_get_path_returns_too_short(self, sample_data):
        """Test getting path returns when path is too short."""
        close, _, _ = sample_data
        t0 = close.index[0]

        result = _get_path_returns(close, t0, t0)
        assert result is None


# =============================================================================
# META-LABELING CORE TESTS
# =============================================================================


class TestMetaLabelingCore:
    def test_get_events_meta_basic(self, sample_data):
        """Test getting meta events."""
        close, volatility, side = sample_data
        t_events = pd.DatetimeIndex(close.index[:50])

        events = get_events_meta(
            close=close,
            t_events=t_events,
            pt_mult=1.0,
            sl_mult=1.0,
            trgt=volatility,
            max_holding=10,
            side=side,
        )

        if not events.empty:
            assert "t1" in events.columns
            assert "trgt" in events.columns
            assert "side" in events.columns

    def test_get_events_meta_empty(self, sample_data):
        """Test getting meta events with empty input."""
        close, volatility, side = sample_data

        # Empty t_events
        t_events = pd.DatetimeIndex([])

        events = get_events_meta(
            close=close,
            t_events=t_events,
            pt_mult=1.0,
            sl_mult=1.0,
            trgt=volatility,
            max_holding=10,
            side=side,
        )

        assert events.empty

    def test_get_bins_basic(self, sample_data):
        """Test getting bins for meta labels."""
        close, volatility, side = sample_data

        # Create simple events DataFrame
        events = pd.DataFrame({
            "t1": close.index[10:20],
            "trgt": volatility.iloc[:10].values,
            "side": side.iloc[:10].values,
            "pt": [0.01] * 10,
            "sl": [-0.01] * 10,
        }, index=close.index[:10])

        result = get_bins(events, close)

        assert "ret" in result.columns
        assert "bin" in result.columns
        assert result["bin"].isin([0, 1]).all()


# =============================================================================
# WALK-FORWARD CV TESTS
# =============================================================================


class TestWalkForwardCV:
    def test_cv_init(self):
        """Test CV initialization."""
        cv = WalkForwardCV(n_splits=5, min_train_size=100, embargo_pct=0.01)

        assert cv.n_splits == 5
        assert cv.min_train_size == 100
        assert cv.embargo_pct == 0.01

    def test_cv_split(self, sample_features):
        """Test CV split generation."""
        events = pd.DataFrame(index=sample_features.index)
        events["t1"] = sample_features.index.shift(5, freq="h")

        cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.0)
        splits = cv.split(sample_features, events)

        assert len(splits) > 0
        for train_idx, val_idx in splits:
            assert len(train_idx) >= 10
            assert len(val_idx) > 0

    def test_cv_process_split_too_small(self, sample_features):
        """Test that small splits are filtered out."""
        events = pd.DataFrame(index=sample_features.index[:20])

        cv = WalkForwardCV(n_splits=3, min_train_size=50)  # Large min train size
        splits = cv.split(sample_features.iloc[:20], events)

        # Should have fewer splits due to min_train_size constraint
        assert len(splits) < 3


# =============================================================================
# OPTUNA OBJECTIVE TESTS
# =============================================================================


class TestOptunaObjective:
    def test_sample_barrier_params(self, mocker):
        """Test sampling barrier parameters."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        params = _sample_barrier_params(trial)

        assert "pt_mult" in params
        assert "sl_mult" in params
        assert "max_holding" in params

    def test_validate_meta_events_empty(self):
        """Test validating empty events."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=10)

        valid, reason = _validate_meta_events(pd.DataFrame(), config)

        assert not valid
        assert "empty" in reason

    def test_validate_meta_events_too_small(self, sample_features):
        """Test validating events that are too small."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=100)

        events = pd.DataFrame({"bin": [1] * 10}, index=sample_features.index[:10])
        valid, reason = _validate_meta_events(events, config)

        assert not valid
        assert "not enough" in reason

    def test_validate_meta_events_one_class(self, sample_features):
        """Test validating events with only one class."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=5)

        events = pd.DataFrame({"bin": [1] * 50}, index=sample_features.index[:50])
        valid, reason = _validate_meta_events(events, config)

        assert not valid
        assert "only one class" in reason

    def test_validate_meta_events_valid(self, sample_features):
        """Test validating valid events."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=5)

        events = pd.DataFrame(
            {"bin": [0] * 25 + [1] * 25},
            index=sample_features.index[:50]
        )
        valid, reason = _validate_meta_events(events, config)

        assert valid
        assert reason == "OK"

    def test_align_features_events_meta(self, sample_features):
        """Test aligning features with events."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=5)

        events = pd.DataFrame(
            {"bin": [0, 1] * 25},
            index=sample_features.index[:50]
        )

        X, y, events_aligned = _align_features_events_meta(
            sample_features, events, config
        )

        assert X is not None
        assert y is not None
        assert len(X) == len(events)

    def test_align_features_events_meta_too_small(self, sample_features):
        """Test aligning when result is too small."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=100)

        events = pd.DataFrame(
            {"bin": [0, 1] * 5},
            index=sample_features.index[:10]
        )

        X, y, events_aligned = _align_features_events_meta(
            sample_features, events, config
        )

        assert X is None
        assert y is None

    def test_sample_model_params(self, mocker):
        """Test sampling model parameters."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        search_space = {"param1": ("categorical", [1, 2, 3])}
        config = MetaOptimizationConfig("primary", "meta", random_state=42)

        params = _sample_model_params(trial, search_space, config)

        assert "random_state" in params
        assert params["random_state"] == 42
        assert "param1" in params


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    def test_create_objective(self, sample_data, sample_features, mocker):
        """Test creating objective function."""
        close, volatility, side = sample_data
        config = MetaOptimizationConfig("primary", "meta", min_train_size=5, n_splits=2)

        # Mock model class
        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.predict.return_value = np.array([0, 1] * 10)

        search_space = {"n_estimators": ("categorical", [10])}

        # Mock expensive operations
        mocker.patch(
            "src.labelling.label_meta.opti._generate_trial_events_meta",
            return_value=pd.DataFrame({
                "bin": [0, 1] * 25,
                "t1": sample_features.index[:50],
            }, index=sample_features.index[:50])
        )

        objective = create_objective(
            config,
            sample_features,
            close,
            volatility,
            side,
            mock_model_cls,
            search_space,
        )

        assert callable(objective)


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestAdditionalCoverage:
    def test_get_events_meta_basic_call(self, sample_data):
        """Test get_events_meta basic call."""
        close, volatility, side = sample_data
        t_events = pd.DatetimeIndex(close.index[:30])

        events = get_events_meta(
            close=close,
            t_events=t_events,
            pt_mult=1.0,
            sl_mult=1.0,
            trgt=volatility,
            max_holding=5,
            side=side,
        )

        # Should return events (may be empty depending on data)
        assert isinstance(events, pd.DataFrame)

    def test_get_bins_with_different_sides(self, sample_data):
        """Test get_bins with both long and short positions."""
        close, volatility, side = sample_data

        # Create events with mixed sides
        events = pd.DataFrame({
            "t1": close.index[5:15],
            "trgt": volatility.iloc[:10].values,
            "side": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            "pt": [0.02] * 10,
            "sl": [-0.01] * 10,
        }, index=close.index[:10])

        result = get_bins(events, close)

        assert "bin" in result.columns
        assert result["bin"].isin([0, 1]).all()

    def test_cv_with_embargo(self, sample_features):
        """Test CV with embargo percentage."""
        events = pd.DataFrame(index=sample_features.index)
        events["t1"] = sample_features.index.shift(3, freq="h")

        cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.05)
        splits = cv.split(sample_features, events)

        assert len(splits) >= 1

    def test_sample_barrier_params_different_values(self, mocker):
        """Test sampling with different parameter values."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_categorical.side_effect = [2.0, 1.5, 15]  # pt, sl, max_holding

        params = _sample_barrier_params(trial)

        assert params["pt_mult"] == 2.0
        assert params["sl_mult"] == 1.5
        assert params["max_holding"] == 15

    def test_validate_meta_events_sufficient(self, sample_features):
        """Test validation with sufficient samples."""
        config = MetaOptimizationConfig("primary", "meta", min_train_size=10)

        events = pd.DataFrame(
            {"bin": [0] * 50 + [1] * 50},
            index=sample_features.index
        )

        valid, reason = _validate_meta_events(events, config)
        assert valid
        assert reason == "OK"

    def test_triple_barrier_search_space(self):
        """Test that search space is properly defined."""
        assert "pt_mult" in TRIPLE_BARRIER_SEARCH_SPACE
        assert "sl_mult" in TRIPLE_BARRIER_SEARCH_SPACE
        assert "max_holding" in TRIPLE_BARRIER_SEARCH_SPACE

        # Verify types
        for key, value in TRIPLE_BARRIER_SEARCH_SPACE.items():
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert value[0] == "categorical"

    def test_get_path_returns_normal_case(self, sample_data):
        """Test path returns for normal case."""
        close, _, _ = sample_data

        t0 = close.index[5]
        t1 = close.index[15]

        ret = _get_path_returns(close, t0, t1)

        assert isinstance(ret, pd.Series)
        assert len(ret) == 11  # 15-5+1 bars

    def test_build_events_dataframe_meta_alignment(self, sample_data):
        """Test that events dataframe aligns correctly."""
        close, volatility, side = sample_data
        t_events = close.index[10:30]

        events = _build_events_dataframe_meta(t_events, volatility, side)

        assert len(events) == 20
        assert events.index.equals(t_events)
        assert (events["trgt"] == volatility.loc[t_events]).all()
        assert (events["side"] == side.loc[t_events]).all()

    def test_cv_iterator(self, sample_features):
        """Test WalkForwardCV as an iterator."""
        events = pd.DataFrame(index=sample_features.index[:50])
        events["t1"] = sample_features.index[5:55]

        cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.0)

        # Test that split returns list of tuples
        splits = cv.split(sample_features.iloc[:50], events)
        assert isinstance(splits, list)
        assert all(isinstance(s, tuple) for s in splits)
        assert all(len(s) == 2 for s in splits)

    def test_meta_config_creation(self):
        """Test MetaOptimizationConfig creation with various params."""
        config = MetaOptimizationConfig(
            primary_model_name="lightgbm",
            meta_model_name="xgboost",
            n_trials=100,
            n_splits=10,
            min_train_size=200,
            random_state=123,
        )

        assert config.primary_model_name == "lightgbm"
        assert config.meta_model_name == "xgboost"
        assert config.n_trials == 100
        assert config.n_splits == 10
        assert config.random_state == 123

    def test_get_events_meta_with_small_data(self, sample_data):
        """Test get_events_meta with minimal data."""
        close, volatility, side = sample_data

        # Use very small subset
        t_events = pd.DatetimeIndex(close.index[5:10])

        events = get_events_meta(
            close=close,
            t_events=t_events,
            pt_mult=1.0,
            sl_mult=1.0,
            trgt=volatility,
            max_holding=3,
            side=side,
        )

        assert isinstance(events, pd.DataFrame)

    def test_get_bins_empty_events(self, sample_data):
        """Test get_bins with minimal events."""
        close, _, _ = sample_data

        # Create minimal events
        events = pd.DataFrame({
            "t1": [close.index[5]],
            "trgt": [0.01],
            "side": [1],
            "pt": [0.02],
            "sl": [-0.01],
        }, index=[close.index[0]])

        result = get_bins(events, close)

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "bin" in result.columns


# =============================================================================
# CV AND FOLD EVALUATION TESTS
# =============================================================================


class TestCVAndFoldEvaluation:
    def test_evaluate_fold_meta_basic(self, sample_features):
        """Test fold evaluation with mock model."""
        X = sample_features
        y = pd.Series(np.random.choice([0, 1], len(X)), index=X.index)

        train_idx = np.arange(50)
        val_idx = np.arange(50, 70)

        # Create mock model class
        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.predict.return_value = np.random.choice([0, 1], 20)

        score, bal_acc, f1_w = _evaluate_fold_meta(
            X, y, train_idx, val_idx, mock_model_cls, {"random_state": 42}
        )

        assert score is not None
        assert bal_acc is not None
        assert f1_w is not None
        assert 0 <= score <= 1

    def test_evaluate_fold_meta_single_class(self, sample_features):
        """Test fold evaluation when train has only one class."""
        X = sample_features
        # All same class in training
        y = pd.Series([0] * len(X), index=X.index)

        train_idx = np.arange(50)
        val_idx = np.arange(50, 70)

        mock_model_cls = MagicMock()

        score, bal_acc, f1_w = _evaluate_fold_meta(
            X, y, train_idx, val_idx, mock_model_cls, {}
        )

        # Should return None for single class
        assert score is None
        assert bal_acc is None
        assert f1_w is None

    def test_run_cv_scoring_meta(self, sample_features, mocker):
        """Test CV scoring integration."""
        X = sample_features.iloc[:80]
        y = pd.Series(np.random.choice([0, 1], 80), index=X.index)
        events = pd.DataFrame(index=X.index)
        events["t1"] = X.index.shift(5, freq="h")

        config = MetaOptimizationConfig("primary", "meta", n_splits=3, min_train_size=10)

        # Mock model
        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.predict.return_value = np.random.choice([0, 1], 20)

        try:
            score, bal_acc, f1_w = _run_cv_scoring_meta(
                X, y, events, mock_model_cls, {"random_state": 42}, config
            )
            assert 0 <= score <= 1
        except optuna.TrialPruned:
            # Expected if CV produces no valid folds
            pass

    def test_process_chunk_events_meta(self, sample_data):
        """Test processing a chunk of events."""
        close, volatility, side = sample_data
        chunk = pd.DatetimeIndex(close.index[:20])

        tb_params = {"pt_mult": 1.0, "sl_mult": 1.0, "max_holding": 5}

        result = _process_chunk_events_meta(chunk, close, volatility, side, tb_params)

        assert isinstance(result, pd.DataFrame)

    def test_generate_trial_events_meta_serial(self, sample_data, sample_features):
        """Test serial event generation."""
        close, volatility, side = sample_data
        config = MetaOptimizationConfig(
            "primary", "meta",
            parallelize_labeling=False,
            parallel_min_events=1000,
        )

        tb_params = {"pt_mult": 1.0, "sl_mult": 1.0, "max_holding": 5}

        result = _generate_trial_events_meta(
            sample_features.iloc[:50], close, volatility, side, tb_params, config
        )

        assert isinstance(result, pd.DataFrame)

    def test_subsample_features(self, sample_features):
        """Test feature subsampling."""
        config = MetaOptimizationConfig(
            "primary", "meta",
            data_fraction=0.5,
            min_train_size=5,
            n_splits=3,
        )

        result = _subsample_features(sample_features, config)

        assert len(result) < len(sample_features)

    def test_subsample_features_full(self, sample_features):
        """Test no subsampling when fraction is 1.0."""
        config = MetaOptimizationConfig(
            "primary", "meta",
            data_fraction=1.0,
        )

        result = _subsample_features(sample_features, config)

        assert len(result) == len(sample_features)


# =============================================================================
# WALK FORWARD CV DETAILED TESTS
# =============================================================================


class TestWalkForwardCVDetailed:
    def test_apply_purging(self, sample_features):
        """Test purging removes overlapping samples."""
        cv = WalkForwardCV(n_splits=3, min_train_size=10)

        # Create events with overlapping t1
        events = pd.DataFrame(index=sample_features.index)
        events["t1"] = sample_features.index.shift(20, freq="h")  # Long overlap

        train_idx = np.arange(50)
        val_idx = np.arange(50, 70)

        purged = cv._apply_purging(train_idx, val_idx, sample_features, events)

        # Some indices should be removed
        assert len(purged) <= len(train_idx)

    def test_process_split_with_embargo(self, sample_features):
        """Test split processing with embargo."""
        cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.05)

        events = pd.DataFrame(index=sample_features.index)

        train_idx = np.arange(50)
        val_idx = np.arange(50, 70)

        result = cv._process_split(train_idx, val_idx, sample_features, events, 5)

        if result is not None:
            train_result, val_result = result
            # Embargo should reduce training size
            assert len(train_result) <= len(train_idx) - 5

    def test_process_split_too_few_samples(self, sample_features):
        """Test that small splits return None."""
        cv = WalkForwardCV(n_splits=3, min_train_size=100)

        events = pd.DataFrame(index=sample_features.index)

        train_idx = np.arange(50)  # Less than min_train_size
        val_idx = np.arange(50, 70)

        result = cv._process_split(train_idx, val_idx, sample_features, events, 0)

        assert result is None
