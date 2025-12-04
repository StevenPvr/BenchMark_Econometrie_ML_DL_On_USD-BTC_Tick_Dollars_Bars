"""Tests for src.labelling.label_meta.opti.logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import optuna

from src.labelling.label_meta.opti.logic import (
    WalkForwardCV,
    _sample_model_params,
    _evaluate_fold,
    create_objective,
)
from src.labelling.label_meta.utils import MetaOptimizationConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Generate sample features for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    features = pd.DataFrame(
        np.random.randn(100, 5),
        index=dates,
        columns=[f"f{i}" for i in range(5)],
    )
    return features


@pytest.fixture
def sample_meta_label() -> pd.Series:
    """Generate sample meta labels for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    return pd.Series(np.random.choice([0, 1], 100), index=dates)


# =============================================================================
# WALK-FORWARD CV TESTS
# =============================================================================


class TestWalkForwardCV:
    """Tests for WalkForwardCV class."""

    def test_init_default(self) -> None:
        """Test CV initialization with default parameters."""
        cv = WalkForwardCV()
        assert cv.n_splits == 5
        assert cv.min_train_size == 500
        assert cv.embargo_pct == 0.01

    def test_init_custom(self) -> None:
        """Test CV initialization with custom parameters."""
        cv = WalkForwardCV(n_splits=3, min_train_size=100, embargo_pct=0.05)
        assert cv.n_splits == 3
        assert cv.min_train_size == 100
        assert cv.embargo_pct == 0.05

    def test_split_basic(self, sample_features: pd.DataFrame) -> None:
        """Test basic split generation."""
        cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.0)
        splits = cv.split(sample_features)

        assert len(splits) > 0
        for train_idx, val_idx in splits:
            assert len(train_idx) >= 10
            assert len(val_idx) > 0
            # Train indices should be before validation indices
            if len(train_idx) > 0 and len(val_idx) > 0:
                assert train_idx[-1] < val_idx[0]

    def test_split_with_embargo(self, sample_features: pd.DataFrame) -> None:
        """Test split with embargo percentage."""
        cv = WalkForwardCV(n_splits=3, min_train_size=10, embargo_pct=0.05)
        splits = cv.split(sample_features)

        assert len(splits) >= 1

    def test_split_min_train_size_constraint(
        self, sample_features: pd.DataFrame
    ) -> None:
        """Test that min_train_size is respected."""
        cv = WalkForwardCV(n_splits=5, min_train_size=100)
        # With 100 samples and min_train_size=100, first folds should be skipped
        splits = cv.split(sample_features)

        for train_idx, _ in splits:
            assert len(train_idx) >= 100

    def test_split_no_valid_splits(self, sample_features: pd.DataFrame) -> None:
        """Test when no valid splits can be created."""
        cv = WalkForwardCV(n_splits=10, min_train_size=200)
        # With only 100 samples and min_train_size=200, no valid splits
        splits = cv.split(sample_features)

        assert len(splits) == 0


# =============================================================================
# PARAMETER SAMPLING TESTS
# =============================================================================


class TestSampleModelParams:
    """Tests for _sample_model_params function."""

    def test_sample_categorical(self, mocker: MagicMock) -> None:
        """Test sampling categorical parameters."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_categorical.return_value = "value1"

        search_space = {"param1": ("categorical", ["value1", "value2"])}
        config = MetaOptimizationConfig("primary", "meta", random_state=42)

        params = _sample_model_params(trial, search_space, config)

        assert params["random_state"] == 42
        assert params["param1"] == "value1"
        trial.suggest_categorical.assert_called_once()

    def test_sample_float(self, mocker: MagicMock) -> None:
        """Test sampling float parameters."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.5

        search_space = {"param1": ("float", [0.0, 1.0])}
        config = MetaOptimizationConfig("primary", "meta")

        params = _sample_model_params(trial, search_space, config)

        assert params["param1"] == 0.5
        trial.suggest_float.assert_called_once()

    def test_sample_float_log_scale(self, mocker: MagicMock) -> None:
        """Test sampling float parameters with log scale."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.01

        search_space = {"param1": ("float", [0.001, 1.0, "log"])}
        config = MetaOptimizationConfig("primary", "meta")

        params = _sample_model_params(trial, search_space, config)

        assert params["param1"] == 0.01
        trial.suggest_float.assert_called_once_with(
            "param1", 0.001, 1.0, log=True
        )

    def test_sample_int(self, mocker: MagicMock) -> None:
        """Test sampling integer parameters."""
        trial = mocker.Mock(spec=optuna.Trial)
        trial.suggest_int.return_value = 10

        search_space = {"param1": ("int", [1, 100])}
        config = MetaOptimizationConfig("primary", "meta")

        params = _sample_model_params(trial, search_space, config)

        assert params["param1"] == 10
        trial.suggest_int.assert_called_once()


# =============================================================================
# FOLD EVALUATION TESTS
# =============================================================================


class TestEvaluateFold:
    """Tests for _evaluate_fold function."""

    def test_evaluate_fold_basic(
        self, sample_features: pd.DataFrame, sample_meta_label: pd.Series
    ) -> None:
        """Test basic fold evaluation."""
        train_idx = np.arange(50)
        val_idx = np.arange(50, 70)

        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.predict.return_value = np.random.choice([0, 1], 20)

        precision = _evaluate_fold(
            sample_features,
            sample_meta_label,
            train_idx,
            val_idx,
            mock_model_cls,
            {"random_state": 42},
        )

        assert 0.0 <= precision <= 1.0
        mock_model_cls.assert_called_once()
        mock_instance.fit.assert_called_once()
        mock_instance.predict.assert_called_once()


# =============================================================================
# OBJECTIVE FUNCTION TESTS
# =============================================================================


class TestCreateObjective:
    """Tests for create_objective function."""

    def test_create_objective_callable(
        self, sample_features: pd.DataFrame, sample_meta_label: pd.Series
    ) -> None:
        """Test that create_objective returns a callable."""
        config = MetaOptimizationConfig(
            "primary", "meta", min_train_size=10, n_splits=3
        )

        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        mock_instance.predict.return_value = np.array([0, 1] * 10)

        search_space = {"n_estimators": ("int", [10, 100])}

        objective = create_objective(
            config,
            sample_features,
            sample_meta_label,
            mock_model_cls,
            search_space,
        )

        assert callable(objective)

    def test_create_objective_no_splits_raises(
        self, sample_features: pd.DataFrame, sample_meta_label: pd.Series
    ) -> None:
        """Test that create_objective raises when no valid splits."""
        config = MetaOptimizationConfig(
            "primary", "meta", min_train_size=200, n_splits=5
        )

        mock_model_cls = MagicMock()
        search_space = {"n_estimators": ("int", [10, 100])}

        with pytest.raises(ValueError, match="No valid CV splits"):
            create_objective(
                config,
                sample_features,
                sample_meta_label,
                mock_model_cls,
                search_space,
            )


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestMetaOptimizationConfig:
    """Tests for MetaOptimizationConfig dataclass."""

    def test_config_defaults(self) -> None:
        """Test config defaults."""
        config = MetaOptimizationConfig("primary", "meta")
        assert config.n_trials == 50
        assert config.n_splits == 5
        assert config.min_train_size == 500
        assert config.random_state == 42  # DEFAULT_RANDOM_STATE

    def test_config_custom(self) -> None:
        """Test config with custom values."""
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
        assert config.min_train_size == 200
        assert config.random_state == 123
