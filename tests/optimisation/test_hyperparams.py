"""Tests for src/optimisation/hyperparams.py module."""

from __future__ import annotations

import os
import sys

import optuna
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.optimisation.hyperparams import (
    RidgeHyperparams,
    LassoHyperparams,
    XGBoostHyperparams,
    LightGBMHyperparams,
    CatBoostHyperparams,
    RandomForestHyperparams,
    LSTMHyperparams,
    get_hyperparam_space,
    list_available_models,
    DEFAULT_HYPERPARAMS,
)


class TestRidgeHyperparams:
    """Test cases for RidgeHyperparams class."""

    def test_suggest_returns_dict(self):
        """Should return dictionary of hyperparameters."""
        space = RidgeHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        assert isinstance(params, dict)
        assert "alpha" in params

    def test_alpha_is_positive(self):
        """Alpha should be positive."""
        space = RidgeHyperparams()
        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            params = space.suggest(trial)
            assert params["alpha"] > 0

    def test_custom_alpha_range(self):
        """Should use custom alpha range."""
        space = RidgeHyperparams(alpha_log_range=(-2.0, 2.0))
        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            params = space.suggest(trial)
            assert 10**-2.0 <= params["alpha"] <= 10**2.0


class TestLassoHyperparams:
    """Test cases for LassoHyperparams class."""

    def test_suggest_returns_dict(self):
        """Should return dictionary of hyperparameters."""
        space = LassoHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        assert isinstance(params, dict)
        assert "alpha" in params

    def test_alpha_is_positive(self):
        """Alpha should be positive."""
        space = LassoHyperparams()
        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            params = space.suggest(trial)
            assert params["alpha"] > 0


class TestXGBoostHyperparams:
    """Test cases for XGBoostHyperparams class."""

    def test_suggest_returns_expected_keys(self):
        """Should return all expected XGBoost hyperparameters."""
        space = XGBoostHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        expected_keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_weight",
        ]
        for key in expected_keys:
            assert key in params

    def test_values_in_range(self):
        """Values should be within specified ranges."""
        space = XGBoostHyperparams(
            n_estimators_range=(100, 200),
            max_depth_range=(3, 6),
        )
        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            params = space.suggest(trial)
            assert 100 <= params["n_estimators"] <= 200
            assert 3 <= params["max_depth"] <= 6


class TestLightGBMHyperparams:
    """Test cases for LightGBMHyperparams class."""

    def test_suggest_returns_expected_keys(self):
        """Should return all expected LightGBM hyperparameters."""
        space = LightGBMHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        expected_keys = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        ]
        for key in expected_keys:
            assert key in params


class TestCatBoostHyperparams:
    """Test cases for CatBoostHyperparams class."""

    def test_suggest_returns_expected_keys(self):
        """Should return all expected CatBoost hyperparameters."""
        space = CatBoostHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        expected_keys = [
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "bagging_temperature",
            "random_strength",
        ]
        for key in expected_keys:
            assert key in params


class TestRandomForestHyperparams:
    """Test cases for RandomForestHyperparams class."""

    def test_suggest_returns_expected_keys(self):
        """Should return all expected Random Forest hyperparameters."""
        space = RandomForestHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        expected_keys = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        ]
        for key in expected_keys:
            assert key in params

    def test_max_features_options(self):
        """max_features should be from valid options."""
        space = RandomForestHyperparams(max_features=["sqrt", "log2"])
        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            params = space.suggest(trial)
            assert params["max_features"] in ["sqrt", "log2"]


class TestLSTMHyperparams:
    """Test cases for LSTMHyperparams class."""

    def test_suggest_returns_expected_keys(self):
        """Should return all expected LSTM hyperparameters."""
        space = LSTMHyperparams()
        study = optuna.create_study()
        trial = study.ask()

        params = space.suggest(trial)

        expected_keys = [
            "hidden_size",
            "num_layers",
            "dropout",
            "sequence_length",
            "learning_rate",
            "batch_size",
            "epochs",
        ]
        for key in expected_keys:
            assert key in params

    def test_dropout_zero_with_single_layer(self):
        """Dropout should be 0 when num_layers is 1."""
        space = LSTMHyperparams(num_layers_range=(1, 1))  # Force single layer
        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            params = space.suggest(trial)
            if params["num_layers"] == 1:
                assert params["dropout"] == 0.0


class TestGetHyperparamSpace:
    """Test cases for get_hyperparam_space function."""

    def test_returns_correct_space(self):
        """Should return correct space for each model."""
        space = get_hyperparam_space("xgboost")
        assert isinstance(space, XGBoostHyperparams)

    def test_case_insensitive(self):
        """Should be case insensitive."""
        space1 = get_hyperparam_space("XGBoost")
        space2 = get_hyperparam_space("xgboost")
        space3 = get_hyperparam_space("XGBOOST")

        assert type(space1) == type(space2) == type(space3)

    def test_accepts_aliases(self):
        """Should accept model name aliases."""
        assert isinstance(get_hyperparam_space("xgb"), XGBoostHyperparams)
        assert isinstance(get_hyperparam_space("lgbm"), LightGBMHyperparams)
        assert isinstance(get_hyperparam_space("rf"), RandomForestHyperparams)
        assert isinstance(get_hyperparam_space("cb"), CatBoostHyperparams)

    def test_raises_for_unknown_model(self):
        """Should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_hyperparam_space("unknown_model")


class TestListAvailableModels:
    """Test cases for list_available_models function."""

    def test_returns_list(self):
        """Should return a list."""
        result = list_available_models()
        assert isinstance(result, list)

    def test_contains_expected_models(self):
        """Should contain all expected models."""
        result = list_available_models()

        expected = ["ridge", "lasso", "xgboost", "lightgbm", "catboost", "randomforest", "lstm"]
        for model in expected:
            assert model in result


class TestDefaultHyperparams:
    """Test cases for DEFAULT_HYPERPARAMS dictionary."""

    def test_all_spaces_have_suggest_method(self):
        """All spaces should have a suggest method."""
        for name, space in DEFAULT_HYPERPARAMS.items():
            assert hasattr(space, "suggest"), f"{name} missing suggest method"

    def test_all_spaces_return_dict(self):
        """All spaces should return a dictionary from suggest."""
        study = optuna.create_study()

        for name, space in DEFAULT_HYPERPARAMS.items():
            trial = study.ask()
            params = space.suggest(trial)
            assert isinstance(params, dict), f"{name} did not return dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
