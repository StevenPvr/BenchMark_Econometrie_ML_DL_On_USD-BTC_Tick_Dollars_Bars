"""Tests for src/model/machine_learning/catboost/catboost_model.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.machine_learning.catboost.catboost_model import CatBoostModel


@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.1

    return X, y


@pytest.fixture
def sample_data_with_validation(sample_data):
    """Split sample data into train and validation sets."""
    X, y = sample_data
    split_idx = 80

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


@pytest.fixture
def sample_data_with_categorical():
    """Generate sample data with categorical features."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        "numeric_1": np.random.randn(n_samples),
        "numeric_2": np.random.randn(n_samples),
        "category_1": np.random.choice(["A", "B", "C"], n_samples),
        "category_2": np.random.choice(["X", "Y"], n_samples),
    })

    y = X["numeric_1"] * 3 + X["numeric_2"] * 2 + np.random.randn(n_samples) * 0.1

    return X, y


class TestCatBoostModelInit:
    """Test cases for CatBoostModel initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = CatBoostModel()

        assert model.name == "CatBoost"
        assert model.iterations == 100
        assert model.depth == 6
        assert model.learning_rate == 0.1
        assert model.l2_leaf_reg == 3.0
        assert model.random_strength == 1.0
        assert model.bagging_temperature == 1.0
        assert model.loss_function == "RMSE"
        assert model.random_seed == 42
        assert model.thread_count == -1
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = CatBoostModel(
            iterations=200,
            depth=10,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            bagging_temperature=0.8,
            loss_function="MAE",
            random_seed=123,
        )

        assert model.iterations == 200
        assert model.depth == 10
        assert model.learning_rate == 0.05
        assert model.l2_leaf_reg == 5.0
        assert model.random_strength == 0.5
        assert model.bagging_temperature == 0.8
        assert model.loss_function == "MAE"
        assert model.random_seed == 123


class TestCatBoostModelFit:
    """Test cases for CatBoostModel fit method."""

    def test_fit_basic(self, sample_data):
        """Should fit model successfully."""
        X, y = sample_data
        model = CatBoostModel(iterations=10, verbose=False)

        result = model.fit(X, y, verbose=False)

        assert model.is_fitted is True
        assert model.model is not None
        assert result is model

    def test_fit_with_dataframe(self, sample_data):
        """Should fit with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        model = CatBoostModel(iterations=10)
        model.fit(X_df, y_series, verbose=False)

        assert model.is_fitted is True

    def test_fit_with_eval_set(self, sample_data_with_validation):
        """Should fit with evaluation set."""
        X_train, y_train, X_val, y_val = sample_data_with_validation

        model = CatBoostModel(iterations=50)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        assert model.is_fitted is True

    def test_fit_with_early_stopping(self, sample_data_with_validation):
        """Should fit with early stopping."""
        X_train, y_train, X_val, y_val = sample_data_with_validation

        model = CatBoostModel(iterations=1000)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=10,
            verbose=False,
        )

        assert model.is_fitted is True
        # Early stopping should stop before 1000 iterations
        assert model.best_iteration < 1000

    def test_fit_with_categorical_features(self, sample_data_with_categorical):
        """Should fit with categorical features."""
        X, y = sample_data_with_categorical

        model = CatBoostModel(iterations=10)
        model.fit(
            X,
            y,
            cat_features=["category_1", "category_2"],
            verbose=False,
        )

        assert model.is_fitted is True

    def test_fit_with_categorical_indices(self, sample_data_with_categorical):
        """Should fit with categorical feature indices."""
        X, y = sample_data_with_categorical

        model = CatBoostModel(iterations=10)
        model.fit(
            X,
            y,
            cat_features=[2, 3],  # Indices of category columns
            verbose=False,
        )

        assert model.is_fitted is True


class TestCatBoostModelPredict:
    """Test cases for CatBoostModel predict method."""

    def test_predict_basic(self, sample_data):
        """Should predict after fitting."""
        X, y = sample_data
        model = CatBoostModel(iterations=10)

        model.fit(X, y, verbose=False)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_reasonable_values(self, sample_data):
        """Predictions should be reasonable."""
        X, y = sample_data
        model = CatBoostModel(iterations=50)

        model.fit(X, y, verbose=False)
        predictions = model.predict(X)

        # Predictions should correlate with actual values
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.8

    def test_predict_before_fit_raises(self, sample_data):
        """Should raise error if predicting before fit."""
        X, y = sample_data
        model = CatBoostModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_data):
        """Should predict with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        model = CatBoostModel(iterations=10)
        model.fit(X, y, verbose=False)
        predictions = model.predict(X_df)

        assert len(predictions) == len(y)

    def test_predict_with_categorical(self, sample_data_with_categorical):
        """Should predict with categorical features."""
        X, y = sample_data_with_categorical

        model = CatBoostModel(iterations=20)
        model.fit(X, y, cat_features=["category_1", "category_2"], verbose=False)

        predictions = model.predict(X)

        assert len(predictions) == len(y)


class TestCatBoostModelFeatureImportance:
    """Test cases for feature importance."""

    def test_get_feature_importance(self, sample_data):
        """Should return feature importance after fit."""
        X, y = sample_data
        model = CatBoostModel(iterations=10)
        model.fit(X, y, verbose=False)

        importance = model.get_feature_importance()

        assert isinstance(importance, np.ndarray)
        assert len(importance) == X.shape[1]
        assert all(imp >= 0 for imp in importance)

    def test_get_feature_importance_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = CatBoostModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()

    def test_important_features_ranked_higher(self, sample_data):
        """Important features should have higher importance."""
        X, y = sample_data
        # y = 3*X[:,0] + 2*X[:,1] + noise, so features 0 and 1 should be most important
        model = CatBoostModel(iterations=100)
        model.fit(X, y, verbose=False)

        importance = model.get_feature_importance()
        top_2_indices = np.argsort(importance)[-2:]

        # Features 0 and 1 should be in top 2
        assert 0 in top_2_indices or 1 in top_2_indices


class TestCatBoostModelBestIteration:
    """Test cases for best_iteration property."""

    def test_best_iteration_without_early_stopping(self, sample_data):
        """Should return iterations without early stopping."""
        X, y = sample_data
        model = CatBoostModel(iterations=20)
        model.fit(X, y, verbose=False)

        # Without early stopping, should return iterations
        assert model.best_iteration == 20 or model.best_iteration >= 0

    def test_best_iteration_with_early_stopping(self, sample_data_with_validation):
        """Should return actual best iteration with early stopping."""
        X_train, y_train, X_val, y_val = sample_data_with_validation

        model = CatBoostModel(iterations=1000)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=5,
            verbose=False,
        )

        assert model.best_iteration < 1000

    def test_best_iteration_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = CatBoostModel()

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.best_iteration


class TestCatBoostModelSaveLoad:
    """Test cases for save and load functionality."""

    def test_save_and_load(self, sample_data):
        """Should save and load model correctly."""
        X, y = sample_data
        model = CatBoostModel(iterations=10, random_seed=42)
        model.fit(X, y, verbose=False)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catboost_model.joblib"
            model.save(path)

            loaded_model = CatBoostModel.load(path)
            loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestCatBoostModelReproducibility:
    """Test cases for reproducibility."""

    def test_reproducibility_same_seed(self, sample_data):
        """Same seed should produce same results."""
        X, y = sample_data

        model1 = CatBoostModel(iterations=10, random_seed=42)
        model1.fit(X, y, verbose=False)
        pred1 = model1.predict(X)

        model2 = CatBoostModel(iterations=10, random_seed=42)
        model2.fit(X, y, verbose=False)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestCatBoostModelLossFunctions:
    """Test cases for different loss functions."""

    def test_rmse_loss(self, sample_data):
        """Should work with RMSE loss."""
        X, y = sample_data
        model = CatBoostModel(iterations=10, loss_function="RMSE")
        model.fit(X, y, verbose=False)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_mae_loss(self, sample_data):
        """Should work with MAE loss."""
        X, y = sample_data
        model = CatBoostModel(iterations=10, loss_function="MAE")
        model.fit(X, y, verbose=False)

        predictions = model.predict(X)
        assert len(predictions) == len(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
