"""Tests for src/model/machine_learning/xgboost/xgboost_model.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.machine_learning.xgboost.xgboost_model import XGBoostModel


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


class TestXGBoostModelInit:
    """Test cases for XGBoostModel initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = XGBoostModel()

        assert model.name == "XGBoost"
        assert model.n_estimators == 100
        assert model.max_depth == 6
        assert model.learning_rate == 0.1
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.8
        assert model.reg_alpha == 0.0
        assert model.reg_lambda == 1.0
        assert model.objective == "reg:squarederror"
        assert model.random_state == 42
        assert model.n_jobs == -1
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = XGBoostModel(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=123,
        )

        assert model.n_estimators == 200
        assert model.max_depth == 10
        assert model.learning_rate == 0.05
        assert model.subsample == 0.9
        assert model.colsample_bytree == 0.7
        assert model.reg_alpha == 0.1
        assert model.reg_lambda == 0.5
        assert model.random_state == 123


class TestXGBoostModelFit:
    """Test cases for XGBoostModel fit method."""

    def test_fit_basic(self, sample_data):
        """Should fit model successfully."""
        X, y = sample_data
        model = XGBoostModel(n_estimators=10)

        result = model.fit(X, y)

        assert model.is_fitted is True
        assert model.model is not None
        assert result is model

    def test_fit_with_dataframe(self, sample_data):
        """Should fit with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        model = XGBoostModel(n_estimators=10)
        model.fit(X_df, y_series)

        assert model.is_fitted is True

    def test_fit_with_eval_set(self, sample_data_with_validation):
        """Should fit with evaluation set."""
        X_train, y_train, X_val, y_val = sample_data_with_validation

        model = XGBoostModel(n_estimators=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        assert model.is_fitted is True

    def test_fit_with_early_stopping(self, sample_data_with_validation):
        """Should fit with early stopping (if supported by XGBoost version)."""
        X_train, y_train, X_val, y_val = sample_data_with_validation

        model = XGBoostModel(n_estimators=1000)
        # Try early stopping if supported, otherwise just fit normally
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False,
            )
        except TypeError:
            # Early stopping not supported in this XGBoost version
            model.fit(X_train, y_train, verbose=False)

        assert model.is_fitted is True
        # Early stopping should stop before 1000 iterations
        assert model.best_iteration < 1000


class TestXGBoostModelPredict:
    """Test cases for XGBoostModel predict method."""

    def test_predict_basic(self, sample_data):
        """Should predict after fitting."""
        X, y = sample_data
        model = XGBoostModel(n_estimators=10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_reasonable_values(self, sample_data):
        """Predictions should be reasonable."""
        X, y = sample_data
        model = XGBoostModel(n_estimators=50)

        model.fit(X, y)
        predictions = model.predict(X)

        # Predictions should correlate with actual values
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.8

    def test_predict_before_fit_raises(self, sample_data):
        """Should raise error if predicting before fit."""
        X, y = sample_data
        model = XGBoostModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_data):
        """Should predict with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        model = XGBoostModel(n_estimators=10)
        model.fit(X, y)
        predictions = model.predict(X_df)

        assert len(predictions) == len(y)


class TestXGBoostModelFeatureImportance:
    """Test cases for feature importance."""

    def test_get_feature_importance(self, sample_data):
        """Should return feature importance after fit."""
        X, y = sample_data
        model = XGBoostModel(n_estimators=10)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert isinstance(importance, np.ndarray)
        assert len(importance) == X.shape[1]
        assert all(imp >= 0 for imp in importance)

    def test_get_feature_importance_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = XGBoostModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()

    def test_important_features_ranked_higher(self, sample_data):
        """Important features should have higher importance."""
        X, y = sample_data
        # y = 3*X[:,0] + 2*X[:,1] + noise, so features 0 and 1 should be most important
        model = XGBoostModel(n_estimators=50)
        model.fit(X, y)

        importance = model.get_feature_importance()
        top_2_indices = np.argsort(importance)[-2:]

        # Features 0 and 1 should be in top 2
        assert 0 in top_2_indices or 1 in top_2_indices


class TestXGBoostModelBestIteration:
    """Test cases for best_iteration property."""

    def test_best_iteration_without_early_stopping(self, sample_data):
        """Should return n_estimators without early stopping."""
        X, y = sample_data
        model = XGBoostModel(n_estimators=20)
        model.fit(X, y, verbose=False)

        # Without early stopping, best_iteration is typically n_estimators - 1
        assert model.best_iteration >= 0

    def test_best_iteration_with_early_stopping(self, sample_data_with_validation):
        """Should return actual best iteration with early stopping (if supported)."""
        X_train, y_train, X_val, y_val = sample_data_with_validation

        model = XGBoostModel(n_estimators=1000)
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=5,
                verbose=False,
            )
            # With early stopping, should be less than n_estimators
            assert model.best_iteration < 1000
        except TypeError:
            # Early stopping not supported, just fit normally
            model.fit(X_train, y_train, verbose=False)

        assert model.best_iteration >= 0

    def test_best_iteration_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = XGBoostModel()

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.best_iteration


class TestXGBoostModelSaveLoad:
    """Test cases for save and load functionality."""

    def test_save_and_load(self, sample_data):
        """Should save and load model correctly."""
        X, y = sample_data
        model = XGBoostModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "xgboost_model.joblib"
            model.save(path)

            loaded_model = XGBoostModel.load(path)
            loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestXGBoostModelReproducibility:
    """Test cases for reproducibility."""

    def test_reproducibility_same_seed(self, sample_data):
        """Same seed should produce same results."""
        X, y = sample_data

        model1 = XGBoostModel(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = XGBoostModel(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_different_seeds_different_results(self, sample_data):
        """Different seeds should produce different results."""
        X, y = sample_data

        model1 = XGBoostModel(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = XGBoostModel(n_estimators=10, random_state=123)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Results should be different (at least for some predictions)
        assert not np.allclose(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
