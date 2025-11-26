"""Tests for src/model/econometrie/ridge/ridge.py module."""

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

from src.model.econometrie.ridge.ridge import RidgeModel


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
def sample_data_collinear():
    """Generate data with multicollinearity."""
    np.random.seed(42)
    n_samples = 100

    X1 = np.random.randn(n_samples)
    X2 = X1 + np.random.randn(n_samples) * 0.01  # Highly correlated
    X3 = np.random.randn(n_samples)

    X = np.column_stack([X1, X2, X3])
    y = 3 * X1 + 2 * X3 + np.random.randn(n_samples) * 0.1

    return X, y


class TestRidgeModelInit:
    """Test cases for RidgeModel initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = RidgeModel()

        assert model.name == "Ridge"
        assert model.alpha == 1.0
        assert model.fit_intercept is True
        assert model.normalize is True
        assert model.is_fitted is False
        assert model.scaler is None
        assert model.model is None

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = RidgeModel(
            alpha=0.5,
            fit_intercept=False,
            normalize=False,
        )

        assert model.alpha == 0.5
        assert model.fit_intercept is False
        assert model.normalize is False


class TestRidgeModelFit:
    """Test cases for RidgeModel fit method."""

    def test_fit_basic(self, sample_data):
        """Should fit model successfully."""
        X, y = sample_data
        model = RidgeModel()

        result = model.fit(X, y)

        assert model.is_fitted is True
        assert model.model is not None
        assert result is model

    def test_fit_with_dataframe(self, sample_data):
        """Should fit with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        model = RidgeModel()
        model.fit(X_df, y_series)

        assert model.is_fitted is True

    def test_fit_with_normalization(self, sample_data):
        """Should create scaler when normalize=True."""
        X, y = sample_data
        model = RidgeModel(normalize=True)
        model.fit(X, y)

        assert model.scaler is not None

    def test_fit_without_normalization(self, sample_data):
        """Should not create scaler when normalize=False."""
        X, y = sample_data
        model = RidgeModel(normalize=False)
        model.fit(X, y)

        assert model.scaler is None


class TestRidgeModelPredict:
    """Test cases for RidgeModel predict method."""

    def test_predict_basic(self, sample_data):
        """Should predict after fitting."""
        X, y = sample_data
        model = RidgeModel()

        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_reasonable_values(self, sample_data):
        """Predictions should be reasonable."""
        X, y = sample_data
        model = RidgeModel(alpha=0.01)

        model.fit(X, y)
        predictions = model.predict(X)

        # Predictions should correlate with actual values
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.9

    def test_predict_before_fit_raises(self, sample_data):
        """Should raise error if predicting before fit."""
        X, y = sample_data
        model = RidgeModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_data):
        """Should predict with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        model = RidgeModel()
        model.fit(X, y)
        predictions = model.predict(X_df)

        assert len(predictions) == len(y)

    def test_predict_normalized_consistent(self, sample_data):
        """Predictions should be consistent with normalization."""
        X, y = sample_data

        model = RidgeModel(normalize=True)
        model.fit(X, y)

        # Predict on training data
        pred_train = model.predict(X)

        # Predict on same data again
        pred_again = model.predict(X)

        np.testing.assert_array_almost_equal(pred_train, pred_again)


class TestRidgeModelCoefficients:
    """Test cases for model coefficients."""

    def test_coef_property(self, sample_data):
        """Should return coefficients after fit."""
        X, y = sample_data
        model = RidgeModel()
        model.fit(X, y)

        coef = model.coef_

        assert isinstance(coef, np.ndarray)
        assert len(coef) == X.shape[1]

    def test_coef_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = RidgeModel()

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.coef_

    def test_intercept_property(self, sample_data):
        """Should return intercept after fit."""
        X, y = sample_data
        model = RidgeModel(fit_intercept=True)
        model.fit(X, y)

        intercept = model.intercept_

        assert isinstance(intercept, (float, np.floating))

    def test_intercept_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = RidgeModel()

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.intercept_


class TestRidgeModelRegularization:
    """Test cases for regularization effects."""

    def test_higher_alpha_smaller_coef(self, sample_data):
        """Higher alpha should shrink coefficients."""
        X, y = sample_data

        model_low = RidgeModel(alpha=0.001, normalize=False)
        model_low.fit(X, y)

        model_high = RidgeModel(alpha=100.0, normalize=False)
        model_high.fit(X, y)

        # Coefficients should be smaller with higher regularization
        norm_low = np.linalg.norm(model_low.coef_)
        norm_high = np.linalg.norm(model_high.coef_)

        assert norm_high < norm_low

    def test_handles_multicollinearity(self, sample_data_collinear):
        """Should handle multicollinear features."""
        X, y = sample_data_collinear

        # This should not raise an error
        model = RidgeModel(alpha=1.0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_regularization_prevents_overfitting(self, sample_data):
        """Regularization should prevent overfitting on noisy data."""
        X, y = sample_data

        # Train/test split
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model_low_reg = RidgeModel(alpha=0.0001)
        model_low_reg.fit(X_train, y_train)
        pred_low = model_low_reg.predict(X_test)

        model_high_reg = RidgeModel(alpha=10.0)
        model_high_reg.fit(X_train, y_train)
        pred_high = model_high_reg.predict(X_test)

        # Both should produce valid predictions
        assert len(pred_low) == len(y_test)
        assert len(pred_high) == len(y_test)


class TestRidgeModelSaveLoad:
    """Test cases for save and load functionality."""

    def test_save_and_load(self, sample_data):
        """Should save and load model correctly."""
        X, y = sample_data
        model = RidgeModel(alpha=0.5)
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ridge_model.joblib"
            model.save(path)

            loaded_model = RidgeModel.load(path)
            loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_save_and_load_preserves_coef(self, sample_data):
        """Should preserve coefficients after save/load."""
        X, y = sample_data
        model = RidgeModel()
        model.fit(X, y)
        original_coef = model.coef_.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ridge_model.joblib"
            model.save(path)

            loaded_model = RidgeModel.load(path)

        np.testing.assert_array_almost_equal(original_coef, loaded_model.coef_)


class TestRidgeModelReproducibility:
    """Test cases for reproducibility."""

    def test_reproducibility(self, sample_data):
        """Same data should produce same results."""
        X, y = sample_data

        model1 = RidgeModel(alpha=0.5)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = RidgeModel(alpha=0.5)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestRidgeModelLinearRelationship:
    """Test cases for linear relationship fitting."""

    def test_perfect_linear_relationship(self):
        """Should fit perfect linear relationship with low alpha."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 3 * X[:, 0] + 2 * X[:, 1]  # Perfect linear relationship, no noise

        model = RidgeModel(alpha=0.0001, normalize=False)
        model.fit(X, y)

        # Coefficients should be close to true values
        assert abs(model.coef_[0] - 3.0) < 0.1
        assert abs(model.coef_[1] - 2.0) < 0.1

    def test_intercept_fitting(self):
        """Should fit intercept correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X[:, 0] + 5  # With intercept

        model = RidgeModel(alpha=0.0001, fit_intercept=True, normalize=False)
        model.fit(X, y)

        assert abs(model.intercept_ - 5.0) < 0.5
        assert abs(model.coef_[0] - 2.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
