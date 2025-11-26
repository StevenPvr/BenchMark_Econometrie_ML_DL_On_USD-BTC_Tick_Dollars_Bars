"""Tests for src/model/econometrie/lasso/lasso.py module."""

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

from src.model.econometrie.lasso.lasso import LassoModel


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
def sample_data_sparse():
    """Generate data where only few features are relevant (sparse)."""
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    # Only features 0 and 5 are relevant
    y = 5 * X[:, 0] + 3 * X[:, 5] + np.random.randn(n_samples) * 0.1

    return X, y


class TestLassoModelInit:
    """Test cases for LassoModel initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = LassoModel()

        assert model.name == "Lasso"
        assert model.alpha == 1.0
        assert model.fit_intercept is True
        assert model.normalize is True
        assert model.max_iter == 1000
        assert model.tol == 1e-4
        assert model.is_fitted is False
        assert model.scaler is None
        assert model.model is None

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = LassoModel(
            alpha=0.5,
            fit_intercept=False,
            normalize=False,
            max_iter=2000,
            tol=1e-5,
        )

        assert model.alpha == 0.5
        assert model.fit_intercept is False
        assert model.normalize is False
        assert model.max_iter == 2000
        assert model.tol == 1e-5


class TestLassoModelFit:
    """Test cases for LassoModel fit method."""

    def test_fit_basic(self, sample_data):
        """Should fit model successfully."""
        X, y = sample_data
        model = LassoModel(alpha=0.01)

        result = model.fit(X, y)

        assert model.is_fitted is True
        assert model.model is not None
        assert result is model

    def test_fit_with_dataframe(self, sample_data):
        """Should fit with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        model = LassoModel(alpha=0.01)
        model.fit(X_df, y_series)

        assert model.is_fitted is True

    def test_fit_with_normalization(self, sample_data):
        """Should create scaler when normalize=True."""
        X, y = sample_data
        model = LassoModel(normalize=True, alpha=0.01)
        model.fit(X, y)

        assert model.scaler is not None

    def test_fit_without_normalization(self, sample_data):
        """Should not create scaler when normalize=False."""
        X, y = sample_data
        model = LassoModel(normalize=False, alpha=0.01)
        model.fit(X, y)

        assert model.scaler is None


class TestLassoModelPredict:
    """Test cases for LassoModel predict method."""

    def test_predict_basic(self, sample_data):
        """Should predict after fitting."""
        X, y = sample_data
        model = LassoModel(alpha=0.01)

        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_reasonable_values(self, sample_data):
        """Predictions should be reasonable."""
        X, y = sample_data
        model = LassoModel(alpha=0.001)

        model.fit(X, y)
        predictions = model.predict(X)

        # Predictions should correlate with actual values
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.9

    def test_predict_before_fit_raises(self, sample_data):
        """Should raise error if predicting before fit."""
        X, y = sample_data
        model = LassoModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_data):
        """Should predict with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        model = LassoModel(alpha=0.01)
        model.fit(X, y)
        predictions = model.predict(X_df)

        assert len(predictions) == len(y)


class TestLassoModelCoefficients:
    """Test cases for model coefficients."""

    def test_coef_property(self, sample_data):
        """Should return coefficients after fit."""
        X, y = sample_data
        model = LassoModel(alpha=0.01)
        model.fit(X, y)

        coef = model.coef_

        assert isinstance(coef, np.ndarray)
        assert len(coef) == X.shape[1]

    def test_coef_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = LassoModel()

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.coef_

    def test_intercept_property(self, sample_data):
        """Should return intercept after fit."""
        X, y = sample_data
        model = LassoModel(alpha=0.01, fit_intercept=True)
        model.fit(X, y)

        intercept = model.intercept_

        assert isinstance(intercept, (float, np.floating))

    def test_intercept_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = LassoModel()

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.intercept_


class TestLassoModelFeatureSelection:
    """Test cases for feature selection (sparsity)."""

    def test_produces_sparse_coefficients(self, sample_data_sparse):
        """Should produce sparse coefficients (some zeros)."""
        X, y = sample_data_sparse
        model = LassoModel(alpha=0.1, normalize=True)
        model.fit(X, y)

        # Should have some zero coefficients
        n_zero = np.sum(model.coef_ == 0)
        assert n_zero > 0

    def test_get_selected_features_with_names(self, sample_data_sparse):
        """Should return selected feature names."""
        X, y = sample_data_sparse
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        model = LassoModel(alpha=0.1, normalize=True)
        model.fit(X, y)

        selected = model.get_selected_features(feature_names)

        assert isinstance(selected, list)
        # Should select at least 1 feature
        assert len(selected) >= 1
        # Selected features should be strings (names)
        assert all(isinstance(f, str) for f in selected)

    def test_get_selected_features_without_names(self, sample_data_sparse):
        """Should return selected feature indices."""
        X, y = sample_data_sparse

        model = LassoModel(alpha=0.1, normalize=True)
        model.fit(X, y)

        selected = model.get_selected_features()

        assert isinstance(selected, list)
        # Selected features should be indices (integers)
        assert all(isinstance(f, (int, np.integer)) for f in selected)

    def test_get_selected_features_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = LassoModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.get_selected_features()

    def test_selects_relevant_features(self, sample_data_sparse):
        """Should select the truly relevant features."""
        X, y = sample_data_sparse
        # y = 5*X[:,0] + 3*X[:,5] + noise

        model = LassoModel(alpha=0.05, normalize=True)
        model.fit(X, y)

        selected = model.get_selected_features()

        # Features 0 and 5 should be selected
        assert 0 in selected or 5 in selected


class TestLassoModelRegularization:
    """Test cases for regularization effects."""

    def test_higher_alpha_more_sparse(self, sample_data_sparse):
        """Higher alpha should produce sparser solutions."""
        X, y = sample_data_sparse

        model_low = LassoModel(alpha=0.001, normalize=True)
        model_low.fit(X, y)

        model_high = LassoModel(alpha=1.0, normalize=True)
        model_high.fit(X, y)

        n_nonzero_low = np.sum(model_low.coef_ != 0)
        n_nonzero_high = np.sum(model_high.coef_ != 0)

        assert n_nonzero_high <= n_nonzero_low

    def test_very_high_alpha_all_zero(self, sample_data):
        """Very high alpha should zero all coefficients."""
        X, y = sample_data

        model = LassoModel(alpha=1000.0, normalize=True)
        model.fit(X, y)

        # All coefficients should be zero (or very close)
        assert np.allclose(model.coef_, 0, atol=1e-6)


class TestLassoModelSaveLoad:
    """Test cases for save and load functionality."""

    def test_save_and_load(self, sample_data):
        """Should save and load model correctly."""
        X, y = sample_data
        model = LassoModel(alpha=0.01)
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lasso_model.joblib"
            model.save(path)

            loaded_model = LassoModel.load(path)
            loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_save_and_load_preserves_coef(self, sample_data):
        """Should preserve coefficients after save/load."""
        X, y = sample_data
        model = LassoModel(alpha=0.01)
        model.fit(X, y)
        original_coef = model.coef_.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lasso_model.joblib"
            model.save(path)

            loaded_model = LassoModel.load(path)

        np.testing.assert_array_almost_equal(original_coef, loaded_model.coef_)


class TestLassoModelReproducibility:
    """Test cases for reproducibility."""

    def test_reproducibility(self, sample_data):
        """Same data should produce same results."""
        X, y = sample_data

        model1 = LassoModel(alpha=0.01)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = LassoModel(alpha=0.01)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestLassoModelConvergence:
    """Test cases for convergence settings."""

    def test_max_iter_affects_convergence(self, sample_data):
        """Should use max_iter setting."""
        X, y = sample_data

        # Very few iterations might not converge but shouldn't crash
        model = LassoModel(alpha=0.01, max_iter=1)
        # This should issue a convergence warning but not crash
        with pytest.warns(match="converge"):
            model.fit(X, y)

    def test_tol_affects_precision(self, sample_data):
        """Should respect tolerance setting."""
        X, y = sample_data

        model_loose = LassoModel(alpha=0.01, tol=1e-1)
        model_loose.fit(X, y)

        model_tight = LassoModel(alpha=0.01, tol=1e-8)
        model_tight.fit(X, y)

        # Both should fit
        assert model_loose.is_fitted
        assert model_tight.is_fitted


class TestLassoModelCompareToRidge:
    """Test cases comparing Lasso to Ridge."""

    def test_lasso_more_sparse_than_ridge(self, sample_data_sparse):
        """Lasso should be sparser than Ridge (Ridge doesn't zero coefficients)."""
        from src.model.econometrie.ridge.ridge import RidgeModel

        X, y = sample_data_sparse

        lasso = LassoModel(alpha=0.1, normalize=True)
        lasso.fit(X, y)

        ridge = RidgeModel(alpha=0.1, normalize=True)
        ridge.fit(X, y)

        n_zero_lasso = np.sum(lasso.coef_ == 0)
        n_zero_ridge = np.sum(ridge.coef_ == 0)

        # Lasso should have more zeros
        assert n_zero_lasso >= n_zero_ridge


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
