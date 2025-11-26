"""Tests for src/model/machine_learning/rf/random_forest_model.py module."""

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

from src.model.machine_learning.rf.random_forest_model import RandomForestModel


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
def sample_data_large():
    """Generate larger sample for OOB testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

    return X, y


class TestRandomForestModelInit:
    """Test cases for RandomForestModel initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = RandomForestModel()

        assert model.name == "RandomForest"
        assert model.n_estimators == 100
        assert model.max_depth is None
        assert model.min_samples_split == 2
        assert model.min_samples_leaf == 1
        assert model.max_features == "sqrt"
        assert model.bootstrap is True
        assert model.oob_score is False
        assert model.random_state == 42
        assert model.n_jobs == -1
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = RandomForestModel(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.5,
            bootstrap=False,
            oob_score=False,
            random_state=123,
        )

        assert model.n_estimators == 200
        assert model.max_depth == 10
        assert model.min_samples_split == 5
        assert model.min_samples_leaf == 2
        assert model.max_features == 0.5
        assert model.bootstrap is False
        assert model.random_state == 123


class TestRandomForestModelFit:
    """Test cases for RandomForestModel fit method."""

    def test_fit_basic(self, sample_data):
        """Should fit model successfully."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10)

        result = model.fit(X, y)

        assert model.is_fitted is True
        assert model.model is not None
        assert result is model

    def test_fit_with_dataframe(self, sample_data):
        """Should fit with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        model = RandomForestModel(n_estimators=10)
        model.fit(X_df, y_series)

        assert model.is_fitted is True

    def test_fit_with_max_depth(self, sample_data):
        """Should fit with limited max_depth."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.fit(X, y)

        assert model.is_fitted is True

    def test_fit_with_oob_score(self, sample_data_large):
        """Should compute OOB score when enabled."""
        X, y = sample_data_large
        model = RandomForestModel(n_estimators=50, oob_score=True)
        model.fit(X, y)

        assert model.is_fitted is True
        assert 0 <= model.oob_score_ <= 1


class TestRandomForestModelPredict:
    """Test cases for RandomForestModel predict method."""

    def test_predict_basic(self, sample_data):
        """Should predict after fitting."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_reasonable_values(self, sample_data):
        """Predictions should be reasonable."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=50)

        model.fit(X, y)
        predictions = model.predict(X)

        # Predictions should correlate with actual values
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.8

    def test_predict_before_fit_raises(self, sample_data):
        """Should raise error if predicting before fit."""
        X, y = sample_data
        model = RandomForestModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_data):
        """Should predict with pandas DataFrame."""
        X, y = sample_data
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)
        predictions = model.predict(X_df)

        assert len(predictions) == len(y)


class TestRandomForestModelFeatureImportance:
    """Test cases for feature importance."""

    def test_get_feature_importance(self, sample_data):
        """Should return feature importance after fit."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert isinstance(importance, np.ndarray)
        assert len(importance) == X.shape[1]
        assert all(imp >= 0 for imp in importance)
        assert abs(sum(importance) - 1.0) < 1e-6  # Should sum to 1

    def test_get_feature_importance_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = RandomForestModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()

    def test_important_features_ranked_higher(self, sample_data):
        """Important features should have higher importance."""
        X, y = sample_data
        # y = 3*X[:,0] + 2*X[:,1] + noise, so features 0 and 1 should be most important
        model = RandomForestModel(n_estimators=100)
        model.fit(X, y)

        importance = model.get_feature_importance()
        top_2_indices = np.argsort(importance)[-2:]

        # Features 0 and 1 should be in top 2
        assert 0 in top_2_indices or 1 in top_2_indices


class TestRandomForestModelOOBScore:
    """Test cases for OOB score."""

    def test_oob_score_enabled(self, sample_data_large):
        """Should compute OOB score when enabled."""
        X, y = sample_data_large
        model = RandomForestModel(n_estimators=50, oob_score=True)
        model.fit(X, y)

        oob = model.oob_score_

        assert 0 <= oob <= 1

    def test_oob_score_disabled_raises(self, sample_data):
        """Should raise error when OOB not enabled."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, oob_score=False)
        model.fit(X, y)

        with pytest.raises(ValueError, match="oob_score"):
            _ = model.oob_score_

    def test_oob_score_before_fit_raises(self):
        """Should raise error if called before fit."""
        model = RandomForestModel(oob_score=True)

        with pytest.raises(ValueError, match="not fitted"):
            _ = model.oob_score_


class TestRandomForestModelSaveLoad:
    """Test cases for save and load functionality."""

    def test_save_and_load(self, sample_data):
        """Should save and load model correctly."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rf_model.joblib"
            model.save(path)

            loaded_model = RandomForestModel.load(path)
            loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


class TestRandomForestModelReproducibility:
    """Test cases for reproducibility."""

    def test_reproducibility_same_seed(self, sample_data):
        """Same seed should produce same results."""
        X, y = sample_data

        model1 = RandomForestModel(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = RandomForestModel(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_different_seeds_different_results(self, sample_data):
        """Different seeds should produce different results."""
        X, y = sample_data

        model1 = RandomForestModel(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = RandomForestModel(n_estimators=10, random_state=123)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Results should be different (at least for some predictions)
        assert not np.allclose(pred1, pred2)


class TestRandomForestModelMaxFeatures:
    """Test cases for max_features parameter."""

    def test_max_features_sqrt(self, sample_data):
        """Should work with sqrt max_features."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, max_features="sqrt")
        model.fit(X, y)

        assert model.is_fitted is True

    def test_max_features_log2(self, sample_data):
        """Should work with log2 max_features."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, max_features="log2")
        model.fit(X, y)

        assert model.is_fitted is True

    def test_max_features_fraction(self, sample_data):
        """Should work with fractional max_features."""
        X, y = sample_data
        model = RandomForestModel(n_estimators=10, max_features=0.5)
        model.fit(X, y)

        assert model.is_fitted is True


class TestRandomForestModelComplexity:
    """Test cases for model complexity."""

    def test_deeper_trees_better_fit(self, sample_data):
        """Deeper trees should fit training data better."""
        X, y = sample_data

        model_shallow = RandomForestModel(n_estimators=20, max_depth=2)
        model_shallow.fit(X, y)
        pred_shallow = model_shallow.predict(X)

        model_deep = RandomForestModel(n_estimators=20, max_depth=None)
        model_deep.fit(X, y)
        pred_deep = model_deep.predict(X)

        mse_shallow = np.mean((y - pred_shallow) ** 2)
        mse_deep = np.mean((y - pred_deep) ** 2)

        assert mse_deep <= mse_shallow

    def test_more_trees_more_stable(self, sample_data):
        """More trees should give more stable predictions."""
        X, y = sample_data

        # Run multiple times and check variance
        predictions_few = []
        predictions_many = []

        for seed in range(5):
            model_few = RandomForestModel(n_estimators=5, random_state=seed)
            model_few.fit(X, y)
            predictions_few.append(model_few.predict(X))

            model_many = RandomForestModel(n_estimators=50, random_state=seed)
            model_many.fit(X, y)
            predictions_many.append(model_many.predict(X))

        var_few = np.var(predictions_few, axis=0).mean()
        var_many = np.var(predictions_many, axis=0).mean()

        assert var_many <= var_few


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
