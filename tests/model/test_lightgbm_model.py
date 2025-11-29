import pytest
import numpy as np
import pandas as pd
from src.model.lightgbm_model import LightGBMModel

@pytest.fixture
def data():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    y = np.array([0, 1, 0, 1])
    return X, y

def test_lightgbm_init():
    model = LightGBMModel(n_estimators=50, learning_rate=0.05)
    assert model.name == "LightGBM"
    assert model.n_estimators == 50
    assert model.learning_rate == 0.05
    assert not model.is_fitted

def test_lightgbm_fit_predict(data):
    X, y = data
    model = LightGBMModel(n_estimators=10, random_state=42)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None
    assert model.feature_names == ["feature_0", "feature_1", "feature_2"]

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_lightgbm_fit_predict_dataframe(data):
    X, y = data
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    y_series = pd.Series(y)

    model = LightGBMModel(n_estimators=10, random_state=42)
    model.fit(X_df, y_series)

    assert model.feature_names == ["a", "b", "c"]

    preds = model.predict(X_df)
    assert len(preds) == len(X)

    # Predict with missing columns should fail
    with pytest.raises(ValueError, match="Missing features"):
        model.predict(X_df[["a", "b"]])

def test_lightgbm_feature_importance(data):
    X, y = data
    model = LightGBMModel(n_estimators=10, random_state=42)

    with pytest.raises(ValueError, match="Model not fitted"):
        model.get_feature_importance()

    model.fit(X, y)
    imp = model.get_feature_importance()
    assert len(imp) == X.shape[1]

def test_lightgbm_best_iteration(data):
    X, y = data
    model = LightGBMModel(n_estimators=10, random_state=42)
    model.fit(X, y)
    # Without early stopping, best_iteration should be n_estimators or similar,
    # but specific to implementation detail.
    # Just check it returns an int.
    assert isinstance(model.best_iteration, int)

def test_lightgbm_predict_unfitted(data):
    X, _ = data
    model = LightGBMModel()
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(X)
