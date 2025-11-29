import pytest
import numpy as np
import pandas as pd
from src.model.catboost_model import CatBoostModel

@pytest.fixture
def data():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    y = np.array([-1, 0, -1, 0])
    return X, y

def test_catboost_init():
    model = CatBoostModel(iterations=50, learning_rate=0.05)
    assert model.name == "CatBoost"
    assert model.iterations == 50
    assert model.learning_rate == 0.05
    assert not model.is_fitted

def test_catboost_fit_predict(data):
    X, y = data
    model = CatBoostModel(iterations=10, random_seed=42)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_catboost_fit_predict_dataframe(data):
    X, y = data
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    y_series = pd.Series(y)

    model = CatBoostModel(iterations=10, random_seed=42)
    model.fit(X_df, y_series)

    preds = model.predict(X_df)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0})

def test_catboost_feature_importance(data):
    X, y = data
    model = CatBoostModel(iterations=10, random_seed=42)

    with pytest.raises(ValueError, match="Model not fitted"):
        model.get_feature_importance()

    model.fit(X, y)
    imp = model.get_feature_importance()
    assert len(imp) == X.shape[1]

def test_catboost_predict_unfitted(data):
    X, _ = data
    model = CatBoostModel()
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(X)
