import pytest
import numpy as np
import pandas as pd
from src.model.ridge_classifier import RidgeClassifierModel

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

def test_ridge_init():
    model = RidgeClassifierModel(alpha=0.5)
    assert model.name == "RidgeClassifier"
    assert model.alpha == 0.5
    assert not model.is_fitted

def test_ridge_fit_predict(data):
    X, y = data
    model = RidgeClassifierModel(alpha=1.0)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_ridge_coef(data):
    X, y = data
    model = RidgeClassifierModel()
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_ridge_unfitted_properties():
    model = RidgeClassifierModel()
    with pytest.raises(ValueError):
        _ = model.coef_
    with pytest.raises(ValueError):
        _ = model.intercept_
