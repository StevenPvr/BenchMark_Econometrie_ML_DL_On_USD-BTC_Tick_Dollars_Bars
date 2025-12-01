import pytest
import numpy as np
import pandas as pd
from src.model.logistic_classifier import LogisticClassifierModel

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

def test_logistic_init():
    model = LogisticClassifierModel(C=0.5)
    assert model.name == "LogisticClassifier"
    assert model.C == 0.5
    assert model.max_iter == 300  # Updated default
    assert not model.is_fitted

def test_logistic_fit_predict(data):
    X, y = data
    model = LogisticClassifierModel(C=1.0)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_logistic_coef(data):
    X, y = data
    model = LogisticClassifierModel()
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_logistic_unfitted_properties():
    model = LogisticClassifierModel()
    with pytest.raises(ValueError):
        _ = model.coef_
    with pytest.raises(ValueError):
        _ = model.intercept_

def test_logistic_multiclass():
    # Test with 3 classes (triple-barrier)
    X = np.random.rand(30, 5)
    y = np.array([-1, 0, 1] * 10)

    model = LogisticClassifierModel(C=1.0)
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds).issubset({-1, 0, 1})

    probs = model.predict_proba(X)
    assert probs.shape == (30, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
