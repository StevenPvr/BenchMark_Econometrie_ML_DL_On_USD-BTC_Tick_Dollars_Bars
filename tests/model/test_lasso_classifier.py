import pytest
import numpy as np
import pandas as pd
from src.model.lasso_classifier import LassoClassifierModel

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

def test_lasso_init():
    model = LassoClassifierModel(C=0.5)
    assert model.name == "LassoClassifier"
    assert model.C == 0.5
    assert not model.is_fitted

def test_lasso_fit_predict(data):
    X, y = data
    model = LassoClassifierModel(C=1.0)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_lasso_selected_features(data):
    X, y = data
    # Increase C to keep more features, or decrease to select fewer
    model = LassoClassifierModel(C=10.0)
    model.fit(X, y)

    selected = model.get_selected_features()
    assert isinstance(selected, list)
    assert len(selected) <= X.shape[1]

    # Test with names
    names = ["f1", "f2", "f3"]
    selected_names = model.get_selected_features(feature_names=names)
    assert set(selected_names).issubset(set(names))

def test_lasso_coef(data):
    X, y = data
    model = LassoClassifierModel()
    model.fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None

def test_lasso_unfitted_properties():
    model = LassoClassifierModel()
    with pytest.raises(ValueError):
        _ = model.coef_
    with pytest.raises(ValueError):
        _ = model.intercept_
