import pytest
import numpy as np
import pandas as pd
from src.model.baseline.random_baseline import RandomBaseline

@pytest.fixture
def data():
    X = np.zeros((20, 2))
    # 60% class 1, 40% class 0
    y = np.array([1]*12 + [0]*8)
    return X, y

def test_random_init():
    model = RandomBaseline(random_state=123)
    assert model.name == "RandomBaseline"
    assert model.random_state == 123
    assert not model.is_fitted

def test_random_fit_predict(data):
    X, y = data
    model = RandomBaseline(random_state=42)

    model.fit(X, y)
    assert model.is_fitted
    assert model.classes_ is not None
    assert model.class_probs_ is not None

    # Check probabilities roughly match
    # class 0: 8/20 = 0.4
    # class 1: 12/20 = 0.6
    # Note: unique returns sorted classes, so 0 then 1
    probs = model.get_distribution_params()["probabilities"]
    np.testing.assert_allclose(probs, [0.4, 0.6])

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probs_pred = model.predict_proba(X)
    assert probs_pred.shape == (len(X), 2)
    np.testing.assert_allclose(probs_pred[0], [0.4, 0.6])

def test_random_predict_unfitted(data):
    X, _ = data
    model = RandomBaseline()
    with pytest.raises(ValueError):
        model.predict(X)
