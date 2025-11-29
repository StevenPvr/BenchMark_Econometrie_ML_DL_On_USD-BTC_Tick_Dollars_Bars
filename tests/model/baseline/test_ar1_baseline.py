import pytest
import numpy as np
import pandas as pd
from src.model.baseline.ar1_baseline import AR1Baseline

@pytest.fixture
def data():
    # Sequence: 0, 1, 0, 1, 0, 1...
    # P(1|0) = 1.0, P(0|1) = 1.0
    y = np.array([0, 1] * 10) # 20 elements. Ends with 1.
    X = np.zeros((20, 1))
    return X, y

def test_ar1_fit_predict(data):
    X, y = data
    model = AR1Baseline()

    model.fit(X, y)
    assert model.is_fitted

    trans_mat = model.get_transition_matrix()
    # classes: 0, 1
    # P(0|0) should be 0, P(1|0) should be 1
    # P(0|1) should be 1, P(1|1) should be 0
    # Note: last element is 1. Previous is 0.
    # y[t] vs y[t-1].
    # 0 -> 1 transitions. 1 -> 0 transitions.
    np.testing.assert_allclose(trans_mat, [[0, 1], [1, 0]])

    # Predict
    # Last value was 1. Next should be 0. Then 1. Then 0.
    X_pred = np.zeros((3, 1))
    preds = model.predict(X_pred)
    np.testing.assert_array_equal(preds, [0, 1, 0])

    # Proba
    # 1st step: P(.|1) = [1, 0] (since next is 0)
    # 2nd step: P(.|0) = [0, 1] (since prev prediction was 0)
    probs = model.predict_proba(X_pred)
    np.testing.assert_allclose(probs[0], [1, 0])
    np.testing.assert_allclose(probs[1], [0, 1])

def test_ar1_predict_with_actuals(data):
    X, y = data
    model = AR1Baseline()
    model.fit(X, y) # last is 1

    # y_test = [1, 1]
    # Prediction 0: from last train (1) -> predict 0
    # Prediction 1: from y_test[0] (1) -> predict 0
    y_test = np.array([1, 1])
    X_test = np.zeros((2, 1))

    preds = model.predict_with_actuals(X_test, y_test)
    np.testing.assert_array_equal(preds, [0, 0])

def test_ar1_unfitted(data):
    X, _ = data
    model = AR1Baseline()
    with pytest.raises(ValueError):
        model.predict(X)
