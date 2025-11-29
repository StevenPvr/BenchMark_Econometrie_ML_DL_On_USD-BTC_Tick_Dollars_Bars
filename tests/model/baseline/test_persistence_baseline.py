import pytest
import numpy as np
import pandas as pd
from src.model.baseline.persistence_baseline import PersistenceBaseline

@pytest.fixture
def data():
    X = np.zeros((5, 1))
    y = np.array([0, 1, 0, 1, 0])
    return X, y

def test_persistence_fit_predict(data):
    X, y = data
    model = PersistenceBaseline()

    model.fit(X, y)
    assert model.is_fitted
    assert model.get_last_value() == 0

    # Predict should return last value (0) for all samples
    preds = model.predict(X)
    assert np.all(preds == 0)

    # Predict proba should be 1.0 for class 0
    probs = model.predict_proba(X)
    # Classes are 0, 1. Index 0 is class 0.
    assert np.all(probs[:, 0] == 1.0)
    assert np.all(probs[:, 1] == 0.0)

def test_persistence_predict_with_actuals(data):
    X, y = data
    model = PersistenceBaseline()
    model.fit(X, y) # last value is 0

    # New data for testing
    # y_actual = [1, 1, 0]
    # Prediction:
    # 0: last train value = 0
    # 1: y_actual[0] = 1
    # 2: y_actual[1] = 1
    y_test = np.array([1, 1, 0])
    X_test = np.zeros((3, 1))

    preds = model.predict_with_actuals(X_test, y_test)
    np.testing.assert_array_equal(preds, [0, 1, 1])

def test_persistence_unfitted(data):
    X, _ = data
    model = PersistenceBaseline()
    with pytest.raises(ValueError):
        model.predict(X)
