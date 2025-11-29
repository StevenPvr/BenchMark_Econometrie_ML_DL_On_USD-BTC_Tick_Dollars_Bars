import pytest
import numpy as np
import pandas as pd
from src.model.random_forest_model import RandomForestModel

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

def test_rf_init():
    model = RandomForestModel(n_estimators=50)
    assert model.name == "RandomForest"
    assert model.n_estimators == 50
    assert not model.is_fitted

def test_rf_fit_predict(data):
    X, y = data
    model = RandomForestModel(n_estimators=10, random_state=42)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_rf_feature_importance(data):
    X, y = data
    model = RandomForestModel(n_estimators=10, random_state=42)

    with pytest.raises(ValueError, match="Model not fitted"):
        model.get_feature_importance()

    model.fit(X, y)
    imp = model.get_feature_importance()
    assert len(imp) == X.shape[1]

def test_rf_oob_score(data):
    X, y = data
    # Need enough samples for OOB
    X_large = np.tile(X, (5, 1))
    y_large = np.tile(y, 5)
    model = RandomForestModel(n_estimators=20, oob_score=True, random_state=42)
    model.fit(X_large, y_large)
    assert isinstance(model.oob_score_, float)

def test_rf_oob_score_not_enabled(data):
    X, y = data
    model = RandomForestModel(n_estimators=10, oob_score=False, random_state=42)
    model.fit(X, y)
    with pytest.raises(ValueError, match="oob_score was not enabled"):
        _ = model.oob_score_
