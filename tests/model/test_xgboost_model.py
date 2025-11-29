import pytest
import numpy as np
import pandas as pd
from src.model.xgboost_model import XGBoostModel

@pytest.fixture
def data():
    # Use -1, 0, 1 labels to test mapping
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ])
    y = np.array([-1, 0, 1, 0, -1])
    return X, y

def test_xgboost_init():
    model = XGBoostModel(n_estimators=50, learning_rate=0.05)
    assert model.name == "XGBoost"
    assert model.n_estimators == 50
    assert model.learning_rate == 0.05
    assert not model.is_fitted

def test_xgboost_fit_predict(data):
    X, y = data
    model = XGBoostModel(n_estimators=10, random_state=42)

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None
    # Check if mapping is correct
    assert -1 in model._label_to_xgb
    assert 0 in model._label_to_xgb
    assert 1 in model._label_to_xgb

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0, 1})

    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

def test_xgboost_fit_predict_dataframe(data):
    X, y = data
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    y_series = pd.Series(y)

    model = XGBoostModel(n_estimators=10, random_state=42)
    model.fit(X_df, y_series)

    preds = model.predict(X_df)
    assert len(preds) == len(X)
    assert set(preds).issubset({-1, 0, 1})

def test_xgboost_feature_importance(data):
    X, y = data
    model = XGBoostModel(n_estimators=10, random_state=42)

    with pytest.raises(ValueError, match="Model not fitted"):
        model.get_feature_importance()

    model.fit(X, y)
    imp = model.get_feature_importance()
    assert len(imp) == X.shape[1]

def test_xgboost_predict_unfitted(data):
    X, _ = data
    model = XGBoostModel()
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.predict(X)
