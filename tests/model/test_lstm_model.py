import pytest
import numpy as np
import pandas as pd
from src.model.lstm_model import LSTMModel

@pytest.fixture
def data():
    # Need enough data for sequence_length
    # 20 samples, 3 features
    X = np.random.rand(20, 3)
    # labels: -1, 0, 1
    y = np.random.choice([-1, 0, 1], size=20)
    return X, y

def test_lstm_init():
    model = LSTMModel(input_size=3, hidden_size=10, sequence_length=5)
    assert model.name == "LSTM"
    assert model.hidden_size == 10
    assert model.sequence_length == 5
    assert not model.is_fitted

def test_lstm_fit_predict(data):
    X, y = data
    model = LSTMModel(
        input_size=3,
        hidden_size=5,
        sequence_length=3,
        epochs=1,
        batch_size=4,
    )

    model.fit(X, y)
    assert model.is_fitted
    assert model.model is not None
    assert model.scaler is not None

    # Predict
    # Needs at least sequence_length rows
    X_pred = X[:5]
    preds = model.predict(X_pred)
    assert len(preds) == 5 - 3 # = 2
    assert set(preds).issubset({-1, 0, 1})

    probs = model.predict_proba(X_pred)
    assert probs.shape == (2, 3) # 3 classes
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

def test_lstm_fit_with_validation(data):
    X, y = data
    model = LSTMModel(
        input_size=3,
        hidden_size=5,
        sequence_length=3,
        epochs=1,
    )
    # Split data
    X_train, y_train = X[:15], y[:15]
    X_val, y_val = X[15:], y[15:]

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    history = model.get_history()
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["val_loss"]) > 0

def test_lstm_insufficient_data(data):
    X, y = data
    model = LSTMModel(sequence_length=50) # more than data
    with pytest.raises(ValueError, match="Not enough data"):
        model.fit(X, y)

def test_lstm_predict_insufficient_data(data):
    X, y = data
    model = LSTMModel(sequence_length=3, epochs=1)
    model.fit(X, y)

    X_short = X[:2] # less than sequence_length
    with pytest.raises(ValueError, match="Not enough data"):
        model.predict(X_short)

def test_lstm_multiclass_structure(data):
    # Ensure it handles different number of classes
    X, _ = data
    # Only 2 classes
    y_binary = np.random.choice([0, 1], size=20)

    model = LSTMModel(sequence_length=3, epochs=1)
    model.fit(X, y_binary)

    probs = model.predict_proba(X[:5])
    assert probs.shape[1] == 2
