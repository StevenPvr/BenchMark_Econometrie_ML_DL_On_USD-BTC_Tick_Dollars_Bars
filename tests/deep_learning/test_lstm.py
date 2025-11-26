"""Tests for src/model/deep_learning/lstm_model.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.deep_learning.lstm_model import LSTMModel


@pytest.fixture
def sample_time_series():
    """Generate sample time series data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    # Simple pattern: y depends on X with some lag
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.1

    return X, y


@pytest.fixture
def sample_time_series_with_validation(sample_time_series):
    """Split time series into train and validation sets."""
    X, y = sample_time_series
    split_idx = 150

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


class TestLSTMModelInit:
    """Test cases for LSTMModel initialization."""

    def test_init_default_params(self):
        """Should initialize with default parameters."""
        model = LSTMModel()

        assert model.name == "LSTM"
        assert model.input_size == 1
        assert model.hidden_size == 32
        assert model.num_layers == 1
        assert model.dropout == 0.0
        assert model.sequence_length == 10
        assert model.learning_rate == 0.001
        assert model.epochs == 100
        assert model.batch_size == 32
        assert model.patience == 10
        assert model.device_str == "auto"
        assert model.is_fitted is False

    def test_init_custom_params(self):
        """Should initialize with custom parameters."""
        model = LSTMModel(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            sequence_length=20,
            learning_rate=0.01,
            epochs=50,
            batch_size=64,
            patience=5,
            device="cpu",
        )

        assert model.input_size == 5
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout == 0.2
        assert model.sequence_length == 20
        assert model.learning_rate == 0.01
        assert model.epochs == 50
        assert model.batch_size == 64
        assert model.patience == 5
        assert model.device_str == "cpu"


class TestLSTMModelGetDevice:
    """Test cases for device selection."""

    def test_get_device_cpu(self):
        """Should select CPU device."""
        model = LSTMModel(device="cpu")
        device = model._get_device()

        assert str(device) == "cpu"

    def test_get_device_auto(self):
        """Should auto-select available device."""
        model = LSTMModel(device="auto")
        device = model._get_device()

        # Should be one of the valid devices
        assert str(device) in ["cpu", "cuda", "mps"]


class TestLSTMModelCreateSequences:
    """Test cases for sequence creation."""

    def test_create_sequences_basic(self, sample_time_series):
        """Should create sequences correctly."""
        X, y = sample_time_series
        model = LSTMModel(sequence_length=10)

        X_seq, y_seq = model._create_sequences(X, y)

        # Should have n_samples - sequence_length sequences
        expected_len = len(X) - model.sequence_length
        assert X_seq.shape[0] == expected_len
        assert y_seq is not None
        assert y_seq.shape[0] == expected_len

        # Each sequence should have shape (sequence_length, n_features)
        assert X_seq.shape[1] == model.sequence_length
        assert X_seq.shape[2] == X.shape[1]

    def test_create_sequences_without_y(self, sample_time_series):
        """Should create sequences without y."""
        X, _ = sample_time_series
        model = LSTMModel(sequence_length=10)

        X_seq, y_seq = model._create_sequences(X, None)

        assert X_seq.shape[0] == len(X) - model.sequence_length
        assert y_seq is None


class TestLSTMModelFit:
    """Test cases for LSTMModel fit method."""

    def test_fit_basic(self, sample_time_series):
        """Should fit model successfully."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )

        result = model.fit(X, y, verbose=False)

        assert model.is_fitted is True
        assert model.model is not None
        assert result is model

    def test_fit_with_dataframe(self, sample_time_series):
        """Should fit with pandas DataFrame."""
        X, y = sample_time_series
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])  # type: ignore[call-arg]
        y_series = pd.Series(y)

        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )
        model.fit(X_df, y_series, verbose=False)

        assert model.is_fitted is True

    def test_fit_with_validation(self, sample_time_series_with_validation):
        """Should fit with validation set."""
        X_train, y_train, X_val, y_val = sample_time_series_with_validation

        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=10,
            patience=3,
            device="cpu",
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

        assert model.is_fitted is True
        # Should have validation loss history
        assert len(model._history["val_loss"]) > 0

    def test_fit_early_stopping(self, sample_time_series_with_validation):
        """Should stop early when validation loss doesn't improve."""
        X_train, y_train, X_val, y_val = sample_time_series_with_validation

        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=1000,  # High number
            patience=3,
            device="cpu",
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

        # Should have stopped before 1000 epochs
        assert len(model._history["train_loss"]) < 1000

    def test_fit_insufficient_data_raises(self):
        """Should raise error with insufficient data."""
        model = LSTMModel(sequence_length=100)
        X = np.random.randn(50, 3)  # Less than sequence_length
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="Not enough data"):
            model.fit(X, y, verbose=False)


class TestLSTMModelPredict:
    """Test cases for LSTMModel predict method."""

    def test_predict_basic(self, sample_time_series):
        """Should predict after fitting."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",  # Force CPU to avoid GPU memory issues
        )

        model.fit(X, y, verbose=False)

        # Clear any cached GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        # Predictions are for sequences, so fewer than original
        assert len(predictions) == len(X) - model.sequence_length

    def test_predict_before_fit_raises(self, sample_time_series):
        """Should raise error if predicting before fit."""
        X, _ = sample_time_series
        model = LSTMModel()

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_with_dataframe(self, sample_time_series):
        """Should predict with pandas DataFrame."""
        X, y = sample_time_series
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])  # type: ignore[call-arg]

        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )
        model.fit(X, y, verbose=False)
        predictions = model.predict(X_df)

        assert len(predictions) == len(X) - model.sequence_length

    def test_predict_insufficient_data_raises(self, sample_time_series):
        """Should raise error with insufficient prediction data."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=20,
            epochs=5,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        # Try to predict with less data than sequence_length
        X_short = X[:10]

        with pytest.raises(ValueError, match="Not enough data"):
            model.predict(X_short)


class TestLSTMModelHistory:
    """Test cases for training history."""

    def test_get_history(self, sample_time_series):
        """Should return training history."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        history = model.get_history()

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 5

    def test_history_returns_copy(self, sample_time_series):
        """Should return a copy of history."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=3,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        history = model.get_history()
        history["train_loss"] = []

        # Original history should be unchanged
        assert len(model._history["train_loss"]) == 3


class TestLSTMModelSaveLoad:
    """Test cases for save and load functionality."""

    def test_save_and_load(self, sample_time_series):
        """Should save and load model correctly."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )
        model.fit(X, y, verbose=False)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "lstm_model.joblib"
            model.save(path)

            loaded_model = LSTMModel.load(path)
            loaded_predictions = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions, decimal=4
        )


class TestLSTMModelArchitecture:
    """Test cases for model architecture."""

    def test_build_model(self, sample_time_series):
        """Should build LSTM network."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=32,
            num_layers=2,
            sequence_length=5,
            epochs=1,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        # Model should have been built
        assert model.model is not None

    def test_different_hidden_sizes(self, sample_time_series):
        """Different hidden sizes should work."""
        X, y = sample_time_series

        for hidden_size in [8, 32, 64]:
            model = LSTMModel(
                hidden_size=hidden_size,
                sequence_length=5,
                epochs=2,
                device="cpu",
            )
            model.fit(X, y, verbose=False)
            predictions = model.predict(X)

            assert len(predictions) > 0

    def test_multiple_layers(self, sample_time_series):
        """Multiple LSTM layers should work."""
        X, y = sample_time_series

        model = LSTMModel(
            hidden_size=16,
            num_layers=3,
            dropout=0.1,
            sequence_length=5,
            epochs=2,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        assert model.is_fitted


class TestLSTMModelScaling:
    """Test cases for data scaling."""

    def test_scaler_created(self, sample_time_series):
        """Should create scaler during fit."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=2,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        assert model.scaler is not None

    def test_scaling_consistency(self, sample_time_series):
        """Predictions should be consistent with scaling."""
        X, y = sample_time_series
        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=3,
            device="cpu",
        )
        model.fit(X, y, verbose=False)

        # Predict on same data twice
        pred1 = model.predict(X)
        pred2 = model.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestLSTMModel1DInput:
    """Test cases for 1D input handling."""

    def test_1d_input(self):
        """Should handle 1D input (single feature)."""
        np.random.seed(42)
        n_samples = 100

        # 1D input
        X = np.random.randn(n_samples)
        y = 0.5 * X + np.random.randn(n_samples) * 0.1

        model = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=3,
            device="cpu",
        )
        model.fit(X, y, verbose=False)
        predictions = model.predict(X)

        assert len(predictions) == len(X) - model.sequence_length


class TestLSTMModelReproducibility:
    """Test cases for reproducibility."""

    def test_reproducibility_with_seed(self, sample_time_series):
        """Setting random seeds should give reproducible results."""
        import torch

        X, y = sample_time_series

        # Set seeds
        np.random.seed(42)
        torch.manual_seed(42)

        model1 = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )
        model1.fit(X, y, verbose=False)
        pred1 = model1.predict(X)

        # Reset seeds
        np.random.seed(42)
        torch.manual_seed(42)

        model2 = LSTMModel(
            hidden_size=16,
            sequence_length=5,
            epochs=5,
            device="cpu",
        )
        model2.fit(X, y, verbose=False)
        pred2 = model2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
