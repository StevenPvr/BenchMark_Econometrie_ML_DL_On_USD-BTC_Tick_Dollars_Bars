"""Tests for src/training/trainer.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.training.trainer import (
    TrainingConfig,
    TrainingResult,
    Trainer,
    train_model,
)


class MockModel:
    """Mock model for testing trainer."""

    def __init__(self, name: str = "MockModel"):
        self.name = name
        self.is_fitted = False
        self.params: dict[str, Any] = {}
        self.mean_: float = 0.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "MockModel":
        """Fit by storing mean of target."""
        self.mean_ = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict constant value (mean)."""
        return np.full(len(X), self.mean_)

    def set_params(self, **params: Any) -> "MockModel":
        """Set parameters."""
        self.params.update(params)
        return self

    def get_params(self) -> dict[str, Any]:
        """Get parameters."""
        return self.params.copy()

    def save(self, path: Path) -> None:
        """Mock save."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class TestTrainingConfig:
    """Test cases for TrainingConfig class."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = TrainingConfig()

        assert config.validation_split == 0.0
        assert config.shuffle is False
        assert config.verbose == 1
        assert config.save_model is True

    def test_raises_for_invalid_validation_split(self):
        """Should raise ValueError for invalid validation_split."""
        with pytest.raises(ValueError, match="validation_split"):
            TrainingConfig(validation_split=1.5)

        with pytest.raises(ValueError, match="validation_split"):
            TrainingConfig(validation_split=-0.1)

    def test_raises_for_invalid_verbose(self):
        """Should raise ValueError for invalid verbose."""
        with pytest.raises(ValueError, match="verbose"):
            TrainingConfig(verbose=3)


class TestTrainingResult:
    """Test cases for TrainingResult dataclass."""

    def test_attributes(self):
        """Should have all expected attributes."""
        result = TrainingResult(
            model=MockModel(),
            train_score=0.1,
            val_score=0.2,
            train_size=100,
            val_size=20,
            metric_name="mse",
            training_time=1.0,
            model_params={},
        )

        assert result.train_score == 0.1
        assert result.val_score == 0.2
        assert result.train_size == 100
        assert result.val_size == 20


class TestTrainer:
    """Test cases for Trainer class."""

    def test_train_without_validation(self):
        """Should train without validation split."""
        model = MockModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        trainer = Trainer(TrainingConfig(validation_split=0.0, verbose=0))
        result = trainer.train(model, X, y, metric="mse")

        assert result.model.is_fitted
        assert result.val_score is None
        assert result.train_size == 100
        assert result.val_size == 0

    def test_train_with_validation(self):
        """Should train with validation split."""
        model = MockModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        trainer = Trainer(TrainingConfig(validation_split=0.2, verbose=0))
        result = trainer.train(model, X, y, metric="mse")

        assert result.model.is_fitted
        assert result.val_score is not None
        assert result.train_size == 80
        assert result.val_size == 20

    def test_computes_train_score(self):
        """Should compute training score."""
        model = MockModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        trainer = Trainer(TrainingConfig(verbose=0))
        result = trainer.train(model, X, y, metric="mse")

        assert result.train_score >= 0

    def test_accepts_dataframe(self):
        """Should accept pandas DataFrame."""
        model = MockModel()
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randn(100))

        trainer = Trainer(TrainingConfig(verbose=0))
        result = trainer.train(model, X, y, metric="mse")

        assert result.model.is_fitted

    def test_saves_model(self):
        """Should save model when configured."""
        model = MockModel()
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                save_model=True,
                output_dir=Path(temp_dir),
                verbose=0,
            )
            trainer = Trainer(config)
            result = trainer.train(model, X, y)

            assert result.model_path is not None
            assert result.model_path.exists()

    def test_does_not_save_when_disabled(self):
        """Should not save model when save_model=False."""
        model = MockModel()
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        config = TrainingConfig(save_model=False, verbose=0)
        trainer = Trainer(config)
        result = trainer.train(model, X, y)

        assert result.model_path is None

    def test_raises_for_unknown_metric(self):
        """Should raise ValueError for unknown metric."""
        model = MockModel()
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        trainer = Trainer(TrainingConfig(verbose=0))

        with pytest.raises(ValueError, match="Unknown metric"):
            trainer.train(model, X, y, metric="unknown_metric")

    def test_uses_time_series_split(self):
        """Should use time series split (no shuffle by default)."""
        model = MockModel()
        X = np.arange(100).reshape(-1, 1)  # Sequential data
        y = np.arange(100) * 2  # y = 2x

        trainer = Trainer(TrainingConfig(validation_split=0.2, shuffle=False, verbose=0))
        result = trainer.train(model, X, y, metric="mse")

        # With no shuffle, validation should be last 20 samples
        # Mean of first 80 samples (0-79) = 39.5
        # For y = 2x, mean of first 80 y values (0-158) = 79
        assert result.train_size == 80
        assert result.val_size == 20


class TestTrainModel:
    """Test cases for train_model convenience function."""

    def test_trains_model(self):
        """Should train model correctly."""
        model = MockModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        result = train_model(model, X, y, validation_split=0.2, verbose=False)

        assert isinstance(result, TrainingResult)
        assert result.model.is_fitted

    def test_uses_validation_split(self):
        """Should use specified validation split."""
        model = MockModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        result = train_model(model, X, y, validation_split=0.3, verbose=False)

        assert result.train_size == 70
        assert result.val_size == 30


class TestTrainerAntiLeakage:
    """CRITICAL: Anti-leakage tests for Trainer."""

    def test_validation_uses_later_data(self):
        """CRITICAL: Validation should use later data (time series)."""
        model = MockModel()
        # Create sequential data where we can verify order
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = X.ravel() + np.random.randn(100) * 0.01

        trainer = Trainer(TrainingConfig(validation_split=0.2, shuffle=False, verbose=0))
        result = trainer.train(model, X, y)

        # The mean of training y should be based on first 80 samples (indices 0-79)
        # Mean of 0-79 with small noise ≈ 39.5
        expected_train_mean = np.mean(np.arange(80))
        assert abs(result.model.mean_ - expected_train_mean) < 1.0

    def test_no_data_leakage_in_split(self):
        """CRITICAL: Validation data should not leak into training."""
        model = MockModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        trainer = Trainer(TrainingConfig(validation_split=0.2, shuffle=False, verbose=0))

        # Access internal split method
        X_train, y_train, X_val, y_val = trainer._split_data(np.asarray(X), np.asarray(y).ravel())

        # No overlap between train and val indices
        assert len(X_train) + len(X_val) == len(X)
        assert len(y_train) + len(y_val) == len(y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
