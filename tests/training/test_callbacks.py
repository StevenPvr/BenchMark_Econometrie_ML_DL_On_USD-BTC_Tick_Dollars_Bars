"""Tests for src/training/callbacks.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TrainingHistory,
    ProgressLogger,
    CallbackList,
)


class MockModel:
    """Mock model for callback testing."""

    def __init__(self):
        self.saved = False
        self.save_path: Path | None = None

    def save(self, path: Path) -> None:
        """Mock save method."""
        self.saved = True
        self.save_path = path


class TestEarlyStopping:
    """Test cases for EarlyStopping callback."""

    def test_default_values(self):
        """Should have sensible defaults."""
        es = EarlyStopping()

        assert es.monitor == "val_loss"
        assert es.patience == 10
        assert es.min_delta == 0.0
        assert es.mode == "min"

    def test_raises_for_invalid_mode(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="invalid")

    def test_on_train_begin_resets_state(self):
        """on_train_begin should reset internal state."""
        es = EarlyStopping(mode="min")
        es._best_value = 0.5
        es._wait = 5

        es.on_train_begin()

        assert es._best_value == float("inf")
        assert es._wait == 0

    def test_continues_with_improvement(self):
        """Should return True when metric improves."""
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        es.on_train_begin()

        result = es.on_epoch_end(0, {"val_loss": 1.0})
        assert result is True

        result = es.on_epoch_end(1, {"val_loss": 0.9})
        assert result is True

    def test_stops_after_patience(self):
        """Should return False after patience epochs without improvement."""
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=False)
        es.on_train_begin()

        # First epoch sets best value
        es.on_epoch_end(0, {"val_loss": 0.5})

        # No improvement
        es.on_epoch_end(1, {"val_loss": 0.6})
        es.on_epoch_end(2, {"val_loss": 0.7})
        result = es.on_epoch_end(3, {"val_loss": 0.8})

        assert result is False
        assert es.stopped_epoch == 3

    def test_max_mode(self):
        """Should work correctly in max mode."""
        es = EarlyStopping(monitor="val_acc", patience=2, mode="max", verbose=False)
        es.on_train_begin()

        es.on_epoch_end(0, {"val_acc": 0.5})
        result = es.on_epoch_end(1, {"val_acc": 0.6})  # Improvement

        assert result is True
        assert es.best_value == 0.6

    def test_min_delta(self):
        """Should require min_delta improvement."""
        es = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.1, mode="min", verbose=False)
        es.on_train_begin()

        es.on_epoch_end(0, {"val_loss": 1.0})
        es.on_epoch_end(1, {"val_loss": 0.95})  # Not enough improvement
        result = es.on_epoch_end(2, {"val_loss": 0.92})  # Still not enough

        assert result is False  # Stopped

    def test_continues_when_monitor_missing(self):
        """Should continue training when monitored metric is missing."""
        es = EarlyStopping(monitor="val_loss", verbose=False)
        es.on_train_begin()

        result = es.on_epoch_end(0, {"other_metric": 0.5})

        assert result is True

    def test_best_epoch_property(self):
        """Should track best epoch."""
        es = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=False)
        es.on_train_begin()

        es.on_epoch_end(0, {"val_loss": 1.0})
        es.on_epoch_end(1, {"val_loss": 0.5})  # Best
        es.on_epoch_end(2, {"val_loss": 0.6})

        assert es.best_epoch == 1


class TestModelCheckpoint:
    """Test cases for ModelCheckpoint callback."""

    def test_saves_model_on_improvement(self):
        """Should save model when metric improves."""
        model = MockModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                filepath=Path(temp_dir) / "model.pt",
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=False,
            )
            checkpoint.on_train_begin()

            # First epoch
            checkpoint.on_epoch_end(0, {"val_loss": 1.0, "model": model})
            assert model.saved

            model.saved = False
            # Improvement
            checkpoint.on_epoch_end(1, {"val_loss": 0.5, "model": model})
            assert model.saved

    def test_does_not_save_without_improvement(self):
        """Should not save when save_best_only and no improvement."""
        model = MockModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                filepath=Path(temp_dir) / "model.pt",
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=False,
            )
            checkpoint.on_train_begin()

            checkpoint.on_epoch_end(0, {"val_loss": 0.5, "model": model})
            model.saved = False

            # No improvement
            checkpoint.on_epoch_end(1, {"val_loss": 0.6, "model": model})
            assert not model.saved

    def test_always_saves_when_not_best_only(self):
        """Should always save when save_best_only=False."""
        model = MockModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                filepath=Path(temp_dir) / "model.pt",
                save_best_only=False,
                verbose=False,
            )
            checkpoint.on_train_begin()

            checkpoint.on_epoch_end(0, {"val_loss": 0.5, "model": model})
            assert model.saved

            model.saved = False
            checkpoint.on_epoch_end(1, {"val_loss": 0.6, "model": model})
            assert model.saved

    def test_returns_true(self):
        """Should always return True (doesn't stop training)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(filepath=Path(temp_dir) / "model.pt")
            checkpoint.on_train_begin()

            result = checkpoint.on_epoch_end(0, {})
            assert result is True


class TestTrainingHistory:
    """Test cases for TrainingHistory callback."""

    def test_records_metrics(self):
        """Should record metrics history."""
        history = TrainingHistory()
        history.on_train_begin()

        history.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.2})
        history.on_epoch_end(1, {"loss": 0.8, "val_loss": 1.0})
        history.on_epoch_end(2, {"loss": 0.6, "val_loss": 0.9})

        assert history.history["loss"] == [1.0, 0.8, 0.6]
        assert history.history["val_loss"] == [1.2, 1.0, 0.9]

    def test_ignores_model_key(self):
        """Should not record 'model' key."""
        history = TrainingHistory()
        history.on_train_begin()

        history.on_epoch_end(0, {"loss": 1.0, "model": MockModel()})

        assert "model" not in history.history

    def test_resets_on_train_begin(self):
        """on_train_begin should reset history."""
        history = TrainingHistory()
        history.history = {"loss": [1.0, 0.8]}

        history.on_train_begin()

        assert history.history == {}

    def test_get_best_epoch_min(self):
        """Should find best epoch for minimizing."""
        history = TrainingHistory()
        history.on_train_begin()

        history.on_epoch_end(0, {"loss": 1.0})
        history.on_epoch_end(1, {"loss": 0.5})  # Best
        history.on_epoch_end(2, {"loss": 0.7})

        assert history.get_best_epoch("loss", mode="min") == 1

    def test_get_best_epoch_max(self):
        """Should find best epoch for maximizing."""
        history = TrainingHistory()
        history.on_train_begin()

        history.on_epoch_end(0, {"acc": 0.7})
        history.on_epoch_end(1, {"acc": 0.9})  # Best
        history.on_epoch_end(2, {"acc": 0.8})

        assert history.get_best_epoch("acc", mode="max") == 1

    def test_get_best_epoch_raises_for_missing(self):
        """Should raise ValueError for missing metric."""
        history = TrainingHistory()
        history.on_train_begin()

        with pytest.raises(ValueError, match="not found"):
            history.get_best_epoch("nonexistent")

    def test_returns_true(self):
        """Should always return True."""
        history = TrainingHistory()
        history.on_train_begin()

        result = history.on_epoch_end(0, {"loss": 1.0})
        assert result is True


class TestProgressLogger:
    """Test cases for ProgressLogger callback."""

    def test_returns_true(self):
        """Should always return True."""
        logger_cb = ProgressLogger()

        result = logger_cb.on_epoch_end(0, {"loss": 1.0})
        assert result is True

    def test_logs_every_n_epochs(self):
        """Should only log every N epochs."""
        logger_cb = ProgressLogger(log_every=5)

        # This mainly tests that no exceptions are raised
        for epoch in range(10):
            logger_cb.on_epoch_end(epoch, {"loss": 1.0 - epoch * 0.1})


class TestCallbackList:
    """Test cases for CallbackList class."""

    def test_calls_all_callbacks(self):
        """Should call all callbacks."""
        history = TrainingHistory()
        logger_cb = ProgressLogger(log_every=100)  # Don't actually log

        callbacks = CallbackList([history, logger_cb])
        callbacks.on_train_begin()
        callbacks.on_epoch_end(0, {"loss": 1.0})

        assert history.history["loss"] == [1.0]

    def test_stops_when_callback_returns_false(self):
        """Should return False when any callback returns False."""
        es = EarlyStopping(monitor="val_loss", patience=1, verbose=False)
        history = TrainingHistory()

        callbacks = CallbackList([es, history])
        callbacks.on_train_begin()

        callbacks.on_epoch_end(0, {"val_loss": 0.5})
        callbacks.on_epoch_end(1, {"val_loss": 0.6})  # No improvement
        result = callbacks.on_epoch_end(2, {"val_loss": 0.7})  # Patience exhausted

        assert result is False

    def test_continues_when_all_return_true(self):
        """Should return True when all callbacks return True."""
        history = TrainingHistory()
        logger_cb = ProgressLogger(log_every=100)

        callbacks = CallbackList([history, logger_cb])
        callbacks.on_train_begin()

        result = callbacks.on_epoch_end(0, {"loss": 1.0})
        assert result is True

    def test_append_callback(self):
        """Should allow appending callbacks."""
        callbacks = CallbackList()
        callbacks.append(TrainingHistory())

        assert len(callbacks.callbacks) == 1

    def test_empty_callback_list(self):
        """Should work with empty callback list."""
        callbacks = CallbackList()

        callbacks.on_train_begin()
        result = callbacks.on_epoch_end(0, {})
        callbacks.on_train_end()

        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
