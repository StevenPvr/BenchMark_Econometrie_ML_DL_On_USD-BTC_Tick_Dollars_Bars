"""Training callbacks for monitoring and control.

This module provides callbacks for:
- Early stopping based on validation metrics
- Model checkpointing
- Training history logging
- Progress reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# CALLBACK PROTOCOL
# ============================================================================


class TrainingCallback(Protocol):
    """Protocol for training callbacks."""

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Called at the beginning of training."""
        ...

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called at the end of training."""
        ...

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at the beginning of each epoch."""
        ...

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Called at the end of each epoch.

        Returns:
            True to continue training, False to stop.
        """
        ...


# ============================================================================
# EARLY STOPPING
# ============================================================================


@dataclass
class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Attributes:
        monitor: Metric to monitor (e.g., "val_loss", "val_mse").
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: "min" for minimizing, "max" for maximizing.
        restore_best_weights: If True, restore model weights from best epoch.
        verbose: If True, log early stopping events.
    """

    monitor: str = "val_loss"
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"
    restore_best_weights: bool = True
    verbose: bool = True

    # Internal state
    _best_value: float = field(default=float("inf"), init=False, repr=False)
    _best_epoch: int = field(default=0, init=False, repr=False)
    _wait: int = field(default=0, init=False, repr=False)
    _stopped_epoch: int = field(default=0, init=False, repr=False)
    _best_weights: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize best value based on mode."""
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
        if self.mode == "min":
            self._best_value = float("inf")
        else:
            self._best_value = float("-inf")

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < self._best_value - self.min_delta
        return current > self._best_value + self.min_delta

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Reset state at training start."""
        self._wait = 0
        self._stopped_epoch = 0
        if self.mode == "min":
            self._best_value = float("inf")
        else:
            self._best_value = float("-inf")
        self._best_weights = None

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Log if training was stopped early."""
        if self._stopped_epoch > 0 and self.verbose:
            logger.info("Training stopped early at epoch %d", self._stopped_epoch)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at epoch start (no-op)."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Check for improvement and decide whether to stop.

        Args:
            epoch: Current epoch number.
            logs: Dict with metric values (must contain self.monitor).

        Returns:
            True to continue training, False to stop.
        """
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose:
                logger.warning(
                    "Early stopping monitor '%s' not in logs. Available: %s",
                    self.monitor,
                    list(logs.keys()),
                )
            return True

        if self._is_improvement(current):
            self._best_value = current
            self._best_epoch = epoch
            self._wait = 0
            # Store weights if restore_best_weights is enabled
            # Note: actual weight storage depends on model implementation
            if self.verbose:
                logger.debug(
                    "Epoch %d: %s improved to %.6f",
                    epoch,
                    self.monitor,
                    current,
                )
        else:
            self._wait += 1
            if self.verbose:
                logger.debug(
                    "Epoch %d: %s=%.6f did not improve (best=%.6f, wait=%d/%d)",
                    epoch,
                    self.monitor,
                    current,
                    self._best_value,
                    self._wait,
                    self.patience,
                )

            if self._wait >= self.patience:
                self._stopped_epoch = epoch
                if self.verbose:
                    logger.info(
                        "Early stopping triggered at epoch %d. "
                        "Best %s=%.6f at epoch %d",
                        epoch,
                        self.monitor,
                        self._best_value,
                        self._best_epoch,
                    )
                return False

        return True

    @property
    def stopped_epoch(self) -> int:
        """Return the epoch at which training stopped (0 if not stopped)."""
        return self._stopped_epoch

    @property
    def best_epoch(self) -> int:
        """Return the epoch with the best monitored value."""
        return self._best_epoch

    @property
    def best_value(self) -> float:
        """Return the best monitored value."""
        return self._best_value


# ============================================================================
# MODEL CHECKPOINT
# ============================================================================


@dataclass
class ModelCheckpoint:
    """Save the model during training.

    Attributes:
        filepath: Path to save the model.
        monitor: Metric to monitor for saving best model.
        save_best_only: If True, only save when monitored metric improves.
        mode: "min" or "max".
        verbose: If True, log saving events.
    """

    filepath: str | Path
    monitor: str = "val_loss"
    save_best_only: bool = True
    mode: str = "min"
    verbose: bool = True

    # Internal state
    _best_value: float = field(default=float("inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize best value based on mode."""
        if self.mode == "min":
            self._best_value = float("inf")
        else:
            self._best_value = float("-inf")
        self.filepath = Path(self.filepath)

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < self._best_value
        return current > self._best_value

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Reset state at training start."""
        if self.mode == "min":
            self._best_value = float("inf")
        else:
            self._best_value = float("-inf")

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Called at training end (no-op)."""
        pass

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at epoch start (no-op)."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Check if model should be saved.

        Args:
            epoch: Current epoch number.
            logs: Dict with metric values and 'model' key.

        Returns:
            Always True (doesn't stop training).
        """
        logs = logs or {}
        current = logs.get(self.monitor)
        model = logs.get("model")

        if model is None:
            return True

        should_save = False
        if self.save_best_only:
            if current is not None and self._is_improvement(current):
                self._best_value = current
                should_save = True
        else:
            should_save = True

        if should_save and hasattr(model, "save"):
            filepath = Path(self.filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            model.save(filepath)
            if self.verbose:
                logger.info(
                    "Epoch %d: saved model to %s (%s=%.6f)",
                    epoch,
                    filepath,
                    self.monitor,
                    current if current is not None else float("nan"),
                )

        return True


# ============================================================================
# TRAINING HISTORY
# ============================================================================


@dataclass
class TrainingHistory:
    """Record training metrics history.

    Attributes:
        history: Dict mapping metric names to lists of values.
    """

    history: dict[str, list[float]] = field(default_factory=dict)
    _start_time: datetime | None = field(default=None, init=False, repr=False)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Reset history at training start."""
        self.history = {}
        self._start_time = datetime.now()

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Record total training time."""
        if self._start_time is not None:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            self.history["training_time_seconds"] = [elapsed]

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at epoch start (no-op)."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Record metrics for this epoch.

        Args:
            epoch: Current epoch number.
            logs: Dict with metric values.

        Returns:
            Always True (doesn't stop training).
        """
        logs = logs or {}
        for key, value in logs.items():
            if key == "model":
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(value))
        return True

    def get_best_epoch(self, metric: str, mode: str = "min") -> int:
        """Get the epoch with the best value for a metric.

        Args:
            metric: Metric name.
            mode: "min" or "max".

        Returns:
            Epoch number (0-indexed).
        """
        if metric not in self.history:
            raise ValueError(f"Metric '{metric}' not found in history")

        values = self.history[metric]
        if mode == "min":
            return int(np.argmin(values))
        return int(np.argmax(values))


# ============================================================================
# PROGRESS LOGGER
# ============================================================================


@dataclass
class ProgressLogger:
    """Log training progress.

    Attributes:
        log_every: Log every N epochs.
        metrics: List of metrics to log (None = all).
    """

    log_every: int = 1
    metrics: list[str] | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Log training start."""
        logger.info("Training started")

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Log training end."""
        logger.info("Training completed")

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Called at epoch start (no-op)."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Log metrics for this epoch.

        Args:
            epoch: Current epoch number.
            logs: Dict with metric values.

        Returns:
            Always True (doesn't stop training).
        """
        if (epoch + 1) % self.log_every != 0:
            return True

        logs = logs or {}
        metrics_to_log = self.metrics or [k for k in logs if k != "model"]

        parts = [f"Epoch {epoch + 1}"]
        for metric in metrics_to_log:
            if metric in logs:
                value = logs[metric]
                if isinstance(value, float):
                    parts.append(f"{metric}={value:.6f}")

        logger.info(" - ".join(parts))
        return True


# ============================================================================
# CALLBACK LIST MANAGER
# ============================================================================


class CallbackList:
    """Manage multiple callbacks.

    Example:
        >>> callbacks = CallbackList([
        ...     EarlyStopping(patience=10),
        ...     ModelCheckpoint("model.pt"),
        ...     TrainingHistory(),
        ... ])
        >>> callbacks.on_train_begin()
        >>> for epoch in range(100):
        ...     # ... training code ...
        ...     if not callbacks.on_epoch_end(epoch, {"val_loss": loss}):
        ...         break
        >>> callbacks.on_train_end()
    """

    def __init__(self, callbacks: list[TrainingCallback] | None = None) -> None:
        """Initialize callback list.

        Args:
            callbacks: List of callbacks.
        """
        self.callbacks = callbacks or []

    def append(self, callback: TrainingCallback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Call on_train_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Call on_train_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Call on_epoch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> bool:
        """Call on_epoch_end on all callbacks.

        Args:
            epoch: Current epoch.
            logs: Metrics dict.

        Returns:
            False if any callback returned False (stop training), True otherwise.
        """
        continue_training = True
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs)
            if result is False:
                continue_training = False
        return continue_training
