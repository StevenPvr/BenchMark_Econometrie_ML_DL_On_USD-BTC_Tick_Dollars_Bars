"""
Training utilities and mixins.

This module provides training-related utilities and mixins
that can be used by model classes.
"""

from __future__ import annotations

from abc import ABC
from typing import Any


class TrainableModel(ABC):
    """
    Mixin class for models that support advanced training features.

    This mixin provides a common interface for:
    - Early stopping
    - Learning rate scheduling
    - Validation during training
    - Training callbacks
    """

    def on_train_begin(self, **kwargs: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        """Called at the end of each epoch."""
        pass
