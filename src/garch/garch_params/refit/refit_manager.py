"""Refit manager for EGARCH models.

This module provides unified refit management:
- Combines window management and scheduling
- Tracks refit history
- Manages parameter updates
- Ensures temporal causality

Unifies refit logic previously duplicated between training_garch and garch_eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.garch.garch_params.estimation import ConvergenceResult, estimate_egarch_mle
from src.utils import get_logger

logger = get_logger(__name__)


# ===================== Windows and Schedules (merged) =====================


@dataclass(frozen=True)
class Window:
    """Training window specification.

    Attributes:
        start: Window start index (inclusive).
        end: Window end index (exclusive).
    """

    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def validate(self) -> None:
        if self.start < 0:
            raise ValueError(f"Window start ({self.start}) must be >= 0")
        if self.end <= self.start:
            raise ValueError(f"Window end ({self.end}) must be > start ({self.start})")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"Window[{self.start}:{self.end}] (size={self.size})"


class ExpandingWindow:
    """Expanding window manager using [start:current_pos)."""

    def __init__(self, start: int = 0) -> None:
        self.start = start

    def compute_window(self, current_pos: int) -> Window:
        if current_pos <= self.start:
            raise ValueError(f"current_pos ({current_pos}) must be > start ({self.start})")
        window = Window(start=self.start, end=current_pos)
        window.validate()
        return window

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"ExpandingWindow(start={self.start})"


class RollingWindow:
    """Rolling window manager using (current_pos - size, current_pos)."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size ({window_size}) must be > 0")
        self.window_size = window_size

    def compute_window(self, current_pos: int) -> Window:
        if current_pos <= 0:
            raise ValueError(f"current_pos ({current_pos}) must be > 0")
        start = max(0, current_pos - self.window_size)
        window = Window(start=start, end=current_pos)
        window.validate()
        return window

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"RollingWindow(size={self.window_size})"


def create_window_manager(
    window_type: str, window_size: int | None = None, start: int = 0
) -> ExpandingWindow | RollingWindow:
    if window_type == "expanding":
        return ExpandingWindow(start=start)
    if window_type == "rolling":
        if window_size is None:
            raise ValueError("window_size required for rolling window")
        return RollingWindow(window_size=window_size)
    raise ValueError(f"Invalid window_type: {window_type}. Must be 'expanding' or 'rolling'.")


@dataclass(frozen=True)
class RefitSchedule:
    """Refit schedule specification."""

    frequency: int
    start_pos: int
    end_pos: int

    def __post_init__(self) -> None:
        if self.frequency < 0:
            raise ValueError(f"frequency ({self.frequency}) must be >= 0")
        if self.start_pos < 0:
            raise ValueError(f"start_pos ({self.start_pos}) must be >= 0")
        if self.end_pos <= self.start_pos:
            raise ValueError(f"end_pos ({self.end_pos}) must be > start_pos ({self.start_pos})")

    def get_refit_positions(self) -> list[int]:
        return list(range(self.start_pos, self.end_pos, self.frequency))

    def should_refit(self, position: int) -> bool:
        if self.frequency == 0:
            return False
        if position < self.start_pos or position >= self.end_pos:
            return False
        return (position - self.start_pos) % self.frequency == 0

    def count_refits(self) -> int:
        return len(self.get_refit_positions())

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"RefitSchedule(freq={self.frequency}, start={self.start_pos}, end={self.end_pos}, "
            f"n_refits={self.count_refits()})"
        )


def create_periodic_schedule(frequency: int, start_pos: int, end_pos: int) -> RefitSchedule:
    return RefitSchedule(frequency=frequency, start_pos=start_pos, end_pos=end_pos)


## Note: compute_next_refit_pos and is_refit_position were removed as unused helpers.


@dataclass
class RefitEvent:
    """Record of a single refit event.

    Attributes:
        position: Position where refit occurred.
        window: Training window used for refit.
        params: Estimated parameters.
        convergence: Convergence result.
    """

    position: int
    window: Window
    params: dict[str, float]
    convergence: ConvergenceResult

    def __str__(self) -> str:
        """String representation."""
        return (
            f"RefitEvent(pos={self.position}, {self.window}, "
            f"converged={self.convergence.converged}, "
            f"loglik={self.convergence.final_loglik:.2f})"
        )


class RefitManager:
    """Manage periodic refit for EGARCH models.

    Handles:
    - Window management (expanding/rolling)
    - Refit scheduling
    - Parameter estimation
    - History tracking
    """

    def __init__(
        self,
        *,
        frequency: int,
        window_type: str,
        window_size: int | None = None,
        o: int = 1,
        p: int = 1,
        dist: str = "student",
        use_fixed_params: bool = False,
    ) -> None:
        """Initialize refit manager.

        Args:
            frequency: Refit every N steps.
            window_type: Window type ('expanding' or 'rolling').
            window_size: Rolling window size (required if window_type='rolling').
            o: ARCH order.
            p: GARCH order.
            dist: Distribution name.

        Raises:
            ValueError: If parameters are invalid.
        """
        self.frequency = frequency
        self.window_type = window_type
        self.window_size = window_size
        self.o = o
        self.p = p
        self.dist = dist
        self.use_fixed_params = use_fixed_params

        self.window_manager: ExpandingWindow | RollingWindow = create_window_manager(
            window_type, window_size
        )
        self.refit_history: list[RefitEvent] = []
        self.current_params: dict[str, float] | None = None

    def should_refit(self, position: int, start_pos: int) -> bool:
        """Check if refit should occur at position.

        Args:
            position: Current position.
            start_pos: First refit position.

        Returns:
            True if refit should occur.
        """
        if self.use_fixed_params:
            return False
        if self.frequency == 0:
            return False  # No refit if frequency is 0
        if position < start_pos:
            return False
        return (position - start_pos) % self.frequency == 0

    def compute_refit_window(self, position: int) -> Window:
        """Compute training window for refit at position.

        Args:
            position: Refit position (exclusive end of window).

        Returns:
            Training window.
        """
        return self.window_manager.compute_window(position)

    def perform_refit(
        self,
        residuals: np.ndarray,
        position: int,
    ) -> tuple[dict[str, float], ConvergenceResult]:
        """Perform refit at position.

        Args:
            residuals: Full residual series.
            position: Refit position.

        Returns:
            Tuple of (parameters, convergence_result).

        Raises:
            RuntimeError: If refit fails.
        """
        window = self.compute_refit_window(position)
        logger.info("Refit at position %d: %s", position, window)

        # Extract training window with temporal boundary check
        window_end = min(window.end, len(residuals))
        residuals_window = residuals[window.start : window_end]

        # Estimate parameters
        params, convergence = estimate_egarch_mle(
            residuals_window,
            o=self.o,
            p=self.p,
            dist=self.dist,
        )

        # Record refit event
        event = RefitEvent(
            position=position,
            window=window,
            params=params,
            convergence=convergence,
        )
        self.refit_history.append(event)

        # Update current parameters
        self.current_params = params

        if not convergence.converged:
            logger.warning("Refit at position %d failed to converge", position)

        return params, convergence

    def get_current_params(self) -> dict[str, float]:
        """Get current parameters.

        Returns:
            Current parameter dictionary.

        Raises:
            RuntimeError: If no refit has been performed yet.
        """
        if self.current_params is None:
            msg = "No refit performed yet - current_params is None"
            raise RuntimeError(msg)
        return self.current_params

    def get_convergence_rate(self) -> float:
        """Compute convergence rate across all refits.

        Returns:
            Convergence rate in [0, 1].
        """
        if not self.refit_history:
            return 0.0
        n_converged = sum(1 for event in self.refit_history if event.convergence.converged)
        return n_converged / len(self.refit_history)

    def get_summary(self) -> dict[str, Any]:
        """Get refit summary statistics.

        Returns:
            Dictionary with refit statistics.
        """
        if not self.refit_history:
            return {
                "n_refits": 0,
                "n_converged": 0,
                "convergence_rate": 0.0,
                "mean_loglik": None,
            }

        n_converged = sum(1 for event in self.refit_history if event.convergence.converged)
        convergence_rate = n_converged / len(self.refit_history)

        logliks = [
            event.convergence.final_loglik
            for event in self.refit_history
            if event.convergence.converged
        ]
        mean_loglik = float(np.mean(logliks)) if logliks else None

        return {
            "n_refits": len(self.refit_history),
            "n_converged": n_converged,
            "convergence_rate": convergence_rate,
            "mean_loglik": mean_loglik,
        }

    def log_summary(self, min_rate: float = 0.95) -> None:
        """Log refit summary with warnings if convergence rate is low.

        Args:
            min_rate: Minimum acceptable convergence rate.
        """
        summary = self.get_summary()

        if summary["n_refits"] == 0:
            logger.warning("No refits performed")
            return

        conv_rate = summary["convergence_rate"]
        logger.info(
            "Refit summary: %d refits, %d converged (%.1f%%)",
            summary["n_refits"],
            summary["n_converged"],
            conv_rate * 100,
        )

        if summary["mean_loglik"] is not None:
            logger.info("Mean log-likelihood: %.2f", summary["mean_loglik"])

        if conv_rate < min_rate:
            logger.warning(
                "Low refit convergence rate: %.1f%% < %.1f%% threshold",
                conv_rate * 100,
                min_rate * 100,
            )

    def create_schedule(self, start_pos: int, end_pos: int) -> RefitSchedule:
        """Create refit schedule for position range.

        Args:
            start_pos: First refit position.
            end_pos: Last position (exclusive).

        Returns:
            Refit schedule.
        """
        return create_periodic_schedule(self.frequency, start_pos, end_pos)

    def clear_history(self) -> None:
        """Clear refit history."""
        self.refit_history.clear()
        self.current_params = None

    def __str__(self) -> str:
        """String representation."""
        return (
            f"RefitManager(freq={self.frequency}, "
            f"window_type={self.window_type}, "
            f"o={self.o}, p={self.p}, dist={self.dist}, "
            f"n_refits={len(self.refit_history)})"
        )
