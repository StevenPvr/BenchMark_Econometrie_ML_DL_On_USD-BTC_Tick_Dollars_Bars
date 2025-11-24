"""Refit management module for EGARCH models.

This module provides unified refit management:
- Window management (expanding/rolling)
- Refit scheduling
- Refit manager (combines windows + schedule)

Eliminates code duplication between training_garch and garch_eval.
"""

from __future__ import annotations

from src.garch.garch_params.refit.refit_manager import (
    ExpandingWindow,
    RefitEvent,
    RefitManager,
    RefitSchedule,
    RollingWindow,
    Window,
    create_periodic_schedule,
    create_window_manager,
)

__all__ = [
    # Refit Manager
    "RefitManager",
    "RefitEvent",
    # Windows
    "Window",
    "ExpandingWindow",
    "RollingWindow",
    "create_window_manager",
    # Schedule
    "RefitSchedule",
    "create_periodic_schedule",
]
