"""Thread-safe rate limiter for API calls."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING

from src.constants import (
    FETCHING_MAX_REQUEST_WEIGHT_PER_MINUTE,
    FETCHING_REQUEST_WEIGHT_PER_CALL,
)
from src.utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Deque

logger = get_logger(__name__)


class RateLimiter:
    """Thread-safe rate limiter for API calls with weight-based limiting.

    This limiter is shared across ALL threads/workers to enforce a global
    rate limit across the entire application.

    Attributes:
        max_weight_per_minute: Maximum allowed weight per minute.
        weight_per_call: Weight consumed by each call.
        call_times: Deque storing timestamps of API calls.
        lock: Threading lock for thread safety.
    """

    def __init__(self, max_weight_per_minute: int, weight_per_call: int) -> None:
        """Initialize the rate limiter.

        Args:
            max_weight_per_minute: Maximum API weight allowed per minute.
            weight_per_call: Weight consumed by each API call.
        """
        self.max_weight_per_minute = max_weight_per_minute
        self.weight_per_call = weight_per_call
        self.call_times: Deque[float] = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits.

        Thread-safe: uses a lock to ensure accurate tracking across all workers.
        """
        while True:
            with self.lock:
                current_time = time.time()
                self._remove_expired_calls(current_time)
                current_weight = len(self.call_times) * self.weight_per_call

                if current_weight < self.max_weight_per_minute:
                    self.call_times.append(current_time)
                    return

                wait_time = self._compute_wait_time(current_time)

            if wait_time > 0:
                logger.info(
                    "Rate limit reached (%d/%d), waiting %.2f seconds",
                    current_weight,
                    self.max_weight_per_minute,
                    wait_time,
                )
                time.sleep(wait_time)

    def _remove_expired_calls(self, current_time: float) -> None:
        """Remove calls older than 1 minute from the deque.

        Args:
            current_time: Current timestamp in seconds.
        """
        while self.call_times and current_time - self.call_times[0] > 60:
            self.call_times.popleft()

    def _compute_wait_time(self, current_time: float) -> float:
        """Compute wait time until oldest call expires.

        Args:
            current_time: Current timestamp in seconds.

        Returns:
            Wait time in seconds.
        """
        if self.call_times:
            return 60 - (current_time - self.call_times[0]) + 0.1
        return 0.1


# Global rate limiter instance shared across ALL workers
rate_limiter = RateLimiter(
    FETCHING_MAX_REQUEST_WEIGHT_PER_MINUTE, FETCHING_REQUEST_WEIGHT_PER_CALL
)


__all__ = [
    "RateLimiter",
    "rate_limiter",
]
