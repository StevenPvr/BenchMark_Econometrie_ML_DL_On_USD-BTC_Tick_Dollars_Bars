"""Data fetching module for BTC/USD tick data.

This module provides functionality to download cryptocurrency tick data
from exchanges using the ccxt library, with support for parallel fetching,
rate limiting, and incremental data storage.

Public API:
    download_ticks_in_date_range: Download trades for a date range.
    RateLimiter: Thread-safe rate limiter for API calls.
    main: Entry point for automated fetching.
    main_loop: Entry point for daemon mode.
"""

from __future__ import annotations

from src.data_fetching.fetching import RateLimiter, download_ticks_in_date_range
from src.data_fetching.main import main, main_loop

__all__ = [
    "RateLimiter",
    "download_ticks_in_date_range",
    "main",
    "main_loop",
]
