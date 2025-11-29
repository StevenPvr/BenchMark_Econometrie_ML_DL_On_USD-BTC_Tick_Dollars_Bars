"""Data fetching module for BTC/USD tick data."""

from __future__ import annotations # type: ignore[import-untyped]

def download_ticks_in_date_range() -> None:
    from src.data_fetching.fetching import download_ticks_in_date_range as _impl
    _impl()


__all__ = ["download_ticks_in_date_range"]
