"""Data fetching module for S&P 500 tickers and historical data."""

from __future__ import annotations

from src.data_fetching.data_fetching import download_sp500_data, fetch_sp500_tickers

__all__ = ["fetch_sp500_tickers", "download_sp500_data"]
