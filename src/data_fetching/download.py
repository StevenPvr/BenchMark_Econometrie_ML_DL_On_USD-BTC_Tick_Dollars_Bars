"""Download functions for fetching ticker data from yfinance."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from src.constants import (
    DATA_FETCH_END_DATE,
    DATA_FETCH_START_DATE,
    NORMALIZED_DATE_COLUMN,
)
from src.utils import get_logger, parse_date_value

logger = get_logger(__name__)


def get_date_range() -> tuple[datetime, datetime]:
    """Return the configured historical download window."""
    return DATA_FETCH_START_DATE, DATA_FETCH_END_DATE


def _validate_ticker_input(ticker: str) -> None:
    """Validate ticker symbol input.

    Args:
        ticker: Ticker symbol.

    Raises:
        ValueError: If ticker is empty or only whitespace.
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol must be a non-empty string")


def _validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """Validate the date range for download.

    Args:
        start_date: Start date.
        end_date: End date.

    Raises:
        ValueError: If start_date >= end_date.
    """
    if start_date >= end_date:
        msg = f"Invalid date range: start_date {start_date} >= end_date {end_date}"
        raise ValueError(msg)


def _check_date_range_coverage(
    min_date: datetime,
    max_date: datetime,
    start_date: datetime,
    end_date: datetime,
    ticker: str,
) -> bool:
    """Check if date range covers the requested period.

    Args:
        min_date: Minimum date in the data.
        max_date: Maximum date in the data.
        start_date: Requested start date.
        end_date: Requested end date.
        ticker: Ticker symbol for logging.

    Returns:
        True if date range is valid, False otherwise.
    """
    if min_date > end_date or max_date < start_date:
        logger.warning(
            "Data for ticker %s outside requested range: [%s, %s]",
            ticker,
            min_date,
            max_date,
        )
        return False
    return True


def _has_date_column(ticker_data: pd.DataFrame, ticker: str) -> bool:
    """Return True if ticker_data exposes a Date column."""
    if NORMALIZED_DATE_COLUMN in ticker_data.columns:
        return True
    logger.warning("No '%s' column in data for ticker %s", NORMALIZED_DATE_COLUMN, ticker)
    return False


def _normalize_date_value(value: object, ticker: str, label: str) -> pd.Timestamp | None:
    """Convert arbitrary date representations to naive timestamps.

    Delegates to src.utils.parse_date_value() for consistency.
    """
    context = f"{label} for ticker {ticker}"
    return parse_date_value(value, context=context, allow_none=False)


def _extract_date_bounds(
    ticker_data: pd.DataFrame, ticker: str
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (min_date, max_date) if they can be derived from ticker_data."""
    min_raw = ticker_data[NORMALIZED_DATE_COLUMN].min()
    max_raw = ticker_data[NORMALIZED_DATE_COLUMN].max()
    min_date = _normalize_date_value(min_raw, ticker, "Min")
    max_date = _normalize_date_value(max_raw, ticker, "Max")
    if min_date is None or max_date is None:
        return None
    return min_date, max_date


def is_valid_ticker_data(
    ticker_data: pd.DataFrame,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> bool:
    """Check that downloaded data is non-empty and spans the requested window."""
    if ticker_data.empty:
        logger.warning("Empty data received for ticker %s", ticker)
        return False
    if not _has_date_column(ticker_data, ticker):
        return False
    bounds = _extract_date_bounds(ticker_data, ticker)
    if bounds is None:
        return False
    min_date, max_date = bounds
    return _check_date_range_coverage(min_date, max_date, start_date, end_date, ticker)


def _handle_download_error(ticker: str, error: Exception) -> None:
    """Handle download errors with appropriate logging.

    Args:
        ticker: Ticker symbol.
        error: Exception raised during download.
    """
    logger.error("Error downloading data for ticker %s: %s", ticker, error)


def _download_yfinance_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Download raw data from yfinance with a clear fallback.

    Tries ``yf.download`` first (used in offline tests via monkeypatch).
    If the result is empty or raises, falls back to ``Ticker.history``
    which unit tests stub explicitly. Always exposes a ``Date`` column
    for downstream validation regardless of index name.

    Args:
        ticker: Ticker symbol.
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame with OHLCV data and a Date column.
        Stock splits and dividends are excluded.
    """
    hist: pd.DataFrame
    try:
        # Prefer the vectorized path when provided by tests (monkeypatched)
        hist = yf.download(  # type: ignore[attr-defined]
            ticker, start=start_date, end=end_date, progress=False, auto_adjust=True
        )
        if hist is None or getattr(hist, "empty", True):
            raise ValueError("yf.download returned empty data")
    except Exception:
        # Fallback to per-ticker history when download is unavailable/empty
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start_date, end=end_date, actions=False)

    # Ensure a 'Date' column exists for validation regardless of index name
    hist = hist.copy()

    # Handle MultiIndex columns from yf.download() (e.g., ('Close', 'A'))
    if isinstance(hist.columns, pd.MultiIndex):
        # Flatten MultiIndex to first level
        hist.columns = hist.columns.get_level_values(0)

    if NORMALIZED_DATE_COLUMN not in hist.columns:
        hist[NORMALIZED_DATE_COLUMN] = hist.index
    hist = hist.reset_index(drop=True)
    return hist


def download_ticker_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame | None:
    """Download and validate ticker data from yfinance."""
    try:
        _validate_ticker_input(ticker)
        _validate_date_range(start_date, end_date)
    except ValueError as exc:
        logger.warning("Invalid input for ticker %s: %s", ticker, exc)
        return None

    try:
        ticker_data = _download_yfinance_data(ticker, start_date, end_date)
    except Exception as exc:  # pragma: no cover - defensive
        _handle_download_error(ticker, exc)
        return None

    if not is_valid_ticker_data(ticker_data, ticker, start_date, end_date):
        return None

    logger.info(
        "Downloaded %d rows of data for %s between %s and %s",
        len(ticker_data),
        ticker,
        start_date.date(),
        end_date.date(),
    )
    return ticker_data
