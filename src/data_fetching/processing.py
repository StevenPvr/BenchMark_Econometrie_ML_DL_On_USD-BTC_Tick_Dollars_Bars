"""Data processing functions for cleaning and transforming ticker data."""

from __future__ import annotations

import pandas as pd

from src.constants import (
    NORMALIZED_ADJ_CLOSE_COLUMN,
    NORMALIZED_CLOSE_COLUMN,
    NORMALIZED_DATE_COLUMN,
    NORMALIZED_HIGH_COLUMN,
    NORMALIZED_LOW_COLUMN,
    NORMALIZED_TICKERS_COLUMN,
    NORMALIZED_VOLUME_COLUMN,
    YF_ADJ_CLOSE_COLUMN,
    YF_CLOSE_COLUMN,
    YF_DATE_COLUMN,
    YF_HIGH_COLUMN,
    YF_LOW_COLUMN,
    YF_OPEN_COLUMN,
    YF_VOLUME_COLUMN,
)
from src.utils import get_logger, validate_dataframe_not_empty, validate_required_columns

logger = get_logger(__name__)


def _validate_ticker_data_not_empty(ticker_data: pd.DataFrame, ticker: str) -> None:
    """Validate that ticker data is not empty.

    Args:
        ticker_data: DataFrame to validate.
        ticker: Ticker symbol for logging.

    Raises:
        ValueError: If ticker_data is empty.
    """
    validate_dataframe_not_empty(ticker_data, f"ticker data for {ticker}")


def _validate_required_columns(ticker_data: pd.DataFrame, ticker: str) -> None:
    """Validate that required columns exist in ticker data.

    Args:
        ticker_data: DataFrame to validate.
        ticker: Ticker symbol for logging.

    Raises:
        KeyError: If any required columns are missing.
    """
    # Require a minimal set needed downstream; High/Low are optional in tests
    required_min = [
        NORMALIZED_DATE_COLUMN,
        NORMALIZED_CLOSE_COLUMN,
        NORMALIZED_VOLUME_COLUMN,
        NORMALIZED_HIGH_COLUMN,
        NORMALIZED_LOW_COLUMN,
    ]
    validate_required_columns(ticker_data, required_min, f"ticker data for {ticker}")


def _normalize_column_names(ticker_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower snake_case.

    Args:
        ticker_data: Input DataFrame.

    Returns:
        DataFrame with normalized column names.
    """
    rename_map = {
        YF_DATE_COLUMN: NORMALIZED_DATE_COLUMN,
        YF_HIGH_COLUMN: NORMALIZED_HIGH_COLUMN,
        YF_LOW_COLUMN: NORMALIZED_LOW_COLUMN,
        YF_CLOSE_COLUMN: NORMALIZED_CLOSE_COLUMN,
        YF_ADJ_CLOSE_COLUMN: NORMALIZED_ADJ_CLOSE_COLUMN,
        YF_VOLUME_COLUMN: NORMALIZED_VOLUME_COLUMN,
    }
    return ticker_data.rename(columns=rename_map)


def _drop_na_rows(ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop rows with missing values and log the amount of data lost.

    Args:
        ticker_data: Input DataFrame.
        ticker: Ticker symbol for logging.

    Returns:
        Cleaned DataFrame without missing values.
    """
    before_count = len(ticker_data)
    cleaned = ticker_data.dropna()
    dropped_count = before_count - len(cleaned)
    if dropped_count > 0:
        logger.info(
            "Dropped %d rows with NaN values for ticker %s",
            dropped_count,
            ticker,
        )
    return cleaned


def _drop_unwanted_columns(ticker_data: pd.DataFrame) -> pd.DataFrame:
    """Drop unwanted columns that are not needed for analysis.

    Args:
        ticker_data: Input DataFrame.

    Returns:
        DataFrame with unwanted columns removed.
    """
    # Remove both original case and normalized case column names
    unwanted_columns = [
        YF_OPEN_COLUMN,
        YF_OPEN_COLUMN.lower(),
    ]  # Columns to exclude from final dataset
    return ticker_data.drop(columns=unwanted_columns, errors="ignore")


def _add_ticker_column(ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add ticker symbol as a separate column.

    Args:
        ticker_data: Input DataFrame.
        ticker: Ticker symbol.

    Returns:
        DataFrame with an added tickers column.
    """
    ticker_data = ticker_data.copy()
    ticker_data[NORMALIZED_TICKERS_COLUMN] = ticker
    return ticker_data


def process_ticker_data(ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Full processing pipeline for a single ticker."""
    _validate_ticker_data_not_empty(ticker_data, ticker)

    ticker_data = _normalize_column_names(ticker_data)
    _validate_required_columns(ticker_data, ticker)

    ticker_data = _drop_unwanted_columns(ticker_data)
    ticker_data = _drop_na_rows(ticker_data, ticker)
    ticker_data = _add_ticker_column(ticker_data, ticker)

    logger.info(
        "Processed data for ticker %s with %d rows",
        ticker,
        len(ticker_data),
    )
    return ticker_data
