"""High level orchestration for S&P 500 ticker and price downloads."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from src.constants import DATA_FETCH_LOG_PROGRESS_INTERVAL
from src.data_fetching.download import download_ticker_data, get_date_range
from src.data_fetching.processing import process_ticker_data
from src.data_fetching.reporting import combine_and_save_data
from src.data_fetching.wikipedia import (
    fetch_wikipedia_html,
    load_tickers,
    parse_tickers_from_table,
    save_tickers_to_csv,
)
from src.path import WEIGHTED_LOG_RETURNS_FILE
from src.utils import get_logger, save_parquet_and_csv, extract_date_range

logger = get_logger(__name__)


def fetch_sp500_tickers() -> None:
    """Fetch S&P 500 tickers from Wikipedia and persist them to CSV.

    The low-level HTTP, HTML parsing and normalization logic is delegated to
    the wikipedia.py submodule. This function only orchestrates the workflow.
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia")

    html_content = fetch_wikipedia_html()
    tickers: list[str] = parse_tickers_from_table(html_content)
    save_tickers_to_csv(tickers)

    logger.info("Retrieved %d tickers from Wikipedia", len(tickers))


def download_sp500_data() -> None:
    """Download and persist historical S&P 500 data via yfinance.

    All validation, IO details and edge cases live in the dedicated
    submodules (download.py, processing.py, reporting.py). This orchestration
    layer remains intentionally thin to ease auditing and testing.
    """
    tickers = load_tickers()
    start_date, end_date = get_date_range()

    logger.info(
        "Downloading historical data for %d tickers between %s and %s",
        len(tickers),
        start_date.date(),
        end_date.date(),
    )

    data_list: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        ticker_data = download_ticker_data(ticker, start_date, end_date)

        if ticker_data is None:
            failed_tickers.append(ticker)
            logger.warning("No data returned for ticker %s", ticker)
        else:
            processed_data = process_ticker_data(ticker_data, ticker)
            data_list.append(processed_data)

        if i % DATA_FETCH_LOG_PROGRESS_INTERVAL == 0 or i == total:
            logger.info("Progress: %d/%d tickers processed", i, total)

    combine_and_save_data(data_list, failed_tickers)


def _download_sp500_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Download S&P 500 index data from yfinance.

    Args:
        start_date: Start date for data fetching.
        end_date: End date for data fetching.

    Returns:
        DataFrame with S&P 500 data.

    Raises:
        RuntimeError: If download fails.
        ValueError: If data is empty.
    """
    logger.info("Fetching S&P 500 index (^GSPC) from %s to %s", start_date, end_date)

    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)

    if sp500 is None:
        msg = "Failed to download S&P 500 data - yfinance returned None"
        raise RuntimeError(msg)

    if sp500.empty:
        msg = "Downloaded S&P 500 data is empty"
        raise ValueError(msg)

    return sp500


def _process_sp500_columns(sp500: pd.DataFrame) -> pd.DataFrame:
    """Process S&P 500 DataFrame columns to match expected format.

    Args:
        sp500: Raw S&P 500 DataFrame from yfinance.

    Returns:
        Processed DataFrame with standardized columns.

    Raises:
        ValueError: If required columns are missing.
    """
    # Handle MultiIndex columns from yfinance
    if isinstance(sp500.columns, pd.MultiIndex):
        # Flatten MultiIndex: take only first level (column names)
        sp500.columns = sp500.columns.get_level_values(0)

    # Reset index to get date as column
    sp500 = sp500.reset_index()

    # Rename columns to match expected format (lowercase)
    sp500.columns = [
        col.lower() if isinstance(col, str) else str(col).lower() for col in sp500.columns
    ]

    # Ensure we have required columns
    if "close" not in sp500.columns or "date" not in sp500.columns:
        msg = f"Missing required columns. Available columns: {list(sp500.columns)}"
        raise ValueError(msg)

    return sp500


def _compute_log_returns(sp500: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for S&P 500 data.

    Args:
        sp500: Processed S&P 500 DataFrame with 'close' column.

    Returns:
        DataFrame with computed log returns.
    """
    logger.info("Computing log returns for S&P 500 index")
    sp500["log_return"] = pd.NA
    sp500.loc[1:, "log_return"] = np.log(sp500["close"] / sp500["close"].shift(1))

    # Remove first row with NaN log_return
    sp500 = sp500.dropna(subset=["log_return"]).reset_index(drop=True)

    # Rename log_return to weighted_log_return for compatibility
    sp500 = sp500.rename(columns={"log_return": "weighted_log_return"})

    return sp500


def _create_result_dataframe(sp500: pd.DataFrame) -> pd.DataFrame:
    """Create final result DataFrame with required columns.

    Args:
        sp500: DataFrame with computed log returns.

    Returns:
        Final DataFrame with date, weighted_log_return, and weighted_closing.
    """
    return pd.DataFrame(
        {
            "date": sp500["date"],
            "weighted_log_return": sp500["weighted_log_return"],
            "weighted_closing": sp500["close"],
        }
    )


def fetch_sp500_index_and_compute_log_returns(
    output_file: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Fetch S&P 500 index from yfinance and compute log returns.

    Downloads the S&P 500 index (^GSPC) historical data, computes log returns,
    and saves to weighted_log_returns file for ARIMA/GARCH pipeline.

    Args:
        output_file: Path to save the weighted log returns CSV. If None, uses default.
        start_date: Start date for data fetching (YYYY-MM-DD). If None, uses get_date_range().
        end_date: End date for data fetching (YYYY-MM-DD). If None, uses get_date_range().

    Raises:
        ValueError: If data is empty or missing required columns.
        RuntimeError: If yfinance download fails.
    """
    if output_file is None:
        output_file = str(WEIGHTED_LOG_RETURNS_FILE)

    # Use same date range as ticker data
    if start_date is None or end_date is None:
        start_dt, end_dt = get_date_range()
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

    try:
        # Download and process S&P 500 data
        sp500_raw = _download_sp500_data(start_date, end_date)
        sp500_processed = _process_sp500_columns(sp500_raw)
        sp500_with_returns = _compute_log_returns(sp500_processed)
        result_df = _create_result_dataframe(sp500_with_returns)

        # Save to file
        logger.info("Saving S&P 500 log returns to %s", output_file)
        save_parquet_and_csv(result_df, output_file)

        start_date_str, end_date_str = extract_date_range(
            result_df, date_col="date", as_string=True
        )
        logger.info(
            "S&P 500 index fetching complete: %d dates, period %s → %s",
            len(result_df),
            start_date_str,
            end_date_str,
        )

    except Exception as e:
        logger.error("Failed to fetch S&P 500 index: %s", e)
        raise RuntimeError(f"Failed to fetch S&P 500 index: {e}") from e
