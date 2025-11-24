"""Wikipedia-related functions for fetching and parsing S&P 500 tickers."""

from __future__ import annotations

import csv
from io import StringIO
import urllib.error
from urllib.parse import urlparse
import urllib.request

import pandas as pd

from src.constants import (
    SP500_TICKERS_FILE,
    SP500_WIKI_URL,
    WIKIPEDIA_REQUEST_TIMEOUT,
    WIKIPEDIA_USER_AGENT,
    WIKIPEDIA_SYMBOL_COLUMN,
    WIKIPEDIA_TICKER_COLUMN,
)
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)


def fetch_wikipedia_html() -> bytes:
    """Fetch the raw HTML content of the S&P 500 Wikipedia page."""
    parsed = urlparse(str(SP500_WIKI_URL))
    if parsed.scheme.lower() != "https":
        raise ValueError("SP500_WIKI_URL must use HTTPS")

    headers = {"User-Agent": WIKIPEDIA_USER_AGENT}
    req = urllib.request.Request(str(SP500_WIKI_URL), headers=headers)

    try:
        with urllib.request.urlopen(
            req, timeout=WIKIPEDIA_REQUEST_TIMEOUT
        ) as response:  # nosec B310
            return response.read()
    except urllib.error.URLError as exc:
        msg = f"Failed to fetch Wikipedia page: {exc}"
        raise RuntimeError(msg) from exc


def _read_html_tables(html_content: bytes) -> list[pd.DataFrame]:
    """Parse HTML tables from Wikipedia HTML content.

    Args:
        html_content: HTML content as bytes.

    Returns:
        List of DataFrames parsed from HTML.

    Raises:
        RuntimeError: If no tables can be parsed.
    """
    try:
        html_str = html_content.decode("utf-8", errors="ignore")
        tables = pd.read_html(StringIO(html_str))
    except (ValueError, ImportError) as exc:
        msg = f"Failed to parse HTML tables from Wikipedia: {exc}"
        raise RuntimeError(msg) from exc

    if not tables:
        raise RuntimeError("No tables found in Wikipedia HTML content")

    return tables


def _extract_sp500_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Extract the S&P 500 constituents table from a list of tables.

    Args:
        tables: List of DataFrames parsed from the Wikipedia page.

    Returns:
        The DataFrame corresponding to the S&P 500 constituents.

    Raises:
        RuntimeError: If no suitable table is found.
    """
    for table in tables:
        columns = {str(c).lower() for c in table.columns}
        if WIKIPEDIA_SYMBOL_COLUMN.lower() in columns or WIKIPEDIA_TICKER_COLUMN in columns:
            return table

    raise RuntimeError("S&P 500 constituents table not found in Wikipedia tables")


def _extract_tickers_from_table(sp500_table: pd.DataFrame) -> list[str]:
    """Extract ticker symbols from the S&P 500 table.

    Args:
        sp500_table: DataFrame containing the S&P 500 constituents.

    Returns:
        List of ticker symbols.

    Raises:
        KeyError: If 'Symbol' column is not present.
        RuntimeError: If no tickers are found.
    """
    if WIKIPEDIA_SYMBOL_COLUMN not in sp500_table.columns:
        msg = (
            f"'{WIKIPEDIA_SYMBOL_COLUMN}' column not found. "
            f"Available columns: {list(sp500_table.columns)}"
        )
        raise KeyError(msg)

    tickers = sp500_table[WIKIPEDIA_SYMBOL_COLUMN].tolist()
    if not tickers:
        raise RuntimeError(f"No tickers found in '{WIKIPEDIA_SYMBOL_COLUMN}' column")

    return tickers


def _normalize_tickers(tickers: list[str]) -> list[str]:
    """Normalize ticker symbols by replacing dots with dashes.

    Args:
        tickers: List of ticker symbols.

    Returns:
        List of normalized ticker symbols.
    """
    return [ticker.replace(".", "-") for ticker in tickers]


def parse_tickers_from_table(html_content: bytes) -> list[str]:
    """Full parsing pipeline from raw HTML to cleaned ticker list."""
    tables = _read_html_tables(html_content)
    sp500_table = _extract_sp500_table(tables)
    tickers = _extract_tickers_from_table(sp500_table)
    normalized = _normalize_tickers(tickers)
    unique_sorted = sorted(set(normalized))
    logger.info("Parsed %d unique tickers from Wikipedia", len(unique_sorted))
    return unique_sorted


def save_tickers_to_csv(tickers: list[str]) -> None:
    """Persist the list of tickers to CSV in the data directory."""
    ensure_output_dir(SP500_TICKERS_FILE)
    df = pd.DataFrame({WIKIPEDIA_TICKER_COLUMN: tickers})
    # Use safe CSV parameters for consistency across the project
    df.to_csv(
        SP500_TICKERS_FILE,
        index=False,
        sep=",",
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    logger.info("Saved %d tickers to %s", len(tickers), SP500_TICKERS_FILE)


def load_tickers() -> list[str]:
    """Load tickers from the persisted CSV file."""
    if not SP500_TICKERS_FILE.exists():
        msg = f"Tickers file not found: {SP500_TICKERS_FILE}. " "Run fetch_sp500_tickers() first."
        raise FileNotFoundError(msg)

    tickers_df = pd.read_csv(SP500_TICKERS_FILE)
    return tickers_df[WIKIPEDIA_TICKER_COLUMN].tolist()
