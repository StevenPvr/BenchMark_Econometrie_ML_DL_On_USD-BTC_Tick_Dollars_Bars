"""ccxt-based tick downloader for crypto symbols.

Downloads trades in chunks, saves to Parquet dataset, and supports resume.
This module orchestrates the download process using submodules for:
- rate_limiter: API rate limiting
- exchange: Exchange connection
- trades: Trade fetching and processing
- storage: Parquet storage
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd

from src.constants import FETCHING_CHUNK_DAYS
from src.data_fetching.exchange import build_exchange, validate_symbol
from src.data_fetching.rate_limiter import RateLimiter, rate_limiter
from src.data_fetching.storage import append_trades_to_dataset
from src.data_fetching.trades import (
    deduplicate_chunk,
    fetch_all_trades_in_range,
    fetch_all_trades_parallel,
    filter_date_range,
    get_date_bounds,
    optimize_trades_memory,
)
from src.utils import get_logger

if TYPE_CHECKING:
    import ccxt as ccxt_module

logger = get_logger(__name__)


def download_ticks_in_date_range(
    parallel: bool = True,
    max_workers: int = 6,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Download trades for SYMBOL from EXCHANGE_ID for the specified date range.

    Downloads data in chunks and appends to existing dataset without loading
    the full parquet file in memory.

    Args:
        parallel: Whether to use parallel fetching within each chunk.
        max_workers: Number of parallel workers per chunk.
        start_date: Start date in YYYY-MM-DD format. Defaults to START_DATE.
        end_date: End date in YYYY-MM-DD format. Defaults to END_DATE.
    """
    if start_date is None or end_date is None:
        start_dt, end_dt = get_date_bounds()
    else:
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    exchange = build_exchange()
    validate_symbol(exchange)

    total_new_trades = _fetch_and_save_chunks(
        start_ts, end_ts, start_dt, end_dt, parallel, max_workers, exchange
    )

    if total_new_trades == 0:
        logger.warning("No trades returned for this date range")
        return

    logger.info("New trades fetched: %d", total_new_trades)


def _fetch_and_save_chunks(
    start_ts: int,
    end_ts: int,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    parallel: bool,
    max_workers: int,
    exchange: ccxt_module.Exchange,
) -> int:
    """Fetch trades in chunks and save to dataset.

    Args:
        start_ts: Start timestamp in milliseconds.
        end_ts: End timestamp in milliseconds.
        start_dt: Start datetime.
        end_dt: End datetime.
        parallel: Whether to use parallel fetching.
        max_workers: Number of parallel workers.
        exchange: ccxt exchange instance.

    Returns:
        Total number of trades fetched.
    """
    total_new_trades = 0
    chunk_ms = int(timedelta(days=FETCHING_CHUNK_DAYS).total_seconds() * 1000)
    chunk_start = start_ts

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + chunk_ms, end_ts)
        logger.info(
            "Chunk fetch: %s to %s",
            pd.to_datetime(chunk_start, unit="ms", utc=True),
            pd.to_datetime(chunk_end, unit="ms", utc=True),
        )

        if parallel and max_workers > 1:
            trades = fetch_all_trades_parallel(chunk_start, chunk_end, max_workers)
        else:
            trades = fetch_all_trades_in_range(exchange, chunk_start, chunk_end)

        if not trades:
            logger.info("Chunk returned no trades, moving to next window")
            chunk_start = chunk_end + 1
            continue

        new_df = _process_chunk_trades(trades, start_dt, end_dt)
        if new_df is None or new_df.empty:
            chunk_start = chunk_end + 1
            continue

        append_trades_to_dataset(new_df)
        total_new_trades += len(new_df)
        chunk_start = chunk_end

    return total_new_trades


def _process_chunk_trades(
    trades: list[dict[str, object]],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame | None:
    """Process and filter trades from a chunk.

    Args:
        trades: List of trade dictionaries.
        start_dt: Start datetime for filtering.
        end_dt: End datetime for filtering.

    Returns:
        Processed DataFrame or None if empty.
    """
    new_df = pd.DataFrame(trades)
    if new_df.empty:
        logger.warning("Chunk DataFrame empty after normalization")
        return None

    new_df = filter_date_range(new_df, start_dt, end_dt)
    new_df = optimize_trades_memory(new_df)
    if new_df.empty:
        logger.warning("No trades within date range after filtering")
        return None

    new_df = deduplicate_chunk(new_df)
    if "timestamp" in new_df.columns:
        new_df = new_df.sort_values("timestamp").reset_index(drop=True)

    return new_df


# Re-export for backwards compatibility
__all__ = [
    "RateLimiter",
    "download_ticks_in_date_range",
    "rate_limiter",
]
