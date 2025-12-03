"""Trade fetching and processing utilities."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.constants import (
    END_DATE,
    EXCHANGE_ID,
    FETCHING_MAX_TRADES_PER_CALL,
    START_DATE,
    SYMBOL,
)
from src.data_fetching.exchange import ccxt
from src.data_fetching.rate_limiter import rate_limiter
from src.utils import get_logger

if TYPE_CHECKING:
    import ccxt as ccxt_module

logger = get_logger(__name__)


def get_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start/end datetimes for configured date range.

    Returns:
        Tuple of (start_dt, end_dt) where end is exclusive.
    """
    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt = pd.to_datetime(END_DATE, utc=True)
    return start_dt, end_dt


def filter_date_range(
    df: pd.DataFrame,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Limit trades to date range.

    Args:
        df: DataFrame with trades.
        start_dt: Start datetime (inclusive). Uses constants if None.
        end_dt: End datetime (exclusive). Uses constants if None.

    Returns:
        Filtered DataFrame.
    """
    if df.empty or "timestamp" not in df.columns:
        return df

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", errors="coerce", utc=True
        )
    df = df.dropna(subset=["timestamp"])

    if start_dt is None or end_dt is None:
        start_dt, end_dt = get_date_bounds()

    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
    return df.loc[mask].reset_index(drop=True)


def optimize_trades_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Drop heavyweight columns and downcast to trim memory usage.

    Args:
        df: DataFrame with trades.

    Returns:
        Memory-optimized DataFrame.
    """
    if df.empty:
        return df

    drop_cols = [col for col in ("info", "datetime") if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", errors="coerce", utc=True
        )
    df = df.dropna(subset=["timestamp"]) if "timestamp" in df.columns else df

    if "amount" in df.columns:
        df["amount"] = np.asarray(pd.to_numeric(df["amount"], errors="coerce"), dtype=np.float32)
    if "price" in df.columns:
        df["price"] = np.asarray(pd.to_numeric(df["price"], errors="coerce"), dtype=np.float32)
    if "side" in df.columns and df["side"].dtype == object:
        df["side"] = df["side"].astype("category")

    return df


def deduplicate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate trades within a single chunk.

    Args:
        df: DataFrame with trades.

    Returns:
        Deduplicated DataFrame.
    """
    subset_cols = [col for col in ("id", "timestamp") if col in df.columns]
    if not subset_cols:
        return df

    before = len(df)
    deduped = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    removed = before - len(deduped)
    if removed > 0:
        logger.info("Removed %d duplicate trades in chunk", removed)
    return deduped


def compute_time_ranges(
    start_ts: int, end_ts: int, max_workers: int
) -> list[tuple[str, int, int, int]]:
    """Compute non-overlapping time ranges for parallel workers.

    Args:
        start_ts: Start timestamp in milliseconds.
        end_ts: End timestamp in milliseconds.
        max_workers: Number of parallel workers.

    Returns:
        List of tuples (exchange_id, start_ts, end_ts, worker_id).
    """
    total_range = max(1, end_ts - start_ts)
    slice_size = max(1, total_range // max_workers)

    time_ranges: list[tuple[str, int, int, int]] = []
    worker_start = start_ts
    worker_id = 0

    while worker_start < end_ts:
        worker_end = min(worker_start + slice_size, end_ts)
        time_ranges.append((EXCHANGE_ID, worker_start, worker_end, worker_id))
        worker_start = worker_end
        worker_id += 1

    return time_ranges


def concat_and_sort_trades(
    dfs: Sequence[pd.DataFrame],
) -> list[dict[str, object]]:
    """Concatenate DataFrames and sort by timestamp.

    Args:
        dfs: Sequence of DataFrames to concatenate.

    Returns:
        List of trade dictionaries sorted by timestamp.
    """
    try:
        final_df = pd.concat(list(dfs), ignore_index=True)
        if "timestamp" in final_df.columns:
            final_df = final_df.sort_values("timestamp")
        logger.info("Parallel fetch completed: total %d trades collected", len(final_df))
        return final_df.to_dict("records")
    except Exception as e:
        logger.error("Error concatenating trade chunks: %s", e)
        return []


def fetch_trades_for_time_range(
    args: tuple[str, int, int, int],
) -> list[dict[str, object]]:
    """Fetch trades for a specific time range in a separate thread.

    Args:
        args: Tuple of (exchange_class_name, start_ts, end_ts, worker_id).

    Returns:
        List of trade dictionaries.
    """
    exchange_class_name, start_ts, end_ts, worker_id = args

    try:
        local_ccxt = ccxt
        if local_ccxt is None:
            import importlib

            local_ccxt = importlib.import_module("ccxt")
        if not hasattr(local_ccxt, exchange_class_name):
            msg = f"Exchange {exchange_class_name} not found in ccxt"
            raise ValueError(msg)
        exchange = getattr(local_ccxt, exchange_class_name)()
    except Exception as e:
        logger.error("Worker %d: Failed to create exchange instance: %s", worker_id, e)
        return []

    logger.info(
        "Worker %d: Fetching trades from %s to %s",
        worker_id,
        pd.to_datetime(start_ts, unit="ms", utc=True),
        pd.to_datetime(end_ts, unit="ms", utc=True),
    )

    try:
        return fetch_all_trades_in_range(exchange, start_ts, end_ts)
    except Exception as e:
        logger.error("Worker %d: Error fetching trades: %s", worker_id, e)
        return []


def fetch_all_trades_parallel(
    start_ts: int, end_ts: int, max_workers: int = 6
) -> list[dict[str, object]]:
    """Fetch all trades within timestamp range using parallel workers.

    Divides the time range into chunks and processes them in parallel.

    Args:
        start_ts: Start timestamp in milliseconds.
        end_ts: End timestamp in milliseconds.
        max_workers: Number of parallel workers.

    Returns:
        List of trade dictionaries sorted by timestamp.
    """
    time_ranges = compute_time_ranges(start_ts, end_ts, max_workers)

    logger.info(
        "Starting parallel fetch with %d workers for time range %s to %s",
        len(time_ranges),
        pd.to_datetime(start_ts, unit="ms", utc=True),
        pd.to_datetime(end_ts, unit="ms", utc=True),
    )

    all_trades_dfs: list[pd.DataFrame] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_worker = {
            executor.submit(fetch_trades_for_time_range, time_range): time_range[3]
            for time_range in time_ranges
        }

        for future in concurrent.futures.as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                trades = future.result()
                if trades:
                    df_chunk = optimize_trades_memory(pd.DataFrame(trades))
                    if not df_chunk.empty:
                        all_trades_dfs.append(df_chunk)
                        logger.info(
                            "Worker %d completed: fetched %d trades",
                            worker_id,
                            len(df_chunk),
                        )
                    else:
                        logger.info("Worker %d completed: 0 trades", worker_id)
            except Exception as e:
                logger.error("Worker %d failed: %s", worker_id, e)

    if not all_trades_dfs:
        logger.warning("No trades collected from any worker")
        return []

    return concat_and_sort_trades(all_trades_dfs)


def fetch_all_trades_in_range(
    exchange: ccxt_module.Exchange, start_ts: int, end_ts: int
) -> list[dict[str, object]]:
    """Fetch all trades within timestamp range using iterative API calls.

    Args:
        exchange: ccxt exchange instance.
        start_ts: Start timestamp in milliseconds.
        end_ts: End timestamp in milliseconds.

    Returns:
        List of trade dictionaries.
    """
    start_dt = pd.to_datetime(start_ts, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="ms", utc=True)
    all_trades_dfs: list[pd.DataFrame] = []
    current_since = start_ts
    total_trades_count = 0

    logger.info("Fetching trades from %s to %s", start_dt, end_dt)

    while current_since < end_ts:
        batch_result = _fetch_single_batch(
            exchange, current_since, start_dt, end_dt, total_trades_count
        )
        if batch_result is None:
            break

        valid_trades, new_since, batch_count = batch_result
        all_trades_dfs.append(valid_trades)
        total_trades_count += batch_count
        current_since = new_since

        if current_since >= end_ts:
            logger.info("Reached end timestamp, stopping iteration")
            break

    if not all_trades_dfs:
        return []

    return concat_and_sort_trades(all_trades_dfs)


def _fetch_single_batch(
    exchange: ccxt_module.Exchange,
    current_since: int,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    total_trades_count: int,
) -> tuple[pd.DataFrame, int, int] | None:
    """Fetch a single batch of trades from the exchange.

    Args:
        exchange: ccxt exchange instance.
        current_since: Timestamp to fetch from (milliseconds).
        start_dt: Start datetime for filtering.
        end_dt: End datetime for filtering.
        total_trades_count: Running total of trades fetched.

    Returns:
        Tuple of (valid_trades_df, next_since_ts, batch_count) or None if done.
    """
    try:
        rate_limiter.wait_if_needed()
        trades = exchange.fetch_trades(
            SYMBOL, since=current_since, limit=FETCHING_MAX_TRADES_PER_CALL
        )

        if not trades:
            logger.info("No more trades returned, stopping iteration")
            return None

        temp_df = optimize_trades_memory(pd.DataFrame(trades))

        if "timestamp" not in temp_df.columns or temp_df.empty:
            logger.warning("No timestamp column in trades data, skipping batch")
            return None

        mask = (temp_df["timestamp"] >= start_dt) & (temp_df["timestamp"] < end_dt)
        valid_trades = temp_df.loc[mask].copy()

        if valid_trades.empty:
            logger.info("No valid trades in current batch, stopping iteration")
            return None

        last_timestamp = valid_trades["timestamp"].max()
        next_since = int(last_timestamp.timestamp() * 1000) + 1
        batch_count = len(valid_trades)

        _log_batch_progress(total_trades_count, batch_count, last_timestamp, len(trades))

        if len(trades) < FETCHING_MAX_TRADES_PER_CALL:
            logger.info(
                "Received less than max trades per call (%d < %d), likely reached end",
                len(trades),
                FETCHING_MAX_TRADES_PER_CALL,
            )

        return valid_trades, next_since, batch_count

    except Exception as e:
        logger.error("Error fetching trades batch: %s", e)
        return None


def _log_batch_progress(
    total_count: int, batch_count: int, last_timestamp: pd.Timestamp, raw_count: int
) -> None:
    """Log progress information for a batch fetch.

    Args:
        total_count: Running total of trades fetched.
        batch_count: Number of valid trades in this batch.
        last_timestamp: Timestamp of the last trade in the batch.
        raw_count: Raw number of trades returned by API.
    """
    last_trade_datetime = pd.to_datetime(last_timestamp, unit="ns", utc=True)
    logger.info(
        "Fetched %d trades so far (current batch: %d valid) - Last trade: %s",
        total_count + batch_count,
        batch_count,
        last_trade_datetime.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


__all__ = [
    "compute_time_ranges",
    "concat_and_sort_trades",
    "deduplicate_chunk",
    "fetch_all_trades_in_range",
    "fetch_all_trades_parallel",
    "fetch_trades_for_time_range",
    "filter_date_range",
    "get_date_bounds",
    "optimize_trades_memory",
]
