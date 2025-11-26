"""ccxt-based tick downloader for crypto symbols.

Implementation that:
- validates exchange/symbol availability,
- downloads trades in 2-week chunks for robustness,
- saves incrementally to CSV/Parquet after each chunk,
- supports resume from existing data.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque

import pandas as pd  # type: ignore[import-untyped]

# Optional dependency; tests monkeypatch this attribute directly
try:  # pragma: no cover - best-effort import
    import ccxt as _ccxt  # type: ignore
except Exception:  # pragma: no cover - missing ccxt is acceptable in tests
    _ccxt = None
ccxt = _ccxt

from src.constants import END_DATE, EXCHANGE_ID, START_DATE, SYMBOL
from src.path import DATASET_RAW_CSV, DATASET_RAW_PARQUET
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)

# Fetching configuration
MAX_TRADES_PER_CALL: int = 1000  # ccxt typically limits to 1000 trades per call

# Chunk configuration: 2 weeks per chunk for incremental saving
CHUNK_DAYS: int = 14  # 2 weeks per chunk

# Rate limiting configuration for Binance
MAX_REQUEST_WEIGHT_PER_MINUTE: int = 6000  # Binance limit: 6000 weight per minute
REQUEST_WEIGHT_PER_CALL: int = 1  # Each fetch_trades call costs 1 weight


class RateLimiter:
    """Thread-safe rate limiter for API calls with weight-based limiting.

    This limiter is shared across ALL threads/workers to enforce a global
    rate limit of 6000 API calls per minute across the entire application.
    """

    def __init__(self, max_weight_per_minute: int, weight_per_call: int):
        self.max_weight_per_minute = max_weight_per_minute
        self.weight_per_call = weight_per_call
        self.call_times: Deque[float] = deque()  # Store timestamps of API calls
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits.

        Thread-safe: uses a lock to ensure accurate tracking across all workers.
        """
        while True:
            with self.lock:
                current_time = time.time()

                # Remove calls older than 1 minute
                while self.call_times and current_time - self.call_times[0] > 60:
                    self.call_times.popleft()

                # Calculate current weight used in the last minute
                current_weight = len(self.call_times) * self.weight_per_call

                # If under the limit, record call and proceed
                if current_weight < self.max_weight_per_minute:
                    self.call_times.append(current_time)
                    return

                # Calculate wait time
                if self.call_times:
                    wait_time = 60 - (current_time - self.call_times[0]) + 0.1
                else:
                    wait_time = 0.1

            # Sleep outside the lock
            if wait_time > 0:
                logger.info("Rate limit reached (%d/%d), waiting %.2f seconds",
                           current_weight, self.max_weight_per_minute, wait_time)
                time.sleep(wait_time)


# Global rate limiter instance shared across ALL workers
rate_limiter = RateLimiter(MAX_REQUEST_WEIGHT_PER_MINUTE, REQUEST_WEIGHT_PER_CALL)


def _build_exchange() -> Any:
    """Create ccxt exchange instance using EXCHANGE_ID."""
    global ccxt
    if ccxt is None:
        import importlib
        ccxt = importlib.import_module("ccxt")
    if not hasattr(ccxt, EXCHANGE_ID):
        raise ValueError(f"Exchange {EXCHANGE_ID} not found in ccxt")
    return getattr(ccxt, EXCHANGE_ID)()


def _validate_symbol(exchange: Any) -> None:
    """Ensure requested symbol exists and is active."""
    markets = exchange.load_markets()
    if SYMBOL not in markets:
        raise ValueError(f"Symbole {SYMBOL} non disponible sur {EXCHANGE_ID}")
    market_info = markets.get(SYMBOL, {})
    if isinstance(market_info, dict) and not market_info.get("active", True):
        raise ValueError(f"Symbole {SYMBOL} inactif sur {EXCHANGE_ID}")


def _compute_since_timestamp() -> int:
    """Compute 'since' timestamp in milliseconds.

    Uses START_DATE when provided; falls back to one-week lookback.
    """
    try:
        start_dt = pd.to_datetime(START_DATE)
    except Exception:
        start_dt = datetime.utcnow() - timedelta(days=7)
    return int(start_dt.timestamp() * 1000)


def _fetch_trades_for_time_range(args: tuple[Any, int, int, int]) -> list[dict]:
    """Fetch trades for a specific time range in a separate thread.

    Args:
        args: Tuple of (exchange_class_name, start_ts, end_ts, worker_id)
    """
    exchange_class_name, start_ts, end_ts, worker_id = args

    # Create a new exchange instance for this thread
    try:
        global ccxt
        if ccxt is None:
            import importlib
            ccxt = importlib.import_module("ccxt")
        if not hasattr(ccxt, exchange_class_name):
            raise ValueError(f"Exchange {exchange_class_name} not found in ccxt")
        exchange = getattr(ccxt, exchange_class_name)()
    except Exception as e:
        logger.error("Worker %d: Failed to create exchange instance: %s", worker_id, e)
        return []

    logger.info("Worker %d: Fetching trades from %s to %s",
               worker_id,
               pd.to_datetime(start_ts, unit='ms', utc=True),
               pd.to_datetime(end_ts, unit='ms', utc=True))

    try:
        # Use the global rate limiter shared across all workers
        # This ensures the 6000 calls/minute limit is respected globally
        return _fetch_all_trades_in_range(exchange, start_ts, end_ts)
    except Exception as e:
        logger.error("Worker %d: Error fetching trades: %s", worker_id, e)
        return []


def _fetch_all_trades_parallel(start_ts: int, end_ts: int, max_workers: int = 6) -> list[dict]:
    """Fetch all trades within timestamp range using parallel workers.

    Divides the time range into chunks and processes them in parallel.
    Returns a list of dictionaries (records) to maintain backward compatibility,
    but internally uses DataFrames for memory efficiency during collection.
    """
    # Calculate time range per worker
    total_range = end_ts - start_ts
    range_per_worker = total_range // max_workers

    # Create time ranges for each worker
    time_ranges = []
    for i in range(max_workers):
        worker_start = start_ts + (i * range_per_worker)
        worker_end = start_ts + ((i + 1) * range_per_worker) if i < max_workers - 1 else end_ts
        time_ranges.append((EXCHANGE_ID, worker_start, worker_end, i))

    logger.info("Starting parallel fetch with %d workers for time range %s to %s",
               max_workers,
               pd.to_datetime(start_ts, unit='ms', utc=True),
               pd.to_datetime(end_ts, unit='ms', utc=True))

    # Use ThreadPoolExecutor for parallel processing
    all_trades_dfs: list[pd.DataFrame] = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_worker = {executor.submit(_fetch_trades_for_time_range, time_range): time_range[3]
                           for time_range in time_ranges}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                trades = future.result()
                if trades:
                    # Convert to DataFrame immediately if not already
                    df_chunk = pd.DataFrame(trades)
                    if not df_chunk.empty:
                        all_trades_dfs.append(df_chunk)
                        logger.info("Worker %d completed: fetched %d trades", worker_id, len(df_chunk))
                    else:
                        logger.info("Worker %d completed: 0 trades", worker_id)
            except Exception as e:
                logger.error("Worker %d failed: %s", worker_id, e)

    if not all_trades_dfs:
        logger.warning("No trades collected from any worker")
        return []

    # Concatenate all DataFrames
    try:
        final_df = pd.concat(all_trades_dfs, ignore_index=True)
        
        # Sort trades by timestamp to ensure chronological order
        if "timestamp" in final_df.columns:
            final_df = final_df.sort_values("timestamp")
            
        logger.info("Parallel fetch completed: total %d trades collected", len(final_df))
        return final_df.to_dict('records')
    except Exception as e:
        logger.error("Error concatenating trade chunks: %s", e)
        return []


def _fetch_all_trades_in_range(exchange: Any, start_ts: int, end_ts: int) -> list[dict]:
    """Fetch all trades within timestamp range using iterative API calls.

    Since ccxt limits results per call, we iterate until we get all data.
    """
    all_trades_dfs: list[pd.DataFrame] = []
    current_since = start_ts
    total_trades_count = 0

    logger.info("Fetching trades from %s to %s", pd.to_datetime(start_ts, unit='ms', utc=True),
                pd.to_datetime(end_ts, unit='ms', utc=True))

    while current_since < end_ts:
        try:
            # Respect rate limits
            rate_limiter.wait_if_needed()

            trades = exchange.fetch_trades(SYMBOL, since=current_since, limit=MAX_TRADES_PER_CALL)

            if not trades:
                logger.info("No more trades returned, stopping iteration")
                break

            # Convert to DataFrame immediately
            temp_df = pd.DataFrame(trades)
            
            if "timestamp" in temp_df.columns:
                temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], unit="ms", utc=True, errors="coerce")
                temp_df = temp_df.dropna(subset=["timestamp"])

                # Filter trades that are within our range
                mask = (temp_df["timestamp"] >= pd.to_datetime(start_ts, unit='ms', utc=True)) & \
                       (temp_df["timestamp"] <= pd.to_datetime(end_ts, unit='ms', utc=True))
                valid_trades = temp_df[mask].copy()  # Ensure we have a DataFrame

                if valid_trades.empty:
                    logger.info("No valid trades in current batch, stopping iteration")
                    break

                # Add valid trades to our collection as DataFrame
                all_trades_dfs.append(valid_trades)
                total_trades_count += len(valid_trades)

                # Update current_since to the timestamp of the last trade + 1ms
                last_timestamp = valid_trades["timestamp"].max()
                current_since = int(last_timestamp.timestamp() * 1000) + 1

                # Log batch information with timestamp
                last_trade_datetime = pd.to_datetime(last_timestamp, unit='ns', utc=True)
                logger.info("Fetched %d trades so far (current batch: %d valid) - Last trade: %s",
                           total_trades_count, len(valid_trades), last_trade_datetime.strftime('%Y-%m-%d %H:%M:%S UTC'))

                # Safety check: if we got less than MAX_TRADES_PER_CALL, we're probably at the end
                if len(trades) < MAX_TRADES_PER_CALL:
                    logger.info("Received less than max trades per call (%d < %d), likely reached end of data", len(trades), MAX_TRADES_PER_CALL)
                    break

                # Check if we're still within our date range
                if current_since >= end_ts:
                    logger.info("Reached end timestamp, stopping iteration")
                    break

            else:
                logger.warning("No timestamp column in trades data, skipping batch")
                break

        except Exception as e:
            logger.error("Error fetching trades batch: %s", e)
            break

    if not all_trades_dfs:
        return []
        
    # Concatenate and return as list of dicts for compatibility
    try:
        final_df = pd.concat(all_trades_dfs, ignore_index=True)
        logger.info("Total trades collected: %d", len(final_df))
        return final_df.to_dict('records')
    except Exception as e:
        logger.error("Error concatenating trade chunks: %s", e)
        return []


def _filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    """Limit trades to START_DATE..END_DATE if END_DATE is provided."""
    if df.empty or "timestamp" not in df.columns:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt = pd.to_datetime(END_DATE, utc=True)
    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
    return df.loc[mask].reset_index(drop=True)


def _persist_trades(df: pd.DataFrame) -> None:
    """Write cleaned trades to CSV and Parquet outputs (overwrite mode)."""
    ensure_output_dir(DATASET_RAW_PARQUET)
    df.to_parquet(DATASET_RAW_PARQUET, index=False)
    df.to_csv(DATASET_RAW_CSV, index=False, lineterminator="\n")
    logger.info("Saved %d trades to %s / %s", len(df), DATASET_RAW_PARQUET, DATASET_RAW_CSV)


def _load_existing_trades() -> pd.DataFrame:
    """Load existing trades from parquet file if it exists."""
    if Path(DATASET_RAW_PARQUET).exists():
        try:
            df = pd.read_parquet(DATASET_RAW_PARQUET)
            logger.info("Loaded %d existing trades from %s", len(df), DATASET_RAW_PARQUET)
            return df
        except Exception as e:
            logger.warning("Could not load existing trades: %s", e)
    return pd.DataFrame()


def _append_and_save_trades(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Append new trades to existing and save incrementally.

    Returns the combined DataFrame.
    """
    if existing_df.empty:
        combined_df = new_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Deduplicate
    subset_cols = [col for col in ("id", "timestamp") if col in combined_df.columns]
    if subset_cols:
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
        if len(combined_df) < before:
            logger.info("Removed %d duplicate trades", before - len(combined_df))

    # Sort by timestamp
    if "timestamp" in combined_df.columns:
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    # Save immediately
    _persist_trades(combined_df)

    return combined_df


def _get_last_timestamp(df: pd.DataFrame) -> int | None:
    """Get the last timestamp from existing data in milliseconds."""
    if df.empty or "timestamp" not in df.columns:
        return None

    last_ts = df["timestamp"].max()
    # Ensure we have a scalar value for the NaN check
    if isinstance(last_ts, pd.Series):
        last_ts = last_ts.iloc[0] if not last_ts.empty else None

    # Check if the scalar value is NaN
    if last_ts is None or pd.isna(last_ts):
        return None

    # Convert to milliseconds if it's a datetime
    if isinstance(last_ts, pd.Timestamp):
        return int(last_ts.timestamp() * 1000)
    return int(last_ts)


def _generate_chunk_ranges(start_ts: int, end_ts: int, chunk_days: int = CHUNK_DAYS) -> list[tuple[int, int]]:
    """Generate list of (start_ts, end_ts) tuples for each chunk.

    Each chunk covers `chunk_days` days (default 14 days = 2 weeks).
    """
    chunk_ms = chunk_days * 24 * 60 * 60 * 1000  # Convert days to milliseconds
    chunks = []

    current_start = start_ts
    while current_start < end_ts:
        current_end = min(current_start + chunk_ms, end_ts)
        chunks.append((current_start, current_end))
        current_start = current_end

    logger.info("Generated %d chunks of %d days each", len(chunks), chunk_days)
    return chunks


def download_ticks_in_date_range(parallel: bool = True, max_workers: int = 12) -> None:
    """Download trades for SYMBOL from EXCHANGE_ID for the date range in constants.

    Downloads data for the date range specified in START_DATE/END_DATE and appends
    to existing data. Data is merged chronologically and deduplicated.

    This allows you to:
    1. Change START_DATE/END_DATE in constants.py to a new week
    2. Run the script to fetch that week's data
    3. Data is automatically appended to existing file (no overwrite)

    Args:
        parallel: Whether to use parallel fetching within each chunk (default: True)
        max_workers: Number of parallel workers per chunk (default: 6)
    """
    # Compute timestamp range using START_DATE and END_DATE from constants
    start_ts = int(pd.to_datetime(START_DATE, utc=True).timestamp() * 1000)
    end_ts = int(pd.to_datetime(END_DATE, utc=True).timestamp() * 1000)

    # Validate exchange first
    exchange = _build_exchange()
    _validate_symbol(exchange)

    # Load existing trades if any (for append capability)
    existing_df = _load_existing_trades()

    # Always use the dates from constants.py (allows fetching any date range)
    actual_start_ts = start_ts
    logger.info(
        "Fetching date range from constants: %s to %s",
        pd.to_datetime(start_ts, unit="ms", utc=True).strftime("%Y-%m-%d"),
        pd.to_datetime(end_ts, unit="ms", utc=True).strftime("%Y-%m-%d"),
    )

    # Check if we already have data for this exact range (skip if complete)
    if not existing_df.empty and "timestamp" in existing_df.columns:
        existing_start = existing_df["timestamp"].min()
        existing_end = existing_df["timestamp"].max()

        # Convert to ms if needed
        if isinstance(existing_start, pd.Timestamp):
            existing_start_ts = int(existing_start.timestamp() * 1000)
            existing_end_ts = int(existing_end.timestamp() * 1000)
        else:
            existing_start_ts = int(existing_start)
            existing_end_ts = int(existing_end)

        # Check if requested range is fully within existing data
        if existing_start_ts <= start_ts and existing_end_ts >= end_ts:
            logger.info("=" * 60)
            logger.info("DATE RANGE ALREADY DOWNLOADED")
            logger.info("Requested: %s to %s", START_DATE, END_DATE)
            logger.info("Existing data covers: %s to %s",
                       pd.to_datetime(existing_start_ts, unit="ms", utc=True).strftime("%Y-%m-%d"),
                       pd.to_datetime(existing_end_ts, unit="ms", utc=True).strftime("%Y-%m-%d"))
            logger.info("Total trades: %d", len(existing_df))
            logger.info("=" * 60)
            logger.info("To force re-download, delete the existing data file first.")
            return

        logger.info(
            "Existing data: %s to %s (%d trades)",
            pd.to_datetime(existing_start_ts, unit="ms", utc=True).strftime("%Y-%m-%d"),
            pd.to_datetime(existing_end_ts, unit="ms", utc=True).strftime("%Y-%m-%d"),
            len(existing_df),
        )

    logger.info("=" * 60)
    logger.info("FETCHING DATE RANGE")
    logger.info("Range: %s to %s", START_DATE, END_DATE)
    logger.info("=" * 60)

    # Fetch all trades for the full date range (no chunking needed for weekly fetches)
    if parallel and max_workers > 1:
        trades = _fetch_all_trades_parallel(actual_start_ts, end_ts, max_workers)
    else:
        trades = _fetch_all_trades_in_range(exchange, actual_start_ts, end_ts)

    if not trades:
        logger.warning("No trades returned for this date range")
        return

    # Convert to DataFrame
    new_df = pd.DataFrame(trades)
    if new_df.empty:
        logger.warning("DataFrame empty after normalization")
        return

    # Apply date filtering
    new_df = _filter_date_range(new_df)
    if new_df.empty:
        logger.warning("No trades within date range after filtering")
        return

    # Append and save (merges with existing, deduplicates, sorts chronologically)
    combined_df = _append_and_save_trades(new_df, existing_df)

    # Show summary
    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info("New trades fetched:    %d", len(new_df))
    logger.info("Total trades in file:  %d", len(combined_df))

    if "timestamp" in combined_df.columns:
        data_start = combined_df["timestamp"].min()
        data_end = combined_df["timestamp"].max()
        if isinstance(data_start, pd.Timestamp):
            logger.info("Data range: %s to %s",
                       data_start.strftime("%Y-%m-%d %H:%M"),
                       data_end.strftime("%Y-%m-%d %H:%M"))

    logger.info("=" * 60)
    logger.info("To fetch more data, update START_DATE/END_DATE in constants.py and rerun.")
