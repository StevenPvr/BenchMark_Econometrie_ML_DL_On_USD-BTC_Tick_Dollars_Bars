"""ccxt-based tick downloader for crypto symbols.

Implementation that:
- validates exchange/symbol availability,
- downloads trades in 2-week chunks for robustness,
- saves each chunk to a Parquet dataset without reloading existing data,
- supports resume from existing data using metadata only.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any, Deque
import uuid

import pandas as pd  # type: ignore[import-untyped]

# Optional dependency; tests monkeypatch this attribute directly
try:  # pragma: no cover - best-effort import
    import ccxt as _ccxt  # type: ignore
except Exception:  # pragma: no cover - missing ccxt is acceptable in tests
    _ccxt = None
ccxt = _ccxt

try:  # pragma: no cover - best-effort import
    import pyarrow as _pa  # type: ignore[import-untyped]
    import pyarrow.parquet as _pq  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - pyarrow may be missing in some envs
    _pa = None
    _pq = None

pa = _pa
pq = _pq

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
    # Non-overlapping ranges: start inclusive, end exclusive
    total_range = max(1, end_ts - start_ts)
    slice_size = max(1, total_range // max_workers)

    time_ranges = []
    worker_start = start_ts
    worker_id = 0
    while worker_start < end_ts:
        worker_end = min(worker_start + slice_size, end_ts)
        time_ranges.append((EXCHANGE_ID, worker_start, worker_end, worker_id))
        worker_start = worker_end
        worker_id += 1

    logger.info("Starting parallel fetch with %d workers for time range %s to %s",
               len(time_ranges),
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
                    df_chunk = _optimize_trades_memory(pd.DataFrame(trades))
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
    start_dt = pd.to_datetime(start_ts, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="ms", utc=True)
    all_trades_dfs: list[pd.DataFrame] = []
    current_since = start_ts
    total_trades_count = 0

    logger.info("Fetching trades from %s to %s", start_dt, end_dt)

    while current_since < end_ts:
        try:
            # Respect rate limits
            rate_limiter.wait_if_needed()

            trades = exchange.fetch_trades(SYMBOL, since=current_since, limit=MAX_TRADES_PER_CALL)

            if not trades:
                logger.info("No more trades returned, stopping iteration")
                break

            temp_df = _optimize_trades_memory(pd.DataFrame(trades))

            if "timestamp" not in temp_df.columns or temp_df.empty:
                logger.warning("No timestamp column in trades data, skipping batch")
                break

            # Filter trades that are within our range
            mask = (temp_df["timestamp"] >= start_dt) & (temp_df["timestamp"] < end_dt)
            valid_trades = temp_df.loc[mask].copy()

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
            last_trade_datetime = pd.to_datetime(last_timestamp, unit="ns", utc=True)
            logger.info("Fetched %d trades so far (current batch: %d valid) - Last trade: %s",
                        total_trades_count, len(valid_trades),
                        last_trade_datetime.strftime("%Y-%m-%d %H:%M:%S UTC"))

            # Safety check: if we got less than MAX_TRADES_PER_CALL, we're probably at the end
            if len(trades) < MAX_TRADES_PER_CALL:
                logger.info("Received less than max trades per call (%d < %d), likely reached end of data", len(trades), MAX_TRADES_PER_CALL)
                break

            # Check if we're still within our date range
            if current_since >= end_ts:
                logger.info("Reached end timestamp, stopping iteration")
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


def _filter_date_range(df: pd.DataFrame, start_dt: pd.Timestamp | None = None, end_dt: pd.Timestamp | None = None) -> pd.DataFrame:
    """Limit trades to date range."""
    if df.empty or "timestamp" not in df.columns:
        return df

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):  # type: ignore[attr-defined]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    # Use provided dates or fall back to constants
    if start_dt is None or end_dt is None:
        start_dt, end_dt = _get_date_bounds()

    mask = (df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)
    return df.loc[mask].reset_index(drop=True)


def _optimize_trades_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Drop heavyweight columns and downcast to trim memory usage."""
    if df.empty:
        return df

    drop_cols = [col for col in ("info", "datetime") if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):  # type: ignore[attr-defined]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]) if "timestamp" in df.columns else df

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amount"] = df["amount"].astype("float32")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["price"] = df["price"].astype("float32")
    if "side" in df.columns and df["side"].dtype == object:
        df["side"] = df["side"].astype("category")

    return df


def _get_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start/end datetimes for configured date range (end exclusive)."""
    start_dt = pd.to_datetime(START_DATE, utc=True)
    end_dt = pd.to_datetime(END_DATE, utc=True)
    return start_dt, end_dt


def _iter_parquet_files(dataset_path: Path) -> list[Path]:
    """Return list of parquet files for a dataset path (file or directory)."""
    if dataset_path.is_file():
        return [dataset_path]
    if dataset_path.is_dir():
        return sorted(p for p in dataset_path.glob("*.parquet") if p.is_file())
    return []


def _get_existing_timestamp_range() -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Read min/max timestamps from parquet metadata without loading full data."""
    dataset_path = Path(DATASET_RAW_PARQUET)
    files = _iter_parquet_files(dataset_path)
    if not files or pq is None:
        return None, None, 0

    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None
    total_rows = 0

    for file_path in files:
        try:
            pf = pq.ParquetFile(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read metadata from %s: %s", file_path, exc)
            continue

        if pf.metadata:
            total_rows += pf.metadata.num_rows

        try:
            col_index = pf.schema_arrow.get_field_index("timestamp")
        except (KeyError, ValueError):
            continue

        for rg_idx in range(pf.metadata.num_row_groups):  # type: ignore[union-attr]
            stats = pf.metadata.row_group(rg_idx).column(col_index).statistics  # type: ignore[union-attr]
            if stats is None or stats.min is None or stats.max is None:
                continue

            rg_min = pd.to_datetime(stats.min, errors="coerce", utc=True)
            rg_max = pd.to_datetime(stats.max, errors="coerce", utc=True)

            if pd.notna(rg_min) and (min_ts is None or rg_min < min_ts):
                min_ts = rg_min
            if pd.notna(rg_max) and (max_ts is None or rg_max > max_ts):
                max_ts = rg_max

    return min_ts, max_ts, total_rows


def _ensure_parquet_dataset_path(path: Path) -> Path:
    """Guarantee that DATASET_RAW_PARQUET is a directory-based dataset."""
    ensure_output_dir(path)

    # Check for consolidated file and convert it to base partition
    consolidated_file = path.parent / "dataset_raw_final.parquet"
    if consolidated_file.exists() and consolidated_file.is_file():
        logger.info("Found consolidated file %s, converting to base partition", consolidated_file)
        path.mkdir(parents=True, exist_ok=True)
        consolidated_file.rename(path / "part-00000.parquet")
        logger.info("Converted consolidated file to base partition part-00000.parquet")
        return path

    if path.exists() and path.is_file():
        migrated_src = path.with_suffix(path.suffix + ".single")
        path.rename(migrated_src)
        path.mkdir(parents=True, exist_ok=True)
        migrated_src.rename(path / "part-00000.parquet")
        logger.info("Converted single parquet file to dataset directory at %s", path)
        return path

    path.mkdir(parents=True, exist_ok=True)
    return path


def _deduplicate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate trades within a single chunk."""
    subset_cols = [col for col in ("id", "timestamp") if col in df.columns]
    if not subset_cols:
        return df

    before = len(df)
    deduped = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    removed = before - len(deduped)
    if removed > 0:
        logger.info("Removed %d duplicate trades in chunk", removed)
    return deduped


def _append_trades_to_dataset(df: pd.DataFrame) -> None:
    """Append trades to a parquet dataset without loading existing data."""
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required to append parquet data")

    dataset_dir = _ensure_parquet_dataset_path(Path(DATASET_RAW_PARQUET))
    table = pa.Table.from_pandas(df, preserve_index=False)
    filename = f"part-{int(time.time() * 1_000_000)}-{uuid.uuid4().hex[:8]}.parquet"
    target_path = dataset_dir / filename
    pq.write_table(table, target_path, compression="snappy")
    logger.info("Saved %d trades to %s", len(df), target_path)



def download_ticks_in_date_range(parallel: bool = True, max_workers: int = 6,
                               start_date: str | None = None, end_date: str | None = None) -> None:
    """Download trades for SYMBOL from EXCHANGE_ID for the specified date range.

    Downloads data for the date range specified by parameters or constants and appends
    to existing data without loading the full parquet file in memory.

    This allows you to:
    1. Specify custom date range or use defaults from constants
    2. Run the script to fetch data for that period
    3. Data is automatically appended to existing dataset directory (no overwrite)

    Args:
        parallel: Whether to use parallel fetching within each chunk (default: True)
        max_workers: Number of parallel workers per chunk (default: 6)
        start_date: Start date in YYYY-MM-DD format (default: START_DATE from constants)
        end_date: End date in YYYY-MM-DD format (default: END_DATE from constants)
    """
    if start_date is None or end_date is None:
        start_dt, end_dt = _get_date_bounds()
    else:
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

    # Compute timestamp range
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    # Validate exchange first
    exchange = _build_exchange()
    _validate_symbol(exchange)

    fetch_start_ts = start_ts

    total_new_trades = 0

    chunk_ms = int(timedelta(days=CHUNK_DAYS).total_seconds() * 1000)
    chunk_start = fetch_start_ts

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + chunk_ms, end_ts)
        logger.info("Chunk fetch: %s to %s",
                    pd.to_datetime(chunk_start, unit="ms", utc=True),
                    pd.to_datetime(chunk_end, unit="ms", utc=True))

        if parallel and max_workers > 1:
            trades = _fetch_all_trades_parallel(chunk_start, chunk_end, max_workers)
        else:
            trades = _fetch_all_trades_in_range(exchange, chunk_start, chunk_end)

        if not trades:
            logger.info("Chunk returned no trades, moving to next window")
            chunk_start = chunk_end + 1
            continue

        new_df = pd.DataFrame(trades)
        if new_df.empty:
            logger.warning("Chunk DataFrame empty after normalization")
            chunk_start = chunk_end + 1
            continue

        # Apply date filtering and downcasting to keep memory low
        new_df = _filter_date_range(new_df, start_dt, end_dt)
        new_df = _optimize_trades_memory(new_df)
        if new_df.empty:
            logger.warning("No trades within date range after filtering")
            chunk_start = chunk_end + 1
            continue

        new_df = _deduplicate_chunk(new_df)
        if "timestamp" in new_df.columns:
            new_df = new_df.sort_values("timestamp").reset_index(drop=True)

        _append_trades_to_dataset(new_df)
        total_new_trades += len(new_df)

        chunk_start = chunk_end

    if total_new_trades == 0:
        logger.warning("No trades returned for this date range")
        return

    logger.info("New trades fetched: %d", total_new_trades)
