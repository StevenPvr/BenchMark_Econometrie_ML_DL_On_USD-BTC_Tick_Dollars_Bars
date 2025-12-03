"""Tests for data_fetching modules."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Mock dependencies if they are not installed
sys.modules["ccxt"] = MagicMock()
sys.modules["pyarrow"] = MagicMock()
sys.modules["pyarrow.parquet"] = MagicMock()

from src.data_fetching.exchange import build_exchange, validate_symbol
from src.data_fetching.fetching import download_ticks_in_date_range
from src.data_fetching.rate_limiter import RateLimiter
from src.data_fetching.storage import (
    ensure_parquet_dataset_path,
    get_existing_timestamp_range,
    iter_parquet_files,
)
from src.data_fetching.trades import (
    compute_time_ranges,
    concat_and_sort_trades,
    deduplicate_chunk,
    fetch_all_trades_in_range,
    fetch_all_trades_parallel,
    fetch_trades_for_time_range,
    filter_date_range,
    get_date_bounds,
    optimize_trades_memory,
)


# --- Fixtures ---


@pytest.fixture
def mock_ccxt() -> MagicMock:
    """Mock the ccxt module."""
    with patch("src.data_fetching.exchange.ccxt") as mock:
        yield mock


@pytest.fixture
def mock_rate_limiter() -> MagicMock:
    """Mock the rate limiter."""
    with patch("src.data_fetching.trades.rate_limiter") as mock:
        yield mock


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Create a sample trades DataFrame for testing."""
    data = {
        "timestamp": pd.to_datetime(
            ["2023-01-01 10:00:00", "2023-01-01 10:01:00"], utc=True
        ),
        "price": [20000.0, 20005.0],
        "amount": [0.1, 0.2],
        "id": ["1", "2"],
        "side": ["buy", "sell"],
    }
    return pd.DataFrame(data)


# --- RateLimiter Tests ---


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_init(self) -> None:
        """Test RateLimiter initialization."""
        limiter = RateLimiter(6000, 1)
        assert limiter.max_weight_per_minute == 6000
        assert limiter.weight_per_call == 1
        assert len(limiter.call_times) == 0

    @patch("time.time")
    @patch("time.sleep")
    def test_wait_if_needed_no_wait(
        self, mock_sleep: MagicMock, mock_time: MagicMock
    ) -> None:
        """Test that wait_if_needed does not sleep when under limit."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter(10, 1)

        limiter.wait_if_needed()
        assert len(limiter.call_times) == 1
        mock_sleep.assert_not_called()

    @patch("time.time")
    @patch("time.sleep")
    def test_wait_if_needed_waits_when_at_limit(
        self, mock_sleep: MagicMock, mock_time: MagicMock
    ) -> None:
        """Test that wait_if_needed sleeps when at rate limit."""
        mock_time.return_value = 1000.0
        limiter = RateLimiter(1, 1)

        def side_effect_sleep(seconds: float) -> None:
            mock_time.return_value += seconds + 0.1

        mock_sleep.side_effect = side_effect_sleep

        limiter.wait_if_needed()
        limiter.wait_if_needed()

        assert mock_sleep.called
        assert len(limiter.call_times) == 1

    def test_remove_expired_calls(self) -> None:
        """Test that expired calls are removed from the deque."""
        limiter = RateLimiter(100, 1)
        limiter.call_times = deque([1000.0, 1030.0, 1050.0])

        limiter._remove_expired_calls(1065.0)

        assert len(limiter.call_times) == 2
        assert 1000.0 not in limiter.call_times

    def test_compute_wait_time_with_calls(self) -> None:
        """Test wait time computation with existing calls."""
        limiter = RateLimiter(100, 1)
        limiter.call_times = deque([1000.0])

        wait_time = limiter._compute_wait_time(1050.0)

        assert wait_time == pytest.approx(10.1, rel=0.1)

    def test_compute_wait_time_empty(self) -> None:
        """Test wait time computation with empty call queue."""
        limiter = RateLimiter(100, 1)

        wait_time = limiter._compute_wait_time(1050.0)

        assert wait_time == 0.1


# --- Exchange Tests ---


def test_build_exchange(mock_ccxt: MagicMock) -> None:
    """Test exchange building."""
    with patch("src.data_fetching.exchange.EXCHANGE_ID", "binance"):
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        exchange = build_exchange()
        assert exchange == mock_exchange
        mock_ccxt.binance.assert_called_once()


def test_build_exchange_missing_id(mock_ccxt: MagicMock) -> None:
    """Test exchange building with missing exchange ID."""
    with patch("src.data_fetching.exchange.EXCHANGE_ID", "unknown_exchange"):
        del mock_ccxt.unknown_exchange
        with pytest.raises(ValueError, match="Exchange unknown_exchange not found"):
            build_exchange()


def test_validate_symbol() -> None:
    """Test symbol validation with valid symbol."""
    mock_exchange = MagicMock()
    mock_exchange.load_markets.return_value = {"BTC/USDT": {"active": True}}

    with patch("src.data_fetching.exchange.SYMBOL", "BTC/USDT"):
        validate_symbol(mock_exchange)


def test_validate_symbol_missing() -> None:
    """Test symbol validation with missing symbol."""
    mock_exchange = MagicMock()
    mock_exchange.load_markets.return_value = {}

    with patch("src.data_fetching.exchange.SYMBOL", "BTC/USDT"):
        with pytest.raises(ValueError, match="Symbole BTC/USDT non disponible"):
            validate_symbol(mock_exchange)


def test_validate_symbol_inactive() -> None:
    """Test symbol validation with inactive symbol."""
    mock_exchange = MagicMock()
    mock_exchange.load_markets.return_value = {"BTC/USDT": {"active": False}}

    with patch("src.data_fetching.exchange.SYMBOL", "BTC/USDT"):
        with pytest.raises(ValueError, match="Symbole BTC/USDT inactif"):
            validate_symbol(mock_exchange)


# --- Trades Processing Tests ---


def test_optimize_trades_memory(sample_trades_df: pd.DataFrame) -> None:
    """Test memory optimization of trades DataFrame."""
    df = sample_trades_df.copy()
    df["info"] = "some info"
    df["datetime"] = "2023-01-01T10:00:00Z"
    df["amount"] = df["amount"].astype(float)

    optimized = optimize_trades_memory(df)

    assert "info" not in optimized.columns
    assert "datetime" not in optimized.columns
    assert optimized["amount"].dtype == "float32"
    assert optimized["price"].dtype == "float32"
    assert isinstance(optimized["side"].dtype, pd.CategoricalDtype)


def test_optimize_trades_memory_empty() -> None:
    """Test memory optimization with empty DataFrame."""
    df = pd.DataFrame()
    result = optimize_trades_memory(df)
    assert result.empty


def test_deduplicate_chunk(sample_trades_df: pd.DataFrame) -> None:
    """Test deduplication of trades chunk."""
    df = pd.concat([sample_trades_df, sample_trades_df.iloc[[0]]], ignore_index=True)
    assert len(df) == 3

    deduped = deduplicate_chunk(df)
    assert len(deduped) == 2
    assert deduped.iloc[0]["id"] == sample_trades_df.iloc[0]["id"]


def test_deduplicate_chunk_no_id_or_timestamp() -> None:
    """Test deduplication when no id or timestamp columns."""
    df = pd.DataFrame({"price": [1, 2, 3]})
    result = deduplicate_chunk(df)
    assert len(result) == 3


def test_filter_date_range(sample_trades_df: pd.DataFrame) -> None:
    """Test date range filtering."""
    start_dt = pd.to_datetime("2023-01-01 10:00:30", utc=True)
    end_dt = pd.to_datetime("2023-01-01 10:02:00", utc=True)

    filtered = filter_date_range(sample_trades_df, start_dt, end_dt)
    assert len(filtered) == 1
    assert filtered.iloc[0]["timestamp"] == pd.to_datetime(
        "2023-01-01 10:01:00", utc=True
    )


def test_filter_date_range_empty() -> None:
    """Test date range filtering with empty DataFrame."""
    df = pd.DataFrame()
    result = filter_date_range(df)
    assert result.empty


def test_filter_date_range_no_timestamp_column() -> None:
    """Test date range filtering without timestamp column."""
    df = pd.DataFrame({"price": [1, 2, 3]})
    result = filter_date_range(df)
    assert len(result) == 3


def test_compute_time_ranges() -> None:
    """Test time range computation for parallel workers."""
    ranges = compute_time_ranges(1000, 3000, 2)

    assert len(ranges) == 2
    assert ranges[0][1] == 1000  # start_ts
    assert ranges[1][2] == 3000  # end_ts


def test_concat_and_sort_trades() -> None:
    """Test concatenation and sorting of trade DataFrames."""
    df1 = pd.DataFrame({"timestamp": [2], "id": ["2"]})
    df2 = pd.DataFrame({"timestamp": [1], "id": ["1"]})

    result = concat_and_sort_trades([df1, df2])

    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "2"


def test_concat_and_sort_trades_empty() -> None:
    """Test concatenation with empty list."""
    with patch("pandas.concat", side_effect=Exception("Error")):
        result = concat_and_sort_trades([pd.DataFrame()])
    assert result == []


# --- Fetching Logic Tests ---


@patch("src.data_fetching.trades.fetch_all_trades_in_range")
@patch("src.data_fetching.trades.ccxt")
def test_fetch_trades_for_time_range(
    mock_ccxt_module: MagicMock, mock_fetch_inner: MagicMock
) -> None:
    """Test fetching trades for a time range."""
    mock_exchange = MagicMock()
    mock_ccxt_module.binance.return_value = mock_exchange

    args = ("binance", 1000, 2000, 1)
    mock_fetch_inner.return_value = [{"id": "1"}]

    result = fetch_trades_for_time_range(args)

    assert result == [{"id": "1"}]
    mock_fetch_inner.assert_called_once()


@patch("concurrent.futures.ThreadPoolExecutor")
def test_fetch_all_trades_parallel(mock_executor: MagicMock) -> None:
    """Test parallel trade fetching."""
    mock_future1 = MagicMock()
    mock_future1.result.return_value = [{"id": "1", "timestamp": 1000}]

    mock_future2 = MagicMock()
    mock_future2.result.return_value = [{"id": "2", "timestamp": 2000}]

    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

    with patch(
        "concurrent.futures.as_completed", return_value=[mock_future1, mock_future2]
    ):
        with patch("src.data_fetching.trades.EXCHANGE_ID", "binance"):
            result = fetch_all_trades_parallel(1000, 3000, max_workers=2)

    assert len(result) == 2


def test_fetch_all_trades_in_range_basic() -> None:
    """Test basic trade fetching in range."""
    mock_exchange = MagicMock()
    start_ts = 1672531200000  # 2023-01-01
    end_ts = 1672531260000  # + 60s

    trades_batch = [
        {"id": "1", "timestamp": 1672531200000, "price": 100, "amount": 1},
        {"id": "2", "timestamp": 1672531210000, "price": 100, "amount": 1},
    ]

    mock_exchange.fetch_trades.side_effect = [trades_batch, []]

    with patch("src.data_fetching.trades.rate_limiter.wait_if_needed"):
        with patch("src.data_fetching.trades.SYMBOL", "BTC/USDT"):
            with patch("src.data_fetching.trades.FETCHING_MAX_TRADES_PER_CALL", 2):
                result = fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)

    assert len(result) == 2
    assert mock_exchange.fetch_trades.call_count == 2


@patch("src.data_fetching.fetching.build_exchange")
@patch("src.data_fetching.fetching.validate_symbol")
@patch("src.data_fetching.fetching.fetch_all_trades_parallel")
@patch("src.data_fetching.fetching.append_trades_to_dataset")
@patch("src.data_fetching.fetching.get_date_bounds")
def test_download_ticks_in_date_range(
    mock_get_bounds: MagicMock,
    mock_append: MagicMock,
    mock_fetch_parallel: MagicMock,
    mock_validate: MagicMock,
    mock_build: MagicMock,
) -> None:
    """Test download_ticks_in_date_range function."""
    mock_get_bounds.return_value = (
        pd.to_datetime("2023-01-01", utc=True),
        pd.to_datetime("2023-01-02", utc=True),
    )

    mock_fetch_parallel.return_value = [{"id": "1", "timestamp": 1672531200000}]

    download_ticks_in_date_range(
        parallel=True, start_date="2023-01-01", end_date="2023-01-02"
    )

    mock_build.assert_called_once()
    mock_validate.assert_called_once()
    mock_fetch_parallel.assert_called()
    mock_append.assert_called()


# --- Exception Handling Tests ---


@patch("src.data_fetching.trades.fetch_all_trades_in_range")
@patch("src.data_fetching.trades.ccxt")
def test_fetch_trades_for_time_range_exception(
    mock_ccxt_module: MagicMock, mock_fetch_inner: MagicMock
) -> None:
    """Test exception handling during trade fetching."""
    mock_exchange = MagicMock()
    mock_ccxt_module.binance.return_value = mock_exchange
    args = ("binance", 1000, 2000, 1)

    mock_fetch_inner.side_effect = Exception("Fetch error")
    result = fetch_trades_for_time_range(args)
    assert result == []

    mock_ccxt_module.binance.side_effect = Exception("Exchange error")
    result = fetch_trades_for_time_range(args)
    assert result == []


@patch("concurrent.futures.ThreadPoolExecutor")
def test_fetch_all_trades_parallel_exceptions(mock_executor: MagicMock) -> None:
    """Test exception handling in parallel fetching."""
    mock_future1 = MagicMock()
    mock_future1.result.side_effect = Exception("Worker error")

    mock_future2 = MagicMock()
    mock_future2.result.return_value = []

    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

    with patch(
        "concurrent.futures.as_completed", return_value=[mock_future1, mock_future2]
    ):
        with patch("src.data_fetching.trades.EXCHANGE_ID", "binance"):
            result = fetch_all_trades_parallel(1000, 3000, max_workers=2)

    assert result == []


def test_fetch_all_trades_in_range_edge_cases() -> None:
    """Test edge cases in trade fetching."""
    mock_exchange = MagicMock()
    start_ts = 1000
    end_ts = 2000

    # Exchange error
    mock_exchange.fetch_trades.side_effect = Exception("Network error")
    with patch("src.data_fetching.trades.rate_limiter.wait_if_needed"):
        result = fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)
    assert result == []

    # Empty trades returned
    mock_exchange.reset_mock()
    mock_exchange.fetch_trades.side_effect = None
    mock_exchange.fetch_trades.return_value = []
    with patch("src.data_fetching.trades.rate_limiter.wait_if_needed"):
        result = fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)
    assert result == []


# --- Storage Tests ---


def test_iter_parquet_files(tmp_path: Path) -> None:
    """Test parquet file iteration."""
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "1.parquet").touch()
    (d / "2.parquet").touch()
    (d / "other.txt").touch()

    files = iter_parquet_files(d)
    assert len(files) == 2

    f = tmp_path / "single.parquet"
    f.touch()
    files = iter_parquet_files(f)
    assert len(files) == 1

    files = iter_parquet_files(tmp_path / "missing")
    assert len(files) == 0


def test_ensure_parquet_dataset_path_dir_exists(tmp_path: Path) -> None:
    """Test ensuring parquet dataset path when directory exists."""
    d = tmp_path / "dataset.parquet"
    d.mkdir()
    path = ensure_parquet_dataset_path(d)
    assert path == d
    assert path.is_dir()


def test_ensure_parquet_dataset_path_migration_single_file(tmp_path: Path) -> None:
    """Test migration from single file to directory."""
    p = tmp_path / "dataset.parquet"
    p.touch()

    path = ensure_parquet_dataset_path(p)

    assert path == p
    assert path.is_dir()
    assert (path / "part-00000.parquet").exists()


def test_ensure_parquet_dataset_path_migration_consolidated(tmp_path: Path) -> None:
    """Test migration from consolidated file."""
    parent = tmp_path / "data"
    parent.mkdir()
    p = parent / "dataset.parquet"
    consolidated = parent / "dataset_raw_final.parquet"
    consolidated.touch()

    path = ensure_parquet_dataset_path(p)

    assert path == p
    assert path.is_dir()
    assert (path / "part-00000.parquet").exists()
    assert not consolidated.exists()


def test_get_existing_timestamp_range_empty() -> None:
    """Test getting timestamp range from empty dataset."""
    with patch("src.data_fetching.storage.DATASET_RAW_PARQUET", "dummy"):
        with patch("src.data_fetching.storage.iter_parquet_files", return_value=[]):
            min_ts, max_ts, count = get_existing_timestamp_range()
            assert min_ts is None
            assert max_ts is None
            assert count == 0


@patch("src.data_fetching.fetching.build_exchange")
@patch("src.data_fetching.fetching.validate_symbol")
@patch("src.data_fetching.fetching.fetch_all_trades_parallel")
@patch("src.data_fetching.fetching.get_date_bounds")
def test_download_ticks_in_date_range_no_trades(
    mock_get_bounds: MagicMock,
    mock_fetch_parallel: MagicMock,
    mock_validate: MagicMock,
    mock_build: MagicMock,
) -> None:
    """Test download when no trades returned."""
    mock_get_bounds.return_value = (
        pd.to_datetime("2023-01-01", utc=True),
        pd.to_datetime("2023-01-02", utc=True),
    )
    mock_fetch_parallel.return_value = []

    download_ticks_in_date_range(parallel=True)

    mock_fetch_parallel.assert_called()


# --- Date Bounds Tests ---


class TestGetDateBounds:
    """Tests for get_date_bounds function."""

    def test_get_date_bounds_returns_timestamps(self) -> None:
        """Test that get_date_bounds returns valid timestamps."""
        with patch("src.data_fetching.trades.START_DATE", "2023-01-01"):
            with patch("src.data_fetching.trades.END_DATE", "2023-12-31"):
                start, end = get_date_bounds()

                assert isinstance(start, pd.Timestamp)
                assert isinstance(end, pd.Timestamp)
                assert start.year == 2023
                assert end.year == 2023

    def test_get_date_bounds_utc_aware(self) -> None:
        """Test that returned timestamps are UTC-aware."""
        with patch("src.data_fetching.trades.START_DATE", "2023-06-15"):
            with patch("src.data_fetching.trades.END_DATE", "2023-07-15"):
                start, end = get_date_bounds()

                assert start.tzinfo is not None
                assert end.tzinfo is not None


class TestIterParquetFilesExtra:
    """Additional tests for iter_parquet_files."""

    def test_iter_parquet_files_sorted(self, tmp_path: Path) -> None:
        """Test that files are returned in sorted order."""
        (tmp_path / "part-00002.parquet").write_bytes(b"dummy")
        (tmp_path / "part-00000.parquet").write_bytes(b"dummy")
        (tmp_path / "part-00001.parquet").write_bytes(b"dummy")

        result = iter_parquet_files(tmp_path)

        assert len(result) == 3
        assert "part-00000" in str(result[0])
        assert "part-00001" in str(result[1])
        assert "part-00002" in str(result[2])

    def test_iter_parquet_files_nonexistent_path(self, tmp_path: Path) -> None:
        """Test with non-existent path."""
        result = iter_parquet_files(tmp_path / "nonexistent")
        assert result == []
