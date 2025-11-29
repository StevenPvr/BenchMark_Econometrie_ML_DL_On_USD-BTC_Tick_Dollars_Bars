
import pytest
import pandas as pd
import threading
import time
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, call, Mock
from collections import deque
from datetime import datetime
import sys
import numpy as np

# Mock dependencies if they are not installed
# This is handled by the module import but we want to ensure we can control them in tests
sys.modules['ccxt'] = MagicMock()
sys.modules['pyarrow'] = MagicMock()
sys.modules['pyarrow.parquet'] = MagicMock()

from src.data_fetching import fetching
from src.data_fetching.fetching import (
    RateLimiter,
    _build_exchange,
    _validate_symbol,
    _fetch_trades_for_time_range,
    _fetch_all_trades_parallel,
    _fetch_all_trades_in_range,
    _filter_date_range,
    _optimize_trades_memory,
    _deduplicate_chunk,
    _iter_parquet_files,
    _ensure_parquet_dataset_path,
    download_ticks_in_date_range,
)

# --- Fixtures ---

@pytest.fixture
def mock_ccxt():
    with patch('src.data_fetching.fetching.ccxt') as mock:
        yield mock

@pytest.fixture
def mock_rate_limiter():
    with patch('src.data_fetching.fetching.rate_limiter') as mock:
        yield mock

@pytest.fixture
def sample_trades_df():
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:01:00'], utc=True),
        'price': [20000.0, 20005.0],
        'amount': [0.1, 0.2],
        'id': ['1', '2'],
        'side': ['buy', 'sell']
    }
    return pd.DataFrame(data)

# --- RateLimiter Tests ---

class TestRateLimiter:
    def test_init(self):
        limiter = RateLimiter(6000, 1)
        assert limiter.max_weight_per_minute == 6000
        assert limiter.weight_per_call == 1
        assert len(limiter.call_times) == 0

    @patch('time.time')
    @patch('time.sleep')
    def test_wait_if_needed_no_wait(self, mock_sleep, mock_time):
        mock_time.return_value = 1000.0
        limiter = RateLimiter(10, 1)

        # First call should pass immediately
        limiter.wait_if_needed()
        assert len(limiter.call_times) == 1
        mock_sleep.assert_not_called()

    @patch('time.time')
    @patch('time.sleep')
    def test_wait_if_needed_wait(self, mock_sleep, mock_time):
        mock_time.return_value = 1000.0
        limiter = RateLimiter(1, 1)

        def side_effect_sleep(seconds):
            mock_time.return_value += seconds + 0.1 # Advance time

        mock_sleep.side_effect = side_effect_sleep

        # First call passes
        limiter.wait_if_needed()
        # Second call triggers sleep
        limiter.wait_if_needed()

        assert mock_sleep.called
        assert len(limiter.call_times) == 1

# --- Helper Functions Tests ---

def test_build_exchange(mock_ccxt):
    with patch('src.data_fetching.fetching.EXCHANGE_ID', 'binance'):
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange

        exchange = _build_exchange()
        assert exchange == mock_exchange
        mock_ccxt.binance.assert_called_once()

def test_build_exchange_missing_id(mock_ccxt):
    with patch('src.data_fetching.fetching.EXCHANGE_ID', 'unknown_exchange'):
        del mock_ccxt.unknown_exchange # Ensure it doesn't exist
        with pytest.raises(ValueError, match="Exchange unknown_exchange not found"):
            _build_exchange()

def test_validate_symbol():
    mock_exchange = MagicMock()
    mock_exchange.load_markets.return_value = {
        'BTC/USDT': {'active': True}
    }

    with patch('src.data_fetching.fetching.SYMBOL', 'BTC/USDT'):
        # Should not raise
        _validate_symbol(mock_exchange)

def test_validate_symbol_missing():
    mock_exchange = MagicMock()
    mock_exchange.load_markets.return_value = {}

    with patch('src.data_fetching.fetching.SYMBOL', 'BTC/USDT'):
        with pytest.raises(ValueError, match="Symbole BTC/USDT non disponible"):
            _validate_symbol(mock_exchange)

def test_validate_symbol_inactive():
    mock_exchange = MagicMock()
    mock_exchange.load_markets.return_value = {
        'BTC/USDT': {'active': False}
    }

    with patch('src.data_fetching.fetching.SYMBOL', 'BTC/USDT'):
        with pytest.raises(ValueError, match="Symbole BTC/USDT inactif"):
            _validate_symbol(mock_exchange)

def test_optimize_trades_memory(sample_trades_df):
    # Add some cols to drop or convert
    df = sample_trades_df.copy()
    df['info'] = 'some info'
    df['datetime'] = '2023-01-01T10:00:00Z'
    df['amount'] = df['amount'].astype(float) # float64 default

    optimized = _optimize_trades_memory(df)

    assert 'info' not in optimized.columns
    assert 'datetime' not in optimized.columns
    assert optimized['amount'].dtype == 'float32'
    assert optimized['price'].dtype == 'float32'
    assert isinstance(optimized['side'].dtype, pd.CategoricalDtype)

def test_deduplicate_chunk(sample_trades_df):
    df = pd.concat([sample_trades_df, sample_trades_df.iloc[[0]]], ignore_index=True)
    assert len(df) == 3

    deduped = _deduplicate_chunk(df)
    assert len(deduped) == 2
    assert deduped.iloc[0]['id'] == sample_trades_df.iloc[0]['id']

def test_filter_date_range(sample_trades_df):
    start_dt = pd.to_datetime('2023-01-01 10:00:30', utc=True)
    end_dt = pd.to_datetime('2023-01-01 10:02:00', utc=True)

    filtered = _filter_date_range(sample_trades_df, start_dt, end_dt)
    assert len(filtered) == 1
    assert filtered.iloc[0]['timestamp'] == pd.to_datetime('2023-01-01 10:01:00', utc=True)

# --- Fetching Logic Tests ---

@patch('src.data_fetching.fetching._fetch_all_trades_in_range')
@patch('src.data_fetching.fetching.ccxt') # Global ccxt
def test_fetch_trades_for_time_range(mock_ccxt_module, mock_fetch_inner):
    mock_exchange = MagicMock()
    mock_ccxt_module.binance.return_value = mock_exchange

    args = ('binance', 1000, 2000, 1)

    mock_fetch_inner.return_value = [{'id': '1'}]

    result = _fetch_trades_for_time_range(args)

    assert result == [{'id': '1'}]
    mock_fetch_inner.assert_called_once()


@patch('concurrent.futures.ThreadPoolExecutor')
def test_fetch_all_trades_parallel(mock_executor):
    # Setup mock futures
    mock_future1 = MagicMock()
    mock_future1.result.return_value = [{'id': '1', 'timestamp': 1000}]

    mock_future2 = MagicMock()
    mock_future2.result.return_value = [{'id': '2', 'timestamp': 2000}]

    # Configure executor context manager
    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

    # Configure as_completed to return futures
    with patch('concurrent.futures.as_completed', return_value=[mock_future1, mock_future2]):
        with patch('src.data_fetching.fetching.EXCHANGE_ID', 'binance'):
            result = _fetch_all_trades_parallel(1000, 3000, max_workers=2)

    assert len(result) == 2
    # Check if sorted
    assert result[0]['id'] == '1'
    assert result[1]['id'] == '2'

def test_fetch_all_trades_in_range_basic():
    mock_exchange = MagicMock()
    start_ts = 1672531200000 # 2023-01-01
    end_ts =   1672531260000 # + 60s

    trades_batch = [
        {'id': '1', 'timestamp': 1672531200000, 'price': 100, 'amount': 1},
        {'id': '2', 'timestamp': 1672531210000, 'price': 100, 'amount': 1}
    ]

    mock_exchange.fetch_trades.side_effect = [trades_batch, []]

    with patch('src.data_fetching.fetching.rate_limiter.wait_if_needed'):
        with patch('src.data_fetching.fetching.SYMBOL', 'BTC/USDT'):
            with patch('src.data_fetching.fetching.MAX_TRADES_PER_CALL', 2):
                 result = _fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)

    assert len(result) == 2
    assert result[0]['id'] == '1'
    assert result[1]['id'] == '2'
    assert mock_exchange.fetch_trades.call_count == 2

@patch('src.data_fetching.fetching._build_exchange')
@patch('src.data_fetching.fetching._validate_symbol')
@patch('src.data_fetching.fetching._fetch_all_trades_parallel')
@patch('src.data_fetching.fetching._append_trades_to_dataset')
@patch('src.data_fetching.fetching._get_date_bounds')
def test_download_ticks_in_date_range(mock_get_bounds, mock_append, mock_fetch_parallel, mock_validate, mock_build):
    mock_get_bounds.return_value = (
        pd.to_datetime('2023-01-01', utc=True),
        pd.to_datetime('2023-01-02', utc=True)
    )

    # Mock return trades
    mock_fetch_parallel.return_value = [{'id': '1', 'timestamp': 1672531200000}]

    download_ticks_in_date_range(parallel=True, start_date='2023-01-01', end_date='2023-01-02')

    mock_build.assert_called_once()
    mock_validate.assert_called_once()
    mock_fetch_parallel.assert_called()
    mock_append.assert_called()

# --- Exception Handling Tests ---

@patch('src.data_fetching.fetching._fetch_all_trades_in_range')
@patch('src.data_fetching.fetching.ccxt')
def test_fetch_trades_for_time_range_exception(mock_ccxt_module, mock_fetch_inner):
    mock_exchange = MagicMock()
    mock_ccxt_module.binance.return_value = mock_exchange
    args = ('binance', 1000, 2000, 1)

    # Exception during fetch
    mock_fetch_inner.side_effect = Exception("Fetch error")
    result = _fetch_trades_for_time_range(args)
    assert result == []

    # Exception during exchange creation
    mock_ccxt_module.binance.side_effect = Exception("Exchange error")
    result = _fetch_trades_for_time_range(args)
    assert result == []

@patch('concurrent.futures.ThreadPoolExecutor')
def test_fetch_all_trades_parallel_exceptions(mock_executor):
    mock_future1 = MagicMock()
    mock_future1.result.side_effect = Exception("Worker error")

    mock_future2 = MagicMock()
    mock_future2.result.return_value = [] # Empty result

    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

    with patch('concurrent.futures.as_completed', return_value=[mock_future1, mock_future2]):
        with patch('src.data_fetching.fetching.EXCHANGE_ID', 'binance'):
            result = _fetch_all_trades_parallel(1000, 3000, max_workers=2)

    assert result == []

@patch('concurrent.futures.ThreadPoolExecutor')
def test_fetch_all_trades_parallel_concat_error(mock_executor):
    mock_future1 = MagicMock()
    mock_future1.result.return_value = [{'id': '1'}] # No timestamp, might cause sort error or just concat

    mock_executor_instance = mock_executor.return_value.__enter__.return_value
    mock_executor_instance.submit.return_value = mock_future1

    with patch('concurrent.futures.as_completed', return_value=[mock_future1]):
        with patch('src.data_fetching.fetching.EXCHANGE_ID', 'binance'):
             with patch('pandas.concat', side_effect=Exception("Concat error")):
                result = _fetch_all_trades_parallel(1000, 2000, max_workers=1)

    assert result == []

def test_fetch_all_trades_in_range_edge_cases():
    mock_exchange = MagicMock()
    start_ts = 1000
    end_ts = 2000

    # 1. Exchange error
    mock_exchange.fetch_trades.side_effect = Exception("Network error")
    with patch('src.data_fetching.fetching.rate_limiter.wait_if_needed'):
        result = _fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)
    assert result == []

    # 2. Empty trades returned
    mock_exchange.reset_mock()
    mock_exchange.fetch_trades.side_effect = None
    mock_exchange.fetch_trades.return_value = []
    with patch('src.data_fetching.fetching.rate_limiter.wait_if_needed'):
        result = _fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)
    assert result == []

    # 3. Trades without timestamp
    mock_exchange.reset_mock()
    mock_exchange.fetch_trades.return_value = [{'id': '1'}] # no timestamp
    with patch('src.data_fetching.fetching.rate_limiter.wait_if_needed'):
        result = _fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)
    assert result == []

    # 4. Trades outside range
    mock_exchange.reset_mock()
    mock_exchange.fetch_trades.return_value = [{'id': '1', 'timestamp': 3000}]
    with patch('src.data_fetching.fetching.rate_limiter.wait_if_needed'):
        result = _fetch_all_trades_in_range(mock_exchange, start_ts, end_ts)
    assert result == []

# --- File System & Migration Tests ---

def test_iter_parquet_files(tmp_path):
    # Test directory
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "1.parquet").touch()
    (d / "2.parquet").touch()
    (d / "other.txt").touch()

    files = _iter_parquet_files(d)
    assert len(files) == 2

    # Test file
    f = tmp_path / "single.parquet"
    f.touch()
    files = _iter_parquet_files(f)
    assert len(files) == 1

    # Test non-existent
    files = _iter_parquet_files(tmp_path / "missing")
    assert len(files) == 0

def test_ensure_parquet_dataset_path_dir_exists(tmp_path):
    d = tmp_path / "dataset.parquet"
    d.mkdir()
    path = _ensure_parquet_dataset_path(d)
    assert path == d
    assert path.is_dir()

def test_ensure_parquet_dataset_path_migration_single_file(tmp_path):
    # Setup single file pretending to be the dataset path
    p = tmp_path / "dataset.parquet"
    p.touch()

    path = _ensure_parquet_dataset_path(p)

    assert path == p
    assert path.is_dir()
    assert (path / "part-00000.parquet").exists()
    # The .single file is moved into the directory as part-00000, so it shouldn't exist outside
    assert not (tmp_path / "dataset.parquet.single").exists()

def test_ensure_parquet_dataset_path_migration_consolidated(tmp_path):
    # Consolidated file exists in parent
    parent = tmp_path / "data"
    parent.mkdir()
    p = parent / "dataset.parquet"
    consolidated = parent / "dataset_raw_final.parquet"
    consolidated.touch()

    path = _ensure_parquet_dataset_path(p)

    assert path == p
    assert path.is_dir()
    assert (path / "part-00000.parquet").exists()
    assert not consolidated.exists()

def test_get_existing_timestamp_range_empty():
    with patch('src.data_fetching.fetching.DATASET_RAW_PARQUET', 'dummy'):
        with patch('src.data_fetching.fetching._iter_parquet_files', return_value=[]):
            min_ts, max_ts, count = fetching._get_existing_timestamp_range()
            assert min_ts is None
            assert max_ts is None
            assert count == 0

@patch('src.data_fetching.fetching._build_exchange')
@patch('src.data_fetching.fetching._validate_symbol')
@patch('src.data_fetching.fetching._fetch_all_trades_parallel')
@patch('src.data_fetching.fetching._get_date_bounds')
def test_download_ticks_in_date_range_no_trades(mock_get_bounds, mock_fetch_parallel, mock_validate, mock_build):
    mock_get_bounds.return_value = (pd.to_datetime('2023-01-01', utc=True), pd.to_datetime('2023-01-02', utc=True))
    mock_fetch_parallel.return_value = [] # No trades

    download_ticks_in_date_range(parallel=True)

    # Should complete without error
    mock_fetch_parallel.assert_called()

@patch('src.data_fetching.fetching._build_exchange')
@patch('src.data_fetching.fetching._validate_symbol')
@patch('src.data_fetching.fetching._fetch_all_trades_parallel')
@patch('src.data_fetching.fetching._get_date_bounds')
def test_download_ticks_in_date_range_empty_df_after_filter(mock_get_bounds, mock_fetch_parallel, mock_validate, mock_build):
    mock_get_bounds.return_value = (pd.to_datetime('2023-01-01', utc=True), pd.to_datetime('2023-01-02', utc=True))
    # Return trades outside range
    mock_fetch_parallel.return_value = [{'id': '1', 'timestamp': 1000}]

    with patch('src.data_fetching.fetching._filter_date_range') as mock_filter:
        mock_filter.return_value = pd.DataFrame() # Filter returns empty
        download_ticks_in_date_range(parallel=True)

    mock_fetch_parallel.assert_called()
