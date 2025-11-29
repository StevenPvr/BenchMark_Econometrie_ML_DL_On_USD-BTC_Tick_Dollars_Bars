
import pytest
import pandas as pd
import threading
import time
from unittest.mock import MagicMock, patch, call, Mock
from collections import deque
from datetime import datetime
import sys

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

        # First call passes
        limiter.wait_if_needed()

        # Second call should trigger sleep because limit is 1
        # We need to simulate time passing for the second call inside the loop
        # But since the loop is `while True`, we need to break it or mock it carefully.
        # However, wait_if_needed logic:
        # 1. checks if current_weight < max. If yes, append and return.
        # 2. else calc wait_time, sleep, and loop again.

        # To test this without infinite loop, we can make `time.sleep` change the mock_time
        # so that the next iteration passes.

        def side_effect_sleep(seconds):
            mock_time.return_value += seconds + 0.1 # Advance time

        mock_sleep.side_effect = side_effect_sleep

        # Second call
        limiter.wait_if_needed()

        assert mock_sleep.called
        assert len(limiter.call_times) == 1 # Since we popped old one in reality or just appended?
        # In this logic:
        # Call 1: time=1000. call_times=[1000].
        # Call 2: time=1000. current_weight=1. max=1. limit reached.
        # wait_time = 60 - (1000 - 1000) + 0.1 = 60.1
        # sleep(60.1). time becomes 1060.2
        # Loop again. time=1060.2.
        # Remove calls older than 60s: 1060.2 - 1000 > 60. call_times pops 1000. call_times=[].
        # current_weight=0. < 1.
        # Append 1060.2. Return.

        assert len(limiter.call_times) == 1
        assert limiter.call_times[0] > 1000

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

    # sample has 10:00:00 and 10:01:00.
    # 10:00:00 is < start_dt (should be removed)
    # 10:01:00 is >= start_dt and < end_dt (kept)

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

    # First call returns some trades, second call returns empty (end)
    trades_batch = [
        {'id': '1', 'timestamp': 1672531200000, 'price': 100, 'amount': 1},
        {'id': '2', 'timestamp': 1672531210000, 'price': 100, 'amount': 1}
    ]

    mock_exchange.fetch_trades.side_effect = [trades_batch, []]

    # We also need to patch RateLimiter.wait_if_needed to avoid sleeps
    with patch('src.data_fetching.fetching.rate_limiter.wait_if_needed'):
        with patch('src.data_fetching.fetching.SYMBOL', 'BTC/USDT'):
            with patch('src.data_fetching.fetching.MAX_TRADES_PER_CALL', 2): # Force it to continue if len >= max
                 # But in the code: if len(trades) < MAX_TRADES_PER_CALL: break
                 # So if we set max to 2, and return 2, it continues.

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
