
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_bars():
    """Create a sample DataFrame mimicking dollar bars."""
    n_rows = 1000
    dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="min")

    # Generate synthetic price data (random walk)
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n_rows)
    price = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    high = price * (1 + np.abs(np.random.normal(0, 0.002, n_rows)))
    low = price * (1 - np.abs(np.random.normal(0, 0.002, n_rows)))
    open_ = price * (1 + np.random.normal(0, 0.001, n_rows))
    close = price
    volume = np.random.randint(100, 1000, n_rows).astype(float)

    # Buy/Sell volume (approx 50/50 split)
    buy_volume = volume * np.random.uniform(0.4, 0.6, n_rows)
    sell_volume = volume - buy_volume

    # Duration (seconds)
    duration = np.random.randint(10, 300, n_rows).astype(float)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "duration_sec": duration,
        "log_return": returns,
        "datetime": dates
    }, index=dates)

    # Add timestamp columns (ms)
    df["timestamp_open"] = dates.astype(np.int64) // 10**6
    df["timestamp_close"] = (dates + pd.to_timedelta(duration, unit="s")).astype(np.int64) // 10**6

    return df

@pytest.fixture
def aligned_data():
    """Create bars and ticks that are perfectly aligned."""
    # Define 2 bars
    # Bar 1: t=0 to t=100
    # Bar 2: t=100 to t=200

    bars_data = {
        "timestamp_open": [0, 100],
        "timestamp_close": [100, 200],
        "open": [100.0, 105.0],
        "high": [110.0, 115.0],
        "low": [90.0, 100.0],
        "close": [105.0, 110.0],
        "duration_sec": [100.0, 100.0]
    }
    df_bars = pd.DataFrame(bars_data)

    # Define ticks
    # Bar 1 ticks: 10, 50, 90
    # Bar 2 ticks: 110, 150, 190
    ticks_data = {
        "timestamp": [10, 50, 90, 110, 150, 190],
        "price": [100.0, 105.0, 102.0, 108.0, 112.0, 110.0],
        "quantity": [10.0, 20.0, 10.0, 15.0, 5.0, 20.0]
    }
    df_ticks = pd.DataFrame(ticks_data)

    return df_ticks, df_bars

@pytest.fixture
def sample_log_returns(sample_bars):
    return sample_bars["log_return"]

@pytest.fixture
def sample_trades():
    """Create a sample DataFrame mimicking raw trades for classification."""
    n_rows = 100
    dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="s")

    np.random.seed(42)
    price = 100 + np.cumsum(np.random.normal(0, 0.1, n_rows))
    volume = np.random.randint(1, 10, n_rows).astype(float)

    df = pd.DataFrame({
        "price": price,
        "quantity": volume,
        "timestamp": dates
    }, index=dates)

    return df
