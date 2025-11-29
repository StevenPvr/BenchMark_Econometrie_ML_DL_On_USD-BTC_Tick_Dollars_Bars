from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
# Find project root by looking for .git, pyproject.toml, or setup.py
_project_root = _script_dir.parent
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import shutil
from unittest.mock import MagicMock

# Import constant to ensure consistency
from src.analyse_features.config import TARGET_COLUMN

@pytest.fixture
def sample_df():
    """Create a simple random DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, n_rows),
        "feature_2": np.random.normal(0, 1, n_rows),
        "feature_3": np.random.uniform(0, 1, n_rows),
        TARGET_COLUMN: np.random.normal(0, 1, n_rows)
    })
    return df

@pytest.fixture
def correlated_df():
    """Create a DataFrame with correlated features."""
    np.random.seed(42)
    n_rows = 100
    x = np.random.normal(0, 1, n_rows)
    df = pd.DataFrame({
        "feat_a": x,
        "feat_b": x * 0.9 + np.random.normal(0, 0.1, n_rows), # Highly correlated
        "feat_c": np.random.normal(0, 1, n_rows),             # Uncorrelated
        TARGET_COLUMN: x * 0.5 + np.random.normal(0, 0.5, n_rows)
    })
    return df

@pytest.fixture
def non_stationary_df():
    """Create a DataFrame with non-stationary features."""
    np.random.seed(42)
    n_rows = 200
    # Random walk
    rw = np.cumsum(np.random.normal(0, 1, n_rows))
    df = pd.DataFrame({
        "stationary": np.random.normal(0, 1, n_rows),
        "non_stationary": rw,
        TARGET_COLUMN: np.random.normal(0, 1, n_rows)
    })
    return df

@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to provide a temporary directory."""
    return tmp_path

@pytest.fixture
def sample_bars():
    """Create a sample dollar bars DataFrame for feature testing."""
    np.random.seed(42)
    n = 100
    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1h")
    
    # Generate realistic OHLCV data
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    
    # OHLC from prices with some variation
    opens = prices
    highs = prices + np.abs(np.random.randn(n) * 0.3)
    lows = prices - np.abs(np.random.randn(n) * 0.3)
    closes = prices + np.random.randn(n) * 0.2
    volumes = np.random.uniform(0.1, 1.0, n)
    
    df = pd.DataFrame({
        "datetime": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    df = df.set_index("datetime")
    
    # Compute log returns after DataFrame creation
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    
    # Add buy_volume and sell_volume for VPIN and volume imbalance tests
    # Split volume randomly between buy and sell
    buy_ratio = np.random.uniform(0.3, 0.7, n)
    df["buy_volume"] = volumes * buy_ratio
    df["sell_volume"] = volumes * (1 - buy_ratio)
    
    # Add datetime as column for temporal tests
    df["datetime"] = df.index
    
    return df

@pytest.fixture
def sample_trades():
    """Create a sample trades DataFrame for trade classification testing."""
    np.random.seed(42)
    n = 100
    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1min")
    
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.uniform(0.01, 0.1, n)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "amount": volumes,
    })
    return df

@pytest.fixture
def aligned_data():
    """Create aligned tick and bar data for microstructure volatility and volume imbalance testing.
    
    Returns tuple of (df_ticks, df_bars) for testing microstructure features.
    """
    np.random.seed(42)
    
    # Create bars
    n_bars = 6
    bar_timestamps = pd.date_range(start="2023-01-01", periods=n_bars, freq="1h")
    bar_prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    df_bars = pd.DataFrame({
        "timestamp_open": bar_timestamps,
        "timestamp_close": bar_timestamps + pd.Timedelta(hours=1),
        "open": bar_prices,
        "high": bar_prices + np.abs(np.random.randn(n_bars) * 0.3),
        "low": bar_prices - np.abs(np.random.randn(n_bars) * 0.3),
        "close": bar_prices + np.random.randn(n_bars) * 0.2,
    })
    
    # Create ticks (3 ticks per bar for microstructure tests)
    # For volume imbalance, we need specific pattern: first bar has 3 ticks with known pattern
    n_ticks = n_bars * 3
    tick_timestamps = []
    tick_prices = []
    tick_quantities = []
    
    for i, bar_start in enumerate(bar_timestamps):
        if i == 0:
            # First bar: specific pattern for volume imbalance test
            # Tick 1: price 100, qty 10 (first trade, assume buy)
            tick_timestamps.append(bar_start)
            tick_prices.append(100.0)
            tick_quantities.append(10.0)
            # Tick 2: price 105 (up), qty 20 (buy)
            tick_timestamps.append(bar_start + pd.Timedelta(minutes=20))
            tick_prices.append(105.0)
            tick_quantities.append(20.0)
            # Tick 3: price 102 (down), qty 10 (sell)
            tick_timestamps.append(bar_start + pd.Timedelta(minutes=40))
            tick_prices.append(102.0)
            tick_quantities.append(10.0)
        else:
            # Other bars: random pattern
            for j in range(3):
                tick_timestamps.append(bar_start + pd.Timedelta(minutes=j*20))
                tick_prices.append(bar_prices[i] + np.random.randn() * 0.1)
                tick_quantities.append(np.random.uniform(5.0, 15.0))
    
    df_ticks = pd.DataFrame({
        "timestamp": tick_timestamps,
        "price": tick_prices,
        "quantity": tick_quantities,
        "amount": tick_quantities,  # Alias for compatibility
    })
    
    # Sort by timestamp to ensure chronological order for tick rule
    df_ticks = df_ticks.sort_values("timestamp").reset_index(drop=True)
    
    return df_ticks, df_bars
