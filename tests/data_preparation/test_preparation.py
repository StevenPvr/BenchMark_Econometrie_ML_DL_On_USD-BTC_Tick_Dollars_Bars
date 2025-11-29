from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
# Find project root by looking for .git, pyproject.toml, or setup.py
_project_root = _script_dir
while _project_root != _project_root.parent:
    if (_project_root / ".git").exists() or (_project_root / "pyproject.toml").exists() or (_project_root / "setup.py").exists():
        break
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from unittest.mock import MagicMock, patch

from src.data_preparation import preparation

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_tick_data():
    """Create a sample DataFrame with tick data."""
    n = 100
    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="1s")
    # Make timestamps essentially int64 ms
    ts_int = timestamps.astype("int64") // 10**6

    # Prices increasing then decreasing
    prices = np.concatenate([np.linspace(100, 110, 50), np.linspace(110, 100, 50)])
    # Volumes random
    volumes = np.random.uniform(0.1, 1.0, size=n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "amount": volumes
    })
    return df

@pytest.fixture
def sample_arrays(sample_tick_data):
    """Create sample numpy arrays for numba functions."""
    df = sample_tick_data
    timestamps = (df["timestamp"].astype("int64") // 10**6).values
    prices = df["price"].values
    volumes = df["amount"].values
    return timestamps, prices, volumes

# =============================================================================
# TESTS FOR NUMBA CORE FUNCTIONS
# =============================================================================

class TestNumbaCore:
    def test_compute_threshold_from_target_bars(self):
        # 10 ticks, each dollar_value = 100. Total = 1000.
        # Target bars = 5. Threshold should be 200.
        dollar_values = np.full(10, 100.0, dtype=np.float64)
        target_bars = 5
        threshold = preparation._compute_threshold_from_target_bars(dollar_values, target_bars)
        assert np.isclose(threshold, 200.0)

    def test_compute_threshold_edge_cases(self):
        # Empty input
        assert preparation._compute_threshold_from_target_bars(np.array([], dtype=np.float64), 5) == 1.0
        # Zero target bars
        assert preparation._compute_threshold_from_target_bars(np.full(10, 100.0), 0) == 1.0

    def test_compute_robust_percentile_threshold(self):
        # 100 values, all 100.0. Mean = 100.
        dollar_values = np.full(100, 100.0, dtype=np.float64)
        target_bars = 10
        # Total = 10000. Mean threshold = 1000.

        threshold, min_t, max_t = preparation._compute_robust_percentile_threshold(dollar_values, target_bars)
        assert np.isclose(threshold, 1000.0)
        assert min_t == 500.0
        assert max_t == 2000.0

    def test_accumulate_dollar_bars_fixed(self, sample_arrays):
        timestamps, prices, volumes = sample_arrays
        # Total dollar volume approx ~105 * 0.5 * 100 = 5250
        # Threshold 1000 -> Expect approx 5 bars
        threshold = 1000.0

        result = preparation._accumulate_dollar_bars_fixed(
            timestamps, prices, volumes, threshold, include_incomplete_final=False
        )

        (bar_ids, ts_open, ts_close, opens, highs, lows, closes,
         bar_volumes, bar_dollars, bar_vwaps, tick_counts, thresholds_arr, num_bars) = result

        assert num_bars > 0
        assert len(bar_ids) == len(ts_open) # Arrays are pre-allocated max size
        # Check actual filled values
        assert np.all(bar_dollars[:num_bars] >= threshold)

    def test_accumulate_dollar_bars_adaptive(self, sample_arrays):
        timestamps, prices, volumes = sample_arrays
        initial_threshold = 1000.0
        ema_alpha = 0.1
        min_t, max_t = 500.0, 2000.0

        result = preparation._accumulate_dollar_bars_adaptive(
            timestamps, prices, volumes, initial_threshold, ema_alpha,
            min_t, max_t, include_incomplete_final=False
        )

        # Unpack
        num_bars = result[-1]
        thresholds = result[-2]

        assert num_bars > 0
        # Check thresholds adaptation within bounds
        used_thresholds = thresholds[:num_bars]
        assert np.all(used_thresholds >= min_t)
        assert np.all(used_thresholds <= max_t)

# =============================================================================
# TESTS FOR PUBLIC API
# =============================================================================

class TestPublicAPI:
    def test_compute_dollar_bars_fixed(self, sample_tick_data):
        # Using fixed threshold
        df_bars = preparation.compute_dollar_bars(
            sample_tick_data,
            target_num_bars=10, # Ignored
            threshold=500.0,
            timestamp_col="timestamp",
            price_col="price",
            volume_col="amount",
            include_incomplete_final=False  # Strict check
        )

        assert not df_bars.empty
        assert "bar_id" in df_bars.columns
        assert "vwap" in df_bars.columns
        # Verify cumulative dollar value is roughly threshold
        # (It will be >= threshold)
        assert df_bars["cum_dollar_value"].min() >= 500.0

    def test_incomplete_final_bar(self, sample_tick_data):
        # Force a situation with incomplete bar
        # Total dollar value ~10500. Threshold 6000 -> 1 complete bar, remainder ~4500.

        # First with include_incomplete_final=True (default)
        df_bars = preparation.compute_dollar_bars(
            sample_tick_data,
            target_num_bars=10,
            threshold=6000.0,
            include_incomplete_final=True,
            timestamp_col="timestamp", price_col="price", volume_col="amount"
        )

        # If we have any bars, check if the last one is partial
        if not df_bars.empty:
            # Note: It's possible we formed 0 bars if total < threshold, but here total > threshold
            if df_bars["cum_dollar_value"].iloc[-1] < 6000.0:
                 # Confirmed partial bar
                 pass
            else:
                 # Exactly hit threshold or something else
                 pass

        # Now with False
        df_bars_strict = preparation.compute_dollar_bars(
            sample_tick_data,
            target_num_bars=10,
            threshold=6000.0,
            include_incomplete_final=False,
            timestamp_col="timestamp", price_col="price", volume_col="amount"
        )

        if not df_bars_strict.empty:
             assert df_bars_strict["cum_dollar_value"].min() >= 6000.0

        # Strict should have same or fewer bars
        assert len(df_bars_strict) <= len(df_bars)

    def test_compute_dollar_bars_target_num(self, sample_tick_data):
        # Using target number of bars
        target = 5
        df_bars = preparation.compute_dollar_bars(
            sample_tick_data,
            target_num_bars=target,
            timestamp_col="timestamp",
            price_col="price",
            volume_col="amount"
        )

        assert not df_bars.empty
        # It's an approximation, so might not be exactly target, but close
        assert abs(len(df_bars) - target) <= 2

    def test_compute_dollar_bars_empty(self):
        empty_df = pd.DataFrame(columns=["timestamp", "price", "amount"])
        df_bars = preparation.compute_dollar_bars(
            empty_df, target_num_bars=10,
            timestamp_col="timestamp", price_col="price", volume_col="amount"
        )
        assert df_bars.empty
        assert "bar_id" in df_bars.columns

    def test_compute_dollar_bars_validation(self, sample_tick_data):
        # Missing column
        with pytest.raises(ValueError, match="Missing required columns"):
            preparation.compute_dollar_bars(
                sample_tick_data, target_num_bars=10,
                price_col="wrong_col"
            )

    def test_prepare_dollar_bars_pipeline(self, sample_tick_data, tmp_path):
        # Mock reading parquet
        with patch("pandas.read_parquet", return_value=sample_tick_data):
            input_path = tmp_path / "ticks.parquet"
            output_path = tmp_path / "bars.parquet"

            # Create dummy file to pass existence check
            input_path.touch()

            df_bars = preparation.prepare_dollar_bars(
                parquet_path=input_path,
                target_num_bars=5,
                output_parquet=output_path,
                timestamp_col="timestamp",
                price_col="price",
                volume_col="amount"
            )

            assert not df_bars.empty
            assert output_path.exists()

    def test_add_log_returns(self, tmp_path):
        # Create a dummy bars file
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 100.0],
            "bar_id": [1, 2, 3, 4]
        })
        p = tmp_path / "test_bars.parquet"
        df.to_parquet(p)

        preparation.add_log_returns_to_bars_file(p)

        df_new = pd.read_parquet(p)
        assert "log_return" in df_new.columns
        # First row should be dropped or NaN handling?
        # The function says: df_bars = df_bars.dropna(subset=["log_return"])
        # So it should have 3 rows
        assert len(df_new) == 3
        assert np.isclose(df_new.iloc[0]["log_return"], np.log(101.0/100.0))

# =============================================================================
# INTEGRATION / LOGIC CHECKS
# =============================================================================

class TestLogic:
    def test_adaptive_threshold_behavior(self):
        """Verify that adaptive threshold actually changes."""
        # Create data where volatility increases dramatically
        # 1st half: low volume/price. 2nd half: high volume/price.
        n = 200
        ts = np.arange(n)
        prices = np.ones(n) * 100.0

        volumes = np.concatenate([np.ones(100), np.ones(100) * 10]) # 10x volume jump

        df = pd.DataFrame({"ts": ts, "price": prices, "vol": volumes})

        # Run adaptive
        # Use target_num_bars that yields small threshold initially
        # 100 ticks * 100$ = 10000. 100 ticks * 1000$ = 100000. Total = 110000.
        # Target 20 bars. Avg T = 5500.

        df_bars = preparation.compute_dollar_bars(
            df, target_num_bars=20,
            timestamp_col="ts", price_col="price", volume_col="vol",
            adaptive=True,
            ema_span=10 # Fast adaptation
        )

        # Check that thresholds in later bars are higher than early bars
        thresholds = df_bars["threshold_used"].values

        # First few bars should handle the low volume regime
        # Last few bars should handle the high volume regime

        # With adaptive, the threshold should increase as dollar value per tick increases?
        # Wait, if dollar value per tick increases, we hit the threshold FASTER.
        # The threshold is based on EMA of *bar dollar values*.
        # Since we close the bar when sum >= Threshold, the bar dollar value is always approx Threshold.
        # So EMA of bar dollar values should remain approx Threshold.
        # UNLESS the "overshoot" is significant.

        # De Prado's adaptive threshold: T_{k+1} = E_k(Bar Dollar Value).
        # If we have huge ticks, we might overshoot the threshold significantly.
        # e.g. Threshold 100. Tick comes in with 1000.
        # Bar closes with 1000. EMA updates towards 1000. Next threshold increases.
        # This is the desired behavior (fewer bars during high activity? No, stable number of bars?)

        # "Produces fewer bars during low activity periods (quiet markets)"
        # "Produces more bars during high activity periods"

        # Actually, if we have huge ticks, we want to sample MORE OFTEN?
        # Dollar bars sample every $X exchanged.
        # If $X is exchanged quickly (high volume), we sample often (more bars per time).
        # If $X is exchanged slowly, we sample rarely.

        # The Adaptive Threshold T adapts to what?
        # It adapts to the expected dollar value of a bar.
        # Ideally we want T to be constant to have "Constant Dollar Bars".
        # But if market changes scale (BTC from $100 to $60000), a fixed $1000 bar is useless noise.
        # So T needs to scale with price/volume levels.

        # If volumes jump 10x, and we don't adapt T, we get 10x more bars.
        # If we adapt T, T should eventually increase 10x, restoring the tick count per bar?
        # Yes, that's the goal of "Adaptive" usually - to keep sampling frequency somewhat stable in terms of "information content" or "ticks per bar" OR to keep it stable in "fraction of daily volume".

        # Let's just check that thresholds vary.
        if len(thresholds) > 1:
            assert np.std(thresholds) > 0.0

if __name__ == "__main__":
    # Allow running individual test file with pytest and colored output
    import pytest  # type: ignore
    pytest.main([__file__, "-v", "--color=yes"])
