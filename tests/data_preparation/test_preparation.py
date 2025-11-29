
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.data_preparation.preparation import (
    _compute_threshold_from_target_bars,
    _compute_robust_percentile_threshold,
    _accumulate_dollar_bars_fixed,
    _accumulate_dollar_bars_adaptive,
    compute_dollar_bars,
    prepare_dollar_bars,
    add_log_returns_to_bars_file,
    _create_empty_bars_df
)

# Mock src.config_logging to avoid console spam or errors
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("src.data_preparation.preparation.logger") as mock:
        yield mock

@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "timestamp": np.arange(100) * 1000, # seconds
        "price": [100.0] * 100,
        "amount": [1.0] * 100
    })

class TestNumbaFunctions:
    """Tests for the internal numba-optimized functions."""

    def test_compute_threshold_from_target_bars(self):
        # Case 1: Normal calculation
        # total_dollar = 1000, target = 10 -> threshold = 100
        dollar_values = np.array([100.0] * 10, dtype=np.float64)
        threshold = _compute_threshold_from_target_bars(dollar_values, 10)
        assert threshold == 100.0

        # Case 2: Empty input
        assert _compute_threshold_from_target_bars(np.array([], dtype=np.float64), 10) == 1.0

        # Case 3: Zero target
        assert _compute_threshold_from_target_bars(dollar_values, 0) == 1.0

    def test_compute_robust_percentile_threshold(self):
        # Create data with an outlier
        # 9 values of 10.0, 1 value of 1000.0
        # Sum = 1090. Mean = 109.
        # Target 10 bars -> Mean threshold = 109.
        dollar_values = np.array([10.0]*9 + [1000.0], dtype=np.float64)

        # Function returns (threshold, min, max)
        # threshold is simple mean = 109.0
        # min is 0.5 * threshold = 54.5
        # max is 2.0 * threshold = 218.0
        t, t_min, t_max = _compute_robust_percentile_threshold(dollar_values, 10, 50.0)

        assert t == 109.0
        assert t_min == 54.5
        assert t_max == 218.0

        # Empty input
        assert _compute_robust_percentile_threshold(np.array([], dtype=np.float64), 10) == (1.0, 1.0, 1.0)

    def test_accumulate_dollar_bars_fixed(self):
        # Timestamps: 0, 1, 2...
        timestamps = np.arange(10, dtype=np.int64)
        # Prices constant 10
        prices = np.full(10, 10.0, dtype=np.float64)
        # Volumes constant 1
        volumes = np.full(10, 1.0, dtype=np.float64)
        # Dollar value per tick = 10.
        # Threshold = 25.
        # Bar 1: tick 0 (10), tick 1 (20), tick 2 (30) -> closes at tick 2.
        # Bar 2: tick 3 (10), tick 4 (20), tick 5 (30) -> closes at tick 5.
        # Bar 3: tick 6 (10), tick 7 (20), tick 8 (30) -> closes at tick 8.
        # Tick 9 remains (10).

        result = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes, threshold=25.0, include_incomplete_final=False
        )

        (bar_ids, ts_open, ts_close, opens, highs, lows, closes,
         bar_volumes, bar_dollars, bar_vwaps, tick_counts, thresholds, num_bars) = result

        assert num_bars == 3
        # Check volumes: 3 ticks * 1.0 = 3.0
        assert np.all(bar_volumes[:3] == 3.0)
        # Check dollars: 3 ticks * 10.0 = 30.0
        assert np.all(bar_dollars[:3] == 30.0)
        # Check tick counts
        assert np.all(tick_counts[:3] == 3)

        # With include_incomplete_final=True
        result_incomplete = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes, threshold=25.0, include_incomplete_final=True
        )
        num_bars_inc = result_incomplete[-1]
        assert num_bars_inc == 4
        # Last bar has 1 tick
        assert result_incomplete[10][3] == 1

    def test_accumulate_dollar_bars_adaptive(self):
        # Similar setup
        timestamps = np.arange(10, dtype=np.int64)
        prices = np.full(10, 10.0, dtype=np.float64)
        volumes = np.full(10, 1.0, dtype=np.float64)

        # Initial threshold 25.
        # EMA alpha 1.0 (instant update) -> threshold becomes last bar's dollar value
        # Min/Max bounds wide enough to not interfere

        # Bar 1 closes at tick 2 (cum=30). New threshold = 1.0*30 + 0 = 30.
        # Bar 2 closes at tick 5 (cum=30). New threshold = 30.

        result = _accumulate_dollar_bars_adaptive(
            timestamps, prices, volumes,
            initial_threshold=25.0, ema_alpha=1.0,
            min_threshold=1.0, max_threshold=1000.0,
            include_incomplete_final=False
        )

        thresholds_out = result[11]
        num_bars = result[12]

        assert num_bars == 3
        # First bar used initial threshold 25
        assert thresholds_out[0] == 25.0
        # Second bar used updated threshold 30 (from first bar's dollar volume)
        assert thresholds_out[1] == 30.0


class TestComputeDollarBars:
    """Tests for the public compute_dollar_bars function."""

    def test_validation(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_dollar_bars(df, target_num_bars=10, timestamp_col="ts", price_col="p", volume_col="v")

    def test_empty_df(self):
        df = pd.DataFrame({
            "timestamp": [],
            "price": [],
            "amount": []
        })
        bars = compute_dollar_bars(df, target_num_bars=10)
        assert bars.empty
        assert "bar_id" in bars.columns

    def test_fixed_threshold_mode(self, basic_df):
        # Total dollar volume = 100 * 100 * 1 = 10000
        # Threshold = 1000 -> Should get 10 bars

        bars = compute_dollar_bars(
            basic_df,
            target_num_bars=0, # Ignored
            threshold=1000.0,
            include_incomplete_final=False
        )

        assert len(bars) == 10
        assert np.allclose(bars["cum_dollar_value"], 1000.0)
        assert np.allclose(bars["volume"], 10.0) # 10 ticks * 1.0
        assert np.allclose(bars["n_ticks"], 10)

    def test_target_num_bars_mode(self, basic_df):
        # Total dollar volume = 10000.
        # Target 5 bars -> Threshold should be 2000.

        bars = compute_dollar_bars(
            basic_df,
            target_num_bars=5,
            include_incomplete_final=False
        )

        assert len(bars) == 5
        # Each bar should have roughly 2000 dollar volume
        # Since data is uniform, it should be exact
        assert np.allclose(bars["cum_dollar_value"], 2000.0)
        assert np.allclose(bars["threshold_used"], 2000.0)

    def test_adaptive_mode(self):
        # Create data that forces overshoot to trigger adaptation
        # We need enough data to form at least 2 bars

        # Initial calibration:
        # Ticks 0-9: 10 * 100 = 1000.
        # Tick 10: 10000.
        # Ticks 11-100: 90 * 100 = 9000.

        n_ticks = 100
        prices = [100.0] * n_ticks
        amounts = [1.0] * 10 + [100.0] + [1.0] * (n_ticks - 11)

        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": prices,
            "amount": amounts
        })

        # Total dollar approx 1000 + 10000 + 8900 = 19900.
        # Target 10 bars -> T_init = 1990.

        bars = compute_dollar_bars(
            df,
            target_num_bars=10,
            adaptive=True,
            ema_span=5,
            include_incomplete_final=False
        )

        # Bar 1 should close at tick 10 (cum > 1990).
        # Its dollar volume will be ~11000.
        # New threshold will adapt towards 11000.
        # Subsequent bars will use this higher threshold.

        thresholds = bars["threshold_used"].values
        # We expect at least 2 bars with different thresholds
        assert len(bars) >= 2
        assert len(np.unique(thresholds)) > 1

    def test_datetime_conversion(self):
        # Test with datetime objects instead of int64
        df = pd.DataFrame({
            "timestamp": pd.date_range("2021-01-01", periods=100, freq="s"),
            "price": [100.0] * 100,
            "amount": [1.0] * 100
        })

        bars = compute_dollar_bars(
            df,
            target_num_bars=5
        )

        assert len(bars) == 5
        assert pd.api.types.is_datetime64_any_dtype(bars["datetime_open"])


class TestPrepareDollarBars:
    """Tests for the prepare_dollar_bars pipeline function."""

    def test_end_to_end(self, tmp_path, basic_df):
        # Create input parquet
        input_path = tmp_path / "ticks.parquet"
        basic_df.to_parquet(input_path)

        output_path = tmp_path / "bars.parquet"

        # Run preparation
        df_bars = prepare_dollar_bars(
            parquet_path=input_path,
            output_parquet=output_path,
            target_num_bars=5,
            timestamp_col="timestamp",
            price_col="price",
            volume_col="amount"
        )

        assert len(df_bars) == 5
        assert output_path.exists()

        # Verify saved file
        df_loaded = pd.read_parquet(output_path)
        assert len(df_loaded) == 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            prepare_dollar_bars(Path("non_existent.parquet"), target_num_bars=10)


class TestAddLogReturns:
    """Tests for add_log_returns_to_bars_file."""

    def test_add_log_returns(self, tmp_path):
        # Create dummy bars
        bars_path = tmp_path / "bars.parquet"
        df = pd.DataFrame({
            "close": [100.0, 110.0, 100.0, 90.0]
        })
        df.to_parquet(bars_path)

        add_log_returns_to_bars_file(bars_path)

        df_new = pd.read_parquet(bars_path)
        assert "log_return" in df_new.columns
        # First row should be dropped because of NaN log return
        assert len(df_new) == 3

        # Check values
        # log(110/100) approx 0.0953
        expected = np.log(110.0/100.0)
        assert np.isclose(df_new.iloc[0]["log_return"], expected)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            add_log_returns_to_bars_file(Path("non_existent.parquet"))
