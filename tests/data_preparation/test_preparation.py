
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
    _validate_dollar_bars_params,
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


class TestParameterValidation:
    """Tests for parameter validation in dollar bars computation."""

    def test_validate_ema_span_positive(self):
        """ema_span must be positive when provided."""
        with pytest.raises(ValueError, match="ema_span must be positive"):
            _validate_dollar_bars_params(ema_span=0, threshold_bounds=None, calibration_fraction=0.5)

        with pytest.raises(ValueError, match="ema_span must be positive"):
            _validate_dollar_bars_params(ema_span=-10, threshold_bounds=None, calibration_fraction=0.5)

        # Valid case - should not raise
        _validate_dollar_bars_params(ema_span=100, threshold_bounds=None, calibration_fraction=0.5)
        _validate_dollar_bars_params(ema_span=None, threshold_bounds=None, calibration_fraction=0.5)

    def test_validate_threshold_bounds(self):
        """threshold_bounds must have min < max and both positive."""
        # Invalid: min >= max
        with pytest.raises(ValueError, match="min.*must be < max"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=(2.0, 1.0), calibration_fraction=0.5)

        with pytest.raises(ValueError, match="min.*must be < max"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=(1.0, 1.0), calibration_fraction=0.5)

        # Invalid: min_mult <= 0
        with pytest.raises(ValueError, match="min_mult must be positive"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=(0.0, 2.0), calibration_fraction=0.5)

        with pytest.raises(ValueError, match="min_mult must be positive"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=(-0.5, 2.0), calibration_fraction=0.5)

        # Valid case
        _validate_dollar_bars_params(ema_span=100, threshold_bounds=(0.5, 2.0), calibration_fraction=0.5)

    def test_validate_calibration_fraction(self):
        """calibration_fraction must be in (0, 1]."""
        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=None, calibration_fraction=0.0)

        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=None, calibration_fraction=-0.1)

        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            _validate_dollar_bars_params(ema_span=100, threshold_bounds=None, calibration_fraction=1.5)

        # Valid cases
        _validate_dollar_bars_params(ema_span=100, threshold_bounds=None, calibration_fraction=0.2)
        _validate_dollar_bars_params(ema_span=100, threshold_bounds=None, calibration_fraction=1.0)

    def test_compute_dollar_bars_validates_params(self, basic_df):
        """compute_dollar_bars should validate parameters."""
        with pytest.raises(ValueError, match="ema_span must be positive"):
            compute_dollar_bars(basic_df, target_num_bars=10, ema_span=0)

        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            compute_dollar_bars(basic_df, target_num_bars=10, calibration_fraction=0.0)


class TestDataLeakagePrevention:
    """Tests to verify no data leakage in dollar bars generation."""

    def test_calibration_uses_only_prefix(self):
        """Verify calibration uses ONLY first N% of ticks, not future data."""
        # Create dataset with strong trend: first 20% has low prices, last 80% has high prices
        n_ticks = 1000
        # First 20% (200 ticks): price=100, volume=1 -> dollar_value=100 each
        # Last 80% (800 ticks): price=1000, volume=1 -> dollar_value=1000 each
        prices = [100.0] * 200 + [1000.0] * 800
        volumes = [1.0] * n_ticks

        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": prices,
            "amount": volumes
        })

        # With calibration_fraction=0.2, only first 200 ticks should be used
        # Prefix total = 200 * 100 = 20,000
        # Target 10 bars in prefix -> threshold should be ~2,000 (not ~82,000 from full data)
        bars = compute_dollar_bars(
            df,
            target_num_bars=10,
            calibration_fraction=0.2,
            include_incomplete_final=False
        )

        # The threshold should be around 2000 (from prefix), not ~82000 (from full data)
        # If full data was used: total = 20000 + 800000 = 820000 -> threshold = 82000
        # With prefix only: total_prefix = 20000, target_prefix = 2 bars -> threshold = 10000
        # Wait, target_prefix = 10 * 0.2 = 2 bars, so threshold = 20000 / 2 = 10000
        # Actually the formula scales: target_bars_calibration = target * (prefix_size / total_size)
        # = 10 * (200/1000) = 2 bars, threshold = 20000 / 2 = 10000

        # Verify threshold is derived from prefix only (should be ~10000, not ~82000)
        threshold_used = bars["threshold_used"].iloc[0]
        assert threshold_used < 20000, f"Threshold {threshold_used} too high - may be using future data"

    def test_no_future_data_in_threshold(self):
        """Verify threshold is not affected by data after calibration period."""
        n_ticks = 500

        # Scenario 1: Uniform data throughout
        df_uniform = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": [100.0] * n_ticks,
            "amount": [1.0] * n_ticks
        })

        # Scenario 2: Same first 20%, but different last 80% (10x larger dollar values)
        df_diff_tail = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": [100.0] * 100 + [1000.0] * 400,
            "amount": [1.0] * n_ticks
        })

        bars_uniform = compute_dollar_bars(
            df_uniform,
            target_num_bars=10,
            calibration_fraction=0.2,
            include_incomplete_final=False
        )

        bars_diff = compute_dollar_bars(
            df_diff_tail,
            target_num_bars=10,
            calibration_fraction=0.2,
            include_incomplete_final=False
        )

        # Both should have the same initial threshold since calibration uses same prefix
        # (first 100 ticks have same dollar values in both datasets)
        threshold_uniform = bars_uniform["threshold_used"].iloc[0]
        threshold_diff = bars_diff["threshold_used"].iloc[0]

        assert threshold_uniform == threshold_diff, (
            f"Thresholds differ ({threshold_uniform} vs {threshold_diff}) - "
            "future data may be leaking into calibration"
        )

    def test_bars_temporal_monotonicity(self):
        """Verify output bars are strictly chronologically ordered."""
        # Create dataset with scrambled timestamps (should be sorted internally)
        n_ticks = 200
        timestamps = np.random.permutation(np.arange(n_ticks) * 1000)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "price": np.random.uniform(95, 105, n_ticks),
            "amount": np.random.uniform(0.5, 2.0, n_ticks)
        })

        bars = compute_dollar_bars(
            df,
            target_num_bars=20,
            include_incomplete_final=False
        )

        # Verify temporal ordering
        assert bars["datetime_open"].is_monotonic_increasing, "datetime_open not monotonic"
        assert bars["datetime_close"].is_monotonic_increasing, "datetime_close not monotonic"

        # Verify each bar's open <= close
        assert (bars["datetime_open"] <= bars["datetime_close"]).all(), (
            "Some bars have datetime_open > datetime_close"
        )

        # Verify bar continuity (close of bar i <= open of bar i+1)
        if len(bars) > 1:
            closes = bars["timestamp_close"].values[:-1]
            opens = bars["timestamp_open"].values[1:]
            assert (closes <= opens).all(), "Bars overlap in time"

    def test_exclude_calibration_prefix_removes_early_bars(self):
        """Verify exclude_calibration_prefix removes bars from calibration period."""
        n_ticks = 500
        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": [100.0] * n_ticks,
            "amount": [1.0] * n_ticks
        })

        # Without exclusion
        bars_all = compute_dollar_bars(
            df,
            target_num_bars=10,
            calibration_fraction=0.2,
            exclude_calibration_prefix=False,
            include_incomplete_final=False
        )

        # With exclusion
        bars_excluded = compute_dollar_bars(
            df,
            target_num_bars=10,
            calibration_fraction=0.2,
            exclude_calibration_prefix=True,
            include_incomplete_final=False
        )

        # Should have fewer bars when exclusion is enabled
        assert len(bars_excluded) < len(bars_all), (
            "exclude_calibration_prefix should reduce bar count"
        )

        # All remaining bars should have timestamp_close > calibration end timestamp
        # Calibration end = timestamp of tick at index (500 * 0.2 - 1) = 99 -> ts = 99000
        calibration_end_ts = int((n_ticks * 0.2 - 1) * 1000)
        assert (bars_excluded["timestamp_close"] > calibration_end_ts).all(), (
            "Some bars from calibration prefix remain after exclusion"
        )

        # bar_ids should be re-indexed starting from 0
        assert bars_excluded["bar_id"].iloc[0] == 0, "bar_ids not re-indexed after exclusion"
        assert list(bars_excluded["bar_id"]) == list(range(len(bars_excluded))), (
            "bar_ids not sequential after exclusion"
        )

    def test_incomplete_final_bar_excluded_by_default(self):
        """Verify incomplete final bar is excluded with default settings."""
        # Create dataset where last portion doesn't form a complete bar
        # 10 ticks, each with dollar value 100. Threshold ~333 (target 3 bars).
        # Full bars: 0-3 (400), 4-6 (300)... actually this is tricky
        # Let's be more explicit

        n_ticks = 10
        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": [100.0] * n_ticks,
            "amount": [1.0] * n_ticks
        })
        # Total dollar = 1000. Target 3 bars -> threshold = 333.33

        bars_default = compute_dollar_bars(
            df,
            target_num_bars=3,
            include_incomplete_final=False  # Default
        )

        bars_with_incomplete = compute_dollar_bars(
            df,
            target_num_bars=3,
            include_incomplete_final=True
        )

        # With incomplete=True should have more bars
        assert len(bars_with_incomplete) >= len(bars_default), (
            "include_incomplete_final=True should include at least as many bars"
        )


class TestAdaptiveModeVariations:
    """Tests for adaptive mode with different EMA alpha values."""

    @pytest.mark.parametrize("ema_span", [5, 20, 100])
    def test_adaptive_mode_different_spans(self, ema_span):
        """Test adaptive mode produces valid bars with different EMA spans."""
        # Create data with varying dollar values
        n_ticks = 500
        amounts = []
        for i in range(n_ticks):
            if (i // 50) % 2 == 0:
                amounts.append(1.0)
            else:
                amounts.append(5.0)

        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": [100.0] * n_ticks,
            "amount": amounts
        })

        bars = compute_dollar_bars(
            df,
            target_num_bars=20,
            adaptive=True,
            ema_span=ema_span,
            include_incomplete_final=False
        )

        # Should produce some bars
        assert len(bars) > 0, f"No bars generated with ema_span={ema_span}"
        # Verify expected columns exist
        assert "threshold_used" in bars.columns
        # Verify thresholds are positive
        assert (bars["threshold_used"] > 0).all()

    def test_adaptive_mode_shows_variation(self):
        """Test that adaptive mode with fast EMA shows threshold variation."""
        # This test uses a spike pattern that forces threshold variation
        n_ticks = 100
        # First bars use small amounts, then a spike, then small again
        prices = [100.0] * n_ticks
        amounts = [1.0] * 10 + [100.0] + [1.0] * (n_ticks - 11)

        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": prices,
            "amount": amounts
        })

        bars = compute_dollar_bars(
            df,
            target_num_bars=10,
            adaptive=True,
            ema_span=5,  # Fast adaptation
            include_incomplete_final=False
        )

        # With a spike and fast EMA, thresholds should vary
        thresholds = bars["threshold_used"].values
        if len(bars) >= 2:
            # After the spike bar, threshold should have changed
            assert len(np.unique(thresholds)) > 1, (
                "Thresholds don't vary despite spike in data"
            )

    def test_adaptive_with_bounds(self):
        """Test that threshold bounds are respected in adaptive mode."""
        n_ticks = 300
        # Create data that would cause large threshold swings
        amounts = [1.0] * 100 + [50.0] * 50 + [1.0] * 150

        df = pd.DataFrame({
            "timestamp": np.arange(n_ticks) * 1000,
            "price": [100.0] * n_ticks,
            "amount": amounts
        })

        # Compute initial threshold estimate
        total_dollar = sum(100.0 * a for a in amounts)
        target_bars = 15
        expected_threshold = total_dollar / target_bars

        # Set tight bounds
        min_mult, max_mult = 0.8, 1.2
        min_threshold = expected_threshold * min_mult
        max_threshold = expected_threshold * max_mult

        bars = compute_dollar_bars(
            df,
            target_num_bars=target_bars,
            adaptive=True,
            ema_span=10,  # Fast adaptation
            threshold_bounds=(min_mult, max_mult),
            include_incomplete_final=False
        )

        # All thresholds should be within bounds (approximately)
        # Note: initial threshold might be slightly different due to calibration
        thresholds = bars["threshold_used"].values
        initial_threshold = thresholds[0]
        actual_min = initial_threshold * min_mult
        actual_max = initial_threshold * max_mult

        # Check bounds are approximately respected (allow small tolerance)
        assert thresholds.min() >= actual_min * 0.99, (
            f"Threshold {thresholds.min()} below min bound {actual_min}"
        )
        assert thresholds.max() <= actual_max * 1.01, (
            f"Threshold {thresholds.max()} above max bound {actual_max}"
        )
