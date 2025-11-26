"""Tests for data_preparation.preparation module (Dollar Bars - De Prado)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd # type: ignore[import-untyped]

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_preparation.preparation import compute_dollar_bars, prepare_dollar_bars


class TestDollarBars:
    """Test cases for Dollar Bars construction (De Prado methodology)."""

    def test_compute_dollar_bars_basic(self):
        """Compute dollar bars on a small synthetic set with adaptive threshold."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "price": [1.0, 1.0, 1.0, 1.0],
                "amount": [1.0, 1.0, 1.0, 1.0],
            }
        )
        bars = compute_dollar_bars(
            df,
            volume_col="amount",
            target_ticks_per_bar=2,
            ema_span=1,
            calibration_ticks=4,
        )

        # 4 ticks with dollar_value=1 each, threshold=2 -> 2 bars
        assert len(bars) == 2

        # Verify first bar structure
        first_bar = bars.iloc[0]
        assert first_bar["n_ticks"] == 2
        assert abs(first_bar["cum_dollar_value"] - 2) < 1e-9
        assert abs(first_bar["threshold_used"] - 2) < 1e-9
        assert first_bar["bar_id"] == 0
        assert first_bar["timestamp_open"] == 1
        assert first_bar["timestamp_close"] == 2
        assert first_bar["duration_sec"] == (2 - 1) / 1000
        assert first_bar["datetime_close"] == pd.to_datetime(2, unit="ms", utc=True)
        assert first_bar["datetime_open"] == pd.to_datetime(1, unit="ms", utc=True)

        # Verify second bar
        second_bar = bars.iloc[1]
        assert second_bar["n_ticks"] == 2
        assert abs(second_bar["cum_dollar_value"] - 2) < 1e-9
        assert second_bar["bar_id"] == 1
        assert second_bar["timestamp_open"] == 3
        assert second_bar["timestamp_close"] == 4

    def test_compute_dollar_bars_fixed_threshold(self):
        """Test with a fixed threshold value."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5, 6],
                "price": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "amount": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        # Each tick has dollar_value = 100 * 10 = 1000
        # With threshold=2500, we need 3 ticks per bar (3000 >= 2500)
        bars = compute_dollar_bars(df, threshold=2500.0, volume_col="amount")

        assert len(bars) == 2
        assert bars.iloc[0]["n_ticks"] == 3
        assert bars.iloc[0]["cum_dollar_value"] == 3000.0

    def test_compute_dollar_bars_ohlc(self):
        """Test OHLC computation within bars."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "price": [100.0, 105.0, 95.0, 102.0],  # Varying prices
                "amount": [10.0, 10.0, 10.0, 10.0],
            }
        )
        # threshold = 2000 means 2 ticks per bar (each tick ~1000 dollar value)
        bars = compute_dollar_bars(df, threshold=2000.0, volume_col="amount")

        # First bar: ticks 0,1 -> prices 100, 105
        first_bar = bars.iloc[0]
        assert first_bar["open"] == 100.0  # First tick price
        assert first_bar["high"] == 105.0  # Max price
        assert first_bar["low"] == 100.0  # Min price
        assert first_bar["close"] == 105.0  # Last tick price

        # Second bar: ticks 2,3 -> prices 95, 102
        second_bar = bars.iloc[1]
        assert second_bar["open"] == 95.0
        assert second_bar["high"] == 102.0
        assert second_bar["low"] == 95.0
        assert second_bar["close"] == 102.0

    def test_compute_dollar_bars_vwap(self):
        """Test VWAP (Volume Weighted Average Price) computation."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2],
                "price": [100.0, 200.0],
                "amount": [10.0, 10.0],
            }
        )
        # Single bar with both ticks
        # VWAP = (100*10 + 200*10) / (10 + 10) = 3000 / 20 = 150
        bars = compute_dollar_bars(df, threshold=5000.0, volume_col="amount")

        assert len(bars) == 1
        assert abs(bars.iloc[0]["vwap"] - 150.0) < 1e-9
        assert bars.iloc[0]["volume"] == 20.0
        assert bars.iloc[0]["cum_dollar_value"] == 3000.0

    def test_compute_dollar_bars_incomplete_final_bar(self):
        """Test that incomplete final bar is included (De Prado methodology)."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5],
                "price": [100.0] * 5,
                "amount": [10.0] * 5,
            }
        )
        # threshold=2500 -> 3 ticks per bar, so 5 ticks = 1 full bar + 2 ticks partial
        bars = compute_dollar_bars(df, threshold=2500.0, volume_col="amount")

        assert len(bars) == 2
        assert bars.iloc[0]["n_ticks"] == 3  # Full bar
        assert bars.iloc[1]["n_ticks"] == 2  # Partial bar (De Prado includes these)

    def test_compute_dollar_bars_empty_df(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "price", "amount"])  # type: ignore
        bars = compute_dollar_bars(df, threshold=1000.0, volume_col="amount")

        assert len(bars) == 0
        assert "bar_id" in bars.columns
        assert "vwap" in bars.columns

    def test_compute_dollar_bars_datetime_input(self):
        """Test with datetime timestamp column."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 00:00:00", "2024-01-01 00:00:01"]
                ),
                "price": [100.0, 100.0],
                "amount": [10.0, 10.0],
            }
        )
        bars = compute_dollar_bars(df, threshold=5000.0, volume_col="amount")

        assert len(bars) == 1
        assert bars.iloc[0]["datetime_close"].year == 2024

    def test_compute_dollar_bars_all_output_columns(self):
        """Verify all expected output columns are present (De Prado spec)."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "price": [100.0] * 4,
                "amount": [10.0] * 4,
            }
        )
        bars = compute_dollar_bars(df, threshold=2000.0, volume_col="amount")

        expected_columns = {
            "bar_id",
            "timestamp_open",
            "timestamp_close",
            "datetime_open",
            "datetime_close",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cum_dollar_value",
            "vwap",
            "n_ticks",
            "threshold_used",
            "duration_sec",
        }
        assert set(bars.columns) == expected_columns

    def test_prepare_dollar_bars_pipeline(self):
        """Full pipeline should write output files."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "price": [100.0, 101.0, 100.0, 99.0],
                "amount": [1.0, 1.0, 1.0, 1.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_parquet = Path(temp_dir) / "ticks.parquet"
            output_parquet = Path(temp_dir) / "bars.parquet"
            output_csv = Path(temp_dir) / "bars.csv"

            df.to_parquet(input_parquet, index=False)

            bars = prepare_dollar_bars(
                parquet_path=input_parquet,
                output_parquet=output_parquet,
                output_csv=output_csv,
                target_ticks_per_bar=2,
                ema_span=1,
                calibration_ticks=4,
            )

            assert output_parquet.exists()
            assert output_csv.exists()
            assert not bars.empty
            assert {"bar_id", "timestamp_close", "duration_sec", "vwap"} <= set(
                bars.columns
            )

    def test_large_transaction_single_bar(self):
        """Test that a single large transaction closes a bar immediately."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "price": [100.0, 100.0, 100.0],
                "amount": [100.0, 1.0, 1.0],  # First tick is huge
            }
        )
        # threshold=5000 but first tick has dollar_value=10000
        bars = compute_dollar_bars(df, threshold=5000.0, volume_col="amount")

        # First bar should contain only the first tick
        assert bars.iloc[0]["n_ticks"] == 1
        assert bars.iloc[0]["cum_dollar_value"] == 10000.0


class TestAdaptiveThreshold:
    """Tests for De Prado's adaptive threshold mechanism."""

    def test_threshold_adapts_over_time(self):
        """Verify threshold changes after each bar (EWMA update)."""
        # Create data with varying transaction sizes to trigger threshold adaptation
        # Key: the closing tick of each bar has large dollar value, causing EMA to grow
        np.random.seed(42)
        n_ticks = 200
        prices = np.concatenate([
            np.full(100, 100.0),   # Low price period
            np.full(100, 300.0),   # High price period (3x)
        ])
        # Varying amounts to create bars with different dollar values
        amounts = np.random.uniform(5, 20, n_ticks)

        df = pd.DataFrame({
            "timestamp": list(range(1, n_ticks + 1)),
            "price": prices,
            "amount": amounts,
        })

        # With adaptive=True (default), threshold should adapt
        bars = compute_dollar_bars(
            df,
            volume_col="amount",
            target_ticks_per_bar=10,
            ema_span=5,  # Fast adaptation
            calibration_ticks=20,
            adaptive=True,
        )

        # Check that threshold_used varies across bars
        thresholds = bars["threshold_used"]
        # With varying amounts and price jump, thresholds should change
        threshold_std = thresholds.std()
        assert threshold_std > 0, f"Adaptive threshold should vary, got std={threshold_std}"

        # Later bars (high price period) should have higher avg threshold
        if len(bars) >= 4:
            first_quarter_mean = thresholds[: len(thresholds) // 4].mean()
            last_quarter_mean = thresholds[-len(thresholds) // 4 :].mean()
            assert last_quarter_mean > first_quarter_mean, (
                f"Threshold should increase: first={first_quarter_mean:.0f}, "
                f"last={last_quarter_mean:.0f}"
            )

    def test_fixed_vs_adaptive_comparison(self):
        """Compare fixed and adaptive threshold behavior."""
        df = pd.DataFrame(
            {
                "timestamp": list(range(1, 51)),
                "price": [100.0] * 50,
                "amount": [10.0] * 50,
            }
        )

        # Fixed threshold
        bars_fixed = compute_dollar_bars(
            df, threshold=5000.0, volume_col="amount", adaptive=False
        )

        # Adaptive threshold (calibrated to similar initial value)
        bars_adaptive = compute_dollar_bars(
            df,
            volume_col="amount",
            target_ticks_per_bar=5,
            ema_span=10,
            calibration_ticks=10,
            adaptive=True,
        )

        # Fixed: all thresholds should be identical
        assert bars_fixed["threshold_used"].nunique() == 1

        # Both should produce bars (exact count may differ)
        assert len(bars_fixed) > 0
        assert len(bars_adaptive) > 0


class TestDollarBarsEdgeCases:
    """Edge cases and error handling tests."""

    def test_missing_columns_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({"timestamp": [1, 2], "price": [100.0, 100.0]})
        try:
            compute_dollar_bars(df, threshold=1000.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Missing required columns" in str(e)

    def test_threshold_too_high(self):
        """Test behavior when threshold is never reached."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2],
                "price": [100.0, 100.0],
                "amount": [1.0, 1.0],
            }
        )
        # threshold=1000000 is way higher than total dollar value (200)
        bars = compute_dollar_bars(df, threshold=1000000.0, volume_col="amount")

        # Should still return one partial bar
        assert len(bars) == 1
        assert bars.iloc[0]["n_ticks"] == 2


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
