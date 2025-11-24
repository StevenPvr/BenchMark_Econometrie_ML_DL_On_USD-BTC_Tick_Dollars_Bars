"""Tests for data_preparation.preparation module."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile

import pandas as pd  # type: ignore

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_preparation.preparation import compute_dollar_bars, prepare_dollar_bars


class TestDollarBars:
    """Test cases for dollar bar construction."""

    def test_compute_dollar_bars_basic(self):
        """Compute one dollar bar on a small synthetic set."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "price": [1, 1, 1, 1],
                "amount": [1, 1, 1, 1],
            }
        )
        bars = compute_dollar_bars(
            df,
            volume_col="amount",
            target_ticks_per_bar=2,
            ema_span=1,
            calibration_ticks=4,
        )

        assert len(bars) == 2  # 4 ticks -> 2 bars of ~2 ticks
        first_bar = bars.iloc[0]
        assert first_bar["n_ticks"] == 2
        assert abs(first_bar["cum_dollar_value"] - 2) < 1e-9
        assert abs(first_bar["threshold_used"] - 2) < 1e-9
        assert first_bar["bar_id"] == 0
        assert first_bar["timestamp_open"] == 1
        assert first_bar["timestamp_close"] == 2
        assert first_bar["duration_sec"] == (2 - 1) / 1000
        assert first_bar["datetime_close"] == pd.to_datetime(2, unit="ms", utc=True)

        second_bar = bars.iloc[1]
        assert second_bar["n_ticks"] == 2
        assert abs(second_bar["cum_dollar_value"] - 2) < 1e-9
        assert second_bar["bar_id"] == 1
        assert second_bar["timestamp_open"] == 3
        assert second_bar["timestamp_close"] == 4

    def test_prepare_dollar_bars_pipeline(self):
        """Full pipeline should write output files."""
        df = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "price": [100, 101, 100, 99],
                "amount": [1, 1, 1, 1],
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
            assert {"bar_id", "timestamp_close", "duration_sec"} <= set(bars.columns)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
