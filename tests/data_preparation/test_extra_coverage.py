
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

from src.data_preparation.preparation import (
    _accumulate_dollar_bars_adaptive,
    _accumulate_dollar_bars_fixed,
    compute_dollar_bars,
    run_dollar_bars_pipeline_batch
)
from src.data_preparation.main import main

# --- Preparation Tests ---

class TestPreparationEdgeCases:
    """Extra tests to cover edge cases in preparation.py."""

    def test_adaptive_with_incomplete_final(self):
        """Cover lines 310-322 in preparation.py: include_incomplete_final for adaptive."""
        timestamps = np.arange(10, dtype=np.int64)
        prices = np.full(10, 10.0, dtype=np.float64)
        volumes = np.full(10, 1.0, dtype=np.float64)

        result = _accumulate_dollar_bars_adaptive(
            timestamps, prices, volumes,
            initial_threshold=25.0, ema_alpha=1.0,
            min_threshold=1.0, max_threshold=1000.0,
            include_incomplete_final=True
        )

        num_bars = result[12]
        assert num_bars == 4
        assert result[10][3] == 1

    def test_adaptive_varying_prices_high_low(self):
        """Cover lines 263, 265 in preparation.py: High/Low updates in adaptive loop."""
        timestamps = np.arange(3, dtype=np.int64)
        prices = np.array([10.0, 12.0, 8.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        result = _accumulate_dollar_bars_adaptive(
            timestamps, prices, volumes,
            initial_threshold=50.0, ema_alpha=1.0,
            min_threshold=1.0, max_threshold=1000.0,
            include_incomplete_final=True
        )

        num_bars = result[12]
        assert num_bars == 1
        highs = result[4]
        lows = result[5]

        assert highs[0] == 12.0
        assert lows[0] == 8.0

    def test_adaptive_empty_input(self):
        """Cover lines 208-210: Empty input check."""
        empty_i64 = np.array([], dtype=np.int64)
        empty_f64 = np.array([], dtype=np.float64)

        result = _accumulate_dollar_bars_adaptive(
            empty_i64, empty_f64, empty_f64,
            10.0, 0.5, 5.0, 20.0, True
        )
        assert result[12] == 0

    def test_fixed_varying_prices_high_low(self):
        """Cover lines 403, 405 in preparation.py: High/Low updates in fixed loop."""
        timestamps = np.arange(3, dtype=np.int64)
        prices = np.array([10.0, 12.0, 8.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Threshold 50
        result = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes,
            threshold=50.0,
            include_incomplete_final=True
        )

        num_bars = result[12]
        assert num_bars == 1
        highs = result[4]
        lows = result[5]
        assert highs[0] == 12.0
        assert lows[0] == 8.0

    def test_fixed_empty_input(self):
        """Cover lines 359-361: Empty input check for fixed."""
        empty_i64 = np.array([], dtype=np.int64)
        empty_f64 = np.array([], dtype=np.float64)

        result = _accumulate_dollar_bars_fixed(
            empty_i64, empty_f64, empty_f64,
            10.0, True
        )
        assert result[12] == 0

    def test_compute_dollar_bars_invalid_prices_volumes(self):
        """Cover lines 538-539, 606-608: invalid prices warning and adaptive logging."""
        df = pd.DataFrame({
            "timestamp": [0, 1],
            "price": [-10.0, 10.0],
            "amount": [1.0, -1.0]
        })
        compute_dollar_bars(df, target_num_bars=1, adaptive=True)

    def test_compute_dollar_bars_calibration_fraction_clamping(self):
        """Cover lines 552-553, 556-557: calibration_fraction clamping."""
        df = pd.DataFrame({
            "timestamp": [0, 1],
            "price": [10.0, 10.0],
            "amount": [1.0, 1.0]
        })
        compute_dollar_bars(df, target_num_bars=1, calibration_fraction=-0.1)
        compute_dollar_bars(df, target_num_bars=1, calibration_fraction=1.5)

    def test_run_dollar_bars_pipeline_batch_exception(self, tmp_path):
        """Cover lines 908-910: Exception in batch loop."""
        input_dir = tmp_path / "input_batch"
        input_dir.mkdir()
        p1 = input_dir / "part1.parquet"
        p1.touch()

        with patch("src.data_preparation.preparation.pd.read_parquet", side_effect=Exception("Read failed")):
             with patch("src.data_preparation.preparation.prepare_dollar_bars", side_effect=Exception("Processing failed")):
                 with pytest.raises(ValueError, match="No bars were generated"):
                     run_dollar_bars_pipeline_batch(
                         input_dir=input_dir,
                         target_num_bars=10
                     )


# --- Main CLI Tests ---

class TestMainCLIEdgeCases:

    def test_main_value_error(self):
        """Cover ValueError catch block in main.py."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input:
            mock_input.exists.return_value = True
            mock_input.is_dir.return_value = False

            with patch("src.data_preparation.main.run_dollar_bars_pipeline", side_effect=ValueError("Bad data")):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_main_file_not_found_error(self):
        """Cover FileNotFoundError catch block in main.py."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input:
            mock_input.exists.return_value = True
            mock_input.is_dir.return_value = False

            with patch("src.data_preparation.main.run_dollar_bars_pipeline", side_effect=FileNotFoundError("Missing")):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_main_success_with_stats_logging(self):
        """Cover lines 83-85: Logging stats when bars are generated."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input_path:
            mock_input_path.exists.return_value = True
            mock_input_path.is_dir.return_value = False

            # Return a DataFrame that satisfies !empty and 'close' in columns
            df_mock = pd.DataFrame({
                "close": [100.0, 105.0],
                "datetime_close": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")]
            }).set_index("datetime_close")

            with patch("src.data_preparation.main.run_dollar_bars_pipeline", return_value=df_mock):
                with patch("src.data_preparation.main.add_log_returns_to_bars_file"):
                    main()
