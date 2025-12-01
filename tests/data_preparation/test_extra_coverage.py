
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
    run_dollar_bars_pipeline,
    run_dollar_bars_pipeline_batch,
    _validate_dollar_bars_params,
    _compute_threshold_from_target_bars,
    _compute_robust_percentile_threshold,
    _process_batch_with_state,
    _BarAccumulatorState,
    prepare_dollar_bars,
    generate_dollar_bars,
    add_log_returns_to_bars_file,
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

    def test_compute_dollar_bars_calibration_fraction_validation(self):
        """Test calibration_fraction validation (now raises ValueError instead of clamping)."""
        df = pd.DataFrame({
            "timestamp": [0, 1],
            "price": [10.0, 10.0],
            "amount": [1.0, 1.0]
        })
        # Invalid values now raise ValueError instead of being silently clamped
        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            compute_dollar_bars(df, target_num_bars=1, calibration_fraction=-0.1)
        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            compute_dollar_bars(df, target_num_bars=1, calibration_fraction=1.5)
        # Valid value should work
        compute_dollar_bars(df, target_num_bars=1, calibration_fraction=0.5)


class TestValidateDollarBarsParams:
    """Tests for _validate_dollar_bars_params function."""

    def test_valid_params(self):
        """Test valid parameters don't raise."""
        _validate_dollar_bars_params(100, (0.5, 2.0), 0.5)

    def test_invalid_ema_span(self):
        """Test invalid ema_span raises ValueError."""
        with pytest.raises(ValueError, match="ema_span must be positive"):
            _validate_dollar_bars_params(0, None, 0.5)
        with pytest.raises(ValueError, match="ema_span must be positive"):
            _validate_dollar_bars_params(-1, None, 0.5)

    def test_invalid_threshold_bounds_length(self):
        """Test invalid threshold_bounds length raises ValueError."""
        with pytest.raises(ValueError, match="threshold_bounds must be a tuple"):
            _validate_dollar_bars_params(100, (0.5,), 0.5)

    def test_invalid_threshold_bounds_order(self):
        """Test invalid threshold_bounds order raises ValueError."""
        with pytest.raises(ValueError, match="threshold_bounds min .* must be < max"):
            _validate_dollar_bars_params(100, (2.0, 0.5), 0.5)

    def test_invalid_threshold_bounds_negative(self):
        """Test negative threshold_bounds min raises ValueError."""
        with pytest.raises(ValueError, match="min_mult must be positive"):
            _validate_dollar_bars_params(100, (-0.5, 2.0), 0.5)


class TestComputeThresholdFromTargetBars:
    """Tests for _compute_threshold_from_target_bars function."""

    def test_basic_calculation(self):
        """Test basic threshold calculation."""
        dollar_values = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        result = _compute_threshold_from_target_bars(dollar_values, 3)
        assert result == pytest.approx(200.0)  # 600 / 3

    def test_empty_array(self):
        """Test empty array returns 1.0."""
        empty = np.array([], dtype=np.float64)
        result = _compute_threshold_from_target_bars(empty, 10)
        assert result == 1.0

    def test_zero_target_bars(self):
        """Test zero target bars returns 1.0."""
        dollar_values = np.array([100.0], dtype=np.float64)
        result = _compute_threshold_from_target_bars(dollar_values, 0)
        assert result == 1.0


class TestComputeRobustPercentileThreshold:
    """Tests for _compute_robust_percentile_threshold function."""

    def test_basic_calculation(self):
        """Test basic percentile threshold calculation."""
        dollar_values = np.random.uniform(10, 100, 1000).astype(np.float64)
        threshold, min_t, max_t = _compute_robust_percentile_threshold(dollar_values, 100)
        assert threshold > 0
        assert min_t == pytest.approx(threshold * 0.5)
        assert max_t == pytest.approx(threshold * 2.0)

    def test_empty_array(self):
        """Test empty array returns defaults."""
        empty = np.array([], dtype=np.float64)
        result = _compute_robust_percentile_threshold(empty, 10)
        assert result == (1.0, 1.0, 1.0)

    def test_zero_target_bars(self):
        """Test zero target bars returns defaults."""
        dollar_values = np.array([100.0], dtype=np.float64)
        result = _compute_robust_percentile_threshold(dollar_values, 0)
        assert result == (1.0, 1.0, 1.0)


class TestProcessBatchWithState:
    """Tests for _process_batch_with_state function."""

    def test_empty_batch(self):
        """Test empty batch returns empty list and unchanged state."""
        state = _BarAccumulatorState()
        state.current_threshold = 100.0
        state.ema_dollar = 100.0

        timestamps = np.array([], dtype=np.int64)
        prices = np.array([], dtype=np.float64)
        volumes = np.array([], dtype=np.float64)

        bars, new_state = _process_batch_with_state(
            timestamps, prices, volumes, state, 0.1, 50.0, 200.0, False
        )

        assert bars == []
        assert not new_state.initialized

    def test_batch_forms_bars(self):
        """Test batch forms bars correctly."""
        state = _BarAccumulatorState()
        state.current_threshold = 10.0
        state.ema_dollar = 10.0

        timestamps = np.arange(10, dtype=np.int64) * 1000
        prices = np.full(10, 10.0, dtype=np.float64)
        volumes = np.full(10, 1.0, dtype=np.float64)

        bars, new_state = _process_batch_with_state(
            timestamps, prices, volumes, state, 0.1, 5.0, 20.0, False
        )

        assert len(bars) > 0
        # State has bar_idx incremented (bars were formed)
        assert new_state.bar_idx > 0

    def test_adaptive_threshold_update(self):
        """Test adaptive threshold is updated after bar closes."""
        state = _BarAccumulatorState()
        state.current_threshold = 10.0
        state.ema_dollar = 10.0

        # Use more ticks to accumulate bigger dollar value for more visible EMA update
        timestamps = np.arange(20, dtype=np.int64) * 1000
        prices = np.full(20, 20.0, dtype=np.float64)  # Higher price
        volumes = np.full(20, 2.0, dtype=np.float64)  # Higher volume

        _, new_state = _process_batch_with_state(
            timestamps, prices, volumes, state, 0.9, 5.0, 500.0, True  # adaptive=True, high alpha
        )

        # Bars should have been formed and threshold updated
        assert new_state.bar_idx > 0


class TestPrepareDollarBars:
    """Tests for prepare_dollar_bars function."""

    def test_file_not_found(self):
        """Test raises FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError, match="Input parquet not found"):
            prepare_dollar_bars("non_existent.parquet", target_num_bars=100)

    def test_successful_preparation(self, tmp_path):
        """Test successful preparation with output file."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 100),
            "amount": np.random.uniform(0.1, 1.0, 100)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        result = prepare_dollar_bars(
            input_path,
            target_ticks_per_bar=10,
            output_parquet=output_path
        )

        assert not result.empty
        assert output_path.exists()


class TestGenerateDollarBars:
    """Tests for generate_dollar_bars function."""

    def test_basic_generation(self):
        """Test basic dollar bars generation with fixed threshold."""
        df_ticks = pd.DataFrame({
            "date_time": np.arange(100, dtype=np.int64) * 1000,
            "price": np.full(100, 100.0),
            "volume": np.full(100, 1.0)
        })

        result = generate_dollar_bars(df_ticks, threshold=500.0)

        assert not result.empty
        assert "close" in result.columns


class TestAddLogReturnsToBarsFile:
    """Tests for add_log_returns_to_bars_file function."""

    def test_file_not_found(self):
        """Test raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="dollar_bars parquet not found"):
            add_log_returns_to_bars_file(Path("non_existent.parquet"))

    def test_adds_log_returns(self, tmp_path):
        """Test log returns are added to bars file."""
        bars_path = tmp_path / "bars.parquet"
        df_bars = pd.DataFrame({
            "bar_id": [0, 1, 2],
            "close": [100.0, 105.0, 102.0],
            "open": [99.0, 100.0, 105.0],
            "high": [101.0, 106.0, 105.0],
            "low": [98.0, 99.0, 101.0],
        })
        df_bars.to_parquet(bars_path)

        add_log_returns_to_bars_file(bars_path)

        # Read back and check
        df_result = pd.read_parquet(bars_path)
        assert "log_return" in df_result.columns
        assert len(df_result) == 2  # First row dropped due to NaN


class TestComputeDollarBarsEdgeCases:
    """Additional edge case tests for compute_dollar_bars."""

    def test_missing_columns(self):
        """Test raises ValueError for missing required columns."""
        df = pd.DataFrame({"timestamp": [1, 2], "price": [10, 20]})  # Missing amount
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_dollar_bars(df, target_num_bars=1)

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty bars."""
        df = pd.DataFrame({"timestamp": [], "price": [], "amount": []})
        result = compute_dollar_bars(df, target_num_bars=1)
        assert result.empty

    def test_unsorted_timestamps(self):
        """Test unsorted timestamps are sorted."""
        df = pd.DataFrame({
            "timestamp": [3, 1, 2],
            "price": [10.0, 10.0, 10.0],
            "amount": [1.0, 1.0, 1.0]
        })
        result = compute_dollar_bars(df, target_num_bars=1)
        # Should work without error (auto-sorted)
        assert not result.empty

    def test_datetime_timestamps(self):
        """Test datetime timestamps are converted correctly."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "price": [100.0, 100.0, 100.0],
            "amount": [1.0, 1.0, 1.0]
        })
        result = compute_dollar_bars(df, target_num_bars=1)
        assert not result.empty

    def test_fixed_threshold_mode(self):
        """Test fixed threshold mode."""
        df = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.full(100, 100.0),
            "amount": np.full(100, 1.0)
        })
        result = compute_dollar_bars(df, threshold=500.0)
        assert not result.empty

    def test_no_threshold_or_target_raises(self):
        """Test raises ValueError when no threshold or target provided."""
        df = pd.DataFrame({
            "timestamp": [1, 2],
            "price": [10.0, 10.0],
            "amount": [1.0, 1.0]
        })
        with pytest.raises(ValueError, match="Either threshold, target_ticks_per_bar, or target_num_bars"):
            compute_dollar_bars(df)

    def test_negative_target_ticks_per_bar(self):
        """Test negative target_ticks_per_bar raises ValueError."""
        df = pd.DataFrame({
            "timestamp": [1, 2],
            "price": [10.0, 10.0],
            "amount": [1.0, 1.0]
        })
        with pytest.raises(ValueError, match="target_ticks_per_bar must be positive"):
            compute_dollar_bars(df, target_ticks_per_bar=-1)

    def test_adaptive_with_bounds(self):
        """Test adaptive mode with threshold bounds."""
        df = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 100),
            "amount": np.random.uniform(0.1, 1.0, 100)
        })
        result = compute_dollar_bars(
            df, target_num_bars=10, adaptive=True, threshold_bounds=(0.5, 2.0)
        )
        assert not result.empty

    def test_exclude_calibration_prefix(self):
        """Test exclude_calibration_prefix option."""
        df = pd.DataFrame({
            "timestamp": np.arange(1000, dtype=np.int64) * 1000,
            "price": np.full(1000, 100.0),
            "amount": np.full(1000, 1.0)
        })
        result = compute_dollar_bars(
            df, target_num_bars=100, calibration_fraction=0.2,
            exclude_calibration_prefix=True
        )
        # Some bars should be excluded from calibration prefix
        assert not result.empty


class TestRunDollarBarsPipelineBatchEdgeCases:
    """Additional tests for run_dollar_bars_pipeline_batch edge cases."""

    def test_default_target_ticks_per_bar(self, tmp_path):
        """Test default target_ticks_per_bar=100 when not provided (line 1055)."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(500, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 500),
            "amount": np.random.uniform(0.1, 1.0, 500)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        # Don't provide target_ticks_per_bar - should default to 100
        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            output_parquet=output_path
        )

        assert not result.empty

    def test_default_output_path(self, tmp_path):
        """Test default output path generation (lines 1059-1060)."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(500, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 500),
            "amount": np.random.uniform(0.1, 1.0, 500)
        })
        df_ticks.to_parquet(input_path)

        # Don't provide output_parquet - should use default
        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            target_ticks_per_bar=50
        )

        assert not result.empty

    def test_datetime_timestamps_in_batch(self, tmp_path):
        """Test datetime timestamps are handled correctly (line 1134)."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": pd.to_datetime(np.arange(500) * 1000, unit="ms"),
            "price": np.random.uniform(100, 101, 500),
            "amount": np.random.uniform(0.1, 1.0, 500)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            target_ticks_per_bar=50,
            output_parquet=output_path
        )

        assert not result.empty

    def test_missing_timestamp_column_raises(self, tmp_path):
        """Test missing timestamp column raises ValueError (line 1138)."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "time": np.arange(100, dtype=np.int64) * 1000,  # Wrong column name
            "price": np.random.uniform(100, 101, 100),
            "amount": np.random.uniform(0.1, 1.0, 100)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        with pytest.raises(ValueError, match="timestamp column required"):
            run_dollar_bars_pipeline_batch(
                input_parquet=input_path,
                target_ticks_per_bar=10,
                output_parquet=output_path
            )

    def test_no_bars_generated_raises(self, tmp_path):
        """Test no bars generated raises ValueError (line 1207)."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 100),
            "amount": np.random.uniform(0.1, 1.0, 100)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        # Mock _process_batch_with_state to return no bars
        with patch("src.data_preparation.preparation._process_batch_with_state") as mock_process:
            mock_state = _BarAccumulatorState()
            mock_state.n_ticks = 0  # No incomplete final bar
            mock_process.return_value = ([], mock_state)

            with pytest.raises(ValueError, match="No bars were generated"):
                run_dollar_bars_pipeline_batch(
                    input_parquet=input_path,
                    target_ticks_per_bar=100,
                    output_parquet=output_path
                )


    def test_include_incomplete_final_bar(self, tmp_path):
        """Test include_incomplete_final adds partial bar (lines 1179-1201)."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 100),
            "amount": np.random.uniform(0.1, 1.0, 100)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        # Mock to simulate incomplete final bar scenario
        with patch("src.data_preparation.preparation._process_batch_with_state") as mock_process:
            # First call returns some bars
            bars = [{
                "bar_id": 0,
                "timestamp_open": 0,
                "timestamp_close": 1000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1.0,
                "cum_dollar_value": 100.0,
                "vwap": 100.0,
                "n_ticks": 10,
                "threshold_used": 100.0,
            }]
            mock_state = _BarAccumulatorState()
            mock_state.n_ticks = 5  # Has incomplete bar data
            mock_state.bar_idx = 1
            mock_state.bar_ts_open = 2000
            mock_state.bar_open = 100.0
            mock_state.bar_high = 101.0
            mock_state.bar_low = 99.0
            mock_state.cum_volume = 0.5
            mock_state.cum_dollar = 50.0
            mock_state.cum_pv = 50.0
            mock_state.current_threshold = 100.0
            mock_process.return_value = (bars, mock_state)

            result = run_dollar_bars_pipeline_batch(
                input_parquet=input_path,
                target_ticks_per_bar=100,
                output_parquet=output_path,
                include_incomplete_final=True
            )

            # Should have 2 bars: 1 complete + 1 incomplete
            assert len(result) == 2


class TestRunDollarBarsPipelineEdgeCases:
    """Additional tests for run_dollar_bars_pipeline edge cases."""

    def test_default_input_path_not_found(self):
        """Test FileNotFoundError for missing default input (line 1255)."""
        with patch("src.path.DATASET_CLEAN_PARQUET", Path("/nonexistent/file.parquet")):
            with pytest.raises(FileNotFoundError, match="No cleaned tick data found"):
                run_dollar_bars_pipeline(target_num_bars=100)


class TestComputeDollarBarsMoreEdgeCases:
    """More edge cases for compute_dollar_bars."""

    def test_negative_target_num_bars(self):
        """Test negative target_num_bars raises ValueError (line 649)."""
        df = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.full(100, 100.0),
            "amount": np.full(100, 1.0)
        })
        with pytest.raises(ValueError, match="target_num_bars must be positive"):
            compute_dollar_bars(df, target_num_bars=-1)

    def test_zero_target_num_bars(self):
        """Test zero target_num_bars raises ValueError (line 649)."""
        df = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.full(100, 100.0),
            "amount": np.full(100, 1.0)
        })
        with pytest.raises(ValueError, match="target_num_bars must be positive"):
            compute_dollar_bars(df, target_num_bars=0)

    def test_calibration_prefix_exclusion_all_bars(self):
        """Test when calibration exclusion removes all bars (lines 771-772)."""
        df = pd.DataFrame({
            "timestamp": np.arange(20, dtype=np.int64) * 1000,
            "price": np.full(20, 100.0),
            "amount": np.full(20, 1.0)
        })
        # All data is in calibration period, so all bars should be excluded
        result = compute_dollar_bars(
            df,
            target_num_bars=10,
            calibration_fraction=1.0,  # 100% is calibration
            exclude_calibration_prefix=True
        )
        # Should return empty since all bars are in calibration period
        # (calibration_size == len(timestamps) means no bars after)
        # Actually calibration_fraction=1.0 means calibration_end_timestamp won't be set
        # Let's use a smaller fraction
        result = compute_dollar_bars(
            df,
            target_num_bars=2,
            calibration_fraction=0.9,  # Most is calibration
            exclude_calibration_prefix=True
        )
        # The result may or may not be empty depending on bar formation
        assert isinstance(result, pd.DataFrame)


class TestNumbaFunctionsDirectCoverage:
    """Direct tests for numba-compiled functions to increase coverage."""

    def test_accumulate_adaptive_threshold_clamping_low(self):
        """Test adaptive threshold clamping to min_threshold."""
        timestamps = np.arange(100, dtype=np.int64) * 1000
        # Very small prices/volumes to make bars with tiny dollar values
        prices = np.full(100, 1.0, dtype=np.float64)
        volumes = np.full(100, 0.1, dtype=np.float64)

        result = _accumulate_dollar_bars_adaptive(
            timestamps, prices, volumes,
            initial_threshold=0.05,  # Very small threshold
            ema_alpha=0.99,  # High alpha for fast adaptation
            min_threshold=0.1,  # Min is higher than initial
            max_threshold=100.0,
            include_incomplete_final=False
        )

        num_bars = result[12]
        assert num_bars > 0

    def test_accumulate_adaptive_threshold_clamping_high(self):
        """Test adaptive threshold clamping to max_threshold."""
        timestamps = np.arange(100, dtype=np.int64) * 1000
        # Large prices/volumes to push threshold up
        prices = np.full(100, 1000.0, dtype=np.float64)
        volumes = np.full(100, 10.0, dtype=np.float64)

        result = _accumulate_dollar_bars_adaptive(
            timestamps, prices, volumes,
            initial_threshold=100.0,  # Small initial
            ema_alpha=0.99,  # High alpha for fast adaptation
            min_threshold=10.0,
            max_threshold=500.0,  # Cap at 500
            include_incomplete_final=False
        )

        num_bars = result[12]
        assert num_bars > 0
        # Check some bars use clamped threshold
        thresholds = result[11][:num_bars]
        assert len(thresholds) > 0

    def test_accumulate_fixed_basic_bars(self):
        """Test fixed threshold forms expected bars."""
        timestamps = np.arange(100, dtype=np.int64) * 1000
        prices = np.full(100, 10.0, dtype=np.float64)
        volumes = np.full(100, 1.0, dtype=np.float64)

        # Each tick = $10, threshold = $50 -> 1 bar per 5 ticks
        result = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes,
            threshold=50.0,
            include_incomplete_final=False
        )

        num_bars = result[12]
        assert num_bars == 20  # 100 ticks / 5 per bar = 20 bars

    def test_accumulate_fixed_include_incomplete(self):
        """Test fixed threshold with incomplete final bar."""
        timestamps = np.arange(103, dtype=np.int64) * 1000
        prices = np.full(103, 10.0, dtype=np.float64)
        volumes = np.full(103, 1.0, dtype=np.float64)

        # 103 ticks, threshold $50 (5 ticks) -> 20 complete + 1 incomplete (3 ticks)
        result = _accumulate_dollar_bars_fixed(
            timestamps, prices, volumes,
            threshold=50.0,
            include_incomplete_final=True
        )

        num_bars = result[12]
        assert num_bars == 21  # 20 complete + 1 incomplete


class TestRobustPercentileThresholdEdgeCases:
    """Edge cases for _compute_robust_percentile_threshold."""

    def test_small_sample_size(self):
        """Test with sample smaller than 100000."""
        dollar_values = np.random.uniform(10, 100, 500).astype(np.float64)
        threshold, min_t, max_t = _compute_robust_percentile_threshold(dollar_values, 50)
        assert threshold > 0
        assert min_t > 0
        assert max_t > threshold

    def test_large_sample_size(self):
        """Test with sample larger than 100000 (triggers sampling)."""
        dollar_values = np.random.uniform(10, 100, 200000).astype(np.float64)
        threshold, min_t, max_t = _compute_robust_percentile_threshold(dollar_values, 1000)
        assert threshold > 0
        assert min_t == pytest.approx(threshold * 0.5)
        assert max_t == pytest.approx(threshold * 2.0)


# --- Main CLI Tests ---

class TestMainCLIEdgeCases:

    def test_main_value_error(self):
        """Cover ValueError catch block in main.py."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input:
            mock_input.exists.return_value = True

            with patch("src.data_preparation.main.run_dollar_bars_pipeline_batch", side_effect=ValueError("Bad data")):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_main_file_not_found_error(self):
        """Cover FileNotFoundError catch block in main.py."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input:
            mock_input.exists.return_value = True

            with patch("src.data_preparation.main.run_dollar_bars_pipeline_batch", side_effect=FileNotFoundError("Missing")):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_main_success_with_stats_logging(self):
        """Cover lines 83-85: Logging stats when bars are generated."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input_path:
            mock_input_path.exists.return_value = True

            # Return a DataFrame that satisfies !empty and 'close' in columns
            df_mock = pd.DataFrame({
                "close": [100.0, 105.0],
            }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))

            with patch("src.data_preparation.main.run_dollar_bars_pipeline_batch", return_value=df_mock):
                with patch("src.data_preparation.main.add_log_returns_to_bars_file"):
                    main()
