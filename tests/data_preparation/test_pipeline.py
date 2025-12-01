
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys
import os

# Need to ensure src is in path for imports to work correctly
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_preparation.preparation import (
    run_dollar_bars_pipeline,
    run_dollar_bars_pipeline_batch
)

@pytest.fixture
def mock_logger():
    with patch("src.data_preparation.preparation.logger") as mock:
        yield mock

@pytest.fixture
def mock_prepare_dollar_bars():
    with patch("src.data_preparation.preparation.prepare_dollar_bars") as mock:
        yield mock

@pytest.fixture
def sample_bars_df():
    return pd.DataFrame({
        "datetime_close": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-01 10:05:00"]),
        "close": [100.0, 101.0]
    })

class TestRunDollarBarsPipeline:

    def test_run_dollar_bars_pipeline_defaults(self, mock_prepare_dollar_bars, sample_bars_df, mock_logger, tmp_path):
        """Test pipeline with default paths (mocking them)."""
        mock_prepare_dollar_bars.return_value = sample_bars_df

        # Use real temporary files instead of mocking Path.exists
        input_path = tmp_path / "input.parquet"
        input_path.touch()
        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline(
            target_num_bars=100,
            input_parquet=input_path,
            output_parquet=output_path,
            threshold=50.0
        )

        assert result.equals(sample_bars_df)
        mock_prepare_dollar_bars.assert_called_once_with(
            parquet_path=input_path,
            target_ticks_per_bar=None,
            target_num_bars=100,
            output_parquet=output_path,
            threshold=50.0,
            adaptive=False,
            threshold_bounds=None,
            calibration_fraction=1.0,
            include_incomplete_final=False,
            exclude_calibration_prefix=False,
        )

    def test_run_dollar_bars_pipeline_missing_input(self):
        """Test pipeline raises FileNotFoundError when input is missing."""
        with pytest.raises(FileNotFoundError):
             run_dollar_bars_pipeline(
                target_num_bars=100,
                input_parquet="non_existent_file.parquet"
            )

    def test_run_dollar_bars_pipeline_default_paths_logic(self, mock_prepare_dollar_bars, sample_bars_df, tmp_path):
        """Test that default paths are resolved correctly."""
        mock_prepare_dollar_bars.return_value = sample_bars_df

        # Create dummy file for default input
        mock_input = tmp_path / "mock_input.parquet"
        mock_input.touch()

        mock_output_dir = tmp_path / "mock_output"
        mock_output = mock_output_dir / "bars.parquet"

        with patch("src.path.DATASET_CLEAN_PARQUET", mock_input), \
             patch("src.data_preparation.preparation.DOLLAR_IMBALANCE_BARS_PARQUET", mock_output):

             run_dollar_bars_pipeline(target_num_bars=100)

             args, kwargs = mock_prepare_dollar_bars.call_args
             assert kwargs['parquet_path'] == mock_input

             # Check output path robustly
             output_path = str(kwargs['output_parquet'])
             expected_prefix = str(mock_output_dir / "dollar_imbalance_bars_")
             assert output_path.startswith(expected_prefix)

    def test_run_dollar_bars_pipeline_with_target_ticks_per_bar(self, mock_prepare_dollar_bars, sample_bars_df, tmp_path):
        """Test pipeline with target_ticks_per_bar parameter."""
        mock_prepare_dollar_bars.return_value = sample_bars_df

        input_path = tmp_path / "input.parquet"
        input_path.touch()
        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline(
            target_ticks_per_bar=100,
            input_parquet=input_path,
            output_parquet=output_path
        )

        assert result.equals(sample_bars_df)
        args, kwargs = mock_prepare_dollar_bars.call_args
        assert kwargs['target_ticks_per_bar'] == 100


class TestRunDollarBarsPipelineBatch:

    def test_batch_processing_success(self, tmp_path):
        """Test batch processing with valid parquet file."""
        # Create a valid parquet file with tick data
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(1000, dtype=np.int64) * 1000,  # ms timestamps
            "price": np.random.uniform(100, 101, 1000),
            "amount": np.random.uniform(0.1, 1.0, 1000)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            target_ticks_per_bar=50,
            output_parquet=output_path,
            batch_size=500
        )

        assert not result.empty
        assert output_path.exists()
        assert "bar_id" in result.columns
        assert "close" in result.columns

    def test_batch_processing_missing_file(self):
        """Test raises FileNotFoundError if input file missing."""
        with pytest.raises(FileNotFoundError, match="Input parquet not found"):
            run_dollar_bars_pipeline_batch(
                input_parquet="non_existent.parquet",
                target_ticks_per_bar=100
            )

    def test_batch_processing_adaptive(self, tmp_path):
        """Test batch processing with adaptive threshold."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(500, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 500),
            "amount": np.random.uniform(0.1, 1.0, 500)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            target_ticks_per_bar=25,
            output_parquet=output_path,
            adaptive=True
        )

        assert not result.empty

    def test_batch_processing_with_threshold_bounds(self, tmp_path):
        """Test batch processing with threshold bounds."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(500, dtype=np.int64) * 1000,
            "price": np.random.uniform(100, 101, 500),
            "amount": np.random.uniform(0.1, 1.0, 500)
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            target_ticks_per_bar=25,
            output_parquet=output_path,
            adaptive=True,
            threshold_bounds=(0.5, 2.0)
        )

        assert not result.empty

    def test_batch_processing_include_incomplete_final(self, tmp_path):
        """Test batch processing with include_incomplete_final=True."""
        input_path = tmp_path / "input.parquet"
        df_ticks = pd.DataFrame({
            "timestamp": np.arange(100, dtype=np.int64) * 1000,
            "price": np.full(100, 100.0),
            "amount": np.full(100, 0.01)  # Small volumes to ensure incomplete final bar
        })
        df_ticks.to_parquet(input_path)

        output_path = tmp_path / "output.parquet"

        result = run_dollar_bars_pipeline_batch(
            input_parquet=input_path,
            target_ticks_per_bar=200,  # More than available ticks to force incomplete
            output_parquet=output_path,
            include_incomplete_final=True
        )

        # Should have at least one bar (the incomplete final one)
        assert len(result) >= 1
