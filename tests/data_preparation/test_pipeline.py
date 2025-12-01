
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
            output_parquet=output_path,
            target_num_bars=100,
            threshold=50.0,
            adaptive=False,
            threshold_bounds=None,
            calibration_fraction=1.0,
            include_incomplete_final=False,  # Default changed to False
            exclude_calibration_prefix=False,  # New parameter
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


class TestRunDollarBarsPipelineBatch:

    def test_batch_processing_success(self, mock_prepare_dollar_bars, sample_bars_df, mock_logger, tmp_path):
        """Test batch processing iterates over files and consolidates results."""
        input_dir = tmp_path / "input_parts"
        input_dir.mkdir()

        # Create dummy parquet files
        (input_dir / "part1.parquet").touch()
        (input_dir / "part2.parquet").touch()

        output_path = tmp_path / "output.parquet"

        # Mock pandas read_parquet for checking file length logging
        mock_prepare_dollar_bars.return_value = sample_bars_df

        # Mock pyarrow.parquet.ParquetWriter and pd.read_parquet

        with patch("pyarrow.parquet.ParquetWriter") as mock_writer_cls, \
             patch("src.data_preparation.preparation.pd.read_parquet") as mock_read_parquet:

            mock_writer_instance = mock_writer_cls.return_value

            consolidated_df = pd.concat([sample_bars_df, sample_bars_df])
            mock_read_parquet.side_effect = [
                pd.DataFrame({"dummy": [1]*10}), # part 1 log
                pd.DataFrame({"dummy": [1]*10}), # part 2 log
                consolidated_df # final read for consolidation
            ]

            result = run_dollar_bars_pipeline_batch(
                input_dir=input_dir,
                target_num_bars=100,
                output_parquet=output_path
            )

            assert mock_prepare_dollar_bars.call_count == 2
            assert mock_writer_cls.call_count == 1
            assert mock_writer_instance.write_table.call_count == 2
            mock_writer_instance.close.assert_called_once()

            assert result.equals(consolidated_df.sort_values("datetime_close").drop_duplicates(subset=["datetime_close"]).reset_index(drop=True))

    def test_batch_processing_no_files(self, tmp_path):
        """Test raises ValueError if no parquet files found."""
        input_dir = tmp_path / "empty_dir"
        input_dir.mkdir()

        with pytest.raises(ValueError, match="No parquet files found"):
            run_dollar_bars_pipeline_batch(input_dir=input_dir, target_num_bars=100)

    def test_batch_processing_missing_dir(self):
        """Test raises FileNotFoundError if directory missing."""
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            run_dollar_bars_pipeline_batch(input_dir="non_existent", target_num_bars=100)

    def test_batch_processing_no_bars_generated(self, mock_prepare_dollar_bars, tmp_path):
        """Test raises ValueError if all files yield empty bars."""
        input_dir = tmp_path / "input_parts"
        input_dir.mkdir()
        (input_dir / "part1.parquet").touch()

        mock_prepare_dollar_bars.return_value = pd.DataFrame() # Empty result

        # We also need to mock read_parquet because it is called to log ticks count
        with patch("src.data_preparation.preparation.pd.read_parquet", return_value=pd.DataFrame({"a":[1]})):
             with pytest.raises(ValueError, match="No bars were generated"):
                run_dollar_bars_pipeline_batch(
                    input_dir=input_dir,
                    target_num_bars=100
                )
