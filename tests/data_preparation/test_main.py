
import pytest
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_preparation.main import main
from src.path import DATASET_CLEAN_PARQUET, DOLLAR_BARS_PARQUET, DOLLAR_BARS_CSV

@pytest.fixture
def mock_logger():
    with patch("src.data_preparation.main.logger") as mock:
        yield mock

@pytest.fixture
def mock_pipeline():
    with patch("src.data_preparation.main.run_dollar_bars_pipeline") as mock:
        yield mock

@pytest.fixture
def mock_batch_pipeline():
    with patch("src.data_preparation.main.run_dollar_bars_pipeline_batch") as mock:
        yield mock

@pytest.fixture
def mock_add_log_returns():
    with patch("src.data_preparation.main.add_log_returns_to_bars_file") as mock:
        yield mock

class TestMainCLI:

    def test_main_single_file_success(self, mock_pipeline, mock_add_log_returns, mock_logger):
        """Test main execution flow for single file input."""
        # Mock DATASET_CLEAN_PARQUET existence and is_dir
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input_path:
            mock_input_path.exists.return_value = True
            mock_input_path.is_dir.return_value = False

            mock_pipeline.return_value = MagicMock(spec=pd.DataFrame)
            mock_pipeline.return_value.empty = False
            mock_pipeline.return_value.__len__.return_value = 100

            # Run main
            main()

            # Verify pipeline call
            mock_pipeline.assert_called_once()
            _, kwargs = mock_pipeline.call_args
            assert kwargs['target_num_bars'] == 500_000

            # Verify log returns added
            mock_add_log_returns.assert_called_once()

    def test_main_directory_batch_success(self, mock_batch_pipeline, mock_add_log_returns, mock_logger):
        """Test main execution flow for directory input (batch processing)."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input_path:
            mock_input_path.exists.return_value = True
            mock_input_path.is_dir.return_value = True

            mock_batch_pipeline.return_value = MagicMock(spec=pd.DataFrame)
            mock_batch_pipeline.return_value.empty = False
            mock_batch_pipeline.return_value.__len__.return_value = 100

            main()

            mock_batch_pipeline.assert_called_once()
            mock_add_log_returns.assert_called_once()

    def test_main_input_not_found(self, mock_logger):
        """Test exit when input not found."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input_path:
            mock_input_path.exists.return_value = False

            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 1
            mock_logger.error.assert_any_call("Input data not found: %s", mock_input_path)

    def test_main_pipeline_error(self, mock_pipeline, mock_logger):
        """Test error handling during pipeline execution."""
        with patch("src.data_preparation.main.DATASET_CLEAN_PARQUET") as mock_input_path:
            mock_input_path.exists.return_value = True
            mock_input_path.is_dir.return_value = False

            mock_pipeline.side_effect = Exception("Pipeline failed")

            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 1
            mock_logger.exception.assert_called()
