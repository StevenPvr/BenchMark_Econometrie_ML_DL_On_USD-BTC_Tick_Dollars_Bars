"""Unit tests for src/data_cleaning/main.py."""

import sys
from unittest.mock import MagicMock

import pytest

from src.data_cleaning.main import main


class TestMain:
    """Tests for the main entry point."""

    def test_main_success(self, mocker):
        """Test successful execution of main."""
        mock_clean = mocker.patch("src.data_cleaning.main.clean_ticks_data")
        mock_setup_logging = mocker.patch("src.data_cleaning.main.setup_logging")
        mock_logger = mocker.patch("src.data_cleaning.main.logger")

        main()

        mock_setup_logging.assert_called_once()
        mock_clean.assert_called_once()
        mock_logger.info.assert_any_call("Data cleaning completed successfully")

    def test_main_known_error(self, mocker):
        """Test main handling known errors (e.g. ValueError)."""
        err = ValueError("Bad data")
        mock_clean = mocker.patch("src.data_cleaning.main.clean_ticks_data", side_effect=err)
        mock_setup_logging = mocker.patch("src.data_cleaning.main.setup_logging")
        mock_sys_exit = mocker.patch("sys.exit")
        mock_logger = mocker.patch("src.data_cleaning.main.logger")

        main()

        mock_clean.assert_called_once()
        mock_logger.error.assert_called_with("Data cleaning failed: %s", err)
        mock_sys_exit.assert_called_once_with(1)

    def test_main_unexpected_error(self, mocker):
        """Test main handling unexpected exceptions."""
        err = Exception("Boom")
        mock_clean = mocker.patch("src.data_cleaning.main.clean_ticks_data", side_effect=err)
        mock_setup_logging = mocker.patch("src.data_cleaning.main.setup_logging")
        mock_sys_exit = mocker.patch("sys.exit")
        mock_logger = mocker.patch("src.data_cleaning.main.logger")

        main()

        mock_clean.assert_called_once()
        mock_logger.exception.assert_called_with("Unexpected error during data cleaning: %s", err)
        mock_sys_exit.assert_called_once_with(1)
