"""Unit tests for src/data_cleaning/main.py."""

import sys
from unittest.mock import patch

from src.data_cleaning.main import main


class TestMain:
    """Tests for the main entry point."""

    def test_main_success(self):
        """Test successful execution of main."""
        with patch("src.data_cleaning.main.clean_ticks_data") as mock_clean, \
             patch("src.data_cleaning.main.setup_logging") as mock_setup_logging, \
             patch("src.data_cleaning.main.logger") as mock_logger:

            main()

            mock_setup_logging.assert_called_once()
            mock_clean.assert_called_once()
            mock_logger.info.assert_any_call("Data cleaning completed successfully")

    def test_main_known_error(self):
        """Test main handling known errors (e.g. ValueError)."""
        err = ValueError("Bad data")
        with patch("src.data_cleaning.main.clean_ticks_data", side_effect=err) as mock_clean, \
             patch("src.data_cleaning.main.setup_logging") as mock_setup_logging, \
             patch("sys.exit") as mock_sys_exit, \
             patch("src.data_cleaning.main.logger") as mock_logger:

            main()

            mock_clean.assert_called_once()
            mock_logger.error.assert_called_with("Data cleaning failed: %s", err)
            mock_sys_exit.assert_called_once_with(1)
            mock_setup_logging.assert_called_once()

    def test_main_unexpected_error(self):
        """Test main handling unexpected exceptions."""
        err = Exception("Boom")
        with patch("src.data_cleaning.main.clean_ticks_data", side_effect=err) as mock_clean, \
             patch("src.data_cleaning.main.setup_logging") as mock_setup_logging, \
             patch("sys.exit") as mock_sys_exit, \
             patch("src.data_cleaning.main.logger") as mock_logger:

            main()

            mock_clean.assert_called_once()
            mock_logger.exception.assert_called_with("Unexpected error during data cleaning: %s", err)
            mock_sys_exit.assert_called_once_with(1)
            mock_setup_logging.assert_called_once()
