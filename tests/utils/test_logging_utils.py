"""Tests for src.utils.logging_utils."""

import pytest
import pandas as pd
import numpy as np
from src.utils.logging_utils import log_series_summary, log_split_summary, save_plot

class TestLoggingUtils:
    def test_log_series_summary(self, mocker):
        """Test logging series summary."""
        mock_get_logger = mocker.patch("src.config_logging.get_logger")
        mock_logger = mock_get_logger.return_value
        series = pd.Series([1, 2, 3, np.nan])

        log_series_summary(series, "TestSeries")

        assert mock_logger.info.called
        # Check if some stats were logged
        calls = [c.args[0] for c in mock_logger.info.call_args_list]
        assert any("TestSeries summary" in c for c in calls)
        assert any("Mean" in c for c in calls)
        assert any("Null values" in c for c in calls)

    def test_log_split_summary(self, mocker):
        """Test logging split summary."""
        mock_get_logger = mocker.patch("src.config_logging.get_logger")
        mock_logger = mock_get_logger.return_value
        train = pd.DataFrame({"a": [1, 2]})
        test = pd.DataFrame({"a": [3]})

        log_split_summary(train, test, split_date="2024-01-01")

        calls = [c.args[0] for c in mock_logger.info.call_args_list]
        assert any("Train/Test split summary" in c for c in calls)
        assert any("Train size: 2" in c for c in calls)
        assert any("Split date: 2024-01-01" in c for c in calls)

    def test_save_plot(self, mocker, tmp_path):
        """Test saving plot."""
        mock_plt = mocker.patch("src.utils.logging_utils.plt")
        mock_figure = mocker.Mock()

        path = tmp_path / "plot.png"
        save_plot(mock_figure, path)

        assert path.parent.exists() # ensure_output_dir called
        mock_figure.savefig.assert_called_with(path, dpi=300, bbox_inches='tight')
        mock_plt.close.assert_called_with(mock_figure)
