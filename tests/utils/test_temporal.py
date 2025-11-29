"""Tests for src.utils.temporal."""

import pytest
import pandas as pd
from src.utils.temporal import (
    compute_timeseries_split_indices,
    log_split_dates,
    validate_temporal_order_series,
    validate_temporal_split,
)

class TestTemporal:
    def test_compute_timeseries_split_indices(self):
        """Test split indices computation."""
        series = pd.Series(range(100))

        # Ratio 0.8
        train_end, test_start = compute_timeseries_split_indices(series, train_ratio=0.8)
        assert train_end == 80
        assert test_start == 80

        # Min train size
        train_end, test_start = compute_timeseries_split_indices(series, min_train_size=50)
        assert train_end == 50
        assert test_start == 50

    def test_log_split_dates(self, mocker):
        """Test logging split dates."""
        mock_get_logger = mocker.patch("src.config_logging.get_logger")
        mock_logger = mock_get_logger.return_value

        dates = pd.date_range("2024-01-01", periods=10)
        train = pd.DataFrame(index=dates[:8])
        test = pd.DataFrame(index=dates[8:])

        log_split_dates(train, test)

        calls = [c.args[0] for c in mock_logger.info.call_args_list]
        assert any("Train date range" in c for c in calls)
        assert any("Test date range" in c for c in calls)

    def test_validate_temporal_order_series_monotonic(self):
        """Test valid monotonic series."""
        dates = pd.date_range("2024-01-01", periods=5)
        series = pd.Series(range(5), index=dates)
        assert validate_temporal_order_series(series)

    def test_validate_temporal_order_series_not_monotonic(self):
        """Test invalid monotonic series."""
        dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-02")
        ]
        series = pd.Series(range(3), index=dates)
        assert not validate_temporal_order_series(series)

    def test_validate_temporal_split_valid(self):
        """Test valid split."""
        dates = pd.date_range("2024-01-01", periods=10)
        train = pd.Series(range(5), index=dates[:5])
        test = pd.Series(range(5), index=dates[5:])
        assert validate_temporal_split(train, test)

    def test_validate_temporal_split_invalid(self):
        """Test invalid split (overlap/order)."""
        dates = pd.date_range("2024-01-01", periods=10)
        train = pd.Series(range(6), index=dates[:6]) # Ends at index 5
        test = pd.Series(range(5), index=dates[4:9]) # Starts at index 4
        # train max is index 5. test min is index 4.
        # train_end (5) < test_start (4) is False.
        assert not validate_temporal_split(train, test)
