"""Tests for src.utils.datetime_utils."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.utils.datetime_utils import (
    normalize_timestamp_to_datetime,
    parse_date_value,
    extract_date_range,
    filter_by_date_range,
    format_dates_to_string,
)

class TestNormalizeTimestampToDatetime:
    def test_timezone_naive(self):
        """Test with timezone naive timestamp."""
        ts = pd.Timestamp("2024-01-01 12:00:00")
        result = normalize_timestamp_to_datetime(ts)
        assert result == ts
        assert result.tzinfo is None

    def test_timezone_aware(self):
        """Test with timezone aware timestamp."""
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        result = normalize_timestamp_to_datetime(ts)
        assert result.tzinfo is None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12

class TestParseDateValue:
    def test_parse_string(self):
        """Test parsing string date."""
        result = parse_date_value("2024-01-01")
        assert result == pd.Timestamp("2024-01-01")

    def test_parse_datetime(self):
        """Test parsing python datetime."""
        dt = datetime(2024, 1, 1, 12, 0)
        result = parse_date_value(dt)
        assert result == pd.Timestamp("2024-01-01 12:00:00")

    def test_parse_timestamp_naive(self):
        """Test parsing naive Timestamp."""
        ts = pd.Timestamp("2024-01-01")
        result = parse_date_value(ts)
        assert result == ts

    def test_parse_timestamp_aware(self):
        """Test parsing aware Timestamp."""
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        result = parse_date_value(ts)
        assert result.tzinfo is None
        assert result.year == 2024

    def test_parse_series(self):
        """Test parsing Series (extract first element)."""
        s = pd.Series([pd.Timestamp("2024-01-01")])
        result = parse_date_value(s)
        assert result == pd.Timestamp("2024-01-01")

    def test_invalid_input_allow_none(self):
        """Test invalid input with allow_none=True."""
        assert parse_date_value("invalid", allow_none=True) is None

    def test_invalid_input_no_allow_none(self, mocker):
        """Test invalid input with allow_none=False."""
        mock_logger = mocker.patch("src.utils.datetime_utils.get_logger")
        result = parse_date_value("invalid", allow_none=False)
        assert result is None
        mock_logger.return_value.warning.assert_called()

    def test_nat_input(self):
        """Test NaT input."""
        assert parse_date_value(pd.NaT, allow_none=True) is None

    def test_parse_scalar_exception(self, mocker):
        """Test exception in parsing scalar."""
        # Force ValueError during pd.Timestamp()
        # "invalid" string usually raises ValueError, but let's be sure
        mock_logger = mocker.patch("src.utils.datetime_utils.get_logger")

        # Test with allow_none=False
        result = parse_date_value("invalid_date_string_xyz", allow_none=False)
        assert result is None
        mock_logger.return_value.warning.assert_called()

class TestExtractDateRange:
    def test_extract_from_dataframe(self):
        """Test extracting range from DataFrame."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")]
        })
        min_date, max_date = extract_date_range(df, as_string=True)
        assert min_date == "2024-01-01 00:00:00"
        assert max_date == "2024-01-05 00:00:00"

    def test_extract_as_timestamp(self):
        """Test extracting range as timestamps."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")]
        })
        min_date, max_date = extract_date_range(df, as_string=False)
        assert isinstance(min_date, pd.Timestamp)
        assert min_date == pd.Timestamp("2024-01-01")

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"date": []})
        min_date, max_date = extract_date_range(df)
        assert min_date is None
        assert max_date is None

    def test_missing_column(self):
        """Test with missing date column."""
        df = pd.DataFrame({"other": [1, 2]})
        min_date, max_date = extract_date_range(df)
        assert min_date is None
        assert max_date is None

    def test_extract_min_max_series_error(self):
        """Test error when min/max returns Series (e.g. duplicate columns)."""
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")]
        })
        # Duplicate column
        df = pd.concat([df, df], axis=1)
        # df["date"] will return a DataFrame

        with pytest.raises(TypeError, match="min/max returned a Series"):
            extract_date_range(df, "date")

    def test_extract_date_range_parse_error(self):
        """Test parsing error in date range."""
        df = pd.DataFrame({"date": ["invalid"]})
        # Pandas raises ValueError (DateParseError) with its own message
        with pytest.raises(ValueError):
            extract_date_range(df)

class TestFilterByDateRange:
    def test_filter_with_date_column(self):
        """Test filtering with date column."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [1, 2, 3]
        })
        filtered = filter_by_date_range(df, "2024-01-01", "2024-01-02")
        assert len(filtered) == 2
        assert filtered["value"].tolist() == [1, 2]

    def test_filter_with_index(self):
        """Test filtering with DatetimeIndex."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        )
        filtered = filter_by_date_range(df, "2024-01-02", "2024-01-03")
        assert len(filtered) == 2
        assert filtered["value"].tolist() == [2, 3]

    def test_filter_empty_result_raises(self):
        """Test raising error when result is empty."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "value": [1]
        })
        with pytest.raises(ValueError, match="No data available"):
            filter_by_date_range(df, "2025-01-01", "2025-01-02")

    def test_filter_empty_result_no_raise(self):
        """Test not raising error when result is empty."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "value": [1]
        })
        filtered = filter_by_date_range(df, "2025-01-01", "2025-01-02", raise_if_empty=False)
        assert filtered.empty

    def test_missing_column(self):
        """Test error when date column is missing."""
        df = pd.DataFrame({"value": [1]})
        with pytest.raises(KeyError):
            filter_by_date_range(df, "2024-01-01", "2024-01-02")

class TestFormatDatesToString:
    def test_format_series(self):
        """Test formatting Series."""
        dates = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        result = format_dates_to_string(dates, "%Y-%m-%d")
        assert result.tolist() == ["2024-01-01", "2024-01-02"]

    def test_format_index(self):
        """Test formatting DatetimeIndex."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        result = format_dates_to_string(dates, "%Y/%m/%d")
        assert result.tolist() == ["2024/01/01", "2024/01/02"]

    def test_format_list(self):
        """Test formatting list."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        result = format_dates_to_string(dates, "%Y%m%d")
        assert result.tolist() == ["20240101", "20240102"]

    def test_default_format(self):
        """Test default format."""
        dates = [pd.Timestamp("2024-01-01")]
        # Default is usually %Y-%m-%d
        result = format_dates_to_string(dates)
        assert result.tolist() == ["2024-01-01"]
