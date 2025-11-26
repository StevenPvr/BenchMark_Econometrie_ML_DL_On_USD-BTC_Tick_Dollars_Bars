"""Tests for src/utils/datetime_utils.py module."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import cast

import pandas as pd # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.datetime_utils import (
    extract_date_range,
    filter_by_date_range,
    format_dates_to_string,
    normalize_timestamp_to_datetime,
    parse_date_value,
)


class TestNormalizeTimestampToDatetime:
    """Test cases for normalize_timestamp_to_datetime function."""

    def test_timezone_aware_to_naive(self):
        """Should convert timezone-aware to timezone-naive."""
        ts = cast(pd.Timestamp, pd.Timestamp("2024-01-01 12:00:00", tz="UTC"))
        result = normalize_timestamp_to_datetime(ts)

        assert result.tzinfo is None
        assert result.hour == 12

    def test_timezone_naive_unchanged(self):
        """Timezone-naive should remain unchanged."""
        ts = cast(pd.Timestamp, pd.Timestamp("2024-01-01 12:00:00"))
        result = normalize_timestamp_to_datetime(ts)

        assert result == ts

    def test_utc_timezone(self):
        """UTC timezone should be stripped."""
        ts = cast(pd.Timestamp, pd.Timestamp("2024-01-01", tz="UTC"))
        result = normalize_timestamp_to_datetime(ts)

        assert result.tzinfo is None

    def test_non_utc_timezone(self):
        """Non-UTC timezone should be stripped."""
        ts = cast(pd.Timestamp, pd.Timestamp("2024-01-01 12:00:00", tz="US/Eastern"))
        result = normalize_timestamp_to_datetime(ts)

        assert result.tzinfo is None


class TestParseDateValue:
    """Test cases for parse_date_value function."""

    def test_parses_string(self):
        """Should parse string date."""
        result = parse_date_value("2024-01-01")

        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parses_timestamp(self):
        """Should parse pandas Timestamp."""
        ts = pd.Timestamp("2024-06-15 14:30:00")
        result = parse_date_value(ts)

        assert result == ts

    def test_datetime_not_directly_supported(self):
        """datetime.datetime not directly supported - returns None."""
        # Note: parse_date_value doesn't directly support datetime.datetime
        # It only supports pd.Series, pd.Timestamp, str, int, float
        dt = datetime(2024, 3, 20, 10, 30, 0)
        result = parse_date_value(dt, allow_none=True)

        # datetime.datetime returns None (unsupported type)
        assert result is None

    def test_datetime_via_timestamp(self):
        """datetime can be parsed via pd.Timestamp conversion."""
        dt = datetime(2024, 3, 20, 10, 30, 0)
        ts = pd.Timestamp(dt)  # Convert to Timestamp first
        result = parse_date_value(ts)

        assert result is not None
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 20

    def test_parses_integer(self):
        """Should parse integer (unix timestamp)."""
        # 2024-01-01 00:00:00 in nanoseconds
        result = parse_date_value(1704067200000000000)

        assert isinstance(result, pd.Timestamp)

    def test_parses_float(self):
        """Should parse float (unix timestamp)."""
        result = parse_date_value(1704067200.0)

        assert isinstance(result, pd.Timestamp)

    def test_parses_series_first_element(self):
        """Should extract first element from Series."""
        series = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
        result = parse_date_value(series)

        assert result == pd.Timestamp("2024-01-01")

    def test_empty_series_returns_none(self):
        """Empty Series should return None."""
        series = pd.Series([], dtype="datetime64[ns]")
        result = parse_date_value(series, allow_none=True)

        assert result is None

    def test_invalid_returns_none_with_allow_none(self):
        """Invalid input should return None when allow_none=True."""
        result = parse_date_value("invalid_date_string", allow_none=True)

        assert result is None

    def test_nat_returns_none(self):
        """NaT should return None."""
        result = parse_date_value(pd.NaT, allow_none=True)

        assert result is None

    def test_timezone_aware_normalized(self):
        """Timezone-aware timestamps should be normalized."""
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        result = parse_date_value(ts)

        assert result is not None
        assert result.tzinfo is None


class TestExtractDateRange:
    """Test cases for extract_date_range function."""

    def test_extracts_min_max(self):
        """Should extract min and max dates."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10),
        })
        min_date, max_date = extract_date_range(df)

        assert min_date is not None and isinstance(min_date, str)
        assert max_date is not None and isinstance(max_date, str)
        assert "2024-01-01" in min_date
        assert "2024-01-10" in max_date

    def test_empty_dataframe_returns_none(self):
        """Empty DataFrame should return (None, None)."""
        df = pd.DataFrame(columns=pd.Index(["date", "value"]))
        min_date, max_date = extract_date_range(df)

        assert min_date is None
        assert max_date is None

    def test_missing_column_returns_none(self):
        """Missing date column should return (None, None)."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        min_date, max_date = extract_date_range(df)

        assert min_date is None
        assert max_date is None

    def test_as_string_option(self):
        """as_string=True should return strings."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        })
        min_date, max_date = extract_date_range(df, as_string=True)

        assert isinstance(min_date, str)
        assert isinstance(max_date, str)

    def test_as_timestamp_option(self):
        """as_string=False should return Timestamps."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        })
        min_date, max_date = extract_date_range(df, as_string=False)

        assert isinstance(min_date, pd.Timestamp)
        assert isinstance(max_date, pd.Timestamp)

    def test_custom_date_column(self):
        """Should work with custom date column name."""
        df = pd.DataFrame({
            "my_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": range(5),
        })
        min_date, max_date = extract_date_range(df, date_col="my_date")

        assert min_date is not None
        assert max_date is not None


class TestFilterByDateRange:
    """Test cases for filter_by_date_range function."""

    def test_filters_correctly(self):
        """Should filter to date range."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "value": range(30),
        })
        result = filter_by_date_range(df, "2024-01-10", "2024-01-20")

        assert len(result) == 11  # 10th to 20th inclusive

    def test_inclusive_bounds(self):
        """Bounds should be inclusive."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10),
        })
        result = filter_by_date_range(df, "2024-01-01", "2024-01-03")

        assert len(result) == 3
        assert result.iloc[0]["date"] == pd.Timestamp("2024-01-01")
        assert result.iloc[-1]["date"] == pd.Timestamp("2024-01-03")

    def test_empty_result_raises_with_flag(self):
        """Empty result should raise when raise_if_empty=True."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10),
        })
        with pytest.raises(ValueError, match="No data available"):
            filter_by_date_range(df, "2025-01-01", "2025-12-31", raise_if_empty=True)

    def test_empty_result_returns_empty_without_flag(self):
        """Empty result should return empty DataFrame when raise_if_empty=False."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10),
        })
        result = filter_by_date_range(df, "2025-01-01", "2025-12-31", raise_if_empty=False)

        assert len(result) == 0

    def test_datetime_index(self):
        """Should work with DatetimeIndex."""
        df = pd.DataFrame({
            "value": range(10),
        }, index=pd.date_range("2024-01-01", periods=10, freq="D"))

        result = filter_by_date_range(df, "2024-01-03", "2024-01-07")

        assert len(result) == 5

    def test_date_column(self):
        """Should work with date column."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10),
        })
        result = filter_by_date_range(df, "2024-01-03", "2024-01-07")

        assert len(result) == 5

    def test_string_dates_converted(self):
        """String dates in column should be converted."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [1, 2, 3],
        })
        result = filter_by_date_range(df, "2024-01-01", "2024-01-02")

        assert len(result) == 2

    def test_missing_date_column_raises(self):
        """Missing date column should raise KeyError."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(KeyError, match="not found"):
            filter_by_date_range(df, "2024-01-01", "2024-01-02")


class TestFormatDatesToString:
    """Test cases for format_dates_to_string function."""

    def test_default_format(self):
        """Should use default format."""
        dates = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
        result = format_dates_to_string(dates)

        assert result.iloc[0] == "2024-01-01"

    def test_custom_format(self):
        """Should use custom format."""
        dates = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
        result = format_dates_to_string(dates, date_format="%d/%m/%Y")

        assert result.iloc[0] == "01/01/2024"

    def test_handles_series(self):
        """Should handle Series input."""
        dates = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])
        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_handles_datetimeindex(self):
        """Should handle DatetimeIndex input."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_handles_list(self):
        """Should handle list input."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_year_month_day_format(self):
        """Should format as YYYYMMDD."""
        dates = pd.Series([pd.Timestamp("2024-06-15")])
        result = format_dates_to_string(dates, date_format="%Y%m%d")

        assert result.iloc[0] == "20240615"

    def test_preserves_length(self):
        """Output should have same length as input."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        result = format_dates_to_string(dates)

        assert len(result) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
