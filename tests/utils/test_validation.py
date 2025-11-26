"""Tests for src/utils/validation.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.validation import (
    has_both_splits,
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
    validate_series,
    validate_ticker_id,
    validate_train_ratio,
)


class TestHasBothSplits:
    """Test cases for has_both_splits function."""

    def test_returns_true_with_both_splits(self):
        """Should return True when both train and test splits exist."""
        df = pd.DataFrame({
            "data": [1, 2, 3, 4],
            "split": ["train", "train", "test", "test"]
        })
        assert has_both_splits(df) is True

    def test_returns_false_with_single_split(self):
        """Should return False when only one split exists."""
        df = pd.DataFrame({
            "data": [1, 2, 3],
            "split": ["train", "train", "train"]
        })
        assert has_both_splits(df) is False

    def test_returns_false_without_split_column(self):
        """Should return False when split column is missing."""
        df = pd.DataFrame({"data": [1, 2, 3]})
        assert has_both_splits(df) is False

    def test_custom_split_column(self):
        """Should work with custom split column name."""
        df = pd.DataFrame({
            "data": [1, 2],
            "my_split": ["train", "test"]
        })
        assert has_both_splits(df, split_column="my_split") is True

    def test_returns_true_with_more_than_two_splits(self):
        """Should return True when more than two splits exist."""
        df = pd.DataFrame({
            "data": [1, 2, 3],
            "split": ["train", "val", "test"]
        })
        assert has_both_splits(df) is True


class TestValidateDataframeNotEmpty:
    """Test cases for validate_dataframe_not_empty function."""

    def test_passes_for_non_empty_dataframe(self):
        """Should not raise for non-empty DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        validate_dataframe_not_empty(df)  # Should not raise

    def test_raises_for_empty_dataframe(self):
        """Should raise ValueError for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe_not_empty(df)

    def test_custom_name_in_error(self):
        """Should include custom name in error message."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="MyData is empty"):
            validate_dataframe_not_empty(df, name="MyData")


class TestValidateFileExists:
    """Test cases for validate_file_exists function."""

    def test_passes_for_existing_file(self):
        """Should not raise for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            validate_file_exists(temp_path)  # Should not raise
        finally:
            os.unlink(temp_path)

    def test_raises_for_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_file_exists("/nonexistent/path/file.txt")

    def test_accepts_path_object(self):
        """Should accept Path object."""
        with pytest.raises(FileNotFoundError):
            validate_file_exists(Path("/nonexistent/file.txt"))


class TestValidateRequiredColumns:
    """Test cases for validate_required_columns function."""

    def test_passes_with_all_columns(self):
        """Should not raise when all required columns exist."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_required_columns(df, ["a", "b"])  # Should not raise

    def test_raises_for_missing_columns(self):
        """Should raise ValueError when columns are missing."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_error_lists_missing_columns(self):
        """Should list missing columns in error message."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="c"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_empty_required_columns(self):
        """Should pass with empty required columns list."""
        df = pd.DataFrame({"a": [1]})
        validate_required_columns(df, [])  # Should not raise


class TestValidateSeries:
    """Test cases for validate_series function."""

    def test_passes_for_valid_series(self):
        """Should not raise for valid series."""
        series = pd.Series([1, 2, 3])
        validate_series(series)  # Should not raise

    def test_raises_for_empty_series(self):
        """Should raise ValueError for empty series."""
        series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="is empty"):
            validate_series(series)

    def test_raises_for_all_null_series(self):
        """Should raise ValueError when all values are null."""
        series = pd.Series([None, None, None])
        with pytest.raises(ValueError, match="only null values"):
            validate_series(series)

    def test_custom_name_in_error(self):
        """Should include custom name in error message."""
        series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="MySeries is empty"):
            validate_series(series, name="MySeries")

    def test_passes_with_some_null(self):
        """Should pass when only some values are null."""
        series = pd.Series([1, None, 3])
        validate_series(series)  # Should not raise


class TestValidateTickerId:
    """Test cases for validate_ticker_id function."""

    def test_passes_for_valid_positive_id(self):
        """Should not raise for valid positive ID."""
        validate_ticker_id(1)  # Should not raise
        validate_ticker_id(100)  # Should not raise

    def test_passes_for_zero_id(self):
        """Should not raise for zero ID."""
        validate_ticker_id(0)  # Should not raise

    def test_raises_for_negative_id(self):
        """Should raise ValueError for negative ID."""
        with pytest.raises(ValueError, match="Invalid ticker ID"):
            validate_ticker_id(-1)

    def test_raises_for_non_integer(self):
        """Should raise ValueError for non-integer ID."""
        with pytest.raises(ValueError, match="Invalid ticker ID"):
            validate_ticker_id(1.5)  # type: ignore[arg-type]

    def test_raises_for_string(self):
        """Should raise ValueError for string ID."""
        with pytest.raises(ValueError, match="Invalid ticker ID"):
            validate_ticker_id("BTC")  # type: ignore[arg-type]


class TestValidateTrainRatio:
    """Test cases for validate_train_ratio function."""

    def test_passes_for_valid_ratio(self):
        """Should not raise for valid ratio."""
        validate_train_ratio(0.8)  # Should not raise
        validate_train_ratio(0.5)  # Should not raise
        validate_train_ratio(0.1)  # Should not raise

    def test_raises_for_zero(self):
        """Should raise ValueError for zero ratio."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_train_ratio(0.0)

    def test_raises_for_one(self):
        """Should raise ValueError for ratio of 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_train_ratio(1.0)

    def test_raises_for_negative(self):
        """Should raise ValueError for negative ratio."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_train_ratio(-0.5)

    def test_raises_for_greater_than_one(self):
        """Should raise ValueError for ratio > 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_train_ratio(1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
