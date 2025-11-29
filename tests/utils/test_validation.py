"""Tests for src.utils.validation."""

import pytest
import pandas as pd
import numpy as np
from src.utils.validation import (
    has_both_splits,
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
    validate_series,
    validate_ticker_id,
    validate_train_ratio,
)

class TestValidation:
    def test_has_both_splits(self):
        """Test split validation."""
        df = pd.DataFrame({"split": ["train", "test"]})
        assert has_both_splits(df)

        df = pd.DataFrame({"split": ["train", "train"]})
        assert not has_both_splits(df)

        df = pd.DataFrame({"other": [1]})
        assert not has_both_splits(df)

    def test_validate_dataframe_not_empty(self):
        """Test empty dataframe validation."""
        df = pd.DataFrame({"a": [1]})
        validate_dataframe_not_empty(df) # Should pass

        df = pd.DataFrame()
        with pytest.raises(ValueError, match="is empty"):
            validate_dataframe_not_empty(df)

    def test_validate_file_exists(self, tmp_path):
        """Test file existence validation."""
        path = tmp_path / "test.txt"
        path.touch()

        validate_file_exists(path) # Should pass

        with pytest.raises(FileNotFoundError):
            validate_file_exists(tmp_path / "missing.txt")

    def test_validate_required_columns(self):
        """Test column validation."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        validate_required_columns(df, ["a", "b"]) # Should pass

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ["a", "c"])

    def test_validate_series(self):
        """Test series validation."""
        s = pd.Series([1, 2])
        validate_series(s) # Should pass

        # Empty
        with pytest.raises(ValueError, match="is empty"):
            validate_series(pd.Series(dtype=float))

        # All null
        with pytest.raises(ValueError, match="contains only null values"):
            validate_series(pd.Series([np.nan, np.nan]))

    def test_validate_ticker_id(self):
        """Test ticker ID validation."""
        validate_ticker_id(123) # Should pass

        with pytest.raises(ValueError):
            validate_ticker_id(-1)

        with pytest.raises(ValueError):
            validate_ticker_id("123") # type: ignore

    def test_validate_train_ratio(self):
        """Test train ratio validation."""
        validate_train_ratio(0.8) # Should pass

        with pytest.raises(ValueError):
            validate_train_ratio(1.5)

        with pytest.raises(ValueError):
            validate_train_ratio(-0.1)
