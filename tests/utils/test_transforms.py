"""Tests for src/utils/transforms.py module."""

from __future__ import annotations

import os
import sys

import pandas as pd  # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.transforms import (
    extract_features_and_target,
    filter_by_split,
    remove_metadata_columns,
    stable_ticker_id,
)


class TestExtractFeaturesAndTarget:
    """Test cases for extract_features_and_target function."""

    def test_extracts_features_and_target(self):
        """Should extract features and target correctly."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [10, 20, 30]
        })

        X, y = extract_features_and_target(df, target_column="target")

        assert list(X.columns) == ["feature1", "feature2"]
        assert list(y) == [10, 20, 30]

    def test_uses_specified_feature_columns(self):
        """Should use only specified feature columns."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "target": [10, 20]
        })

        X, y = extract_features_and_target(df, target_column="target", feature_columns=["a", "c"])

        assert list(X.columns) == ["a", "c"]
        assert "b" not in X.columns

    def test_returns_copies(self):
        """Should return copies, not views."""
        df = pd.DataFrame({
            "feature": [1, 2, 3],
            "target": [10, 20, 30]
        })

        X, y = extract_features_and_target(df, target_column="target")
        X.iloc[0, 0] = 999

        assert df["feature"].iloc[0] == 1  # Original unchanged

    def test_all_columns_except_target_as_features(self):
        """Should use all columns except target when features not specified."""
        df = pd.DataFrame({
            "a": [1],
            "b": [2],
            "c": [3],
            "target": [10]
        })

        X, y = extract_features_and_target(df, target_column="target")

        assert set(X.columns) == {"a", "b", "c"}


class TestFilterBySplit:
    """Test cases for filter_by_split function."""

    def test_filters_by_split_value(self):
        """Should filter rows by split value."""
        df = pd.DataFrame({
            "data": [1, 2, 3, 4],
            "split": ["train", "train", "test", "test"]
        })

        result = filter_by_split(df, split_column="split", split_value="train")

        assert len(result) == 2
        assert list(result["data"]) == [1, 2]

    def test_returns_copy(self):
        """Should return a copy, not a view."""
        df = pd.DataFrame({
            "data": [1, 2],
            "split": ["train", "test"]
        })

        result = filter_by_split(df, split_column="split", split_value="train")
        result.iloc[0, 0] = 999

        assert df["data"].iloc[0] == 1  # Original unchanged

    def test_empty_result_when_no_match(self):
        """Should return empty DataFrame when no rows match."""
        df = pd.DataFrame({
            "data": [1, 2],
            "split": ["train", "train"]
        })

        result = filter_by_split(df, split_column="split", split_value="test")

        assert len(result) == 0

    def test_preserves_columns(self):
        """Should preserve all columns in result."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "split": ["train", "test"]
        })

        result = filter_by_split(df, split_column="split", split_value="train")

        assert list(result.columns) == ["a", "b", "split"]


class TestRemoveMetadataColumns:
    """Test cases for remove_metadata_columns function."""

    def test_removes_specified_columns(self):
        """Should remove specified metadata columns."""
        df = pd.DataFrame({
            "feature1": [1, 2],
            "feature2": [3, 4],
            "metadata": [5, 6]
        })

        result = remove_metadata_columns(df, metadata_columns=["metadata"])

        assert "metadata" not in result.columns
        assert "feature1" in result.columns
        assert "feature2" in result.columns

    def test_ignores_nonexistent_columns(self):
        """Should ignore columns that don't exist."""
        df = pd.DataFrame({
            "feature1": [1, 2],
            "feature2": [3, 4]
        })

        # Should not raise
        result = remove_metadata_columns(df, metadata_columns=["nonexistent"])

        assert list(result.columns) == ["feature1", "feature2"]

    def test_removes_multiple_columns(self):
        """Should remove multiple metadata columns."""
        df = pd.DataFrame({
            "feature": [1],
            "meta1": [2],
            "meta2": [3],
            "meta3": [4]
        })

        result = remove_metadata_columns(df, metadata_columns=["meta1", "meta2", "meta3"])

        assert list(result.columns) == ["feature"]

    def test_empty_metadata_list(self):
        """Should return unchanged DataFrame with empty metadata list."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        result = remove_metadata_columns(df, metadata_columns=[])

        assert list(result.columns) == ["a", "b"]


class TestStableTickerId:
    """Test cases for stable_ticker_id function."""

    def test_returns_integer(self):
        """Should return an integer ID."""
        result = stable_ticker_id("BTC/USD")

        assert isinstance(result, int)

    def test_deterministic(self):
        """Should return same ID for same ticker."""
        id1 = stable_ticker_id("BTC/USD")
        id2 = stable_ticker_id("BTC/USD")

        assert id1 == id2

    def test_different_tickers_different_ids(self):
        """Should return different IDs for different tickers."""
        id_btc = stable_ticker_id("BTC/USD")
        id_eth = stable_ticker_id("ETH/USD")

        assert id_btc != id_eth

    def test_salt_changes_id(self):
        """Salt should change the ID."""
        id_no_salt = stable_ticker_id("BTC/USD")
        id_with_salt = stable_ticker_id("BTC/USD", salt="mysalt")

        assert id_no_salt != id_with_salt

    def test_same_salt_same_id(self):
        """Same salt should produce same ID."""
        id1 = stable_ticker_id("BTC/USD", salt="test")
        id2 = stable_ticker_id("BTC/USD", salt="test")

        assert id1 == id2

    def test_id_is_positive(self):
        """Generated ID should be positive."""
        result = stable_ticker_id("BTC/USD")

        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
