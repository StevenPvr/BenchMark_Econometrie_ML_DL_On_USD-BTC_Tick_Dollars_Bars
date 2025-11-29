"""Tests for src.utils.transforms."""

import pytest
import pandas as pd
from src.utils.transforms import (
    extract_features_and_target,
    filter_by_split,
    remove_metadata_columns,
    stable_ticker_id,
)

class TestTransforms:
    def test_extract_features_and_target(self):
        """Test feature and target extraction."""
        df = pd.DataFrame({
            "f1": [1], "f2": [2], "target": [3]
        })
        X, y = extract_features_and_target(df, "target")
        assert list(X.columns) == ["f1", "f2"]
        assert y.name == "target"

        # With feature columns specified
        X, y = extract_features_and_target(df, "target", feature_columns=["f1"])
        assert list(X.columns) == ["f1"]

    def test_filter_by_split(self):
        """Test filtering by split column."""
        df = pd.DataFrame({
            "val": [1, 2, 3],
            "split": ["train", "train", "test"]
        })
        train = filter_by_split(df, "split", "train")
        assert len(train) == 2
        assert all(train["split"] == "train")

    def test_remove_metadata_columns(self):
        """Test removing metadata columns."""
        df = pd.DataFrame({
            "data": [1], "meta1": [2], "meta2": [3]
        })
        clean = remove_metadata_columns(df, ["meta1", "meta2", "meta3"]) # meta3 not in df
        assert list(clean.columns) == ["data"]

    def test_stable_ticker_id(self):
        """Test stable ticker ID generation."""
        id1 = stable_ticker_id("BTC/USD")
        id2 = stable_ticker_id("BTC/USD")
        assert isinstance(id1, int)
        assert id1 == id2

        id3 = stable_ticker_id("ETH/USD")
        assert id1 != id3
