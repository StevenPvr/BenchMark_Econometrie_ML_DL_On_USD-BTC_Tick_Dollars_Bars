"""Data transformation utilities."""

from __future__ import annotations

import hashlib
import pandas as pd  # type: ignore[import-untyped]
from typing import cast


def extract_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from DataFrame."""
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y  # type: ignore[return-value]


def filter_by_split(df: pd.DataFrame, split_column: str, split_value: str) -> pd.DataFrame:
    """Filter DataFrame by split value."""
    mask = df[split_column] == split_value
    result = df[mask]
    # Ensure we return a DataFrame even if pandas infers differently
    return result.copy() if isinstance(result, pd.DataFrame) else pd.DataFrame()


def remove_metadata_columns(df: pd.DataFrame, metadata_columns: list[str]) -> pd.DataFrame:
    """Remove metadata columns from DataFrame."""
    return df.drop(columns=[col for col in metadata_columns if col in df.columns])


def stable_ticker_id(ticker: str, salt: str = "") -> int:
    """Generate stable integer ID for ticker (BTC/USD context)."""
    # For BTC/USD, we typically only have one symbol, but this function
    # remains for compatibility with multi-asset extensions
    combined = f"{ticker}{salt}".encode('utf-8')
    hash_obj = hashlib.md5(combined)
    # Take first 8 bytes and convert to int
    return int(hash_obj.hexdigest()[:16], 16)
