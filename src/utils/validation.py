"""Data validation utilities."""

from __future__ import annotations

from pathlib import Path
import pandas as pd  # type: ignore[import-untyped]


def has_both_splits(df: pd.DataFrame, split_column: str = "split") -> bool:
    """Check if DataFrame has both train and test splits."""
    if split_column not in df.columns:
        return False
    return df[split_column].nunique() >= 2


def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that DataFrame is not empty."""
    if df.empty:
        raise ValueError(f"{name} is empty")


def validate_file_exists(path: Path | str, description: str | None = None) -> None:
    """Validate that file exists.

    Args:
        path: Path to the file to validate.
        description: Optional description for error message.
    """
    if not Path(path).exists():
        msg = f"{description} not found: {path}" if description else f"File not found: {path}"
        raise FileNotFoundError(msg)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], description: str | None = None) -> None:
    """Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        description: Optional description for error message.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"{description}: missing required columns: {missing}" if description else f"Missing required columns: {missing}"
        raise ValueError(msg)


def validate_series(series: pd.Series, name: str = "Series") -> None:
    """Validate a pandas Series."""
    if series.empty:
        raise ValueError(f"{name} is empty")
    if series.isnull().all():
        raise ValueError(f"{name} contains only null values")


def validate_ticker_id(ticker_id: int) -> None:
    """Validate ticker ID format."""
    if not isinstance(ticker_id, int) or ticker_id < 0:
        raise ValueError(f"Invalid ticker ID: {ticker_id}")


def validate_train_ratio(ratio: float) -> None:
    """Validate train/test ratio."""
    if not 0 < ratio < 1:
        raise ValueError(f"Train ratio must be between 0 and 1, got: {ratio}")
