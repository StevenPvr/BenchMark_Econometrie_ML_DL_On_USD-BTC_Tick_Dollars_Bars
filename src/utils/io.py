"""I/O utilities for loading and saving data files.

This module provides functions for:
- Loading DataFrames (Parquet, CSV with automatic fallback)
- Saving DataFrames (Parquet + CSV)
- JSON file operations
- File system utilities (ensure directories exist)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.utils.validation import (
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
)

__all__ = [
    "get_parquet_path",
    "load_parquet_file",
    "load_csv_file",
    "read_dataset_file",
    "save_parquet_and_csv",
    "write_placeholder_file",
    "ensure_output_dir",
    "load_dataframe",
    "load_and_validate_dataframe",
    "load_json_data",
    "save_json_pretty",
]


def get_parquet_path(dataset_path: Path) -> Path:
    """Get Parquet path from dataset path.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        Path to Parquet file.

    Raises:
        ValueError: If dataset_path is not a valid file path.
    """
    if not dataset_path.name or dataset_path.name == ".":
        raise ValueError(f"Invalid dataset path: {dataset_path}")

    if dataset_path.suffix.lower() == ".csv":
        return dataset_path.with_suffix(".parquet")
    elif dataset_path.suffix.lower() == ".parquet":
        return dataset_path
    else:
        return dataset_path.with_suffix(".parquet")


def load_parquet_file(parquet_path: Path) -> pd.DataFrame | None:
    """Load dataset from Parquet file.

    Args:
        parquet_path: Path to Parquet file.

    Returns:
        DataFrame if successful, None otherwise.
    """
    logger = get_logger(__name__)

    if not parquet_path.exists():
        return None

    logger.info(f"Loading dataset from {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if df.empty:
            raise ValueError(f"Dataset is empty: {parquet_path}")
        return df
    except Exception as e:
        logger.warning(f"Failed to load Parquet file {parquet_path}: {e}")
        return None


def load_csv_file(csv_path: Path) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If dataset is empty.
    """
    logger = get_logger(__name__)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    logger.info(f"Loading dataset from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format='mixed')
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Dataset is empty: {csv_path}") from e
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")
    return df


def read_dataset_file(dataset_path: Path) -> pd.DataFrame:
    """Read dataset file (Parquet preferred, CSV fallback).

    Args:
        dataset_path: Path to the dataset file (CSV or Parquet).

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty.
    """
    parquet_path = get_parquet_path(dataset_path)

    # Try Parquet first
    df = load_parquet_file(parquet_path)
    if df is not None:
        return df

    # Fallback to CSV
    try:
        return load_csv_file(dataset_path)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path} " f"(tried {parquet_path} and {dataset_path})"
        ) from err


def ensure_output_dir(path: Path) -> None:
    """Ensure parent directory exists for a given path.

    Creates parent directories if they don't exist. Useful for ensuring
    output directories exist before saving files.

    Args:
        path: File path whose parent directory should be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def save_parquet_and_csv(df: pd.DataFrame, output_file: Path | str) -> None:
    """Save DataFrame to both parquet and CSV formats.

    Args:
        df: DataFrame to save.
        output_file: Path to save the parquet file (CSV will have same name with .csv extension).
    """
    logger = get_logger(__name__)
    output_path = Path(output_file)
    ensure_output_dir(output_path)

    # Save parquet
    df.to_parquet(output_path, index=False)

    # Save CSV with same name
    from src.constants import (
        CSV_ENCODING_DEFAULT,
        CSV_LINETERMINATOR_DEFAULT,
        CSV_QUOTING_DEFAULT,
        CSV_SEPARATOR_DEFAULT,
    )

    csv_output_file = output_path.with_suffix(".csv")
    df.to_csv(
        csv_output_file,
        index=False,
        sep=CSV_SEPARATOR_DEFAULT,
        encoding=CSV_ENCODING_DEFAULT,
        quoting=CSV_QUOTING_DEFAULT,
        lineterminator=CSV_LINETERMINATOR_DEFAULT,
    )
    logger.info(f"Saved to parquet: {output_path}")
    logger.info(f"Saved to CSV: {csv_output_file}")


def write_placeholder_file(path: Path) -> None:
    """Write a minimal non-empty placeholder file.

    Useful as a fallback when plotting or file generation fails.
    Creates parent directories if they don't exist.

    Args:
        path: Output file path.
    """
    ensure_output_dir(path)
    path.write_bytes(b"placeholder")


def _resolve_file_format(path_obj: Path, file_format: str | None) -> str:
    """Determine file format from path extension or explicit format."""
    if file_format is not None:
        return file_format
    return path_obj.suffix.lower().lstrip(".")


def _load_file_by_format(path_obj: Path, file_format: str) -> pd.DataFrame:
    """Load DataFrame based on file format."""
    if file_format == "parquet":
        return pd.read_parquet(path_obj)
    if file_format == "csv":
        return pd.read_csv(path_obj)
    # Try read_dataset_file for automatic fallback
    return read_dataset_file(path_obj)


def _convert_date_columns(df: pd.DataFrame, date_columns: list[str] | None) -> pd.DataFrame:
    """Convert specified columns to datetime type."""
    if date_columns is None:
        return df
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed')
    return df


def _validate_and_sort_dataframe(
    df: pd.DataFrame,
    path_name: str,
    required_columns: list[str] | set[str] | None,
    validate_not_empty: bool,
    sort_by: list[str] | None,
) -> pd.DataFrame:
    """Validate and sort DataFrame."""
    if required_columns is not None:
        validate_required_columns(df, required_columns)

    if validate_not_empty:
        validate_dataframe_not_empty(df, f"Data from {path_name}")

    if sort_by is not None:
        df = df.sort_values(sort_by).reset_index(drop=True)

    return df


def load_dataframe(
    path: Path | str,
    *,
    date_columns: list[str] | None = None,
    required_columns: list[str] | set[str] | None = None,
    validate_not_empty: bool = True,
    sort_by: list[str] | None = None,
    file_format: str | None = None,
) -> pd.DataFrame:
    """Load DataFrame from CSV or Parquet with automatic validation.

    Comprehensive loading function that handles format detection, date parsing,
    validation, and sorting in a single call. Consolidates common loading patterns.

    Args:
        path: File path. Format auto-detected from extension if file_format=None.
        date_columns: Columns to convert to datetime64 type.
        required_columns: Columns that must exist (raises KeyError if missing).
        validate_not_empty: If True, raise ValueError if DataFrame is empty.
        sort_by: Columns to sort by after loading (e.g., ["ticker", "date"]).
        file_format: Force format ('csv', 'parquet', or None for auto-detection).

    Returns:
        Loaded and validated DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If required columns are missing.
        ValueError: If DataFrame is empty and validate_not_empty=True.

    Examples:
        Basic loading:
        >>> df = load_dataframe("data/prices.parquet")

        With validation and sorting:
        >>> df = load_dataframe(
        ...     "data/prices.csv",
        ...     date_columns=["date"],
        ...     required_columns=["date", "ticker", "close"],
        ...     sort_by=["ticker", "date"]
        ... )

        Force CSV format:
        >>> df = load_dataframe("data.txt", file_format="csv")

    Usage in project:
        - Replaces patterns in data_cleaning/validation.py:88
        - Replaces patterns in arima/stationnarity_check/utils.py:28
        - Replaces 20+ similar loading patterns
    """
    path_obj = Path(path)
    validate_file_exists(path_obj, "Data file")

    # Determine format
    resolved_format = _resolve_file_format(path_obj, file_format)

    # Load file
    df = _load_file_by_format(path_obj, resolved_format)

    # Convert date columns
    df = _convert_date_columns(df, date_columns)

    # Validate and sort
    df = _validate_and_sort_dataframe(
        df, path_obj.name, required_columns, validate_not_empty, sort_by
    )

    return df


def load_json_data(
    path: Path | str,
    *,
    required_keys: list[str] | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Load JSON file with validation of required keys.

    Args:
        path: Path to JSON file.
        required_keys: Keys that must exist in the loaded dict.
        context: Optional context label for error messages (unused placeholder).

    Returns:
        Loaded dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If required keys are missing.
        json.JSONDecodeError: If file is not valid JSON.

    Examples:
        Basic loading:
        >>> metadata = load_json_data("results/metadata.json")

        With validation:
        >>> config = load_json_data(
        ...     "config.json",
        ...     required_keys=["model_type", "params"]
        ... )

    Usage in project:
        - Replaces 15+ json.load() patterns
        - Standardizes JSON loading with validation
    """
    path_obj = Path(path)
    validate_file_exists(path_obj, "JSON file")

    with open(path_obj) as f:
        data = json.load(f)

    if required_keys is not None:
        missing_keys = set(required_keys) - set(data.keys())
        if missing_keys:
            msg = f"Missing required keys in {path_obj.name}: {sorted(missing_keys)}"
            raise KeyError(msg)

    return data


def save_json_pretty(
    data: dict | list,
    output_path: Path | str,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Save JSON with pretty formatting and automatic directory creation.

    Args:
        data: Dictionary or list to save as JSON.
        output_path: Path to save JSON file.
        indent: Indentation level for pretty printing.
        sort_keys: If True, sort dictionary keys alphabetically.

    Examples:
        Save metrics:
        >>> save_json_pretty(
        ...     {"mae": 0.123, "rmse": 0.456},
        ...     "results/metrics.json"
        ... )

        Save config:
        >>> save_json_pretty(
        ...     {"model": "GARCH", "params": {"p": 1, "q": 1}},
        ...     "config/model_config.json",
        ...     sort_keys=True
        ... )

    Usage in project:
        - Replaces 20+ json.dump() patterns
        - Standardizes JSON formatting across project
    """
    path_obj = Path(output_path)
    ensure_output_dir(path_obj)

    with open(path_obj, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)


def _apply_column_renames(df: pd.DataFrame, column_renames: dict[str, str] | None) -> pd.DataFrame:
    """Apply column renaming to DataFrame.

    Args:
        df: Input DataFrame.
        column_renames: Dictionary mapping old column names to new names.

    Returns:
        DataFrame with renamed columns.
    """
    if not column_renames:
        return df

    logger = get_logger(__name__)
    result_df = df.copy()

    for old_col, new_col in column_renames.items():
        if old_col in result_df.columns and new_col not in result_df.columns:
            result_df = result_df.rename(columns={old_col: new_col})
            logger.info(f"Renamed column '{old_col}' to '{new_col}' for internal processing")

    return result_df


def _apply_sorting(df: pd.DataFrame, sort_by: list[str]) -> pd.DataFrame:
    """Apply sorting to DataFrame.

    Args:
        df: Input DataFrame.
        sort_by: Columns to sort by.

    Returns:
        Sorted DataFrame.
    """
    if sort_by and all(col in df.columns for col in sort_by):
        return df.sort_values(sort_by).reset_index(drop=True)
    return df


def _apply_nan_filtering(
    df: pd.DataFrame,
    drop_na_subset: list[str] | None,
    validate_not_empty: bool,
) -> pd.DataFrame:
    """Apply NaN filtering to DataFrame.

    Args:
        df: Input DataFrame.
        drop_na_subset: Columns to check for NaN values.
        validate_not_empty: Whether to validate DataFrame is not empty after filtering.

    Returns:
        DataFrame with NaN rows removed.
    """
    if not drop_na_subset:
        return df

    logger = get_logger(__name__)
    initial_len = len(df)
    result_df = df.dropna(subset=drop_na_subset)

    if len(result_df) < initial_len:
        logger.info(
            f"Dropped {initial_len - len(result_df)} rows with NaN values in {drop_na_subset}"
        )

    if validate_not_empty:
        validate_dataframe_not_empty(result_df, "Data after NaN removal")

    return result_df


def _try_standard_load(
    input_path: Path,
    date_columns: list[str],
    required_columns: list[str],
    sort_by: list[str],
    validate_not_empty: bool,
) -> pd.DataFrame:
    """Try loading with standard column names and validation."""
    return load_dataframe(
        input_path,
        date_columns=date_columns,
        required_columns=required_columns,
        validate_not_empty=validate_not_empty,
        sort_by=sort_by,
    )


def _fallback_load_with_transforms(
    input_path: Path,
    column_renames: dict[str, str] | None,
    required_columns: list[str],
    validate_not_empty: bool,
    sort_by: list[str],
) -> pd.DataFrame:
    """Fallback loading with column renaming and validation."""
    df = read_dataset_file(input_path)
    df = _apply_column_renames(df, column_renames)
    validate_required_columns(df, required_columns)

    if validate_not_empty:
        validate_dataframe_not_empty(df, "Data")

    df = _apply_sorting(df, sort_by)
    return df


def load_and_validate_dataframe(
    input_file: str,
    date_columns: list[str],
    required_columns: list[str],
    sort_by: list[str],
    validate_not_empty: bool = True,
    drop_na_subset: list[str] | None = None,
    column_renames: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load and validate DataFrame with flexible preprocessing options.

    Extended version of load_dataframe with additional validation and transformation options.
    Provides flexible DataFrame loading with column renaming, NaN handling, and sorting.

    Args:
        input_file: Path to data file (Parquet or CSV).
        date_columns: Columns to parse as dates.
        required_columns: Required columns that must be present.
        sort_by: Columns to sort by.
        validate_not_empty: Whether to validate that DataFrame is not empty.
        drop_na_subset: Columns to check for NaN values to drop.
        column_renames: Optional column renames to apply.

    Returns:
        Loaded and validated DataFrame.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If DataFrame is empty when validation is required.
        KeyError: If required columns are missing.
    """
    input_path = Path(input_file)

    # Try loading with standard column names first
    try:
        df = _try_standard_load(
            input_path, date_columns, required_columns, sort_by, validate_not_empty
        )
    except (ValueError, KeyError):
        # Fallback: load without validation and apply transformations
        df = _fallback_load_with_transforms(
            input_path, column_renames, required_columns, validate_not_empty, sort_by
        )

    # Apply final transformations
    df = _apply_nan_filtering(df, drop_na_subset, validate_not_empty)

    return df
