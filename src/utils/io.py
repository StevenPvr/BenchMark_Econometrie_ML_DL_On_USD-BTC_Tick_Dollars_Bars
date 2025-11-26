"""I/O utilities for file operations."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def ensure_output_dir(file_path: Path | str) -> None:
    """Ensure the output directory for a file exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_parquet_path(csv_path: Path | str) -> Path:
    """Convert CSV path to Parquet path."""
    return Path(csv_path).with_suffix('.parquet')


def load_and_validate_dataframe(path: Path | str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load and validate a DataFrame."""
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if required_columns and not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_csv_file(path: Path | str) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(path)


def load_dataframe(
    path: Path | str,
    date_columns: list[str] | None = None,
    validate_not_empty: bool = True,
) -> pd.DataFrame:
    """Load a DataFrame from CSV or Parquet.

    Args:
        path: Path to the file.
        date_columns: Columns to parse as datetime.
        validate_not_empty: If True, raise ValueError if DataFrame is empty.

    Returns:
        Loaded DataFrame.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == '.csv':
        df = pd.read_csv(path, parse_dates=date_columns)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if validate_not_empty and df.empty:
        raise ValueError(f"DataFrame loaded from {path} is empty")

    return df


def load_json_data(path: Path | str) -> dict[str, Any]:
    """Load JSON data from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_parquet_file(path: Path | str) -> pd.DataFrame:
    """Load a Parquet file."""
    return pd.read_parquet(path)


def read_dataset_file(path: Path | str) -> pd.DataFrame:
    """Read a dataset file (CSV or Parquet)."""
    return load_dataframe(path)


def save_json_pretty(data: dict[str, Any], path: Path | str) -> None:
    """Save data as pretty JSON."""
    ensure_output_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_parquet_and_csv(df: pd.DataFrame, parquet_path: Path | str, csv_path: Path | str | None = None) -> None:
    """Save DataFrame to both Parquet and CSV formats."""
    ensure_output_dir(parquet_path)
    df.to_parquet(parquet_path, index=False)

    if csv_path is None:
        csv_path = get_parquet_path(parquet_path).with_suffix('.csv')

    ensure_output_dir(csv_path)
    df.to_csv(csv_path, index=False, lineterminator='\n')


def write_placeholder_file(path: Path | str, content: str = "Placeholder file") -> None:
    """Write a placeholder file."""
    ensure_output_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
