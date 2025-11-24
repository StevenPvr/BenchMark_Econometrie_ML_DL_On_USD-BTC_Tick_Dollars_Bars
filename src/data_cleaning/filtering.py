"""Dataset filtering functions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_cleaning.validation import assert_not_empty
from src.path import DATASET_FILTERED_FILE, DATASET_FILTERED_PARQUET_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def validate_filtering_inputs(raw_df: pd.DataFrame, valid_tickers: Iterable[str]) -> None:
    """Validate inputs for dataset filtering.

    Args:
        raw_df: Raw dataset DataFrame.
        valid_tickers: Iterable of valid ticker symbols.

    Raises:
        KeyError: If ``tickers`` column is missing.
        ValueError: If ``valid_tickers`` is empty or ``raw_df`` is empty.
    """
    assert_not_empty(raw_df, name="raw dataset")

    if "tickers" not in raw_df.columns:
        msg = "Missing required column: 'tickers'"
        raise KeyError(msg)

    valid_tickers = list(valid_tickers)
    if not valid_tickers:
        msg = "No valid tickers to filter. Dataset would be empty."
        raise ValueError(msg)


def write_filtered_dataset(filtered_df: pd.DataFrame, output_file: Path | None = None) -> Path:
    """Persist a filtered dataset to both CSV and Parquet formats.

    Args:
        filtered_df: Filtered dataset DataFrame.
        output_file: Optional explicit output file path. If not provided,
            ``DATASET_FILTERED_FILE`` is used. The parquet file will be saved
            with the same name but .parquet extension.

    Returns:
        Path to the written CSV file.

    Raises:
        OSError: If file cannot be written.
    """
    if output_file is None:
        output_file = DATASET_FILTERED_FILE

    # At this point, output_file is guaranteed to be a Path
    assert output_file is not None

    from src.utils.io import ensure_output_dir

    ensure_output_dir(output_file)

    # Determine parquet file path
    parquet_file = output_file.with_suffix(".parquet")
    if output_file == DATASET_FILTERED_FILE:
        parquet_file = DATASET_FILTERED_PARQUET_FILE

    try:
        # Save CSV with explicit parameters to ensure proper formatting
        # Use QUOTE_MINIMAL to escape values containing commas or quotes
        # Use lineterminator='\n' for Unix-style line endings
        filtered_df.to_csv(
            output_file,
            index=False,
            sep=",",
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        logger.info("Saved filtered dataset to CSV: %s", output_file)

        # Save Parquet format
        filtered_df.to_parquet(parquet_file, index=False)
        logger.info("Saved filtered dataset to Parquet: %s", parquet_file)
    except OSError as e:
        msg = f"Failed to save filtered dataset to {output_file}: {e}"
        raise OSError(msg) from e

    return output_file
