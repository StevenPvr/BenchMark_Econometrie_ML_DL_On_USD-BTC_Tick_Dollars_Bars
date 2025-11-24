"""Reporting functions for generating fetch reports and summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.constants import (
    DATA_FETCH_END_DATE,
    DATA_FETCH_START_DATE,
    NORMALIZED_DATE_COLUMN,
    NORMALIZED_TICKERS_COLUMN,
)
from src.path import DATASET_FILE, FETCH_REPORT_FILE
from src.utils import (
    ensure_output_dir,
    extract_date_range,
    get_logger,
    save_json_pretty,
    save_parquet_and_csv,
)

logger = get_logger(__name__)


def _combine_dataframes(data_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of ticker DataFrames into a single dataset.

    Ensures chronological order by sorting by date and ticker to prevent
    any potential data leakage from unordered data.

    Args:
        data_list: List of processed ticker DataFrames.

    Returns:
        Concatenated DataFrame sorted by date and ticker. Empty if data_list is empty.
    """
    if not data_list:
        logger.warning("No data to combine, returning empty DataFrame")
        return pd.DataFrame()

    # Pre-process each DataFrame to normalize datetime columns before concatenation
    processed_data_list = []
    for df in data_list:
        df = df.copy()
        if NORMALIZED_DATE_COLUMN in df.columns and pd.api.types.is_datetime64_any_dtype(
            df[NORMALIZED_DATE_COLUMN]
        ):
            # Normalize all datetime columns to tz-naive UTC before concatenation
            df[NORMALIZED_DATE_COLUMN] = pd.to_datetime(
                df[NORMALIZED_DATE_COLUMN], utc=True
            ).dt.tz_localize(None)
        processed_data_list.append(df)

    combined = pd.concat(processed_data_list, ignore_index=True)

    # Ensure chronological order: sort by date (ascending) then by tickers
    # This prevents any potential data leakage from unordered data
    if NORMALIZED_DATE_COLUMN in combined.columns:
        # All dates should now be tz-naive, so sorting should work safely
        combined = combined.sort_values(by=NORMALIZED_DATE_COLUMN, ascending=True, kind="mergesort")
        if NORMALIZED_TICKERS_COLUMN in combined.columns:
            combined = combined.sort_values(
                by=[NORMALIZED_DATE_COLUMN, NORMALIZED_TICKERS_COLUMN],
                ascending=[True, True],
                kind="mergesort",
            )
        combined = combined.reset_index(drop=True)

    logger.info("Combined dataset has %d rows", len(combined))
    return combined


def _build_fetch_report(
    dataset: pd.DataFrame,
    failed_tickers: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable report for the fetch process.

    Args:
        dataset: Combined dataset of all successful tickers.
        failed_tickers: List of tickers for which download failed.

    Returns:
        Dictionary with metadata about the fetch process.
    """
    n_rows = int(len(dataset))
    unique_tickers = (
        sorted(dataset[NORMALIZED_TICKERS_COLUMN].unique()) if not dataset.empty else []
    )
    n_unique_tickers = int(len(unique_tickers))

    realized_start, realized_end = extract_date_range(
        dataset, date_col=NORMALIZED_DATE_COLUMN, as_string=True
    )

    report: dict[str, Any] = {
        "requested_start_date": str(DATA_FETCH_START_DATE.date()),
        "requested_end_date": str(DATA_FETCH_END_DATE.date()),
        "realized_start_date": realized_start,
        "realized_end_date": realized_end,
        "n_rows": n_rows,
        "n_unique_tickers": n_unique_tickers,
        "tickers": unique_tickers,
        "failed_tickers": failed_tickers,
    }
    return report


def _save_fetch_report(report: dict[str, Any]) -> None:
    """Persist the fetch report to JSON.

    Delegates to src.utils.save_json_pretty() for consistency.

    Args:
        report: Report dictionary to save.
    """
    save_json_pretty(report, FETCH_REPORT_FILE, indent=2, sort_keys=True)
    logger.info("Saved fetch report to %s", FETCH_REPORT_FILE)


def _save_dataset(dataset: pd.DataFrame) -> None:
    """Persist the combined dataset to disk.

    Saves both parquet and CSV formats for flexibility.

    Args:
        dataset: Combined dataset DataFrame.
    """
    # Ensure output directory exists and save dataset
    ensure_output_dir(DATASET_FILE.with_suffix(".parquet"))
    # Use shared utility to enforce consistent, safe CSV parameters
    save_parquet_and_csv(dataset, DATASET_FILE.with_suffix(".parquet"))
    logger.info(
        "Saved dataset with %d rows to %s(.parquet/.csv)",
        len(dataset),
        DATASET_FILE,
    )


def combine_and_save_data(
    data_list: list[pd.DataFrame],
    failed_tickers: list[str],
) -> None:
    """Combine all ticker data, save dataset and fetch report."""

    if not data_list:
        raise RuntimeError("No data downloaded to combine")

    dataset = _combine_dataframes(data_list)

    if dataset.empty:
        raise RuntimeError("Dataset is empty after removing NaN values")

    _save_dataset(dataset)

    report = _build_fetch_report(dataset, failed_tickers)
    _save_fetch_report(report)

    logger.info(
        "Fetch summary: %d rows, %d tickers, %d failed tickers",
        len(dataset),
        len(report.get("tickers", [])),
        len(failed_tickers),
    )
