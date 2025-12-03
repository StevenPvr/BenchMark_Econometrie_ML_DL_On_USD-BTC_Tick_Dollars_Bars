"""Tick-level cleaning pipeline for crypto trades (ccxt output).

Implements robust outlier detection methods suitable for financial markets
and dollar bar construction (De Prado methodology).

All outlier detection methods are CAUSAL (use only past data) to prevent
temporal data leakage.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.constants import OUTLIER_LEGACY_MAX_PCT_CHANGE, SYMBOL
from src.path import DATASET_CLEAN_PARQUET
from src.utils import ensure_output_dir, get_logger

from src.data_cleaning.outliers import (
    OutlierReport,
    filter_outliers_robust,
    filter_price_outliers,
    merge_outlier_reports,
)
from src.data_cleaning.parquet_io import (
    clean_stale_output,
    list_partition_files,
)

logger = get_logger(__name__)

__all__ = [
    "OutlierReport",
    "clean_ticks_data",
]


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate trades using (timestamp, id) keys when available.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trade data.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    subset: list[str] = []
    for col in ("timestamp", "id"):
        if col in df.columns:
            subset.append(col)
    if not subset:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate trades", removed)
    return df


def _drop_missing_essentials(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    """Drop rows missing any essential columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter.
    required : Sequence[str]
        Column names that must not contain NaN values.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with NaN rows removed.
    """
    before = len(df)
    df = df.dropna(subset=list(required))
    removed = before - len(df)
    if removed > 0:
        logger.info("Dropped %d rows with missing required values", removed)
    return df


def _validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that specified columns exist and are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    columns : list[str]
        Column names that must be numeric.

    Raises
    ------
    KeyError
        If a required column is not found.
    TypeError
        If a column is not numeric.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}")


def _strip_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove heavyweight or unused columns before saving.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.

    Returns
    -------
    pd.DataFrame
        DataFrame with unwanted columns removed.
    """
    unwanted = ["id", "info", "symbol"]
    return df.drop(columns=[c for c in unwanted if c in df.columns], errors="ignore")


def _persist_clean_dataset(df: pd.DataFrame) -> None:
    """Save cleaned dataset to parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to save.
    """
    ensure_output_dir(DATASET_CLEAN_PARQUET)
    df.to_parquet(DATASET_CLEAN_PARQUET, index=False)
    logger.info("Saved cleaned trades to %s", DATASET_CLEAN_PARQUET)


def _clean_partition_dataframe(
    df: pd.DataFrame,
    use_robust_outliers: bool,
    symbol: str,
) -> tuple[pd.DataFrame, OutlierReport | None]:
    """Apply the cleaning pipeline to a single partition.

    Parameters
    ----------
    df : pd.DataFrame
        Partition DataFrame to clean.
    use_robust_outliers : bool
        Whether to use robust outlier detection.
    symbol : str
        Trading pair symbol.

    Returns
    -------
    tuple[pd.DataFrame, OutlierReport | None]
        Cleaned DataFrame and optional outlier report.
    """
    df = _drop_missing_essentials(df, required=("timestamp", "price", "amount"))
    _validate_numeric_columns(df, ["price", "amount"])
    df = _drop_duplicates(df)

    report: OutlierReport | None
    if use_robust_outliers:
        df, report = filter_outliers_robust(df)
    else:
        df = filter_price_outliers(df, max_pct_change=OUTLIER_LEGACY_MAX_PCT_CHANGE)
        report = None

    df = _strip_unwanted_columns(df)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df, report


def _process_partition(
    partition_file: Path,
    idx: int,
    total: int,
    use_robust_outliers: bool,
    symbol: str,
) -> tuple[pd.DataFrame | None, OutlierReport | None]:
    """Load and clean a single partition file.

    Parameters
    ----------
    partition_file : Path
        Path to the partition parquet file.
    idx : int
        Current partition index (1-based).
    total : int
        Total number of partitions.
    use_robust_outliers : bool
        Whether to use robust outlier detection.
    symbol : str
        Trading pair symbol.

    Returns
    -------
    tuple[pd.DataFrame | None, OutlierReport | None]
        Cleaned DataFrame (or None if empty) and outlier report.
    """
    logger.info("Cleaning partition %d/%d: %s", idx, total, partition_file.name)

    df_partition = pd.read_parquet(partition_file, engine="pyarrow")
    if df_partition.empty:
        logger.info("Partition %s is empty, skipping", partition_file.name)
        return None, None

    df_cleaned, report = _clean_partition_dataframe(
        df_partition,
        use_robust_outliers=use_robust_outliers,
        symbol=symbol,
    )

    if df_cleaned.empty:
        logger.info("Partition %s yielded no rows after cleaning", partition_file.name)
        return None, report

    return df_cleaned, report


def clean_ticks_data(
    use_robust_outliers: bool = True,
    symbol: str | None = None,
    partition_dir: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """End-to-end cleaning pipeline for tick data downloaded via ccxt.

    All outlier detection methods are CAUSAL (use only past data via
    expanding windows) to prevent temporal data leakage.

    Parameters
    ----------
    use_robust_outliers : bool, default=True
        If True, use robust MAD-based outlier detection suitable for
        financial markets and dollar bars. If False, use legacy simple
        percentage-change filter.
    symbol : str | None, optional
        Trading pair symbol (e.g., "BTC/USDT"). Defaults to SYMBOL constant.
    partition_dir : Path | None, optional
        Directory containing partitioned parquet files.
    output_path : Path | None, optional
        Path for the cleaned parquet output.

    Raises
    ------
    ValueError
        If no data remains after cleaning.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("Starting tick data cleaning")
    symbol = symbol or SYMBOL
    partitions = list_partition_files(partition_dir=partition_dir)
    output_path = output_path or DATASET_CLEAN_PARQUET

    ensure_output_dir(output_path)
    clean_stale_output(output_path)

    temp_output = output_path.with_suffix(".tmp.parquet")
    aggregated_report: OutlierReport | None = None
    writer: pq.ParquetWriter | None = None
    total_written = 0

    try:
        for idx, partition_file in enumerate(partitions, start=1):
            df_cleaned, report = _process_partition(
                partition_file, idx, len(partitions), use_robust_outliers, symbol
            )

            if use_robust_outliers and report is not None:
                aggregated_report = merge_outlier_reports(aggregated_report, report)

            if df_cleaned is None:
                continue

            table = pa.Table.from_pandas(df_cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_output, table.schema, compression="snappy")
            writer.write_table(table)
            total_written += len(df_cleaned)
    finally:
        if writer is not None:
            writer.close()

    if total_written == 0:
        if temp_output.exists():
            temp_output.unlink()
        raise ValueError("No data remaining after cleaning")

    temp_output.replace(output_path)

    if use_robust_outliers and aggregated_report is not None:
        aggregated_report.final_ticks = total_written
        aggregated_report.log_summary()

    logger.info("Saved cleaned trades to %s", output_path)
