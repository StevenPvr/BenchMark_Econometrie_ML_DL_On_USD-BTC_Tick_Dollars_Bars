"""Parquet I/O operations for tick data cleaning.

Handles streaming reads/writes for large parquet datasets with PyArrow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.constants import PARQUET_BATCH_SIZE
from src.path import DATASET_RAW_PARQUET, RAW_PARTITIONS_DIR
from src.utils import ensure_output_dir, get_logger

logger = get_logger(__name__)

__all__ = [
    "clean_stale_output",
    "get_parquet_partitions",
    "list_partition_files",
    "load_raw_trades",
]


def get_parquet_partitions(partitions_root: Path) -> list[Path]:
    """Find and validate parquet partition files in a directory.

    Parameters
    ----------
    partitions_root : Path
        Directory to search for partition files.

    Returns
    -------
    list[Path]
        Sorted list of partition file paths.

    Raises
    ------
    FileNotFoundError
        If directory does not exist.
    ValueError
        If no parquet files found.
    """
    if not partitions_root.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {partitions_root}")

    parquet_parts = sorted(
        p for p in partitions_root.glob("part-*")
        if ".parquet" in p.name.lower()
    )
    if not parquet_parts:
        raise ValueError(f"No parquet partition files found in {partitions_root}")

    return parquet_parts


def list_partition_files(partition_dir: Path | None = None) -> list[Path]:
    """Return sorted parquet partitions from the raw directory.

    Parameters
    ----------
    partition_dir : Path | None
        Directory containing partition files. Defaults to RAW_PARTITIONS_DIR.

    Returns
    -------
    list[Path]
        Sorted list of partition file paths.
    """
    partitions_root = Path(partition_dir) if partition_dir is not None else RAW_PARTITIONS_DIR
    return get_parquet_partitions(partitions_root)


def clean_stale_output(output_path: Path) -> None:
    """Remove stale output files or directories before writing.

    Parameters
    ----------
    output_path : Path
        Target output path to clean.
    """
    for stale_path in (output_path, output_path.with_suffix(".tmp.parquet")):
        if stale_path.exists():
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            else:
                stale_path.unlink()


def load_raw_trades(
    partition_dir: Path | None = None,
    output_path: Path = DATASET_RAW_PARQUET,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Load raw trades from partitioned parquet files.

    Streams partitions with PyArrow to handle very large datasets efficiently.

    Parameters
    ----------
    partition_dir : Path | None
        Directory containing part-*.parquet files.
        Defaults to RAW_PARTITIONS_DIR.
    output_path : Path
        Path where the consolidated parquet should be written.
    use_cache : bool
        If True, reuse existing consolidated file when fresher than partitions.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with all raw trades.

    Raises
    ------
    FileNotFoundError
        If partition directory does not exist.
    ValueError
        If no partitions found or result is empty.
    RuntimeError
        If partitions appear corrupted.
    """
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    partitions_root = Path(partition_dir) if partition_dir is not None else RAW_PARTITIONS_DIR
    parquet_parts = get_parquet_partitions(partitions_root)

    total_size = sum(f.stat().st_size for f in parquet_parts)
    logger.info(
        "Loading %d partition(s) from %s (%.2f GB)",
        len(parquet_parts),
        partitions_root,
        total_size / (1024**3),
    )

    try:
        dataset = ds.dataset(parquet_parts, format="parquet")
    except OSError as exc:
        if "timed out" in str(exc).lower():
            raise RuntimeError(
                f"Parquet partitions in {partitions_root} appear to be corrupted"
            ) from exc
        raise

    latest_partition_mtime = max(f.stat().st_mtime for f in parquet_parts)
    ensure_output_dir(output_path)

    if use_cache and output_path.exists() and output_path.stat().st_mtime >= latest_partition_mtime:
        logger.info("Using cached consolidated parquet at %s", output_path)
        df_cached = pd.read_parquet(output_path, engine="pyarrow")
        if df_cached.empty:
            raise ValueError("Raw trades dataset is empty")
        return df_cached

    clean_stale_output(output_path)

    temp_output = output_path.with_suffix(".tmp.parquet")
    logger.info("Streaming merge to %s", output_path)

    try:
        with pq.ParquetWriter(temp_output, dataset.schema, compression="snappy") as writer:
            for batch in dataset.to_batches(batch_size=PARQUET_BATCH_SIZE, use_threads=True):
                writer.write_batch(batch)
        temp_output.replace(output_path)
    except OSError as exc:
        if "timed out" in str(exc).lower():
            raise RuntimeError(
                f"Parquet partitions in {partitions_root} appear to be corrupted"
            ) from exc
        raise

    df = pd.read_parquet(output_path, engine="pyarrow")
    if df.empty:
        raise ValueError("Raw trades dataset is empty")

    logger.info("Consolidation complete. Total rows: %d", len(df))
    return df
