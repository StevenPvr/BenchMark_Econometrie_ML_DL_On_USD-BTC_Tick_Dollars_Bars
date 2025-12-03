"""Parquet storage utilities for trade data."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.path import DATASET_RAW_PARQUET
from src.utils import ensure_output_dir, get_logger

if TYPE_CHECKING:
    import pyarrow as pa_module
    import pyarrow.parquet as pq_module

logger = get_logger(__name__)

# Optional dependencies; tests monkeypatch these attributes
try:
    import pyarrow as _pa
    import pyarrow.parquet as _pq
except ImportError:
    _pa = None  # type: ignore[assignment]
    _pq = None  # type: ignore[assignment]

pa: pa_module | None = _pa  # type: ignore[name-defined]
pq: pq_module | None = _pq  # type: ignore[name-defined]


def iter_parquet_files(dataset_path: Path) -> list[Path]:
    """Return list of parquet files for a dataset path.

    Args:
        dataset_path: Path to a file or directory.

    Returns:
        Sorted list of parquet file paths.
    """
    if dataset_path.is_file():
        return [dataset_path]
    if dataset_path.is_dir():
        return sorted(p for p in dataset_path.glob("*.parquet") if p.is_file())
    return []


def get_existing_timestamp_range() -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Read min/max timestamps from parquet metadata without loading full data.

    Returns:
        Tuple of (min_timestamp, max_timestamp, total_rows).
    """
    dataset_path = Path(DATASET_RAW_PARQUET)
    files = iter_parquet_files(dataset_path)
    if not files or pq is None:
        return None, None, 0

    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None
    total_rows = 0

    for file_path in files:
        result = _read_parquet_timestamp_stats(file_path)
        if result is None:
            continue

        rg_min, rg_max, rows = result
        total_rows += rows

        if rg_min is not None and (min_ts is None or rg_min < min_ts):
            min_ts = rg_min
        if rg_max is not None and (max_ts is None or rg_max > max_ts):
            max_ts = rg_max

    return min_ts, max_ts, total_rows


def _read_parquet_timestamp_stats(
    file_path: Path,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int] | None:
    """Read timestamp statistics from a parquet file's metadata.

    Args:
        file_path: Path to the parquet file.

    Returns:
        Tuple of (min_ts, max_ts, num_rows) or None if unreadable.
    """
    try:
        pf = pq.ParquetFile(file_path)  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("Could not read metadata from %s: %s", file_path, exc)
        return None

    num_rows = pf.metadata.num_rows if pf.metadata else 0

    try:
        col_index = pf.schema_arrow.get_field_index("timestamp")
    except (KeyError, ValueError):
        return None, None, num_rows

    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None

    for rg_idx in range(pf.metadata.num_row_groups):  # type: ignore[union-attr]
        stats = pf.metadata.row_group(rg_idx).column(col_index).statistics  # type: ignore[union-attr]
        if stats is None or stats.min is None or stats.max is None:
            continue

        rg_min = pd.to_datetime(stats.min, errors="coerce", utc=True)
        rg_max = pd.to_datetime(stats.max, errors="coerce", utc=True)

        if pd.notna(rg_min) and (min_ts is None or rg_min < min_ts):
            min_ts = rg_min
        if pd.notna(rg_max) and (max_ts is None or rg_max > max_ts):
            max_ts = rg_max

    return min_ts, max_ts, num_rows


def ensure_parquet_dataset_path(path: Path) -> Path:
    """Guarantee that path is a directory-based dataset.

    Handles migration from single-file to directory format.

    Args:
        path: Target dataset path.

    Returns:
        Path to the dataset directory.
    """
    ensure_output_dir(path)

    consolidated_file = path.parent / "dataset_raw_final.parquet"
    if consolidated_file.exists() and consolidated_file.is_file():
        logger.info(
            "Found consolidated file %s, converting to base partition", consolidated_file
        )
        path.mkdir(parents=True, exist_ok=True)
        consolidated_file.rename(path / "part-00000.parquet")
        logger.info("Converted consolidated file to base partition part-00000.parquet")
        return path

    if path.exists() and path.is_file():
        migrated_src = path.with_suffix(path.suffix + ".single")
        path.rename(migrated_src)
        path.mkdir(parents=True, exist_ok=True)
        migrated_src.rename(path / "part-00000.parquet")
        logger.info("Converted single parquet file to dataset directory at %s", path)
        return path

    path.mkdir(parents=True, exist_ok=True)
    return path


def append_trades_to_dataset(df: pd.DataFrame) -> None:
    """Append trades to a parquet dataset without loading existing data.

    Args:
        df: DataFrame with trades to append.

    Raises:
        RuntimeError: If pyarrow is not available.
    """
    if pa is None or pq is None:
        msg = "pyarrow is required to append parquet data"
        raise RuntimeError(msg)

    dataset_dir = ensure_parquet_dataset_path(Path(DATASET_RAW_PARQUET))
    table = pa.Table.from_pandas(df, preserve_index=False)  # type: ignore[union-attr]
    filename = f"part-{int(time.time() * 1_000_000)}-{uuid.uuid4().hex[:8]}.parquet"
    target_path = dataset_dir / filename
    pq.write_table(table, target_path, compression="snappy")  # type: ignore[union-attr]
    logger.info("Saved %d trades to %s", len(df), target_path)


__all__ = [
    "append_trades_to_dataset",
    "ensure_parquet_dataset_path",
    "get_existing_timestamp_range",
    "iter_parquet_files",
    "pa",
    "pq",
]
