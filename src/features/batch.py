"""Batch processing pipeline for large datasets.

This module handles memory-efficient feature engineering for large datasets
using PyArrow iter_batches to minimize memory usage.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from src.utils import get_logger
from src.features.compute import apply_lags, compute_all_features
from src.features.pipeline import (
    TRAIN_RATIO,
    compute_timestamp_features,
    drop_initial_nan_rows,
    drop_timestamp_columns,
    interpolate_sporadic_nan,
    shift_target_to_future_return,
)
from src.path import (
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_PARQUET,
    DOLLAR_BARS_PARQUET,
    FEATURES_DIR,
    SCALERS_DIR,
)

logger = get_logger(__name__)

# Default batch size (number of bars per batch)
DEFAULT_BATCH_SIZE = 200_000

# Overlap to handle rolling windows and lags at batch boundaries
# Should be >= max(rolling window size, max lag) used in features
BATCH_OVERLAP = 200

__all__ = [
    "run_batch_pipeline",
    "DEFAULT_BATCH_SIZE",
    "BATCH_OVERLAP",
]


def run_batch_pipeline(
    batch_size: int = DEFAULT_BATCH_SIZE,
    overlap: int = BATCH_OVERLAP,
    sample_fraction: float = 1.0,
) -> None:
    """Run feature engineering with batch processing for large datasets.

    Processes dollar bars in batches using PyArrow iter_batches to minimize
    memory usage. Maintains an overlap buffer between batches to handle
    rolling windows and lags correctly.

    Train/test split is applied DURING batch processing to avoid reloading
    the entire dataset at the end.

    Args:
        batch_size: Number of bars per batch (default 200K).
        overlap: Number of rows overlap between batches for rolling windows/lags.
        sample_fraction: Fraction of data to use (0.0-1.0). Default 1.0 (all data).
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE - BATCH MODE")
    logger.info("=" * 60)
    logger.info(f"  Batch size: {batch_size:,}")
    logger.info(f"  Overlap: {overlap:,}")
    if sample_fraction < 1.0:
        logger.info(f"  Sample fraction: {sample_fraction:.1%} (TEST MODE)")

    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(
            f"Input file not found: {DOLLAR_BARS_PARQUET}. "
            "Please run data_preparation first."
        )

    parquet_file = pq.ParquetFile(DOLLAR_BARS_PARQUET)
    total_rows_full = parquet_file.metadata.num_rows

    if sample_fraction < 1.0:
        total_rows = int(total_rows_full * sample_fraction)
        logger.info(f"  Full dataset: {total_rows_full:,} bars")
        logger.info(f"  Sampling {sample_fraction:.1%}: {total_rows:,} bars")
    else:
        total_rows = total_rows_full

    logger.info(f"  Total input bars: {total_rows:,}")
    logger.info(f"  Train/test split: {TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    batch_output_dir = FEATURES_DIR / "batches"

    if batch_output_dir.exists():
        for old_batch in batch_output_dir.glob("*.parquet"):
            old_batch.unlink()
        logger.info("  Cleaned up old batch files")

    batch_output_dir.mkdir(parents=True, exist_ok=True)

    batch_files: list[Path] = []
    total_features_rows = 0

    overlap_buffer: pd.DataFrame | None = None
    last_processed_timestamp: int | None = None
    batch_num = 0
    input_rows_processed = 0

    for arrow_batch in parquet_file.iter_batches(batch_size=batch_size):
        if input_rows_processed >= total_rows:
            logger.info(f"  Reached sample limit ({total_rows:,} rows), stopping")
            break

        df_batch = arrow_batch.to_pandas()

        remaining_rows = total_rows - input_rows_processed
        if len(df_batch) > remaining_rows:
            df_batch = df_batch.iloc[:remaining_rows].copy()
            logger.info(f"  Truncated batch to {len(df_batch):,} rows (sample limit)")

        input_rows_processed += len(df_batch)
        batch_num += 1

        logger.info(f"\n{'=' * 50}")
        logger.info(f"BATCH {batch_num}")
        logger.info(f"  Raw batch rows: {len(df_batch):,}")

        if overlap_buffer is not None:
            df_batch = pd.concat([overlap_buffer, df_batch], ignore_index=True)
            logger.info(f"  With overlap: {len(df_batch):,} rows")

        if len(df_batch) > overlap:
            overlap_buffer = df_batch.tail(overlap).copy()
        else:
            overlap_buffer = df_batch.copy()

        df_batch = compute_timestamp_features(df_batch)
        df_features = compute_all_features(df_batch)

        del df_batch
        gc.collect()

        df_features_lagged = apply_lags(df_features)

        del df_features
        gc.collect()

        df_features_clean = drop_initial_nan_rows(df_features_lagged)

        del df_features_lagged
        gc.collect()

        df_features_clean = interpolate_sporadic_nan(df_features_clean)
        df_features_clean = shift_target_to_future_return(
            df_features_clean, target_col="log_return"
        )

        if last_processed_timestamp is not None and len(df_features_clean) > 0:
            ts_col = None
            for col in ["timestamp_close", "timestamp_open", "timestamp"]:
                if col in df_features_clean.columns:
                    ts_col = col
                    break

            if ts_col is not None:
                before_filter = len(df_features_clean)
                df_features_clean = df_features_clean[
                    df_features_clean[ts_col] > last_processed_timestamp
                ].copy()
                filtered_count = before_filter - len(df_features_clean)
                if filtered_count > 0:
                    logger.info(
                        f"  Filtered {filtered_count} overlap rows "
                        f"(timestamp <= {last_processed_timestamp})"
                    )
            else:
                rows_to_skip = min(overlap, len(df_features_clean) - 1)
                if rows_to_skip > 0:
                    df_features_clean = df_features_clean.iloc[rows_to_skip:].copy()
                    logger.info(
                        f"  After removing overlap prefix (index-based): "
                        f"{len(df_features_clean):,} rows"
                    )

        if len(df_features_clean) == 0:
            logger.warning(f"  Batch {batch_num} produced no valid rows, skipping")
            continue

        ts_col = None
        for col in ["timestamp_close", "timestamp_open", "timestamp"]:
            if col in df_features_clean.columns:
                ts_col = col
                break
        if ts_col is not None:
            last_processed_timestamp = int(df_features_clean[ts_col].max())

        batch_file = batch_output_dir / f"batch_{batch_num:04d}.parquet"
        df_features_clean.to_parquet(batch_file, index=False)
        batch_files.append(batch_file)
        total_features_rows += len(df_features_clean)

        logger.info(f"  Saved {len(df_features_clean):,} rows to {batch_file.name}")

        del df_features_clean
        gc.collect()

    # Consolidate batches
    _consolidate_batches(batch_files, total_features_rows, batch_output_dir)


def _consolidate_batches(
    batch_files: list[Path],
    total_features_rows: int,
    batch_output_dir: Path,
) -> None:
    """Consolidate batch files into final output with train/test split.

    Args:
        batch_files: List of batch parquet files.
        total_features_rows: Total number of rows across all batches.
        batch_output_dir: Directory containing batch files.
    """
    logger.info(f"\n{'=' * 50}")
    logger.info("CONSOLIDATING BATCHES WITH TRAIN/TEST SPLIT")
    logger.info(f"  Total batch files: {len(batch_files)}")
    logger.info(f"  Total rows: {total_features_rows:,}")

    if not batch_files:
        raise ValueError("No batches were generated")

    train_split_idx = int(total_features_rows * TRAIN_RATIO)
    logger.info(f"  Train split at row: {train_split_idx:,}")

    # Collect all unique columns
    logger.info("  Scanning batch schemas...")
    all_columns: set[str] = set()
    for batch_file in batch_files:
        pf = pq.ParquetFile(batch_file)
        schema = pf.schema_arrow
        batch_cols = set(schema.names)
        all_columns.update(batch_cols)

    unified_columns = sorted(all_columns)
    logger.info(f"  Unified schema: {len(unified_columns)} columns")

    cumulative_rows = 0

    writers: dict[str, pq.ParquetWriter | None] = {
        "tree_based": None,
        "linear": None,
        "lstm": None,
    }
    output_paths = {
        "tree_based": DATASET_FEATURES_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_PARQUET,
    }

    for batch_idx, batch_file in enumerate(batch_files):
        logger.info(
            f"  Processing batch {batch_idx + 1}/{len(batch_files)}: {batch_file.name}"
        )

        df_batch = pd.read_parquet(batch_file)
        batch_len = len(df_batch)

        missing_cols = set(unified_columns) - set(df_batch.columns)
        if missing_cols:
            missing_df = pd.DataFrame(
                np.nan,
                index=df_batch.index,
                columns=pd.Index(list(missing_cols)),
            )
            df_batch = pd.concat([df_batch, missing_df], axis=1)
            df_batch = df_batch[unified_columns].copy()

        batch_start = cumulative_rows
        batch_end = cumulative_rows + batch_len

        if batch_end <= train_split_idx:
            df_batch = df_batch.copy()
            df_batch["split"] = "train"
        elif batch_start >= train_split_idx:
            if not missing_cols:
                df_batch = df_batch.copy()
            df_batch["split"] = "test"
        else:
            local_split = train_split_idx - batch_start
            if not missing_cols:
                df_batch = df_batch.copy()
            df_batch["split"] = "test"
            df_batch.iloc[:local_split, df_batch.columns.get_loc("split")] = "train"

        cumulative_rows = batch_end

        df_batch = drop_timestamp_columns(cast(pd.DataFrame, df_batch))
        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        for key in writers:
            if writers[key] is None:
                try:
                    writers[key] = pq.ParquetWriter(
                        output_paths[key],
                        table.schema,
                        compression="snappy",
                    )
                except Exception as e:
                    logger.error(f"Failed to create ParquetWriter for {key}: {e}")
                    continue
            if writers[key] is not None:
                cast(pq.ParquetWriter, writers[key]).write_table(table)

        del df_batch, table
        gc.collect()

    logger.info("  All batches consolidated")

    for key in writers:
        if writers[key] is not None:
            cast(pq.ParquetWriter, writers[key]).close()

    if batch_output_dir.exists():
        shutil.rmtree(batch_output_dir)
        logger.info(f"  Cleaned up batch directory: {batch_output_dir}")

    logger.info(f"  Saved to: {DATASET_FEATURES_PARQUET}")
    logger.info(f"  Saved to: {DATASET_FEATURES_LINEAR_PARQUET}")
    logger.info(f"  Saved to: {DATASET_FEATURES_LSTM_PARQUET}")

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE (BATCH MODE)")
    logger.info("=" * 60)
    logger.info(f"  Total features rows: {total_features_rows:,}")
    logger.info(f"  Train rows: {train_split_idx:,}")
    logger.info(f"  Test rows: {total_features_rows - train_split_idx:,}")
    logger.info("")
    logger.info("NEXT STEP: Run clear_features to apply:")
    logger.info("  1. PCA reduction on correlated features")
    logger.info("  2. Scaler fitting on PCA-transformed features (train only)")
    logger.info("  3. Normalization (z-score for linear, minmax for LSTM)")
