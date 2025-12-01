"""Main script for clear_features - Complete feature transformation pipeline.

This module orchestrates:
1. Standardization (z-score) - applied before PCA
2. Normalization (min-max for LSTM) - applied before PCA
3. Group-based PCA reduction (fit on train, transform all to avoid leakage)

Pipeline flow:
    features/main.py -> clear_features/main.py -> ready for training

    Input: Raw features (dataset_features*.parquet)
    Output: Transformed and normalized features (same files, overwritten)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
from datetime import datetime
from typing import Any, cast

import gc

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

# Batch size for parquet writing (number of rows per batch)
SAVE_BATCH_SIZE = 500_000

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    INPUT_DATASETS,
    OUTPUT_DATASETS,
    META_COLUMNS,
    PCA_ARTIFACTS_DIR,
    PCA_CONFIG,
    TARGET_COLUMN,
)
from src.clear_features.pca_reducer import GroupPCAReducer, IncrementalGroupPCAReducer
from src.clear_features.scaler_applier import ScalerApplier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a feature dataset."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def save_dataset(df: pd.DataFrame, path: Path, batch_size: int = SAVE_BATCH_SIZE) -> None:
    """Save dataset to parquet by batch (memory efficient)."""
    logger.info(f"Saving dataset to {path} (batch_size={batch_size:,})")

    # Remove duplicate columns (keep first occurrence)
    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Found {len(duplicated_cols)} duplicate columns, removing duplicates: {duplicated_cols[:10]}")
        df = df.loc[:, ~df.columns.duplicated()]

    # Remove technical columns (bar_id, bar_id_lag*) - not needed in output
    cols_to_drop = [c for c in df.columns if c == "bar_id" or c.startswith("bar_id_lag")]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Removed {len(cols_to_drop)} technical columns: {cols_to_drop}")

    # Save by batch using ParquetWriter
    n_rows = len(df)
    n_batches = (n_rows + batch_size - 1) // batch_size

    parquet_writer: pq.ParquetWriter | None = None

    for batch_num in range(n_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, n_rows)

        df_batch = df.iloc[start_idx:end_idx]
        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(path, table.schema, compression="snappy")

        parquet_writer.write_table(table)

        logger.info(f"  Batch {batch_num + 1}/{n_batches}: saved rows {start_idx:,}-{end_idx:,}")

        del df_batch, table
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()

    logger.info(f"Saved {n_rows:,} rows, {len(df.columns)} columns")


def _to_scalar_int(value: Any) -> int:
    """Safely convert pandas/numpy result to scalar int."""
    # Handle scalar types directly
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)

    # Handle pandas Series
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return 0
        if len(value) == 1:
            return int(value.iloc[0])
        # If Series has multiple elements, sum to get scalar
        sum_result = value.sum()
        # Recursively handle the result (should be scalar now)
        return _to_scalar_int(sum_result)

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.size == 1:
            return int(value.item())
        # If array has multiple elements, sum to get scalar
        sum_result = value.sum()
        return _to_scalar_int(sum_result)

    # Try direct conversion, with fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        # Last resort: try to convert to numpy array and sum
        try:
            arr = np.asarray(value)
            return _to_scalar_int(arr)
        except Exception:
            raise ValueError(f"Cannot convert {type(value)} to scalar int: {value}")


def fill_nan_with_zero(df: pd.DataFrame, meta_columns: list[str], target_column: str) -> pd.DataFrame:
    """Fill NaN values with 0 in feature columns (before PCA).

    This handles features like bars_since_* where NaN means "event hasn't occurred yet".

    Args:
        df: DataFrame with features
        meta_columns: Metadata columns to skip
        target_column: Target column to skip

    Returns:
        DataFrame with NaN filled by 0
    """
    df = df.copy()
    excluded = set(meta_columns) | {target_column}
    feature_cols = [c for c in df.columns if c not in excluded and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    if feature_cols:
        df[feature_cols] = df[feature_cols].fillna(0)

    return df


def clean_nan_values(
    df: pd.DataFrame,
    meta_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean NaN values from feature columns.

    Strategy:
    - For numeric features: fill with median computed on TRAIN ONLY (no data leakage)
    - Remove rows with NaN in target or metadata columns

    Args:
        df: DataFrame with features (must have 'split' column)
        meta_columns: Metadata columns to preserve
        target_column: Target column name

    Returns:
        Cleaned DataFrame and stats dict
    """
    stats: dict[str, Any] = {
        "rows_before": len(df),
        "nan_cols": {},
        "rows_removed": 0,
    }

    # Identify feature columns
    feature_cols = [
        c for c in df.columns
        if c not in meta_columns and c != target_column
    ]

    # Get training data mask for computing statistics (prevent data leakage)
    if "split" in df.columns:
        train_mask = df["split"] == "train"
    else:
        # Fallback: use all data if no split column (with warning)
        logger.warning("No 'split' column found - using full dataset for median computation")
        train_mask = pd.Series(True, index=df.index)

    # Count NaN per column before cleaning
    for col in feature_cols:
        nan_count = _to_scalar_int(df[col].isna().sum())
        if nan_count > 0:
            stats["nan_cols"][col] = nan_count

    # Fill NaN in feature columns with median computed on TRAIN ONLY
    for col in feature_cols:
        nan_count = _to_scalar_int(df[col].isna().sum())
        if nan_count > 0:
            # Compute median ONLY on training data to prevent leakage
            median_val = df.loc[train_mask, col].median()
            # Convert to float and check if NaN (handles case where all values are NaN)
            median_float = float(median_val) if isinstance(median_val, (int, float)) else 0.0
            if np.isnan(median_float):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    # Remove rows with NaN in target
    if target_column in df.columns:
        nan_target_mask = df[target_column].isna()
        nan_count = _to_scalar_int(nan_target_mask.sum())
        if nan_count > 0:
            rows_before = len(df)
            df = df.loc[~nan_target_mask].copy()
            stats["rows_removed"] = rows_before - len(df)

    stats["rows_after"] = len(df)
    stats["total_nan_features"] = len(stats["nan_cols"])

    return df, stats


def run_full_pipeline(
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run the complete clear_features pipeline.

    Pipeline order (to avoid data leakage):
    1. Standardization (z-score) - BEFORE PCA
    2. Normalization (min-max for LSTM) - BEFORE PCA
    3. Group-based PCA reduction (fit on train only, transform all)

    Args:
        dry_run: If True, don't overwrite files

    Returns:
        Summary dictionary with results
    """
    logger.info("=" * 60)
    logger.info("CLEAR FEATURES PIPELINE")
    logger.info("=" * 60)
    logger.info("Pipeline: Standardize -> Normalize -> PCA")
    logger.info(f"PCA variance threshold: {PCA_CONFIG['variance_explained_threshold']}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    # Create output directories
    CLEAR_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PCA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "steps": {},
        "datasets": {},
    }

    # =========================================================================
    # STEP 1: Load pre-fitted scalers
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Load pre-fitted scalers")
    logger.info("=" * 60)

    scaler_applier = ScalerApplier()
    scaler_applier.load_scalers()

    # =========================================================================
    # STEP 2: Process each dataset - Apply scaling BEFORE PCA fitting
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Apply standardization/normalization to all datasets")
    logger.info("=" * 60)

    scaled_datasets: dict[str, pd.DataFrame] = {}

    for dataset_name, input_path in INPUT_DATASETS.items():
        logger.info(f"\n--- Scaling {dataset_name} ---")

        # Load dataset
        df = load_dataset(input_path)

        # Apply scaling based on dataset type (BEFORE PCA)
        if dataset_name == "linear":
            logger.info("  Applying z-score standardization...")
            df = scaler_applier.apply_zscore(df)
        elif dataset_name == "lstm":
            logger.info("  Applying min-max normalization...")
            df = scaler_applier.apply_minmax(df)
        else:
            logger.info("  No scaling for tree_based (not needed)")

        scaled_datasets[dataset_name] = df

    # =========================================================================
    # STEP 3: Fit PCA on scaled training data (tree_based as reference)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Group-based PCA reduction")
    logger.info("=" * 60)
    logger.info("Fitting PCA on TRAIN data only to avoid data leakage")

    # Use tree_based as reference for PCA fitting (unscaled, but PCA will work on groups)
    df_ref = scaled_datasets["tree_based"]

    pca_reducer = GroupPCAReducer()
    pca_summary = pca_reducer.fit(df_ref)  # Internally uses only train split
    pca_reducer.save_artifacts()

    results["steps"]["pca"] = {
        "original_features": pca_summary.original_n_features,
        "final_features": pca_summary.final_n_features,
        "groups_processed": len(pca_summary.groups_processed),
        "groups_skipped": len(pca_summary.groups_skipped),
        "features_removed": len(pca_summary.features_removed),
        "components_added": len(pca_summary.features_added),
    }

    logger.info(
        f"PCA: {pca_summary.original_n_features} -> {pca_summary.final_n_features} features"
    )
    logger.info(f"Groups processed: {len(pca_summary.groups_processed)}")

    # =========================================================================
    # STEP 4: Apply PCA transform and save all datasets
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Apply PCA transform and save datasets")
    logger.info("=" * 60)

    for dataset_name in INPUT_DATASETS:
        output_path = OUTPUT_DATASETS[dataset_name]
        logger.info(f"\n--- Processing {dataset_name} ---")

        df = scaled_datasets[dataset_name]
        original_cols = len(df.columns)

        # Apply PCA (fitted on train, applied to all)
        logger.info("  Applying PCA (fitted on train)...")
        df = pca_reducer.transform(df)
        after_pca_cols = len(df.columns)

        # Clean NaN values (fill with median)
        logger.info("  Cleaning NaN values...")
        df, nan_stats = clean_nan_values(df, META_COLUMNS, TARGET_COLUMN)
        if nan_stats["total_nan_features"] > 0:
            logger.info(
                f"  Cleaned NaN in {nan_stats['total_nan_features']} features"
            )
        if nan_stats["rows_removed"] > 0:
            logger.warning(
                f"  Removed {nan_stats['rows_removed']} rows with NaN target"
            )

        final_cols = len(df.columns)

        results["datasets"][dataset_name] = {
            "original_columns": original_cols,
            "after_pca": after_pca_cols,
            "final_columns": final_cols,
            "rows": len(df),
            "nan_cleaned": nan_stats,
        }

        logger.info(f"  {dataset_name}: {original_cols} -> {final_cols} columns")

        # Save to output path (with _clear suffix)
        if not dry_run:
            save_dataset(df, output_path)
        else:
            logger.info(f"  [DRY RUN] Would save to {output_path}")

    # Save run summary
    summary_file = CLEAR_FEATURES_DIR / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("CLEAR FEATURES PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Artifacts saved to: {CLEAR_FEATURES_DIR}")
    logger.info("Datasets ready for training!")

    return results


def run_batch_pipeline(
    batch_size: int = 100_000,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run the clear_features pipeline with batch processing.

    Memory-efficient version that processes data in batches:
    1. Fit PCA incrementally on train batches
    2. Fit scalers incrementally on PCA-transformed train batches
    3. Transform and save each dataset batch-by-batch

    Args:
        batch_size: Number of rows per batch
        dry_run: If True, don't overwrite files

    Returns:
        Summary dictionary with results
    """
    from src.clear_features.scaler_applier import ScalerFitter

    logger.info("=" * 60)
    logger.info("CLEAR FEATURES PIPELINE - BATCH MODE")
    logger.info("=" * 60)
    logger.info(f"Batch size: {batch_size:,}")
    logger.info(f"PCA variance threshold: {PCA_CONFIG['variance_explained_threshold']}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    # Create output directories
    CLEAR_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PCA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "batch_mode": True,
        "batch_size": batch_size,
        "steps": {},
        "datasets": {},
    }

    # =========================================================================
    # STEP 1: Fit PCA incrementally on tree_based (train only)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Incremental PCA fitting on TRAIN data")
    logger.info("=" * 60)

    pca_reducer = IncrementalGroupPCAReducer()

    # Use tree_based as reference dataset for PCA fitting
    ref_path = INPUT_DATASETS["tree_based"]
    parquet_file = pq.ParquetFile(ref_path)
    total_rows = parquet_file.metadata.num_rows

    logger.info(f"Reference dataset: {ref_path}")
    logger.info(f"Total rows: {total_rows:,}")

    train_batches_count = 0
    train_rows_total = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()

        # Filter to train only for PCA fitting
        if "split" in df_batch.columns:
            df_train = df_batch[df_batch["split"] == "train"]
        else:
            logger.warning("No 'split' column found, using entire batch")
            df_train = df_batch

        if len(df_train) > 0:
            # Fill NaN with 0 before PCA fitting
            df_train = fill_nan_with_zero(df_train, META_COLUMNS, TARGET_COLUMN)
            train_batches_count += 1
            train_rows_total += len(df_train)
            pca_reducer.partial_fit(df_train)

        del df_batch, df_train
        gc.collect()

    logger.info(f"PCA partial_fit complete: {train_batches_count} batches, {train_rows_total:,} train rows")

    # Finalize PCA fitting
    pca_summary = pca_reducer.finalize_fit()
    pca_reducer.save_artifacts()

    results["steps"]["pca"] = {
        "original_features": pca_summary.original_n_features,
        "final_features": pca_summary.final_n_features,
        "groups_processed": len(pca_summary.groups_processed),
        "groups_skipped": len(pca_summary.groups_skipped),
        "features_removed": len(pca_summary.features_removed),
        "components_added": len(pca_summary.features_added),
    }

    logger.info(
        f"PCA: {pca_summary.original_n_features} -> {len(pca_summary.features_added)} components"
    )

    # =========================================================================
    # STEP 2: Fit scalers on PCA-transformed TRAIN data
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Fit scalers on PCA-transformed TRAIN data")
    logger.info("=" * 60)

    scaler_fitter = ScalerFitter()

    # Second pass through train data to fit scalers on PCA-transformed features
    parquet_file = pq.ParquetFile(ref_path)
    scaler_batches = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()

        # Filter to train only
        if "split" in df_batch.columns:
            df_train = df_batch[df_batch["split"] == "train"]
        else:
            df_train = df_batch

        if len(df_train) > 0:
            # Fill NaN and apply PCA transform
            df_train = fill_nan_with_zero(df_train, META_COLUMNS, TARGET_COLUMN)
            df_train = pca_reducer.transform(df_train)

            # Fit scalers on PCA-transformed data
            scaler_fitter.partial_fit(df_train)
            scaler_batches += 1

        del df_batch, df_train
        gc.collect()

    logger.info(f"Scaler fitting complete: {scaler_batches} batches")

    # Finalize scaler fitting
    zscore_scaler, minmax_scaler = scaler_fitter.finalize()

    # Create scaler applier with fitted scalers
    scaler_applier = ScalerApplier()
    scaler_applier.set_scalers(zscore_scaler, minmax_scaler)
    scaler_applier.save_scalers()

    results["steps"]["scalers"] = {
        "columns_fitted": len(zscore_scaler.columns_) if zscore_scaler.columns_ else 0,
    }

    # =========================================================================
    # STEP 3: Transform and save each dataset batch-by-batch
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Transform and save datasets (batch mode)")
    logger.info("=" * 60)

    for dataset_name in INPUT_DATASETS:
        input_path = INPUT_DATASETS[dataset_name]
        output_path = OUTPUT_DATASETS[dataset_name]

        logger.info(f"\n--- Processing {dataset_name} ---")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")

        if dry_run:
            logger.info(f"  [DRY RUN] Would transform {input_path} -> {output_path}")
            continue

        parquet_file = pq.ParquetFile(input_path)
        total_rows = parquet_file.metadata.num_rows
        n_batches = (total_rows + batch_size - 1) // batch_size

        logger.info(f"  Total rows: {total_rows:,}, batches: {n_batches}")

        writer: pq.ParquetWriter | None = None
        rows_processed = 0
        batch_num = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_num += 1
            df_batch = batch.to_pandas()

            # Fill NaN with 0 before any transformation
            df_batch = fill_nan_with_zero(df_batch, META_COLUMNS, TARGET_COLUMN)

            # Apply PCA transform
            df_batch = pca_reducer.transform(df_batch)

            # Apply scaling AFTER PCA based on dataset type
            if dataset_name == "linear":
                df_batch = scaler_applier.apply_zscore(df_batch)
            elif dataset_name == "lstm":
                df_batch = scaler_applier.apply_minmax(df_batch)

            # Clean any remaining NaN values
            df_batch, nan_stats = clean_nan_values(df_batch, META_COLUMNS, TARGET_COLUMN)

            # Remove technical columns (bar_id, bar_id_lag*) - not needed in output
            cols_to_drop = [c for c in df_batch.columns if c == "bar_id" or c.startswith("bar_id_lag")]
            if cols_to_drop:
                df_batch = df_batch.drop(columns=cols_to_drop)

            rows_processed += len(df_batch)

            # Write batch
            table = pa.Table.from_pandas(df_batch, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

            writer.write_table(table)

            if batch_num % 10 == 0 or batch_num == n_batches:
                logger.info(f"  Batch {batch_num}/{n_batches}: {rows_processed:,} rows processed")

            del df_batch, table
            gc.collect()

        if writer is not None:
            writer.close()
            logger.info(f"  Saved {rows_processed:,} rows to {output_path}")

        results["datasets"][dataset_name] = {
            "rows": rows_processed,
        }

    # Save run summary
    summary_file = CLEAR_FEATURES_DIR / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("CLEAR FEATURES PIPELINE COMPLETE (BATCH MODE)")
    logger.info("=" * 60)
    logger.info(f"Artifacts saved to: {CLEAR_FEATURES_DIR}")
    logger.info("Datasets ready for training!")

    return results


def main():
    """Main entry point - uses batch mode automatically."""
    run_batch_pipeline(batch_size=100_000, dry_run=False)


if __name__ == "__main__":
    main()
