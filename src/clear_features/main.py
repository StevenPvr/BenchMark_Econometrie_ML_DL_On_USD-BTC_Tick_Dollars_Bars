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

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    INPUT_DATASETS,
    META_COLUMNS,
    OUTPUT_DATASETS,
    PCA_ARTIFACTS_DIR,
    PCA_CONFIG,
    SAVE_BATCH_SIZE,
    TARGET_COLUMN,
)
from src.clear_features.pca_reducer import GroupPCAReducer, IncrementalGroupPCAReducer
from src.clear_features.scaler_applier import ScalerApplier, ScalerFitter
from src.config_logging import get_logger

logger = get_logger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a feature dataset from parquet file.

    Args:
        path: Path to the parquet file.

    Returns:
        DataFrame with loaded features.
    """
    logger.info("Loading dataset from %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def _remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns from DataFrame, keeping first occurrence.

    Args:
        df: DataFrame potentially containing duplicate columns.

    Returns:
        DataFrame with duplicate columns removed.
    """
    if not df.columns.duplicated().any():
        return df

    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    logger.warning(
        "Found %d duplicate columns, removing duplicates: %s",
        len(duplicated_cols),
        duplicated_cols[:10],
    )
    return df.loc[:, ~df.columns.duplicated()]


def _remove_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove technical columns (bar_id, bar_id_lag*) not needed in output.

    Args:
        df: DataFrame containing potential technical columns.

    Returns:
        DataFrame with technical columns removed.
    """
    cols_to_drop = [
        c for c in df.columns if c == "bar_id" or c.startswith("bar_id_lag")
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Removed %d technical columns: %s", len(cols_to_drop), cols_to_drop)
    return df


def save_dataset(
    df: pd.DataFrame, path: Path, batch_size: int = SAVE_BATCH_SIZE
) -> None:
    """Save dataset to parquet by batch (memory efficient).

    Args:
        df: DataFrame to save.
        path: Output path for the parquet file.
        batch_size: Number of rows per batch for writing.
    """
    logger.info("Saving dataset to %s (batch_size=%d)", path, batch_size)

    df = _remove_duplicate_columns(df)
    df = _remove_technical_columns(df)

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
        logger.info(
            "  Batch %d/%d: saved rows %d-%d", batch_num + 1, n_batches, start_idx, end_idx
        )

        del df_batch, table
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()

    logger.info("Saved %d rows, %d columns", n_rows, len(df.columns))


def _to_scalar_int(value: Any) -> int:
    """Safely convert pandas/numpy result to scalar int.

    Args:
        value: Value to convert (scalar, Series, or ndarray).

    Returns:
        Integer representation of the value.

    Raises:
        ValueError: If conversion fails.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)

    if isinstance(value, pd.Series):
        if len(value) == 0:
            return 0
        if len(value) == 1:
            return int(value.iloc[0])
        return _to_scalar_int(value.sum())

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.size == 1:
            return int(value.item())
        return _to_scalar_int(value.sum())

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            arr = np.asarray(value)
            return _to_scalar_int(arr)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert {type(value)} to scalar int: {value}"
            ) from exc


def fill_nan_with_zero(
    df: pd.DataFrame, meta_columns: list[str], target_column: str
) -> pd.DataFrame:
    """Fill NaN values with 0 in feature columns (before PCA).

    Handles features like bars_since_* where NaN means "event hasn't occurred yet".

    Args:
        df: DataFrame with features.
        meta_columns: Metadata columns to skip.
        target_column: Target column to skip.

    Returns:
        DataFrame with NaN filled by 0.
    """
    df = df.copy()
    excluded = set(meta_columns) | {target_column}
    numeric_dtypes = [np.float64, np.float32, np.int64, np.int32]
    feature_cols = [
        c for c in df.columns if c not in excluded and df[c].dtype in numeric_dtypes
    ]

    if feature_cols:
        df[feature_cols] = df[feature_cols].fillna(0)

    return df


def _compute_feature_columns(
    df: pd.DataFrame, meta_columns: list[str], target_column: str
) -> list[str]:
    """Get feature columns excluding metadata and target.

    Args:
        df: DataFrame with features.
        meta_columns: Metadata columns to exclude.
        target_column: Target column to exclude.

    Returns:
        List of feature column names.
    """
    return [c for c in df.columns if c not in meta_columns and c != target_column]


def _get_train_mask(df: pd.DataFrame) -> pd.Series:
    """Get boolean mask for training data.

    Args:
        df: DataFrame with potential 'split' column.

    Returns:
        Boolean Series indicating training rows.
    """
    if "split" in df.columns:
        return df["split"] == "train"
    logger.warning("No 'split' column found - using full dataset for median computation")
    return pd.Series(True, index=df.index)


def _count_nan_per_column(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, int]:
    """Count NaN values per feature column.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature column names.

    Returns:
        Dict mapping column name to NaN count (only non-zero counts).
    """
    nan_cols = {}
    for col in feature_cols:
        nan_count = _to_scalar_int(df[col].isna().sum())
        if nan_count > 0:
            nan_cols[col] = nan_count
    return nan_cols


def _fill_nan_with_train_median(
    df: pd.DataFrame, feature_cols: list[str], train_mask: pd.Series
) -> pd.DataFrame:
    """Fill NaN in feature columns with median computed on train only.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature column names.
        train_mask: Boolean mask for training rows.

    Returns:
        DataFrame with NaN filled.
    """
    for col in feature_cols:
        nan_count = _to_scalar_int(df[col].isna().sum())
        if nan_count > 0:
            median_val = df.loc[train_mask, col].median()
            median_float = float(median_val) if isinstance(median_val, (int, float)) else 0.0
            if np.isnan(median_float):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
    return df


def clean_nan_values(
    df: pd.DataFrame,
    meta_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean NaN values from feature columns.

    Strategy:
    - For numeric features: fill with median computed on TRAIN ONLY (no data leakage)
    - Remove rows with NaN in target or metadata columns

    Args:
        df: DataFrame with features (must have 'split' column).
        meta_columns: Metadata columns to preserve.
        target_column: Target column name.

    Returns:
        Tuple of (cleaned DataFrame, stats dict).
    """
    stats: dict[str, Any] = {
        "rows_before": len(df),
        "nan_cols": {},
        "rows_removed": 0,
    }

    feature_cols = _compute_feature_columns(df, meta_columns, target_column)
    train_mask = _get_train_mask(df)

    stats["nan_cols"] = _count_nan_per_column(df, feature_cols)
    df = _fill_nan_with_train_median(df, feature_cols, train_mask)

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


def _log_pipeline_header(dry_run: bool) -> None:
    """Log pipeline header information."""
    logger.info("=" * 60)
    logger.info("CLEAR FEATURES PIPELINE")
    logger.info("=" * 60)
    logger.info("Pipeline: Standardize -> Normalize -> PCA")
    logger.info("PCA variance threshold: %s", PCA_CONFIG["variance_explained_threshold"])
    logger.info("Dry run: %s", dry_run)
    logger.info("=" * 60)


def _scale_dataset(
    df: pd.DataFrame, dataset_name: str, scaler_applier: ScalerApplier
) -> pd.DataFrame:
    """Apply appropriate scaling based on dataset type.

    Args:
        df: DataFrame to scale.
        dataset_name: Name of dataset (linear, lstm, or tree_based).
        scaler_applier: Scaler applier instance.

    Returns:
        Scaled DataFrame.
    """
    if dataset_name == "linear":
        logger.info("  Applying z-score standardization...")
        return scaler_applier.apply_zscore(df)
    elif dataset_name == "lstm":
        logger.info("  Applying min-max normalization...")
        return scaler_applier.apply_minmax(df)
    else:
        logger.info("  No scaling for tree_based (not needed)")
        return df


def run_full_pipeline(dry_run: bool = False) -> dict[str, Any]:
    """Run the complete clear_features pipeline.

    Pipeline order (to avoid data leakage):
    1. Standardization (z-score) - BEFORE PCA
    2. Normalization (min-max for LSTM) - BEFORE PCA
    3. Group-based PCA reduction (fit on train only, transform all)

    Args:
        dry_run: If True, don't overwrite files.

    Returns:
        Summary dictionary with results.
    """
    _log_pipeline_header(dry_run)

    CLEAR_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PCA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "steps": {},
        "datasets": {},
    }

    # STEP 1: Load pre-fitted scalers
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Load pre-fitted scalers")
    logger.info("=" * 60)

    scaler_applier = ScalerApplier()
    scaler_applier.load_scalers()

    # STEP 2: Apply scaling to all datasets
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Apply standardization/normalization to all datasets")
    logger.info("=" * 60)

    scaled_datasets: dict[str, pd.DataFrame] = {}
    for dataset_name, input_path in INPUT_DATASETS.items():
        logger.info("\n--- Scaling %s ---", dataset_name)
        df = load_dataset(input_path)
        df = _scale_dataset(df, dataset_name, scaler_applier)
        scaled_datasets[dataset_name] = df

    # STEP 3: Fit PCA on scaled training data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Group-based PCA reduction")
    logger.info("=" * 60)
    logger.info("Fitting PCA on TRAIN data only to avoid data leakage")

    df_ref = scaled_datasets["tree_based"]
    pca_reducer = GroupPCAReducer()
    pca_summary = pca_reducer.fit(df_ref)
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
        "PCA: %d -> %d features",
        pca_summary.original_n_features,
        pca_summary.final_n_features,
    )
    logger.info("Groups processed: %d", len(pca_summary.groups_processed))

    # STEP 4: Apply PCA transform and save all datasets
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Apply PCA transform and save datasets")
    logger.info("=" * 60)

    for dataset_name in INPUT_DATASETS:
        output_path = OUTPUT_DATASETS[dataset_name]
        logger.info("\n--- Processing %s ---", dataset_name)

        df = scaled_datasets[dataset_name]
        original_cols = len(df.columns)

        logger.info("  Applying PCA (fitted on train)...")
        df = pca_reducer.transform(df)
        after_pca_cols = len(df.columns)

        logger.info("  Cleaning NaN values...")
        df, nan_stats = clean_nan_values(df, META_COLUMNS, TARGET_COLUMN)
        if nan_stats["total_nan_features"] > 0:
            logger.info("  Cleaned NaN in %d features", nan_stats["total_nan_features"])
        if nan_stats["rows_removed"] > 0:
            logger.warning("  Removed %d rows with NaN target", nan_stats["rows_removed"])

        final_cols = len(df.columns)

        results["datasets"][dataset_name] = {
            "original_columns": original_cols,
            "after_pca": after_pca_cols,
            "final_columns": final_cols,
            "rows": len(df),
            "nan_cleaned": nan_stats,
        }

        logger.info("  %s: %d -> %d columns", dataset_name, original_cols, final_cols)

        if not dry_run:
            save_dataset(df, output_path)
        else:
            logger.info("  [DRY RUN] Would save to %s", output_path)

    # Save run summary
    summary_file = CLEAR_FEATURES_DIR / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("CLEAR FEATURES PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Artifacts saved to: %s", CLEAR_FEATURES_DIR)
    logger.info("Datasets ready for training!")

    return results


def _fit_pca_incrementally(
    pca_reducer: IncrementalGroupPCAReducer,
    ref_path: Path,
    batch_size: int,
) -> tuple[int, int]:
    """Fit PCA incrementally on training data batches.

    Args:
        pca_reducer: Incremental PCA reducer instance.
        ref_path: Path to reference dataset.
        batch_size: Number of rows per batch.

    Returns:
        Tuple of (number of batches, total train rows).
    """
    parquet_file = pq.ParquetFile(ref_path)
    total_rows = parquet_file.metadata.num_rows
    logger.info("Reference dataset: %s", ref_path)
    logger.info("Total rows: %d", total_rows)

    train_batches_count = 0
    train_rows_total = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()

        if "split" in df_batch.columns:
            df_train = df_batch[df_batch["split"] == "train"]
        else:
            logger.warning("No 'split' column found, using entire batch")
            df_train = df_batch

        if len(df_train) > 0:
            df_train = fill_nan_with_zero(df_train, META_COLUMNS, TARGET_COLUMN)
            train_batches_count += 1
            train_rows_total += len(df_train)
            pca_reducer.partial_fit(df_train)

        del df_batch, df_train
        gc.collect()

    logger.info(
        "PCA partial_fit complete: %d batches, %d train rows",
        train_batches_count,
        train_rows_total,
    )
    return train_batches_count, train_rows_total


def _fit_scalers_on_pca_data(
    scaler_fitter: ScalerFitter,
    pca_reducer: IncrementalGroupPCAReducer,
    ref_path: Path,
    batch_size: int,
) -> int:
    """Fit scalers on PCA-transformed training data.

    Args:
        scaler_fitter: Scaler fitter instance.
        pca_reducer: Fitted PCA reducer.
        ref_path: Path to reference dataset.
        batch_size: Number of rows per batch.

    Returns:
        Number of batches processed.
    """
    parquet_file = pq.ParquetFile(ref_path)
    scaler_batches = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()

        if "split" in df_batch.columns:
            df_train = df_batch[df_batch["split"] == "train"]
        else:
            df_train = df_batch

        if len(df_train) > 0:
            df_train = fill_nan_with_zero(df_train, META_COLUMNS, TARGET_COLUMN)
            df_train = pca_reducer.transform(df_train)
            scaler_fitter.partial_fit(df_train)
            scaler_batches += 1

        del df_batch, df_train
        gc.collect()

    logger.info("Scaler fitting complete: %d batches", scaler_batches)
    return scaler_batches


def _transform_and_save_dataset(
    dataset_name: str,
    input_path: Path,
    output_path: Path,
    pca_reducer: IncrementalGroupPCAReducer,
    scaler_applier: ScalerApplier,
    batch_size: int,
) -> int:
    """Transform a single dataset and save to output path.

    Args:
        dataset_name: Name of dataset (linear, lstm, or tree_based).
        input_path: Path to input parquet file.
        output_path: Path for output parquet file.
        pca_reducer: Fitted PCA reducer.
        scaler_applier: Scaler applier instance.
        batch_size: Number of rows per batch.

    Returns:
        Total rows processed.
    """
    logger.info("\n--- Processing %s ---", dataset_name)
    logger.info("  Input: %s", input_path)
    logger.info("  Output: %s", output_path)

    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    n_batches = (total_rows + batch_size - 1) // batch_size

    logger.info("  Total rows: %d, batches: %d", total_rows, n_batches)

    writer: pq.ParquetWriter | None = None
    rows_processed = 0
    batch_num = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_num += 1
        df_batch = batch.to_pandas()

        df_batch = fill_nan_with_zero(df_batch, META_COLUMNS, TARGET_COLUMN)
        df_batch = pca_reducer.transform(df_batch)

        if dataset_name == "linear":
            df_batch = scaler_applier.apply_zscore(df_batch)
        elif dataset_name == "lstm":
            df_batch = scaler_applier.apply_minmax(df_batch)

        df_batch, _ = clean_nan_values(df_batch, META_COLUMNS, TARGET_COLUMN)
        df_batch = _remove_technical_columns(df_batch)

        rows_processed += len(df_batch)

        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

        writer.write_table(table)

        if batch_num % 10 == 0 or batch_num == n_batches:
            logger.info(
                "  Batch %d/%d: %d rows processed", batch_num, n_batches, rows_processed
            )

        del df_batch, table
        gc.collect()

    if writer is not None:
        writer.close()
        logger.info("  Saved %d rows to %s", rows_processed, output_path)

    return rows_processed


def run_batch_pipeline(
    batch_size: int = 100_000,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the clear_features pipeline with batch processing.

    Memory-efficient version that processes data in batches:
    1. Fit PCA incrementally on train batches
    2. Fit scalers incrementally on PCA-transformed train batches
    3. Transform and save each dataset batch-by-batch

    Args:
        batch_size: Number of rows per batch.
        dry_run: If True, don't overwrite files.

    Returns:
        Summary dictionary with results.
    """
    logger.info("=" * 60)
    logger.info("CLEAR FEATURES PIPELINE - BATCH MODE")
    logger.info("=" * 60)
    logger.info("Batch size: %d", batch_size)
    logger.info("PCA variance threshold: %s", PCA_CONFIG["variance_explained_threshold"])
    logger.info("Dry run: %s", dry_run)
    logger.info("=" * 60)

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

    # STEP 1: Fit PCA incrementally
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Incremental PCA fitting on TRAIN data")
    logger.info("=" * 60)

    pca_reducer = IncrementalGroupPCAReducer()
    ref_path = INPUT_DATASETS["tree_based"]

    _fit_pca_incrementally(pca_reducer, ref_path, batch_size)
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
        "PCA: %d -> %d components",
        pca_summary.original_n_features,
        len(pca_summary.features_added),
    )

    # STEP 2: Fit scalers on PCA-transformed train data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Fit scalers on PCA-transformed TRAIN data")
    logger.info("=" * 60)

    scaler_fitter = ScalerFitter()
    _fit_scalers_on_pca_data(scaler_fitter, pca_reducer, ref_path, batch_size)

    zscore_scaler, minmax_scaler = scaler_fitter.finalize()
    scaler_applier = ScalerApplier()
    scaler_applier.set_scalers(zscore_scaler, minmax_scaler)
    scaler_applier.save_scalers()

    results["steps"]["scalers"] = {
        "columns_fitted": len(zscore_scaler.columns_) if zscore_scaler.columns_ else 0,
    }

    # STEP 3: Transform and save each dataset
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Transform and save datasets (batch mode)")
    logger.info("=" * 60)

    for dataset_name in INPUT_DATASETS:
        input_path = INPUT_DATASETS[dataset_name]
        output_path = OUTPUT_DATASETS[dataset_name]

        if dry_run:
            logger.info(
                "  [DRY RUN] Would transform %s -> %s", input_path, output_path
            )
            continue

        rows_processed = _transform_and_save_dataset(
            dataset_name,
            input_path,
            output_path,
            pca_reducer,
            scaler_applier,
            batch_size,
        )

        results["datasets"][dataset_name] = {"rows": rows_processed}

    # Save run summary
    summary_file = CLEAR_FEATURES_DIR / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("CLEAR FEATURES PIPELINE COMPLETE (BATCH MODE)")
    logger.info("=" * 60)
    logger.info("Artifacts saved to: %s", CLEAR_FEATURES_DIR)
    logger.info("Datasets ready for training!")

    return results


def main() -> None:
    """Main entry point - uses batch mode automatically."""
    run_batch_pipeline(batch_size=100_000, dry_run=False)


if __name__ == "__main__":
    main()
