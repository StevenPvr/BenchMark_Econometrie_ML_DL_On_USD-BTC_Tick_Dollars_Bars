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

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    DATASETS,
    META_COLUMNS,
    PCA_ARTIFACTS_DIR,
    PCA_CONFIG,
    TARGET_COLUMN,
)
from src.clear_features.pca_reducer import GroupPCAReducer
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


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save dataset to parquet (overwrites existing)."""
    logger.info(f"Saving dataset to {path}")

    # Remove duplicate columns (keep first occurrence)
    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Found {len(duplicated_cols)} duplicate columns, removing duplicates: {duplicated_cols[:10]}")
        df = df.loc[:, ~df.columns.duplicated()]

    # Remove bar_id column if it exists
    if "bar_id" in df.columns:
        df = df.drop(columns=["bar_id"])
        logger.info("Removed 'bar_id' column")

    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns")


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

    for dataset_name, dataset_path in DATASETS.items():
        logger.info(f"\n--- Scaling {dataset_name} ---")

        # Load dataset
        df = load_dataset(dataset_path)

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

    for dataset_name, dataset_path in DATASETS.items():
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

        # Save (overwrite)
        if not dry_run:
            save_dataset(df, dataset_path)
        else:
            logger.info(f"  [DRY RUN] Would save to {dataset_path}")

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


def main():
    """Main entry point."""
    run_full_pipeline(dry_run=False)


if __name__ == "__main__":
    main()
