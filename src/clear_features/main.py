"""Main script for clear_features - Complete feature transformation pipeline.

This module orchestrates:
1. Non-linear correlation analysis (Spearman) to identify correlated feature groups
2. PCA reduction on correlated feature clusters
3. Log transformation on non-stationary features
4. Normalization (z-score for linear, minmax for LSTM)

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
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    CORRELATION_CONFIG,
    DATASETS,
    LOG_TRANSFORM_ARTIFACTS_DIR,
    META_COLUMNS,
    PCA_ARTIFACTS_DIR,
    PCA_CONFIG,
    REFERENCE_DATASET,
    TARGET_COLUMN,
)
from src.clear_features.log_transformer import LogTransformer
from src.clear_features.nonlinear_correlation import NonLinearCorrelationAnalyzer
from src.clear_features.pca_reducer import WeightedPCAReducer
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


def get_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract training data based on split column."""
    if "split" not in df.columns:
        raise ValueError("Dataset must have 'split' column")

    train_mask = df["split"] == "train"
    df_train = df.loc[train_mask].copy()
    logger.info(f"Training data: {len(df_train)} rows")
    return df_train


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
    - For numeric features: fill with median (robust to outliers)
    - Remove rows with NaN in target or metadata columns

    Args:
        df: DataFrame with features
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

    # Count NaN per column before cleaning
    for col in feature_cols:
        nan_count = _to_scalar_int(df[col].isna().sum())
        if nan_count > 0:
            stats["nan_cols"][col] = nan_count

    # Fill NaN in feature columns with median
    for col in feature_cols:
        nan_count = _to_scalar_int(df[col].isna().sum())
        if nan_count > 0:
            median_val = df[col].median()
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

    Pipeline order:
    1. Non-linear correlation analysis (Spearman) to identify correlated feature groups
    2. PCA reduction on correlated clusters
    3. Log transformation (fit on train PCA, then transform all datasets)
    4. Normalization (apply pre-fitted scalers: z-score for linear, minmax for LSTM)

    Args:
        dry_run: If True, don't overwrite files

    Returns:
        Summary dictionary with results
    """
    logger.info("=" * 60)
    logger.info("CLEAR FEATURES PIPELINE")
    logger.info("=" * 60)
    logger.info("Pipeline: Correlation -> PCA -> Log Transform -> Normalize")
    logger.info(f"Correlation threshold: {CORRELATION_CONFIG['threshold']}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    # Create output directories
    CLEAR_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PCA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_TRANSFORM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "steps": {},
        "datasets": {},
    }

    # =========================================================================
    # STEP 1: Load reference dataset and fit transformers
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Load reference dataset and training data")
    logger.info("=" * 60)

    ref_path = DATASETS[REFERENCE_DATASET]
    df_ref = load_dataset(ref_path)
    df_train = get_train_data(df_ref)

    # =========================================================================
    # STEP 2: Non-linear correlation analysis
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Non-linear correlation analysis (Spearman)")
    logger.info("=" * 60)

    corr_analyzer = NonLinearCorrelationAnalyzer(
        method=cast(str, CORRELATION_CONFIG["method"]),
        threshold=cast(float, CORRELATION_CONFIG["threshold"]),
        min_cluster_size=cast(int, CORRELATION_CONFIG["min_cluster_size"]),
    )

    sample_size_val = CORRELATION_CONFIG["sample_size"]
    corr_result = corr_analyzer.analyze(
        df_train,
        sample_size=cast(int | None, sample_size_val),
    )
    corr_analyzer.save_results()

    results["steps"]["correlation"] = {
        "method": corr_result.method,
        "threshold": corr_result.threshold_used,
        "n_clusters": len(corr_result.clusters),
        "n_clustered_features": sum(len(c.features) for c in corr_result.clusters),
        "n_unclustered_features": len(corr_result.unclustered_features),
    }

    logger.info(f"Found {len(corr_result.clusters)} clusters to merge")
    for cluster in corr_result.clusters[:5]:  # Show first 5
        logger.info(
            f"  Cluster {cluster.cluster_id}: {len(cluster.features)} features, "
            f"avg_corr={cluster.avg_correlation:.3f}"
        )
    if len(corr_result.clusters) > 5:
        logger.info(f"  ... and {len(corr_result.clusters) - 5} more clusters")

    # =========================================================================
    # STEP 3: PCA reduction on correlated clusters
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: PCA reduction on correlated clusters")
    logger.info("=" * 60)

    # Get clusters for PCA
    clusters_for_pca = corr_analyzer.get_clusters_for_pca()

    if clusters_for_pca:
        pca_reducer = WeightedPCAReducer(clusters=clusters_for_pca)
        pca_summary = pca_reducer.fit(df_train)
        pca_reducer.save_artifacts()

        results["steps"]["pca"] = {
            "original_features": pca_summary.original_n_features,
            "final_features": pca_summary.final_n_features,
            "features_removed": len(pca_summary.features_removed),
            "components_added": len(pca_summary.features_added),
        }

        logger.info(
            f"PCA: {pca_summary.original_n_features} -> {pca_summary.final_n_features} features"
        )
    else:
        pca_reducer = None
        results["steps"]["pca"] = {"skipped": True, "reason": "No clusters found"}
        logger.info("No clusters to reduce - skipping PCA")

    # =========================================================================
    # STEP 4: Fit log transformer (on PCA-transformed data)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Fit log transformer")
    logger.info("=" * 60)

    if pca_reducer:
        df_train_pca = pca_reducer.transform(df_train)
    else:
        df_train_pca = df_train

    log_transformer = LogTransformer()
    log_result = log_transformer.fit(df_train_pca)
    log_transformer.save_artifacts()

    results["steps"]["log_transform"] = {
        "features_transformed": len(log_result.features_transformed),
        "features_skipped": len(log_result.features_skipped),
    }

    logger.info(f"Log transform: {len(log_result.features_transformed)} features")

    # =========================================================================
    # STEP 5: Load pre-fitted scalers
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Load pre-fitted scalers")
    logger.info("=" * 60)

    scaler_applier = ScalerApplier()
    scaler_applier.load_scalers()

    # =========================================================================
    # STEP 6: Transform all datasets
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Transform all datasets")
    logger.info("=" * 60)

    for dataset_name, dataset_path in DATASETS.items():
        logger.info(f"\n--- Processing {dataset_name} ---")

        # Load dataset
        df = load_dataset(dataset_path)
        original_cols = len(df.columns)

        # 6a. Apply PCA (if clusters were found)
        if pca_reducer:
            logger.info("  Applying PCA...")
            df = pca_reducer.transform(df)
        after_pca_cols = len(df.columns)

        # 6b. Apply log transform
        logger.info("  Applying log transform...")
        df = log_transformer.transform(df)

        # 6c. Apply scaling based on dataset type
        if dataset_name == "linear":
            logger.info("  Applying z-score normalization...")
            df = scaler_applier.apply_zscore(df)
        elif dataset_name == "lstm":
            logger.info("  Applying min-max normalization...")
            df = scaler_applier.apply_minmax(df)
        else:
            logger.info("  No normalization for tree_based (not needed)")

        # 6d. Clean NaN values (fill with median)
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
