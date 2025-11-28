"""Main script for clear_features - PCA-based correlated feature reduction.

This module:
1. Loads clustering results from analyse_features
2. Identifies highly correlated feature clusters
3. Applies weighted PCA to reduce dimensionality
4. Saves transformed datasets (overwrites original files)

Usage:
    python -m src.clear_features.main
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.clear_features.config import (
    CLEAR_FEATURES_DIR,
    DATASETS,
    META_COLUMNS,
    PCA_ARTIFACTS_DIR,
    REFERENCE_DATASET,
)
from src.clear_features.pca_reducer import WeightedPCAReducer

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
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows, {len(df.columns)} columns")


def get_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract training data based on split column."""
    if "split" not in df.columns:
        raise ValueError("Dataset must have 'split' column")

    train_mask = df["split"] == "train"
    df_train = df[train_mask].copy()
    logger.info(f"Training data: {len(df_train)} rows")
    return df_train


def run_pca_reduction(
    dry_run: bool = False,
    variance_threshold: float = 0.95,
    min_correlation: float = 0.6,
) -> dict:
    """
    Run PCA reduction on all feature datasets.

    Args:
        dry_run: If True, don't overwrite files, just show what would happen
        variance_threshold: Cumulative variance to retain (default 95%)
        min_correlation: Minimum cluster correlation to apply PCA (default 0.6)

    Returns:
        Summary dictionary with results
    """
    logger.info("=" * 60)
    logger.info("Starting PCA-based feature reduction")
    logger.info(f"Variance threshold: {variance_threshold}")
    logger.info(f"Min correlation: {min_correlation}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)

    # Create output directory
    CLEAR_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PCA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize reducer with custom config
    from src.clear_features.config import PCA_CONFIG

    config = PCA_CONFIG.copy()
    config["variance_explained_threshold"] = variance_threshold
    config["min_cluster_correlation"] = min_correlation

    reducer = WeightedPCAReducer(config=config)

    # Step 1: Load reference dataset and fit PCA
    logger.info(f"\n--- Step 1: Fit PCA on {REFERENCE_DATASET} dataset ---")
    ref_path = DATASETS[REFERENCE_DATASET]
    df_ref = load_dataset(ref_path)
    df_train = get_train_data(df_ref)

    # Fit PCA on training data only
    summary = reducer.fit(df_train)

    logger.info(f"\nPCA Summary:")
    logger.info(f"  Original features: {summary.original_n_features}")
    logger.info(f"  Final features: {summary.final_n_features}")
    logger.info(f"  Features removed: {len(summary.features_removed)}")
    logger.info(f"  PCA components added: {len(summary.features_added)}")

    # Save PCA artifacts
    reducer.save_artifacts()

    # Step 2: Transform all datasets
    logger.info(f"\n--- Step 2: Transform all datasets ---")

    results = {
        "timestamp": datetime.now().isoformat(),
        "variance_threshold": variance_threshold,
        "min_correlation": min_correlation,
        "dry_run": dry_run,
        "datasets": {},
    }

    for dataset_name, dataset_path in DATASETS.items():
        logger.info(f"\nProcessing {dataset_name}...")

        # Load dataset
        df = load_dataset(dataset_path)
        original_shape = df.shape

        # Transform
        df_transformed = reducer.transform(df)
        new_shape = df_transformed.shape

        logger.info(
            f"  {dataset_name}: {original_shape[1]} -> {new_shape[1]} columns"
        )

        results["datasets"][dataset_name] = {
            "original_columns": original_shape[1],
            "new_columns": new_shape[1],
            "rows": new_shape[0],
        }

        # Save (overwrite)
        if not dry_run:
            save_dataset(df_transformed, dataset_path)
        else:
            logger.info(f"  [DRY RUN] Would save to {dataset_path}")

    # Save run summary
    import json

    summary_file = CLEAR_FEATURES_DIR / "last_run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("PCA reduction complete!")
    logger.info(f"Artifacts saved to: {PCA_ARTIFACTS_DIR}")
    logger.info("=" * 60)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply PCA reduction to correlated feature clusters"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't overwrite files, just show what would happen",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Cumulative variance to retain (default: 0.95)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.6,
        help="Minimum cluster correlation to apply PCA (default: 0.6)",
    )

    args = parser.parse_args()

    run_pca_reduction(
        dry_run=args.dry_run,
        variance_threshold=args.variance_threshold,
        min_correlation=args.min_correlation,
    )


if __name__ == "__main__":
    main()
