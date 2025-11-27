"""Main entry point for Ridge pipeline.

Usage:
    python -m src.model.econometrie.ridge.main [--n-trials N] [--skip-optim]

Example:
    # Run full optimization (50 trials by default)
    python -m src.model.econometrie.ridge.main

    # Quick run with 10 trials
    python -m src.model.econometrie.ridge.main --n-trials 10

    # Skip optimization, use default params
    python -m src.model.econometrie.ridge.main --skip-optim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger, setup_logging
from src.model.econometrie.ridge.pipeline import (
    RidgePipeline,
    RidgePipelineConfig,
)
from src.path import DATASET_FEATURES_LINEAR_PARQUET, RIDGE_DIR

logger = get_logger(__name__)

# Train/test split ratio
TRAIN_RATIO = 0.8

# Columns to exclude from features
EXCLUDE_COLS = [
    "bar_id",
    "datetime_close",
    "log_return",
    "label",  # Target
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split the dataset for linear models.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info("Loading data from %s", DATASET_FEATURES_LINEAR_PARQUET)

    if not DATASET_FEATURES_LINEAR_PARQUET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FEATURES_LINEAR_PARQUET}. "
            "Please run 'python -m src.features.main' first."
        )

    df = pd.read_parquet(DATASET_FEATURES_LINEAR_PARQUET)
    logger.info("Loaded dataset: %d rows, %d columns", len(df), len(df.columns))

    # Prepare target (De Prado triple-barrier labels: -1, 0, 1)
    if "label" not in df.columns:
        raise ValueError(
            "Target column 'label' not found in dataset. "
            "Please ensure labels have been generated using triple-barrier labeling."
        )

    y = df["label"].copy()

    # Prepare features (exclude non-feature columns)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()

    logger.info("Features: %d columns", len(feature_cols))

    # Train/test split (temporal, no shuffle)
    n_train = int(len(df) * TRAIN_RATIO)

    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]

    logger.info(
        "Train: %d samples, Test: %d samples (%.0f%% / %.0f%%)",
        len(X_train),
        len(X_test),
        100 * TRAIN_RATIO,
        100 * (1 - TRAIN_RATIO),
    )

    return X_train, X_test, y_train, y_test


def main() -> None:
    """Run the Ridge pipeline."""
    parser = argparse.ArgumentParser(description="Run Ridge optimization pipeline")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--skip-optim",
        action="store_true",
        help="Skip optimization and use default parameters",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )
    parser.add_argument(
        "--purge-gap",
        type=int,
        default=5,
        help="Purge gap between train/test in CV (default: 5)",
    )
    args = parser.parse_args()

    setup_logging()

    logger.info("=" * 60)
    logger.info("RIDGE OPTIMIZATION PIPELINE")
    logger.info("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Configure pipeline
    output_dir = RIDGE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RidgePipelineConfig(
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        purge_gap=args.purge_gap,
        min_train_size=100,
        output_dir=output_dir,
        verbose=True,
    )

    # Run pipeline
    pipeline = RidgePipeline(config)
    result = pipeline.run(
        X_train,
        y_train,
        X_test,
        y_test,
        skip_optimization=args.skip_optim,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Results saved to: %s", output_dir)

    # Return best params for reference
    logger.info("Best parameters: %s", result.best_params)
    logger.info("Test F1 (macro): %.6f", result.evaluation_result.metrics.get("f1_macro", float("nan")))


if __name__ == "__main__":
    main()
