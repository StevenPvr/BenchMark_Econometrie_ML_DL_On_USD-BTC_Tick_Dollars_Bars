"""CLI entry point for CatBoost pipeline.

This module loads data, splits into train/test sets, and runs the complete
CatBoost pipeline: optimization, training, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import setup_logging
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_DATASET_COMPLETE_FILE,
)
from src.model.machine_learning.catboost.pipeline import (
    CatBoostPipeline,
    CatBoostPipelineConfig,
)
from src.path import RESULTS_DIR
from src.utils import get_logger
from src.utils.io import load_dataframe

logger = get_logger(__name__)


def load_and_split_data(
    dataset_path: Path | None = None,
    target_column: str = "log_volatility",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load dataset and split into train/test sets.

    Args:
        dataset_path: Path to dataset file. If None, uses default.
        target_column: Name of the target column.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If required columns are missing.
    """
    if dataset_path is None:
        dataset_path = LIGHTGBM_DATASET_COMPLETE_FILE

    logger.info("Loading dataset from %s", dataset_path)
    df = load_dataframe(
        dataset_path,
        date_columns=["date"],
        required_columns=["split", target_column],
        validate_not_empty=True,
        sort_by=["date"],
    )

    # Split into train and test
    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    if df_train.empty:
        raise ValueError("No training data found in dataset")
    if df_test.empty:
        raise ValueError("No test data found in dataset")

    logger.info(
        "Dataset loaded: %d total observations (%d train, %d test)",
        len(df),
        len(df_train),
        len(df_test),
    )

    # Extract features (all columns except date, split, and target)
    feature_columns = [
        col
        for col in df.columns
        if col not in ["date", "split", target_column]
    ]

    if not feature_columns:
        raise ValueError("No feature columns found in dataset")

    logger.info("Using %d features: %s", len(feature_columns), ", ".join(feature_columns[:10]))

    # Prepare X and y
    X_train = df_train[feature_columns].copy()
    y_train_raw = df_train[target_column]
    X_test = df_test[feature_columns].copy()
    y_test_raw = df_test[target_column]

    # Ensure target columns are Series
    if not isinstance(y_train_raw, pd.Series):
        y_train_raw = pd.Series(y_train_raw, index=df_train.index)
    if not isinstance(y_test_raw, pd.Series):
        y_test_raw = pd.Series(y_test_raw, index=df_test.index)

    # Remove rows with NaN in target
    train_mask = ~pd.isna(y_train_raw)
    test_mask = ~pd.isna(y_test_raw)

    X_train = X_train[train_mask].copy()
    y_train = y_train_raw[train_mask].copy()
    X_test = X_test[test_mask].copy()
    y_test = y_test_raw[test_mask].copy()

    # Ensure types are correct
    assert isinstance(X_train, pd.DataFrame), "X_train must be a DataFrame"
    assert isinstance(y_train, pd.Series), "y_train must be a Series"
    assert isinstance(X_test, pd.DataFrame), "X_test must be a DataFrame"
    assert isinstance(y_test, pd.Series), "y_test must be a Series"

    logger.info(
        "Data prepared: X_train shape=%s, y_train shape=%s, X_test shape=%s, y_test shape=%s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )

    return X_train, y_train, X_test, y_test


def main() -> None:
    """Main CLI function for CatBoost pipeline."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("CatBoost Pipeline")
    logger.info("=" * 60)

    # Load and split data
    try:
        X_train, y_train, X_test, y_test = load_and_split_data()
    except (FileNotFoundError, ValueError) as ex:
        logger.error("Failed to load data: %s", ex)
        raise

    # Configure pipeline
    output_dir = RESULTS_DIR / "catboost"
    config = CatBoostPipelineConfig(
        n_trials=50,
        n_splits=5,
        purge_gap=5,
        min_train_size=200,
        validation_split=0.2,
        metric="mse",
        output_dir=output_dir,
        random_state=DEFAULT_RANDOM_STATE,
        verbose=True,
    )

    # Create and run pipeline
    pipeline = CatBoostPipeline(config)
    result = pipeline.run(X_train, y_train, X_test, y_test)

    logger.info("=" * 60)
    logger.info("CatBoost Pipeline Complete")
    logger.info("=" * 60)
    logger.info("Best parameters: %s", result.best_params)
    logger.info("Test metrics: %s", result.evaluation_result.metrics)


if __name__ == "__main__":
    main()

