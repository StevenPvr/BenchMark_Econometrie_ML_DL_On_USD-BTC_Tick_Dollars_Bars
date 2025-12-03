"""I/O functions for feature pipeline.

This module handles loading and saving data for the feature engineering pipeline.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from src.utils import get_logger
from src.features.pipeline import TRAIN_RATIO, drop_timestamp_columns
from src.features.scalers import (
    MinMaxScalerCustom,
    StandardScalerCustom,
)
from src.path import (
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LINEAR_TEST_PARQUET,
    DATASET_FEATURES_LINEAR_TRAIN_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_LSTM_TEST_PARQUET,
    DATASET_FEATURES_LSTM_TRAIN_PARQUET,
    DATASET_FEATURES_PARQUET,
    DATASET_FEATURES_TEST_PARQUET,
    DATASET_FEATURES_TRAIN_PARQUET,
    DOLLAR_BARS_PARQUET,
    FEATURES_DIR,
    MINMAX_SCALER_FILE,
    SCALERS_DIR,
    ZSCORE_SCALER_FILE,
)
from src.features.pipeline import get_columns_to_scale

logger = get_logger(__name__)

__all__ = [
    "load_input_data",
    "save_outputs",
    "save_train_test_splits",
    "fit_and_save_scalers",
]


def load_input_data() -> pd.DataFrame:
    """Load input data from log_returns_split.parquet.

    Returns:
        DataFrame with dollar bars and log returns.

    Raises:
        FileNotFoundError: If input file not found.
    """
    logger.info("Loading input data from %s", DOLLAR_BARS_PARQUET)

    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(
            f"Input file not found: {DOLLAR_BARS_PARQUET}. "
            "Please run data_preparation first."
        )

    df = pd.read_parquet(DOLLAR_BARS_PARQUET)

    logger.info(
        "Loaded %d rows, %d columns. Columns: %s",
        len(df),
        len(df.columns),
        list(df.columns),
    )

    return df


def save_outputs(df_features: pd.DataFrame) -> None:
    """Save feature dataset to files.

    NOTE: Only saves the raw (unscaled) dataset. Linear and LSTM datasets
    with normalization will be created by clear_features module after
    PCA reduction and log transformation.

    Args:
        df_features: Raw features for ML models.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    df_features = drop_timestamp_columns(df_features)

    logger.info("Saving dataset_features to %s", DATASET_FEATURES_PARQUET)
    df_features.to_parquet(DATASET_FEATURES_PARQUET, index=False)
    logger.info(
        "Saved dataset_features: %d rows, %d columns",
        len(df_features),
        len(df_features.columns),
    )

    logger.info(
        "Saving dataset_features_linear to %s (unscaled, will be normalized in clear_features)",
        DATASET_FEATURES_LINEAR_PARQUET,
    )
    df_features.to_parquet(DATASET_FEATURES_LINEAR_PARQUET, index=False)

    logger.info(
        "Saving dataset_features_lstm to %s (unscaled, will be normalized in clear_features)",
        DATASET_FEATURES_LSTM_PARQUET,
    )
    df_features.to_parquet(DATASET_FEATURES_LSTM_PARQUET, index=False)


def _split_dataframe_for_output(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into chronological train/test slices.

    Uses an existing 'split' column if present; otherwise applies an 80/20
    chronological split and tags the rows with a new 'split' column.

    Args:
        df: DataFrame to split.
        train_ratio: Ratio for training set.

    Returns:
        Tuple of (df_train, df_test).
    """
    if "split" in df.columns:
        mask_train = df["split"] == "train"
        mask_test = df["split"] == "test"
        df_train = df.loc[mask_train].copy()
        df_test = df.loc[mask_test].copy()

        if df_train.empty or df_test.empty:
            logger.warning(
                "Existing split column is empty; falling back to ratio split"
            )
        else:
            return df_train, df_test

    split_idx = int(len(df) * train_ratio)
    df_train = df.iloc[:split_idx].copy()
    df_train["split"] = "train"
    df_test = df.iloc[split_idx:].copy()
    df_test["split"] = "test"
    return df_train, df_test


def save_train_test_splits() -> None:
    """Create and save 80/20 train/test splits for all feature variants."""
    datasets = [
        (
            DATASET_FEATURES_PARQUET,
            DATASET_FEATURES_TRAIN_PARQUET,
            DATASET_FEATURES_TEST_PARQUET,
        ),
        (
            DATASET_FEATURES_LINEAR_PARQUET,
            DATASET_FEATURES_LINEAR_TRAIN_PARQUET,
            DATASET_FEATURES_LINEAR_TEST_PARQUET,
        ),
        (
            DATASET_FEATURES_LSTM_PARQUET,
            DATASET_FEATURES_LSTM_TRAIN_PARQUET,
            DATASET_FEATURES_LSTM_TEST_PARQUET,
        ),
    ]

    for input_path, train_parquet, test_parquet in datasets:
        if not input_path.exists():
            logger.warning("Cannot create split: missing %s", input_path)
            continue

        df = pd.read_parquet(input_path)
        df_train, df_test = _split_dataframe_for_output(df)

        df_train.to_parquet(train_parquet, index=False)
        df_test.to_parquet(test_parquet, index=False)

        logger.info(
            "Saved train/test splits for %s -> train: %s, test: %s",
            input_path.name,
            train_parquet.name,
            test_parquet.name,
        )


def fit_and_save_scalers(df_train: pd.DataFrame) -> None:
    """Fit scalers on training data and save them for later use.

    NOTE: Scalers are fit here but NOT applied. The actual normalization
    happens in clear_features after PCA reduction and log transformation.

    CRITICAL: Scalers are fit on TRAIN only to prevent data leakage.

    Args:
        df_train: Training DataFrame.
    """
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    # Z-SCORE FOR LINEAR MODELS
    logger.info("Fitting z-score scaler for linear models (NOT applying)...")

    cols_zscore = get_columns_to_scale(
        df_train,
        exclude_target=True,
        include_log_return_lags=False,
    )

    zscore_scaler = StandardScalerCustom()
    zscore_scaler.fit(df_train, cols_zscore)
    zscore_scaler.save(ZSCORE_SCALER_FILE)

    logger.info(
        "Z-score scaler fit on %d columns, saved to %s",
        len(cols_zscore),
        ZSCORE_SCALER_FILE,
    )

    # MIN-MAX FOR LSTM
    logger.info("Fitting min-max scaler for LSTM (NOT applying)...")

    cols_minmax = get_columns_to_scale(
        df_train,
        exclude_target=True,
        include_log_return_lags=True,
    )

    minmax_scaler = MinMaxScalerCustom()
    minmax_scaler.fit(df_train, cols_minmax)
    minmax_scaler.save(MINMAX_SCALER_FILE)

    logger.info(
        "Min-max scaler fit on %d columns, saved to %s",
        len(cols_minmax),
        MINMAX_SCALER_FILE,
    )

    logger.info(
        "Scalers saved. Normalization will be applied in clear_features module."
    )
