"""Functions for ticker-level data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    LIGHTGBM_REALIZED_VOL_WINDOW,
    REQUIRED_COLS_TICKER_DATA,
    REQUIRED_COLS_WEIGHTED_RETURNS,
    TRAIN_RATIO_DEFAULT,
)
from src.data_preparation.computations import (
    compute_log_returns_for_tickers,
    compute_log_volume,
    compute_volatility_for_tickers,
)
from src.path import (
    DATA_TICKERS_FULL_FILE,
    DATASET_FILTERED_PARQUET_FILE,
    WEIGHTED_LOG_RETURNS_FILE,
)
from src.utils import (
    get_logger,
    load_and_validate_dataframe,
    load_dataframe,
    log_split_dates,
    save_parquet_and_csv,
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
    validate_temporal_split,
    validate_train_ratio,
)

logger = get_logger(__name__)


def _load_tickers_data(input_file: str) -> pd.DataFrame:
    """Load ticker data from provided path.

    Uses common loading function for consistency and reduced duplication.

    Args:
        input_file: Path to ticker data (Parquet or CSV).

    Returns:
        DataFrame with ticker data sorted by date and ticker.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If DataFrame is empty after cleaning.
        KeyError: If required columns are missing.
    """
    try:
        return load_and_validate_dataframe(
            input_file=input_file,
            date_columns=["date"],
            required_columns=list(REQUIRED_COLS_TICKER_DATA),
            sort_by=["ticker", "date"],
            validate_not_empty=True,
            column_renames={"tickers": "ticker"},  # Handle legacy column name
        )
    except ValueError as err:
        # Standardize empty-data error message for tests and callers
        raise ValueError("Ticker data DataFrame is empty") from err


def _split_single_ticker(
    ticker_data: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Split a single ticker's data temporally using a global split date.

    Args:
        ticker_data: DataFrame for a single ticker.
        split_date: Global date for splitting (train: <= split_date, test: > split_date).

    Returns:
        DataFrame with split column added, or None if too small.
    """
    ticker_data = ticker_data.sort_values(by="date").reset_index(drop=True)

    n_total = len(ticker_data)
    if n_total < 2:
        ticker = ticker_data["ticker"].iloc[0] if len(ticker_data) > 0 else "unknown"
        logger.warning("Ticker %s has less than 2 observations, skipping", ticker)
        return None

    # Split based on global date cutoff
    ticker_data["split"] = "train"
    ticker_data.loc[ticker_data["date"] > split_date, "split"] = "test"

    return ticker_data


def _split_all_tickers(
    data: pd.DataFrame,
    split_date: pd.Timestamp,
) -> list[pd.DataFrame]:
    """Split each ticker's data temporally using a global split date.

    Args:
        data: DataFrame with ticker time series data.
        split_date: Global date for splitting all tickers.

    Returns:
        List of DataFrames with split column added.
    """
    split_dfs: list[pd.DataFrame] = []

    for ticker in data["ticker"].unique():
        ticker_data = cast(pd.DataFrame, data[data["ticker"] == ticker].copy())
        split_ticker = _split_single_ticker(ticker_data, split_date)
        if split_ticker is not None:
            split_dfs.append(split_ticker)

    return split_dfs


def _validate_split_result(result_df: pd.DataFrame, function_name: str) -> None:
    """Validate temporal split result.

    Args:
        result_df: DataFrame with split column.
        function_name: Name of calling function for logging.
    """
    validate_temporal_split(
        result_df,
        ticker_col="ticker",
        function_name=function_name,
    )
    log_split_dates(
        result_df,
        ticker_col="ticker",
        function_name=function_name,
    )


def _calculate_global_split_date(data: pd.DataFrame, train_ratio: float) -> pd.Timestamp:
    """Calculate global split date based on temporal range.

    NOTE ON GLOBAL SPLIT STRATEGY:
    This implementation uses a single global split date applied to all tickers.
    This approach has trade-offs:

    PROS:
    - Ensures strict temporal separation across all tickers
    - Simpler to implement and validate
    - Prevents any temporal overlap in ensemble predictions

    CONS:
    - Tickers with different listing dates may have unequal train/test ratios
    - Potential survivorship bias if tickers were delisted before test period
    - Newer tickers (IPOs) may have less training data

    ALTERNATIVE APPROACHES CONSIDERED:
    - Per-ticker split: Would give equal ratios but allow temporal overlap
    - Minimum observation filter: Implemented via validation in downstream code

    For production use, consider filtering tickers with insufficient observations
    in either split to ensure balanced evaluation.

    Args:
        data: DataFrame with date column.
        train_ratio: Proportion of temporal range for training.

    Returns:
        Global split date for all tickers.
    """
    min_date = data["date"].min()
    max_date = data["date"].max()
    temporal_range = max_date - min_date
    split_date = min_date + temporal_range * train_ratio

    logger.info(
        f"Global temporal split: {min_date.date()} to {max_date.date()}, "
        f"split date: {split_date.date()} (train_ratio={train_ratio:.2f})"
    )
    logger.info(
        "NOTE: Global split date may result in different train/test ratios per ticker. "
        "See function docstring for trade-offs and alternatives."
    )

    return split_date


def _remove_contaminated_test_observations(split_df: pd.DataFrame) -> pd.DataFrame:
    """Remove test observations contaminated by rolling windows spanning train/test boundary.

    ANTI-LEAKAGE MECHANISM:
    When features are computed using rolling windows (e.g., 5-day volatility),
    the first (window_size - 1) observations in the test set have features computed
    using data from BOTH train and test periods. This creates data leakage.

    This function removes these contaminated observations to ensure clean evaluation.

    Example with 5-day window:
        Split date: 2022-07-26
        Test observation 2022-07-27: window uses [2022-07-23 to 2022-07-27] ❌ CONTAMINATED
        Test observation 2022-07-28: window uses [2022-07-24 to 2022-07-28] ❌ CONTAMINATED
        Test observation 2022-07-29: window uses [2022-07-25 to 2022-07-29] ❌ CONTAMINATED
        Test observation 2022-08-01: window uses [2022-07-26 to 2022-08-01] ❌ CONTAMINATED
        Test observation 2022-08-02: window uses [2022-07-27 to 2022-08-02] ✅ CLEAN

    Args:
        split_df: DataFrame with split column and features computed using rolling windows.

    Returns:
        DataFrame with contaminated test observations removed.
    """
    n_contaminated = LIGHTGBM_REALIZED_VOL_WINDOW - 1

    if n_contaminated == 0:
        logger.info("No test observations to remove (window size = 1)")
        return split_df

    # Count initial test observations per ticker
    n_initial_test = (split_df["split"] == "test").sum()
    logger.info(
        f"Anti-leakage: removing first {n_contaminated} test observations per ticker "
        f"to prevent rolling window contamination (window_size={LIGHTGBM_REALIZED_VOL_WINDOW})"
    )

    # Process each ticker independently
    clean_dfs: list[pd.DataFrame] = []

    for ticker in split_df["ticker"].unique():
        mask = split_df["ticker"] == ticker
        ticker_df = split_df[mask].copy()
        if len(ticker_df) == 0:
            continue
        assert "date" in ticker_df.columns, f"Missing 'date' column for ticker {ticker}"
        ticker_df = ticker_df.sort_values(by="date").reset_index(drop=True)  # type: ignore[arg-type]

        # Keep all train observations
        train_df = cast(pd.DataFrame, ticker_df[ticker_df["split"] == "train"].copy())

        # Remove first N test observations
        test_df = cast(pd.DataFrame, ticker_df[ticker_df["split"] == "test"].copy())
        if len(test_df) > n_contaminated:
            test_df = test_df.iloc[n_contaminated:].copy()
            clean_dfs.extend([train_df, test_df])
        elif len(test_df) > 0:
            # Ticker has test data but not enough clean observations
            logger.warning(
                f"Ticker {ticker}: only {len(test_df)} test observations, "
                f"all contaminated. Keeping only train data."
            )
            clean_dfs.append(train_df)
        else:
            # No test data for this ticker
            clean_dfs.append(train_df)

    result_df = pd.concat(clean_dfs, ignore_index=True)
    n_final_test = (result_df["split"] == "test").sum()
    n_removed = n_initial_test - n_final_test
    pct_removed = 100 * n_removed / n_initial_test if n_initial_test > 0 else 0

    logger.info(
        f"Removed {n_removed} contaminated test observations ({pct_removed:.2f}% of test set). "
        f"Clean test set: {n_final_test} observations."
    )

    return result_df


def _perform_tickers_temporal_split(
    data: pd.DataFrame,
    train_ratio: float,
) -> pd.DataFrame:
    """Perform temporal split on ticker data using a global split date.

    Calculates a global split date based on train_ratio of the total temporal range,
    then applies this date to all tickers. This ensures no temporal overlap between
    train and test sets when viewed globally across all tickers.

    Args:
        data: DataFrame with ticker time series data.
        train_ratio: Proportion of temporal range for training.

    Returns:
        DataFrame with split column added ('train' or 'test').

    Raises:
        ValueError: If DataFrame is empty or too small for splitting.
    """
    validate_dataframe_not_empty(data, "Input ticker data")
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    split_date = _calculate_global_split_date(data, train_ratio)
    split_dfs = _split_all_tickers(data, split_date)

    if not split_dfs:
        msg = "No ticker data could be split"
        raise ValueError(msg)

    result_df = pd.concat(split_dfs, ignore_index=True)
    _validate_split_result(result_df, "_perform_tickers_temporal_split")

    return result_df


def _drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not required in outputs.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with unnecessary columns removed.
    """
    cols_to_drop: list[str] = []

    # Drop raw price columns not required downstream; keep only derived metrics.
    # Keep 'close' and 'volume' explicitly — downstream analyses rely on closing prices and volume.
    for col in ("open",):
        if col in df.columns:
            cols_to_drop.append(col)

    # Drop weighted_log_return (not needed once merged into higher-level dataset)
    if "weighted_log_return" in df.columns:
        cols_to_drop.append("weighted_log_return")

    # Keep volume and log_volatility for RF models
    if cols_to_drop:
        logger.info("Dropping columns not required in outputs: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def _log_split_summary_stats(df: pd.DataFrame, output_file: str) -> None:
    """Log summary statistics for split ticker dataset.

    Uses common log_split_summary for consistency.

    Args:
        df: DataFrame with split column.
        output_file: Path where data was saved.
    """
    from src.utils import log_split_summary

    # Split the dataframe by split column for consistent logging
    train_df = pd.DataFrame(df[df["split"] == "train"])
    test_df = pd.DataFrame(df[df["split"] == "test"])

    log_split_summary(train_df, test_df, output_file)


def _compute_all_features(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Compute all features (log returns, volatility, volume).

    Args:
        df: DataFrame with ticker data.

    Returns:
        Tuple of (DataFrame with features, number of rows removed).
    """
    n_removed = 0

    # Compute log returns if not present
    if "log_return" not in df.columns:
        logger.info("Computing log_return per ticker")
        df = compute_log_returns_for_tickers(df)
        n_removed = len(df)  # This would be more accurate but keeping simple

    logger.info("Computing log volatility per ticker")
    df = compute_volatility_for_tickers(df)

    df = compute_log_volume(df)

    return df, n_removed


def _prepare_output_dataframe(split_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final output DataFrame with required columns and comprehensive logging.

    Args:
        split_df: DataFrame with all features and split column.

    Returns:
        Cleaned DataFrame with required columns only.

    Raises:
        KeyError: If required columns are missing.
    """
    n_initial = len(split_df)
    logger.info(f"Starting output preparation with {n_initial} rows")

    split_df = _drop_unnecessary_columns(split_df)
    split_df = split_df.rename(columns={"ticker": "tickers"})
    required_output_cols = [
        "date",
        "tickers",
        "close",
        "high",
        "low",
        "volume",
        "log_return",
        "log_volatility",
        "split",
    ]
    missing = [c for c in required_output_cols if c not in split_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for LightGBM pipeline: {missing}")

    split_df = cast(pd.DataFrame, split_df[required_output_cols])

    # Replace infinities
    split_df = split_df.replace([np.inf, -np.inf], pd.NA)

    # Log NaN counts per column BEFORE dropping
    required_cols = ["close", "high", "low", "volume", "log_return", "log_volatility"]
    for col in required_cols:
        n_na = split_df[col].isna().sum()
        if n_na > 0:
            pct_na = 100 * n_na / n_initial
            logger.warning(f"Column '{col}' has {n_na} NaN values ({pct_na:.2f}% of data)")

    # Drop NaN rows
    split_df = cast(
        pd.DataFrame,
        split_df.dropna(subset=required_cols),
    )
    n_after_dropna = len(split_df)
    n_dropped_na = n_initial - n_after_dropna

    if n_dropped_na > 0:
        pct_dropped_na = 100 * n_dropped_na / n_initial
        logger.warning(
            f"Dropped {n_dropped_na} rows ({pct_dropped_na:.2f}%) due to NaN in required columns"
        )

    # Filter zero high/low (data quality issue)
    absence_mask = (split_df["high"] == 0) & (split_df["low"] == 0)
    n_zero_hl = absence_mask.sum()
    if n_zero_hl > 0:
        logger.warning(f"Removing {n_zero_hl} rows with zero high/low prices (data quality issue)")

    split_df = cast(pd.DataFrame, split_df.loc[~absence_mask].reset_index(drop=True))

    n_final = len(split_df)
    n_total_dropped = n_initial - n_final
    pct_total_dropped = 100 * n_total_dropped / n_initial
    logger.info(
        f"Output preparation complete: {n_initial} → {n_final} rows "
        f"({n_total_dropped} dropped total, {pct_total_dropped:.2f}%)"
    )

    return split_df


def _prepare_output_dataframe_no_feature_computation(split_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final output DataFrame without computing features (already computed).

    This variant is used when features have been computed BEFORE the split,
    ensuring no data leakage from rolling windows crossing split boundaries.

    Args:
        split_df: DataFrame with all features already computed and split column.

    Returns:
        Cleaned DataFrame with required columns only.

    Raises:
        KeyError: If required columns are missing.
    """
    return _prepare_output_dataframe(split_df)


def _compute_features_and_prepare_output(
    split_df: pd.DataFrame, function_name: str
) -> pd.DataFrame:
    """Compute features and prepare output DataFrame for ticker data.

    Args:
        split_df: DataFrame with split column.
        function_name: Name of calling function for validation.

    Returns:
        Prepared DataFrame with all features computed.
    """
    # Compute features after split to preserve original split counts
    if "log_return" not in split_df.columns:
        logger.info("Computing log_return per ticker")
        split_df = compute_log_returns_for_tickers(split_df)
    logger.info("Computing log volatility per ticker")
    split_df = compute_volatility_for_tickers(split_df)
    split_df = compute_log_volume(split_df)

    _validate_split_result(split_df, function_name)
    return _prepare_output_dataframe(split_df)


def split_tickers_train_test(
    train_ratio: float = TRAIN_RATIO_DEFAULT,
    input_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Split ticker time series data into train and test sets.

    ANTI-LEAKAGE DESIGN:
    To prevent data leakage from rolling features crossing the train/test boundary:
    1. Computes all features (log_return, log_volatility) BEFORE temporal split
    2. Performs temporal split using global split date
    3. Removes first (window_size - 1) observations from test set to ensure clean evaluation

    This ensures test set observations have features computed purely from test period data,
    preventing any information from the training period influencing test predictions.

    Performs temporal split using a global split date calculated from train_ratio
    (80% train / 20% test by default) applied to the entire temporal range.
    This ensures no temporal overlap between train and test sets across all tickers.
    Saves result to Parquet with 'split' column indicating train/test.

    Args:
        train_ratio: Proportion of temporal range for training (default: TRAIN_RATIO_DEFAULT).
        input_file: Path to ticker data CSV. If None, uses default.
        output_file: Path to save split data Parquet. If None, uses default.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If train_ratio is not between 0 and 1.
    """
    validate_train_ratio(train_ratio)

    if input_file is None:
        input_file = str(DATASET_FILTERED_PARQUET_FILE)
    if output_file is None:
        output_file = str(DATA_TICKERS_FULL_FILE)

    # CRITICAL: Compute features BEFORE split to ensure consistency
    ticker_data = _load_tickers_data(input_file)
    logger.info("Computing all features BEFORE temporal split to prevent leakage")
    ticker_data, _ = _compute_all_features(ticker_data)
    ticker_data = compute_log_volume(ticker_data)

    # Perform temporal split
    split_df = _perform_tickers_temporal_split(ticker_data, train_ratio)

    # ANTI-LEAKAGE: Remove contaminated test observations
    split_df = _remove_contaminated_test_observations(split_df)

    # Prepare final output
    split_df = _prepare_output_dataframe_no_feature_computation(split_df)
    _validate_split_result(split_df, "split_tickers_train_test")

    save_parquet_and_csv(split_df, output_file)
    _log_split_summary_stats(split_df, output_file)


def _load_weighted_returns_file(
    weighted_returns_file: Path | str,
) -> pd.DataFrame | None:
    """Load weighted returns file (Parquet or CSV).

    Delegates to src.utils.load_dataframe() with exception handling.

    Args:
        weighted_returns_file: Path to weighted log returns file.

    Returns:
        DataFrame with weighted returns or None if file doesn't exist.
    """
    weighted_returns_path = Path(weighted_returns_file)

    try:
        weighted_df = load_dataframe(
            weighted_returns_path,
            date_columns=["date"],
            required_columns=list(REQUIRED_COLS_WEIGHTED_RETURNS),
            validate_not_empty=False,
        )
        return weighted_df
    except FileNotFoundError:
        logger.warning(
            "Weighted returns file not found: %s. Skipping weighted_log_return column.",
            weighted_returns_path,
        )
        return None


def _merge_weighted_returns(df: pd.DataFrame, weighted_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weighted returns into DataFrame.

    Args:
        df: DataFrame with date column.
        weighted_df: DataFrame with weighted log returns.

    Returns:
        DataFrame with weighted_log_return column added.
    """
    validate_required_columns(weighted_df, REQUIRED_COLS_WEIGHTED_RETURNS)
    df["date"] = pd.to_datetime(df["date"])

    df = df.merge(
        weighted_df[["date", "weighted_log_return"]],
        on="date",
        how="left",
    )

    n_missing_weighted = df["weighted_log_return"].isna().sum()
    if n_missing_weighted > 0:
        logger.warning(
            "%d rows have missing weighted_log_return (dates not in weighted returns file)",
            n_missing_weighted,
        )
    else:
        logger.info("Successfully added weighted_log_return column")

    return df


def _add_weighted_returns_from_file(
    df: pd.DataFrame,
    weighted_returns_file: Path | str | None,
) -> pd.DataFrame:
    """Add weighted log returns from file to DataFrame.

    Loads from Parquet if available (faster), otherwise falls back to CSV.

    Args:
        df: DataFrame with date column.
        weighted_returns_file: Path to weighted log returns file (Parquet or CSV).

    Returns:
        DataFrame with weighted_log_return column added if file exists.
    """
    if weighted_returns_file is None:
        weighted_returns_file = WEIGHTED_LOG_RETURNS_FILE

    assert weighted_returns_file is not None  # Type hint for mypy
    weighted_df = _load_weighted_returns_file(weighted_returns_file)
    if weighted_df is None:
        return df

    return _merge_weighted_returns(df, weighted_df)


def add_log_returns_to_ticker_parquet(
    input_file: Path | str,
    output_file: Path | str | None = None,
    weighted_returns_file: Path | str | None = None,
) -> pd.DataFrame:
    """Add log returns and weighted log returns to ticker-level parquet file.

    Computes log returns per ticker from close prices and adds weighted log returns
    by merging with weighted log returns file.

    Args:
        input_file: Path to input parquet file with ticker data.
        output_file: Path to output parquet file. If None, overwrites input file.
        weighted_returns_file: Path to weighted log returns CSV file. If None, uses
            default WEIGHTED_LOG_RETURNS_FILE from constants.

    Returns:
        DataFrame with log_return and weighted_log_return columns added.

    Raises:
        FileNotFoundError: If input file or weighted returns file doesn't exist.
        ValueError: If required columns are missing.
    """
    input_path = Path(input_file)
    validate_file_exists(input_path, "Input file")

    logger.info("Loading ticker data from %s", input_path)
    df = pd.read_parquet(input_path)

    df, _ = _compute_all_features(df)
    df = _add_weighted_returns_from_file(df, weighted_returns_file)

    output_path = input_path if output_file is None else Path(output_file)
    save_parquet_and_csv(df, output_path)
    logger.info("Saved ticker data with log returns to %s", output_path)

    return df
