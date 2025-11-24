"""Module for saving GARCH dataset from ARIMA evaluation results.

Keeps `src.*` imports intact (do not remove).
The function `save_garch_dataset` merges ARIMA predictions into the split dataset.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import pandas as pd

from src.arima.evaluation_arima.utils import detect_value_column  # type: ignore
from src.path import (  # type: ignore
    GARCH_DATASET_FILE,
    ROLLING_PREDICTIONS_ARIMA_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)
from src.utils import get_logger  # type: ignore
from src.utils.datetime_utils import format_dates_to_string  # type: ignore
from src.utils.io import ensure_output_dir, load_csv_file  # type: ignore

logger = get_logger(__name__)

_SPLIT_FILE_OVERRIDE: Path | None = None
_DATASET_FILE_OVERRIDE: Path | None = None


def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 'date' column as string with DATE_FORMAT_DEFAULT."""
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in split file.")
    df = df.copy()
    df["date"] = format_dates_to_string(df.loc[:, "date"])
    return df


def _results_to_df(results: dict[str, Any]) -> pd.DataFrame:
    """Convert evaluation results dict to a tidy dataframe."""
    dates = format_dates_to_string(results["dates"])
    df = pd.DataFrame(
        {
            "date": dates,
            "y_true": np.asarray(results["y_true"], dtype=float),
            "arima_pred": np.asarray(results["y_pred"], dtype=float),
            "arima_resid": np.asarray(results["residuals"], dtype=float),
        }
    )
    return df


def _load_from_rolling_predictions() -> pd.DataFrame | None:
    """Load ARIMA predictions from rolling_predictions.csv if available.

    Maps columns:
    - y_pred -> arima_pred
    - residual -> arima_resid

    Returns:
        DataFrame with date, arima_pred, arima_resid, or None if file doesn't exist.
    """
    if not ROLLING_PREDICTIONS_ARIMA_FILE.exists():
        return None

    try:
        df = pd.read_csv(ROLLING_PREDICTIONS_ARIMA_FILE, parse_dates=["date"])
        df["date"] = format_dates_to_string(df.loc[:, "date"])

        # Check for required columns
        required_cols = {"date", "y_pred", "residual"}
        if not required_cols.issubset(df.columns):
            logger.warning(
                f"rolling_predictions.csv missing required columns. Expected: {required_cols}, "
                f"found: {set(df.columns)}"
            )
            return None

        # Map columns to expected names (matching dataset_garch.csv format)
        result_df = pd.DataFrame(
            {
                "date": df["date"],
                "arima_pred": pd.to_numeric(df["y_pred"], errors="coerce"),
                "arima_resid": pd.to_numeric(df["residual"], errors="coerce"),
            }
        )
        logger.info(f"Loaded ARIMA predictions from {ROLLING_PREDICTIONS_ARIMA_FILE}")
        return result_df
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not load from rolling_predictions.csv: {e}")
        return None


def _extract_walk_forward_residuals(
    backtest_history: pd.DataFrame, split_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Extract train residuals from walk-forward backtest via date alignment.

    This uses out-of-sample residuals produced by a walk-forward backtest
    and aligns them with the split dataset to select the training window,
    avoiding look-ahead bias from in-sample Kalman filtering.

    Args:
        backtest_history: DataFrame from backtest_full_series() or walk_forward_backtest.
            Must include at least 'date' and 'arima_resid'.
        split_df: Split dataset with 'date' and 'split' columns.

    Returns:
        DataFrame with ['date', 'arima_resid'] for train dates, or None when unavailable.
    """
    if not {"date", "arima_resid"}.issubset(backtest_history.columns):
        logger.warning(
            "Backtest history lacks required ['date','arima_resid'] columns; "
            "falling back to in-sample fitted values."
        )
        return None

    # Normalize dates then join to select the train portion only
    hist = backtest_history.copy()
    hist["date"] = format_dates_to_string(hist.loc[:, "date"])
    split = split_df[["date", "split"]].copy()
    split["date"] = format_dates_to_string(split.loc[:, "date"])
    merged = hist.merge(split, on="date", how="inner")
    train_mask = merged["split"] == "train"
    train_resid = merged.loc[train_mask, ["date", "arima_resid"]].dropna().reset_index(drop=True)

    logger.info(
        f"Using {len(train_resid)} walk-forward train residuals from backtest (no look-ahead bias)"
    )
    return train_resid


def _compute_train_residuals(
    split_df: pd.DataFrame,
    fitted_model: Any | None,
    value_col: str = "weighted_log_return",
    backtest_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute train residuals from walk-forward backtest ONLY.

    IMPORTANT: REQUIRES walk-forward residuals to avoid look-ahead bias.
    The in-sample fallback has been removed to ensure methodological rigor.

    Args:
        split_df: Split dataset DataFrame.
        fitted_model: Fitted ARIMA model (NOT USED - kept for backward compatibility).
        value_col: Column name for target values.
        backtest_history: DataFrame from backtest_full_series() - REQUIRED.

    Returns:
        DataFrame with date and arima_resid for train data.

    Raises:
        ValueError: If backtest_history is None or empty (walk-forward required).
    """
    # MANDATORY: Use walk-forward residuals (no look-ahead bias)
    if backtest_history is None or backtest_history.empty:
        msg = (
            "Walk-forward backtest residuals are REQUIRED for GARCH. "
            "In-sample residuals have look-ahead bias from Kalman filtering. "
            "Ensure backtest_full_series() is called with include_test=False "
            "before save_garch_dataset()."
        )
        logger.error(msg)
        raise ValueError(msg)

    walk_forward_result = _extract_walk_forward_residuals(backtest_history, split_df)
    if walk_forward_result is None or walk_forward_result.empty:
        msg = (
            "Walk-forward residuals extraction failed. "
            "Check that backtest_history contains 'date' and 'arima_resid' columns, "
            "and that dates align with split dataset."
        )
        logger.error(msg)
        raise ValueError(msg)

    return walk_forward_result


def _load_and_validate_split_dataset() -> pd.DataFrame:
    """Load and validate split dataset.

    Returns:
        Validated split dataset DataFrame.

    Raises:
        FileNotFoundError: If split file is missing.
        ValueError: If split column is missing.
    """
    split_source: Path | str | None = _SPLIT_FILE_OVERRIDE
    if split_source is None:
        eval_module = sys.modules.get("src.arima.evaluation_arima.evaluation_arima")
        if eval_module is not None and hasattr(eval_module, "WEIGHTED_LOG_RETURNS_SPLIT_FILE"):
            split_source = eval_module.WEIGHTED_LOG_RETURNS_SPLIT_FILE
    if split_source is None:
        split_source = WEIGHTED_LOG_RETURNS_SPLIT_FILE

    split_path = Path(split_source)
    if not split_path.exists():
        msg = f"Split file not found: {split_path}"
        raise FileNotFoundError(msg)

    split_df = load_csv_file(split_path)
    split_df = _ensure_date_col(split_df)

    if "split" not in split_df.columns:
        msg = "Split file must contain a 'split' column"
        raise ValueError(msg)

    return split_df


def _get_test_residuals(
    results: dict[str, Any] | None,
    backtest_residuals: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Get test residuals (prefer evaluation results to avoid duplication).

    Priority:
    1) results dict (walk-forward test residuals)
    2) backtest_residuals (if explicitly provided as test residuals)
    3) rolling_predictions.csv
    """
    # 1) Prefer residuals derived from evaluation (walk-forward on test)
    if results is not None:
        test_resid_df = _results_to_df(results)
        if "arima_resid" in test_resid_df.columns:
            return pd.DataFrame(test_resid_df[["date", "arima_resid"]].copy())
        return pd.DataFrame(test_resid_df)

    # 2) Accept an explicit DataFrame if provided
    if backtest_residuals is not None and not backtest_residuals.empty:
        expected_cols = {"date", "arima_resid"}
        if not expected_cols.issubset(backtest_residuals.columns):
            missing = expected_cols - set(backtest_residuals.columns)
            raise KeyError(f"Backtest residuals missing required columns: {sorted(missing)}")
        backtest_df = backtest_residuals.copy()
        backtest_df["date"] = format_dates_to_string(backtest_df.loc[:, "date"])
        return pd.DataFrame(backtest_df[["date", "arima_resid"]].copy())

    # 3) Attempt loading from persisted predictions
    loaded_resid_df: pd.DataFrame | None = _load_from_rolling_predictions()
    if loaded_resid_df is None:
        logger.warning(
            "No results dict provided and rolling_predictions.csv not available. "
            "GARCH dataset will be created without test residuals."
        )
    return loaded_resid_df


def _combine_residuals(
    train_resid_df: pd.DataFrame,
    test_resid_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine train and test residuals into a single DataFrame.

    Args:
        train_resid_df: DataFrame with train residuals.
        test_resid_df: Optional DataFrame with test residuals.

    Returns:
        Combined DataFrame with all residuals.

    Raises:
        ValueError: If no residuals are available.
    """
    all_resid_df = None
    if train_resid_df is not None and not train_resid_df.empty:
        all_resid_df = train_resid_df.copy()
        if test_resid_df is not None and not test_resid_df.empty:
            all_resid_df = pd.concat([all_resid_df, test_resid_df], ignore_index=True)
    elif test_resid_df is not None and not test_resid_df.empty:
        all_resid_df = test_resid_df.copy()

    if all_resid_df is None or all_resid_df.empty:
        msg = "No residuals available (neither train nor test). Cannot create GARCH dataset."
        raise ValueError(msg)

    combined = cast(pd.DataFrame, all_resid_df)
    combined = (
        combined.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )
    return combined


def _merge_residuals_with_split(
    split_df: pd.DataFrame,
    all_resid_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Merge residuals with split dataset.

    Args:
        split_df: Split dataset DataFrame.
        all_resid_df: DataFrame with all residuals.
        value_col: Column name for target values.

    Returns:
        Merged DataFrame with split and residuals.

    Raises:
        ValueError: If value column is missing or no residuals found after merge.
        TypeError: If all_resid_df is not a DataFrame.
    """
    required_cols = ["date", "split"]
    if value_col not in split_df.columns:
        msg = f"Value column '{value_col}' not found in split dataset"
        raise ValueError(msg)
    required_cols.append(value_col)

    split_subset = split_df[required_cols].copy()
    if not isinstance(all_resid_df, pd.DataFrame):
        msg = "all_resid_df must be a DataFrame"
        raise TypeError(msg)
    resid_subset = all_resid_df[["date", "arima_resid"]].copy()
    merged = split_subset.merge(resid_subset, on="date", how="left")

    n_with_resid = int(merged["arima_resid"].notna().sum())
    if n_with_resid == 0:
        msg = "No valid residuals found after merge. Check date alignment."
        raise ValueError(msg)

    return merged


def _log_residuals_summary(merged: pd.DataFrame) -> None:
    """Log summary of merged residuals.

    Args:
        merged: Merged DataFrame with residuals.
    """
    n_with_resid = int(merged["arima_resid"].notna().sum())
    train_mask = merged["split"] == "train"
    test_mask = merged["split"] == "test"
    n_train_resid = int(merged.loc[train_mask, "arima_resid"].notna().sum())
    n_test_resid = int(merged.loc[test_mask, "arima_resid"].notna().sum())

    logger.info(
        f"Merged residuals: {n_with_resid} observations with residuals "
        f"(train: {n_train_resid}, test: {n_test_resid})"
    )


def _save_garch_dataset_file(merged: pd.DataFrame) -> Path:
    """Save merged dataset to GARCH_DATASET_FILE.

    Args:
        merged: Merged DataFrame to save.

    Returns:
        Path to saved file.
    """
    # Filter out rows with missing residuals before saving
    merged_filtered = merged.dropna(subset=["arima_resid"]).reset_index(drop=True)

    # Respect module-level monkeypatching by default and a private override when set.
    # Why: tests patch save_data_for_garch.GARCH_DATASET_FILE directly. Re-reading from
    # constants would ignore that patch. We only use _DATASET_FILE_OVERRIDE when explicitly set.
    out_path = Path(GARCH_DATASET_FILE)
    if _DATASET_FILE_OVERRIDE is not None:
        out_path = _DATASET_FILE_OVERRIDE
    ensure_output_dir(out_path)
    merged_filtered.to_csv(out_path, index=False)

    n_filtered = len(merged_filtered)
    n_original = len(merged)
    logger.info(
        f"Saved GARCH dataset → {out_path} ({n_original} rows, {n_filtered} with valid residuals)"
    )
    if n_filtered < n_original:
        logger.warning(f"Filtered out {n_original - n_filtered} rows with missing residuals")
    return out_path


def save_garch_dataset(
    results: dict[str, Any] | None = None,
    fitted_model: Any | None = None,
    train_series: pd.Series | None = None,
    value_col: str | None = None,
    backtest_residuals: pd.DataFrame | None = None,
) -> Path:
    """Merge ARIMA predictions/residuals into split dataset and save to GARCH_DATASET_FILE.

    Computes train residuals from fitted model and test residuals from evaluation results
    or walk-forward backtest (when provided).
    Saves only essential columns: date, split, arima_resid.

    Args:
        results: Optional evaluation results dict with test predictions/residuals.
            If None, tries to load from rolling_predictions.csv.
        fitted_model: Fitted ARIMA model for computing train residuals.
        train_series: Optional train series for residual computation
            (unused, kept for compatibility).
        value_col: Column name for target values in split dataset.
            If None, auto-detects.
        backtest_residuals: Optional DataFrame produced by walk-forward backtest with columns
            ['date', 'arima_resid']. When provided, supersedes evaluation residuals.

    Returns:
        Path to saved GARCH dataset file.

    Raises:
        FileNotFoundError: If split file is missing.
        ValueError: If required data is missing.
    """
    split_df = _load_and_validate_split_dataset()

    if value_col is None:
        value_col = detect_value_column(split_df)

    # Compute residuals robustly: prefer train-only residuals from backtest history,
    # and test residuals from evaluation results (walk-forward).
    train_resid_df = _compute_train_residuals(
        split_df, fitted_model, value_col, backtest_history=backtest_residuals
    )
    test_resid_df = _get_test_residuals(results, None)
    all_resid_df = _combine_residuals(train_resid_df, test_resid_df)

    merged = _merge_residuals_with_split(split_df, all_resid_df, value_col)

    _log_residuals_summary(merged)
    return _save_garch_dataset_file(merged)


def regenerate_garch_dataset_from_rolling_predictions() -> Path:
    """Regenerate GARCH dataset from rolling_predictions.csv file.

    DEPRECATED: This function is no longer recommended as it requires walk-forward
    backtest residuals for train data. Use the main evaluation pipeline instead
    which generates walk-forward residuals via backtest_full_series().

    This function loads ARIMA predictions/residuals from rolling_predictions.csv
    (test set only) and merges them with the split dataset. Train residuals
    MUST be provided separately via backtest_full_series() to avoid look-ahead bias.

    Returns:
        Path to saved GARCH dataset file.

    Raises:
        FileNotFoundError: If rolling_predictions.csv or split file doesn't exist.
        ValueError: If walk-forward backtest residuals are not available.
    """
    logger.warning(
        "regenerate_garch_dataset_from_rolling_predictions() is DEPRECATED. "
        "Walk-forward backtest residuals are required for train data. "
        "This will fail unless backtest_full_series() has been run separately."
    )

    # NOTE: fitted_model=None will trigger the ValueError in _compute_train_residuals
    # This is intentional - walk-forward residuals are REQUIRED
    return save_garch_dataset(results=None, fitted_model=None, backtest_residuals=None)
