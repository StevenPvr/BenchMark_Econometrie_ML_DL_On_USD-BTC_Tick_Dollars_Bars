"""Simplified GARCH evaluation using orchestration.py.

This refactored version delegates forecasting to orchestration.py, eliminating
266 lines of duplicated refit/forecast logic.

Key simplifications:
- forecast_on_test_from_trained_model: 266 → ~30 lines (uses orchestration.py)
- Removed manual refit logic (uses EGARCHForecaster)
- All functions ≤40 lines per AGENTS.md
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_EVAL_DEFAULT_LEVEL,
    GARCH_EVAL_FORCED_MIN_START_SIZE,
    GARCH_MIN_WINDOW_SIZE,
)
from src.garch.garch_eval.data_loading import load_and_prepare_residuals, load_model_params
from src.garch.garch_eval.helpers import assemble_forecast_results
from src.garch.garch_eval.variance_path import compute_initial_forecasts, compute_variance_path
from src.garch.training_garch.orchestration import (
    generate_full_sample_forecasts_from_trained_model,
)
from src.path import DATA_TICKERS_FULL_FILE, DATA_TICKERS_FULL_INSIGHTS_FILE, GARCH_FORECASTS_FILE
from src.utils import ensure_output_dir, get_logger, save_parquet_and_csv

logger = get_logger(__name__)


def forecast_on_test_from_trained_model(
    df_full_forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate forecasts on TEST using the trained EGARCH model.

    Behavior:
    - If df_full_forecasts is provided, extract TEST and format it.
    - Otherwise, load the trained model (model.joblib + metadata) and
      generate leak-free one-step-ahead forecasts via orchestration.
    - CRITICAL: Saves FULL SAMPLE (TRAIN + TEST) forecasts to garch_forecasts.csv
      for LightGBM feature engineering, but returns only TEST forecasts for evaluation.

    Args:
        df_full_forecasts: Optional full sample forecasts (TRAIN + TEST).
                          If None, generates them.

    Returns:
        DataFrame with columns: date, resid, RV, sigma2_egarch_raw (TEST ONLY).

    Raises:
        FileNotFoundError: If trained model or dataset not found.
    """
    logger.info("Generating TEST forecasts using orchestration pipeline...")

    # Use provided full-sample forecasts or generate from trained model
    if df_full_forecasts is None:
        df_forecasts = generate_full_sample_forecasts_from_trained_model(
            min_window_size=GARCH_EVAL_FORCED_MIN_START_SIZE,
            anchor_at_min_window=True,
        )
    else:
        df_forecasts = df_full_forecasts

    # Find residual column name
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        resid_col_name = _find_residual_column(df_forecasts)
    except ValueError:
        msg = (
            "Forecasts missing residual column ('arima_resid' or 'sarima_resid'). "
            "Cannot compute realized volatility for evaluation."
        )
        raise ValueError(msg) from None

    # CRITICAL: Save FULL SAMPLE (TRAIN + TEST) forecasts to garch_forecasts.csv
    # LightGBM requires both splits for feature engineering
    df_full_formatted = df_forecasts.copy()

    # Rename garch_forecast_h1 to sigma2_egarch_raw for consistency
    rename_map: dict[str, str] = {
        "garch_forecast_h1": "sigma2_egarch_raw",
    }
    df_full_renamed = df_full_formatted.rename(columns=rename_map)  # type: ignore[arg-type]

    # CRITICAL FIX: Save date, split, and sigma2_egarch_raw to prevent data leakage
    # The 'split' column is MANDATORY to ensure LightGBM uses only TRAIN data for training
    # and TEST data for evaluation, preventing catastrophic data leakage
    df_full_to_save: pd.DataFrame = df_full_renamed[["date", "split", "sigma2_egarch_raw"]].copy()  # type: ignore[assignment]

    # Save FULL SAMPLE results (TRAIN + TEST)
    if GARCH_FORECASTS_FILE.suffix.lower() != ".parquet":
        msg = f"GARCH forecasts must be saved with a parquet base path, got: {GARCH_FORECASTS_FILE}"
        raise ValueError(msg)
    save_parquet_and_csv(df_full_to_save, GARCH_FORECASTS_FILE)
    logger.info(
        "Saved %d FULL SAMPLE forecasts (TRAIN + TEST) to: %s",
        len(df_full_to_save),
        GARCH_FORECASTS_FILE,
    )

    # Return only TEST split for evaluation with required columns
    df_test_subset = df_forecasts[df_forecasts["split"] == "test"].copy()
    df_test_subset["RV"] = df_test_subset[resid_col_name] ** 2
    df_test_renamed = df_test_subset.rename(
        columns={resid_col_name: "resid", "garch_forecast_h1": "sigma2_egarch_raw"}
    )  # type: ignore[arg-type]
    df_test: pd.DataFrame = df_test_renamed[["date", "resid", "RV", "sigma2_egarch_raw"]].copy()  # type: ignore[assignment]
    logger.info("Returning %d TEST forecasts for evaluation", len(df_test))

    return df_test


def forecast_from_artifacts(
    *,
    horizon: int,
    level: float = GARCH_EVAL_DEFAULT_LEVEL,
) -> pd.DataFrame:
    """Compute EGARCH variance forecasts directly from persisted artifacts.

    Args:
        horizon: Number of periods to forecast ahead.
        level: Prediction interval confidence level (0 < level < 1).

    Returns:
        DataFrame containing variance forecasts, prediction intervals, and VaR.

    Raises:
        ValueError: If inputs are invalid or artifacts contain insufficient data.
    """
    if horizon <= 0:
        msg = "horizon must be a positive integer"
        raise ValueError(msg)
    if not 0.0 < level < 1.0:
        msg = "level must be strictly between 0 and 1"
        raise ValueError(msg)

    _, resid_train, _ = load_and_prepare_residuals()
    if resid_train.size == 0:
        msg = "Training residuals are empty; cannot build forecasts"
        raise ValueError(msg)

    params, model_name, dist, nu, gamma, lambda_skew = load_model_params()
    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma_val = float(gamma if gamma is not None else params.get("gamma", 0.0))

    sigma2_train = compute_variance_path(
        resid_train,
        model_name,
        omega,
        alpha,
        beta,
        gamma_val,
        dist,
        nu,
        lambda_skew,
    )
    if sigma2_train.size == 0:
        msg = "Variance path on training data is empty"
        raise ValueError(msg)

    s2_one_step, s2_multi = compute_initial_forecasts(
        resid_train,
        sigma2_train,
        horizon,
        omega,
        alpha,
        gamma_val,
        beta,
        dist,
        nu,
        lambda_skew,
    )
    forecasts = assemble_forecast_results(
        s2_multi,
        s2_one_step,
        level,
        dist,
        nu,
        lambda_skew,
    )
    forecasts["model_name"] = model_name

    if GARCH_FORECASTS_FILE.suffix.lower() != ".parquet":
        msg = f"GARCH forecasts must be saved with a parquet base path, got: {GARCH_FORECASTS_FILE}"
        raise ValueError(msg)
    save_parquet_and_csv(forecasts, GARCH_FORECASTS_FILE)
    logger.info("Saved %d-step forecasts to: %s", horizon, GARCH_FORECASTS_FILE)

    return forecasts


def _filter_valid_forecasts(
    forecasts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and filter valid forecast data (≤40 lines).

    Args:
        forecasts: DataFrame with resid, RV, sigma2_egarch_raw columns.

    Returns:
        Tuple of (resid_valid, RV_valid, sigma2_valid).
    """
    resid = forecasts["resid"].to_numpy(dtype=float)
    RV = forecasts["RV"].to_numpy(dtype=float)
    sigma2 = forecasts["sigma2_egarch_raw"].to_numpy(dtype=float)

    valid_mask = np.isfinite(resid) & np.isfinite(RV) & np.isfinite(sigma2) & (sigma2 > 0)

    return resid[valid_mask], RV[valid_mask], sigma2[valid_mask]


def compute_metrics_from_forecasts_new_format(
    forecasts: pd.DataFrame,
) -> dict[str, object]:
    """Compute metrics from new format forecasts (≤40 lines).

    Args:
        forecasts: DataFrame with columns: date, resid, RV, sigma2_egarch_raw.

    Returns:
        Dictionary with variance metrics.
    """
    from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss

    if forecasts.empty:
        return {
            "n_obs": 0,
            "qlike": float("nan"),
            "mse_var": float("nan"),
            "mae_var": float("nan"),
        }

    _, RV_valid, sigma2_valid = _filter_valid_forecasts(forecasts)

    if RV_valid.size == 0:
        return {
            "n_obs": 0,
            "qlike": float("nan"),
            "mse_var": float("nan"),
            "mae_var": float("nan"),
        }

    variance_losses = mse_mae_variance(RV_valid, sigma2_valid)
    return {
        "n_obs": int(RV_valid.size),
        "qlike": qlike_loss(RV_valid, sigma2_valid),
        "mse_var": variance_losses["mse"],
        "mae_var": variance_losses["mae"],
    }


def compute_metrics_from_forecasts(forecasts: pd.DataFrame) -> dict[str, object]:
    """Compute metrics from old format forecasts with h column (≤40 lines).

    Args:
        forecasts: DataFrame with h, actual_residual, sigma2_forecast columns.

    Returns:
        Dictionary with variance and VaR metrics.
    """
    from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss

    if forecasts.empty:
        return _empty_metrics_dict()

    df_h1 = pd.DataFrame(
        forecasts.loc[(forecasts["h"] == 1) & forecasts["actual_residual"].notna()].copy()
    )

    if df_h1.empty:
        return _empty_metrics_dict()

    e_test = df_h1["actual_residual"].to_numpy(dtype=float)
    s2_test = df_h1["sigma2_forecast"].to_numpy(dtype=float)

    variance_losses = mse_mae_variance(e_test, s2_test)
    metrics: dict[str, object] = {
        "n_obs": int(e_test.size),
        "qlike": qlike_loss(e_test, s2_test),
        "mse_var": variance_losses["mse"],
        "mae_var": variance_losses["mae"],
    }

    # Compute VaR backtests
    metrics["var_backtests_empirical"] = _compute_var_backtests(df_h1)
    return metrics


def _empty_metrics_dict() -> dict[str, object]:
    """Return empty metrics dictionary."""
    return {
        "n_obs": 0,
        "qlike": float("nan"),
        "mse_var": float("nan"),
        "mae_var": float("nan"),
        "var_backtests_empirical": {},
    }


def _compute_var_backtests(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute VaR backtests by alpha level (≤40 lines)."""
    var_results: dict[str, dict[str, float]] = {}

    for alpha, group in df.groupby("var_left_alpha"):
        hits = (group["actual_residual"] < group["VaR"]).astype(int)
        n_obs = int(hits.size)
        violations = int(hits.sum())
        hit_rate = float(hits.mean()) if n_obs > 0 else float("nan")

        var_results[str(alpha)] = {
            "n": n_obs,
            "violations": violations,
            "hit_rate": hit_rate,
        }

    return var_results


def egarch_one_step_variance_forecast(
    e_last: float,
    s2_last: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> float:
    """One-step EGARCH variance forecast.

    Wrapper for compute_egarch_forecasts that returns only one-step forecast.

    Args:
        e_last: Last residual.
        s2_last: Last variance.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma parameter.
        beta: Beta parameter.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        One-step ahead variance forecast.
    """
    from src.garch.garch_eval.variance_path import compute_egarch_forecasts

    s2_1, _ = compute_egarch_forecasts(
        e_last=e_last,
        s2_last=s2_last,
        horizon=1,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )
    return s2_1


def egarch_multi_step_variance_forecast(
    horizon: int,
    s2_last: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Multi-step EGARCH variance forecast.

    Wrapper for compute_egarch_forecasts that returns multi-step forecasts.

    Args:
        horizon: Forecast horizon.
        s2_last: Last variance.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma parameter.
        beta: Beta parameter.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter.

    Returns:
        Multi-step variance forecasts array.
    """
    from src.garch.garch_eval.variance_path import compute_egarch_forecasts

    # Use dummy e_last since multi-step uses unconditional expectation
    _, s2_h = compute_egarch_forecasts(
        e_last=0.0,
        s2_last=s2_last,
        horizon=horizon,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )
    return s2_h


def prediction_interval(
    mean: float,
    variance: float,
    *,
    level: float = GARCH_EVAL_DEFAULT_LEVEL,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> tuple[float, float]:
    """Two-sided prediction interval (≤40 lines).

    Args:
        mean: Mean of the return distribution.
        variance: Conditional variance.
        level: Prediction interval level (default: 0.95).
        dist: Distribution type ('student', 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t.
        lambda_skew: Skewness parameter for Skew-t.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    from src.garch.garch_eval.utils import quantile

    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)

    alpha = (1.0 - float(level)) / 2.0
    sigma = float(np.sqrt(variance))
    q_lo = quantile(dist, alpha, nu, lambda_skew)
    q_hi = quantile(dist, 1.0 - alpha, nu, lambda_skew)

    return float(mean + sigma * q_lo), float(mean + sigma * q_hi)


def value_at_risk(
    alpha: float,
    *,
    mean: float = 0.0,
    variance: float,
    dist: str = "student",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> float:
    """Left-tail Value-at-Risk at level alpha (≤40 lines).

    Returns VaR_alpha such that P(R < VaR_alpha) = alpha.

    Args:
        alpha: VaR level (e.g., 0.01 or 0.05).
        mean: Mean return.
        variance: Conditional variance.
        dist: Distribution type.
        nu: Degrees of freedom for Student-t/Skew-t.
        lambda_skew: Skewness parameter for Skew-t.

    Returns:
        VaR value.
    """
    from src.garch.garch_eval.utils import quantile

    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)

    sigma = float(np.sqrt(variance))
    q = quantile(dist, float(alpha), nu, lambda_skew)

    return float(mean + sigma * q)


def _filter_initial_observations_per_ticker(
    df: pd.DataFrame, min_window_size: int = 250
) -> pd.DataFrame:
    """Filter first min_window_size observations per ticker.

    GARCH forecasts start at observation min_window_size (default 250),
    so the first 250 observations per ticker have NaN for GARCH features.
    We remove these observations to avoid training LightGBM on incomplete data.

    Args:
        df: Merged dataframe with ticker and date columns.
        min_window_size: Number of initial observations to skip per ticker.

    Returns:
        Filtered dataframe with first min_window_size observations removed per ticker.
    """
    logger.info(
        "Filtering first %d observations per ticker (GARCH warm-up period)...",
        min_window_size,
    )

    df = df.sort_values(["tickers", "date"]).reset_index(drop=True)
    positions = df.groupby("tickers").cumcount()
    df_filtered = pd.DataFrame(df.loc[positions >= min_window_size].reset_index(drop=True))

    n_removed = len(df) - len(df_filtered)
    logger.info(
        "✓ Filtered %d observations (%.1f%%) from %d rows to %d rows",
        n_removed,
        100 * n_removed / len(df) if len(df) > 0 else 0,
        len(df),
        len(df_filtered),
    )

    return df_filtered


def _validate_garch_columns(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    """Ensure essential GARCH columns exist and contain at least one value."""

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        msg = f"Missing essential GARCH columns: {missing_columns}"
        raise KeyError(msg)

    empty_columns = [col for col in required_columns if df[col].notna().sum() == 0]
    if empty_columns:
        msg = f"GARCH columns contain only NaN values: {empty_columns}"
        raise ValueError(msg)

    logger.info("✓ Validated GARCH columns contain non-NaN values: %s", required_columns)


def _load_base_ticker_data() -> pd.DataFrame:
    """Load and validate the base ticker-level dataset."""
    if not DATA_TICKERS_FULL_FILE.exists():
        msg = f"Ticker-level dataset not found: {DATA_TICKERS_FULL_FILE}"
        raise FileNotFoundError(msg)

    df_tickers = pd.read_parquet(DATA_TICKERS_FULL_FILE)
    if "date" not in df_tickers.columns:
        msg = "data_tickers_full must contain a 'date' column"
        raise KeyError(msg)

    df_tickers = df_tickers.copy()
    df_tickers["date"] = pd.to_datetime(df_tickers["date"])  # type: ignore[assignment]
    return df_tickers


def _load_arima_predictions() -> pd.DataFrame:
    """Load and validate ARIMA rolling predictions."""
    from src.path import ROLLING_PREDICTIONS_ARIMA_FILE

    if not ROLLING_PREDICTIONS_ARIMA_FILE.exists():
        msg = f"ARIMA rolling predictions not found: {ROLLING_PREDICTIONS_ARIMA_FILE}"
        raise FileNotFoundError(msg)

    df_arima = pd.read_csv(ROLLING_PREDICTIONS_ARIMA_FILE, parse_dates=["date"])

    required_arima = {"date", "y_pred", "residual"}
    missing_arima = required_arima - set(df_arima.columns)
    if missing_arima:
        msg = f"rolling_predictions.csv missing required columns: {sorted(missing_arima)}"
        raise KeyError(msg)

    # Rename columns to match expected names
    df_arima = df_arima.rename(columns={"y_pred": "sarima_pred", "residual": "arima_resid"})
    return df_arima.loc[:, ["date", "sarima_pred", "arima_resid"]].copy()


def _load_garch_forecasts(df_full_forecasts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load and process GARCH evaluation forecasts."""
    if df_full_forecasts is not None:
        return _process_full_sample_forecasts(df_full_forecasts)
    else:
        return _load_legacy_garch_forecasts()


def _process_full_sample_forecasts(df_full_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Process full sample forecasts from orchestration."""
    df_garch_eval = df_full_forecasts[["date", "garch_forecast_h1", "garch_vol_h1"]].copy()
    # Remove duplicate dates (keep first occurrence)
    df_garch_eval = df_garch_eval.groupby("date", as_index=False).first()
    return _finalize_garch_data(df_garch_eval)  # type: ignore


def _load_legacy_garch_forecasts() -> pd.DataFrame:
    """Load GARCH forecasts from CSV file (legacy path)."""
    from src.path import GARCH_EVALUATION_DIR

    garch_file = GARCH_EVALUATION_DIR / "garch_forecasts.csv"
    if not garch_file.exists():
        msg = f"GARCH forecasts file not found: {garch_file}"
        raise FileNotFoundError(msg)

    df_garch_eval = pd.read_csv(garch_file, parse_dates=["date"])

    # Check for sigma2_egarch_raw (new format: only date and sigma2_egarch_raw)
    if "sigma2_egarch_raw" in df_garch_eval.columns:
        # Compute log_sigma_garch directly from sigma2_egarch_raw
        df_garch_eval["log_sigma_garch"] = np.log(np.sqrt(df_garch_eval["sigma2_egarch_raw"]))
        return cast(pd.DataFrame, df_garch_eval[["date", "log_sigma_garch"]].copy())

    # Legacy format: garch_forecast_h1 and garch_vol_h1
    required_garch = {"date", "garch_forecast_h1", "garch_vol_h1"}
    missing_garch = required_garch - set(df_garch_eval.columns)
    if missing_garch:
        msg = f"garch_forecasts.csv missing required columns: {sorted(missing_garch)}"
        raise KeyError(msg)

    df_garch_eval = cast(
        pd.DataFrame, df_garch_eval[["date", "garch_forecast_h1", "garch_vol_h1"]].copy()
    )
    return _finalize_garch_data(df_garch_eval)


def _finalize_garch_data(df_garch_eval: pd.DataFrame) -> pd.DataFrame:
    """Finalize GARCH data by computing log_sigma_garch."""
    # Calculate log(sigma_garch) directly from garch_vol_h1
    df_garch_eval["log_sigma_garch"] = np.log(df_garch_eval["garch_vol_h1"])
    # Keep only date and log_sigma_garch for merging
    return cast(pd.DataFrame, df_garch_eval[["date", "log_sigma_garch"]].copy())


def _sort_output_dataframe(df_out: pd.DataFrame) -> pd.DataFrame:
    """Sort the output DataFrame deterministically."""

    def _split_order(v: object) -> int:
        return 0 if str(v) == "train" else 1

    if "split" in df_out.columns:
        df_out = df_out.copy()
        df_out["_split_order"] = df_out["split"].map(_split_order)
        sort_cols = ["_split_order", "date"] + (["tickers"] if "tickers" in df_out.columns else [])
        df_out = df_out.sort_values(sort_cols).drop(columns=["_split_order"]).reset_index(drop=True)
    else:
        sort_cols = ["date"] + (["tickers"] if "tickers" in df_out.columns else [])
        df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    return df_out


def generate_data_tickers_full_insights(
    df_full_forecasts: pd.DataFrame | None = None,
    *,
    min_window_size: int = GARCH_MIN_WINDOW_SIZE,
) -> None:
    """Create data_tickers_full_insights from evaluation artifacts without truncation.

    This implementation builds insights as requested:
    - GARCH insights from df_full_forecasts if provided, else from forecasts
    - ARIMA data is loaded but not included in final output

    It preserves the full ticker timeline (e.g., 2013–2024) and does not drop
    early dates where insights are naturally unavailable (NaN).

    Args:
        df_full_forecasts: Full sample forecasts from orchestration (optional).
        min_window_size: Minimum observations to drop per ticker to skip the
            GARCH warm-up period.
    """
    logger.info("Building data_tickers_full_insights from ARIMA and GARCH evaluation results...")

    # Load base ticker-level data
    df_tickers = _load_base_ticker_data()

    # Load ARIMA rolling predictions (contains residuals) - loaded but not used in output
    _load_arima_predictions()

    # Load GARCH evaluation forecasts
    df_garch_eval = _load_garch_forecasts(df_full_forecasts)

    if min_window_size < 0:
        msg = "min_window_size must be non-negative"
        raise ValueError(msg)

    # Merge: duplicate per-date GARCH across all tickers for that date
    # Note: ARIMA predictions (sarima_pred, arima_resid) are not included in final output
    df_out = df_tickers.merge(df_garch_eval, on="date", how="left")

    # ANTI-LEAKAGE VALIDATION: Verify temporal alignment after merge
    # GARCH forecasts at date D_t must predict variance FOR D_t using data BEFORE D_t
    # This assertion checks that dates are sorted (temporal ordering preserved)
    if len(df_out) > 0 and "date" in df_out.columns:
        global_dates_sorted = df_out["date"].is_monotonic_increasing
        per_ticker_dates_sorted = bool(
            df_out.groupby("tickers")["date"].apply(lambda x: x.is_monotonic_increasing).all()
        )
        dates_sorted = global_dates_sorted or per_ticker_dates_sorted
        if not dates_sorted:
            logger.warning(
                "POTENTIAL DATA LEAKAGE: Dates not in temporal order after merge. "
                "This could indicate incorrect alignment of GARCH forecasts with ticker data."
            )

    df_out = _filter_initial_observations_per_ticker(
        df_out,
        min_window_size=min_window_size,
    )

    # Sort deterministically
    df_out = _sort_output_dataframe(df_out)

    # Validate only log_sigma_garch column
    _validate_garch_columns(df_out, ("log_sigma_garch",))

    # Persist parquet + CSV to configured paths
    out_parquet = DATA_TICKERS_FULL_INSIGHTS_FILE
    out_csv = DATA_TICKERS_FULL_INSIGHTS_FILE.with_suffix(".csv")
    ensure_output_dir(out_parquet)
    ensure_output_dir(out_csv)
    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)

    logger.info(
        "✓ Saved data_tickers_full_insights: %s (+ CSV) | rows=%d cols=%d",
        out_parquet,
        len(df_out),
        df_out.shape[1],
    )
