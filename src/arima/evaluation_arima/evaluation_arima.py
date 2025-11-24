"""ARIMA model evaluation module (publication-grade, DRY/KISS).

This module intentionally keeps `src.*` imports intact. It does not
introduce new package-level dependencies beyond statsmodels/sklearn/matplotlib.

Exposed functions:
- rolling_forecast
- evaluate_model
- calculate_metrics
- ljung_box_on_residuals
- plot_residuals_acf_with_ljungbox
- save_evaluation_results
- save_ljung_box_results

Note: compute_residuals is imported from src.utils for consistency
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

# Note: We handle date frequency warnings by ensuring series have proper frequency

# Third-party (optional imports are guarded at runtime)
try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf # type: ignore[import-untyped]
    from statsmodels.stats.diagnostic import acorr_ljungbox # type: ignore[import-untyped]
    from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore[import-untyped]
except Exception as _e:  # pragma: no cover
    SARIMAX = None  # type: ignore
    plot_acf = None  # type: ignore
    acorr_ljungbox = None  # type: ignore
    plt = None  # type: ignore

from scipy import stats  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.constants import (  # type: ignore
    ARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS,
    DEFAULT_PLACEHOLDER_DATE,
    EVAL_DPI,
    EVAL_FIGURE_SIZE,
    ROLLING_FORECAST_PROGRESS_INTERVAL,
    TEXT_POSITION_X,
    TEXT_POSITION_Y,
)
from src.path import (  # type: ignore
    ARIMA_RESIDUALS_LJUNGBOX_PLOT,
    LJUNGBOX_RESIDUALS_ARIMA_FILE,
    PREDICTIONS_VS_ACTUAL_ARIMA_PLOT,
    ROLLING_PREDICTIONS_ARIMA_FILE,
    ROLLING_VALIDATION_METRICS_ARIMA_FILE,
)

# Keep project imports as-is (do NOT remove per user's instruction)
from src.utils import (  # type: ignore
    ensure_output_dir,
    format_dates_to_string,
    get_logger,
    save_json_pretty,
    suppress_statsmodels_warnings,
    validate_temporal_order_series,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArimaSpec:
    """Container for ARIMA hyper-parameters."""

    order: tuple[int, int, int]
    trend: str = "c"  # Default trend parameter


# ----------------------------- Utilities ----------------------------- #


def _append_point_to_train_series(
    y_train: pd.Series,
    ts_idx: Any,
    y_true: float,
) -> pd.Series:
    """Append a new point to training series while preserving frequency."""
    train_freq = y_train.index.freq if isinstance(y_train.index, pd.DatetimeIndex) else None
    if train_freq is not None:
        new_idx = pd.date_range(start=ts_idx, periods=1, freq=train_freq)
        new_point = pd.Series([y_true], index=new_idx, name=y_train.name)
    else:
        new_point = pd.Series([y_true], index=[ts_idx], name=y_train.name)
    y_train = pd.concat([y_train, new_point])
    # Ensure frequency is preserved after concat by recreating index if needed
    if isinstance(y_train.index, pd.DatetimeIndex) and y_train.index.freq is None:
        dt_idx = cast(pd.DatetimeIndex, y_train.index)
        inferred = pd.infer_freq(dt_idx) or train_freq or "D"
        y_train.index = pd.date_range(start=dt_idx[0], periods=len(dt_idx), freq=inferred)
    return y_train


def _as_series(x: Iterable[float] | pd.Series, name: str = "y") -> pd.Series:
    """Coerce to pandas Series."""
    if isinstance(x, pd.Series):
        s = x.copy()
        if s.name is None:
            s.name = name
        return s
    s = pd.Series(list(x), name=name)
    return s


def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    """Ensure the index is datetime-like with frequency; if not, fabricate sequential dates.

    - If index is a DatetimeIndex, ensure it has a frequency (infer if missing).
    - Else, create daily dates with frequency "D".
    """
    if isinstance(s.index, pd.DatetimeIndex):
        s2 = s.copy()
        dt_index = cast(pd.DatetimeIndex, s2.index)
        # Ensure frequency is set to avoid statsmodels warnings
        if dt_index.freq is None:
            # Try to infer frequency from the index
            try:
                inferred_freq = pd.infer_freq(dt_index)
                if inferred_freq:
                    # Set frequency on existing index instead of recreating it
                    s2.index.freq = inferred_freq
                else:
                    # Default to irregular frequency if inference fails
                    # Don't force a frequency that doesn't exist
                    pass
            except Exception as e:
                logger.warning(f"Failed to infer frequency: {e}")
                # Don't force a frequency if inference fails
        return s2
    # fabricate deterministic daily index for plotting and merges
    start = pd.Timestamp(DEFAULT_PLACEHOLDER_DATE)
    idx = pd.date_range(start=start, periods=len(s), freq="D")
    s2 = s.copy()
    s2.index = idx
    return s2


def _infer_frequency_from_index(dt_index: pd.DatetimeIndex) -> str | None:
    """Try to infer frequency from DatetimeIndex using pandas infer_freq."""
    try:
        inferred_freq = pd.infer_freq(dt_index)
        return inferred_freq if inferred_freq else None
    except Exception:
        return None


def _infer_frequency_from_differences(dt_index: pd.DatetimeIndex) -> str | None:
    """Try to infer frequency from differences between consecutive dates."""
    try:
        if len(dt_index) <= 1:
            return None

        # Calculate most common difference
        diffs = dt_index.to_series().diff().dropna()
        most_common_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs.iloc[0]

        # Map common differences to frequencies
        if pd.Timedelta(days=1) <= most_common_diff < pd.Timedelta(days=2):
            return "D"
        elif pd.Timedelta(days=7) <= most_common_diff < pd.Timedelta(days=8):
            return "W"
        elif pd.Timedelta(days=30) <= most_common_diff < pd.Timedelta(days=31):
            return "M"
        else:
            return "D"  # Default to daily
    except Exception:
        return None


def _create_datetime_index_with_frequency(
    dt_index: pd.DatetimeIndex, freq: str
) -> pd.DatetimeIndex:
    """Create a new DatetimeIndex with specified frequency."""
    return pd.date_range(start=dt_index[0], periods=len(dt_index), freq=freq)


def _ensure_frequency(y: pd.Series) -> pd.Series:
    """Ensure series has a DatetimeIndex with frequency to avoid statsmodels warnings.

    Args:
        y: Input series (may have DatetimeIndex without frequency).

    Returns:
        Series with DatetimeIndex that has a frequency set.

    Raises:
        ValueError: If frequency cannot be inferred from data and no frequency is set.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        # If not DatetimeIndex, return as-is (statsmodels will handle it)
        return y

    y_copy = y.copy()
    dt_index = cast(pd.DatetimeIndex, y_copy.index)

    # If frequency is already set, return as-is
    if dt_index.freq is not None:
        return y_copy

    # Try to infer frequency from the index
    inferred_freq = _infer_frequency_from_index(dt_index)
    if inferred_freq:
        logger.debug(f"Inferred frequency from index: {inferred_freq}")
        y_copy.index = _create_datetime_index_with_frequency(dt_index, inferred_freq)
        return y_copy

    # If inference fails, try to infer from differences
    diff_freq = _infer_frequency_from_differences(dt_index)
    if diff_freq:
        logger.warning(
            f"Could not infer frequency from index, using difference-based inference: {diff_freq}. "
            f"First dates: {dt_index[:5].tolist()}"
        )
        y_copy.index = _create_datetime_index_with_frequency(dt_index, diff_freq)
        return y_copy

    # CRITICAL: Raise exception instead of silent fallback
    msg = (
        f"Could not infer frequency from DatetimeIndex. "
        f"Index has {len(dt_index)} points. First 5 dates: {dt_index[:5].tolist()}, "
        f"Last 5 dates: {dt_index[-5:].tolist()}. "
        f"Please ensure data has consistent temporal spacing."
    )
    logger.error(msg)
    raise ValueError(msg)


def _fit_arima(y: pd.Series, spec: ArimaSpec) -> Any:
    """Fit ARIMA and return fitted model.

    Ensures series has proper frequency to avoid statsmodels warnings.
    """
    if SARIMAX is None:  # pragma: no cover
        raise RuntimeError("statsmodels is required for SARIMAX.")

    # Ensure frequency is set to avoid warnings
    y_with_freq = _ensure_frequency(y)

    model = SARIMAX(
        y_with_freq,
        order=spec.order,
        seasonal_order=(0, 0, 0, 0),  # No seasonal components for ARIMA
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend=spec.trend,
    )
    return model.fit(disp=False)


def _make_one_step_forecast(fitted_model: Any, y_train: pd.Series, step: int) -> float:
    """Make a one-step forecast or raise a RuntimeError if it fails."""
    try:
        forecast_result = fitted_model.forecast(steps=1)  # type: ignore
        return float(
            forecast_result.iloc[0] if hasattr(forecast_result, "iloc") else forecast_result[0]
        )
    except Exception as exc:
        logger.error(
            "One-step forecast failed at step %s (train_length=%s): %s",
            step + 1,
            len(y_train),
            exc,
        )
        raise RuntimeError(f"One-step forecast failed at step {step + 1}: {exc}") from exc


# ----------------------------- Core API ----------------------------- #


def _setup_rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    trend: str = "c",
) -> tuple[pd.Series, pd.Series, ArimaSpec]:
    """Setup rolling forecast: filter warnings, prepare series, validate order."""
    suppress_statsmodels_warnings()

    y_train = _ensure_datetime_index(_as_series(train_series, "y_train"))
    y_test = _ensure_datetime_index(_as_series(test_series, "y_test"))
    validate_temporal_order_series(y_train, y_test, function_name="rolling_forecast")

    spec = ArimaSpec(order=order, trend=trend)
    return y_train, y_test, spec


def _execute_rolling_forecast_loop(
    y_train: pd.Series,
    y_test: pd.Series,
    spec: ArimaSpec,
    refit_every: int,
    verbose: bool,
) -> tuple[list[float], list[float]]:
    """Execute the main rolling forecast loop."""
    preds: list[float] = []
    actuals: list[float] = []
    fitted_model = None

    for i, (ts_idx, y_true) in enumerate(zip(y_test.index, y_test.values, strict=False)):
        if i == 0 or (refit_every > 0 and i % refit_every == 0) or fitted_model is None:
            fitted_model = _fit_arima(y_train, spec)
            if verbose:
                logger.info(f"Refit at step {i+1}/{len(y_test)} (train n={len(y_train)})")
        pred = _make_one_step_forecast(fitted_model, y_train, i)
        preds.append(pred)
        actuals.append(float(y_true))
        y_train = _append_point_to_train_series(y_train, ts_idx, y_true)
        if verbose and (i + 1) % ROLLING_FORECAST_PROGRESS_INTERVAL == 0:
            logger.info(f"Rolling forecast progress: {i+1}/{len(y_test)}")
    return preds, actuals


def rolling_forecast(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    refit_every: int,
    verbose: bool = False,
    trend: str = "c",
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling one-step-ahead forecast with periodic refits.

    For t in test horizon:
      - If t % refit_every == 0: refit ARIMA on all available data up to t-1
      - Else: reuse previous fitted model and update via append (fast)

    Args:
        train_series: Training data series.
        test_series: Test data series.
        order: ARIMA order (p, d, q).
        refit_every: Number of steps between model refits (default: 20).
        verbose: Whether to log progress (default: False).
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".

    Returns
    -------
    preds : np.ndarray
        Forecasts aligned with test_series.
    actuals : np.ndarray
        Actual values from test_series.
    """
    y_train, y_test, spec = _setup_rolling_forecast(train_series, test_series, order, trend)
    try:
        preds, actuals = _execute_rolling_forecast_loop(y_train, y_test, spec, refit_every, verbose)
    except RuntimeError as exc:
        logger.error(
            "Rolling forecast aborted due to a forecasting failure; propagating error to CLI."
        )
        raise RuntimeError("Rolling forecast failed during evaluation.") from exc
    return np.array(preds, dtype=float), np.array(actuals, dtype=float)


def _create_non_nan_mask(y_true_s: pd.Series, y_pred_s: pd.Series) -> pd.Series:
    """Create a boolean mask for non-NaN values in both series."""
    return y_true_s.notna() & y_pred_s.notna()  # type: ignore[return-value]


def calculate_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Compute standard regression metrics (MSE, RMSE, MAE)."""
    y_true_s = _as_series(y_true, "y_true")
    y_pred_s = _as_series(y_pred, "y_pred")

    # Filter out NaN values
    mask = _create_non_nan_mask(y_true_s, y_pred_s)
    y_true_clean = y_true_s[mask]
    y_pred_clean = y_pred_s[mask]

    # Check if all predictions are NaN
    if len(y_pred_clean) == 0:
        raise RuntimeError("All predictions are NaN")

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae)}


# Note: compute_residuals is now imported from src.utils for consistency


def ljung_box_on_residuals(residuals: Iterable[float], lags: int) -> dict[str, Any]:
    """Run Ljung–Box test on residuals and return dict with statistics and p-values."""
    if acorr_ljungbox is None:  # pragma: no cover
        raise RuntimeError("statsmodels is required for Ljung–Box.")
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]
    lags_list = list(range(1, lags + 1))
    lb = acorr_ljungbox(res, lags=lags_list, return_df=True)
    # Extract statistics for all lags
    q_stat = lb["lb_stat"].values.tolist()
    p_value = lb["lb_pvalue"].values.tolist()
    reject_5pct = (lb["lb_pvalue"] < 0.05).values.tolist()
    out = {
        "lags": lags_list,
        "q_stat": q_stat,
        "p_value": p_value,
        "reject_5pct": reject_5pct,
        "n": int(res.size),
    }
    return out


def _compute_jarque_bera_statistics(res: np.ndarray) -> dict[str, float]:
    """Compute Jarque-Bera test statistics and moments.

    Args:
        res: Cleaned residuals array (finite values only).

    Returns:
        Dict with statistic, p_value, skewness, kurtosis, n.
    """
    jb_result = stats.jarque_bera(res)
    jb_stat_val = cast(float, jb_result[0])
    p_val_val = cast(float, jb_result[1])
    jb_stat = float(jb_stat_val)
    p_val = float(p_val_val)
    skew = float(stats.skew(res))
    kurt = float(stats.kurtosis(res, fisher=True))  # Excess kurtosis (0 for normal)

    return {
        "statistic": jb_stat,
        "p_value": p_val,
        "skewness": skew,
        "kurtosis": kurt,
        "n": int(res.size),
    }


def jarque_bera_test(residuals: Iterable[float]) -> dict[str, float]:
    """Jarque-Bera test for normality of residuals.

    Tests whether residuals have skewness and kurtosis matching a normal distribution.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with statistic, p_value, skewness, kurtosis, n.

    Note:
        - p-value > 0.05 suggests residuals are normally distributed
        - p-value < 0.05 suggests deviation from normality
    """
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if res.size < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "n": int(res.size),
        }

    return _compute_jarque_bera_statistics(res)


def _compute_shapiro_wilk_statistics(res: np.ndarray) -> dict[str, float]:
    """Compute Shapiro-Wilk test statistics.

    Args:
        res: Cleaned residuals array (finite values only).

    Returns:
        Dict with statistic, p_value, n.
    """
    if res.size > 5000:
        logger.warning(
            "Shapiro-Wilk test: sample size %d exceeds 5000. Consider using Jarque-Bera instead.",
            res.size,
        )

    sw_stat, p_val = stats.shapiro(res)

    return {
        "statistic": float(sw_stat),
        "p_value": float(p_val),
        "n": int(res.size),
    }


def shapiro_wilk_test(residuals: Iterable[float]) -> dict[str, float]:
    """Shapiro-Wilk test for normality of residuals.

    More powerful than Jarque-Bera for small samples (n < 2000).
    Tests whether residuals come from a normal distribution.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with statistic, p_value, n.

    Note:
        - p-value > 0.05 suggests residuals are normally distributed
        - p-value < 0.05 suggests deviation from normality
        - Test is most powerful for n between 3 and 5000
    """
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if res.size < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n": int(res.size),
        }

    return _compute_shapiro_wilk_statistics(res)


def _build_critical_values_dict(
    significance_levels: np.ndarray, critical_values: np.ndarray
) -> dict[str, float]:
    """Build critical values dictionary from Anderson-Darling result.

    Args:
        significance_levels: Array of significance levels.
        critical_values: Array of critical values.

    Returns:
        Dict mapping significance level strings to critical values.
    """
    crit_dict = {}
    for sig_level, crit_val in zip(significance_levels, critical_values, strict=False):
        crit_dict[f"{sig_level}%"] = float(crit_val)
    return crit_dict


def _compute_anderson_darling_statistics(res: np.ndarray) -> dict[str, Any]:
    """Compute Anderson-Darling test statistics.

    Args:
        res: Cleaned residuals array (finite values only).

    Returns:
        Dict with statistic, critical_values, significance_levels, n.
    """
    result = stats.anderson(res, dist="norm")

    significance_levels = getattr(result, "significance_level", np.array([]))
    critical_values = getattr(result, "critical_values", np.array([]))
    statistic = getattr(result, "statistic", float("nan"))

    crit_dict = _build_critical_values_dict(significance_levels, critical_values)

    return {
        "statistic": float(statistic),
        "critical_values": crit_dict,
        "significance_levels": (
            significance_levels.tolist() if hasattr(significance_levels, "tolist") else []
        ),
        "n": int(res.size),
    }


def anderson_darling_test(residuals: Iterable[float]) -> dict[str, Any]:
    """Anderson-Darling test for normality of residuals.

    Tests whether residuals come from a normal distribution.
    More sensitive to tails than Jarque-Bera or Shapiro-Wilk.

    H0: Residuals are normally distributed
    H1: Residuals are not normally distributed

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with statistic, critical_values, significance_levels, n.

    Note:
        - If statistic < critical_value at a given significance level,
          do not reject H0 at that level
        - If statistic > critical_value, reject H0
        - Critical values provided for [15%, 10%, 5%, 2.5%, 1%]
    """
    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if res.size < 3:
        return {
            "statistic": float("nan"),
            "critical_values": {},
            "significance_levels": [],
            "n": int(res.size),
        }

    return _compute_anderson_darling_statistics(res)


def run_all_normality_tests(residuals: Iterable[float]) -> dict[str, dict[str, Any]]:
    """Run all three normality tests on residuals.

    Convenience function that runs:
    1. Jarque-Bera test (good for large samples, based on moments)
    2. Shapiro-Wilk test (powerful for small/medium samples)
    3. Anderson-Darling test (sensitive to tail behavior)

    Args:
        residuals: Residual series from ARIMA model.

    Returns:
        Dict with keys: jarque_bera, shapiro_wilk, anderson_darling.
        Each value is a dict with test statistics and results.

    Note:
        Consistent rejection across all tests provides strong evidence
        against normality assumption.
    """
    return {
        "jarque_bera": jarque_bera_test(residuals),
        "shapiro_wilk": shapiro_wilk_test(residuals),
        "anderson_darling": anderson_darling_test(residuals),
    }


def plot_residuals_acf_with_ljungbox(
    residuals: Iterable[float],
    lags: int,
    out_path: Path | None = None,
) -> Path:
    """Plot ACF of residuals and annotate Ljung–Box p-value."""
    if plt is None or plot_acf is None:  # pragma: no cover
        raise RuntimeError("matplotlib/statsmodels required for plotting.")

    res = np.asarray(list(residuals), dtype=float)
    res = res[np.isfinite(res)]

    if out_path is None:
        out_path = Path(ARIMA_RESIDUALS_LJUNGBOX_PLOT)

    out_path = Path(out_path)
    ensure_output_dir(out_path)

    fig = plt.figure(figsize=EVAL_FIGURE_SIZE)
    ax = fig.add_subplot(111)
    plot_acf(res, lags=lags, ax=ax, zero=False)  # Exclude lag 0 (always 1.0)
    ax.set_title(f"Residuals ACF (lags={lags})")

    lb = ljung_box_on_residuals(res, lags=lags)
    # Use the last p-value (cumulative test up to max lag)
    p_value = lb["p_value"][-1] if lb["p_value"] else np.nan
    txt = f"Ljung–Box (lag={lags}) p-value={p_value:.4f}, n={lb['n']}"
    ax.text(TEXT_POSITION_X, TEXT_POSITION_Y, txt, transform=ax.transAxes, va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=EVAL_DPI)
    plt.close(fig)
    logger.info(f"Saved residuals ACF + Ljung–Box: {out_path}")
    return out_path


def plot_predictions_vs_actual(
    predictions_file: Path | str | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Plot ARIMA predictions vs actual values for publication-ready evaluation.

    Creates a time series plot showing:
    - Actual values (y_true)
    - ARIMA predictions (y_pred)
    - Confidence band showing prediction accuracy

    Args:
        predictions_file: Path to rolling_predictions.csv. If None, uses default path.
        out_path: Output path for the plot. If None, uses default path.

    Returns:
        Path to saved plot.

    Raises:
        RuntimeError: If matplotlib is not available or file cannot be read.
    """
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib required for plotting.")

    # Set default paths
    if predictions_file is None:
        predictions_file = ROLLING_PREDICTIONS_ARIMA_FILE
    if out_path is None:
        out_path = PREDICTIONS_VS_ACTUAL_ARIMA_PLOT

    predictions_file = Path(predictions_file)
    out_path = Path(out_path)
    ensure_output_dir(out_path)

    # Load predictions data
    try:
        df = pd.read_csv(predictions_file)
        if df.empty:
            raise ValueError("Predictions file is empty")
    except Exception as e:
        raise RuntimeError(f"Failed to load predictions file {predictions_file}: {e}") from e

    # Validate required columns
    required_cols = ["date", "y_true", "y_pred"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Predictions file missing required columns: {missing_cols}")

    # Convert date column and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna()

    if len(df) == 0:
        raise ValueError("No valid data found after cleaning")

    # Calculate error metrics for annotation
    mse = mean_squared_error(df["y_true"], df["y_pred"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df["y_true"], df["y_pred"])

    # Create publication-ready plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=EVAL_DPI)

    # Plot actual values
    ax.plot(
        df["date"],
        df["y_true"],
        label="Valeurs réelles",
        color="#1f77b4",
        linewidth=2,
        alpha=0.9,
        marker="o",
        markersize=3,
        markerfacecolor="white",
    )

    # Plot predictions
    ax.plot(
        df["date"],
        df["y_pred"],
        label="Prédictions ARIMA",
        color="#ff7f0e",
        linewidth=2,
        alpha=0.9,
        marker="s",
        markersize=3,
        markerfacecolor="white",
    )

    # Add grid and styling
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Date", fontsize=14, fontweight="bold")
    ax.set_ylabel("Rendements logarithmiques", fontsize=14, fontweight="bold")
    ax.set_title(
        "Évaluation ARIMA: Prédictions vs Valeurs Réelles\n"
        + f"(MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Format dates
    ax.tick_params(axis="x", rotation=45)
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Add legend
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)

    # Add performance metrics as text box
    metrics_text = f"""Statistiques de performance:
• MSE: {mse:.6f}
• RMSE: {rmse:.6f}
• MAE: {mae:.6f}
• N observations: {len(df):,}
"""

    # Position text box in upper right
    ax.text(
        0.98,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=1
        ),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=EVAL_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved ARIMA predictions vs actual plot: {out_path}")
    return out_path


def _ensure_path(path_like: Path | str | PathLike[str] | Any) -> Path:
    """Convert arbitrary path-like objects (including mocks) to :class:`Path`."""
    if isinstance(path_like, Path):
        return path_like
    if isinstance(path_like, str):
        return Path(path_like)
    try:
        return Path(path_like)
    except TypeError:
        return cast(Path, path_like)


def _prepare_predictions_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Prepare predictions dataframe from results dict.

    Args:
        results: Evaluation results dictionary.

    Returns:
        DataFrame with date, y_true, y_pred, residual columns.
    """
    y_true = results.get("y_true", results.get("actuals", []))
    y_pred = results.get("y_pred", results.get("predictions", []))
    residuals = results.get("residuals", [])
    if not residuals and y_true and y_pred:
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    df = pd.DataFrame(
        {
            "date": format_dates_to_string(results["dates"]),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "residual": np.asarray(residuals, dtype=float),
        }
    )
    return df


def _prepare_metrics_payload(results: dict[str, Any], n_test: int) -> dict[str, Any]:
    """Prepare metrics payload for JSON serialization.

    Args:
        results: Evaluation results dictionary.
        n_test: Number of test observations.

    Returns:
        Metrics payload dictionary.
    """
    metrics = results.get("metrics", {})
    payload = {
        "order": list(results.get("order", [])),
        "refit_every": int(results["refit_every"]),
        "metrics": metrics,
        "n_test": n_test,
    }
    return payload


def save_evaluation_results(results: dict[str, Any]) -> tuple[Path, Path]:
    """Persist rolling predictions and metrics to disk using constants paths."""
    preds_file = _ensure_path(ROLLING_PREDICTIONS_ARIMA_FILE)
    metrics_file = _ensure_path(ROLLING_VALIDATION_METRICS_ARIMA_FILE)

    ensure_output_dir(preds_file)
    ensure_output_dir(metrics_file)

    df = _prepare_predictions_dataframe(results)
    df.to_csv(preds_file, index=False)

    payload = _prepare_metrics_payload(results, len(df))
    save_json_pretty(payload, metrics_file)

    logger.info(f"Saved predictions → {preds_file}")
    logger.info(f"Saved metrics → {metrics_file}")
    return preds_file, metrics_file


def save_ljung_box_results(lb_result: dict[str, Any]) -> Path:
    """Persist Ljung–Box results as JSON."""
    path = Path(LJUNGBOX_RESIDUALS_ARIMA_FILE)
    ensure_output_dir(path)
    save_json_pretty(lb_result, path)
    logger.info(f"Saved Ljung–Box results → {path}")
    return path


def _setup_backtest_configuration(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    trend: str,
) -> tuple[pd.Series, list[Any], ArimaSpec, int, int]:
    """Setup configuration for full series backtest.

    Returns y_full, original_dates, spec, min_train_size, train_end_index.
    """
    full_series = pd.concat([train_series, test_series]).sort_index()

    if isinstance(full_series.index, pd.DatetimeIndex):
        original_dates = full_series.index.tolist()
    else:
        original_dates = pd.to_datetime(full_series.index).tolist()

    y_full = _ensure_datetime_index(_as_series(full_series, "y_full"))
    spec = ArimaSpec(order=order, trend=trend)
    suppress_statsmodels_warnings()

    min_train_size = min(ARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS, len(y_full) // 4)
    if min_train_size < ARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS // 2:
        min_train_size = min(ARIMA_MIN_TRAIN_SIZE_FOR_RESIDUALS // 2, len(y_full) // 2)

    train_end_index = len(train_series)
    return y_full, original_dates, spec, min_train_size, train_end_index


def _run_backtest_loop(
    y_full: pd.Series,
    spec: ArimaSpec,
    min_train_size: int,
    refit_every: int,
    verbose: bool,
) -> tuple[list[float], list[float]]:
    """Execute backtest loop and return predictions and actuals."""
    preds: list[float] = []
    actuals: list[float] = []
    fitted_model = None

    for i in range(min_train_size, len(y_full)):
        y_train_current = y_full.iloc[:i]
        y_true = float(y_full.iloc[i])

        needs_refit = (
            i == min_train_size
            or (refit_every > 0 and (i - min_train_size) % refit_every == 0)
            or fitted_model is None
        )
        if needs_refit:
            fitted_model = _fit_arima(y_train_current, spec)
            if verbose:
                logger.info(
                    f"Backtest refit at step {i}/{len(y_full)} (train n={len(y_train_current)})"
                )

        pred = _make_one_step_forecast(fitted_model, y_train_current, i)
        preds.append(pred)
        actuals.append(y_true)

        if verbose and (i + 1 - min_train_size) % ROLLING_FORECAST_PROGRESS_INTERVAL == 0:
            logger.info(f"Backtest progress: {i + 1}/{len(y_full)}")

    return preds, actuals


def backtest_full_series(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    refit_every: int,
    verbose: bool = False,
    trend: str = "c",
    include_test: bool = True,
) -> pd.DataFrame:
    """Generate walk-forward backtest residuals to avoid look-ahead bias.

    IMPORTANT: This function uses walk-forward forecasting (out-of-sample) instead
    of in-sample fitted values to avoid look-ahead bias from Kalman filtering.
    This ensures residuals are suitable for unbiased GARCH training.

    Args:
        train_series: Training data series.
        test_series: Test data series (required for temporal context).
        order: ARIMA order (p, d, q).
        refit_every: Number of steps between model refits.
        verbose: Whether to log progress.
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".
        include_test: If True, generates residuals for full series (train+test).
            If False, generates residuals for train data only (stops at train_end_index).
            Use False to generate unbiased train residuals for GARCH without
            contaminating them with test predictions. Default: True.

    Returns:
        DataFrame with columns: date, y_true, y_pred, arima_resid.
        - If include_test=True: residuals for train+test dates
        - If include_test=False: residuals for train dates only

    Note:
        Walk-forward methodology ensures each prediction uses only past data,
        avoiding look-ahead bias that would occur with in-sample fitted values.
    """
    y_full, original_dates, spec, min_train_size, train_end_index = _setup_backtest_configuration(
        train_series, test_series, order, trend
    )

    # Ensure frequency is set to avoid statsmodels warnings
    # _ensure_datetime_index may not always set frequency correctly, so we ensure it here
    y_full = _ensure_frequency(y_full)

    # Determine stop index: only train if include_test is False
    stop_index = len(y_full) if include_test else train_end_index

    try:
        # Execute loop but limit the number of iterations when excluding test horizon
        # We do this by slicing y_full to the desired stop_index
        preds, actuals = _run_backtest_loop(
            y_full.iloc[:stop_index], spec, min_train_size, refit_every, verbose
        )
    except RuntimeError as exc:
        logger.error(
            "Full-series backtest aborted because a single-step forecast failed; propagating error."
        )
        raise RuntimeError("Full-series backtest failed.") from exc

    # Compute residuals
    residuals = np.array(actuals) - np.array(preds)

    # Format dates for the forecasted portion
    # If include_test is False, restrict to the train segment only
    if include_test:
        forecast_dates = original_dates[min_train_size:]
    else:
        forecast_dates = original_dates[min_train_size:train_end_index]

    # Create DataFrame
    history = pd.DataFrame(
        {
            "date": format_dates_to_string(forecast_dates),
            "y_true": actuals,
            "y_pred": preds,
            "arima_resid": residuals,
        }
    )

    return history


def evaluate_model(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int],
    refit_every: int,
    verbose: bool = False,
    model_info: dict[str, Any] | None = None,
    trend: str = "c",
) -> dict[str, Any]:
    """End-to-end evaluation producing walk-forward predictions and metrics.

    Executes the optimized rolling forecast to stay consistent with the
    refit cadence discovered during hyper-parameter search, then computes
    standard regression metrics on the resulting out-of-sample predictions.
    A separate single fit on the concatenated series is performed afterwards
    only to expose diagnostics artifacts (fitted model, full residuals) for
    downstream consumers such as the GARCH pipeline.

    Args:
        train_series: Training data series.
        test_series: Test data series.
        order: ARIMA order (p, d, q).
        refit_every: Optimized refit cadence obtained during training.
        verbose: Whether to log progress (default: False).
        model_info: Optional model metadata dictionary.
        trend: Trend component ("n", "c", "t", "ct"). Default: "c".

    Returns:
        Dictionary with evaluation results including metrics and walk-forward predictions.
    """
    if SARIMAX is None:  # pragma: no cover
        raise RuntimeError("statsmodels is required for SARIMAX.")

    try:
        preds_array, actuals_array = rolling_forecast(
            train_series=train_series,
            test_series=test_series,
            order=order,
            refit_every=refit_every,
            verbose=verbose,
            trend=trend,
        )
    except RuntimeError as exc:
        logger.error(
            "Rolling forecast failed while evaluating the ARIMA model; propagating to CLI."
        )
        raise RuntimeError("ARIMA evaluation aborted because rolling forecast failed.") from exc

    preds = np.asarray(preds_array, dtype=float)
    actuals = np.asarray(actuals_array, dtype=float)
    metrics = calculate_metrics(actuals, preds)
    residuals = (actuals - preds).tolist()

    test_dates = list(_as_series(test_series, "y_test").index)
    dates = format_dates_to_string(test_dates).tolist()

    model_name = "ARIMA"
    if model_info and "params" in model_info:
        model_name = str(model_info["params"])

    results: dict[str, Any] = {
        "model": model_name,
        "order": order,
        "metrics": metrics,
        "predictions": preds.tolist(),
        "actuals": actuals.tolist(),
        "dates": dates,
        "refit_every": refit_every,
        "y_true": actuals.tolist(),
        "y_pred": preds.tolist(),
        "residuals": residuals,
    }

    # Preserve diagnostics artifacts for downstream steps (best-effort)
    full_series = pd.concat([train_series, test_series]).sort_index()
    try:
        full_series_freq = _ensure_frequency(full_series)
        spec = ArimaSpec(order=order, trend=trend)
        suppress_statsmodels_warnings()
        if verbose:
            logger.info(
                f"Fitting ARIMA{order} once on full series (train+test, n={len(full_series_freq)}) "
                f"for DIAGNOSTICS ONLY (has look-ahead bias)"
            )
        fitted_model = _fit_arima(full_series_freq, spec)
        results["_fitted_model"] = fitted_model
        results["_full_residuals"] = fitted_model.resid
        results["_full_series"] = full_series_freq

        # CRITICAL WARNING: These artifacts have look-ahead bias
        logger.warning(
            "IMPORTANT: _fitted_model and _full_residuals are for DIAGNOSTICS ONLY. "
            "They contain look-ahead bias from Kalman filtering on full series. "
            "For GARCH training, use walk-forward backtest residuals from "
            "backtest_full_series(include_test=False) instead."
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Full-series diagnostic fit failed: %s", exc)

    return results
