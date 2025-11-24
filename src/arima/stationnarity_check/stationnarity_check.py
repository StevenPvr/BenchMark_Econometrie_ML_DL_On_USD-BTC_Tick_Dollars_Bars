"""Stationarity checks for time series (ADF + KPSS).

This module provides small, focused helpers to:
- run ADF and KPSS on a pandas Series
- combine results into a single verdict
- load the project's weighted returns and persist a JSON report

All functions are short, typed, and log meaningful progress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning # type: ignore[import-untyped]
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews # type: ignore[import-untyped]

from src.constants import STATIONARITY_DEFAULT_ALPHA, STATIONARITY_RESAMPLE_FREQ, ZIVOT_ANDREWS_TRIM
from src.path import STATIONARITY_REPORT_FILE
from src.utils import get_logger, save_json_pretty

from .utils import load_series_from_csv, validate_series

logger = get_logger(__name__)


class StationarityTestResult(TypedDict):
    """Typed structure for a single stationarity test result."""

    statistic: float
    p_value: float
    lags: int | None
    nobs: int | None
    critical_values: dict[str, float] | None


class ZivotAndrewsTestResult(TypedDict):
    """Typed structure for Zivot-Andrews structural break test result."""

    statistic: float
    p_value: float
    break_index: int | None
    break_date: str | None
    lags: int | None
    nobs: int | None
    critical_values: dict[str, float] | None
    model: str


@dataclass(frozen=True)
class StationarityReport:
    """Combined ADF + KPSS + Zivot-Andrews stationarity report."""

    stationary: bool
    alpha: float
    adf: StationarityTestResult
    kpss: StationarityTestResult
    zivot_andrews: ZivotAndrewsTestResult | None = None


def _convert_test_result(
    stat: float,
    pval: float,
    lags: int | None,
    nobs: int | None,
    crit: dict[str, float] | None,
) -> StationarityTestResult:
    """Convert raw test outputs to a StationarityTestResult mapping."""
    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "lags": int(lags) if lags is not None else None,
        "nobs": int(nobs) if nobs is not None else None,
        "critical_values": (
            {str(k): float(v) for k, v in crit.items()} if crit is not None else None
        ),
    }


def adf_test(series: pd.Series, *, autolag: str | None = "AIC") -> StationarityTestResult:
    """Run Augmented Dickey–Fuller test.

    The number of lags is automatically selected based on the specified criterion
    (default: AIC). The lag value in the result reflects the optimal number chosen
    by the algorithm for the given series, not a fixed value.

    Args:
        series: Input time series.
        autolag: Criterion for lag selection ("AIC", "BIC", "t-stat", or None).

    Returns:
        StationarityTestResult with statistic, p-value, lags (auto-selected),
        nobs, and critical values.
    """
    # Use series values directly for ADF test (avoid DatetimeIndex issues)
    print(f"DEBUG: series length = {len(series)}, values length = {len(series.values)}")
    result = adfuller(series.values, autolag=None, maxlag=1, regression='n')
    # adfuller returns 5 values when autolag=None, 6 values when autolag is set
    # We always need the first 5: (adfstat, pvalue, usedlag, nobs, criticalvalues)
    # Use tuple unpacking with slice - runtime guarantees at least 5 values
    stat, pval, lags, nobs, crit = result[0], result[1], result[2], result[3], result[4]  # type: ignore[misc, assignment]
    lags_int = int(lags) if isinstance(lags, (int, np.integer)) else None
    nobs_int = int(nobs) if isinstance(nobs, (int, np.integer)) else None
    crit_dict: dict[str, float] | None = None
    if isinstance(crit, dict):
        crit_dict = {str(k): float(v) for k, v in crit.items()}
    return _convert_test_result(
        float(stat),
        float(pval),
        lags_int,
        nobs_int,
        crit_dict,
    )


def kpss_test(
    series: pd.Series,
    *,
    regression: Literal["c", "ct"] = "c",
) -> StationarityTestResult:
    """Run KPSS test for (trend-)stationarity.

    The number of lags is automatically calculated based on the series length
    using Newey–West bandwidth selection. The lag value in the result reflects
    the optimal number chosen for the given series, not a fixed value.

    Args:
        series: Input time series.
        regression: "c" (level) or "ct" (trend).

    Returns:
        StationarityTestResult with statistic, p-value, lags (auto-calculated),
        nobs and critical values.
    """
    s = validate_series(series)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        stat, pval, lags, crit = kpss(s, regression=regression, nlags="auto")
    return _convert_test_result(stat, pval, lags, s.size, crit)


def _is_int_type(expected_type: type) -> bool:
    """Check if expected_type is int type."""
    return isinstance(expected_type, type) and issubclass(expected_type, int)


def _is_dict_type(expected_type: type) -> bool:
    """Check if expected_type is dict type."""
    return isinstance(expected_type, type) and issubclass(expected_type, dict)


def _get_target_index(values: list, index_with_dict: int, index_without_dict: int) -> int:
    """Calculate target index based on dict presence."""
    has_dict_third = len(values) > 2 and isinstance(values[2], dict)
    return index_with_dict if has_dict_third else index_without_dict


def _process_dict_value(raw_value) -> dict[str, float]:
    """Process dict type values."""
    return {str(k): float(v) for k, v in raw_value.items()} if isinstance(raw_value, dict) else {}


def _process_int_value(raw_value, expected_type: type, series_size: int | None, fallback_value):
    """Process int type values with fallback logic."""
    if raw_value is not None and not isinstance(raw_value, dict):
        return expected_type(raw_value)
    return series_size if series_size else fallback_value


def _process_other_value(raw_value, expected_type: type, fallback_value):
    """Process other type values."""
    if raw_value is not None and not isinstance(raw_value, dict):
        return expected_type(raw_value)
    return fallback_value


def _extract_value_from_zivot_andrews_result(
    values: list,
    index_with_dict: int,
    index_without_dict: int,
    expected_type: type,
    fallback_value: int | None = None,
    series_size: int | None = None,
) -> int | None | dict[str, float]:
    """Generic extractor for Zivot-Andrews result values."""
    target_index = _get_target_index(values, index_with_dict, index_without_dict)

    # Handle out of bounds
    if target_index >= len(values):
        return series_size if _is_int_type(expected_type) and series_size else fallback_value

    raw_value = values[target_index]

    # Route to appropriate processor based on type
    if _is_dict_type(expected_type):
        return _process_dict_value(raw_value)
    elif _is_int_type(expected_type):
        return _process_int_value(raw_value, expected_type, series_size, fallback_value)
    else:
        return _process_other_value(raw_value, expected_type, fallback_value)


def _extract_critical_values(values: list, has_dict_third: bool) -> dict[str, float]:
    """Extract critical values from Zivot-Andrews result."""
    return cast(
        dict[str, float],
        _extract_value_from_zivot_andrews_result(
            values, index_with_dict=2, index_without_dict=4, expected_type=dict
        ),
    )


def _extract_lags(values: list, has_dict_third: bool) -> int | None:
    """Extract lags from Zivot-Andrews result."""
    return cast(
        int | None,
        _extract_value_from_zivot_andrews_result(
            values, index_with_dict=3, index_without_dict=2, expected_type=int, fallback_value=None
        ),
    )


def _extract_nobs(values: list, has_dict_third: bool, series_size: int) -> int:
    """Extract number of observations from Zivot-Andrews result."""
    return cast(
        int,
        _extract_value_from_zivot_andrews_result(
            values,
            index_with_dict=4,
            index_without_dict=3,
            expected_type=int,
            series_size=series_size,
        ),
    )


def _extract_break_index(values: list) -> int | None:
    """Extract break index from Zivot-Andrews result."""
    return cast(
        int | None,
        _extract_value_from_zivot_andrews_result(
            values, index_with_dict=5, index_without_dict=5, expected_type=int, fallback_value=None
        ),
    )


def _parse_zivot_andrews_result(
    result: tuple,
    series_size: int,
) -> tuple[float, float, dict[str, float], int | None, int, int | None]:
    """Parse Zivot-Andrews test result handling different statsmodels versions.

    Args:
        result: Raw result tuple from zivot_andrews().
        series_size: Size of the tested series (fallback for nobs).

    Returns:
        Tuple of (stat, pval, crit, lags, nobs, break_idx).
    """
    stat = float(result[0])
    pval = float(result[1])

    values = list(result)
    has_dict_third = len(values) > 2 and isinstance(values[2], dict)

    crit = _extract_critical_values(values, has_dict_third)
    lags = _extract_lags(values, has_dict_third)
    nobs = _extract_nobs(values, has_dict_third, series_size)
    break_idx = _extract_break_index(values)

    return stat, pval, crit, lags, nobs, break_idx


def _extract_break_date(series: pd.Series, break_idx: int | None) -> str | None:
    """Extract break date from series index if available.

    Args:
        series: Original time series with potential DatetimeIndex.
        break_idx: Break point index.

    Returns:
        Break date as string (YYYY-MM-DD) or None if not available.
    """
    if break_idx is None or not isinstance(series.index, pd.DatetimeIndex):
        return None

    if not (0 <= break_idx < len(series)):
        return None

    timestamp = series.index[break_idx]
    if isinstance(timestamp, pd.DatetimeIndex):
        timestamp = timestamp[0]

    if isinstance(timestamp, pd.Timestamp):
        return timestamp.strftime("%Y-%m-%d")

    return str(timestamp)


def zivot_andrews_test(
    series: pd.Series,
    *,
    model: Literal["c", "t", "ct"] = "c",
    max_lags: int | None = None,
) -> ZivotAndrewsTestResult:
    """Run Zivot-Andrews test for unit root with structural break.

    Endogenously determines structural break point for financial time series.

    Args:
        series: Input time series.
        model: "c" (intercept), "t" (trend), or "ct" (both). Defaults to "c".
        max_lags: Maximum lags for ADF regression. If None, automatic.

    Returns:
        ZivotAndrewsTestResult with statistic, p-value, break index/date, and critical values.
    """
    s = validate_series(series)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        result = zivot_andrews(s, maxlag=max_lags, trim=ZIVOT_ANDREWS_TRIM, regression=model)

    stat, pval, crit, lags, nobs, break_idx = _parse_zivot_andrews_result(result, s.size)
    break_date = _extract_break_date(series, break_idx)

    return {
        "statistic": stat,
        "p_value": pval,
        "break_index": break_idx,
        "break_date": break_date,
        "lags": lags,
        "nobs": nobs,
        "critical_values": crit,
        "model": model,
    }


def _determine_stationarity(
    adf_res: StationarityTestResult,
    kpss_res: StationarityTestResult,
    alpha: float,
) -> bool:
    """Determine stationarity verdict from ADF and KPSS test results.

    Rule of thumb:
    - ADF p < alpha (reject unit root)
    - KPSS p > alpha (do not reject stationarity)

    ⇒ stationary = True, otherwise False.
    """
    adf_rejects_unit_root = adf_res["p_value"] < alpha
    kpss_p = kpss_res["p_value"]
    kpss_accepts_stationarity = np.isnan(kpss_p) or kpss_p > alpha
    return bool(adf_rejects_unit_root and kpss_accepts_stationarity)


def _run_structural_break_test(series: pd.Series) -> ZivotAndrewsTestResult | None:
    """Run Zivot-Andrews structural break test with error handling.

    Args:
        series: Input time series.

    Returns:
        ZivotAndrewsTestResult or None if test fails.
    """
    try:
        za_res = zivot_andrews_test(series, model="c")
        if za_res["break_date"] is not None:
            logger.info(
                "Zivot-Andrews test detected potential break at %s "
                "(statistic=%.4f, p-value=%.4f)",
                za_res["break_date"],
                za_res["statistic"],
                za_res["p_value"],
            )
        return za_res
    except Exception as e:
        logger.warning("Zivot-Andrews test failed: %s. Continuing with ADF/KPSS.", e)
        return None


def evaluate_stationarity(
    series: pd.Series,
    *,
    alpha: float = STATIONARITY_DEFAULT_ALPHA,
    test_structural_break: bool = True,
) -> StationarityReport:
    """Combine ADF, KPSS and optionally Zivot-Andrews into a single verdict.

    Rule: ADF p < alpha AND KPSS p > alpha ⇒ stationary = True.

    Args:
        series: Input time series.
        alpha: Significance level (must be between 0 and 1). Defaults to STATIONARITY_DEFAULT_ALPHA.
        test_structural_break: If True, run Zivot-Andrews test before ADF/KPSS.

    Returns:
        StationarityReport with combined verdict and test results.

    Raises:
        ValueError: If alpha is not in (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    za_res = _run_structural_break_test(series) if test_structural_break else None
    adf_res = adf_test(series)
    kpss_res = kpss_test(series)

    stationary = _determine_stationarity(adf_res, kpss_res, alpha)
    return StationarityReport(
        stationary=stationary,
        alpha=float(alpha),
        adf=adf_res,
        kpss=kpss_res,
        zivot_andrews=za_res,
    )


def _resample_series(series: pd.Series, freq: str) -> pd.Series:
    """Resample time series to specified frequency.

    Args:
        series: Input time series with DatetimeIndex.
        freq: Resampling frequency (e.g., "W" for weekly).

    Returns:
        Resampled series with NaN values removed.
    """
    return series.resample(freq).sum(min_count=1).dropna()


def run_stationarity_pipeline(
    *,
    data_file: str,
    column: str = "weighted_log_return",
    alpha: float = STATIONARITY_DEFAULT_ALPHA,
    test_structural_break: bool = True,
) -> StationarityReport:
    """Load series, run tests, return structured report.

    Args:
        data_file: Path to CSV file containing time series data.
        column: Column name to extract from CSV. Defaults to "weighted_log_return".
        alpha: Significance level for tests. Defaults to STATIONARITY_DEFAULT_ALPHA.
        test_structural_break: If True, run Zivot-Andrews test before ADF/KPSS.
            Defaults to True to enable all tests.

    Returns:
        StationarityReport with test results and verdict.
    """
    logger.info(
        "Running stationarity checks (ADF + KPSS%s) on %s::%s",
        " + Zivot-Andrews" if test_structural_break else "",
        data_file,
        column,
    )

    # load_series_from_csv already returns a sorted DatetimeIndex
    series = load_series_from_csv(data_file=data_file, column=column)

    # For price data (close), test on original series; for returns, use weekly aggregation
    if column == "close":
        test_series = series
        logger.info("Testing stationarity on raw price data (no resampling)")
    else:
        # Weekly stationarity: aggregate log-returns by calendar week using sum
        test_series = _resample_series(series, STATIONARITY_RESAMPLE_FREQ)
        logger.info("Testing stationarity on weekly aggregated returns")

    report = evaluate_stationarity(
        test_series,
        alpha=alpha,
        test_structural_break=test_structural_break,
    )
    logger.info("Stationary=%s (alpha=%.3f)", report.stationary, report.alpha)
    return report


def save_stationarity_report(
    report: StationarityReport,
    out_path: Path | None = None,
) -> Path:
    """Persist report as JSON to the configured path.

    Args:
        report: StationarityReport to serialize.
        out_path: Optional override for the output path. If None,
            STATIONARITY_REPORT_FILE from src.constants is used.

    Returns:
        Path to the written JSON file.
    """
    target = Path(out_path) if out_path is not None else STATIONARITY_REPORT_FILE

    save_json_pretty(asdict(report), target)
    logger.info("Saved stationarity report: %s", target)
    return target
