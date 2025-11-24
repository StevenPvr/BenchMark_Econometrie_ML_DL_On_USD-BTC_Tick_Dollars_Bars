"""Utility functions for GARCH numerical tests.

Contains helper functions used across the module for:
- Statistical computations (chi-square, Ljung-Box, ARCH-LM)
- Data preparation and validation
- Output formatting
- Test result validation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_NUMERICAL_TEST_NAME_ENGLE_ARCH_LM,
    GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_RESIDUALS,
    GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_SQUARED,
    GARCH_NUMERICAL_TEST_NAME_MCLEOD_LI,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Statistical computation utilities
# ============================================================================


def prepare_lags_list(lags: int | list[int]) -> list[int]:
    """Convert lags to a list of integers.

    Args:
        lags: Single lag or list of lags.

    Returns:
        List of lag integers.

    Raises:
        TypeError: If lags is not int or list.
    """
    if isinstance(lags, int):
        return list(range(1, lags + 1))
    if isinstance(lags, list):
        return sorted(set(int(lag) for lag in lags if lag > 0))
    msg = f"lags must be int or list[int], got {type(lags)}"
    raise TypeError(msg)


def compute_ljung_box_stats(
    res: np.ndarray, lags_list: list[int]
) -> tuple[list[float], list[float]]:
    """Compute Ljung-Box statistics using statsmodels.

    Args:
        res: Residual series (already cleaned).
        lags_list: List of lag values to test.

    Returns:
        Tuple of (lb_stats, lb_pvalues) lists.

    Raises:
        ImportError: If statsmodels is not available.
        RuntimeError: If computation fails.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore
    except ImportError as e:
        msg = f"statsmodels.stats.diagnostic.acorr_ljungbox is required but not available: {e}"
        raise ImportError(msg) from e

    try:
        df = acorr_ljungbox(res, lags=lags_list, return_df=True)
        return [float(x) for x in df["lb_stat"]], [float(x) for x in df["lb_pvalue"]]
    except Exception as e:
        msg = f"Failed to compute Ljung-Box statistics: {e}"
        raise RuntimeError(msg) from e


def compute_arch_lm_statistic(e2: np.ndarray, lags: int) -> float:
    """Compute ARCH-LM test statistic.

    Args:
        e2: Squared residuals.
        lags: Number of lags in regression.

    Returns:
        LM test statistic value.

    Raises:
        ValueError: If computation fails or inputs are invalid.
    """
    n = int(e2.size)
    Y = e2[lags:]
    X = np.ones((n - lags, lags + 1), dtype=float)
    for j in range(1, lags + 1):
        X[:, j] = e2[lags - j : n - j]

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        ss_res = float(np.sum((Y - Y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return (n - lags) * max(0.0, r2)
    except Exception as e:
        msg = f"Failed to compute ARCH-LM statistic: {e}"
        raise ValueError(msg) from e


# ============================================================================
# Output formatting utilities
# ============================================================================


def prepare_output(results: dict[str, Any], n_residuals: int) -> dict[str, Any]:
    """Prepare output dictionary for test results.

    Args:
        results: Dictionary containing test results.
        n_residuals: Number of residuals tested.

    Returns:
        Formatted output dictionary.
    """
    return {
        "source": str(GARCH_DATASET_FILE),
        "n_residuals": n_residuals,
        "tests": {
            "ljung_box_residuals": {
                "name": GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_RESIDUALS,
                "result": results["ljung_box_residuals"],
            },
            "ljung_box_squared": {
                "name": GARCH_NUMERICAL_TEST_NAME_LJUNG_BOX_SQUARED,
                "result": results["ljung_box_squared"],
            },
            "engle_arch_lm": {
                "name": GARCH_NUMERICAL_TEST_NAME_ENGLE_ARCH_LM,
                "result": results["engle_arch_lm"],
            },
            "mcleod_li": {
                "name": GARCH_NUMERICAL_TEST_NAME_MCLEOD_LI,
                "result": results["mcleod_li"],
            },
        },
    }


def log_test_summary(results: dict[str, Any]) -> None:
    """Log summary of test results.

    Args:
        results: Dictionary containing test results.
    """
    logger.info("Test Results Summary:")
    logger.info("  Ljung-Box (residuals): reject=%s", results["ljung_box_residuals"]["reject"])
    logger.info("  Ljung-Box (squared): reject=%s", results["ljung_box_squared"]["reject"])
    logger.info("  Engle ARCH-LM: reject=%s", results["engle_arch_lm"]["reject"])
    logger.info("  McLeod-Li: reject=%s", results["mcleod_li"]["reject"])


# ============================================================================
# Test validation utilities
# ============================================================================


def assert_keys_present(result: dict[str, Any], required_keys: list[str]) -> None:
    """Assert that all required keys are present in result.

    Args:
        result: Test result dictionary.
        required_keys: List of required key names.

    Raises:
        ValueError: If a required key is missing.
    """
    for key in required_keys:
        if key not in result:
            msg = f"Required key '{key}' is missing from result"
            raise ValueError(msg)


def validate_ljung_box_result(result: dict[str, Any], expected_lags: int) -> None:
    """Validate Ljung-Box test result structure.

    Args:
        result: Test result dictionary.
        expected_lags: Expected number of lags.

    Raises:
        ValueError: If result structure is invalid.
    """
    assert_keys_present(result, ["lb_stat", "lb_pvalue", "reject"])
    if len(result["lb_stat"]) != expected_lags:
        msg = f"Expected {expected_lags} lb_stat values, got {len(result['lb_stat'])}"
        raise ValueError(msg)
    if len(result["lb_pvalue"]) != expected_lags:
        msg = f"Expected {expected_lags} lb_pvalue values, got {len(result['lb_pvalue'])}"
        raise ValueError(msg)


def validate_arch_lm_result(result: dict[str, Any], expected_df: int) -> None:
    """Validate ARCH-LM test result structure.

    Args:
        result: Test result dictionary.
        expected_df: Expected degrees of freedom.

    Raises:
        ValueError: If result structure is invalid.
    """
    assert_keys_present(result, ["lm_stat", "p_value", "df", "reject"])
    if result["df"] != expected_df:
        msg = f"Expected df={expected_df}, got df={result['df']}"
        raise ValueError(msg)


def validate_all_tests_result(results: dict[str, Any]) -> None:
    """Validate that all test results are present and valid.

    Args:
        results: Dictionary containing all test results.

    Raises:
        ValueError: If results structure is invalid.
    """
    required_keys = ["ljung_box_residuals", "ljung_box_squared", "engle_arch_lm", "mcleod_li"]
    for key in required_keys:
        if key not in results:
            msg = f"Required test '{key}' is missing from results"
            raise ValueError(msg)
        if not isinstance(results[key], dict):
            msg = f"Test result '{key}' must be a dict, got {type(results[key])}"
            raise ValueError(msg)


__all__ = [
    # Statistical computation utilities
    "prepare_lags_list",
    "compute_ljung_box_stats",
    "compute_arch_lm_statistic",
    # Output formatting utilities
    "prepare_output",
    "log_test_summary",
    # Test validation utilities
    "assert_keys_present",
    "validate_ljung_box_result",
    "validate_arch_lm_result",
    "validate_all_tests_result",
]
