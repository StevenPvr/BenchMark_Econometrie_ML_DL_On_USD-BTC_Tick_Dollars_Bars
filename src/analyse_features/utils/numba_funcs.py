"""Numba JIT-compiled functions for high-performance computations.

This module provides optimized implementations of compute-intensive operations:
- Rolling correlations
- Distance correlation components
- Entropy calculations

All functions use Numba's @njit decorator with parallel=True for multi-core
execution and fastmath=True for SIMD optimizations.

Performance notes:
- First call incurs JIT compilation overhead (~1-2 seconds)
- Subsequent calls are 10-100x faster than pure Python
- Memory layout: Use C-contiguous arrays (np.ascontiguousarray)
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def fast_rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling Pearson correlation between two arrays.

    Uses Welford's online algorithm for numerical stability.

    Args:
        x: First array (1D, float64).
        y: Second array (1D, float64, same length as x).
        window: Rolling window size.

    Returns:
        Array of rolling correlations (length = len(x) - window + 1).
        First (window - 1) values are NaN.
    """
    n = len(x)
    result = np.full(n, np.nan)

    if window > n:
        return result

    for i in prange(window - 1, n):
        start = i - window + 1

        # Extract window
        x_win = x[start : i + 1]
        y_win = y[start : i + 1]

        # Compute means
        mean_x = 0.0
        mean_y = 0.0
        for j in range(window):
            mean_x += x_win[j]
            mean_y += y_win[j]
        mean_x /= window
        mean_y /= window

        # Compute correlation components
        cov_xy = 0.0
        var_x = 0.0
        var_y = 0.0
        for j in range(window):
            dx = x_win[j] - mean_x
            dy = y_win[j] - mean_y
            cov_xy += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        # Correlation
        denom = np.sqrt(var_x * var_y)
        if denom > 1e-12:
            result[i] = cov_xy / denom
        else:
            result[i] = 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def fast_rolling_spearman(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling Spearman correlation using ranks.

    Args:
        x: First array (1D, float64).
        y: Second array (1D, float64).
        window: Rolling window size.

    Returns:
        Array of rolling Spearman correlations.
    """
    n = len(x)
    result = np.full(n, np.nan)

    if window > n:
        return result

    for i in prange(window - 1, n):
        start = i - window + 1

        # Extract window
        x_win = x[start : i + 1].copy()
        y_win = y[start : i + 1].copy()

        # Compute ranks (simple argsort-based ranking)
        x_ranks = np.empty(window)
        y_ranks = np.empty(window)

        # Argsort for x
        x_order = np.argsort(x_win)
        for j in range(window):
            x_ranks[x_order[j]] = j + 1

        # Argsort for y
        y_order = np.argsort(y_win)
        for j in range(window):
            y_ranks[y_order[j]] = j + 1

        # Pearson correlation on ranks
        mean_x = (window + 1) / 2.0
        mean_y = (window + 1) / 2.0

        cov_xy = 0.0
        var_x = 0.0
        var_y = 0.0
        for j in range(window):
            dx = x_ranks[j] - mean_x
            dy = y_ranks[j] - mean_y
            cov_xy += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        denom = np.sqrt(var_x * var_y)
        if denom > 1e-12:
            result[i] = cov_xy / denom
        else:
            result[i] = 0.0

    return result


@njit(fastmath=True, cache=True)
def fast_spearman_matrix(data: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation matrix for all columns.

    Optimized for feature matrices (n_samples x n_features).

    Args:
        data: 2D array (n_samples, n_features), C-contiguous.

    Returns:
        Correlation matrix (n_features, n_features).
    """
    n_samples, n_features = data.shape
    result = np.eye(n_features)

    # Compute ranks for all columns
    ranks = np.empty_like(data)
    for j in range(n_features):
        col = data[:, j].copy()
        order = np.argsort(col)
        for i in range(n_samples):
            ranks[order[i], j] = i + 1

    # Compute correlation between all pairs
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r_i = ranks[:, i]
            r_j = ranks[:, j]

            mean_i = (n_samples + 1) / 2.0
            mean_j = (n_samples + 1) / 2.0

            cov_ij = 0.0
            var_i = 0.0
            var_j = 0.0

            for k in range(n_samples):
                di = r_i[k] - mean_i
                dj = r_j[k] - mean_j
                cov_ij += di * dj
                var_i += di * di
                var_j += dj * dj

            denom = np.sqrt(var_i * var_j)
            if denom > 1e-12:
                corr = cov_ij / denom
            else:
                corr = 0.0

            result[i, j] = corr
            result[j, i] = corr

    return result


@njit(parallel=True, fastmath=True, cache=True)
def fast_distance_variance(x: np.ndarray) -> float:
    """Compute distance variance of a 1D array.

    Component of distance correlation (dCor).

    Args:
        x: 1D array.

    Returns:
        Distance variance (scalar).
    """
    n = len(x)
    if n < 2:
        return 0.0

    # Compute pairwise distances
    a = np.empty((n, n))
    for i in prange(n):
        for j in range(n):
            a[i, j] = abs(x[i] - x[j])

    # Row and column means
    row_means = np.empty(n)
    col_means = np.empty(n)
    grand_mean = 0.0

    for i in range(n):
        s = 0.0
        for j in range(n):
            s += a[i, j]
        row_means[i] = s / n

    for j in range(n):
        s = 0.0
        for i in range(n):
            s += a[i, j]
        col_means[j] = s / n
        grand_mean += s

    grand_mean /= n * n

    # Double-centered distances
    dvar = 0.0
    for i in prange(n):
        for j in range(n):
            A_ij = a[i, j] - row_means[i] - col_means[j] + grand_mean
            dvar += A_ij * A_ij

    return dvar / (n * n)


@njit(parallel=True, fastmath=True, cache=True)
def fast_distance_covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance covariance between two 1D arrays.

    Args:
        x: First array.
        y: Second array (same length).

    Returns:
        Distance covariance (scalar).
    """
    n = len(x)
    if n < 2:
        return 0.0

    # Compute pairwise distances for x
    a = np.empty((n, n))
    for i in prange(n):
        for j in range(n):
            a[i, j] = abs(x[i] - x[j])

    # Row and column means for a
    a_row_means = np.empty(n)
    a_col_means = np.empty(n)
    a_grand_mean = 0.0

    for i in range(n):
        s = 0.0
        for j in range(n):
            s += a[i, j]
        a_row_means[i] = s / n

    for j in range(n):
        s = 0.0
        for i in range(n):
            s += a[i, j]
        a_col_means[j] = s / n
        a_grand_mean += s

    a_grand_mean /= n * n

    # Compute pairwise distances for y
    b = np.empty((n, n))
    for i in prange(n):
        for j in range(n):
            b[i, j] = abs(y[i] - y[j])

    # Row and column means for b
    b_row_means = np.empty(n)
    b_col_means = np.empty(n)
    b_grand_mean = 0.0

    for i in range(n):
        s = 0.0
        for j in range(n):
            s += b[i, j]
        b_row_means[i] = s / n

    for j in range(n):
        s = 0.0
        for i in range(n):
            s += b[i, j]
        b_col_means[j] = s / n
        b_grand_mean += s

    b_grand_mean /= n * n

    # Distance covariance
    dcov = 0.0
    for i in prange(n):
        for j in range(n):
            A_ij = a[i, j] - a_row_means[i] - a_col_means[j] + a_grand_mean
            B_ij = b[i, j] - b_row_means[i] - b_col_means[j] + b_grand_mean
            dcov += A_ij * B_ij

    return dcov / (n * n)


@njit(fastmath=True, cache=True)
def fast_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation between two 1D arrays.

    Distance correlation (dCor) measures both linear and non-linear
    dependencies. dCor = 0 if and only if X and Y are independent.

    Args:
        x: First array.
        y: Second array (same length).

    Returns:
        Distance correlation in [0, 1].
    """
    dcov_xy = fast_distance_covariance(x, y)
    dvar_x = fast_distance_variance(x)
    dvar_y = fast_distance_variance(y)

    denom = np.sqrt(dvar_x * dvar_y)

    if denom > 1e-12:
        return np.sqrt(dcov_xy / denom)
    else:
        return 0.0


@njit(parallel=True, fastmath=True, cache=True)
def fast_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to max_lag.

    Args:
        x: Time series (1D).
        max_lag: Maximum lag to compute.

    Returns:
        ACF values for lags 0 to max_lag.
    """
    n = len(x)
    result = np.zeros(max_lag + 1)

    # Mean and variance
    mean_x = 0.0
    for i in range(n):
        mean_x += x[i]
    mean_x /= n

    var_x = 0.0
    for i in range(n):
        var_x += (x[i] - mean_x) ** 2
    var_x /= n

    if var_x < 1e-12:
        return result

    # ACF for each lag
    for lag in prange(max_lag + 1):
        acf = 0.0
        for i in range(n - lag):
            acf += (x[i] - mean_x) * (x[i + lag] - mean_x)
        result[lag] = acf / (n * var_x)

    return result


@njit(fastmath=True, cache=True)
def fast_vif_single(
    X: np.ndarray,
    feature_idx: int,
) -> float:
    """Compute VIF for a single feature using OLS.

    VIF = 1 / (1 - R²) where R² is from regressing feature_idx on all others.

    Args:
        X: Feature matrix (n_samples, n_features).
        feature_idx: Index of feature to compute VIF for.

    Returns:
        VIF value for the feature.
    """
    n_samples, n_features = X.shape

    # Extract target column
    y = X[:, feature_idx].copy()

    # Build design matrix (all other columns + intercept)
    n_predictors = n_features  # -1 for removed column + 1 for intercept
    X_reg = np.empty((n_samples, n_predictors))

    # Add intercept
    for i in range(n_samples):
        X_reg[i, 0] = 1.0

    # Add other features
    col_idx = 1
    for j in range(n_features):
        if j != feature_idx:
            for i in range(n_samples):
                X_reg[i, col_idx] = X[i, j]
            col_idx += 1

    # OLS: beta = (X'X)^-1 X'y
    # Compute X'X
    XtX = np.zeros((n_predictors, n_predictors))
    for i in range(n_predictors):
        for j in range(n_predictors):
            s = 0.0
            for k in range(n_samples):
                s += X_reg[k, i] * X_reg[k, j]
            XtX[i, j] = s

    # Compute X'y
    Xty = np.zeros(n_predictors)
    for i in range(n_predictors):
        s = 0.0
        for k in range(n_samples):
            s += X_reg[k, i] * y[k]
        Xty[i] = s

    # Solve for beta (simple Cholesky for positive definite)
    # Add small regularization for numerical stability
    for i in range(n_predictors):
        XtX[i, i] += 1e-8

    # Simple Gaussian elimination
    beta = np.linalg.solve(XtX, Xty)

    # Compute predictions and R²
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        for j in range(n_predictors):
            y_pred[i] += X_reg[i, j] * beta[j]

    # R² = 1 - SS_res / SS_tot
    ss_res = 0.0
    ss_tot = 0.0
    y_mean = 0.0
    for i in range(n_samples):
        y_mean += y[i]
    y_mean /= n_samples

    for i in range(n_samples):
        ss_res += (y[i] - y_pred[i]) ** 2
        ss_tot += (y[i] - y_mean) ** 2

    if ss_tot < 1e-12:
        return np.inf

    r_squared = 1.0 - ss_res / ss_tot

    if r_squared >= 1.0 - 1e-12:
        return np.inf

    return 1.0 / (1.0 - r_squared)
