"""Entropy-based features for market regime detection.

This module implements entropy measures to quantify market uncertainty
and predictability:

1. Shannon Entropy (on discretized returns):
    H = -Σ p_i * log(p_i)

2. Approximate Entropy (ApEn): Measures the probability that similar patterns
   remain similar at the next step. Higher ApEn = more randomness.

3. Sample Entropy (SampEn): Improved version of ApEn, less biased on small
   samples and more consistent.

Interpretation:
    - High entropy: Random, unpredictable market
    - Low entropy: Structured, predictable patterns
    - Entropy drop: Often precedes breakouts (market "deciding")

Reference:
    Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
    PNAS, 88(6), 2297-2301.

    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.config_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_shannon_entropy",
    "compute_approximate_entropy",
    "compute_sample_entropy",
]


# =============================================================================
# SHANNON ENTROPY
# =============================================================================


@njit(cache=True)
def _compute_shannon_entropy(counts: NDArray[np.int64], n: int) -> float:
    """Compute Shannon entropy from bin counts (numba optimized).

    H = -Σ p_i * log(p_i)

    Args:
        counts: Array of bin counts.
        n: Total number of observations.

    Returns:
        Shannon entropy in nats (natural log).
    """
    if n == 0:
        return np.nan

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            entropy -= p * np.log(p)

    return entropy


@njit(cache=True)
def _discretize_returns(
    returns: NDArray[np.float64],
    n_bins: int,
) -> NDArray[np.int64]:
    """Discretize returns into bins (numba optimized).

    Args:
        returns: Array of returns.
        n_bins: Number of bins for discretization.

    Returns:
        Array of bin indices.
    """
    # Remove NaN for min/max calculation
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) == 0:
        return np.zeros(len(returns), dtype=np.int64)

    min_val = np.min(valid_returns)
    max_val = np.max(valid_returns)

    if max_val == min_val:
        return np.zeros(len(returns), dtype=np.int64)

    bin_width = (max_val - min_val) / n_bins
    bins = np.zeros(len(returns), dtype=np.int64)

    for i in range(len(returns)):
        if np.isnan(returns[i]):
            bins[i] = -1  # Mark NaN
        else:
            bin_idx = int((returns[i] - min_val) / bin_width)
            bins[i] = min(bin_idx, n_bins - 1)

    return bins


@njit(cache=True)
def _rolling_shannon_entropy(
    returns: NDArray[np.float64],
    window: int,
    n_bins: int,
) -> NDArray[np.float64]:
    """Compute rolling Shannon entropy (numba optimized).

    Args:
        returns: Array of returns.
        window: Rolling window size.
        n_bins: Number of bins for discretization.

    Returns:
        Array of entropy values.
    """
    n = len(returns)
    entropy = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return entropy

    for i in range(window - 1, n):
        # Get window data
        window_data = returns[i - window + 1 : i + 1]

        # Count non-NaN values
        valid_count = 0
        for val in window_data:
            if not np.isnan(val):
                valid_count += 1

        if valid_count < window // 2:
            continue

        # Discretize window
        bins = _discretize_returns(window_data, n_bins)

        # Count occurrences in each bin
        counts = np.zeros(n_bins, dtype=np.int64)
        for b in bins:
            if 0 <= b < n_bins:
                counts[b] += 1

        entropy[i] = _compute_shannon_entropy(counts, valid_count)

    return entropy


def compute_shannon_entropy(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    window: int = 50,
    n_bins: int = 10,
) -> pd.Series:
    """Compute rolling Shannon entropy of discretized returns.

    Shannon entropy measures the uncertainty/randomness in the return
    distribution over a rolling window:

        H = -Σ p_i * log(p_i)

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        window: Rolling window size (default: 50).
        n_bins: Number of bins for discretization (default: 10).

    Returns:
        Series with Shannon entropy values (in nats).

    Example:
        >>> df_bars["entropy"] = compute_shannon_entropy(df_bars, window=50)
    """
    returns = df_bars[return_col].values.astype(np.float64)
    entropy = _rolling_shannon_entropy(returns, window, n_bins)

    # Log statistics
    valid = entropy[~np.isnan(entropy)]
    if len(valid) > 0:
        # Max entropy for uniform distribution: log(n_bins)
        max_entropy = np.log(n_bins)
        normalized_mean = np.mean(valid) / max_entropy
        logger.info(
            "Shannon entropy (window=%d, bins=%d) stats: mean=%.4f (%.1f%% of max), "
            "std=%.4f, min=%.4f, max=%.4f",
            window,
            n_bins,
            np.mean(valid),
            100 * normalized_mean,
            np.std(valid),
            np.min(valid),
            np.max(valid),
        )

    return pd.Series(entropy, index=df_bars.index, name=f"shannon_entropy_{window}")


# =============================================================================
# APPROXIMATE ENTROPY (ApEn)
# =============================================================================


@njit(cache=True)
def _count_similar_patterns(
    data: NDArray[np.float64],
    m: int,
    r: float,
) -> float:
    """Count similar patterns for ApEn/SampEn calculation (numba optimized).

    Args:
        data: Input time series.
        m: Embedding dimension (pattern length).
        r: Tolerance (similarity threshold).

    Returns:
        Phi value (log of average pattern similarity).
    """
    n = len(data)
    if n < m:
        return np.nan

    n_patterns = n - m + 1
    if n_patterns <= 0:
        return np.nan

    count_sum = 0.0

    for i in range(n_patterns):
        count = 0
        for j in range(n_patterns):
            # Check if patterns are similar (Chebyshev distance <= r)
            similar = True
            for k in range(m):
                if abs(data[i + k] - data[j + k]) > r:
                    similar = False
                    break
            if similar:
                count += 1

        # Include self-match for ApEn
        count_sum += np.log(count / n_patterns)

    return count_sum / n_patterns


@njit(cache=True)
def _compute_apen_single(
    data: NDArray[np.float64],
    m: int,
    r: float,
) -> float:
    """Compute Approximate Entropy for a single window (numba optimized).

    ApEn(m, r) = Phi_m(r) - Phi_{m+1}(r)

    Args:
        data: Input time series (should be normalized).
        m: Embedding dimension.
        r: Tolerance threshold.

    Returns:
        Approximate entropy value.
    """
    phi_m = _count_similar_patterns(data, m, r)
    phi_m1 = _count_similar_patterns(data, m + 1, r)

    if np.isnan(phi_m) or np.isnan(phi_m1):
        return np.nan

    return phi_m - phi_m1


@njit(cache=True)
def _rolling_apen(
    returns: NDArray[np.float64],
    window: int,
    m: int,
    r_mult: float,
) -> NDArray[np.float64]:
    """Compute rolling Approximate Entropy (numba optimized).

    Args:
        returns: Array of returns.
        window: Rolling window size.
        m: Embedding dimension.
        r_mult: Tolerance multiplier (r = r_mult after normalization).

    Returns:
        Array of ApEn values.
    """
    n = len(returns)
    apen = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return apen

    for i in range(window - 1, n):
        # Get window data
        window_data = returns[i - window + 1 : i + 1]

        # Remove NaN
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) < window // 2:
            continue

        # Normalize data (std becomes 1)
        std = np.std(valid_data)
        if std < 1e-10:
            # No variance = perfectly regular = 0 entropy
            apen[i] = 0.0
            continue

        normalized = (valid_data - np.mean(valid_data)) / std
        # After normalization, r is just r_mult (since std=1)
        r = r_mult

        apen[i] = _compute_apen_single(normalized, m, r)

    return apen


def compute_approximate_entropy(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    window: int = 50,
    m: int = 2,
    r_mult: float = 0.2,
) -> pd.Series:
    """Compute rolling Approximate Entropy (ApEn).

    ApEn measures the probability that similar patterns of length m remain
    similar when extended to length m+1. Lower ApEn indicates more regular,
    predictable patterns.

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        window: Rolling window size (default: 50).
        m: Embedding dimension (default: 2).
        r_mult: Tolerance multiplier (default: 0.2, r = 0.2 * std).

    Returns:
        Series with ApEn values.

    Example:
        >>> df_bars["apen"] = compute_approximate_entropy(df_bars, window=50)
    """
    returns = df_bars[return_col].values.astype(np.float64)
    apen = _rolling_apen(returns, window, m, r_mult)

    # Log statistics
    valid = apen[~np.isnan(apen)]
    if len(valid) > 0:
        logger.info(
            "Approximate entropy (window=%d, m=%d) stats: mean=%.4f, "
            "std=%.4f, min=%.4f, max=%.4f",
            window,
            m,
            np.mean(valid),
            np.std(valid),
            np.min(valid),
            np.max(valid),
        )

    return pd.Series(apen, index=df_bars.index, name=f"apen_{window}")


# =============================================================================
# SAMPLE ENTROPY (SampEn)
# =============================================================================


@njit(cache=True)
def _count_matches_sampen(
    data: NDArray[np.float64],
    m: int,
    r: float,
) -> tuple[int, int]:
    """Count matches for SampEn calculation (numba optimized).

    Unlike ApEn, SampEn excludes self-matches.

    Args:
        data: Input time series.
        m: Embedding dimension.
        r: Tolerance threshold.

    Returns:
        Tuple of (B_count for m, A_count for m+1).
    """
    n = len(data)
    if n < m + 1:
        return 0, 0

    B = 0  # Count of similar patterns of length m
    A = 0  # Count of similar patterns of length m+1

    n_patterns = n - m

    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            # Check m-length pattern similarity
            similar_m = True
            for k in range(m):
                if abs(data[i + k] - data[j + k]) > r:
                    similar_m = False
                    break

            if similar_m:
                B += 1

                # Check (m+1)-length pattern similarity
                if abs(data[i + m] - data[j + m]) <= r:
                    A += 1

    return B, A


@njit(cache=True)
def _compute_sampen_single(
    data: NDArray[np.float64],
    m: int,
    r: float,
) -> float:
    """Compute Sample Entropy for a single window (numba optimized).

    SampEn = -log(A / B)

    Where A = matches of length m+1, B = matches of length m.

    Args:
        data: Input time series (should be normalized).
        m: Embedding dimension.
        r: Tolerance threshold.

    Returns:
        Sample entropy value.
    """
    B, A = _count_matches_sampen(data, m, r)

    if B == 0:
        # No similar patterns found = maximum entropy (fully random)
        return 5.0
    if A == 0:
        # No matches of length m+1 found - cap at reasonable max
        # This indicates maximum unpredictability
        # Cap value ~5 corresponds to A/B ratio of ~0.007 (very low similarity)
        return 5.0

    return -np.log(A / B)


@njit(cache=True)
def _rolling_sampen(
    returns: NDArray[np.float64],
    window: int,
    m: int,
    r_mult: float,
) -> NDArray[np.float64]:
    """Compute rolling Sample Entropy (numba optimized).

    Args:
        returns: Array of returns.
        window: Rolling window size.
        m: Embedding dimension.
        r_mult: Tolerance multiplier (r = r_mult after normalization).

    Returns:
        Array of SampEn values.
    """
    n = len(returns)
    sampen = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return sampen

    for i in range(window - 1, n):
        # Get window data
        window_data = returns[i - window + 1 : i + 1]

        # Remove NaN
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) < window // 2:
            continue

        # Normalize data (std becomes 1)
        std = np.std(valid_data)
        if std < 1e-10:
            # No variance = perfectly regular = 0 entropy
            sampen[i] = 0.0
            continue

        normalized = (valid_data - np.mean(valid_data)) / std
        # After normalization, r is just r_mult (since std=1)
        r = r_mult

        sampen[i] = _compute_sampen_single(normalized, m, r)

    return sampen


def compute_sample_entropy(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    window: int = 50,
    m: int = 2,
    r_mult: float = 0.2,
) -> pd.Series:
    """Compute rolling Sample Entropy (SampEn).

    SampEn is an improved version of ApEn that:
    - Excludes self-matches (less biased)
    - Is more consistent across different data lengths
    - Is better suited for small samples

    Interpretation:
        - Low SampEn: Regular, predictable patterns
        - High SampEn: Random, unpredictable
        - SampEn drop: May precede breakout (market "deciding")

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        window: Rolling window size (default: 50).
        m: Embedding dimension (default: 2).
        r_mult: Tolerance multiplier (default: 0.2, r = 0.2 * std).

    Returns:
        Series with SampEn values.

    Example:
        >>> df_bars["sampen"] = compute_sample_entropy(df_bars, window=50)
    """
    returns = df_bars[return_col].values.astype(np.float64)
    sampen = _rolling_sampen(returns, window, m, r_mult)

    # Log statistics
    valid = sampen[~np.isnan(sampen)]
    if len(valid) > 0:
        logger.info(
            "Sample entropy (window=%d, m=%d) stats: mean=%.4f, "
            "std=%.4f, min=%.4f, max=%.4f",
            window,
            m,
            np.mean(valid),
            np.std(valid),
            np.min(valid),
            np.max(valid),
        )

    return pd.Series(sampen, index=df_bars.index, name=f"sampen_{window}")
