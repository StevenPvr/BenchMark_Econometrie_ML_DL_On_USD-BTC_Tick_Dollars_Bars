"""Numba-optimized core functions for entropy-based features.

This module contains the low-level Numba functions for computing
entropy measures. These are used by entropy.py.

NOTE: Numba functions use literal values equivalent to EPS (1e-10)
because Numba cannot import Python constants at compile time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "_compute_shannon_entropy",
    "_discretize_returns",
    "_rolling_shannon_entropy",
    "_count_similar_patterns",
    "_compute_apen_single",
    "_rolling_apen",
    "_count_matches_sampen",
    "_compute_sampen_single",
    "_rolling_sampen",
]


# =============================================================================
# SHANNON ENTROPY (Numba optimized)
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


# =============================================================================
# APPROXIMATE ENTROPY (ApEn) - Numba optimized
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
        # Use literal value 1e-10 (equivalent to EPS) - Numba cannot import Python constants
        if std < 1e-10:
            # No variance = perfectly regular = 0 entropy
            apen[i] = 0.0
            continue

        normalized = (valid_data - np.mean(valid_data)) / std
        # After normalization, r is just r_mult (since std=1)
        r = r_mult

        apen[i] = _compute_apen_single(normalized, m, r)

    return apen


# =============================================================================
# SAMPLE ENTROPY (SampEn) - Numba optimized
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
    max_entropy: float,
) -> float:
    """Compute Sample Entropy for a single window (numba optimized).

    SampEn = -log(A / B)

    Where A = matches of length m+1, B = matches of length m.

    Args:
        data: Input time series (should be normalized).
        m: Embedding dimension.
        r: Tolerance threshold.
        max_entropy: Maximum entropy value (adaptive, based on window size).

    Returns:
        Sample entropy value.
    """
    B, A = _count_matches_sampen(data, m, r)

    if B == 0:
        # No similar patterns found = maximum entropy (fully random)
        # Use adaptive max based on window size: log(n)
        return max_entropy
    if A == 0:
        # No matches of length m+1 found - cap at adaptive max
        # This indicates maximum unpredictability
        return max_entropy

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
        # Use literal value 1e-10 (equivalent to EPS) - Numba cannot import Python constants
        if std < 1e-10:
            # No variance = perfectly regular = 0 entropy
            sampen[i] = 0.0
            continue

        normalized = (valid_data - np.mean(valid_data)) / std
        # After normalization, r is just r_mult (since std=1)
        r = r_mult

        # Adaptive max entropy based on effective window size
        max_entropy = np.log(float(len(valid_data)))

        sampen[i] = _compute_sampen_single(normalized, m, r, max_entropy)

    return sampen
