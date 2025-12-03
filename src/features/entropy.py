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

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.utils import get_logger
from src.features.entropy_core import (
    _rolling_apen,
    _rolling_sampen,
    _rolling_shannon_entropy,
)

logger = get_logger(__name__)

__all__ = [
    "compute_shannon_entropy",
    "compute_approximate_entropy",
    "compute_sample_entropy",
]


# =============================================================================
# SHANNON ENTROPY
# =============================================================================


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
