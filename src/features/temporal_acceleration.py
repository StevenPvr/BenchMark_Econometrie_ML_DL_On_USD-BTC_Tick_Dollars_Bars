"""Temporal Acceleration feature computation.

Temporal acceleration measures changes in the sampling rate of dollar bars.
Since dollar bars are sampled by volume rather than time, the duration between
bars varies with market activity. Acceleration captures the dynamics of this
variation:

    a_i = τ_{i-1} - τ_i  (raw acceleration)
    a_i = EMA(τ, k)_{i-1} - EMA(τ, k)_i  (smoothed acceleration)

Where τ_i is the duration (in seconds) of bar i.

Interpretation:
    - a_i > 0: Acceleration (bars forming faster, higher activity)
    - a_i < 0: Deceleration (bars forming slower, lower activity)

The second derivative (jerk) captures inflection points where the market
transitions between acceleration and deceleration regimes.

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 2: Financial Data Structures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_temporal_acceleration",
    "compute_temporal_acceleration_smoothed",
    "compute_temporal_jerk",
]


@njit(cache=True)
def _compute_ema(
    values: NDArray[np.float64],
    span: int,
) -> NDArray[np.float64]:
    """Compute Exponential Moving Average (numba optimized).

    EMA_t = α * x_t + (1 - α) * EMA_{t-1}
    where α = 2 / (span + 1)

    Args:
        values: Input array.
        span: EMA span (window size for decay).

    Returns:
        Array of EMA values.
    """
    n = len(values)
    ema = np.empty(n, dtype=np.float64)

    if n == 0:
        return ema

    alpha = 2.0 / (span + 1.0)
    ema[0] = values[0]

    for i in range(1, n):
        if np.isnan(values[i]):
            ema[i] = ema[i - 1]
        else:
            ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]

    return ema


@njit(cache=True)
def _compute_acceleration(durations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute raw temporal acceleration (numba optimized).

    a_i = τ_{i-1} - τ_i

    Args:
        durations: Array of bar durations (in seconds).

    Returns:
        Array of acceleration values (NaN for first element).
    """
    n = len(durations)
    acceleration = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        acceleration[i] = durations[i - 1] - durations[i]

    return acceleration


@njit(cache=True)
def _compute_acceleration_smoothed(
    durations: NDArray[np.float64],
    span: int,
) -> NDArray[np.float64]:
    """Compute smoothed temporal acceleration using EMA (numba optimized).

    a_i = EMA(τ, k)_{i-1} - EMA(τ, k)_i

    Args:
        durations: Array of bar durations (in seconds).
        span: EMA span for smoothing.

    Returns:
        Array of smoothed acceleration values.
    """
    ema = _compute_ema(durations, span)

    n = len(durations)
    acceleration = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        acceleration[i] = ema[i - 1] - ema[i]

    return acceleration


@njit(cache=True)
def _compute_jerk(acceleration: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute temporal jerk (second derivative) (numba optimized).

    j_i = a_i - a_{i-1} = (τ_{i-2} - τ_{i-1}) - (τ_{i-1} - τ_i)
                       = τ_{i-2} - 2*τ_{i-1} + τ_i

    Jerk captures inflection points where acceleration changes direction.

    Args:
        acceleration: Array of acceleration values.

    Returns:
        Array of jerk values (NaN for first two elements).
    """
    n = len(acceleration)
    jerk = np.full(n, np.nan, dtype=np.float64)

    for i in range(2, n):
        if not np.isnan(acceleration[i]) and not np.isnan(acceleration[i - 1]):
            jerk[i] = acceleration[i] - acceleration[i - 1]

    return jerk


def compute_temporal_acceleration(
    df_bars: pd.DataFrame,
    duration_col: str = "duration_sec",
) -> pd.Series:
    """Compute raw temporal acceleration.

    Raw acceleration measures the change in bar duration:
        a_i = τ_{i-1} - τ_i

    Args:
        df_bars: DataFrame with dollar bars containing duration column.
        duration_col: Name of duration column (in seconds).

    Returns:
        Series with acceleration values (positive = faster bars).

    Example:
        >>> df_bars["acceleration"] = compute_temporal_acceleration(df_bars)
    """
    durations = df_bars[duration_col].values.astype(np.float64)
    acceleration = _compute_acceleration(durations)

    # Log statistics
    valid = acceleration[~np.isnan(acceleration)]
    if len(valid) > 0:
        pct_accelerating = 100 * np.mean(valid > 0)
        logger.info(
            "Temporal acceleration stats: mean=%.2f, std=%.2f, "
            "accelerating=%.1f%%",
            np.mean(valid),
            np.std(valid),
            pct_accelerating,
        )

    return pd.Series(acceleration, index=df_bars.index, name="temporal_acceleration")


def compute_temporal_acceleration_smoothed(
    df_bars: pd.DataFrame,
    duration_col: str = "duration_sec",
    span: int = 20,
) -> pd.Series:
    """Compute smoothed temporal acceleration using EMA.

    Smoothed acceleration reduces noise by applying EMA before differencing:
        a_i = EMA(τ, k)_{i-1} - EMA(τ, k)_i

    Args:
        df_bars: DataFrame with dollar bars containing duration column.
        duration_col: Name of duration column (in seconds).
        span: EMA span for smoothing (default: 20).

    Returns:
        Series with smoothed acceleration values.

    Example:
        >>> df_bars["acceleration_smooth"] = compute_temporal_acceleration_smoothed(
        ...     df_bars, span=20
        ... )
    """
    durations = df_bars[duration_col].values.astype(np.float64)
    acceleration = _compute_acceleration_smoothed(durations, span)

    # Log statistics
    valid = acceleration[~np.isnan(acceleration)]
    if len(valid) > 0:
        pct_accelerating = 100 * np.mean(valid > 0)
        logger.info(
            "Smoothed acceleration (span=%d) stats: mean=%.2f, std=%.2f, "
            "accelerating=%.1f%%",
            span,
            np.mean(valid),
            np.std(valid),
            pct_accelerating,
        )

    return pd.Series(
        acceleration, index=df_bars.index, name=f"temporal_acceleration_ema{span}"
    )


def compute_temporal_jerk(
    df_bars: pd.DataFrame,
    duration_col: str = "duration_sec",
    smoothed: bool = True,
    span: int = 20,
) -> pd.Series:
    """Compute temporal jerk (second derivative of duration).

    Jerk is the rate of change of acceleration, capturing inflection points
    where the market transitions between acceleration and deceleration:
        j_i = a_i - a_{i-1}

    Interpretation:
        - j > 0: Acceleration is increasing (speeding up faster)
        - j < 0: Acceleration is decreasing (slowing down faster)
        - j ≈ 0 after sign change: Inflection point

    Args:
        df_bars: DataFrame with dollar bars containing duration column.
        duration_col: Name of duration column (in seconds).
        smoothed: If True, use smoothed acceleration (default: True).
        span: EMA span if using smoothed acceleration.

    Returns:
        Series with jerk values.

    Example:
        >>> df_bars["jerk"] = compute_temporal_jerk(df_bars, smoothed=True, span=20)
    """
    durations = df_bars[duration_col].values.astype(np.float64)

    if smoothed:
        acceleration = _compute_acceleration_smoothed(durations, span)
    else:
        acceleration = _compute_acceleration(durations)

    jerk = _compute_jerk(acceleration)

    # Log statistics
    valid = jerk[~np.isnan(jerk)]
    if len(valid) > 0:
        logger.info(
            "Temporal jerk stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            np.mean(valid),
            np.std(valid),
            np.min(valid),
            np.max(valid),
        )

    name = f"temporal_jerk_ema{span}" if smoothed else "temporal_jerk"
    return pd.Series(jerk, index=df_bars.index, name=name)
