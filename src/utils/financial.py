"""Financial computation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from typing import cast


def compute_rolling_volume_scaling(
    volume: pd.Series,
    window: int = 21,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling volume scaling factors (BTC/USD context).

    For cryptocurrency trading data, volume scaling helps normalize trading activity
    relative to recent historical levels, which is important for volatility modeling
    and liquidity analysis.

    Args:
        volume: Volume series (in base currency units, e.g., BTC amount)
        window: Rolling window size (default 21 for ~1 trading day in crypto)
        min_periods: Minimum periods for rolling calculation

    Returns:
        Rolling volume scaling factors (volume relative to recent average)
    """
    if min_periods is None:
        min_periods = max(1, window // 3)

    # Compute rolling mean volume
    rolling_volume = volume.rolling(window=window, min_periods=min_periods).mean()

    # Avoid division by zero - use numpy.where for safe division
    safe_denominator = np.where(rolling_volume == 0, np.nan, rolling_volume)

    # Compute weights as volume / rolling_mean_volume
    weights = volume / safe_denominator

    # Fill NaN values with 1.0 (neutral weight)
    weights = weights.fillna(1.0)

    return cast(pd.Series, weights)
