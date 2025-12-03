"""Jump detection features using Bipower Variation.

This module computes features for detecting price jumps (discontinuities)
in the return series:

1. Bipower Variation (BV):
   BV_t^{(k)} = (π/2) * Σ_{j=1}^{k-1} |r_{t-j}| * |r_{t-j-1}|

   Bipower variation is robust to jumps and converges to integrated variance
   in the presence of jumps (unlike realized variance which captures jumps).

2. Jump Component:
   J_t^{(k)} = max(RV_t^{(k)} - BV_t^{(k)}, 0)

   The difference between realized variance and bipower variation isolates
   the jump component of price variation.

3. Jump Ratio:
   JR_t^{(k)} = J_t^{(k)} / RV_t^{(k)}

   Proportion of total variance attributable to jumps.

Interpretation:
    - High BV: High continuous (diffusion) volatility
    - High Jump: Significant price discontinuities detected
    - High JR: Jumps dominate price movements

Reference:
    Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and Bipower
    Variation with Stochastic Volatility and Jumps. Journal of Financial
    Econometrics, 2(1), 1-37.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

from src.constants import EPS, JUMP_MIN_VARIANCE
from src.utils import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

__all__ = [
    "compute_bipower_variation",
    "compute_jump_component",
    "compute_all_jump_features",
]

# NOTE: Numba functions use literal values equivalent to EPS (1e-10) and
# JUMP_MIN_VARIANCE (1e-15) because Numba cannot import Python constants.


@njit(cache=True)
def _rolling_bipower(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Bipower Variation (numba optimized).

    BV = (π/2) * Σ |r_t| * |r_{t-1}|

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of bipower variation values.
    """
    n = len(returns)
    bv = np.full(n, np.nan, dtype=np.float64)

    if n < window + 1:
        return bv

    # Correction factor: π/2 ≈ 1.5708
    mu1 = np.sqrt(2.0 / np.pi)  # E[|Z|] for Z ~ N(0,1)
    correction = 1.0 / (mu1 ** 2)  # = π/2

    for i in range(window, n):
        # Compute bipower sum over window
        bp_sum = 0.0
        valid_pairs = 0

        for j in range(1, window):
            idx = i - window + j + 1
            r_curr = returns[idx]
            r_prev = returns[idx - 1]

            if not np.isnan(r_curr) and not np.isnan(r_prev):
                bp_sum += np.abs(r_curr) * np.abs(r_prev)
                valid_pairs += 1

        if valid_pairs >= window // 2:
            # Scale by correction factor
            bv[i] = correction * bp_sum

    return bv


@njit(cache=True)
def _rolling_realized_variance(
    returns: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Compute rolling Realized Variance (numba optimized).

    RV = Σ r_t²

    Args:
        returns: Array of returns.
        window: Rolling window size.

    Returns:
        Array of realized variance values.
    """
    n = len(returns)
    rv = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return rv

    for i in range(window - 1, n):
        rv_sum = 0.0
        valid_count = 0

        for j in range(window):
            r = returns[i - j]
            if not np.isnan(r):
                rv_sum += r * r
                valid_count += 1

        if valid_count >= window:
            rv[i] = rv_sum

    return rv


def compute_bipower_variation(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Bipower Variation over multiple horizons.

    BV_t^{(k)} = (π/2) * Σ_{j=1}^{k-1} |r_{t-j}| * |r_{t-j-1}|

    Bipower variation is robust to jumps - it estimates integrated variance
    from the continuous (diffusion) component of price movements.

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [10, 20, 50]).

    Returns:
        DataFrame with bipower variation columns for each horizon.

    Example:
        >>> df_bv = compute_bipower_variation(df_bars)
        >>> df_bars = pd.concat([df_bars, df_bv], axis=1)
    """
    if horizons is None:
        horizons = [10, 20, 50]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        bv = _rolling_bipower(returns, k)
        col_name = f"bipower_var_{k}"
        result[col_name] = bv

        # Log statistics
        valid = bv[~np.isnan(bv)]
        if len(valid) > 0:
            logger.info(
                "Bipower Variation (k=%d) stats: mean=%.6f, std=%.6f, "
                "min=%.6f, max=%.6f",
                k,
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
            )

    return result


def compute_jump_component(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Jump Component and Jump Ratio over multiple horizons.

    Jump_t^{(k)} = max(RV_t^{(k)} - BV_t^{(k)}, 0)
    JumpRatio_t^{(k)} = Jump_t^{(k)} / RV_t^{(k)}

    The jump component isolates the discontinuous (jump) part of price
    variation by subtracting the continuous component (bipower variation)
    from total variation (realized variance).

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [10, 20, 50]).

    Returns:
        DataFrame with jump and jump ratio columns for each horizon.

    Example:
        >>> df_jump = compute_jump_component(df_bars)
        >>> df_bars = pd.concat([df_bars, df_jump], axis=1)
    """
    if horizons is None:
        horizons = [10, 20, 50]

    returns = df_bars[return_col].values.astype(np.float64)
    result = pd.DataFrame(index=df_bars.index)

    for k in horizons:
        # Compute realized variance and bipower variation
        rv = _rolling_realized_variance(returns, k)
        bv = _rolling_bipower(returns, k)

        # Jump component (non-negative)
        jump = np.maximum(rv - bv, 0.0)

        # Jump ratio (proportion of variance from jumps)
        jump_ratio = np.where(
            rv > 1e-15,
            jump / rv,
            0.0,
        )

        result[f"jump_{k}"] = jump
        result[f"jump_ratio_{k}"] = jump_ratio

        # Log statistics
        valid_jump = jump[~np.isnan(jump)]
        valid_ratio = jump_ratio[~np.isnan(jump_ratio)]
        if len(valid_jump) > 0:
            logger.info(
                "Jump component (k=%d) stats: mean=%.6f, std=%.6f, "
                "pct_nonzero=%.1f%%",
                k,
                np.mean(valid_jump),
                np.std(valid_jump),
                100 * np.mean(valid_jump > 1e-10),
            )
            logger.info(
                "Jump ratio (k=%d) stats: mean=%.4f, std=%.4f, "
                "max=%.4f",
                k,
                np.mean(valid_ratio),
                np.std(valid_ratio),
                np.max(valid_ratio),
            )

    return result


def compute_all_jump_features(
    df_bars: pd.DataFrame,
    return_col: str = "log_return",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute all jump detection features (convenience function).

    Computes:
    - Bipower variation
    - Jump component
    - Jump ratio

    Args:
        df_bars: DataFrame with return data.
        return_col: Name of return column.
        horizons: List of horizons (default: [10, 20, 50]).

    Returns:
        DataFrame with all jump-related features.

    Example:
        >>> df_jumps = compute_all_jump_features(df_bars)
        >>> df_bars = pd.concat([df_bars, df_jumps], axis=1)
    """
    if horizons is None:
        horizons = [10, 20, 50]

    logger.info("Computing jump detection features...")

    # Bipower variation
    df_bv = compute_bipower_variation(df_bars, return_col, horizons)

    # Jump component and ratio
    df_jump = compute_jump_component(df_bars, return_col, horizons)

    # Combine all features
    result = pd.concat([df_bv, df_jump], axis=1)

    logger.info("Jump features computed: %d columns", len(result.columns))

    return result
