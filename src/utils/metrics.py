"""Statistical and financial metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]
from typing import cast


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - cdf) of chi-squared distribution."""
    return cast(float, stats.chi2.sf(x, df))


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from price series."""
    return cast(pd.Series, np.log(prices / prices.shift(1)))


def compute_residuals(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute residuals between true and predicted values."""
    return cast(pd.Series, y_true - y_pred)
