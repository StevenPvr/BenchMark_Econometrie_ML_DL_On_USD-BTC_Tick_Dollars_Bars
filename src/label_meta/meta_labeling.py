"""Meta-labeling for two-stage classification.

This module implements the meta-labeling methodology from De Prado (2018).
Meta-labeling is a two-stage approach where:
    1. Primary model predicts direction (side: -1, 0, 1)
    2. Secondary model (meta-learner) predicts whether to act on the signal (0, 1)

The meta-label indicates whether the primary model's signal was correct:
    - Meta-label = 1: Primary signal direction matched actual return
    - Meta-label = 0: Primary signal direction was wrong

Reference:
    De Prado, M. L. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons. Chapter 3: Meta-Labeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_meta_labels",
    "get_meta_features",
    "compute_strategy_returns",
    "compute_sharpe_ratio",
    "compute_strategy_metrics",
]


def get_meta_labels(
    events: pd.DataFrame,
    primary_signal: pd.Series,
    min_ret: float = 0.0,
) -> pd.Series:
    """Generate binary meta-labels.

    Meta-labels indicate whether the primary model's directional signal
    was correct (1) or wrong (0). Only generates labels where the primary
    model gives a non-zero signal.

    Logic:
        - If primary_signal == 0: No trade, meta-label = NaN
        - If primary_signal == 1 (Long) and ret > min_ret: Meta-label = 1
        - If primary_signal == 1 (Long) and ret <= min_ret: Meta-label = 0
        - If primary_signal == -1 (Short) and ret < -min_ret: Meta-label = 1
        - If primary_signal == -1 (Short) and ret >= -min_ret: Meta-label = 0

    Args:
        events: DataFrame from get_triple_barrier_events with 'ret' column.
        primary_signal: Series of primary model predictions (-1, 0, 1).
                       Must be aligned with events (same index).
        min_ret: Minimum return threshold for considering a trade correct.

    Returns:
        Series of binary meta-labels (0 or 1), NaN where primary_signal == 0.
    """
    # Ensure alignment
    if not events.index.equals(primary_signal.index):
        common_idx = events.index.intersection(primary_signal.index)
        events = events.loc[common_idx]
        primary_signal = primary_signal.loc[common_idx]
        logger.warning(
            "Index mismatch: using %d common observations", len(common_idx)
        )

    # Get returns from events
    returns = np.asarray(events["ret"].values)
    signals = np.asarray(primary_signal.values)

    # Initialize meta-labels as NaN
    meta_labels = np.full(len(signals), np.nan)

    # Long signals: correct if return > min_ret
    long_mask = signals == 1
    meta_labels[long_mask] = np.where(
        returns[long_mask] > min_ret, 1.0, 0.0
    )

    # Short signals: correct if return < -min_ret
    short_mask = signals == -1
    meta_labels[short_mask] = np.where(
        returns[short_mask] < -min_ret, 1.0, 0.0
    )

    result = pd.Series(meta_labels, index=events.index, name="meta_label")

    # Log statistics
    n_trades = np.sum(~np.isnan(meta_labels))
    n_correct = np.nansum(meta_labels)
    n_wrong = n_trades - n_correct

    logger.info(
        "Meta-labels generated: %d trades (%.1f%% correct, %.1f%% wrong)",
        n_trades,
        100 * n_correct / n_trades if n_trades > 0 else 0,
        100 * n_wrong / n_trades if n_trades > 0 else 0,
    )

    return result


def get_meta_features(
    X: pd.DataFrame,
    primary_signal: pd.Series,
    primary_proba: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create meta-features for the secondary model.

    Args:
        X: Original feature DataFrame.
        primary_signal: Primary model predictions (-1, 0, 1).
        primary_proba: Primary model probability estimates (optional).

    Returns:
        DataFrame with meta-features for training the secondary model.
    """
    result = X.copy()
    result["primary_signal"] = primary_signal.values

    if primary_proba is not None:
        for col in primary_proba.columns:
            result[f"primary_proba_{col}"] = primary_proba[col].values
        result["primary_confidence"] = primary_proba.max(axis=1).values

    return result


def compute_strategy_returns(
    events: pd.DataFrame,
    primary_signal: pd.Series,
    meta_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute strategy returns based on primary and meta signals.

    Args:
        events: DataFrame with 'ret' column from triple-barrier events.
        primary_signal: Primary model predictions (-1, 0, 1).
        meta_signal: Meta-model predictions (0, 1). If None, computes primary-only.

    Returns:
        DataFrame with strategy returns and performance metrics.
    """
    common_idx = events.index
    if primary_signal.index is not events.index:
        common_idx = common_idx.intersection(primary_signal.index)
    if meta_signal is not None and meta_signal.index is not events.index:
        common_idx = common_idx.intersection(meta_signal.index)

    returns = events.loc[common_idx, "ret"].values
    signals = primary_signal.loc[common_idx].values

    primary_returns = returns * signals

    if meta_signal is not None:
        meta_signals = meta_signal.loc[common_idx].values
        meta_signals = np.nan_to_num(meta_signals, nan=0.0)
        meta_returns = returns * signals * meta_signals
    else:
        meta_returns = np.full_like(primary_returns, np.nan)

    result = pd.DataFrame(
        {
            "return": returns,
            "signal": signals,
            "primary_return": primary_returns,
            "meta_return": meta_returns,
        },
        index=common_idx,
    )

    result["primary_cum_return"] = np.exp(np.nancumsum(result["primary_return"])) - 1
    if meta_signal is not None:
        result["meta_cum_return"] = np.exp(np.nancumsum(result["meta_return"])) - 1

    return result


def compute_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    annualization_factor: float = 1.0,
) -> float:
    """Compute Sharpe ratio from returns.

    Args:
        returns: Array of returns (can contain NaN).
        annualization_factor: Factor for annualization (default 1.0).

    Returns:
        Sharpe ratio (0.0 if insufficient data or zero std).
    """
    clean_returns = np.asarray(returns)
    clean_returns = clean_returns[~np.isnan(clean_returns)]

    if len(clean_returns) < 2:
        return 0.0

    mean_ret = np.mean(clean_returns)
    std_ret = np.std(clean_returns, ddof=1)

    if std_ret < 1e-10:
        return 0.0

    sharpe = (mean_ret / std_ret) * np.sqrt(annualization_factor)
    return float(sharpe)


def compute_strategy_metrics(
    strategy_returns: pd.DataFrame,
) -> dict[str, float]:
    """Compute performance metrics for strategy returns.

    Args:
        strategy_returns: DataFrame from compute_strategy_returns.

    Returns:
        Dictionary of performance metrics.
    """
    metrics = {}

    primary_returns = pd.Series(strategy_returns["primary_return"].dropna())
    metrics["primary_total_return"] = np.exp(primary_returns.sum()) - 1
    metrics["primary_sharpe"] = compute_sharpe_ratio(primary_returns)
    metrics["primary_n_trades"] = len(primary_returns)
    metrics["primary_win_rate"] = (primary_returns > 0).mean()
    metrics["primary_mean_return"] = primary_returns.mean()

    if "meta_return" in strategy_returns.columns:
        meta_returns = pd.Series(strategy_returns["meta_return"].dropna())
        if len(meta_returns) > 0:
            metrics["meta_total_return"] = np.exp(meta_returns.sum()) - 1
            metrics["meta_sharpe"] = compute_sharpe_ratio(meta_returns)
            metrics["meta_n_trades"] = len(meta_returns[meta_returns != 0])
            metrics["meta_win_rate"] = (meta_returns[meta_returns != 0] > 0).mean()
            metrics["meta_mean_return"] = meta_returns[meta_returns != 0].mean()

    return metrics
