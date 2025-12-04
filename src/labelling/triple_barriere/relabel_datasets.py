"""
Relabel datasets with triple-barrier labels using Optuna optimization.

Behaviour:
- Uses Optuna TPE sampler for efficient hyperparameter search.
- tp = RISK_REWARD_RATIO * sl (configurable ratio).
- Checks class proportion constraints for each configuration.
- Maximizes Sharpe ratio as objective.
- Applies the best labels to all datasets (tree/linear/lstm).
"""

from __future__ import annotations


import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any, Dict, List, Tuple, cast

import numpy as np
import optuna
import pandas as pd

from src.config_logging import get_logger
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LABELING_RISK_REWARD_RATIO,
    TRIPLE_BARRIER_RELABEL_MIN_CLASS_RATIO,
    TRIPLE_BARRIER_RELABEL_MAX_CLASS_RATIO,
    TRIPLE_BARRIER_RELABEL_INITIAL_CAPITAL,
    TRIPLE_BARRIER_RELABEL_TOTAL_COST_PCT,
    TRIPLE_BARRIER_RELABEL_N_TRIALS,
    TRIPLE_BARRIER_RELABEL_VOL_SPAN,
    TRIPLE_BARRIER_RELABEL_DATASET_FRACTION,
    TRIPLE_BARRIER_RELABEL_SL_MULT_RANGE,
    TRIPLE_BARRIER_RELABEL_MIN_RETURN_RANGE,
    TRIPLE_BARRIER_RELABEL_MAX_HOLDING_RANGE,
)
from src.labelling.triple_barriere.fast_barriers import get_events_primary_fast
from src.labelling.label_primaire.utils import (
    get_daily_volatility,
    load_dollar_bars,
)
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
)

logger = get_logger(__name__)

# Silence Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Configuration (from constants.py)
# ============================================================================

DATASET_PATHS: Dict[str, Path] = {
    "tree": DATASET_FEATURES_FINAL_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
}

SOURCE_DATASET = "linear"

# Optuna search space (from constants.py)
SEARCH_SPACE = {
    "sl_mult": TRIPLE_BARRIER_RELABEL_SL_MULT_RANGE,
    "min_return": TRIPLE_BARRIER_RELABEL_MIN_RETURN_RANGE,
    "max_holding": TRIPLE_BARRIER_RELABEL_MAX_HOLDING_RANGE,
}

# Optuna settings
N_TRIALS = TRIPLE_BARRIER_RELABEL_N_TRIALS
STUDY_NAME = "triple_barrier_optimization"

# Volatility estimation
VOL_SPAN = TRIPLE_BARRIER_RELABEL_VOL_SPAN

# Class proportion constraints
MIN_CLASS_RATIO = TRIPLE_BARRIER_RELABEL_MIN_CLASS_RATIO
MAX_CLASS_RATIO = TRIPLE_BARRIER_RELABEL_MAX_CLASS_RATIO

# Dataset sampling
DATASET_FRACTION = TRIPLE_BARRIER_RELABEL_DATASET_FRACTION


# ============================================================================
# Data Loading
# ============================================================================


def load_features(path: Path) -> pd.DataFrame:
    """Load features parquet and set datetime index."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)
    if "datetime_close" in df.columns:
        df = df.set_index("datetime_close")
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df.loc[~df.index.duplicated(keep="first")]
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature column names (exclude metadata and target)."""
    non_feature_cols = {
        "bar_id",
        "timestamp_open",
        "timestamp_close",
        "datetime_open",
        "datetime_close",
        "threshold_used",
        "log_return",
        "split",
        "label",
    }
    return [c for c in df.columns if c not in non_feature_cols]


def prepare_data(
    vol_span: int = VOL_SPAN,
    fraction: float = DATASET_FRACTION,
    use_train_only: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare source dataset with close prices and volatility.

    Args:
        vol_span: Span for volatility estimation.
        fraction: Fraction of dataset to use (0.0 to 1.0). Takes the last N% of data.
        use_train_only: If True, filter to train split only (avoids data leakage).
    """
    features_df = load_features(DATASET_PATHS[SOURCE_DATASET])

    # Filter to train split only to avoid data leakage during grid search
    if use_train_only and "split" in features_df.columns:
        n_before = len(features_df)
        features_df = features_df[features_df["split"] == "train"].copy()
        logger.info("Filtered to train split: %d -> %d samples", n_before, len(features_df))

    # Sample dataset (take last fraction to preserve time series continuity)
    if 0.0 < fraction < 1.0:
        n_samples = int(len(features_df) * fraction)
        features_df = features_df.iloc[-n_samples:]
        logger.info("Using %.0f%% of dataset: %d samples", fraction * 100, n_samples)

    bars = load_dollar_bars()
    close_raw = cast(pd.Series, bars["close"])

    # Align close to feature window
    close = close_raw.loc[
        (close_raw.index >= features_df.index[0])
        & (close_raw.index <= features_df.index[-1])
    ]
    if close.index.has_duplicates:
        close = close.loc[~close.index.duplicated(keep="first")]

    volatility = get_daily_volatility(close, span=vol_span)

    return cast(pd.DataFrame, features_df), close, volatility


# ============================================================================
# Labelling (Numba-optimized)
# ============================================================================


def compute_labels(
    features_df: pd.DataFrame,
    close: pd.Series,
    volatility: pd.Series,
    sl_mult: float,
    min_return: float,
    max_holding: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute triple-barrier labels using Numba-optimized functions.

    Args:
        features_df: Features DataFrame.
        close: Close price series.
        volatility: Volatility series.
        sl_mult: Stop-loss multiplier.
        min_return: Minimum return threshold.
        max_holding: Maximum holding period.

    Returns:
        Tuple of (labeled_df, events_df).
    """
    t_events = pd.DatetimeIndex(features_df.index)
    pt_mult = LABELING_RISK_REWARD_RATIO * sl_mult

    # Use Numba-optimized function (parallelization is handled internally)
    events = get_events_primary_fast(
        close=close,
        t_events=t_events,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        trgt=volatility,
        max_holding=max_holding,
        min_return=min_return,
    )

    # Build labeled DataFrame
    if events.empty:
        labeled_df = features_df.copy()
        labeled_df["label"] = np.nan
        return labeled_df, events

    common_idx = events.index.intersection(features_df.index)
    labels = events.loc[common_idx, "label"]

    labeled_df = features_df.copy()
    labeled_df["label"] = np.nan
    labeled_df.loc[common_idx, "label"] = labels

    return labeled_df, events


# ============================================================================
# Validation
# ============================================================================


def check_class_proportions(df: pd.DataFrame) -> bool:
    """Check if class proportions are within acceptable bounds."""
    if "label" not in df.columns:
        return False

    labels = df["label"].dropna()
    if labels.empty:
        return False

    counts = labels.value_counts()
    total = len(labels)

    required_classes = [-1, 0, 1]
    for cls in required_classes:
        if cls not in counts:
            return False
        proportion = counts[cls] / total
        if not (MIN_CLASS_RATIO <= proportion <= MAX_CLASS_RATIO):
            return False

    return True


def compute_label_stats(labels: pd.Series) -> Dict[str, float]:
    """Compute label distribution statistics."""
    counts = labels.value_counts(dropna=True).to_dict()
    total = sum(counts.values())
    pct = {k: (v / total * 100) if total > 0 else 0.0 for k, v in counts.items()}
    return {
        "total": total,
        "count_-1": counts.get(-1, 0),
        "count_0": counts.get(0, 0),
        "count_1": counts.get(1, 0),
        "pct_-1": pct.get(-1, 0.0),
        "pct_0": pct.get(0, 0.0),
        "pct_1": pct.get(1, 0.0),
    }


# ============================================================================
# Trading Costs Configuration (from constants.py)
# ============================================================================

INITIAL_CAPITAL: float = TRIPLE_BARRIER_RELABEL_INITIAL_CAPITAL
TOTAL_COST_PCT: float = TRIPLE_BARRIER_RELABEL_TOTAL_COST_PCT


# ============================================================================
# Economic Metrics Scoring
# ============================================================================


def compute_economic_metrics(
    events: pd.DataFrame,
    close: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    cost_pct: float = TOTAL_COST_PCT,
) -> Dict[str, float]:
    """Compute economic metrics from triple-barrier events with realistic costs.

    Uses simple returns. Cost is applied as a percentage of each trade's value.

    Args:
        events: DataFrame with 't1' (exit time), 'label', and 'ret' columns.
        close: Close price series.
        initial_capital: Starting portfolio value (default 10,000€).
        cost_pct: Cost as percentage of trade value (default 1%).

    Returns:
        Dict with sharpe_per_trade, total_return_pct, net_pnl, n_trades, win_rate.
    """
    # Filter valid trades (label != 0, has return)
    trades = events[events["label"] != 0].copy()
    if len(trades) < 10:
        return {
            "sharpe_per_trade": 0.0,
            "total_return_pct": 0.0,
            "net_pnl": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
        }

    # Compute gross returns: label * actual return
    gross_returns = np.asarray(trades["label"] * trades["ret"])

    # Net returns: subtract cost from absolute return
    # If gross return is +1%, cost is 0.01 * 1% = 0.01%, net = +0.99%
    # If gross return is -1%, cost is 0.01 * 1% = 0.01%, net = -1.01%
    net_returns = gross_returns - cost_pct * np.abs(gross_returns)

    n_trades = len(net_returns)

    # Sharpe ratio per trade: mean / std * sqrt(n_trades)
    mean_return = float(np.mean(net_returns))
    std_return = float(np.std(net_returns, ddof=1))
    if std_return > 0:
        sharpe_per_trade = (mean_return / std_return) * np.sqrt(n_trades)
    else:
        sharpe_per_trade = 0.0

    # Win rate (net profitable trades)
    n_profitable = int(np.sum(net_returns > 0))
    win_rate = n_profitable / n_trades if n_trades > 0 else 0.0

    # Total return: sum of returns
    total_return = float(np.sum(net_returns))
    # Convert to percentage
    total_return_pct = total_return * 100

    # Net P&L in euros (approximate for display)
    net_pnl = initial_capital * total_return

    return {
        "sharpe_per_trade": float(sharpe_per_trade),
        "total_return_pct": float(total_return_pct),
        "net_pnl": float(net_pnl),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
    }



# ============================================================================
# Optuna Optimization
# ============================================================================


class TripleBarrierOptimizer:
    """Optuna-based optimizer for triple-barrier labelling parameters."""

    def __init__(
        self,
        search_space: Dict[str, Tuple[float, float]] = SEARCH_SPACE,
        n_trials: int = N_TRIALS,
        vol_span: int = VOL_SPAN,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.vol_span = vol_span

        self.best_params_: Dict[str, Any] | None = None
        self.best_score_: float = float("-inf")
        self.best_metrics_: Dict[str, float] | None = None
        self.study_: optuna.Study | None = None

    def _create_objective(
        self,
        features_df: pd.DataFrame,
        close: pd.Series,
        volatility: pd.Series,
    ) -> Any:
        """Create the Optuna objective function."""

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            sl_mult = trial.suggest_float(
                "sl_mult",
                self.search_space["sl_mult"][0],
                self.search_space["sl_mult"][1],
            )
            min_return = trial.suggest_float(
                "min_return",
                self.search_space["min_return"][0],
                self.search_space["min_return"][1],
            )
            max_holding = trial.suggest_int(
                "max_holding",
                int(self.search_space["max_holding"][0]),
                int(self.search_space["max_holding"][1]),
            )

            # Compute labels and events
            try:
                labeled_df, events = compute_labels(
                    features_df=features_df,
                    close=close,
                    volatility=volatility,
                    sl_mult=sl_mult,
                    min_return=min_return,
                    max_holding=max_holding,
                )
            except Exception as e:
                logger.debug("Trial failed: %s", e)
                return float("-inf")

            # Check class proportions
            if not check_class_proportions(labeled_df):
                label_stats = compute_label_stats(cast(pd.Series, labeled_df["label"]))
                trial.set_user_attr("skip_reason", "class_proportions")
                trial.set_user_attr("label_stats", label_stats)
                return float("-inf")

            # Compute economic metrics (with costs)
            econ_metrics = compute_economic_metrics(events, close)
            # Maximize Sharpe ratio per trade
            score = float(econ_metrics["sharpe_per_trade"])

            # Store metrics in trial
            trial.set_user_attr("sharpe_per_trade", econ_metrics["sharpe_per_trade"])
            trial.set_user_attr("total_return_pct", econ_metrics["total_return_pct"])
            trial.set_user_attr("net_pnl", econ_metrics["net_pnl"])
            trial.set_user_attr("n_trades", econ_metrics["n_trades"])
            trial.set_user_attr("win_rate", econ_metrics["win_rate"])
            label_stats = compute_label_stats(cast(pd.Series, labeled_df["label"]))
            trial.set_user_attr("label_stats", label_stats)

            return score

        return objective

    def fit(
        self,
        features_df: pd.DataFrame,
        close: pd.Series,
        volatility: pd.Series,
    ) -> "TripleBarrierOptimizer":
        """Run Optuna optimization."""
        logger.info("=" * 70)
        logger.info("Starting Optuna optimization with %d trials", self.n_trials)
        logger.info("Search space:")
        for param, (low, high) in self.search_space.items():
            logger.info("  %s: [%.6f, %.6f]", param, low, high)
        logger.info("=" * 70)

        # Create study with TPE sampler
        self.study_ = optuna.create_study(
            direction="maximize",
            study_name=STUDY_NAME,
            sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE),
        )

        # Create objective
        objective = self._create_objective(features_df, close, volatility)

        # Run optimization with progress callback
        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            callbacks=[self._log_trial_callback],
        )

        # Extract best results
        if self.study_.best_trial is not None:
            self.best_params_ = self.study_.best_params.copy()
            self.best_score_ = self.study_.best_value
            self.best_metrics_ = {
                "sharpe_per_trade": self.study_.best_trial.user_attrs.get("sharpe_per_trade", 0.0),
                "total_return_pct": self.study_.best_trial.user_attrs.get("total_return_pct", 0.0),
                "net_pnl": self.study_.best_trial.user_attrs.get("net_pnl", 0.0),
                "n_trades": self.study_.best_trial.user_attrs.get("n_trades", 0),
                "win_rate": self.study_.best_trial.user_attrs.get("win_rate", 0.0),
            }

        self._log_summary()
        return self

    def _log_trial_callback(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """Callback to log trial results."""
        if trial.value is not None and trial.value > float("-inf"):
            is_best = trial.number == study.best_trial.number
            marker = " *** BEST ***" if is_best else ""
            sharpe = trial.user_attrs.get("sharpe_per_trade", 0.0)
            total_return_pct = trial.user_attrs.get("total_return_pct", 0.0)
            n_trades = trial.user_attrs.get("n_trades", 0)
            win_rate = trial.user_attrs.get("win_rate", 0.0)
            # Use f-string for formatting
            logger.info(
                f"Trial {trial.number:3d}: sl={trial.params['sl_mult']:.3f} "
                f"min_ret={trial.params['min_return']:.5f} hold={trial.params['max_holding']:2d} | "
                f"sharpe={sharpe:.2f} ret={total_return_pct:+.1f}% wr={win_rate * 100:.1f}% n={n_trades:,d}{marker}"
            )

    def _log_summary(self) -> None:
        """Log summary of optimization results."""
        if self.study_ is None:
            return

        n_complete = len([t for t in self.study_.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_valid = len([t for t in self.study_.trials if t.value is not None and t.value > float("-inf")])

        logger.info("=" * 70)
        logger.info("Optuna Optimization Complete")
        logger.info("  Total trials: %d", len(self.study_.trials))
        logger.info("  Completed trials: %d", n_complete)
        logger.info("  Valid trials: %d", n_valid)

        if self.best_params_ is not None and self.best_metrics_ is not None:
            logger.info("  Best parameters:")
            logger.info("    sl_mult: %.4f", self.best_params_["sl_mult"])
            logger.info("    min_return: %.6f", self.best_params_["min_return"])
            logger.info("    max_holding: %d", self.best_params_["max_holding"])
            logger.info("  Best metrics (with 1%% transaction costs):")
            logger.info("    Sharpe per trade: %.2f", self.best_metrics_["sharpe_per_trade"])
            logger.info("    Total return: %+.1f%%", self.best_metrics_["total_return_pct"])
            logger.info("    Win rate: %.1f%%", self.best_metrics_["win_rate"] * 100)
            logger.info("    N trades: %d", self.best_metrics_["n_trades"])
        else:
            logger.warning("  No valid parameter combination found!")
        logger.info("=" * 70)

    def get_results_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame for analysis."""
        if self.study_ is None:
            return pd.DataFrame()

        rows = []
        for trial in self.study_.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {
                "trial": trial.number,
                "sl_mult": trial.params.get("sl_mult"),
                "min_return": trial.params.get("min_return"),
                "max_holding": trial.params.get("max_holding"),
                "sharpe_per_trade": trial.user_attrs.get("sharpe_per_trade"),
                "total_return_pct": trial.user_attrs.get("total_return_pct"),
                "win_rate": trial.user_attrs.get("win_rate"),
                "n_trades": trial.user_attrs.get("n_trades"),
                "valid": trial.value is not None and trial.value > float("-inf"),
            }
            label_stats = trial.user_attrs.get("label_stats", {})
            row.update({f"label_{k}": v for k, v in label_stats.items()})
            rows.append(row)
        return pd.DataFrame(rows)


# Backward compatibility alias
TripleBarrierGridSearch = TripleBarrierOptimizer


# ============================================================================
# Apply Labels
# ============================================================================


def apply_labels_to_datasets(
    best_params: Dict[str, Any],
    best_metrics: Dict[str, float],
) -> None:
    """Apply best labels to all dataset variants by recomputing on full data.

    This function recomputes labels for the ENTIRE dataset (not just the
    GridSearch sample) using the optimal parameters found during search.
    """
    logger.info("=" * 70)
    logger.info("Applying optimal parameters to FULL datasets")
    logger.info("  sl_mult: %.2f", best_params["sl_mult"])
    logger.info("  min_return: %.5f", best_params["min_return"])
    logger.info("  max_holding: %d", best_params["max_holding"])
    logger.info("=" * 70)

    # Load full data (100%) - use_train_only=False to apply labels to all splits
    features_df, close, volatility = prepare_data(
        vol_span=VOL_SPAN,
        fraction=1.0,  # Use 100% of data
        use_train_only=False,  # Apply labels to train + val + test
    )

    # Compute labels for full dataset
    logger.info("Computing labels for full dataset (%d samples)...", len(features_df))
    labeled_df, events = compute_labels(
        features_df=features_df,
        close=close,
        volatility=volatility,
        sl_mult=best_params["sl_mult"],
        min_return=best_params["min_return"],
        max_holding=best_params["max_holding"],
    )

    # Compute final metrics
    full_metrics = compute_economic_metrics(events, close)
    full_stats = compute_label_stats(cast(pd.Series, labeled_df["label"]))
    logger.info(
        "Full dataset: return=%+.1f%%, n_trades=%d | -1=%.1f%%, 0=%.1f%%, +1=%.1f%%",
        full_metrics["total_return_pct"],
        full_metrics["n_trades"],
        full_stats["pct_-1"],
        full_stats["pct_0"],
        full_stats["pct_1"],
    )

    # Also add t1 column for purging in opti.py
    if "t1" in events.columns:
        # Convert to timezone-naive datetime to avoid dtype incompatibility warnings
        t1_values = events["t1"].dt.tz_localize(None) if events["t1"].dt.tz is not None else events["t1"]
        labeled_df["t1"] = pd.Series(pd.NaT, index=labeled_df.index, dtype="datetime64[ns]")
        common_idx = events.index.intersection(labeled_df.index)
        labeled_df.loc[common_idx, "t1"] = t1_values.loc[common_idx]

    # Apply to all datasets
    for ds_name, path in DATASET_PATHS.items():
        df_target = load_features(path)
        df_with_labels = df_target.copy()

        # Add label column
        df_with_labels["label"] = np.nan
        common_idx = labeled_df.index.intersection(df_target.index)
        df_with_labels.loc[common_idx, "label"] = labeled_df.loc[common_idx, "label"]

        # Add t1 column for purging
        if "t1" in labeled_df.columns:
            df_with_labels["t1"] = pd.Series(pd.NaT, index=df_with_labels.index, dtype="datetime64[ns]")
            df_with_labels.loc[common_idx, "t1"] = labeled_df.loc[common_idx, "t1"]

        # Save
        df_with_labels.to_parquet(path)

        stats = compute_label_stats(cast(pd.Series, df_with_labels["label"]))
        n_labeled = int(stats["total"])
        n_total = len(df_with_labels)
        pct_labeled = n_labeled / n_total * 100 if n_total > 0 else 0
        logger.info(
            "[%s] Labeled %d/%d (%.1f%%) | -1=%.1f%%, 0=%.1f%%, +1=%.1f%% | %s",
            ds_name,
            n_labeled,
            n_total,
            pct_labeled,
            stats["pct_-1"],
            stats["pct_0"],
            stats["pct_1"],
            path.name,
        )

