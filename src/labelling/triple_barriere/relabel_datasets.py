"""
Relabel datasets with triple-barrier labels using GridSearch optimization.

Behaviour:
- Uses sklearn's ParameterGrid for systematic grid search over barrier parameters.
- tp = RISK_REWARD_RATIO * sl (configurable ratio).
- Checks class proportion constraints for each configuration.
- Runs walk-forward CV (logistic regression, balanced, F1-macro) to score valid configs.
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

import logging
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.labelling.triple_barriere.fast_barriers import get_events_primary_fast
from src.labelling.label_primaire.utils import (
    RISK_REWARD_RATIO,
    get_daily_volatility,
    load_dollar_bars,
)
from src.path import (
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
)

logger = logging.getLogger("relabel_gridsearch")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ============================================================================
# Configuration
# ============================================================================

DATASET_PATHS: Dict[str, Path] = {
    "tree": DATASET_FEATURES_FINAL_PARQUET,
    "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
}

SOURCE_DATASET = "linear"

# GridSearch parameter space (expanded thanks to Numba 130x speedup)
PARAM_GRID: dict[str, list[Any]] = {
    # Stop-loss multiplier (× volatility) - wider range with finer steps
    "sl_mult": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    # Minimum return threshold for neutral label
    "min_return": [0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005],
    # Maximum holding period in bars
    "max_holding": [15, 20, 25, 30, 40, 50],
}
# Total: 12 × 8 × 8 = 768 combinations (~6s with Numba vs ~13min without)

# Volatility estimation
VOL_SPAN = 100

# Class proportion constraints
MIN_CLASS_RATIO = 0.25
MAX_CLASS_RATIO = 0.45

# Dataset sampling (with Numba, we can use more data)
DATASET_FRACTION = 0.5  # Use full dataset (Numba makes this fast)

# Economic metrics weights for scoring (sum to 1.0)
METRIC_WEIGHTS = {
    "sharpe": 1.0,
}


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
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare source dataset with close prices and volatility.

    Args:
        vol_span: Span for volatility estimation.
        fraction: Fraction of dataset to use (0.0 to 1.0). Takes the last N% of data.
    """
    features_df = load_features(DATASET_PATHS[SOURCE_DATASET])

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

    return features_df, close, volatility


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
    pt_mult = RISK_REWARD_RATIO * sl_mult

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
# Economic Metrics Scoring
# ============================================================================


def compute_economic_metrics(
    events: pd.DataFrame,
    close: pd.Series,
) -> Dict[str, float]:
    """Compute economic metrics from triple-barrier events.

    Args:
        events: DataFrame with 't1' (exit time), 'label', and 'ret' columns.
        close: Close price series.

    Returns:
        Dict with sharpe, n_trades.
    """
    # Filter valid trades (label != 0, has return)
    trades = events[events["label"] != 0].copy()
    if len(trades) < 10:
        return {
            "sharpe": float("-inf"),
            "n_trades": 0,
        }

    # Compute returns: label * actual return
    # ret column contains the return achieved
    returns = trades["label"] * trades["ret"]

    # Sharpe Ratio (annualized, assuming ~252 trading days)
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(252)
    else:
        sharpe = 0.0


    return {"sharpe": float(sharpe), "n_trades": len(trades)}



# ============================================================================
# GridSearch
# ============================================================================


class TripleBarrierGridSearch:
    """GridSearch for triple-barrier labelling parameters using economic metrics."""

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        vol_span: int = VOL_SPAN,
    ):
        self.param_grid = param_grid
        self.vol_span = vol_span

        self.best_params_: Dict[str, Any] | None = None
        self.best_score_: float = float("-inf")
        self.best_metrics_: Dict[str, float] | None = None
        self.best_df_: pd.DataFrame | None = None
        self.results_: List[Dict[str, Any]] = []

    def fit(
        self,
        features_df: pd.DataFrame,
        close: pd.Series,
        volatility: pd.Series,
    ) -> "TripleBarrierGridSearch":
        """Run grid search over all parameter combinations."""
        grid = list(ParameterGrid(self.param_grid))
        n_combinations = len(grid)

        logger.info("Starting GridSearch with %d parameter combinations", n_combinations)

        pbar = tqdm(grid, desc="GridSearch", unit="combo", position=0)
        for params in pbar:
            sl_mult = params["sl_mult"]
            min_return = params["min_return"]
            max_holding = params["max_holding"]

            # Update progress bar
            pbar.set_postfix(
                sl=f"{sl_mult:.2f}",
                min_ret=f"{min_return:.5f}",
                hold=max_holding,
                best=f"{self.best_score_:.3f}" if self.best_score_ > float("-inf") else "N/A",
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
                tqdm.write(f"  [FAIL] sl={sl_mult:.2f} min_ret={min_return:.5f} hold={max_holding}: {e}")
                self._record_result(params, float("-inf"), {}, {}, valid=False)
                continue

            # Check class proportions
            label_stats = compute_label_stats(cast(pd.Series, labeled_df["label"]))
            if not check_class_proportions(labeled_df):
                tqdm.write(
                    f"  [SKIP] sl={sl_mult:.2f} min_ret={min_return:.5f} hold={max_holding} | "
                    f"props: -1={label_stats['pct_-1']:.1f}% 0={label_stats['pct_0']:.1f}% +1={label_stats['pct_1']:.1f}%"
                )
                self._record_result(params, float("-inf"), label_stats, {}, valid=False)
                continue

            # Compute economic metrics
            econ_metrics = compute_economic_metrics(events, close)
            score = float(econ_metrics["sharpe"])

            if score == float("-inf"):
                tqdm.write(
                    f"  [SKIP] sl={sl_mult:.2f} min_ret={min_return:.5f} hold={max_holding} | "
                    f"insufficient trades ({econ_metrics['n_trades']})"
                )
                self._record_result(params, score, label_stats, econ_metrics, valid=False)
                continue

            # Log valid result
            is_new_best = score > self.best_score_
            marker = " *** BEST ***" if is_new_best else ""
            tqdm.write(
                f"  [OK] sl={sl_mult:.2f} min_ret={min_return:.5f} hold={max_holding} | "
                f"score={score:.3f} sharpe={econ_metrics['sharpe']:.2f} "
                f"n={econ_metrics['n_trades']}{marker}"
            )

            self._record_result(params, score, label_stats, econ_metrics, valid=True)

            # Update best
            if is_new_best:
                self.best_score_ = score
                self.best_params_ = params.copy()
                self.best_metrics_ = econ_metrics.copy()
                self.best_df_ = labeled_df

        pbar.close()
        self._log_summary()
        return self

    def _record_result(
        self,
        params: Dict[str, Any],
        score: float,
        label_stats: Dict[str, float],
        econ_metrics: Dict[str, float],
        valid: bool,
    ) -> None:
        """Record result for this parameter combination."""
        self.results_.append(
            {
                "params": params.copy(),
                "score": score,
                "label_stats": label_stats.copy(),
                "econ_metrics": econ_metrics.copy(),
                "valid": valid,
            }
        )

    def _log_summary(self) -> None:
        """Log summary of grid search results."""
        valid_count = sum(1 for r in self.results_ if r["valid"])
        logger.info("=" * 70)
        logger.info("GridSearch Complete")
        logger.info("  Total combinations: %d", len(self.results_))
        logger.info("  Valid combinations: %d", valid_count)

        if self.best_params_ is not None and self.best_metrics_ is not None:
            logger.info("  Best parameters:")
            logger.info("    sl_mult: %.2f", self.best_params_["sl_mult"])
            logger.info("    min_return: %.5f", self.best_params_["min_return"])
            logger.info("    max_holding: %d", self.best_params_["max_holding"])
            logger.info("  Best economic metrics:")
            logger.info("    Composite score: %.4f", self.best_score_)
            logger.info("    Sharpe ratio: %.2f", self.best_metrics_["sharpe"])
            logger.info("    N trades: %d", self.best_metrics_["n_trades"])
        else:
            logger.warning("  No valid parameter combination found!")
        logger.info("=" * 70)

    def get_results_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame for analysis."""
        rows = []
        for r in self.results_:
            row = {
                "sl_mult": r["params"]["sl_mult"],
                "min_return": r["params"]["min_return"],
                "max_holding": r["params"]["max_holding"],
                "score": r["score"],
                "valid": r["valid"],
            }
            row.update({f"label_{k}": v for k, v in r["label_stats"].items()})
            row.update({f"econ_{k}": v for k, v in r["econ_metrics"].items()})
            rows.append(row)
        return pd.DataFrame(rows)


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

    # Load full data (100%)
    features_df, close, volatility = prepare_data(
        vol_span=VOL_SPAN,
        fraction=1.0,  # Use 100% of data
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
        "Full dataset: sharpe=%.2f, n_trades=%d | -1=%.1f%%, 0=%.1f%%, +1=%.1f%%",
        full_metrics["sharpe"],
        full_metrics["n_trades"],
        full_stats["pct_-1"],
        full_stats["pct_0"],
        full_stats["pct_1"],
    )

    # Also add t1 column for purging in opti.py
    if "t1" in events.columns:
        labeled_df["t1"] = np.nan
        common_idx = events.index.intersection(labeled_df.index)
        labeled_df.loc[common_idx, "t1"] = events.loc[common_idx, "t1"]

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
            df_with_labels["t1"] = np.nan
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


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run triple-barrier GridSearch and apply best labels."""
    logger.info("Loading source dataset: %s", SOURCE_DATASET)
    features_df, close, volatility = prepare_data(
        vol_span=VOL_SPAN,
        fraction=DATASET_FRACTION,
    )

    logger.info(
        "Dataset loaded: %d samples, %d features",
        len(features_df),
        len(get_feature_columns(features_df)),
    )

    # Run GridSearch
    gridsearch = TripleBarrierGridSearch(
        param_grid=PARAM_GRID,
        vol_span=VOL_SPAN,
    )
    gridsearch.fit(features_df, close, volatility)

    # Check results
    if gridsearch.best_params_ is None or gridsearch.best_metrics_ is None:
        logger.warning("No valid configuration found. Datasets not modified.")
        return

    # Apply best params to ALL data (recomputes labels for full dataset)
    apply_labels_to_datasets(
        best_params=gridsearch.best_params_,
        best_metrics=gridsearch.best_metrics_,
    )

    # Save results summary
    results_df = gridsearch.get_results_dataframe()
    results_path = Path(__file__).parent / "gridsearch_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Results saved to: %s", results_path)


if __name__ == "__main__":
    main()
