"""
Combined Evaluation Module for Primary + Meta Models.

This module evaluates the COMBINED prediction system:
- Primary Model: Predicts direction (side: +1 Long, -1 Short)
- Meta Model: Filters false positives (take_trade: 0 or 1)

Final Position = side × take_trade
- If meta = 1: take the position (Long or Short)
- If meta = 0: skip the trade (position = 0)

Metrics computed:
- Meta model performance (filtering accuracy)
- Combined trading performance (profitability)
- Comparison: Primary only vs Primary + Meta

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado, Chapter 3.6
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import TEST_SPLIT_LABEL
from src.labelling.label_meta.utils import (
    MODEL_REGISTRY,
    get_dataset_for_model,
    get_labeled_dataset_for_primary_model,
    load_dollar_bars,
    load_primary_model,
    get_daily_volatility,
)
from src.labelling.label_meta.opti import get_events_meta, get_bins
from src.model.base import BaseModel
from src.path import (
    LABEL_META_TRAIN_DIR,
    LABEL_META_EVAL_DIR,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TradingMetrics:
    """Trading performance metrics."""

    n_trades: int
    n_profitable: int
    n_losing: int
    win_rate: float
    total_return: float
    mean_return: float
    sharpe_ratio: float | None
    max_drawdown: float | None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_trades": self.n_trades,
            "n_profitable": self.n_profitable,
            "n_losing": self.n_losing,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class CombinedEvaluationResult:
    """Complete evaluation result for combined Primary + Meta system."""

    primary_model_name: str
    meta_model_name: str
    n_test_samples: int

    # Meta model filtering metrics
    meta_accuracy: float
    meta_precision: float
    meta_recall: float
    meta_f1: float

    # Trading metrics: Primary only
    primary_only_metrics: Dict[str, Any]

    # Trading metrics: Primary + Meta (filtered)
    combined_metrics: Dict[str, Any]

    # Improvement stats
    trades_filtered_pct: float
    win_rate_improvement: float

    # Confusion matrix for meta model
    meta_confusion_matrix: List[List[int]]

    # Paths
    primary_model_path: str
    meta_model_path: str
    predictions_path: str

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_model_name": self.primary_model_name,
            "meta_model_name": self.meta_model_name,
            "n_test_samples": self.n_test_samples,
            "meta_accuracy": self.meta_accuracy,
            "meta_precision": self.meta_precision,
            "meta_recall": self.meta_recall,
            "meta_f1": self.meta_f1,
            "primary_only_metrics": self.primary_only_metrics,
            "combined_metrics": self.combined_metrics,
            "trades_filtered_pct": self.trades_filtered_pct,
            "win_rate_improvement": self.win_rate_improvement,
            "meta_confusion_matrix": self.meta_confusion_matrix,
            "primary_model_path": self.primary_model_path,
            "meta_model_path": self.meta_model_path,
            "predictions_path": self.predictions_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {path}")


# =============================================================================
# TRADING METRICS COMPUTATION
# =============================================================================


def compute_trading_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
) -> TradingMetrics:
    """
    Compute trading performance metrics.

    Parameters
    ----------
    returns : np.ndarray
        Actual returns for each period.
    positions : np.ndarray
        Position for each period (-1, 0, or +1).

    Returns
    -------
    TradingMetrics
        Trading performance metrics.
    """
    # Strategy returns = position × actual return
    strategy_returns = positions * returns

    # Filter to actual trades (non-zero positions)
    trade_mask = positions != 0
    trade_returns = strategy_returns[trade_mask]

    n_trades = len(trade_returns)

    if n_trades == 0:
        return TradingMetrics(
            n_trades=0,
            n_profitable=0,
            n_losing=0,
            win_rate=0.0,
            total_return=0.0,
            mean_return=0.0,
            sharpe_ratio=None,
            max_drawdown=None,
        )

    n_profitable = (trade_returns > 0).sum()
    n_losing = (trade_returns < 0).sum()
    win_rate = n_profitable / n_trades if n_trades > 0 else 0.0

    total_return = trade_returns.sum()
    mean_return = trade_returns.mean()

    # Sharpe ratio (annualized, assuming ~252 trading days)
    if trade_returns.std() > 0:
        sharpe_ratio = (mean_return / trade_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = None

    # Max drawdown
    cumulative = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else None

    return TradingMetrics(
        n_trades=n_trades,
        n_profitable=int(n_profitable),
        n_losing=int(n_losing),
        win_rate=win_rate,
        total_return=total_return,
        mean_return=mean_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )


# =============================================================================
# DATA LOADING
# =============================================================================


def load_meta_model(primary_model_name: str, meta_model_name: str) -> BaseModel:
    """Load a trained meta model."""
    model_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"
    model_path = model_dir / "final_meta_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Meta model not found: {model_path}\n"
            "Train meta model first: python -m src.labelling.label_meta.train"
        )

    return BaseModel.load(model_path)


def load_meta_training_results(
    primary_model_name: str,
    meta_model_name: str,
) -> Dict[str, Any]:
    """Load training results for meta model."""
    model_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"
    results_path = model_dir / "training_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Training results not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_trained_meta_models() -> List[Tuple[str, str]]:
    """Get list of trained meta models."""
    if not LABEL_META_TRAIN_DIR.exists():
        return []

    models = []
    for subdir in LABEL_META_TRAIN_DIR.iterdir():
        if subdir.is_dir():
            model_file = subdir / "final_meta_model.joblib"
            if model_file.exists():
                parts = subdir.name.split("_")
                if len(parts) >= 2:
                    primary = parts[0]
                    meta = "_".join(parts[1:])
                    models.append((primary, meta))
    return models


# =============================================================================
# DATA PREPARATION
# =============================================================================


def _load_and_filter_features(primary_model_name: str) -> pd.DataFrame:
    """
    Load TEST features from the primary model's labeled dataset.

    Parameters
    ----------
    primary_model_name : str
        Name of the primary model (e.g., 'lightgbm', 'xgboost').
        This determines which labeled features file to load.
    """
    features_df = get_labeled_dataset_for_primary_model(primary_model_name)

    if "split" in features_df.columns:
        features_df = features_df[features_df["split"] == TEST_SPLIT_LABEL].copy()
        features_df = features_df.drop(columns=["split"])

    if "datetime_close" in features_df.columns:
        features_df = features_df.set_index("datetime_close")
    features_df = features_df.sort_index()

    if features_df.index.has_duplicates:
        features_df = cast(pd.DataFrame, features_df[~features_df.index.duplicated(keep="first")])

    return cast(pd.DataFrame, features_df)


def _remove_non_feature_cols(features_df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns."""
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label",
    ]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    return cast(pd.DataFrame, features_df[feature_cols])


def _handle_missing_values(features_df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by filling with median."""
    for col in features_df.columns:
        is_na_any = features_df[col].isna().any()
        if isinstance(is_na_any, bool) and is_na_any:
            median_val = features_df[col].median()
            is_median_na = pd.isna(median_val)
            if isinstance(is_median_na, bool) and is_median_na:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = features_df[col].fillna(median_val)
    return features_df


def prepare_test_data(
    primary_model: BaseModel,
    meta_model: BaseModel,
    primary_model_name: str,
    meta_model_name: str,
    tb_params: Dict[str, Any],
    vol_window: int = 100,
) -> pd.DataFrame:
    """
    Prepare TEST data with all predictions and actual returns.

    Parameters
    ----------
    primary_model : BaseModel
        Trained primary model.
    meta_model : BaseModel
        Trained meta model.
    primary_model_name : str
        Name of the primary model (used to load correct labeled dataset).
    meta_model_name : str
        Name of the meta model.
    tb_params : Dict[str, Any]
        Triple barrier parameters.
    vol_window : int, optional
        Volatility window span, by default 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - side: Primary model direction prediction (+1/-1)
        - take_trade: Meta model filter prediction (0/1)
        - position: Combined position (side × take_trade)
        - y_true_meta: Actual meta label (was trade profitable?)
        - actual_return: Actual return from the trade
    """
    logger.info("Loading TEST data...")

    features_df = _load_and_filter_features(primary_model_name)
    features_df = _handle_missing_values(features_df)
    X_test = _remove_non_feature_cols(features_df)

    logger.info(f"TEST samples: {len(X_test)}")

    # Load dollar bars for returns
    bars = load_dollar_bars()
    close = bars["close"]

    # Align close to features
    close = close.loc[
        (close.index >= X_test.index[0]) &
        (close.index <= X_test.index[-1])
    ]
    if close.index.has_duplicates:
        close = close.loc[~close.index.duplicated(keep="first")]

    volatility = get_daily_volatility(close, span=vol_window)

    # Generate PRIMARY model predictions (side)
    logger.info("Generating primary model predictions...")
    side_predictions = pd.Series(
        primary_model.predict(X_test),
        index=X_test.index,
        name="side",
    )

    # Filter to valid sides (+1 or -1)
    valid_sides = side_predictions.isin([1, -1])
    side_predictions = side_predictions.loc[valid_sides]
    X_test_valid = X_test.loc[valid_sides]

    logger.info(f"Valid side predictions: {len(side_predictions)}")
    logger.info(f"Side distribution: {side_predictions.value_counts().to_dict()}")

    # Generate META model predictions (take_trade)
    logger.info("Generating meta model predictions...")
    meta_predictions = pd.Series(
        meta_model.predict(X_test_valid),
        index=X_test_valid.index,
        name="take_trade",
    )

    logger.info(f"Meta distribution: {meta_predictions.value_counts().to_dict()}")

    # Generate actual meta labels (ground truth)
    logger.info("Generating actual meta labels...")
    events = get_events_meta(
        close=close,
        t_events=pd.DatetimeIndex(side_predictions.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility,
        max_holding=tb_params["max_holding"],
        side=side_predictions,
    )
    if not events.empty:
        events = get_bins(events, close)

    # Align all data
    common_idx = side_predictions.index.intersection(events.index)
    common_idx = common_idx.intersection(meta_predictions.index)

    result_df = pd.DataFrame({
        "side": side_predictions.loc[common_idx].values,
        "take_trade": meta_predictions.loc[common_idx].values,
        "y_true_meta": events.loc[common_idx, "bin"].values,
        "actual_return": events.loc[common_idx, "ret"].values,
    }, index=common_idx)

    # Compute combined position
    result_df["position"] = result_df["side"] * result_df["take_trade"]

    # Position for primary only (always trade)
    result_df["position_primary_only"] = result_df["side"]

    logger.info(f"Final aligned samples: {len(result_df)}")

    return result_df


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_combined(
    primary_model_name: str,
    meta_model_name: str,
    output_dir: Path | None = None,
) -> CombinedEvaluationResult:
    """
    Evaluate the combined Primary + Meta trading system on TEST set.

    Compares:
    1. Primary model alone (trade every signal)
    2. Primary + Meta (filtered signals)
    """
    if output_dir is None:
        output_dir = LABEL_META_EVAL_DIR / f"{primary_model_name}_{meta_model_name}"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"COMBINED EVALUATION: {primary_model_name} + {meta_model_name}")
    logger.info(f"{'='*60}")

    # Load models
    logger.info("\nLoading models...")
    primary_model = load_primary_model(primary_model_name)
    meta_model = load_meta_model(primary_model_name, meta_model_name)

    # Get actual primary model path (try both patterns)
    primary_model_path_main = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name / f"{primary_model_name}_final_model.joblib"
    primary_model_path_train = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name / f"{primary_model_name}_model.joblib"
    primary_model_path = primary_model_path_main if primary_model_path_main.exists() else primary_model_path_train
    meta_model_path = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}" / "final_meta_model.joblib"

    # Load training results for TB params
    training_results = load_meta_training_results(primary_model_name, meta_model_name)
    tb_params = training_results["triple_barrier_params"]

    # Prepare data
    test_df = prepare_test_data(
        primary_model, meta_model, primary_model_name, meta_model_name, tb_params
    )

    # Save predictions
    predictions_path = output_dir / "test_predictions.parquet"
    test_df.to_parquet(predictions_path)
    logger.info(f"Predictions saved: {predictions_path}")

    # =========================================================================
    # META MODEL METRICS
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("META MODEL FILTERING PERFORMANCE")
    logger.info(f"{'='*60}")

    y_true_meta = test_df["y_true_meta"].values.astype(int)
    y_pred_meta = test_df["take_trade"].values.astype(int)

    meta_accuracy = accuracy_score(y_true_meta, y_pred_meta)
    meta_precision = precision_score(y_true_meta, y_pred_meta, zero_division="warn")
    meta_recall = recall_score(y_true_meta, y_pred_meta, zero_division="warn")
    meta_f1 = f1_score(y_true_meta, y_pred_meta, zero_division="warn")
    meta_cm = confusion_matrix(y_true_meta, y_pred_meta).tolist()

    logger.info(f"Meta Accuracy: {meta_accuracy:.4f}")
    logger.info(f"Meta Precision: {meta_precision:.4f}")
    logger.info(f"Meta Recall: {meta_recall:.4f}")
    logger.info(f"Meta F1: {meta_f1:.4f}")
    logger.info(f"\nConfusion Matrix (Meta):")
    logger.info(f"  TN={meta_cm[0][0]}, FP={meta_cm[0][1]}")
    logger.info(f"  FN={meta_cm[1][0]}, TP={meta_cm[1][1]}")

    # =========================================================================
    # TRADING METRICS: PRIMARY ONLY
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("TRADING METRICS: PRIMARY MODEL ONLY")
    logger.info(f"{'='*60}")

    actual_returns = cast(np.ndarray, test_df["actual_return"].values)
    positions_primary = cast(np.ndarray, test_df["position_primary_only"].values)

    primary_metrics = compute_trading_metrics(actual_returns, positions_primary)
    logger.info(f"Trades: {primary_metrics.n_trades}")
    logger.info(f"Win Rate: {primary_metrics.win_rate:.2%}")
    logger.info(f"Total Return: {primary_metrics.total_return:.4f}")
    logger.info(f"Mean Return: {primary_metrics.mean_return:.6f}")
    if primary_metrics.sharpe_ratio:
        logger.info(f"Sharpe Ratio: {primary_metrics.sharpe_ratio:.2f}")

    # =========================================================================
    # TRADING METRICS: PRIMARY + META (FILTERED)
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("TRADING METRICS: PRIMARY + META (FILTERED)")
    logger.info(f"{'='*60}")

    positions_combined = cast(np.ndarray, test_df["position"].values)

    combined_metrics = compute_trading_metrics(actual_returns, positions_combined)
    logger.info(f"Trades: {combined_metrics.n_trades}")
    logger.info(f"Win Rate: {combined_metrics.win_rate:.2%}")
    logger.info(f"Total Return: {combined_metrics.total_return:.4f}")
    logger.info(f"Mean Return: {combined_metrics.mean_return:.6f}")
    if combined_metrics.sharpe_ratio:
        logger.info(f"Sharpe Ratio: {combined_metrics.sharpe_ratio:.2f}")

    # =========================================================================
    # IMPROVEMENT ANALYSIS
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info(f"{'='*60}")

    trades_filtered = primary_metrics.n_trades - combined_metrics.n_trades
    trades_filtered_pct = trades_filtered / primary_metrics.n_trades * 100 if primary_metrics.n_trades > 0 else 0

    win_rate_improvement = combined_metrics.win_rate - primary_metrics.win_rate

    logger.info(f"Trades filtered: {trades_filtered} ({trades_filtered_pct:.1f}%)")
    logger.info(f"Win rate improvement: {win_rate_improvement:+.2%}")

    if primary_metrics.total_return != 0:
        return_improvement = (combined_metrics.total_return - primary_metrics.total_return) / abs(primary_metrics.total_return) * 100
        logger.info(f"Return improvement: {return_improvement:+.1f}%")

    # Build result
    result = CombinedEvaluationResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        n_test_samples=len(test_df),
        meta_accuracy=meta_accuracy,
        meta_precision=meta_precision,
        meta_recall=meta_recall,
        meta_f1=meta_f1,
        primary_only_metrics=primary_metrics.to_dict(),
        combined_metrics=combined_metrics.to_dict(),
        trades_filtered_pct=trades_filtered_pct,
        win_rate_improvement=win_rate_improvement,
        meta_confusion_matrix=meta_cm,
        primary_model_path=str(primary_model_path),
        meta_model_path=str(meta_model_path),
        predictions_path=str(predictions_path),
    )

    results_path = output_dir / "evaluation_results.json"
    result.save(results_path)

    return result


# =============================================================================
# DISPLAY
# =============================================================================


def print_comparison_table(result: CombinedEvaluationResult) -> None:
    """Print comparison between Primary only and Combined."""
    print("\n" + "=" * 70)
    print("COMPARISON: PRIMARY ONLY vs PRIMARY + META")
    print("=" * 70)

    p = result.primary_only_metrics
    c = result.combined_metrics

    print(f"\n{'Metric':<25} {'Primary Only':>15} {'Primary+Meta':>15} {'Delta':>12}")
    print("-" * 70)

    metrics_to_compare = [
        ("n_trades", "Trades", ""),
        ("win_rate", "Win Rate", "%"),
        ("total_return", "Total Return", ""),
        ("mean_return", "Mean Return", ""),
        ("sharpe_ratio", "Sharpe Ratio", ""),
    ]

    for key, name, suffix in metrics_to_compare:
        p_val = p.get(key)
        c_val = c.get(key)

        if p_val is None and c_val is None:
            continue

        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        c_str = f"{c_val:.4f}" if c_val is not None else "N/A"

        if suffix == "%":
            p_str = f"{p_val:.2%}" if p_val is not None else "N/A"
            c_str = f"{c_val:.2%}" if c_val is not None else "N/A"
            if p_val is not None and c_val is not None:
                delta = c_val - p_val
                delta_str = f"{delta:+.2%}"
            else:
                delta_str = "N/A"
        else:
            if p_val is not None and c_val is not None:
                delta = c_val - p_val
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "N/A"

        print(f"{name:<25} {p_str:>15} {c_str:>15} {delta_str:>12}")

    print("-" * 70)
    print(f"{'Trades Filtered':<25} {'-':>15} {result.trades_filtered_pct:>14.1f}%")
    print("=" * 70)


def print_evaluation_results(result: CombinedEvaluationResult) -> None:
    """Print complete evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {result.primary_model_name} + {result.meta_model_name}")
    print("=" * 70)

    print(f"\nTest Samples: {result.n_test_samples}")

    print("\n--- Meta Model Performance ---")
    print(f"  Accuracy:  {result.meta_accuracy:.4f}")
    print(f"  Precision: {result.meta_precision:.4f}")
    print(f"  Recall:    {result.meta_recall:.4f}")
    print(f"  F1 Score:  {result.meta_f1:.4f}")

    print("\n  Confusion Matrix:")
    cm = result.meta_confusion_matrix
    print(f"    Predicted:  Skip(0)  Take(1)")
    print(f"    Actual 0:   {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"    Actual 1:   {cm[1][0]:>6}   {cm[1][1]:>6}")

    print_comparison_table(result)

    print("\n--- Fichiers ---")
    print(f"  Predictions: {result.predictions_path}")

    print("=" * 70)


# =============================================================================
# INTERACTIVE CLI
# =============================================================================


def select_models() -> Tuple[str, str]:
    """Interactive selection of trained meta models."""
    available = get_available_trained_meta_models()

    print("\n" + "=" * 60)
    print("COMBINED EVALUATION: PRIMARY + META")
    print("=" * 60)
    print("\nEvaluates combined trading system on TEST set.")
    print("\nModeles entraines disponibles:")
    print("-" * 40)

    if not available:
        print("Aucun meta model entraine!")
        print("Lancer d'abord: python -m src.labelling.label_meta.train")
        sys.exit(1)

    for i, (primary, meta) in enumerate(available, 1):
        print(f"  {i}. Primary: {primary}, Meta: {meta}")

    print("-" * 40)

    while True:
        try:
            choice = input("\nChoisir (numero): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    return available[idx]
            print("Choix invalide.")
        except KeyboardInterrupt:
            print("\nAnnule.")
            sys.exit(0)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    primary_model_name, meta_model_name = select_models()
    print(f"\nSelectionne: Primary={primary_model_name}, Meta={meta_model_name}")

    confirm = input("\nLancer l'evaluation? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    result = evaluate_combined(primary_model_name, meta_model_name)

    print_evaluation_results(result)


if __name__ == "__main__":
    main()
