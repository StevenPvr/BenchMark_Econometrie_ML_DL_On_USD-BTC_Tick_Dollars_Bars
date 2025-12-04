"""Evaluation helpers for meta-label models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config_logging import get_logger
from src.constants import (
    META_EVAL_BARS_PER_YEAR,
    META_EVAL_INITIAL_CAPITAL,
    META_EVAL_TOTAL_COST_PCT,
)
from src.model.base import BaseModel
from src.path import LABEL_META_TRAIN_DIR

logger = get_logger(__name__)

# Re-export constants for backward compatibility and convenience
INITIAL_CAPITAL: float = META_EVAL_INITIAL_CAPITAL
TOTAL_COST_PCT: float = META_EVAL_TOTAL_COST_PCT
BARS_PER_YEAR: int = META_EVAL_BARS_PER_YEAR


@dataclass
class TradingMetrics:
    """Trading performance metrics with costs (all monetary values in euros)."""

    n_trades: int
    n_profitable: int
    n_losing: int
    win_rate: float
    # P&L in euros (starting from INITIAL_CAPITAL)
    gross_pnl: float  # Total gross P&L before costs
    net_pnl: float  # Total net P&L after costs
    total_costs: float  # Total costs paid
    final_capital: float  # Final portfolio value
    # Returns as percentages
    gross_return_pct: float
    net_return_pct: float
    # Risk metrics
    sharpe_ratio: float | None
    max_drawdown: float | None  # In euros
    max_drawdown_pct: float | None  # As percentage
    profit_factor: float | None
    avg_win: float | None  # In euros
    avg_loss: float | None  # In euros

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "n_profitable": self.n_profitable,
            "n_losing": self.n_losing,
            "win_rate": self.win_rate,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "total_costs": self.total_costs,
            "final_capital": self.final_capital,
            "gross_return_pct": self.gross_return_pct,
            "net_return_pct": self.net_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }


@dataclass
class CombinedEvaluationResult:
    """Evaluation summary for a meta model attached to a primary model."""

    primary_model_name: str
    meta_model_name: str
    n_test_samples: int
    meta_accuracy: float
    meta_precision: float
    meta_recall: float
    meta_f1: float
    primary_only_metrics: Dict[str, Any]
    combined_metrics: Dict[str, Any]
    trades_filtered_pct: float
    win_rate_improvement: float
    meta_confusion_matrix: List[List[int]]
    primary_model_path: str
    meta_model_path: str
    predictions_path: str
    timestamp: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def compute_trading_metrics(
    log_returns: np.ndarray,
    positions: np.ndarray,
    initial_capital: float = INITIAL_CAPITAL,
    cost_pct: float = TOTAL_COST_PCT,
) -> TradingMetrics:
    """Compute trading metrics from log returns and positions with compounding.

    Simulates a portfolio starting with initial_capital (default 10,000€).
    Each trade uses 100% of current capital (all-in / all-out).
    Gains/losses are reinvested (compounding).

    Costs are applied as a percentage of the absolute P&L for each trade:
    - Win 10€ -> pay 0.03€ costs -> net +9.97€
    - Lose 10€ -> pay 0.03€ costs -> net -10.03€

    Args:
        log_returns: Array of log returns for each bar
        positions: Array of positions (-1, 0, 1) for each bar
        initial_capital: Starting portfolio value in euros (default 10,000€)
        cost_pct: Cost as percentage of trade value (default ~0.3%)

    Returns:
        TradingMetrics with P&L in euros, Sharpe, drawdown, etc.
    """
    trade_mask = positions != 0
    n_trades = int(np.sum(trade_mask))

    if n_trades == 0:
        return TradingMetrics(
            n_trades=0, n_profitable=0, n_losing=0, win_rate=0.0,
            gross_pnl=0.0, net_pnl=0.0, total_costs=0.0,
            final_capital=initial_capital,
            gross_return_pct=0.0, net_return_pct=0.0,
            sharpe_ratio=None, max_drawdown=None, max_drawdown_pct=None,
            profit_factor=None, avg_win=None, avg_loss=None,
        )

    # Simulate trading with compounding
    n_bars = len(log_returns)
    capital_history = np.zeros(n_bars + 1)
    capital_history[0] = initial_capital

    trade_gross_pnl_list: List[float] = []
    trade_net_pnl_list: List[float] = []
    trade_costs_list: List[float] = []

    current_capital = initial_capital

    for i in range(n_bars):
        if positions[i] == 0:
            # No trade, capital unchanged
            capital_history[i + 1] = current_capital
        else:
            # Trade with current capital
            # Gross P&L = capital * log_return * position
            gross_pnl_trade = current_capital * log_returns[i] * positions[i]

            # Cost = cost_pct * |gross P&L|
            cost_trade = cost_pct * abs(gross_pnl_trade)

            # Net P&L = gross - cost
            net_pnl_trade = gross_pnl_trade - cost_trade

            # Update capital (compounding)
            current_capital = current_capital + net_pnl_trade
            capital_history[i + 1] = current_capital

            # Store for stats
            trade_gross_pnl_list.append(gross_pnl_trade)
            trade_net_pnl_list.append(net_pnl_trade)
            trade_costs_list.append(cost_trade)

    # Convert to arrays
    trade_gross_pnl = np.array(trade_gross_pnl_list)
    trade_net_pnl = np.array(trade_net_pnl_list)
    trade_costs = np.array(trade_costs_list)

    # Win/loss stats (based on net P&L)
    n_profitable = int(np.sum(trade_net_pnl > 0))
    n_losing = int(np.sum(trade_net_pnl < 0))
    win_rate = n_profitable / n_trades

    # Totals
    gross_pnl = float(np.sum(trade_gross_pnl))
    total_costs = float(np.sum(trade_costs))
    final_capital = current_capital
    net_pnl = final_capital - initial_capital

    # Returns as percentages
    gross_return_pct = (gross_pnl / initial_capital) * 100
    net_return_pct = (net_pnl / initial_capital) * 100

    # Sharpe ratio (annualized, based on net returns per trade as % of capital at time)
    # Use log returns for Sharpe calculation (more appropriate for compounding)
    trade_log_returns = log_returns[trade_mask] * positions[trade_mask]
    # Adjust for costs (approximate: subtract cost_pct from absolute return)
    trade_net_log_returns = trade_log_returns - cost_pct * np.abs(trade_log_returns)
    mean_ret = float(np.mean(trade_net_log_returns))
    std_ret = float(np.std(trade_net_log_returns, ddof=1)) if n_trades > 1 else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(BARS_PER_YEAR)) if std_ret > 0 else None

    # Max drawdown (on capital history)
    running_max = np.maximum.accumulate(capital_history)
    drawdowns = running_max - capital_history
    max_dd = float(np.max(drawdowns))
    # Max drawdown % relative to peak
    peak_at_max_dd = running_max[np.argmax(drawdowns)]
    max_dd_pct = (max_dd / peak_at_max_dd) * 100 if peak_at_max_dd > 0 else None

    # Profit factor = sum(wins) / |sum(losses)|
    wins = trade_net_pnl[trade_net_pnl > 0]
    losses = trade_net_pnl[trade_net_pnl < 0]
    total_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    total_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else None

    avg_win = float(np.mean(wins)) if len(wins) > 0 else None
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else None

    return TradingMetrics(
        n_trades=n_trades,
        n_profitable=n_profitable,
        n_losing=n_losing,
        win_rate=win_rate,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        total_costs=total_costs,
        final_capital=final_capital,
        gross_return_pct=gross_return_pct,
        net_return_pct=net_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def get_available_trained_meta_models() -> List[tuple[str, str]]:
    """List trained meta models available on disk."""
    if not LABEL_META_TRAIN_DIR.exists():
        return []
    models: List[tuple[str, str]] = []
    for subdir in LABEL_META_TRAIN_DIR.iterdir():
        if subdir.is_dir():
            parts = subdir.name.split("_")
            if len(parts) >= 2:
                primary = parts[0]
                meta = "_".join(parts[1:])
                # Check if model file exists with meta model name
                model_path = subdir / f"{meta}_model.joblib"
                if model_path.exists():
                    models.append((primary, meta))
    return models


def load_meta_model(primary_model_name: str, meta_model_name: str) -> BaseModel:
    """Load a trained meta model."""
    model_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"
    model_path = model_dir / f"{meta_model_name}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Meta model not found: {model_path}")
    return BaseModel.load(model_path)


def load_meta_training_results(primary_model_name: str, meta_model_name: str) -> Dict[str, Any]:
    """Load stored training results for a meta model."""
    results_path = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}" / "training_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Meta training results not found: {results_path}")
    return json.loads(results_path.read_text())


def _remove_non_feature_cols(df: pd.DataFrame, keep_proba: bool = False) -> pd.DataFrame:
    non_feature_cols = {
        "bar_id", "split", "datetime_close", "datetime_open",
        "timestamp_open", "timestamp_close", "label", "prediction",
        "coverage", "t1", "threshold_used", "log_return",
    }
    cols = [c for c in df.columns if c not in non_feature_cols]
    # Remove proba columns unless explicitly kept
    if not keep_proba:
        cols = [c for c in cols if not c.startswith("proba_")]
    return pd.DataFrame(df[cols])


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for col in filled.columns:
        if bool(filled[col].isna().any()):
            median = filled[col].median()
            filled[col] = filled[col].fillna(0.0 if bool(pd.isna(median)) else median)
    return filled


def evaluate_combined_performance(
    primary_model_name: str,
    meta_model_name: str,
) -> CombinedEvaluationResult:
    """Evaluate combined primary + meta model performance on TEST set.

    - Primary model predicts on test set
    - Meta model filters these predictions
    - Compare performance: primary alone vs primary + meta filter
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    from src.labelling.label_meta.utils import get_labeled_dataset_for_primary_model, load_primary_model

    # Load models
    primary_model = load_primary_model(primary_model_name)
    meta_model = load_meta_model(primary_model_name, meta_model_name)

    # Load FULL labeled dataset (contains train with OOF + test with NaN)
    dataset = get_labeled_dataset_for_primary_model(primary_model_name)

    if "split" not in dataset.columns:
        raise ValueError("Dataset missing 'split' column - cannot separate test set")

    test_mask = dataset["split"] == "test"
    dataset = dataset.loc[test_mask].copy()
    logger.info("Evaluating on TEST set: %d samples", len(dataset))

    if len(dataset) == 0:
        raise ValueError("No test samples found in dataset")

    # Prepare base features WITHOUT proba columns (for primary model prediction)
    base_features = _remove_non_feature_cols(dataset, keep_proba=False)
    base_features = _handle_missing_values(base_features)

    # Primary model predictions on test set
    primary_preds = primary_model.predict(base_features)

    # Build meta features: base features + proba columns
    meta_features = base_features.copy()
    predict_proba_fn = getattr(primary_model, "predict_proba", None)
    if predict_proba_fn is not None:
        primary_proba = predict_proba_fn(base_features)
        n_classes = primary_proba.shape[1]
        for i in range(n_classes):
            meta_features[f"proba_{i}"] = primary_proba[:, i]

    features = meta_features

    # Filter to directional predictions only (not neutral)
    directional_mask = primary_preds != 0
    features_dir = features.loc[directional_mask]
    dataset_dir = dataset.loc[directional_mask]
    primary_preds_dir = primary_preds[directional_mask]

    logger.info("Directional predictions: %d/%d", len(features_dir), len(features))

    # Meta model predictions (filter the directional trades)
    meta_preds = meta_model.predict(features_dir)

    # Ground truth labels
    actual_labels = dataset_dir["label"].values

    # Was primary correct? (meta ground truth)
    actual_meta = (primary_preds_dir == actual_labels).astype(int)

    # Combined positions: trade only when meta says "take it" (1)
    combined_positions = np.where(meta_preds == 1, np.sign(primary_preds_dir), 0)

    # Compute meta model metrics (how well does it predict if primary is correct?)
    meta_accuracy = float(accuracy_score(actual_meta, meta_preds))
    meta_precision = float(precision_score(actual_meta, meta_preds))
    meta_recall = float(recall_score(actual_meta, meta_preds))
    meta_f1 = float(f1_score(actual_meta, meta_preds))
    conf_matrix = confusion_matrix(actual_meta, meta_preds).tolist()

    # Get log returns for P&L calculation
    if "log_return" in dataset_dir.columns:
        log_returns = dataset_dir["log_return"].values
    else:
        logger.warning("No 'log_return' column found, using label as proxy")
        log_returns = actual_labels.astype(float) * 0.001  # Fallback

    # Primary positions: sign of prediction (-1, 0, 1)
    primary_positions = np.sign(primary_preds_dir)

    # Combined positions: trade only when meta says "take it" (1)
    combined_positions = np.where(meta_preds == 1, primary_positions, 0)

    # Compute P&L metrics for both strategies
    primary_trading_metrics = compute_trading_metrics(log_returns, primary_positions)
    combined_trading_metrics = compute_trading_metrics(log_returns, combined_positions)

    # Basic win rate stats (for backward compatibility)
    primary_wins = (primary_preds_dir == actual_labels)
    n_primary_trades = len(primary_preds_dir)
    n_primary_wins = int(primary_wins.sum())
    primary_win_rate = n_primary_wins / n_primary_trades if n_primary_trades > 0 else 0.0

    combined_mask = meta_preds == 1
    n_combined_trades = int(combined_mask.sum())
    n_combined_wins = int((primary_wins & combined_mask).sum())
    combined_win_rate = n_combined_wins / n_combined_trades if n_combined_trades > 0 else 0.0

    n_filtered = n_primary_trades - n_combined_trades
    trades_filtered_pct = (n_filtered / n_primary_trades * 100) if n_primary_trades > 0 else 0.0
    win_rate_improvement = combined_win_rate - primary_win_rate

    # Merge basic stats with full trading metrics
    # Note: TradingMetrics.win_rate is based on net P&L > 0 (after costs)
    # classification_win_rate is based on correct predictions
    primary_metrics = {
        **primary_trading_metrics.to_dict(),
        "n_trades": n_primary_trades,
        "classification_win_rate": primary_win_rate,  # correct predictions
        "n_correct": n_primary_wins,
    }
    combined_metrics_dict = {
        **combined_trading_metrics.to_dict(),
        "n_trades": n_combined_trades,
        "classification_win_rate": combined_win_rate,  # correct predictions
        "n_correct": n_combined_wins,
    }

    # Save predictions
    model_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"
    predictions_path = model_dir / "eval_predictions.parquet"
    eval_df = pd.DataFrame({
        "primary_pred": primary_preds_dir,
        "actual_label": actual_labels,
        "meta_pred": meta_preds,
        "actual_meta": actual_meta,
        "combined_position": combined_positions,
    }, index=features_dir.index)
    eval_df.to_parquet(predictions_path)

    logger.info("Primary win rate: %.2f%% (%d/%d trades)",
                primary_win_rate * 100, n_primary_wins, n_primary_trades)
    logger.info("Combined win rate: %.2f%% (%d/%d trades) - filtered %.1f%%",
                combined_win_rate * 100, n_combined_wins, n_combined_trades, trades_filtered_pct)
    logger.info("Win rate improvement: %+.2f%%", win_rate_improvement * 100)

    return CombinedEvaluationResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        n_test_samples=len(features),
        meta_accuracy=meta_accuracy,
        meta_precision=meta_precision,
        meta_recall=meta_recall,
        meta_f1=meta_f1,
        primary_only_metrics=primary_metrics,
        combined_metrics=combined_metrics_dict,
        trades_filtered_pct=trades_filtered_pct,
        win_rate_improvement=win_rate_improvement,
        meta_confusion_matrix=conf_matrix,
        primary_model_path=str(LABEL_META_TRAIN_DIR.parent / "label_primaire_train" / primary_model_name),
        meta_model_path=str(model_dir / "final_meta_model.joblib"),
        predictions_path=str(predictions_path),
    )


__all__ = [
    "CombinedEvaluationResult",
    "TradingMetrics",
    "compute_trading_metrics",
    "evaluate_combined_performance",
    "get_available_trained_meta_models",
    "load_meta_model",
    "load_meta_training_results",
]
