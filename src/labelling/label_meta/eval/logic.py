"""Evaluation helpers for meta-label models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config_logging import get_logger
from src.model.base import BaseModel
from src.path import LABEL_META_TRAIN_DIR

logger = get_logger(__name__)


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


def compute_trading_metrics(returns: np.ndarray, positions: np.ndarray) -> TradingMetrics:
    """Compute simple trading metrics from returns and positions."""
    strategy_returns = returns * positions
    trade_mask = positions != 0
    trade_returns = strategy_returns[trade_mask]

    n_trades = len(trade_returns)
    if n_trades == 0:
        return TradingMetrics(0, 0, 0, 0.0, 0.0, 0.0, None, None)

    n_profitable = int(np.sum(trade_returns > 0))
    n_losing = int(np.sum(trade_returns < 0))
    win_rate = n_profitable / n_trades if n_trades else 0.0
    total_return = float(np.sum(trade_returns))
    mean_return = float(np.mean(trade_returns))
    std_return = float(np.std(trade_returns, ddof=1)) if n_trades > 1 else 0.0
    sharpe_ratio = mean_return / std_return if std_return > 0 else None

    cum_returns = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cum_returns) if len(cum_returns) > 0 else np.array([])
    drawdowns = running_max - cum_returns if len(cum_returns) > 0 else np.array([])
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else None

    return TradingMetrics(
        n_trades=n_trades,
        n_profitable=n_profitable,
        n_losing=n_losing,
        win_rate=win_rate,
        total_return=total_return,
        mean_return=mean_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )


def get_available_trained_meta_models() -> List[tuple[str, str]]:
    """List trained meta models available on disk."""
    if not LABEL_META_TRAIN_DIR.exists():
        return []
    models: List[tuple[str, str]] = []
    for subdir in LABEL_META_TRAIN_DIR.iterdir():
        if subdir.is_dir() and (subdir / "final_meta_model.joblib").exists():
            parts = subdir.name.split("_")
            if len(parts) >= 2:
                primary = parts[0]
                meta = "_".join(parts[1:])
                models.append((primary, meta))
    return models


def load_meta_model(primary_model_name: str, meta_model_name: str) -> BaseModel:
    """Load a trained meta model."""
    model_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"
    model_path = model_dir / "final_meta_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Meta model not found: {model_path}")
    return BaseModel.load(model_path)


def load_meta_training_results(primary_model_name: str, meta_model_name: str) -> Dict[str, Any]:
    """Load stored training results for a meta model."""
    results_path = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}" / "training_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Meta training results not found: {results_path}")
    return json.loads(results_path.read_text())


def _remove_non_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    non_feature_cols = {"bar_id", "split", "datetime_close", "timestamp_open", "label"}
    cols = [c for c in df.columns if c not in non_feature_cols]
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
    """Evaluate combined primary + meta model performance."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    from src.labelling.label_meta.utils import (
        get_labeled_dataset_for_primary_model,
        load_primary_model,
    )

    meta_model = load_meta_model(primary_model_name, meta_model_name)
    primary_model = load_primary_model(primary_model_name)

    dataset = get_labeled_dataset_for_primary_model(primary_model_name)
    if "datetime_close" in dataset.columns:
        dataset = dataset.set_index("datetime_close")
    dataset = dataset.sort_index()

    features = _remove_non_feature_cols(dataset)
    features = _handle_missing_values(features)

    primary_preds = primary_model.predict(features)
    meta_preds = meta_model.predict(features)

    primary_positions = np.sign(primary_preds)
    combined_positions = primary_positions * meta_preds

    returns_arr: np.ndarray = np.asarray(dataset["ret"].values) if "ret" in dataset.columns else np.zeros(len(dataset))

    primary_metrics = compute_trading_metrics(returns_arr, np.asarray(primary_positions))
    combined_metrics = compute_trading_metrics(returns_arr, np.asarray(combined_positions))

    n_filtered = int(np.sum(combined_positions == 0))
    n_total = len(combined_positions)
    trades_filtered_pct = (n_filtered / n_total * 100) if n_total > 0 else 0.0

    win_rate_improvement = combined_metrics.win_rate - primary_metrics.win_rate

    actual_meta = np.where(returns_arr * primary_positions > 0, 1, 0)

    meta_accuracy = float(accuracy_score(actual_meta, meta_preds))
    meta_precision = float(precision_score(actual_meta, meta_preds, zero_division="warn"))
    meta_recall = float(recall_score(actual_meta, meta_preds, zero_division="warn"))
    meta_f1 = float(f1_score(actual_meta, meta_preds, zero_division="warn"))
    conf_matrix = confusion_matrix(actual_meta, meta_preds).tolist()

    model_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"
    predictions_path = model_dir / "eval_predictions.parquet"
    eval_df = pd.DataFrame({
        "primary_pred": primary_preds,
        "meta_pred": meta_preds,
        "combined_position": combined_positions,
        "return": returns_arr,
    }, index=features.index)
    eval_df.to_parquet(predictions_path)

    return CombinedEvaluationResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        n_test_samples=len(features),
        meta_accuracy=meta_accuracy,
        meta_precision=meta_precision,
        meta_recall=meta_recall,
        meta_f1=meta_f1,
        primary_only_metrics=primary_metrics.to_dict(),
        combined_metrics=combined_metrics.to_dict(),
        trades_filtered_pct=trades_filtered_pct,
        win_rate_improvement=win_rate_improvement,
        meta_confusion_matrix=conf_matrix,
        primary_model_path=str(LABEL_META_TRAIN_DIR.parent / "label_primaire" / "train" / primary_model_name),
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
    "_remove_non_feature_cols",
    "_handle_missing_values",
]
