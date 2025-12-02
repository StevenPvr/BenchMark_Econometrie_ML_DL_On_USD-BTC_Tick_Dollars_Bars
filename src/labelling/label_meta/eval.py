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


import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
    get_labeled_dataset_for_primary_model,
    load_primary_model,
)
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
    model_path = model_dir / f"{meta_model_name}_meta_model.joblib"

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
            # Parse directory name: primary_meta
            parts = subdir.name.split("_")
            if len(parts) >= 2:
                primary = parts[0]
                meta = "_".join(parts[1:])
                # Check if model file exists with meta model name
                model_file = subdir / f"{meta}_meta_model.joblib"
                if model_file.exists():
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


def _remove_non_feature_cols(features_df: pd.DataFrame, keep_side: bool = False) -> pd.DataFrame:
    """
    Remove non-feature columns.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with all columns.
    keep_side : bool, optional
        If True, keep 'side' column (needed for meta model), by default False.
    """
    non_feature_cols = {
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label", "t1",
        "prediction",  # OOF predictions from primary model training
    }

    # Add 'side' to excluded columns only if not needed
    if not keep_side:
        non_feature_cols.add("side")

    # Filter out prediction and probability columns
    feature_cols = [
        c for c in features_df.columns
        if c not in non_feature_cols and not c.startswith("proba_")
    ]
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
    filter_neutral_labels: bool = True,
) -> pd.DataFrame:
    """
    Prepare TEST data with all predictions and actual returns.

    Workflow:
    1. Load test features from primary model dataset (with labels)
    2. Predict with primary model → side
    3. Calculate meta labels: meta_label = 1 if (label × side) > 0 else 0
    4. Predict with meta model (features + side) → take_trade
    5. Evaluate both meta model and combined system

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
        Triple barrier parameters (not used, kept for compatibility).
    filter_neutral_labels : bool, optional
        Filter neutral labels (label=0), by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - side: Primary model direction prediction (+1/-1)
        - take_trade: Meta model filter prediction (0/1)
        - position: Combined position (side × take_trade)
        - y_true_meta: Actual meta label (was primary correct?)
        - actual_return: Actual return from the trade
    """
    logger.info("Loading TEST data...")

    features_df = _load_and_filter_features(primary_model_name)
    features_df = _handle_missing_values(features_df)

    # Validate required columns
    if "label" not in features_df.columns:
        raise ValueError("Test dataset must contain 'label' column (triple barrier labels)")

    logger.info(f"TEST samples: {len(features_df)}")

    # STEP 1: Generate PRIMARY model predictions (side)
    logger.info("STEP 1: Generating primary model predictions (side)...")
    X_test_for_primary = _remove_non_feature_cols(features_df)
    side_predictions = pd.Series(
        primary_model.predict(X_test_for_primary),
        index=X_test_for_primary.index,
        name="side",
    )

    # Filter to valid sides (+1 or -1)
    valid_sides = side_predictions.isin([1, -1])
    side_predictions = side_predictions.loc[valid_sides]
    features_df_valid = features_df.loc[valid_sides].copy()

    logger.info(f"Valid side predictions: {len(side_predictions)}")
    logger.info(f"Side distribution: {side_predictions.value_counts().to_dict()}")

    # STEP 2: Calculate TRUE meta labels from dataset
    logger.info("STEP 2: Calculating true meta labels from dataset...")
    label_test = features_df_valid["label"].values

    y_true_meta = np.where(
        label_test * side_predictions.values > 0,
        1,  # Primary was correct
        0   # Primary was incorrect
    )

    # Filter neutral labels if configured
    if filter_neutral_labels:
        neutral_mask = label_test == 0
        n_neutral = neutral_mask.sum()
        if n_neutral > 0:
            logger.info(f"Filtering {n_neutral} neutral labels from test")
            valid_mask = ~neutral_mask
            features_df_valid = features_df_valid[valid_mask].copy()
            side_predictions = side_predictions[valid_mask]
            y_true_meta = y_true_meta[valid_mask]

    # STEP 3: Generate META model predictions (take_trade)
    logger.info("STEP 3: Generating meta model predictions (take_trade)...")

    # Add 'side' column to features for meta model
    features_df_valid["side"] = side_predictions.values

    # Get features for meta model (KEEP 'side' column)
    X_test_for_meta = _remove_non_feature_cols(features_df_valid, keep_side=True)

    meta_predictions = pd.Series(
        meta_model.predict(X_test_for_meta),
        index=X_test_for_meta.index,
        name="take_trade",
    )

    logger.info(f"Meta distribution: {meta_predictions.value_counts().to_dict()}")
    logger.info(f"Meta true labels distribution: {pd.Series(y_true_meta).value_counts().to_dict()}")

    # STEP 4: Get actual returns from dataset (no need to recalculate triple barrier)
    logger.info("STEP 4: Getting actual returns from dataset...")

    # Align all data
    common_idx = side_predictions.index.intersection(meta_predictions.index)

    result_df = pd.DataFrame({
        "side": side_predictions.loc[common_idx].values,
        "take_trade": meta_predictions.loc[common_idx].values,
        "y_true_meta": y_true_meta[side_predictions.index.get_indexer(common_idx)],
    }, index=common_idx)

    # Get actual returns from the dataset (log_return column)
    if "log_return" in features_df.columns:
        returns_aligned = features_df.loc[common_idx, "log_return"]
        result_df["actual_return"] = returns_aligned.values
    else:
        logger.warning("No 'log_return' column in dataset, returns will be NaN")
        result_df["actual_return"] = np.nan

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
    meta_model_path = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}" / f"{meta_model_name}_meta_model.joblib"

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
    # META MODEL CLASSIFICATION METRICS
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("META MODEL CLASSIFICATION PERFORMANCE")
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
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={meta_cm[0][0]}, FP={meta_cm[0][1]}")
    logger.info(f"  FN={meta_cm[1][0]}, TP={meta_cm[1][1]}")

    # Calculate class distribution
    n_total = len(y_true_meta)
    n_take_predicted = y_pred_meta.sum()
    n_skip_predicted = n_total - n_take_predicted
    n_take_actual = y_true_meta.sum()
    n_skip_actual = n_total - n_take_actual

    logger.info(f"\nClass Distribution:")
    logger.info(f"  Predicted: Skip={n_skip_predicted} ({n_skip_predicted/n_total*100:.1f}%), Take={n_take_predicted} ({n_take_predicted/n_total*100:.1f}%)")
    logger.info(f"  Actual:    Skip={n_skip_actual} ({n_skip_actual/n_total*100:.1f}%), Take={n_take_actual} ({n_take_actual/n_total*100:.1f}%)")

    # =========================================================================
    # WIN RATE ANALYSIS
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("WIN RATE ANALYSIS")
    logger.info(f"{'='*60}")

    # Calculate win rate for primary model predictions (side)
    # A "win" is when side prediction matches the actual label
    side_predictions = test_df["side"].to_numpy()
    actual_labels = test_df["y_true_meta"].to_numpy()

    # Primary model win rate (on all predictions)   
    primary_correct = (side_predictions * actual_labels) > 0
    primary_win_rate = primary_correct.sum() / len(primary_correct)

    # Meta filtered win rate (only on trades where take_trade=1)
    take_mask = y_pred_meta == 1
    if take_mask.sum() > 0:
        filtered_correct = primary_correct[take_mask]
        filtered_win_rate = filtered_correct.sum() / len(filtered_correct)
        win_rate_improvement = filtered_win_rate - primary_win_rate
    else:
        filtered_win_rate = 0.0
        win_rate_improvement = 0.0

    logger.info(f"Primary model win rate (all predictions): {primary_win_rate:.2%}")
    logger.info(f"Filtered win rate (meta take_trade=1): {filtered_win_rate:.2%}")
    logger.info(f"Win rate improvement: {win_rate_improvement:+.2%}")
    logger.info(f"Trades filtered: {n_skip_predicted} ({n_skip_predicted/n_total*100:.1f}%)")

    # Build result (classification metrics + win rate)
    result = CombinedEvaluationResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        n_test_samples=len(test_df),
        meta_accuracy=meta_accuracy,
        meta_precision=meta_precision,
        meta_recall=meta_recall,
        meta_f1=meta_f1,
        primary_only_metrics={"win_rate": primary_win_rate},
        combined_metrics={"win_rate": filtered_win_rate},
        trades_filtered_pct=n_skip_predicted / n_total * 100,
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
    """Print classification metrics and win rate."""
    print("\n" + "=" * 70)
    print(f"META MODEL EVALUATION: {result.primary_model_name} + {result.meta_model_name}")
    print("=" * 70)

    print(f"\nTest Samples: {result.n_test_samples}")

    print("\n--- Classification Metrics ---")
    print(f"  Accuracy:  {result.meta_accuracy:.4f}")
    print(f"  Precision: {result.meta_precision:.4f}")
    print(f"  Recall:    {result.meta_recall:.4f}")
    print(f"  F1 Score:  {result.meta_f1:.4f}")

    print("\n  Confusion Matrix:")
    cm = result.meta_confusion_matrix
    print(f"    Predicted:  Skip(0)  Take(1)")
    print(f"    Actual 0:   {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"    Actual 1:   {cm[1][0]:>6}   {cm[1][1]:>6}")

    print("\n--- Win Rate Analysis ---")
    primary_wr = result.primary_only_metrics.get("win_rate", 0.0)
    filtered_wr = result.combined_metrics.get("win_rate", 0.0)
    print(f"  Primary model (all):      {primary_wr:.2%}")
    print(f"  Filtered (take_trade=1):  {filtered_wr:.2%}")
    print(f"  Improvement:              {result.win_rate_improvement:+.2%}")
    print(f"  Trades filtered:          {result.trades_filtered_pct:.1f}%")

    print("\n--- Fichiers ---")
    print(f"  Predictions: {result.predictions_path}")
    print(f"  Results: {result.predictions_path.replace('test_predictions.parquet', 'evaluation_results.json')}")

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
