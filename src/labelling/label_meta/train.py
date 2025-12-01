"""
Meta Model Training with Out-of-Sample Label Generation.

This module implements the META model training pipeline that AVOIDS DATA LEAKAGE.
The meta model learns to FILTER false positives from the primary model.

Pipeline:
1. Load trained PRIMARY model (generates direction: +1 Long, -1 Short)
2. Generate OOS meta-labels on TRAIN using K-Fold Walk-Forward:
   - For each fold k, train meta model on folds 1..k-1, predict on fold k
   - Meta-label = 1 if primary's direction was correct, 0 otherwise
3. Train final META model on entire TRAIN set (one-shot)
4. Generate meta-labels on TEST using the final model
5. Save all predictions for combined evaluation

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
from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import DEFAULT_RANDOM_STATE, TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL
from src.labelling.label_meta.opti import get_events_meta, get_bins
from src.labelling.label_meta.utils import (
    MODEL_REGISTRY,
    MetaOptimizationConfig,
    get_daily_volatility,
    get_dataset_for_model,
    load_dollar_bars,
    load_model_class,
    load_primary_model,
    get_available_primary_models,
)
from src.model.base import BaseModel
from src.path import (
    LABEL_META_OPTI_DIR,
    LABEL_META_TRAIN_DIR,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# WALK-FORWARD K-FOLD CROSS-VALIDATION
# =============================================================================


class WalkForwardKFold:
    """
    Walk-Forward K-Fold Cross-Validation for time series.

    For fold k, trains on ALL data BEFORE fold k (folds 1..k-1).
    Never uses future data for training.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        min_train_size: int = 100,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        t1: pd.Series | None = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation indices for each fold using walk-forward."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        splits = []

        for fold_idx in range(self.n_splits):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < self.n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]

            # Training: ALL data BEFORE this fold
            train_end = val_start

            # Apply embargo
            if embargo_size > 0:
                train_end = max(0, train_end - embargo_size)

            train_indices = indices[:train_end]

            # Apply purging if t1 is provided
            if t1 is not None and len(train_indices) > 0:
                train_indices = self._apply_purge(train_indices, val_indices, X, t1)

            if len(train_indices) >= self.min_train_size and len(val_indices) > 0:
                splits.append((train_indices, val_indices))

        return splits

    def _apply_purge(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        X: pd.DataFrame,
        t1: pd.Series,
    ) -> np.ndarray:
        """Remove training samples whose labels overlap validation period."""
        if len(val_indices) == 0:
            return train_indices

        val_start_time = X.index[val_indices[0]]
        train_times = X.index[train_indices]
        t1_train = t1.reindex(train_times)

        overlap_mask = t1_train.notna() & (t1_train >= val_start_time)
        return train_indices[~overlap_mask.to_numpy()]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MetaEvaluationMetrics:
    """Evaluation metrics for binary meta classification."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
        }


@dataclass
class MetaTrainingResult:
    """Complete training result for meta model."""

    primary_model_name: str
    meta_model_name: str
    meta_model_params: Dict[str, Any]
    triple_barrier_params: Dict[str, Any]
    train_samples: int
    test_samples: int
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    meta_label_distribution_train: Dict[str, Any]
    meta_label_distribution_test: Dict[str, Any]
    n_folds: int
    primary_model_path: str
    meta_model_path: str
    oof_predictions_path: str
    test_predictions_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_model_name": self.primary_model_name,
            "meta_model_name": self.meta_model_name,
            "meta_model_params": self.meta_model_params,
            "triple_barrier_params": self.triple_barrier_params,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "meta_label_distribution_train": self.meta_label_distribution_train,
            "meta_label_distribution_test": self.meta_label_distribution_test,
            "n_folds": self.n_folds,
            "primary_model_path": self.primary_model_path,
            "meta_model_path": self.meta_model_path,
            "oof_predictions_path": self.oof_predictions_path,
            "test_predictions_path": self.test_predictions_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Training results saved to {path}")


# =============================================================================
# PARAMETER LOADING
# =============================================================================


def load_optimized_params(
    primary_model_name: str,
    meta_model_name: str,
    opti_dir: Path | None = None,
) -> Dict[str, Any]:
    """Load optimized parameters from meta model optimization."""
    if opti_dir is None:
        opti_dir = LABEL_META_OPTI_DIR

    opti_file = opti_dir / f"{primary_model_name}_{meta_model_name}_optimization.json"

    if not opti_file.exists():
        raise FileNotFoundError(
            f"Meta optimization results not found: {opti_file}\n"
            f"Run optimization first: python -m src.labelling.label_meta.opti"
        )

    with open(opti_file, "r", encoding="utf-8") as f:
        opti_results = json.load(f)

    return {
        "meta_model_params": opti_results["best_params"],
        "triple_barrier_params": opti_results["best_triple_barrier_params"],
        "best_score": opti_results.get("best_score", None),
        "metric": opti_results.get("metric", "f1_score"),
    }


# =============================================================================
# EVALUATION
# =============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> MetaEvaluationMetrics:
    """Compute binary classification metrics for meta model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division="warn")
    recall = recall_score(y_true, y_pred, zero_division="warn")
    f1 = f1_score(y_true, y_pred, zero_division="warn")

    auc_roc = None
    if y_proba is not None:
        try:
            # For binary, use probability of positive class
            if y_proba.ndim == 2:
                y_proba_pos = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, 0]
            else:
                y_proba_pos = y_proba
            auc_roc = cast(float, roc_auc_score(y_true, y_proba_pos))
        except (ValueError, IndexError):
            pass

    return MetaEvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
    )


# =============================================================================
# DATA PREPARATION
# =============================================================================


def _load_and_filter_features(model_name: str, split: str) -> pd.DataFrame:
    """Load features and filter by split."""
    features_df = get_dataset_for_model(model_name)

    if "split" in features_df.columns:
        features_df = features_df[features_df["split"] == split].copy()
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


def _align_close_to_features(close: pd.Series, features_df: pd.DataFrame) -> pd.Series:
    """Align close prices to feature window."""
    close = close.loc[
        (close.index >= features_df.index[0]) &
        (close.index <= features_df.index[-1])
    ]
    if close.index.has_duplicates:
        close = close.loc[~close.index.duplicated(keep="first")]
    return close


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


def prepare_meta_data(
    primary_model: BaseModel,
    meta_model_name: str,
    tb_params: Dict[str, Any],
    vol_window: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepare train/test data for meta model with labels.

    Returns
    -------
    Tuple containing:
        X_train, X_test, y_train, y_test, t1_train, t1_test, side_train, side_test
    """
    logger.info("Loading datasets...")

    # Load features
    features_train = _load_and_filter_features(meta_model_name, TRAIN_SPLIT_LABEL)
    features_test = _load_and_filter_features(meta_model_name, TEST_SPLIT_LABEL)

    # Handle missing values
    features_train = _handle_missing_values(features_train)
    features_test = _handle_missing_values(features_test)

    # Get feature columns
    X_train_raw = _remove_non_feature_cols(features_train)
    X_test_raw = _remove_non_feature_cols(features_test)

    logger.info(f"Features: train={len(X_train_raw)}, test={len(X_test_raw)}")

    # Load dollar bars
    bars = load_dollar_bars()
    close_raw = cast(pd.Series, bars["close"])

    # Align close to features
    close_train = _align_close_to_features(close_raw, features_train)
    close_test = _align_close_to_features(close_raw, features_test)

    volatility_train = get_daily_volatility(close_train, span=vol_window)
    volatility_test = get_daily_volatility(close_test, span=vol_window)

    # Generate PRIMARY model predictions (side)
    logger.info("Generating primary model predictions (side)...")
    side_train = pd.Series(
        primary_model.predict(X_train_raw),
        index=X_train_raw.index,
        name="side",
    )
    side_test = pd.Series(
        primary_model.predict(X_test_raw),
        index=X_test_raw.index,
        name="side",
    )

    # Filter to valid sides (+1 or -1)
    side_train = side_train.loc[side_train.isin([1, -1])]
    side_test = side_test.loc[side_test.isin([1, -1])]

    logger.info(f"Side distribution train: {side_train.value_counts().to_dict()}")
    logger.info(f"Side distribution test: {side_test.value_counts().to_dict()}")

    # Generate META labels (bin: 1 if trade correct, 0 otherwise)
    logger.info("Generating meta labels for train set...")
    events_train = get_events_meta(
        close=close_train,
        t_events=pd.DatetimeIndex(side_train.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility_train,
        max_holding=tb_params["max_holding"],
        side=side_train,
    )
    if not events_train.empty:
        events_train = get_bins(events_train, close_train)

    logger.info("Generating meta labels for test set...")
    events_test = get_events_meta(
        close=close_test,
        t_events=pd.DatetimeIndex(side_test.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility_test,
        max_holding=tb_params["max_holding"],
        side=side_test,
    )
    if not events_test.empty:
        events_test = get_bins(events_test, close_test)

    # Align features with events
    common_train = X_train_raw.index.intersection(events_train.index)
    common_test = X_test_raw.index.intersection(events_test.index)

    X_train = X_train_raw.loc[common_train]
    y_train = events_train.loc[common_train, "bin"]
    t1_train = events_train.loc[common_train, "t1"]
    side_train_aligned = side_train.loc[common_train]

    X_test = X_test_raw.loc[common_test]
    y_test = events_test.loc[common_test, "bin"]
    t1_test = events_test.loc[common_test, "t1"]
    side_test_aligned = side_test.loc[common_test]

    logger.info(f"Final samples: train={len(X_train)}, test={len(X_test)}")

    return (
        X_train, X_test,
        y_train, y_test,
        t1_train, t1_test,
        side_train_aligned, side_test_aligned,
    )


# =============================================================================
# OUT-OF-SAMPLE LABEL GENERATION
# =============================================================================


def generate_oos_labels_train(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    min_train_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Generate out-of-sample meta labels using Walk-Forward K-Fold.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray | None, np.ndarray]
        - oos_labels: Out-of-sample predictions (0 or 1)
        - oos_proba: Out-of-sample probabilities (or None)
        - coverage_mask: Boolean mask indicating which samples have OOS labels
    """
    logger.info(f"Generating OOS meta labels with Walk-Forward K-Fold ({n_splits} splits)...")

    n_samples = len(y)
    oos_labels = np.full(n_samples, np.nan)
    oos_proba = None
    coverage_mask = np.zeros(n_samples, dtype=bool)

    cv = WalkForwardKFold(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        min_train_size=min_train_size,
    )
    splits = cv.split(X, y, t1=t1)

    if len(splits) == 0:
        logger.warning("No valid splits generated.")
        return oos_labels, oos_proba, coverage_mask

    logger.info(f"Generated {len(splits)} valid Walk-Forward splits")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_start = X.index[train_idx[0]] if len(train_idx) > 0 else "N/A"
        train_end = X.index[train_idx[-1]] if len(train_idx) > 0 else "N/A"
        val_start = X.index[val_idx[0]] if len(val_idx) > 0 else "N/A"
        val_end = X.index[val_idx[-1]] if len(val_idx) > 0 else "N/A"

        logger.info(
            f"Fold {fold_idx + 1}/{len(splits)}: "
            f"train[{len(train_idx)}]={train_start} to {train_end}, "
            f"val[{len(val_idx)}]={val_start} to {val_end}"
        )

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        # Check for degenerate fold
        unique_classes = y_train_fold.unique()
        if len(unique_classes) < 2:
            logger.warning(
                f"Fold {fold_idx + 1}: Only {len(unique_classes)} class(es), skipping"
            )
            continue

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)

        # Predict
        fold_preds = model.predict(X_val_fold)
        oos_labels[val_idx] = fold_preds
        coverage_mask[val_idx] = True

        # Get probabilities if available
        fold_proba = None
        try:
            fold_proba = cast(Any, model).predict_proba(X_val_fold)
            if oos_proba is None:
                n_classes = fold_proba.shape[1]
                oos_proba = np.full((n_samples, n_classes), np.nan)
            oos_proba[val_idx] = fold_proba
        except Exception:
            pass

        fold_label_counts = pd.Series(fold_preds).value_counts().to_dict()
        logger.info(f"  -> Fold {fold_idx + 1} predictions: {fold_label_counts}")

    n_covered = coverage_mask.sum()
    coverage_pct = n_covered / n_samples * 100
    logger.info(f"OOS coverage: {n_covered}/{n_samples} ({coverage_pct:.1f}%)")

    return oos_labels, oos_proba, coverage_mask


# =============================================================================
# TRAINING PIPELINE
# =============================================================================


def train_meta_model(
    primary_model_name: str,
    meta_model_name: str,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    min_train_size: int = 100,
    output_dir: Path | None = None,
) -> MetaTrainingResult:
    """
    Full training pipeline for META model with OUT-OF-SAMPLE label generation.

    STEP 1: Load trained PRIMARY model
    STEP 2: Generate OOS meta-labels on TRAIN using Walk-Forward K-Fold
    STEP 3: Train FINAL meta model on entire TRAIN set (one-shot)
    STEP 4: Generate meta-labels on TEST using final model
    STEP 5: Save ALL outputs
    """
    if output_dir is None:
        output_dir = LABEL_META_TRAIN_DIR / f"{primary_model_name}_{meta_model_name}"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"META MODEL TRAINING: {primary_model_name} -> {meta_model_name}")
    logger.info(f"{'='*60}")

    # Load optimized parameters
    logger.info("\nLoading optimized parameters...")
    optimized = load_optimized_params(primary_model_name, meta_model_name)
    meta_model_params = optimized["meta_model_params"].copy()
    tb_params = optimized["triple_barrier_params"]

    logger.info(f"Triple Barrier: {tb_params}")
    logger.info(f"Meta model params: {meta_model_params}")

    meta_model_params["random_state"] = DEFAULT_RANDOM_STATE

    # Load models
    logger.info("\nLoading primary model...")
    primary_model = load_primary_model(primary_model_name)
    meta_model_class = load_model_class(meta_model_name)

    # Get actual primary model path (try both patterns)
    primary_model_path_main = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name / f"{primary_model_name}_final_model.joblib"
    primary_model_path_train = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name / f"{primary_model_name}_model.joblib"
    primary_model_path = primary_model_path_main if primary_model_path_main.exists() else primary_model_path_train

    # Prepare data
    (
        X_train, X_test,
        y_train, y_test,
        t1_train, t1_test,
        side_train, side_test,
    ) = prepare_meta_data(primary_model, meta_model_name, tb_params)

    # Meta-label distributions (ground truth)
    train_label_counts = y_train.value_counts().to_dict()
    test_label_counts = y_test.value_counts().to_dict()

    train_label_dist = {
        "total": len(y_train),
        "counts": {str(k): int(v) for k, v in train_label_counts.items()},
        "percentages": {str(k): v / len(y_train) * 100 for k, v in train_label_counts.items()},
    }
    test_label_dist = {
        "total": len(y_test),
        "counts": {str(k): int(v) for k, v in test_label_counts.items()},
        "percentages": {str(k): v / len(y_test) * 100 for k, v in test_label_counts.items()},
    }

    logger.info(f"\nMeta-Label Distribution (Ground Truth):")
    logger.info(f"  Train: {train_label_counts}")
    logger.info(f"  Test: {test_label_counts}")

    # =========================================================================
    # STEP 1: Generate OOS Labels on TRAIN (Walk-Forward K-Fold)
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: OOS Meta-Label Generation on TRAIN")
    logger.info(f"{'='*60}")

    oos_train_labels, oos_train_proba, train_coverage_mask = generate_oos_labels_train(
        X_train, y_train, t1_train,
        meta_model_class, meta_model_params,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        min_train_size=min_train_size,
    )

    # Evaluate OOS predictions on train
    if train_coverage_mask.any():
        y_train_covered = cast(np.ndarray, y_train.values[train_coverage_mask])
        oos_preds_covered = cast(np.ndarray, oos_train_labels[train_coverage_mask].astype(int))
        oos_proba_covered = (
            oos_train_proba[train_coverage_mask]
            if oos_train_proba is not None else None
        )

        train_metrics = compute_metrics(y_train_covered, oos_preds_covered, oos_proba_covered)
        logger.info(f"\nTrain (OOS) Metrics: {train_metrics.to_dict()}")
    else:
        logger.warning("No OOS labels generated for train set!")
        train_metrics = MetaEvaluationMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, auc_roc=None
        )

    # =========================================================================
    # STEP 2: Train FINAL Model on Entire TRAIN Set
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Train FINAL Meta Model on Entire TRAIN Set")
    logger.info(f"{'='*60}")

    final_model = meta_model_class(**meta_model_params)
    final_model.fit(X_train, y_train)
    logger.info(f"Final meta model trained on {len(X_train)} samples")

    # =========================================================================
    # STEP 3: Generate Labels on TEST Set
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3: Generate Meta-Labels on TEST Set (OOS)")
    logger.info(f"{'='*60}")

    test_preds = final_model.predict(X_test)
    test_proba = None
    try:
        test_proba = cast(Any, final_model).predict_proba(X_test)
    except Exception:
        pass

    test_metrics = compute_metrics(cast(np.ndarray, y_test.values), test_preds, test_proba)
    logger.info(f"Test Metrics: {test_metrics.to_dict()}")

    # =========================================================================
    # STEP 4: Save ALL Outputs
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 4: Save All Outputs")
    logger.info(f"{'='*60}")

    # Save OOS predictions for train
    oos_train_df = pd.DataFrame({
        "datetime": X_train.index,
        "side": side_train.values,
        "y_true_meta": y_train.values,
        "y_pred_meta_oos": oos_train_labels,
        "has_oos_label": train_coverage_mask,
    })
    if oos_train_proba is not None:
        for i in range(oos_train_proba.shape[1]):
            oos_train_df[f"proba_class_{i}"] = oos_train_proba[:, i]
    oos_train_path = output_dir / "oos_train_meta_labels.parquet"
    oos_train_df.to_parquet(oos_train_path, index=False)
    logger.info(f"OOS train labels saved: {oos_train_path}")

    # Save predictions for test
    test_df = pd.DataFrame({
        "datetime": X_test.index,
        "side": side_test.values,
        "y_true_meta": y_test.values,
        "y_pred_meta_oos": test_preds,
    })
    if test_proba is not None:
        for i in range(test_proba.shape[1]):
            test_df[f"proba_class_{i}"] = test_proba[:, i]
    test_path = output_dir / "oos_test_meta_labels.parquet"
    test_df.to_parquet(test_path, index=False)
    logger.info(f"OOS test labels saved: {test_path}")

    # Save final model
    meta_model_path = output_dir / "final_meta_model.joblib"
    final_model.save(meta_model_path)
    logger.info(f"Final meta model saved: {meta_model_path}")

    # Build result
    result = MetaTrainingResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        meta_model_params=meta_model_params,
        triple_barrier_params=tb_params,
        train_samples=len(X_train),
        test_samples=len(X_test),
        train_metrics=train_metrics.to_dict(),
        test_metrics=test_metrics.to_dict(),
        meta_label_distribution_train=train_label_dist,
        meta_label_distribution_test=test_label_dist,
        n_folds=n_splits,
        primary_model_path=str(primary_model_path),
        meta_model_path=str(meta_model_path),
        oof_predictions_path=str(oos_train_path),
        test_predictions_path=str(test_path),
    )

    results_path = output_dir / "training_results.json"
    result.save(results_path)

    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'='*60}")

    return result


# =============================================================================
# INTERACTIVE CLI
# =============================================================================


def get_available_meta_optimizations() -> List[Tuple[str, str]]:
    """Get list of available meta model optimizations."""
    if not LABEL_META_OPTI_DIR.exists():
        return []

    optimizations = []
    for f in LABEL_META_OPTI_DIR.glob("*_optimization.json"):
        parts = f.stem.replace("_optimization", "").split("_")
        if len(parts) >= 2:
            # Assume format: primary_meta
            primary = parts[0]
            meta = "_".join(parts[1:])
            optimizations.append((primary, meta))
    return optimizations


def select_models() -> Tuple[str, str]:
    """Interactive selection of primary and meta models."""
    available = get_available_meta_optimizations()

    print("\n" + "=" * 60)
    print("META MODEL TRAINING (OOS Label Generation)")
    print("=" * 60)
    print("\nThe meta model filters false positives from the primary model.")
    print("\nOptimizations disponibles:")
    print("-" * 40)

    if not available:
        print("Aucune optimisation meta trouvee!")
        print("Lancer d'abord: python -m src.labelling.label_meta.opti")
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


def print_results(result: MetaTrainingResult) -> None:
    """Print formatted results."""
    print("\n" + "=" * 60)
    print(f"RESULTATS: {result.primary_model_name} -> {result.meta_model_name}")
    print("=" * 60)

    print(f"\nEchantillons: Train={result.train_samples}, Test={result.test_samples}")

    print("\n--- Distribution Meta-Labels ---")
    print("Train:")
    for label, count in result.meta_label_distribution_train["counts"].items():
        pct = result.meta_label_distribution_train["percentages"][label]
        print(f"  Meta={label}: {count} ({pct:.1f}%)")
    print("Test:")
    for label, count in result.meta_label_distribution_test["counts"].items():
        pct = result.meta_label_distribution_test["percentages"][label]
        print(f"  Meta={label}: {count} ({pct:.1f}%)")

    print("\n--- Metriques de Performance ---")
    print("Train (OOS - Walk-Forward K-Fold):")
    for metric, value in result.train_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    print("Test (OOS):")
    for metric, value in result.test_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    print("\n--- Fichiers Sauvegardes ---")
    print(f"  Primary model: {result.primary_model_path}")
    print(f"  Meta model: {result.meta_model_path}")
    print(f"  OOS Train: {result.oof_predictions_path}")
    print(f"  OOS Test: {result.test_predictions_path}")

    print("=" * 60)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    primary_model_name, meta_model_name = select_models()
    print(f"\nSelectionne: Primary={primary_model_name}, Meta={meta_model_name}")

    n_splits = int(input("Nombre de folds K-Fold [5]: ").strip() or "5")
    embargo_input = input("Embargo (%) [1.0]: ").strip() or "1.0"
    embargo_pct = float(embargo_input) / 100
    min_train_size = int(input("Taille min train [100]: ").strip() or "100")

    print("\n" + "-" * 40)
    print(f"Primary: {primary_model_name}")
    print(f"Meta: {meta_model_name}")
    print(f"K-Fold: {n_splits}, Embargo: {embargo_pct * 100:.1f}%")
    print("-" * 40)

    confirm = input("\nLancer? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    result = train_meta_model(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        min_train_size=min_train_size,
    )

    print_results(result)


if __name__ == "__main__":
    main()
