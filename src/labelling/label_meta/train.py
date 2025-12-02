"""
Meta Model Training.

This module implements the META model training pipeline that AVOIDS DATA LEAKAGE.
The meta model learns to FILTER false positives from the primary model.

Pipeline:
1. Load trained PRIMARY model (generates direction: +1 Long, -1 Short)
2. Load dataset with OOF predictions from primary model (avoids leakage)
3. Calculate meta-labels: meta_label = 1 if (true_label × primary_prediction) > 0 else 0
4. Train META model on entire TRAIN set (one-shot)
5. Evaluate on TEST set
6. Save model and predictions

Note: No K-fold needed for meta model as it doesn't generate labels for downstream use.
It uses OOF predictions from primary model, so no additional leakage prevention needed.

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
    get_labeled_dataset_for_primary_model,
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
        "triple_barrier_params": opti_results.get("best_triple_barrier_params", {}),
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
    """Load features with OOF predictions and filter by split."""
    features_df = get_labeled_dataset_for_primary_model(model_name)

    if "split" in features_df.columns:
        features_df = features_df[features_df["split"] == split].copy()
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
    primary_model_name: str,
    filter_neutral_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare TRAIN data only for meta model.

    Meta labels are computed by comparing:
    - Primary model OOF predictions (column "prediction")
    - True triple barrier labels (column "label")

    Meta label = 1 if (label × prediction) > 0 (primary was correct), else 0

    Returns
    -------
    Tuple containing:
        X_train, y_train
    """
    logger.info("Loading train dataset...")

    # Load features from PRIMARY model dataset (contains OOF predictions)
    features_train = _load_and_filter_features(primary_model_name, TRAIN_SPLIT_LABEL)

    # Handle missing values
    features_train = _handle_missing_values(features_train)

    if "label" not in features_train.columns:
        raise ValueError("Train dataset must contain 'label' column (triple barrier labels).")

    if "prediction" not in features_train.columns:
        raise ValueError(
            "Train dataset must contain 'prediction' column (OOF predictions from primary model).\n"
            "Train the primary model first: python -m src.labelling.label_primaire.train"
        )

    # Calculate meta labels for TRAIN using OOF predictions
    label_train = features_train["label"].to_numpy()
    prediction_train = features_train["prediction"].to_numpy()

    y_train_meta = np.where(
        label_train * prediction_train > 0,
        1,  # Primary was correct
        0   # Primary was incorrect
    )

    # Filter neutral labels if configured
    if filter_neutral_labels:
        neutral_mask = label_train == 0
        n_neutral = neutral_mask.sum()
        if n_neutral > 0:
            logger.info(f"Filtering {n_neutral} neutral labels from train")
            valid_mask = ~neutral_mask
            features_train = cast(pd.DataFrame, features_train[valid_mask].copy())
            y_train_meta = y_train_meta[valid_mask]

    y_train = pd.Series(y_train_meta, index=features_train.index, name="meta_label")

    # Add 'side' as a feature (using OOF predictions from primary model)
    features_train["side"] = features_train["prediction"].copy()

    # Get feature columns (KEEP 'side' column for meta model)
    X_train = _remove_non_feature_cols(features_train, keep_side=True)

    # Check for NaN values and identify problematic columns
    nan_mask = cast(pd.Series, X_train.isna().any(axis=1))
    n_nan = nan_mask.sum()
    if n_nan > 0:
        logger.warning(f"Found {n_nan} rows with NaN values")
        # Identify which columns have NaN
        nan_cols = X_train.columns[X_train.isna().any()].tolist()
        logger.warning(f"Columns with NaN: {nan_cols}")
        for col in nan_cols:
            n_nan_col = X_train[col].isna().sum()
            logger.warning(f"  {col}: {n_nan_col} NaN values ({n_nan_col/len(X_train)*100:.2f}%)")

        # Check if it's the 'side' column (predictions)
        if 'side' in nan_cols:
            logger.error("'side' column has NaN! This should not happen on TRAIN split.")
            logger.error("Checking 'prediction' column in features_train...")
            logger.error(f"prediction NaN count: {features_train['prediction'].isna().sum()}")

        # Drop rows with NaN for now
        logger.warning(f"Dropping {n_nan} rows with NaN values")
        X_train = cast(pd.DataFrame, X_train[~nan_mask].copy())
        y_train = cast(pd.Series, y_train[~nan_mask])

    logger.info(f"Features: train={len(X_train)}")
    logger.info(f"Meta label distribution train: {y_train.value_counts().to_dict()}")

    return X_train, y_train


# =============================================================================
# TRAINING PIPELINE
# =============================================================================


def train_meta_model(
    primary_model_name: str,
    meta_model_name: str,
    output_dir: Path | None = None,
) -> MetaTrainingResult:
    """
    Full training pipeline for META model.

    STEP 1: Load trained PRIMARY model
    STEP 2: Load dataset with OOF predictions from primary model
    STEP 3: Calculate meta-labels (using OOF predictions to avoid leakage)
    STEP 4: Train meta model on entire TRAIN set (one-shot)
    STEP 5: Evaluate on TEST set
    STEP 6: Save model and predictions

    Note: No K-fold needed as meta model uses OOF predictions from primary.
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
    tb_params = optimized.get("triple_barrier_params", {})

    logger.info(f"Meta model params: {meta_model_params}")

    meta_model_params["random_state"] = DEFAULT_RANDOM_STATE

    # Disable normalization for linear models (data already z-scored)
    if meta_model_name in ["logistic", "ridge"]:
        meta_model_params["normalize"] = False
        logger.info("Normalization disabled (data already z-scored)")

    # Load models
    logger.info("\nLoading primary model...")
    primary_model = load_primary_model(primary_model_name)
    meta_model_class = load_model_class(meta_model_name)

    # Get actual primary model path (try both patterns)
    primary_model_path_main = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name / f"{primary_model_name}_final_model.joblib"
    primary_model_path_train = LABEL_PRIMAIRE_TRAIN_DIR / primary_model_name / f"{primary_model_name}_model.joblib"
    primary_model_path = primary_model_path_main if primary_model_path_main.exists() else primary_model_path_train

    # Prepare data (TRAIN only)
    X_train, y_train = prepare_meta_data(primary_model_name)

    # Meta-label distributions
    train_label_counts = y_train.value_counts().to_dict()

    train_label_dist = {
        "total": len(y_train),
        "counts": {str(k): int(v) for k, v in train_label_counts.items()},
        "percentages": {str(k): v / len(y_train) * 100 for k, v in train_label_counts.items()},
    }

    logger.info(f"\nMeta-Label Distribution:")
    logger.info(f"  Train: {train_label_counts}")

    # =========================================================================
    # STEP 1: Train Meta Model on TRAIN Set
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: Train Meta Model on TRAIN Set")
    logger.info(f"{'='*60}")

    meta_model = meta_model_class(**meta_model_params)
    meta_model.fit(X_train, y_train)
    logger.info(f"Meta model trained on {len(X_train)} samples")

    # =========================================================================
    # STEP 2: Save Model
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Save Model")
    logger.info(f"{'='*60}")

    # Save model with specific meta model name
    meta_model_path = output_dir / f"{meta_model_name}_meta_model.joblib"
    meta_model.save(meta_model_path)
    logger.info(f"Meta model saved: {meta_model_path}")

    # Empty metrics (evaluation is done in eval.py)
    train_metrics = MetaEvaluationMetrics(
        accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, auc_roc=None
    )

    # Build result
    result = MetaTrainingResult(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
        meta_model_params=meta_model_params,
        triple_barrier_params=tb_params,
        train_samples=len(X_train),
        test_samples=0,
        train_metrics=train_metrics.to_dict(),
        test_metrics={},
        meta_label_distribution_train=train_label_dist,
        meta_label_distribution_test={},
        n_folds=0,  # No K-fold used
        primary_model_path=str(primary_model_path),
        meta_model_path=str(meta_model_path),
        oof_predictions_path="",  # No predictions saved (done in eval.py)
        test_predictions_path="",  # No predictions saved (done in eval.py)
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
    print("META MODEL TRAINING")
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
    print(f"TRAINING COMPLETE: {result.primary_model_name} -> {result.meta_model_name}")
    print("=" * 60)

    print(f"\nEchantillons: Train={result.train_samples}")

    print("\n--- Distribution Meta-Labels (Train) ---")
    for label, count in result.meta_label_distribution_train["counts"].items():
        pct = result.meta_label_distribution_train["percentages"][label]
        print(f"  Meta={label}: {count} ({pct:.1f}%)")

    print("\n--- Fichiers Sauvegardes ---")
    print(f"  Primary model: {result.primary_model_path}")
    print(f"  Meta model: {result.meta_model_path}")

    print("\nPour evaluer le modele, lancer:")
    print("  python -m src.labelling.label_meta.eval")

    print("=" * 60)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    primary_model_name, meta_model_name = select_models()
    print(f"\nSelectionne: Primary={primary_model_name}, Meta={meta_model_name}")

    print("\n" + "-" * 40)
    print(f"Primary: {primary_model_name}")
    print(f"Meta: {meta_model_name}")
    print("-" * 40)

    confirm = input("\nLancer? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    result = train_meta_model(
        primary_model_name=primary_model_name,
        meta_model_name=meta_model_name,
    )

    print_results(result)


if __name__ == "__main__":
    main()
