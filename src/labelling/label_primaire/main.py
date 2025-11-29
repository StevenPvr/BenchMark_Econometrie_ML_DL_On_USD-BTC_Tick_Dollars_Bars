"""
Primary Model Training and Evaluation with Out-of-Fold Predictions.

This module implements a full training pipeline that:
1. Asks user to select a model interactively
2. Loads optimized hyperparameters from config files
3. Generates out-of-fold predictions on train set using PurgedKFold
4. Trains final model on entire train set
5. Generates predictions on test set
6. Evaluates and saves all outputs

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado, Chapter 7
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import DEFAULT_RANDOM_STATE, TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL
from src.labelling.label_primaire.opti import get_events_primary
from src.labelling.label_primaire.utils import (
    MODEL_REGISTRY,
    get_daily_volatility,
    get_dataset_for_model,
    load_dollar_bars,
    load_model_class,
)
from src.model.base import BaseModel
from src.path import (
    DOLLAR_BARS_PARQUET,
    LABEL_PRIMAIRE_OPTI_DIR,
    LABEL_PRIMAIRE_TRAIN_DIR,
    DATASET_FEATURES_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_LABEL_PARQUET,
    DATASET_FEATURES_LSTM_LABEL_PARQUET,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PURGED K-FOLD CROSS-VALIDATION (De Prado Chapter 7)
# =============================================================================


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for time series with embargo.

    Implements the PurgedKFold methodology from De Prado's Chapter 7.
    Prevents data leakage by:
    1. Purging: Removing training samples whose labels overlap validation period
    2. Embargo: Adding a gap between training and validation sets

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    purge_pct : float, default=0.0
        Percentage of samples to purge after each training set.
    embargo_pct : float, default=0.01
        Percentage of samples to embargo between train and validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.0,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        t1: pd.Series | None = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation indices for each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Features with datetime index.
        y : pd.Series, optional
            Labels (not used but kept for sklearn compatibility).
        t1 : pd.Series, optional
            Series mapping start time to end time of each label.
            Used for purging overlapping samples.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            (train_indices, validation_indices) for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate sizes
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        splits = []

        for fold_idx in range(self.n_splits):
            # Validation indices for this fold
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < self.n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]

            # Training indices: everything except validation
            train_indices = np.concatenate([
                indices[:val_start],
                indices[val_end:],
            ])

            # Apply embargo: remove samples too close to validation start
            if embargo_size > 0 and val_start > 0:
                # Find indices just before validation that are within embargo distance
                embargo_start = max(0, val_start - embargo_size)
                embargo_mask = (train_indices >= embargo_start) & (train_indices < val_start)
                train_indices = train_indices[~embargo_mask]

            # Apply purging: remove training samples whose t1 extends into validation
            if t1 is not None:
                train_indices = self._apply_purge(
                    train_indices, val_indices, X, t1
                )

            if len(train_indices) > 0 and len(val_indices) > 0:
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

        # For each training sample, check if its t1 extends into validation
        train_times = X.index[train_indices]
        t1_train = t1.reindex(train_times)

        # Remove samples where t1 >= validation start
        overlap_mask = t1_train.notna() & (t1_train >= val_start_time)
        purged_indices = train_indices[~overlap_mask.to_numpy()]

        n_purged = len(train_indices) - len(purged_indices)
        if n_purged > 0:
            logger.debug(f"Purged {n_purged} samples with overlapping labels")

        return purged_indices


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for classification."""

    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    auc_roc_ovr: float | None  # One-vs-Rest for multiclass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "auc_roc_ovr": self.auc_roc_ovr,
        }


@dataclass
class TrainingResult:
    """Complete training result with all outputs."""

    model_name: str
    model_params: Dict[str, Any]
    triple_barrier_params: Dict[str, Any]
    train_samples: int
    test_samples: int
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    label_distribution_train: Dict[str, Any]
    label_distribution_test: Dict[str, Any]
    n_folds: int
    model_path: str
    oof_predictions_path: str
    test_predictions_path: str
    labeled_dataset_path: str  # Path to labeled dataset in data/features/
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "triple_barrier_params": self.triple_barrier_params,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "label_distribution_train": self.label_distribution_train,
            "label_distribution_test": self.label_distribution_test,
            "n_folds": self.n_folds,
            "model_path": self.model_path,
            "oof_predictions_path": self.oof_predictions_path,
            "test_predictions_path": self.test_predictions_path,
            "labeled_dataset_path": self.labeled_dataset_path,
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
    model_name: str,
    opti_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Load optimized parameters from the optimization step.

    Parameters
    ----------
    model_name : str
        Name of the model.
    opti_dir : Path, optional
        Directory containing optimization results.

    Returns
    -------
    Dict[str, Any]
        Dictionary with model_params and triple_barrier_params.
    """
    if opti_dir is None:
        opti_dir = LABEL_PRIMAIRE_OPTI_DIR

    opti_file = opti_dir / f"{model_name}_optimization.json"

    if not opti_file.exists():
        raise FileNotFoundError(
            f"Optimization results not found: {opti_file}\n"
            f"Run optimization first: python -m src.labelling.label_primaire.opti"
        )

    with open(opti_file, "r", encoding="utf-8") as f:
        opti_results = json.load(f)

    return {
        "model_params": opti_results["best_params"],
        "triple_barrier_params": opti_results["best_triple_barrier_params"],
        "best_score": opti_results.get("best_score", None),
        "metric": opti_results.get("metric", "mcc_weighted"),
    }


# =============================================================================
# EVALUATION
# =============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> EvaluationMetrics:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray, optional
        Predicted probabilities for AUC-ROC.

    Returns
    -------
    EvaluationMetrics
        Computed metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # AUC-ROC for multiclass (One-vs-Rest)
    auc_roc = None
    if y_proba is not None:
        try:
            # Check if we have enough classes
            unique_true = np.unique(y_true)
            if len(unique_true) >= 2 and y_proba.shape[1] >= 2:
                auc_roc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
        except (ValueError, IndexError):
            pass

    return EvaluationMetrics(
        accuracy=accuracy,
        precision_weighted=precision,
        recall_weighted=recall,
        f1_weighted=f1,
        auc_roc_ovr=auc_roc,
    )


# =============================================================================
# DATA PREPARATION
# =============================================================================


def _load_and_filter_features(model_name: str, split: str) -> pd.DataFrame:
    """Load features and filter by split (same logic as opti.py)."""
    features_df = get_dataset_for_model(model_name)

    # Filter to split
    if "split" in features_df.columns:
        features_df = features_df[features_df["split"] == split].copy()
        features_df = features_df.drop(columns=["split"])

    # Set datetime index
    if "datetime_close" in features_df.columns:
        features_df = features_df.set_index("datetime_close")
    features_df = features_df.sort_index()

    # Remove duplicates
    if features_df.index.has_duplicates:
        features_df = features_df[~features_df.index.duplicated(keep="first")]

    return features_df


def _remove_non_feature_cols(features_df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns (same as opti.py)."""
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close",
        "threshold_used", "log_return", "split", "label",
    ]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    return features_df[feature_cols]


def _align_close_to_features(close: pd.Series, features_df: pd.DataFrame) -> pd.Series:
    """Align close prices to feature window (same as opti.py)."""
    close = close.loc[
        (close.index >= features_df.index[0]) &
        (close.index <= features_df.index[-1])
    ]
    if close.index.has_duplicates:
        close = close.loc[~close.index.duplicated(keep="first")]
    return close


def _handle_missing_values(features_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Handle missing values by filling with median (same as opti.py)."""
    for col in features_df.columns:
        if features_df[col].isna().any():
            median_val = features_df[col].median()
            if pd.isna(median_val):
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = features_df[col].fillna(median_val)
    return features_df


def prepare_data(
    model_name: str,
    tb_params: Dict[str, Any],
    vol_window: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load and prepare train/test data with labels.
    Uses the same logic as opti.py for data preparation.

    Returns
    -------
    Tuple containing:
        X_train, X_test, y_train, y_test, t1_train, t1_test
    """
    logger.info("Loading datasets...")

    # Load and filter features for train and test (same logic as opti.py)
    features_train = _load_and_filter_features(model_name, TRAIN_SPLIT_LABEL)
    features_test = _load_and_filter_features(model_name, TEST_SPLIT_LABEL)

    # Handle missing values
    features_train = _handle_missing_values(features_train, model_name)
    features_test = _handle_missing_values(features_test, model_name)

    # Remove non-feature columns
    features_train = _remove_non_feature_cols(features_train)
    features_test = _remove_non_feature_cols(features_test)

    logger.info(f"Features: train={len(features_train)}, test={len(features_test)}")

    # Load dollar bars (same as opti.py)
    bars = load_dollar_bars()
    close_raw = bars["close"]

    # Align close to train features window (same as opti.py)
    close_train = _align_close_to_features(close_raw, features_train)
    volatility_train = get_daily_volatility(close_train, span=vol_window)

    # Align close to test features window
    close_test = _align_close_to_features(close_raw, features_test)
    volatility_test = get_daily_volatility(close_test, span=vol_window)

    # Generate labels for train (same as opti.py)
    logger.info("Generating labels for train set...")
    events_train = get_events_primary(
        close=close_train,
        t_events=pd.DatetimeIndex(features_train.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility_train,
        max_holding=tb_params["max_holding"],
        min_return=tb_params.get("min_return", 0.0),
    )

    # Generate labels for test
    logger.info("Generating labels for test set...")
    events_test = get_events_primary(
        close=close_test,
        t_events=pd.DatetimeIndex(features_test.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility_test,
        max_holding=tb_params["max_holding"],
        min_return=tb_params.get("min_return", 0.0),
    )

    # Align features with labels
    common_train = features_train.index.intersection(events_train.index)
    common_test = features_test.index.intersection(events_test.index)

    X_train = features_train.loc[common_train]
    y_train = events_train.loc[common_train, "label"]
    t1_train = events_train.loc[common_train, "t1"]

    X_test = features_test.loc[common_test]
    y_test = events_test.loc[common_test, "label"]
    t1_test = events_test.loc[common_test, "t1"]

    logger.info(f"Final samples: train={len(X_train)}, test={len(X_test)}")

    return X_train, X_test, y_train, y_test, t1_train, t1_test


# =============================================================================
# LABELED DATASET SAVING
# =============================================================================


def get_labeled_dataset_path(model_name: str) -> Path:
    """Get the path for saving labeled dataset based on model type."""
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]

    path_map = {
        "tree": DATASET_FEATURES_LABEL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_LABEL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_LABEL_PARQUET,
    }

    return path_map.get(dataset_type, DATASET_FEATURES_LABEL_PARQUET)


def save_labeled_dataset(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    t1_train: pd.Series,
    t1_test: pd.Series,
) -> Path:
    """
    Save the complete labeled dataset (train + test) to data/features/.

    The dataset includes:
    - All original features
    - 'label' column with triple barrier labels (-1, 0, 1)
    - 't1' column with label end times
    - 'split' column to distinguish train/test

    Parameters
    ----------
    model_name : str
        Name of the model (determines which dataset file to save).
    X_train, X_test : pd.DataFrame
        Feature DataFrames for train and test.
    y_train, y_test : pd.Series
        Labels for train and test.
    t1_train, t1_test : pd.Series
        Label end times for train and test.

    Returns
    -------
    Path
        Path to the saved labeled dataset.
    """
    # Get original dataset to preserve all columns
    from src.labelling.label_primaire.utils import get_dataset_for_model

    original_df = get_dataset_for_model(model_name)

    # Set datetime index if needed
    if "datetime_close" in original_df.columns:
        original_df = original_df.set_index("datetime_close")

    # Create labeled train dataset
    train_labeled = original_df.loc[X_train.index].copy()
    train_labeled["label"] = y_train
    train_labeled["t1"] = t1_train
    train_labeled["split"] = TRAIN_SPLIT_LABEL

    # Create labeled test dataset
    test_labeled = original_df.loc[X_test.index].copy()
    test_labeled["label"] = y_test
    test_labeled["t1"] = t1_test
    test_labeled["split"] = TEST_SPLIT_LABEL

    # Combine train and test
    labeled_df = pd.concat([train_labeled, test_labeled], axis=0)
    labeled_df = labeled_df.sort_index()

    # Reset index to have datetime_close as column
    labeled_df = labeled_df.reset_index()
    labeled_df = labeled_df.rename(columns={"index": "datetime_close"})

    # Get output path
    output_path = get_labeled_dataset_path(model_name)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(output_path, index=False)

    logger.info(f"Labeled dataset saved: {output_path}")
    logger.info(f"  Total samples: {len(labeled_df)}")
    logger.info(f"  Train: {len(train_labeled)}, Test: {len(test_labeled)}")
    logger.info(f"  Label distribution: {labeled_df['label'].value_counts().to_dict()}")

    return output_path


# =============================================================================
# OUT-OF-FOLD PREDICTIONS
# =============================================================================


def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Generate out-of-fold predictions using PurgedKFold.

    Each sample in the training set gets a prediction from a model
    that was trained without seeing that sample (or temporally overlapping samples).

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    t1 : pd.Series
        End time of each label (for purging).
    model_class : Type[BaseModel]
        Model class to use.
    model_params : Dict[str, Any]
        Model hyperparameters.
    n_splits : int
        Number of CV folds.
    embargo_pct : float
        Embargo percentage.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray | None]
        (predictions, probabilities)
    """
    logger.info(f"Generating out-of-fold predictions with {n_splits} folds...")

    # Initialize arrays for predictions
    oof_preds = np.full(len(y), np.nan)
    oof_proba = None

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    splits = cv.split(X, y, t1=t1)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold_idx + 1}/{len(splits)}: "
                    f"train={len(train_idx)}, val={len(val_idx)}")

        # Get fold data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        # Check for degenerate fold
        unique_classes = y_train_fold.unique()
        if len(unique_classes) < 2:
            logger.warning(f"Fold {fold_idx + 1}: Only {len(unique_classes)} class(es) in training, skipping")
            continue

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        fold_preds = model.predict(X_val_fold)
        oof_preds[val_idx] = fold_preds

        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                fold_proba = model.predict_proba(X_val_fold)
                if oof_proba is None:
                    # Initialize probability array
                    n_classes = fold_proba.shape[1]
                    oof_proba = np.full((len(y), n_classes), np.nan)
                oof_proba[val_idx] = fold_proba
            except Exception as e:
                logger.debug(f"Could not get probabilities: {e}")

    # Log coverage
    valid_preds = ~np.isnan(oof_preds)
    coverage = valid_preds.sum() / len(oof_preds) * 100
    logger.info(f"Out-of-fold coverage: {coverage:.1f}%")

    return oof_preds, oof_proba


# =============================================================================
# TRAINING PIPELINE
# =============================================================================


def train_and_evaluate(
    model_name: str,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    output_dir: Path | None = None,
) -> TrainingResult:
    """
    Full training pipeline with out-of-fold predictions.

    Steps:
    1. Load optimized hyperparameters
    2. Prepare train/test data with labels
    3. Generate out-of-fold predictions on train (using PurgedKFold)
    4. Train final model on entire train set
    5. Generate predictions on test set
    6. Evaluate and save outputs

    Parameters
    ----------
    model_name : str
        Name of the model to train.
    n_splits : int
        Number of CV folds for OOF predictions.
    embargo_pct : float
        Embargo percentage for CV.
    output_dir : Path, optional
        Directory to save outputs.

    Returns
    -------
    TrainingResult
        Complete training results.
    """
    if output_dir is None:
        output_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"TRAINING: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Load optimized parameters
    logger.info("Loading optimized parameters...")
    optimized = load_optimized_params(model_name)
    model_params = optimized["model_params"].copy()
    tb_params = optimized["triple_barrier_params"]

    logger.info(f"Triple Barrier: {tb_params}")
    logger.info(f"Model params: {model_params}")

    # Add random state
    model_params["random_state"] = DEFAULT_RANDOM_STATE

    # Load model class
    model_class = load_model_class(model_name)

    # Prepare data
    X_train, X_test, y_train, y_test, t1_train, t1_test = prepare_data(
        model_name, tb_params
    )

    # Label distributions
    train_label_counts = y_train.value_counts().to_dict()
    test_label_counts = y_test.value_counts().to_dict()

    train_label_dist = {
        "total": len(y_train),
        "counts": {str(k): v for k, v in train_label_counts.items()},
        "percentages": {str(k): v / len(y_train) * 100 for k, v in train_label_counts.items()},
    }
    test_label_dist = {
        "total": len(y_test),
        "counts": {str(k): v for k, v in test_label_counts.items()},
        "percentages": {str(k): v / len(y_test) * 100 for k, v in test_label_counts.items()},
    }

    logger.info(f"Train label distribution: {train_label_counts}")
    logger.info(f"Test label distribution: {test_label_counts}")

    # =========================================================================
    # STEP 0: Save Labeled Dataset to data/features/
    # =========================================================================
    logger.info("\n--- Saving Labeled Dataset ---")
    labeled_dataset_path = save_labeled_dataset(
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        t1_train=t1_train,
        t1_test=t1_test,
    )

    # =========================================================================
    # STEP 1: Out-of-Fold Predictions on Train
    # =========================================================================
    logger.info("\n--- Out-of-Fold Predictions (Train) ---")
    oof_preds, oof_proba = generate_oof_predictions(
        X_train, y_train, t1_train,
        model_class, model_params,
        n_splits=n_splits, embargo_pct=embargo_pct
    )

    # Evaluate OOF predictions (only on samples that have predictions)
    valid_mask = ~np.isnan(oof_preds)
    y_train_valid = y_train.values[valid_mask]
    oof_preds_valid = oof_preds[valid_mask].astype(int)
    oof_proba_valid = oof_proba[valid_mask] if oof_proba is not None else None

    train_metrics = compute_metrics(y_train_valid, oof_preds_valid, oof_proba_valid)
    logger.info(f"Train (OOF) Metrics: {train_metrics.to_dict()}")

    # =========================================================================
    # STEP 2: Train Final Model on Entire Train Set
    # =========================================================================
    logger.info("\n--- Training Final Model ---")
    final_model = model_class(**model_params)
    final_model.fit(X_train, y_train)
    logger.info("Final model trained on entire train set")

    # =========================================================================
    # STEP 3: Predictions on Test Set
    # =========================================================================
    logger.info("\n--- Test Set Predictions ---")
    test_preds = final_model.predict(X_test)
    test_proba = None
    if hasattr(final_model, "predict_proba"):
        try:
            test_proba = final_model.predict_proba(X_test)
        except Exception:
            pass

    test_metrics = compute_metrics(y_test.values, test_preds, test_proba)
    logger.info(f"Test Metrics: {test_metrics.to_dict()}")

    # =========================================================================
    # STEP 4: Save Outputs
    # =========================================================================
    logger.info("\n--- Saving Outputs ---")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "datetime": X_train.index,
        "y_true": y_train.values,
        "y_pred": oof_preds,
    })
    if oof_proba is not None:
        for i in range(oof_proba.shape[1]):
            oof_df[f"proba_class_{i}"] = oof_proba[:, i]
    oof_path = output_dir / f"{model_name}_oof_predictions.parquet"
    oof_df.to_parquet(oof_path, index=False)
    logger.info(f"OOF predictions saved: {oof_path}")

    # Save test predictions
    test_df = pd.DataFrame({
        "datetime": X_test.index,
        "y_true": y_test.values,
        "y_pred": test_preds,
    })
    if test_proba is not None:
        for i in range(test_proba.shape[1]):
            test_df[f"proba_class_{i}"] = test_proba[:, i]
    test_path = output_dir / f"{model_name}_test_predictions.parquet"
    test_df.to_parquet(test_path, index=False)
    logger.info(f"Test predictions saved: {test_path}")

    # Save final model
    model_path = output_dir / f"{model_name}_final_model.joblib"
    final_model.save(model_path)

    # Build and save results
    result = TrainingResult(
        model_name=model_name,
        model_params=model_params,
        triple_barrier_params=tb_params,
        train_samples=len(X_train),
        test_samples=len(X_test),
        train_metrics=train_metrics.to_dict(),
        test_metrics=test_metrics.to_dict(),
        label_distribution_train=train_label_dist,
        label_distribution_test=test_label_dist,
        n_folds=n_splits,
        model_path=str(model_path),
        oof_predictions_path=str(oof_path),
        test_predictions_path=str(test_path),
        labeled_dataset_path=str(labeled_dataset_path),
    )

    results_path = output_dir / f"{model_name}_training_results.json"
    result.save(results_path)

    return result


# =============================================================================
# INTERACTIVE CLI
# =============================================================================


def get_available_optimized_models() -> List[str]:
    """Get list of models that have been optimized."""
    available = []
    for model_name in MODEL_REGISTRY.keys():
        opti_file = LABEL_PRIMAIRE_OPTI_DIR / f"{model_name}_optimization.json"
        if opti_file.exists():
            available.append(model_name)
    return available


def select_model() -> str:
    """Interactive model selection."""
    models = list(MODEL_REGISTRY.keys())
    optimized = get_available_optimized_models()

    print("\n" + "=" * 60)
    print("LABEL PRIMAIRE - TRAINING WITH OUT-OF-FOLD")
    print("=" * 60)
    print("\nThis pipeline:")
    print("  1. Generates out-of-fold predictions on TRAIN (no leakage)")
    print("  2. Trains final model on entire TRAIN set")
    print("  3. Generates predictions on TEST set")
    print("  4. Evaluates and saves all outputs")
    print("\nModeles disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        status = "[optimise]" if model in optimized else "[non optimise]"
        info = MODEL_REGISTRY[model]
        dataset_type = info["dataset"]
        print(f"  {i}. {model:<15} ({dataset_type}) {status}")

    print("-" * 40)

    if not optimized:
        print("\nAucun modele optimise. Lancer d'abord l'optimisation:")
        print("  python -m src.labelling.label_primaire.opti")
        sys.exit(1)

    while True:
        try:
            choice = input("\nChoisir le modele (numero ou nom): ").strip()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                else:
                    print(f"Numero invalide. Choisir entre 1 et {len(models)}")
                    continue
            elif choice.lower() in models:
                selected = choice.lower()
            else:
                print(f"Modele inconnu: {choice}")
                continue

            # Check if optimized
            if selected not in optimized:
                print(f"Le modele '{selected}' n'a pas ete optimise.")
                print("Lancer d'abord: python -m src.labelling.label_primaire.opti")
                continue

            return selected

        except KeyboardInterrupt:
            print("\nAnnule.")
            sys.exit(0)


def print_results(result: TrainingResult) -> None:
    """Print formatted results."""
    print("\n" + "=" * 60)
    print(f"RESULTATS: {result.model_name.upper()}")
    print("=" * 60)

    print(f"\nEchantillons: Train={result.train_samples}, Test={result.test_samples}")

    print("\n--- Distribution des Labels ---")
    print("Train:")
    for label, count in result.label_distribution_train["counts"].items():
        pct = result.label_distribution_train["percentages"][label]
        print(f"  Label {label}: {count} ({pct:.1f}%)")
    print("Test:")
    for label, count in result.label_distribution_test["counts"].items():
        pct = result.label_distribution_test["percentages"][label]
        print(f"  Label {label}: {count} ({pct:.1f}%)")

    print("\n--- Metriques de Performance ---")
    print("Train (Out-of-Fold):")
    for metric, value in result.train_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    print("Test:")
    for metric, value in result.test_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    print("\n--- Fichiers Sauvegardes ---")
    print(f"  Modele: {result.model_path}")
    print(f"  Predictions OOF: {result.oof_predictions_path}")
    print(f"  Predictions Test: {result.test_predictions_path}")

    print("=" * 60)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Interactive model selection
    model_name = select_model()
    print(f"\nModele selectionne: {model_name}")

    # Get CV parameters
    n_splits = int(input("Nombre de folds K-Fold [5]: ").strip() or "5")
    embargo_input = input("Embargo (%) [1.0]: ").strip() or "1.0"
    embargo_pct = float(embargo_input) / 100

    print("\n" + "-" * 40)
    print(f"Modele: {model_name}")
    print(f"K-Fold splits: {n_splits}")
    print(f"Embargo: {embargo_pct * 100:.1f}%")
    print("-" * 40)

    confirm = input("\nLancer l'entrainement? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    # Run training
    result = train_and_evaluate(
        model_name=model_name,
        n_splits=n_splits,
        embargo_pct=embargo_pct,
    )

    # Print results
    print_results(result)


if __name__ == "__main__":
    main()
