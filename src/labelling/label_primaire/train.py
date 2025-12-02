"""
Training module for Primary Label models.

This module trains a primary model using optimized parameters from the
optimization step. It:
1. Loads optimized triple barrier and model parameters
2. Generates labels on the training set
3. Trains the model
4. Saves the trained model for evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import concurrent.futures
import gc
import json
import logging
import multiprocessing
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]


# =============================================================================
# WALK-FORWARD CV FOR OOF PREDICTIONS
# =============================================================================


class WalkForwardCV:
    """Walk-Forward Cross-Validation with purging and embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 500,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        events: pd.DataFrame,
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """Generate train/val indices with purging."""
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        base_splitter = TimeSeriesSplit(n_splits=self.n_splits)

        splits = []
        for train_idx, val_idx in base_splitter.split(np.arange(n_samples)):
            split = self._process_split(train_idx, val_idx, X, events, embargo_size)
            if split is not None:
                splits.append(split)

        return splits

    def _process_split(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        events: pd.DataFrame,
        embargo_size: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Process a single CV split with purging and embargo."""
        if len(train_idx) < self.min_train_size or len(val_idx) == 0:
            return None

        # Purge: remove overlapping training samples
        if "t1" in events.columns:
            train_idx = self._apply_purging(train_idx, val_idx, X, events)

        # Embargo: gap between train and val
        if embargo_size > 0 and len(train_idx) > embargo_size:
            train_idx = train_idx[:-embargo_size]

        if len(train_idx) >= self.min_train_size:
            return (train_idx, val_idx)
        return None

    def _apply_purging(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        events: pd.DataFrame,
    ) -> np.ndarray:
        """Remove training samples whose labels overlap with validation."""
        val_start = X.index[val_idx[0]]
        t1_series = events.iloc[train_idx]["t1"]
        overlap_mask = t1_series.notna() & (t1_series >= val_start)
        return train_idx[~overlap_mask.to_numpy()]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 to float32 to reduce memory by 50%."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    return df


def _free_memory(*objects: Any) -> None:
    """Delete objects and force garbage collection."""
    for obj in objects:
        del obj
    gc.collect()

from src.constants import (
    DEFAULT_RANDOM_STATE,
    TRAIN_SPLIT_LABEL,
)
from src.labelling.label_primaire.utils import (
    MODEL_REGISTRY,
    load_model_class,
    get_dataset_for_model,
    get_daily_volatility,
)
from src.labelling.triple_barriere import get_events_primary_fast
from src.model.base import BaseModel
from src.path import (
    DOLLAR_BARS_PARQUET,
    LABEL_PRIMAIRE_OPTI_DIR,
    LABEL_PRIMAIRE_TRAIN_DIR,
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training."""

    model_name: str
    random_state: int = DEFAULT_RANDOM_STATE
    vol_window: int = 21
    use_class_weight: bool = True  # Balance classes during training
    parallelize_labeling: bool = True  # Use multiprocessing for label generation
    parallel_min_events: int = 10_000  # Minimum events to trigger parallel path
    n_jobs: int | None = -1  # Use all CPU cores
    n_splits_oof: int = 5  # Number of splits for OOF predictions
    min_train_size_oof: int = 500  # Minimum training size per fold


@dataclass
class TrainingResult:
    """Result of the training process."""

    model_name: str
    model_params: Dict[str, Any]
    triple_barrier_params: Dict[str, Any]
    train_samples: int
    label_distribution: Dict[str, Any]
    model_path: str
    events_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "triple_barrier_params": self.triple_barrier_params,
            "train_samples": self.train_samples,
            "label_distribution": self.label_distribution,
            "model_path": self.model_path,
            "events_path": self.events_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Training results saved to {path}")


# =============================================================================
# LABEL GENERATION HELPERS
# =============================================================================


def _resolve_n_jobs(requested: int | None) -> int:
    """Return a valid worker count capped to available CPU cores."""
    max_cores = multiprocessing.cpu_count()
    if requested is None or requested <= 0:
        return max_cores
    return min(requested, max_cores)


def _split_t_events(t_events: pd.DatetimeIndex, n_splits: int) -> List[pd.DatetimeIndex]:
    """Split events into roughly equal chunks while preserving order."""
    chunks = np.array_split(t_events, n_splits)
    return [pd.DatetimeIndex(chunk) for chunk in chunks if len(chunk) > 0]


def _generate_events_chunk(
    close: pd.Series,
    t_events_chunk: pd.DatetimeIndex,
    volatility: pd.Series,
    tb_params: Dict[str, Any],
) -> pd.DataFrame:
    """Generate triple-barrier events for a chunk of timestamps."""
    return get_events_primary_fast(
        close=close,
        t_events=t_events_chunk,
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility,
        max_holding=tb_params["max_holding"],
        min_return=tb_params.get("min_return", 0.0),
    )


def _generate_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    tb_params: Dict[str, Any],
    volatility: pd.Series,
    config: TrainingConfig,
) -> pd.DataFrame:
    """
    Generate triple-barrier events, optionally in parallel.
    """
    if not config.parallelize_labeling:
        return _generate_events_chunk(close, t_events, volatility, tb_params)

    n_events = len(t_events)
    if n_events < config.parallel_min_events:
        return _generate_events_chunk(close, t_events, volatility, tb_params)

    n_jobs = _resolve_n_jobs(config.n_jobs)
    event_chunks = _split_t_events(t_events, n_jobs)

    if len(event_chunks) <= 1:
        return _generate_events_chunk(close, t_events, volatility, tb_params)

    logger.info(
        "Generating %d events in parallel using %d workers",
        n_events,
        len(event_chunks),
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(event_chunks)) as executor:
        futures = [
            executor.submit(
                _generate_events_chunk,
                close,
                chunk,
                volatility,
                tb_params,
            )
            for chunk in event_chunks
        ]
        results = [future.result() for future in futures]

    non_empty = [df for df in results if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty).sort_index()
    # Remove duplicates that can occur at chunk boundaries
    if combined.index.has_duplicates:
        combined = cast(pd.DataFrame, combined[~combined.index.duplicated(keep="first")])
    return combined


def _label_metadata_path(features_path: Path) -> Path:
    """Return path to the label metadata file next to the features parquet."""
    return features_path.with_suffix(".label_meta.json")


def _load_label_metadata(meta_path: Path) -> Dict[str, Any] | None:
    """Load label metadata if present."""
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return cast(Dict[str, Any], json.load(f))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read label metadata at %s; ignoring", meta_path)
        return None


def _save_label_metadata(meta_path: Path, tb_params: Dict[str, Any]) -> None:
    """Persist triple-barrier parameters used for labeling."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tb_params": tb_params,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )


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
        Dictionary with keys:
        - model_params: Model hyperparameters
        - triple_barrier_params: Triple barrier parameters
        - best_score: Best optimization score
        - metric: Optimization metric used
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
        # Triple-barrier params may be absent when labels are precomputed
        "triple_barrier_params": opti_results.get("best_triple_barrier_params", {}),
        "best_score": opti_results["best_score"],
        "metric": opti_results["metric"],
    }


def get_model_output_dir(model_name: str) -> Path:
    """Get the output directory for a specific model."""
    return LABEL_PRIMAIRE_TRAIN_DIR / model_name


def get_features_path(model_name: str) -> Path:
    """Get the base features file path for a model type (without labels)."""
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    return path_map[dataset_type]


def get_labeled_features_path(model_name: str) -> Path:
    """
    Get the labeled features file path for a specific model.

    Each model gets its own labeled features file to avoid overwriting
    when training multiple models with different triple barrier parameters.

    Example: lightgbm -> data/features/dataset_features_final_lightgbm.parquet
    """
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    base_path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    base_path = base_path_map[dataset_type]
    # Insert model name before .parquet extension
    return base_path.parent / f"{base_path.stem}_{model_name}.parquet"


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train_model(
    model_name: str,
    config: TrainingConfig | None = None,
    opti_dir: Path | None = None,
    output_dir: Path | None = None,
) -> TrainingResult:
    """
    Train a primary model using optimized parameters.

    This function:
    1. Loads the full dataset (train + test)
    2. Generates triple barrier labels on the full dataset
    3. Saves labels to the input features file (column 'label')
    4. Trains the model on TRAIN split only
    5. Saves the trained model

    Parameters
    ----------
    model_name : str
        Name of the model to train.
    config : TrainingConfig, optional
        Training configuration.
    opti_dir : Path, optional
        Directory containing optimization results.
    output_dir : Path, optional
        Directory to save trained model and results.

    Returns
    -------
    TrainingResult
        Training results including paths to saved artifacts.
    """
    if config is None:
        config = TrainingConfig(model_name=model_name)
    else:
        config.model_name = model_name

    if output_dir is None:
        output_dir = get_model_output_dir(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {model_name} with optimized parameters")

    # Load optimized parameters
    logger.info("Loading optimized parameters...")
    optimized = load_optimized_params(model_name, opti_dir)
    model_params = optimized["model_params"]
    tb_params = optimized.get("triple_barrier_params", {})

    logger.info(f"Best optimization score ({optimized['metric']}): {optimized['best_score']:.4f}")
    logger.info(f"Model params: {model_params}")

    # Load model class
    model_class = load_model_class(model_name)

    # Determine paths
    base_features_path = get_features_path(model_name)
    labeled_features_path = get_labeled_features_path(model_name)

    # Load dataset with precomputed labels (always use base path for training)
    # The model-specific path will be created after training with predictions
    data_path = base_features_path
    logger.info(f"Loading labeled dataset from {data_path}")
    full_df = pd.read_parquet(data_path)
    full_df = _optimize_dtypes(full_df)  # float64 -> float32
    if "datetime_close" in full_df.columns:
        full_df = full_df.set_index("datetime_close")
    full_df = full_df.sort_index()

    # Validate presence of label
    if "label" not in full_df.columns:
        raise ValueError(
            "Dataset must contain a 'label' column. "
            "Provide pre-labeled data (label + t1 optional)."
        )

    # Build events from existing columns (label + optional t1)
    if "t1" in full_df.columns:
        events = full_df[["label", "t1"]].copy()
    else:
        logger.warning("No 't1' column found; events will contain only labels.")
        events = full_df[["label"]].copy()

    # Prepare training data (TRAIN split only)
    # Extract only TRAIN split, then free full_df
    if "split" in full_df.columns:
        train_mask = full_df["split"] == TRAIN_SPLIT_LABEL
        # Get labels before filtering
        y_train = full_df.loc[train_mask, "label"].copy()
        # Get feature columns
        non_feature_cols = {
            "bar_id", "timestamp_open", "timestamp_close",
            "datetime_open", "datetime_close", "threshold_used",
            "log_return", "split", "label", "t1",
        }
        feature_cols = [c for c in full_df.columns if c not in non_feature_cols]
        X_train = full_df.loc[train_mask, feature_cols].copy()
    else:
        logger.warning("No 'split' column found, using all data for training")
        y_train = full_df["label"].copy()
        non_feature_cols = {
            "bar_id", "timestamp_open", "timestamp_close",
            "datetime_open", "datetime_close", "threshold_used",
            "log_return", "split", "label", "t1",
        }
        feature_cols = [c for c in full_df.columns if c not in non_feature_cols]
        X_train = full_df[feature_cols].copy()

    # Free full_df - no longer needed
    del full_df
    gc.collect()
    logger.info("Released full dataset from memory")

    # Drop NaN labels
    valid_mask = ~y_train.isna()
    X_train = cast(pd.DataFrame, X_train[valid_mask])
    y_train = cast(pd.Series, y_train[valid_mask].astype(np.int8))  # int8 is enough for labels

    # Optimize X_train dtypes
    X_train = _optimize_dtypes(X_train)

    logger.info(f"Training samples: {len(X_train)}")

    # Analyze label distribution
    label_counts = y_train.value_counts().to_dict()
    total = len(y_train)
    label_percentages = {k: v / total * 100 for k, v in label_counts.items()}
    label_stats = {
        "total_events": total,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "label_percentages": {str(k): float(v) for k, v in label_percentages.items()},
    }
    logger.info(f"Label distribution: {label_counts}")

    # Add random state and parallelism to model params
    model_params["random_state"] = config.random_state
    # Force parallelism - use all CPU cores
    if model_name == "catboost":
        model_params["thread_count"] = -1
    else:
        # All other models use n_jobs
        model_params["n_jobs"] = -1

    # Add class weight if supported and requested
    if config.use_class_weight and model_name in ["lightgbm", "xgboost", "random_forest", "ridge", "logistic"]:
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        class_weight = {
            cls: total_samples / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        model_params["class_weight"] = class_weight
        logger.info(f"Using class weights: {class_weight}")

    # Create and train model
    logger.info(f"Training model with params: {model_params}")
    model = model_class(**model_params)
    model.fit(cast(pd.DataFrame, X_train), cast(pd.Series, y_train))

    logger.info("Model training complete")

    # Keep training sample count before freeing memory
    n_train_samples = len(X_train)

    # Free training data - model is trained
    del X_train, y_train
    gc.collect()
    logger.info("Released training data from memory")

    # Save model
    model_path = output_dir / f"{model_name}_model.joblib"
    model.save(model_path)

    # =========================================================================
    # GENERATE OUT-OF-FOLD (OOF) PREDICTIONS ON TRAIN SPLIT
    # =========================================================================
    logger.info("Generating out-of-fold (OOF) predictions on train split...")

    # Reload full dataset to prepare OOF predictions
    full_df = pd.read_parquet(data_path)
    full_df = _optimize_dtypes(full_df)
    if "datetime_close" in full_df.columns:
        full_df = full_df.set_index("datetime_close")
    full_df = full_df.sort_index()

    # Extract train split for OOF predictions
    if "split" in full_df.columns:
        train_mask = full_df["split"] == TRAIN_SPLIT_LABEL
        train_df = cast(pd.DataFrame, full_df[train_mask].copy())
    else:
        # No split column - use all data (shouldn't happen in normal flow)
        train_df = full_df.copy()
        logger.warning("No 'split' column found, using all data for OOF predictions")

    # Prepare features and labels for OOF
    X_train_oof = cast(pd.DataFrame, train_df[feature_cols])
    y_train_oof = cast(pd.Series, train_df["label"])

    # Filter out NaN labels
    valid_mask = ~y_train_oof.isna()
    X_train_oof = cast(pd.DataFrame, X_train_oof[valid_mask])
    y_train_oof = cast(pd.Series, y_train_oof[valid_mask].astype(np.int8))

    # Build events for purging (needs t1 and label)
    if "t1" in train_df.columns:
        events_oof = train_df.loc[valid_mask, ["label", "t1"]].copy()
    else:
        events_oof = train_df.loc[valid_mask, ["label"]].copy()

    logger.info(f"Performing {config.n_splits_oof}-fold CV for OOF predictions on {len(X_train_oof)} samples...")

    # Initialize OOF prediction arrays
    oof_predictions = np.full(len(X_train_oof), np.nan, dtype=np.float32)
    n_classes = len(np.unique(y_train_oof))
    oof_probas = np.full((len(X_train_oof), n_classes), np.nan, dtype=np.float32)

    # Create WalkForwardCV splitter
    cv = WalkForwardCV(
        n_splits=config.n_splits_oof,
        min_train_size=config.min_train_size_oof,
        embargo_pct=0.01,
    )

    splits = cv.split(X_train_oof, events_oof)
    logger.info(f"Generated {len(splits)} valid CV splits")

    if len(splits) == 0:
        logger.warning("No valid CV splits generated, skipping OOF predictions")
    else:
        # Train models for each fold and collect OOF predictions
        X_train_values = X_train_oof.to_numpy()
        y_train_values = y_train_oof.to_numpy()

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"  Fold {fold_idx + 1}/{len(splits)}: train={len(train_idx)}, val={len(val_idx)}")

            # Extract fold data
            X_fold_train = X_train_values[train_idx]
            y_fold_train = y_train_values[train_idx]
            X_fold_val = X_train_values[val_idx]

            # Create and train fold model with same params
            fold_model = model_class(**model_params)
            fold_model.fit(X_fold_train, y_fold_train)

            # Predict on validation fold (OOF)
            oof_predictions[val_idx] = fold_model.predict(X_fold_val)

            # Try to get probabilities
            try:
                oof_probas[val_idx] = cast(Any, fold_model).predict_proba(X_fold_val)
            except Exception:
                pass

            # Free fold model
            del fold_model
            gc.collect()

        logger.info("OOF predictions complete")

        # ====================================================================
        # PREDICT ON REMAINING SAMPLES (LAST FOLD)
        # ====================================================================
        # Identify samples without OOF predictions (last fold in TimeSeriesSplit)
        missing_mask = np.isnan(oof_predictions)
        n_missing = np.sum(missing_mask)

        if n_missing > 0:
            logger.info(f"Generating predictions for remaining {n_missing} samples (last fold)...")

            # Train a model on all samples that have OOF predictions
            # This maintains OOF principle: we don't use target samples in training
            available_mask = ~missing_mask
            X_available = X_train_values[available_mask]
            y_available = y_train_values[available_mask]
            X_missing = X_train_values[missing_mask]

            # Train model on available data
            last_fold_model = model_class(**model_params)
            last_fold_model.fit(X_available, y_available)

            # Predict on missing samples
            oof_predictions[missing_mask] = last_fold_model.predict(X_missing)

            # Try to get probabilities
            try:
                oof_probas[missing_mask] = cast(Any, last_fold_model).predict_proba(X_missing)
            except Exception:
                pass

            # Free model
            del last_fold_model
            gc.collect()

            logger.info(f"Last fold predictions complete: {n_missing} samples covered")

    # =========================================================================
    # RETRAIN FINAL MODEL ON FULL TRAIN SPLIT
    # =========================================================================
    logger.info("Retraining final model on full train split...")

    # The model is already trained on the full train split (before OOF)
    # We keep the already trained model as the final model
    logger.info("Final model already trained on full train split")

    # =========================================================================
    # SAVE DATASET WITH OOF PREDICTIONS
    # =========================================================================
    logger.info("Saving dataset with OOF predictions...")

    # Add OOF predictions to train split only (test will be NaN)
    # Initialize prediction columns with NaN
    full_df["prediction"] = np.nan
    for i in range(n_classes):
        full_df[f"proba_{i}"] = np.nan

    # Fill in OOF predictions for train split
    # Map OOF predictions back to original full_df indices
    train_valid_idx = train_df.loc[valid_mask].index
    full_df.loc[train_valid_idx, "prediction"] = oof_predictions
    for i in range(n_classes):
        full_df.loc[train_valid_idx, f"proba_{i}"] = oof_probas[:, i]

    # Save the full dataset with OOF predictions to model-specific file
    logger.info(f"Saving dataset with OOF predictions to {labeled_features_path}")
    labeled_features_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(labeled_features_path)
    logger.info(f"Dataset with OOF predictions saved to {labeled_features_path}")

    # Log statistics
    n_oof_valid = np.sum(~np.isnan(oof_predictions))
    logger.info(f"OOF predictions: {n_oof_valid}/{len(oof_predictions)} valid samples ({n_oof_valid/len(oof_predictions)*100:.1f}%)")

    # Free memory
    del full_df, train_df, X_train_oof, y_train_oof, events_oof, oof_predictions, oof_probas
    gc.collect()
    logger.info("Released datasets from memory")

    # Save events (labels) for reference
    events_path = output_dir / f"{model_name}_events.parquet"
    events.to_parquet(events_path)
    logger.info(f"Events saved to {events_path}")

    # Free events
    del events
    gc.collect()

    # Build result
    result = TrainingResult(
        model_name=model_name,
        model_params=model_params,
        triple_barrier_params=tb_params,
        train_samples=n_train_samples,
        label_distribution=label_stats,
        model_path=str(model_path),
        events_path=str(events_path),
    )

    # Save training results
    results_path = output_dir / f"{model_name}_training_results.json"
    result.save(results_path)

    return result


# =============================================================================
# INTERACTIVE CLI
# =============================================================================


def get_available_optimized_models() -> list[str]:
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
    print("LABEL PRIMAIRE - TRAINING")
    print("=" * 60)
    print("\nModels disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        status = "[optimise]" if model in optimized else "[non optimise]"
        info = MODEL_REGISTRY[model]
        dataset_type = info["dataset"]
        print(f"  {i}. {model:<15} ({dataset_type}) {status}")

    print("-" * 40)

    if not optimized:
        print("\nAucun modele optimise. Lancer d'abord l'optimisation.")
        print("python -m src.labelling.label_primaire.opti")
        exit(1)

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
            print("\nEntrainement annule.")
            exit(0)


def get_yes_no_input(prompt: str, default: bool = True) -> bool:
    """Get yes/no input with default value."""
    default_str = "O/n" if default else "o/N"
    while True:
        try:
            value = input(f"{prompt} ({default_str}): ").strip().lower()
            if value == "":
                return default
            if value in ["o", "oui", "y", "yes"]:
                return True
            if value in ["n", "non", "no"]:
                return False
            print("Repondre par O (oui) ou N (non)")
        except KeyboardInterrupt:
            print("\nAnnule.")
            exit(0)


def main() -> None:
    """Main entry point with interactive prompts."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Interactive model selection
    model_name = select_model()
    print(f"\nModele selectionne: {model_name}")

    # Load and display optimized params
    try:
        optimized = load_optimized_params(model_name)
        print(f"\nParametres optimises charges:")
        print(f"  Score ({optimized['metric']}): {optimized['best_score']:.4f}")
        print(f"  Triple Barrier: {optimized['triple_barrier_params']}")
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        exit(1)

    # Ask about class weighting
    use_class_weight = get_yes_no_input("\nUtiliser le class weighting?", default=True)

    # Confirm
    print("\n" + "-" * 40)
    print(f"Modele: {model_name}")
    print(f"Class weighting: {'Oui' if use_class_weight else 'Non'}")
    print("-" * 40)

    confirm = input("\nLancer l'entrainement? (O/n): ").strip().lower()
    if confirm == "n":
        print("Entrainement annule.")
        return

    # Create config
    config = TrainingConfig(
        model_name=model_name,
        random_state=DEFAULT_RANDOM_STATE,
        use_class_weight=use_class_weight,
    )

    # Run training
    print("\n")
    result = train_model(
        model_name=model_name,
        config=config,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"ENTRAINEMENT TERMINE: {model_name.upper()}")
    print("=" * 60)
    print(f"Echantillons d'entrainement: {result.train_samples}")
    print(f"\nDistribution des labels:")
    for label, count in result.label_distribution["label_counts"].items():
        pct = result.label_distribution["label_percentages"][label]
        print(f"  Label {label}: {count} ({pct:.1f}%)")
    print(f"\nParametres Triple Barrier:")
    for k, v in result.triple_barrier_params.items():
        print(f"  {k}: {v}")
    print(f"\nModele sauvegarde: {result.model_path}")
    print(f"Events sauvegardes: {result.events_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
