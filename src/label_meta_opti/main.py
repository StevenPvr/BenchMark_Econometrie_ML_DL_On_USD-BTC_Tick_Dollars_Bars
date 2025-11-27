"""Main entry point for label_meta_opti module.

This module optimizes hyperparameters for the meta-model separately from
the primary model. The meta-model is trained on meta-labels (correct/wrong)
and uses different hyperparameters than the primary model.

Steps:
1. Load the primary model and training data.
2. Generate primary predictions on training set.
3. Create meta-labels (was the primary prediction correct?).
4. Optimize LightGBM hyperparameters for binary classification.
5. Save optimized parameters for use in label_meta_train.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib # type: ignore[import-untyped]
import numpy as np
import pandas as pd # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.label_meta.meta_labeling import get_meta_labels
from src.label_meta_opti.optimize import (
    optimize_meta_model,
    save_meta_optimization_results,
)
from src.path import (
    DATASET_FEATURES_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_LABEL_PARQUET,
    DATASET_FEATURES_LSTM_LABEL_PARQUET,
    LABEL_META_OPTI_DIR,
    LABEL_META_OPTIMIZATION_FILE,
    LABEL_PRIMAIRE_MODELS_DIR,
    LABEL_PRIMAIRE_EVENTS_TRAIN_FILE,
)
from src.utils.user_input import ask_choice, ask_integer

logger = get_logger(__name__)

AVAILABLE_MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "rf",
    "ridge",
    "lasso",
    "logistic",
]


def get_model_family(model_name: str) -> str:
    """Return the model family to pick the right labeled dataset."""
    name = model_name.lower()
    if name == "lstm":
        return "lstm"
    if name in ("ridge", "lasso", "logistic", "logistic_regression"):
        return "linear"
    return "ml"


def get_labeled_dataset_path(model_name: str) -> Path:
    """Return parquet labeled dataset path matching the model family."""
    family = get_model_family(model_name)
    if family == "ml":
        return DATASET_FEATURES_LABEL_PARQUET
    if family == "linear":
        return DATASET_FEATURES_LINEAR_LABEL_PARQUET
    if family == "lstm":
        return DATASET_FEATURES_LSTM_LABEL_PARQUET
    raise ValueError(f"Unknown model family for {model_name}")


def load_primary_model(model_name: str) -> Any:
    """Load the trained primary model."""
    path = LABEL_PRIMAIRE_MODELS_DIR / f"{model_name}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Primary model not found at {path}. "
            "Please run 'python -m src.label_primaire_train.main' first."
        )
    logger.info("Loading primary model from %s", path)
    return joblib.load(path)


def load_training_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and train events."""
    dataset_parquet = get_labeled_dataset_path(model_name)

    if not dataset_parquet.exists():
        raise FileNotFoundError(
            f"Features file {dataset_parquet} not found. "
            "Run label_primaire first."
        )

    if not LABEL_PRIMAIRE_EVENTS_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Events file {LABEL_PRIMAIRE_EVENTS_TRAIN_FILE} not found."
        )

    logger.info("Loading features from %s", dataset_parquet)
    df_features = pd.read_parquet(dataset_parquet)

    logger.info("Loading train events from %s", LABEL_PRIMAIRE_EVENTS_TRAIN_FILE)
    events_train = pd.read_parquet(LABEL_PRIMAIRE_EVENTS_TRAIN_FILE)

    return df_features, events_train


def main() -> None:
    print("\n" + "=" * 60)
    print("   LABEL META OPTI - Optimize Meta-Model Hyperparameters")
    print("=" * 60)

    LABEL_META_OPTI_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select Primary Model
    print("\n" + "-" * 30)
    print("PRIMARY MODEL SELECTION")
    print("-" * 30)
    print("Choose the primary model for which to optimize the meta-model.")
    model_name = ask_choice(AVAILABLE_MODELS, prompt="Primary model", default="lightgbm")

    # 2. Load Data
    print("\nLoading data...")
    try:
        df_features, events_train = load_training_data(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Load Primary Model
    try:
        primary_model = load_primary_model(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 4. Generate Primary Predictions
    print("Generating primary model predictions on training set...")

    train_indices = events_train["t_start"].values.astype(int)
    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_train = df_features.iloc[train_indices][feature_cols].to_numpy()
    primary_preds = primary_model.predict(X_train)
    primary_signal = pd.Series(primary_preds, index=events_train.index)

    # 5. Generate Meta Labels
    print("Generating meta-labels...")
    if "ret" not in events_train.columns:
        raise ValueError("events_train file missing 'ret' column needed for meta-labeling.")

    meta_labels = get_meta_labels(
        events=events_train,
        primary_signal=primary_signal
    )

    # Filter valid samples (where primary made a trade)
    valid_mask = ~meta_labels.isna()

    if valid_mask.sum() == 0:
        print("No trades taken by primary model on training set. Cannot optimize meta-model.")
        return

    y_meta = np.asarray(meta_labels[valid_mask].astype(int))
    X_base_valid = X_train[valid_mask]
    primary_signal_valid = primary_preds[valid_mask]

    # Create meta features (original features + primary signal)
    X_meta = np.column_stack([X_base_valid, primary_signal_valid])

    print(f"\nMeta-optimization dataset: {len(y_meta)} samples")
    print(f"Meta-Label Distribution: 1s (correct)={y_meta.sum()}, 0s (wrong)={len(y_meta)-y_meta.sum()}")
    print(f"Accuracy baseline: {max(y_meta.mean(), 1-y_meta.mean()):.2%}")

    # 6. Optimization Settings
    print("\n" + "-" * 30)
    print("OPTIMIZATION SETTINGS")
    print("-" * 30)
    n_trials = ask_integer("Number of Optuna trials", default=100, min_val=10, max_val=500)
    n_splits = ask_integer("Number of CV splits", default=5, min_val=3, max_val=10)
    purge_gap = ask_integer("Purge gap (bars)", default=5, min_val=0, max_val=100)

    # 7. Run Optimization
    print("\n" + "=" * 60)
    print("Starting meta-model optimization (LightGBM binary classifier)...")
    print("=" * 60)

    result = optimize_meta_model(
        X=X_meta,
        y=y_meta,
        n_trials=n_trials,
        n_splits=n_splits,
        purge_gap=purge_gap,
        metric="mcc",  # Matthews Correlation Coefficient for imbalanced binary classification
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Optimization Complete")
    print(f"Best Score (MCC): {result.best_score:.4f}")
    print("=" * 60)

    # 8. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"meta_params_{model_name}_{timestamp}.json"

    save_name = input(f"\nEnter filename to save parameters [{default_name}]: ").strip()
    if not save_name:
        save_name = default_name

    if not save_name.endswith(".json"):
        save_name += ".json"

    save_path = LABEL_META_OPTI_DIR / save_name
    save_meta_optimization_results(result, save_path)
    print(f"\nParameters saved to: {save_path}")

    # Also save to default location
    save_meta_optimization_results(result, LABEL_META_OPTIMIZATION_FILE)
    print(f"Also updated default meta parameters file: {LABEL_META_OPTIMIZATION_FILE}")

    print("\nNext step: Run 'python -m src.label_meta_train.main' to train the meta-model.")


if __name__ == "__main__":
    main()
