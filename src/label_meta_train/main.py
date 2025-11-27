"""Main entry point for label_meta_train module.

This module trains the meta-model on the training split using:
1. A pre-trained primary model (to generate signals).
2. The training dataset (features + true labels/returns).
3. Meta-labeling logic (De Prado).

The meta-model is trained to predict whether the primary model's signal
will be correct (profit > 0) or not.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_meta.meta_labeling import get_meta_labels
from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel
from src.path import (
    DATASET_FEATURES_LABEL_PARQUET,
    DATASET_FEATURES_LABEL_CSV,
    DATASET_FEATURES_LINEAR_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_LABEL_CSV,
    DATASET_FEATURES_LSTM_LABEL_PARQUET,
    DATASET_FEATURES_LSTM_LABEL_CSV,
    LABEL_META_TRAIN_DIR,
    LABEL_META_TRAIN_MODELS_DIR,
    LABEL_PRIMAIRE_MODELS_DIR,
    LABEL_PRIMAIRE_EVENTS_TRAIN_FILE,
    LABEL_PRIMAIRE_OPTIMIZATION_FILE,
)
from src.utils import load_json_data

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


def ask_model_choice() -> str:
    """Ask user to choose a primary model."""
    print("\nAvailable primary models:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")

    while True:
        answer = input("\nChoose a primary model to build meta-model for (number or name): ").strip().lower()

        # Try as number
        try:
            idx = int(answer)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[idx - 1]
        except ValueError:
            pass

        # Try as name
        if answer in AVAILABLE_MODELS:
            return answer

        print("Invalid choice.")


def get_model_family(model_name: str) -> str:
    """Return the model family to pick the right labeled dataset."""
    name = model_name.lower()
    if name == "lstm":
        return "lstm"
    if name in ("ridge", "lasso", "logistic", "logistic_regression"):
        return "linear"
    return "ml"


def get_labeled_dataset_paths(model_name: str) -> tuple[Path, Path]:
    """Return parquet/csv labeled dataset paths matching the model family."""
    family = get_model_family(model_name)
    if family == "ml":
        return DATASET_FEATURES_LABEL_PARQUET, DATASET_FEATURES_LABEL_CSV
    if family == "linear":
        return DATASET_FEATURES_LINEAR_LABEL_PARQUET, DATASET_FEATURES_LINEAR_LABEL_CSV
    if family == "lstm":
        return DATASET_FEATURES_LSTM_LABEL_PARQUET, DATASET_FEATURES_LSTM_LABEL_CSV
    raise ValueError(f"Unknown model family for {model_name}")


def load_primary_model(model_name: str) -> Any:
    """Load the trained primary model."""
    path = LABEL_PRIMAIRE_MODELS_DIR / f"{model_name}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Primary model not found at {path}. "
            "Please run 'python -m src.label_primaire.main' first."
        )
    logger.info("Loading primary model from %s", path)
    return joblib.load(path)


def load_training_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and train events."""
    dataset_parquet, _ = get_labeled_dataset_paths(model_name)

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


def get_primary_params(model_name: str) -> dict[str, Any]:
    """Load optimized parameters from primary model to reuse for meta model."""
    if not LABEL_PRIMAIRE_OPTIMIZATION_FILE.exists():
        logger.warning("No optimization file found. Using default LightGBM params.")
        return {}

    data = load_json_data(LABEL_PRIMAIRE_OPTIMIZATION_FILE)
    opt_model = data.get("model_name", "")

    # Only reuse params if the primary model was LightGBM
    # (since the meta model IS LightGBM)
    if opt_model == "lightgbm":
        logger.info("Reusing LightGBM parameters from primary model optimization.")
        return data.get("best_model_params", {})

    # If primary was not LightGBM, we can't strictly reuse params for a LightGBM meta-model,
    # unless we want to use the default config.
    # The prompt says: "Comme le meta label est lightgbm on peut récupérer les meilleurs paramètres du model primaire lightgbm"
    # This implies if primary is lightgbm, we reuse.

    return {}


def main() -> None:
    print("\n" + "=" * 60)
    print("   LABEL META TRAIN - Train Meta-Model")
    print("=" * 60)

    LABEL_META_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_META_TRAIN_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select Primary Model
    model_name = ask_model_choice()

    # 2. Load Data
    try:
        df_features, events_train = load_training_data(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Filter features for training based on events_train
    # The events_train file contains t_start indices corresponding to df_features.
    # We only train on samples that are in the training set (implied by events_train).

    # We need to reconstruct the training set used for the primary model
    # usually defined by the indices in events_train.

    # Note: df_features contains ALL data. events_train contains only train events.
    # We need to map them.

    # 3. Load Primary Model
    try:
        primary_model = load_primary_model(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 4. Generate Primary Predictions on Train Data
    print("Generating primary model predictions on training set...")

    # Get indices from events
    train_indices = events_train["t_start"].values.astype(int)

    # Extract features for these indices
    # Exclude non-feature columns
    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_train = df_features.iloc[train_indices][feature_cols].to_numpy()

    # Predict
    primary_preds = primary_model.predict(X_train)
    primary_signal = pd.Series(primary_preds, index=events_train.index) # index match events

    # 5. Generate Meta Labels
    # Meta label is 1 if primary model was correct (and took a trade), 0 otherwise.
    # We use get_meta_labels from src.label_meta.meta_labeling

    # Need to align events index with primary_signal index if they differ
    # Here they should match because we built primary_signal from events_train rows.

    print("Generating meta-labels...")
    # events_train needs 'ret' column (it should have it from triple barrier)
    if "ret" not in events_train.columns:
        raise ValueError("events_train file missing 'ret' column needed for meta-labeling.")

    # We might need to reset index of events_train to match what get_meta_labels expects if it expects datetime index
    # But get_meta_labels checks alignment.
    # Let's check get_meta_labels implementation... it uses intersection of indices.
    # Our events_train usually has a RangeIndex or whatever it was saved with.

    meta_labels = get_meta_labels(
        events=events_train,
        primary_signal=primary_signal
    )

    # meta_labels has NaNs where primary_signal == 0 (no trade). We drop those.
    valid_mask = ~meta_labels.isna()

    if valid_mask.sum() == 0:
        print("No trades taken by primary model on training set. Cannot train meta-model.")
        return

    y_meta = meta_labels[valid_mask].astype(int).values
    X_base_valid = X_train[valid_mask]
    primary_signal_valid = primary_preds[valid_mask]

    # 6. Create Meta Features
    # Feature set = Original Features + Primary Signal (or Proba)
    # De Prado suggests adding the probability/signal as a feature.
    X_meta = np.column_stack([X_base_valid, primary_signal_valid])

    print(f"Training Meta-Model (LightGBM) on {len(y_meta)} samples...")
    print(f"Meta-Label Distribution: 1s={y_meta.sum()}, 0s={len(y_meta)-y_meta.sum()}")

    # 7. Configure Meta Model
    # Inherit params from primary if it was LightGBM
    meta_params = get_primary_params(model_name)
    meta_params["random_state"] = DEFAULT_RANDOM_STATE

    # Train
    meta_model = LightGBMModel(**meta_params)
    meta_model.fit(X_meta, y_meta)

    # 8. Save Meta Model
    save_path = LABEL_META_TRAIN_MODELS_DIR / f"meta_model_{model_name}.joblib"
    joblib.dump(meta_model, save_path)

    print(f"\nMeta-model saved to: {save_path}")
    print("Done.")

if __name__ == "__main__":
    main()
