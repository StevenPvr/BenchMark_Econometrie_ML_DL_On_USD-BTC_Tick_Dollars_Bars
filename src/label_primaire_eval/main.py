"""Main entry point for label_primaire_eval module.

This module is responsible for:
1. Loading a trained primary model and its associated metadata (labeling params).
2. Generating Triple Barrier labels on the FULL dataset (ground truth).
3. Evaluating the model on the test split (predictions vs ground truth).
4. Saving the labeled dataset and events for downstream tasks (label_meta).
"""

from __future__ import annotations

import sys
import os
import joblib # type: ignore[import-untyped]
import pandas as pd # type: ignore[import-untyped]
import numpy as np
from typing import Any

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config_logging import get_logger
from src.label_primaire.main import (
    generate_labels,
    load_features,
    compute_classification_metrics,
    save_labeled_datasets
)
from src.path import (
    LABEL_PRIMAIRE_TRAIN_DIR,
    LABEL_PRIMAIRE_EVAL_DIR,
    LABEL_PRIMAIRE_EVENTS_TRAIN_FILE,
    LABEL_PRIMAIRE_EVENTS_TEST_FILE,
    LABEL_PRIMAIRE_EVALUATION_FILE,
    DATASET_FEATURES_LABEL_PARQUET, # Check if we need to explicitly mention this or if save_labeled_datasets handles it
)
from src.utils.user_input import ask_choice
from src.utils import load_json_data, save_json_pretty

logger = get_logger(__name__)

def get_available_models() -> list[str]:
    """List available model files in the train directory."""
    if not LABEL_PRIMAIRE_TRAIN_DIR.exists():
        return []
    return [f.name for f in LABEL_PRIMAIRE_TRAIN_DIR.glob("*.joblib")]

def main() -> None:
    print("\n" + "=" * 60)
    print("   LABEL PRIMAIRE - EVALUATION")
    print("=" * 60)

    LABEL_PRIMAIRE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select Model
    print("\n" + "-"*30)
    print("MODEL SELECTION")
    print("-"*30)

    model_files = get_available_models()
    if not model_files:
        print(f"No trained models found in {LABEL_PRIMAIRE_TRAIN_DIR}.")
        print("Please run 'python -m src.label_primaire_train.main' first.")
        return

    # Sort by modification time
    model_files.sort(key=lambda x: (LABEL_PRIMAIRE_TRAIN_DIR / x).stat().st_mtime, reverse=True)

    choice = ask_choice(model_files, prompt="Choose model to evaluate")
    model_path = LABEL_PRIMAIRE_TRAIN_DIR / choice
    metadata_path = model_path.with_suffix(".meta.json")

    if not metadata_path.exists():
        print(f"Error: Metadata file not found for {choice} ({metadata_path}).")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    metadata = load_json_data(metadata_path)

    model_name = metadata.get("model_name")
    labeling_params = metadata.get("labeling_params")
    feature_cols = metadata.get("feature_cols")

    # Validate required metadata fields
    if model_name is None:
        print("Error: 'model_name' not found in metadata.")
        return
    if labeling_params is None:
        print("Error: 'labeling_params' not found in metadata.")
        return
    if feature_cols is None:
        print("Error: 'feature_cols' not found in metadata.")
        return

    # Type assertions for type checker after validation
    assert model_name is not None
    assert labeling_params is not None
    assert feature_cols is not None

    print(f"Model Type: {model_name.upper()}")
    print(f"Labeling Params: {labeling_params}")

    # 2. Data Loading & Label Generation
    print("\n" + "-"*30)
    print("DATA & LABELS")
    print("-"*30)

    df_features, prices, volatility = load_features()
    n_samples = len(df_features)

    print("Generating ground truth labels on full dataset...")
    events, labels = generate_labels(
        prices=prices,
        volatility=volatility,
        n_samples=n_samples,
        pt_mult=labeling_params["pt_mult"],
        sl_mult=labeling_params["sl_mult"],
        max_holding=int(labeling_params["max_holding_period"]),
        min_ret=labeling_params.get("min_ret", 0.0),
    )

    # 3. Evaluation
    print("\n" + "-"*30)
    print("EVALUATION")
    print("-"*30)

    if "split" in df_features.columns:
        test_mask = df_features["split"] == "test"
        if test_mask.sum() == 0:
             # Fallback
             n_train = int(n_samples * 0.8)
             test_mask = pd.Series([False]*n_train + [True]*(n_samples-n_train), index=df_features.index)
    else:
        n_train = int(n_samples * 0.8)
        test_mask = pd.Series([False]*n_train + [True]*(n_samples-n_train), index=df_features.index)

    train_mask = ~test_mask
    n_train_split = int(train_mask.sum())

    # Prepare X and y for test
    df_labeled = df_features.copy()
    df_labeled["label"] = labels.values

    # Predict on Test
    valid_test_mask = test_mask & df_labeled["label"].notna()

    if valid_test_mask.sum() == 0:
        print("Warning: No valid labeled test samples found.")
        test_metrics = {}
    else:
        X_test = df_labeled.loc[valid_test_mask, feature_cols].to_numpy()
        y_test = df_labeled.loc[valid_test_mask, "label"].astype(int)

        y_pred = model.predict(X_test)

        test_metrics = compute_classification_metrics(y_test.to_numpy(), y_pred)

        print("\nTest Metrics:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
        print(f"  MCC:      {test_metrics['mcc']:.4f}")
        print(f"  Samples:  {test_metrics['n_samples']}")

    # Predict on Train (for reference)
    valid_train_mask = train_mask & df_labeled["label"].notna()
    if valid_train_mask.sum() > 0:
        X_train = df_labeled.loc[valid_train_mask, feature_cols].to_numpy()
        y_train = df_labeled.loc[valid_train_mask, "label"].astype(int)
        y_pred_train = model.predict(X_train)
        train_metrics = compute_classification_metrics(y_train.to_numpy(), y_pred_train)
        print("\nTrain Metrics:")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {train_metrics['f1_macro']:.4f}")
        print(f"  MCC:      {train_metrics['mcc']:.4f}")
    else:
        train_metrics = {}

    # 4. Save Datasets (for label_meta)
    print("\n" + "-"*30)
    print("SAVING ARTIFACTS")
    print("-"*30)

    # Save labeled datasets (Standard Paths for label_meta)
    save_labeled_datasets(
        labels=labels,
        model_name=model_name,
        test_mask=test_mask,
        df_features=df_features,
    )

    # Save Events (Standard Paths for label_meta)
    # Note: path.py constants for events now point to LABEL_PRIMAIRE_EVAL_DIR, so this puts them there.
    # We must ensure label_meta knows to look there (it uses the same constants, so yes).

    events_train = events[events["t_start"] < n_train_split].copy()
    events_test = events[events["t_start"] >= n_train_split].copy()

    events_train.to_parquet(LABEL_PRIMAIRE_EVENTS_TRAIN_FILE)
    events_test.to_parquet(LABEL_PRIMAIRE_EVENTS_TEST_FILE)
    print(f"Saved events to {LABEL_PRIMAIRE_EVAL_DIR}")

    # Save Evaluation Results
    eval_payload = {
        "model_name": model_name,
        "model_path": str(model_path),
        "labeling_params": labeling_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    save_json_pretty(eval_payload, LABEL_PRIMAIRE_EVALUATION_FILE)
    print(f"Saved evaluation report to {LABEL_PRIMAIRE_EVALUATION_FILE}")

    print("\n" + "=" * 60)
    print("Label Primaire Evaluation Complete")
    print("Results are ready for label_meta.")
    print("=" * 60)

if __name__ == "__main__":
    main()
