"""Main entry point for label_primaire_train module.

This module is responsible for:
1. Loading optimized parameters (labeling + model) from label_primaire_opti.
2. Generating labels on the training data using the optimized labeling parameters.
3. Training the primary model using the optimized hyperparameters.
4. Saving the trained model and its configuration for the evaluation step.
"""

from __future__ import annotations

import sys
import os
import joblib # type: ignore[import-untyped]
import pandas as pd # type: ignore[import-untyped]
from datetime import datetime

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config_logging import get_logger
from src.label_primaire.main import generate_labels, load_features, get_model_class, compute_class_weights
from src.path import LABEL_PRIMAIRE_OPTI_DIR, LABEL_PRIMAIRE_TRAIN_DIR
from src.utils.user_input import ask_choice, ask_yes_no
from src.utils import load_json_data, save_json_pretty

logger = get_logger(__name__)

def get_available_param_files() -> list[str]:
    """List available JSON parameter files in the opti directory."""
    if not LABEL_PRIMAIRE_OPTI_DIR.exists():
        return []
    return [f.name for f in LABEL_PRIMAIRE_OPTI_DIR.glob("*.json")]

def main() -> None:
    print("\n" + "=" * 60)
    print("   LABEL PRIMAIRE - TRAINING")
    print("=" * 60)

    LABEL_PRIMAIRE_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select Parameters
    print("\n" + "-"*30)
    print("PARAMETER SELECTION")
    print("-"*30)

    param_files = get_available_param_files()
    if not param_files:
        print(f"No parameter files found in {LABEL_PRIMAIRE_OPTI_DIR}.")
        print("Please run 'python -m src.label_primaire_opti.main' first.")
        return

    # Sort by modification time (newest first)
    param_files.sort(key=lambda x: (LABEL_PRIMAIRE_OPTI_DIR / x).stat().st_mtime, reverse=True)

    choice = ask_choice(param_files, prompt="Choose parameter file to load")
    param_path = LABEL_PRIMAIRE_OPTI_DIR / choice

    print(f"Loading parameters from {param_path}...")
    params_data = load_json_data(param_path)

    model_name = params_data.get("model_name")
    labeling_params = params_data.get("best_labeling_params")
    model_params = params_data.get("best_model_params")

    if not model_name or not labeling_params:
        print("Error: Invalid parameter file format.")
        return

    print(f"Model Type: {model_name.upper()}")
    print(f"Labeling Params: {labeling_params}")
    print(f"Model Hyperparams: {model_params}")

    # 2. Load Data and Generate Labels
    print("\n" + "-"*30)
    print("DATA PREPARATION")
    print("-"*30)

    df_features, prices, volatility = load_features()

    # Use train split for training (if exists), or first 80%
    if "split" in df_features.columns:
        train_mask = df_features["split"] == "train"
    else:
        n_samples = len(df_features)
        n_train = int(n_samples * 0.8)
        train_mask = pd.Series([True] * n_train + [False] * (n_samples - n_train), index=df_features.index)

    print("Generating labels for training...")
    events, labels = generate_labels(
        prices=prices,
        volatility=volatility,
        n_samples=len(df_features),
        pt_mult=labeling_params["pt_mult"],
        sl_mult=labeling_params["sl_mult"],
        max_holding=int(labeling_params["max_holding_period"]),
        min_ret=labeling_params.get("min_ret", 0.0),
    )

    # 3. Train Model
    print("\n" + "-"*30)
    print(f"TRAINING {model_name.upper()}")
    print("-"*30)

    # Prepare training data
    df_labeled = df_features.loc[train_mask].copy()
    y_train_full = labels.loc[train_mask]

    # Filter valid labels
    valid_mask = y_train_full.notna()
    df_train = df_labeled.loc[valid_mask]
    y_train = y_train_full.loc[valid_mask].astype(int)

    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    X_train = df_train[feature_cols].to_numpy()

    print(f"Training on {len(X_train)} samples...")

    model_class = get_model_class(model_name)
    train_params = dict(model_params or {})

    # Add class weights if supported
    if model_name in ("lightgbm", "xgboost", "catboost", "rf"):
        class_weight = compute_class_weights(y_train)
        if class_weight:
            train_params["class_weight"] = class_weight

    model = model_class(**train_params)
    model.fit(X_train, y_train)

    print("Training complete.")

    # 4. Save Model and Metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_model_name = f"model_{model_name}_{timestamp}.joblib"

    save_name = input(f"\nEnter filename to save model [{default_model_name}]: ").strip()
    if not save_name:
        save_name = default_model_name

    if not save_name.endswith(".joblib"):
        save_name += ".joblib"

    save_path = LABEL_PRIMAIRE_TRAIN_DIR / save_name
    joblib.dump(model, save_path)
    print(f"\nModel saved to: {save_path}")

    # Save metadata (critical for Evaluation step)
    metadata_path = save_path.with_suffix(".meta.json")
    metadata = {
        "model_name": model_name,
        "labeling_params": labeling_params,
        "model_params": model_params,
        "feature_cols": feature_cols,
        "train_samples": len(X_train),
        "timestamp": timestamp,
        "source_params_file": choice
    }
    save_json_pretty(metadata, metadata_path)
    print(f"Metadata saved to: {metadata_path}")

    print("\nNext step: Run 'python -m src.label_primaire_eval.main' to evaluate and generate dataset.")

if __name__ == "__main__":
    main()
