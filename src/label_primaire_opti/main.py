"""Main entry point for label_primaire_opti module.

This module is responsible for:
1. Selecting the primary model type (econometric, ML, DL).
2. Optimizing the labeling parameters (Triple Barrier) and model hyperparameters.
3. Saving the best parameters to a user-specified file.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from src.config_logging import get_logger
from src.label_primaire.optimize import (
    AVAILABLE_MODELS,
    optimize_joint_params,
    save_optimization_results,
    normalize_volatility_scale,
)
from src.path import (
    DATASET_FEATURES_PARQUET,
    LABEL_PRIMAIRE_OPTI_DIR,
    LABEL_PRIMAIRE_OPTIMIZATION_FILE,
)
from src.utils.user_input import ask_choice, ask_integer, ask_yes_no, ask_float
from src.utils import ensure_output_dir

logger = get_logger(__name__)

def load_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load features dataset with prices and volatility."""
    if not DATASET_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Features not found: {DATASET_FEATURES_PARQUET}. "
            "Please run 'python -m src.features.main' first."
        )

    df = pd.read_parquet(DATASET_FEATURES_PARQUET)
    logger.info("Loaded %d samples from %s", len(df), DATASET_FEATURES_PARQUET)

    prices = pd.Series(df["close"].values)

    # Extract volatility
    vol_col = "realized_vol_20"
    if vol_col not in df.columns:
        available = [c for c in df.columns if "vol" in c.lower()]
        if not available:
            raise ValueError("No volatility column found in features")
        vol_col = available[0]

    volatility = pd.Series(df[vol_col].values)

    # Rescale volatility
    volatility = normalize_volatility_scale(volatility, prices)

    return df, prices, volatility

def main() -> None:
    print("\n" + "=" * 60)
    print("   LABEL PRIMAIRE - OPTIMIZATION")
    print("=" * 60)

    LABEL_PRIMAIRE_OPTI_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\nLoading data...")
    df_features, prices, volatility = load_features()

    # Prepare data for optimization (use train split if available)
    if "split" in df_features.columns:
        train_mask = df_features["split"] == "train"
        df_features_opt = df_features.loc[train_mask].reset_index(drop=True)
        prices_opt = prices.loc[train_mask.values].reset_index(drop=True)
        volatility_opt = volatility.loc[train_mask.values].reset_index(drop=True)
        print(f"Using train split for optimization: {len(df_features_opt)} samples")
    else:
        # Fallback to 80% split
        n_train = int(len(df_features) * 0.8)
        df_features_opt = df_features.iloc[:n_train].reset_index(drop=True)
        prices_opt = prices.iloc[:n_train].reset_index(drop=True)
        volatility_opt = volatility.iloc[:n_train].reset_index(drop=True)
        print("No 'split' column found; using first 80% for optimization")

    # 2. Select Model
    print("\n" + "-"*30)
    print("MODEL SELECTION")
    print("-"*30)
    model_name = ask_choice(AVAILABLE_MODELS, prompt="Choose a primary model type", default="lightgbm")

    # 3. Optimization Settings
    print("\n" + "-"*30)
    print("OPTIMIZATION SETTINGS")
    print("-"*30)
    n_trials = ask_integer("Number of Optuna trials", default=100, min_val=10, max_val=1000)
    n_splits = ask_integer("Number of CV splits", default=5, min_val=3, max_val=10)
    purge_gap = ask_integer("Purge gap (bars)", default=5, min_val=0, max_val=100)

    # Prepare feature matrix
    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_features_opt.columns if c not in exclude_cols]

    if not feature_cols:
        raise ValueError("No feature columns available for optimization.")

    X = df_features_opt[feature_cols].to_numpy()

    # 4. Run Optimization
    print("\n" + "=" * 60)
    print(f"Starting joint optimization with {model_name.upper()}...")
    print("=" * 60)

    result = optimize_joint_params(
        prices=prices_opt,
        X=X,
        volatility=volatility_opt,
        model_name=model_name,
        n_trials=n_trials,
        n_splits=n_splits,
        purge_gap=purge_gap,
        symmetric_barriers=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Optimization Complete")
    print(f"Best Score (MCC): {result.best_score:.4f}")
    print("=" * 60)

    # 5. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"params_{model_name}_{timestamp}.json"

    save_name = input(f"\nEnter filename to save parameters [{default_name}]: ").strip()
    if not save_name:
        save_name = default_name

    if not save_name.endswith(".json"):
        save_name += ".json"

    save_path = LABEL_PRIMAIRE_OPTI_DIR / save_name
    save_optimization_results(result, save_path)
    print(f"\nParameters saved to: {save_path}")

    # Also save to default/latest location for backward compatibility or easy loading
    save_optimization_results(result, LABEL_PRIMAIRE_OPTIMIZATION_FILE)
    print(f"Also updated default parameters file: {LABEL_PRIMAIRE_OPTIMIZATION_FILE}")

    print("\nNext step: Run 'python -m src.label_primaire_train.main' to train the model.")

if __name__ == "__main__":
    main()
