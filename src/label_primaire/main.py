"""Main entry point for label_primaire module.

This module provides:
1. Joint optimization of labeling + primary model parameters (run ONCE)
2. Generation of triple-barrier labels on FULL dataset
3. Save results for use by label_meta module

Supported primary models:
    - lightgbm (default)
    - xgboost
    - catboost
    - rf (Random Forest)
    - ridge
    - lasso
    - logistic

Usage:
    python -m src.label_primaire.main

Output files:
    - joint_optimization.json: Best labeling + model params
    - events_train.parquet: Events for training split
    - events_test.parquet: Events for test split
    - dataset_features_label.parquet: Features with labels
"""

from __future__ import annotations

import sys
import os

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import (
    DEFAULT_MAX_HOLDING,
    DEFAULT_PT_MULT,
    DEFAULT_SL_MULT,
)
from src.label_primaire.optimize import (
    AVAILABLE_MODELS,
    optimize_joint_params,
    normalize_volatility_scale,
    save_optimization_results,
)
from src.label_primaire.triple_barrier import get_triple_barrier_events
from src.path import (
    DATASET_FEATURES_LABEL_CSV,
    DATASET_FEATURES_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_LABEL_CSV,
    DATASET_FEATURES_LINEAR_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_PARQUET,
    DATASET_FEATURES_LSTM_LABEL_CSV,
    DATASET_FEATURES_LSTM_LABEL_PARQUET,
    DATASET_FEATURES_LSTM_PARQUET,
    DATASET_FEATURES_PARQUET,
    LABEL_PRIMAIRE_DIR,
    LABEL_PRIMAIRE_OPTIMIZATION_FILE,
    LABEL_PRIMAIRE_LABELING_PARAMS_FILE,
    LABEL_PRIMAIRE_EVENTS_TRAIN_FILE,
    LABEL_PRIMAIRE_EVENTS_TEST_FILE,
)
from src.utils import load_json_data, save_json_pretty

logger = get_logger(__name__)


# =============================================================================
# USER INPUT HELPERS
# =============================================================================


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    default_str = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "oui", "o")


def ask_integer(prompt: str, default: int, min_val: int = 1, max_val: int = 10000) -> int:
    """Ask for an integer value."""
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            val = int(answer)
            if min_val <= val <= max_val:
                return val
            print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid integer")


def ask_model_choice(default: str = "lightgbm") -> str:
    """Ask user to choose a primary model."""
    print("\nAvailable primary models:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        marker = " (default)" if model == default else ""
        print(f"  {i}. {model}{marker}")

    while True:
        answer = input(f"\nChoose a model (number or name) [{default}]: ").strip().lower()

        if not answer:
            return default

        # Try as number
        try:
            idx = int(answer)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[idx - 1]
            else:
                print(f"Please enter a number between 1 and {len(AVAILABLE_MODELS)}")
                continue
        except ValueError:
            pass

        # Try as name
        if answer in AVAILABLE_MODELS:
            return answer

        # Try partial match
        matches = [m for m in AVAILABLE_MODELS if m.startswith(answer)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous: {matches}. Please be more specific.")
        else:
            print(f"Unknown model: {answer}. Available: {AVAILABLE_MODELS}")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load features dataset with prices and volatility.

    Returns:
        Tuple of (df_features, prices, volatility rescaled to log-return units).
    """
    if not DATASET_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Features not found: {DATASET_FEATURES_PARQUET}. "
            "Please run 'python -m src.features.main' first."
        )

    df = pd.read_parquet(DATASET_FEATURES_PARQUET)
    logger.info("Loaded %d samples from %s", len(df), DATASET_FEATURES_PARQUET)

    # Extract prices
    prices = pd.Series(df["close"].values)

    # Extract volatility (already computed in features)
    vol_col = "realized_vol_20"
    if vol_col not in df.columns:
        available = [c for c in df.columns if "vol" in c.lower()]
        if not available:
            raise ValueError("No volatility column found in features")
        vol_col = available[0]

    volatility = pd.Series(df[vol_col].values)
    logger.info("Using volatility column: %s", vol_col)

    # Volatility is computed on log returns that are stored in percentage points (x100).
    # Rescale to match the log-return units derived from prices before feeding
    # the triple-barrier.
    volatility = normalize_volatility_scale(volatility, prices)

    return df, prices, volatility


# =============================================================================
# LABEL GENERATION
# =============================================================================


def generate_labels(
    prices: pd.Series,
    volatility: pd.Series,
    n_samples: int,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    min_ret: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate triple-barrier labels on full dataset.

    Args:
        prices: Close price series.
        volatility: Volatility series.
        n_samples: Number of samples in the dataset.
        pt_mult: Profit-taking multiplier.
        sl_mult: Stop-loss multiplier.
        max_holding: Maximum holding period.
        min_ret: Minimum return threshold.

    Returns:
        Tuple of (events DataFrame, labels Series).

    Notes:
        The volatility series is automatically rescaled if it appears to be
        expressed in percentage points (log returns x100) so that triple-barrier
        thresholds align with price returns.
    """
    logger.info("=" * 60)
    logger.info("Generating triple-barrier labels")
    logger.info("  pt_mult: %.3f", pt_mult)
    logger.info("  sl_mult: %.3f", sl_mult)
    logger.info("  max_holding: %d", max_holding)
    logger.info("  min_ret: %.6f", min_ret)
    logger.info("=" * 60)

    volatility = normalize_volatility_scale(volatility, prices)

    events = get_triple_barrier_events(
        prices=prices,
        pt_sl=[pt_mult, sl_mult],
        target_volatility=volatility,
        max_holding_period=max_holding,
        min_ret=min_ret,
    )

    # Create label series aligned with samples
    labels = pd.Series(index=range(n_samples), dtype=float)
    for _, row in events.iterrows():
        t_start = int(row["t_start"])
        if t_start < len(labels):
            labels.iloc[t_start] = row["label"]

    logger.info("Generated %d events, %d valid labels", len(events), labels.notna().sum())

    return events, labels


def save_labeled_datasets(labels: pd.Series) -> None:
    """Save labeled datasets (features + labels) for all model types."""
    # Tree-based models (raw features)
    if DATASET_FEATURES_PARQUET.exists():
        df = pd.read_parquet(DATASET_FEATURES_PARQUET)
        df["label"] = labels.iloc[:len(df)].values
        df.to_parquet(DATASET_FEATURES_LABEL_PARQUET)
        df.to_csv(DATASET_FEATURES_LABEL_CSV, index=False)
        logger.info("Saved labeled features to %s", DATASET_FEATURES_LABEL_PARQUET)

    # Linear models (z-scored)
    if DATASET_FEATURES_LINEAR_PARQUET.exists():
        df = pd.read_parquet(DATASET_FEATURES_LINEAR_PARQUET)
        df["label"] = labels.iloc[:len(df)].values
        df.to_parquet(DATASET_FEATURES_LINEAR_LABEL_PARQUET)
        df.to_csv(DATASET_FEATURES_LINEAR_LABEL_CSV, index=False)
        logger.info("Saved labeled linear features to %s", DATASET_FEATURES_LINEAR_LABEL_PARQUET)

    # LSTM (min-max scaled)
    if DATASET_FEATURES_LSTM_PARQUET.exists():
        df = pd.read_parquet(DATASET_FEATURES_LSTM_PARQUET)
        df["label"] = labels.iloc[:len(df)].values
        df.to_parquet(DATASET_FEATURES_LSTM_LABEL_PARQUET)
        df.to_csv(DATASET_FEATURES_LSTM_LABEL_CSV, index=False)
        logger.info("Saved labeled LSTM features to %s", DATASET_FEATURES_LSTM_LABEL_PARQUET)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point for label_primaire module."""
    print("\n" + "=" * 60)
    print("   LABEL PRIMAIRE - Triple-Barrier Labeling")
    print("=" * 60)
    print("\nThis module will:")
    print("  1. Optimize labeling + primary model parameters (joint)")
    print("  2. Generate labels on the FULL dataset")
    print("  3. Save results for use by label_meta module")
    print("\n")

    # Create output directory
    LABEL_PRIMAIRE_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df_features, prices, volatility = load_features()
    n_samples = len(df_features)
    print(f"Loaded {n_samples} samples with {len(df_features.columns)} features")

    # Restrict optimization to train split when available
    if "split" in df_features.columns:
        train_mask = df_features["split"] == "train"
        df_features_opt = df_features.loc[train_mask].reset_index(drop=True)
        prices_opt = prices.loc[train_mask.values].reset_index(drop=True)
        volatility_opt = volatility.loc[train_mask.values].reset_index(drop=True)
        print(f"Using train split for optimization: {len(df_features_opt)} samples")
    else:
        df_features_opt = df_features
        prices_opt = prices
        volatility_opt = volatility
        print("No 'split' column found; using full dataset for optimization")

    # Check if optimization results already exist
    optimization_file = LABEL_PRIMAIRE_OPTIMIZATION_FILE
    model_name = "lightgbm"  # default

    if optimization_file.exists():
        print(f"\nOptimization results found: {optimization_file}")
        use_existing = ask_yes_no("Use existing optimization results?", default=True)

        if use_existing:
            existing_data = load_json_data(optimization_file)
            labeling_params = existing_data.get("best_labeling_params", {})
            model_params = existing_data.get("best_model_params", {})
            model_name = existing_data.get("model_name", "lightgbm")
            print(f"Model: {model_name.upper()}")
            print(f"Loaded labeling params: {labeling_params}")
            print(f"Loaded model params: {model_params}")
        else:
            labeling_params = None
            model_params = None
    else:
        labeling_params = None
        model_params = None

    # Run optimization if needed
    if labeling_params is None:
        optimize = ask_yes_no("Run joint optimization (labeling + primary model)?", default=True)

        if optimize:
            # Choose primary model
            model_name = ask_model_choice(default="lightgbm")
            print(f"\nSelected primary model: {model_name.upper()}")

            n_trials = ask_integer("Number of Optuna trials", default=50, min_val=10, max_val=500)
            n_splits = ask_integer("Number of CV splits", default=5, min_val=3, max_val=10)
            purge_gap = ask_integer("Purge gap (bars)", default=5, min_val=0, max_val=50)

            # Prepare feature matrix
            exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
            feature_cols = [c for c in df_features_opt.columns if c not in exclude_cols]
            X = df_features_opt[feature_cols].to_numpy()

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

            labeling_params = result.best_labeling_params
            model_params = result.best_model_params

            # Save optimization results
            save_optimization_results(result, optimization_file)
            print(f"\nSaved optimization results to {optimization_file}")

        else:
            # Use default parameters
            labeling_params = {
                "pt_mult": DEFAULT_PT_MULT,
                "sl_mult": DEFAULT_SL_MULT,
                "max_holding_period": DEFAULT_MAX_HOLDING,
                "min_ret": 0.0,
            }
            model_params = {}
            print(f"Using default labeling params: {labeling_params}")

    # Generate labels on full dataset
    print("\n" + "=" * 60)
    print("Generating labels on full dataset...")
    print("=" * 60)

    events, labels = generate_labels(
        prices=prices,
        volatility=volatility,
        n_samples=n_samples,
        pt_mult=labeling_params["pt_mult"],
        sl_mult=labeling_params["sl_mult"],
        max_holding=int(labeling_params["max_holding_period"]),
        min_ret=labeling_params.get("min_ret", 0.0),
    )

    # Save labeled datasets
    save_labeled_datasets(labels)

    # Split events into train/test
    if "split" in df_features.columns:
        train_mask = df_features["split"] == "train"
        n_train = int(train_mask.sum())
    else:
        n_train = int(n_samples * 0.8)

    events_train = events[events["t_start"] < n_train].copy()
    events_test = events[events["t_start"] >= n_train].copy()

    # Save events
    events_train.to_parquet(LABEL_PRIMAIRE_EVENTS_TRAIN_FILE)
    events_test.to_parquet(LABEL_PRIMAIRE_EVENTS_TEST_FILE)
    logger.info("Saved events to %s", LABEL_PRIMAIRE_DIR)

    # Save labeling params separately for easy access
    save_json_pretty(labeling_params, LABEL_PRIMAIRE_LABELING_PARAMS_FILE)

    # Summary
    print("\n" + "=" * 60)
    print("   LABEL PRIMAIRE COMPLETE")
    print("=" * 60)
    print(f"\nPrimary model: {model_name.upper()}")
    print(f"Labeling params: {labeling_params}")
    print(f"Model params (for meta-model): {model_params}")
    print(f"\nEvents: {len(events)} total")
    print(f"  Train: {len(events_train)}")
    print(f"  Test: {len(events_test)}")
    print(f"\nLabels distribution:")
    print(f"  Up (+1): {(labels == 1).sum()}")
    print(f"  Down (-1): {(labels == -1).sum()}")
    print(f"  Neutral (0): {(labels == 0).sum()}")
    print(f"\nOutput directory: {LABEL_PRIMAIRE_DIR}")
    print("\nNext step: Run 'python -m src.label_meta.main' to train meta-model")
    print("=" * 60)


if __name__ == "__main__":
    main()
