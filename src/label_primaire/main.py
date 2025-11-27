"""Main entry point for label_primaire module.

This module provides:
1. Joint optimization of labeling + primary model parameters (run ONCE then reused)
2. Train/evaluate the optimized primary model on train/test splits
3. Generation of triple-barrier labels on FULL dataset
4. Save results for use by label_meta module

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
    - primary_evaluation.json: Train/test metrics with optimized params
    - events_train.parquet: Events for training split
    - events_test.parquet: Events for test split
    - dataset_features_label.parquet: Features with labels
"""

from __future__ import annotations

import sys
import os
from typing import Any

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import joblib  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_meta.main import get_primary_model_path  # Import from label_meta
from src.label_primaire.optimize import (
    AVAILABLE_MODELS,
    compute_class_weights,
    get_model_class,
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
    LABEL_PRIMAIRE_EVALUATION_FILE,
)
from src.utils import load_json_data, save_json_pretty

logger = get_logger(__name__)


# =============================================================================
# USER INPUT HELPERS
# =============================================================================


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask user for a yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "oui", "o"):
            return True
        if answer in ("n", "no", "non"):
            return False
        print("Please answer 'y' or 'n'")


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


def save_primary_model(model: Any, save_path: Path) -> None:
    """Save a trained primary model to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("Saved primary model to %s", save_path)


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
# PARAMETER HANDLING
# =============================================================================


def get_model_family(model_name: str) -> str:
    """Return model family: ml, linear, or lstm."""
    model_name = model_name.lower()
    if model_name == "lstm":
        return "lstm"
    if model_name in ("ridge", "lasso", "logistic", "logistic_regression"):
        return "linear"
    return "ml"


def load_saved_parameters() -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Load previously saved labeling and model parameters if they exist."""
    if LABEL_PRIMAIRE_OPTIMIZATION_FILE.exists():
        data = load_json_data(LABEL_PRIMAIRE_OPTIMIZATION_FILE)
        labeling_params = data.get("best_labeling_params")
        model_params = data.get("best_model_params", {})
        model_name = data.get("model_name")

        if labeling_params:
            return labeling_params, model_params, model_name

    if LABEL_PRIMAIRE_LABELING_PARAMS_FILE.exists():
        params = load_json_data(LABEL_PRIMAIRE_LABELING_PARAMS_FILE)
        return params, None, None

    return None, None, None


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


def save_labeled_datasets(
    labels: pd.Series,
    model_name: str,
    test_mask: pd.Series,
    df_features: pd.DataFrame,
) -> None:
    """
    Save labeled datasets for the selected model family.

    Labels are now preserved on both train and test splits to make the exported
    datasets directly usable for training and evaluation.
    """
    family = get_model_family(model_name)
    labels_for_save = labels.reindex(df_features.index).astype(float)

    if family == "ml":
        df = df_features.copy()
        df["label"] = labels_for_save.values
        df.to_parquet(DATASET_FEATURES_LABEL_PARQUET)
        df.to_csv(DATASET_FEATURES_LABEL_CSV, index=False)
        logger.info("Saved labeled ML features to %s", DATASET_FEATURES_LABEL_PARQUET)

    elif family == "linear":
        if not DATASET_FEATURES_LINEAR_PARQUET.exists():
            logger.warning("Linear features not found at %s", DATASET_FEATURES_LINEAR_PARQUET)
        else:
            df = pd.read_parquet(DATASET_FEATURES_LINEAR_PARQUET)
            df["label"] = labels_for_save.iloc[: len(df)].values
            df.to_parquet(DATASET_FEATURES_LINEAR_LABEL_PARQUET)
            df.to_csv(DATASET_FEATURES_LINEAR_LABEL_CSV, index=False)
            logger.info("Saved labeled LINEAR features to %s", DATASET_FEATURES_LINEAR_LABEL_PARQUET)

    elif family == "lstm":
        if not DATASET_FEATURES_LSTM_PARQUET.exists():
            logger.warning("LSTM features not found at %s", DATASET_FEATURES_LSTM_PARQUET)
        else:
            df = pd.read_parquet(DATASET_FEATURES_LSTM_PARQUET)
            df["label"] = labels_for_save.iloc[: len(df)].values
            df.to_parquet(DATASET_FEATURES_LSTM_LABEL_PARQUET)
            df.to_csv(DATASET_FEATURES_LSTM_LABEL_CSV, index=False)
            logger.info("Saved labeled LSTM features to %s", DATASET_FEATURES_LSTM_LABEL_PARQUET)

    else:
        logger.warning("Unknown model family for %s, no labeled dataset saved.", model_name)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute basic classification metrics while ignoring NaNs."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    valid_mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    y_true_valid = y_true_arr[valid_mask].astype(int)
    y_pred_valid = y_pred_arr[valid_mask].astype(int)

    if len(y_true_valid) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "mcc": 0.0, "n_samples": 0}

    return {
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "f1_macro": float(
            f1_score(
                y_true_valid,
                y_pred_valid,
                average="macro",
                zero_division="warn",
            )
        ),
        "mcc": float(matthews_corrcoef(y_true_valid, y_pred_valid)),
        "n_samples": int(len(y_true_valid)),
    }


def train_and_evaluate_primary_model(
    df_features: pd.DataFrame,
    labels: pd.Series,
    model_name: str,
    model_params: dict[str, Any] | None,
    train_mask: pd.Series,
) -> dict[str, Any] | None:
    """Train the primary model on the train split and evaluate on the test split."""
    df_labeled = df_features.copy()
    df_labeled["label"] = labels.iloc[: len(df_labeled)].values

    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_labeled.columns if c not in exclude_cols]

    if not feature_cols:
        logger.warning("No feature columns available for training the primary model.")
        return None

    valid_mask = df_labeled["label"].notna()
    train_mask_valid = train_mask & valid_mask
    test_mask_valid = (~train_mask) & valid_mask

    n_train = int(train_mask_valid.sum())
    n_test = int(test_mask_valid.sum())

    if n_train == 0 or n_test == 0:
        logger.warning(
            "Insufficient labeled samples for training/evaluation (train=%d, test=%d).",
            n_train,
            n_test,
        )
        return None

    X_train = df_labeled.loc[train_mask_valid, feature_cols].to_numpy()
    y_train = df_labeled.loc[train_mask_valid, "label"].astype(int)
    X_test = df_labeled.loc[test_mask_valid, feature_cols].to_numpy()
    y_test = df_labeled.loc[test_mask_valid, "label"].astype(int)

    model_class = get_model_class(model_name)
    params = dict(model_params or {})
    params.setdefault("random_state", DEFAULT_RANDOM_STATE)

    if model_name in ("lightgbm", "xgboost", "catboost", "rf"):
        class_weight = compute_class_weights(y_train)
        if class_weight:
            params.setdefault("class_weight", class_weight)

    logger.info(
        "Training %s on %d samples, evaluating on %d samples",
        model_name.upper(),
        n_train,
        n_test,
    )

    model = model_class(**params)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    class_weight_serialized = None
    if params.get("class_weight"):
        class_weight_serialized = {str(k): float(v) for k, v in params["class_weight"].items()}

    return {
        "model": model,  # Add trained model to return value
        "train_metrics": compute_classification_metrics(y_train.to_numpy(), train_pred),
        "test_metrics": compute_classification_metrics(y_test.to_numpy(), test_pred),
        "n_train": n_train,
        "n_test": n_test,
        "class_weight": class_weight_serialized,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point for label_primaire module."""
    print("\n" + "=" * 60)
    print("   LABEL PRIMAIRE - Triple-Barrier Labeling")
    print("=" * 60)
    print("\nThis module will:")
    print("  1. Optimize labeling + primary model parameters (once)")
    print("  2. Train the primary model on train split and evaluate on test")
    print("  3. Generate labels on the FULL dataset")
    print("  4. Save results for use by label_meta module")
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
        n_train = int(train_mask.sum())
        test_mask = df_features["split"] == "test"
        if int(test_mask.sum()) == 0:
            test_mask = ~train_mask
        df_features_opt = df_features.loc[train_mask].reset_index(drop=True)
        prices_opt = prices.loc[train_mask.values].reset_index(drop=True)
        volatility_opt = volatility.loc[train_mask.values].reset_index(drop=True)
        print(f"Using train split for optimization: {len(df_features_opt)} samples")
    else:
        n_train = int(n_samples * 0.8)
        train_mask = pd.Series(
            [True] * n_train + [False] * (n_samples - n_train),
            index=df_features.index,
        )
        test_mask = ~train_mask
        df_features_opt = df_features
        prices_opt = prices
        volatility_opt = volatility
        print("No 'split' column found; using full dataset for optimization")

    labeling_params, model_params, saved_model_name = load_saved_parameters()

    # Always allow model choice
    print("\n" + "="*50)
    print("MODEL SELECTION")
    print("="*50)

    if labeling_params is not None and saved_model_name:
        print(f"Found saved parameters for model: {saved_model_name.upper()}")
        use_saved = ask_yes_no(
            f"Use existing parameters for {saved_model_name.upper()}?",
            default=True
        )
        if use_saved:
            model_name = saved_model_name
            params_source = "existing"
            print(f"Using saved parameters for {model_name.upper()}")
        else:
            print("Choosing a different model...")
            model_name = ask_model_choice(default="lightgbm")
            params_source = "optimized"
            labeling_params = None  # Force re-optimization
    else:
        model_name = ask_model_choice(default="lightgbm")
        params_source = "optimized"

    print(f"\nSelected primary model: {model_name.upper()}")

    if labeling_params is None:
        print("\nStarting joint optimization...")
        n_trials = ask_integer("Number of Optuna trials", default=200, min_val=10, max_val=1000)
        n_splits = ask_integer("Number of CV splits", default=5, min_val=3, max_val=10)
        purge_gap = ask_integer("Purge gap (bars)", default=5, min_val=0, max_val=50)

        # Prepare feature matrix
        exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
        feature_cols = [c for c in df_features_opt.columns if c not in exclude_cols]

        if not feature_cols:
            raise ValueError("No feature columns available for optimization.")

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
        model_name = result.model_name

        # Save optimization results
        save_optimization_results(result, LABEL_PRIMAIRE_OPTIMIZATION_FILE)
        print(f"\nSaved optimization results to {LABEL_PRIMAIRE_OPTIMIZATION_FILE}")
    else:
        model_params = model_params or {}
        print("\nSaved labeling parameters detected - skipping optimization step.")
        if saved_model_name:
            print(f"Model: {model_name.upper()}")
        print(f"Labeling params: {labeling_params}")
        if model_params:
            print(f"Model params: {model_params}")
        else:
            print("No saved primary model params found; using model defaults.")

    if labeling_params is None:
        raise RuntimeError("Labeling parameters could not be determined.")

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

    # Save labeled dataset only for the selected model family (labels kept on test split)
    save_labeled_datasets(
        labels=labels,
        model_name=model_name,
        test_mask=test_mask,
        df_features=df_features,
    )

    # Train and evaluate the primary model on the labeled data
    evaluation = train_and_evaluate_primary_model(
        df_features=df_features,
        labels=labels,
        model_name=model_name,
        model_params=model_params,
        train_mask=train_mask,
    )

    # Save the trained primary model
    if evaluation and "model" in evaluation:
        primary_model_path = get_primary_model_path(model_name)
        save_primary_model(evaluation["model"], primary_model_path)
        print(f"Primary model saved to {primary_model_path}")

    if evaluation:
        eval_payload = {
            "model_name": model_name,
            "model_params": model_params or {},
            "labeling_params": labeling_params,
            "params_source": params_source,
            "train_metrics": evaluation["train_metrics"],
            "test_metrics": evaluation["test_metrics"],
            "n_train_labeled": evaluation["n_train"],
            "n_test_labeled": evaluation["n_test"],
            "class_weight": evaluation.get("class_weight"),
        }
        save_json_pretty(eval_payload, LABEL_PRIMAIRE_EVALUATION_FILE)

        print("\nPrimary model evaluation (train/test):")
        print(
            f"  Train - acc: {evaluation['train_metrics']['accuracy']:.4f}, "
            f"f1: {evaluation['train_metrics']['f1_macro']:.4f}, "
            f"mcc: {evaluation['train_metrics']['mcc']:.4f}, "
            f"n={evaluation['n_train']}"
        )
        print(
            f"  Test  - acc: {evaluation['test_metrics']['accuracy']:.4f}, "
            f"f1: {evaluation['test_metrics']['f1_macro']:.4f}, "
            f"mcc: {evaluation['test_metrics']['mcc']:.4f}, "
            f"n={evaluation['n_test']}"
        )
    else:
        print("\nPrimary model training/evaluation skipped (not enough labeled samples).")

    # Split events into train/test
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
    print(f"Labeling params source: {params_source}")
    print(f"Labeling params: {labeling_params}")
    print(f"Model params (for meta-model): {model_params}")
    print(f"\nEvents: {len(events)} total")
    print(f"  Train: {len(events_train)}")
    print(f"  Test: {len(events_test)}")
    print(f"\nLabels distribution:")
    print(f"  Up (+1): {(labels == 1).sum()}")
    print(f"  Down (-1): {(labels == -1).sum()}")
    print(f"  Neutral (0): {(labels == 0).sum()}")
    print("Labels saved for both train and test splits for the selected model family.")
    if evaluation:
        print(
            f"\nTest metrics → acc: {evaluation['test_metrics']['accuracy']:.4f}, "
            f"f1: {evaluation['test_metrics']['f1_macro']:.4f}, "
            f"mcc: {evaluation['test_metrics']['mcc']:.4f}"
        )
    print(f"\nOutput directory: {LABEL_PRIMAIRE_DIR}")
    print("\nNext step: Run 'python -m src.label_meta.main' to train meta-model")
    print("=" * 60)


if __name__ == "__main__":
    main()
