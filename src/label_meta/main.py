"""Main entry point for label_meta module.

This module provides:
1. Interactive selection of a primary model (load trained file or retrain)
2. Meta-model hyperparameter optimization per primary (run once, then reuse)
3. Training meta-model for each primary and applying it on test labels

Usage:
    python -m src.label_meta.main

Prerequisites:
    - Run 'python -m src.label_primaire.main' first to generate labels
"""

from __future__ import annotations

import os
import sys

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_meta.benchmark import compute_metrics
from src.label_meta.config import BenchmarkResult
from src.label_meta.meta_model import load_meta_model, save_meta_model
from src.label_meta.meta_labeling import get_meta_labels
from src.path import (
    DATASET_FEATURES_LABEL_PARQUET,
    DATASET_FEATURES_LABEL_CSV,
    DATASET_FEATURES_LINEAR_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_LABEL_CSV,
    DATASET_FEATURES_LSTM_LABEL_PARQUET,
    DATASET_FEATURES_LSTM_LABEL_CSV,
    LABEL_META_DIR,
    LABEL_PRIMAIRE_DIR,
    LABEL_META_BENCHMARKS_DIR,
    LABEL_META_MODELS_DIR,
    LABEL_META_PARAMS_DIR,
    LABEL_PRIMAIRE_MODELS_DIR,
    LABEL_PRIMAIRE_OPTIMIZATION_FILE,
    LABEL_PRIMAIRE_EVENTS_TRAIN_FILE,
    LABEL_PRIMAIRE_EVENTS_TEST_FILE,
)
from src.label_primaire.optimize import compute_class_weights, get_model_class
from src.optimisation.walk_forward_cv import create_cv_config, walk_forward_cv
from src.model.machine_learning.lightgbm.lightgbm_model import LightGBMModel
from src.utils import load_json_data, save_json_pretty

logger = get_logger(__name__)

# Available models for benchmarking
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


def ask_model_choice() -> str | None:
    """Ask user to choose a model for benchmarking."""
    print("\nAvailable models:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")
    print(f"  0. Exit/Done")

    while True:
        answer = input("\nChoose a model (number or name): ").strip().lower()

        if answer in ("0", "exit", "done", "q", "quit"):
            return None

        # Try as number
        try:
            idx = int(answer)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[idx - 1]
            elif idx == 0:
                return None
            else:
                print(f"Please enter a number between 0 and {len(AVAILABLE_MODELS)}")
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
            print(f"Unknown model: {answer}")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features and events for meta-model training.

    Picks the labeled dataset that matches the chosen model family (ML/linear/LSTM).

    Returns:
        Tuple of (features DataFrame, events_train, events_test).
    """
    dataset_parquet, dataset_csv = get_labeled_dataset_paths(model_name)

    if dataset_parquet.exists():
        df_features = pd.read_parquet(dataset_parquet)
        dataset_used = dataset_parquet
    elif dataset_csv.exists():
        df_features = pd.read_csv(dataset_csv)
        dataset_used = dataset_csv
    else:
        raise FileNotFoundError(
            f"Labeled features for {model_name.upper()} not found. "
            f"Expected {dataset_parquet} or {dataset_csv}. "
            "Please run 'python -m src.label_primaire.main' with the same model family first."
        )

    if not LABEL_PRIMAIRE_EVENTS_TRAIN_FILE.exists() or not LABEL_PRIMAIRE_EVENTS_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Events not found in {LABEL_PRIMAIRE_DIR}. "
            "Please run 'python -m src.label_primaire.main' first."
        )

    events_train = pd.read_parquet(LABEL_PRIMAIRE_EVENTS_TRAIN_FILE)
    events_test = pd.read_parquet(LABEL_PRIMAIRE_EVENTS_TEST_FILE)

    logger.info(
        "Loaded %d features from %s, %d train events, %d test events",
        len(df_features),
        dataset_used,
        len(events_train),
        len(events_test),
    )

    return df_features, events_train, events_test


def build_labels_from_events(
    df_features: pd.DataFrame,
    events_train: pd.DataFrame,
    events_test: pd.DataFrame,
) -> pd.Series:
    """Reconstruct label Series from events files for train and test splits."""
    labels = pd.Series(np.nan, index=df_features.index, dtype=float)
    n_samples = len(df_features)

    for events in (events_train, events_test):
        if not {"t_start", "label"}.issubset(events.columns):
            raise ValueError("Events files must contain 't_start' and 'label' columns.")

        positions = events["t_start"].to_numpy(dtype=int, copy=False)
        valid_mask = (positions >= 0) & (positions < n_samples)
        if not np.all(valid_mask):
            logger.warning("Some event indices fall outside feature range and were ignored.")

        labels.iloc[positions[valid_mask]] = events.loc[valid_mask, "label"].to_numpy()

    return labels


def prepare_splits(
    df_features: pd.DataFrame,
    events_train: pd.DataFrame,
    events_test: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Prepare train/test splits from features and events.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, events_train, events_test).
    """
    df_features = df_features.copy()

    # Get feature columns
    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    # Get labels and backfill missing ones from events
    labels = df_features["label"].copy() if "label" in df_features.columns else pd.Series(
        np.nan, index=df_features.index, dtype=float
    )
    if bool(labels.isna().any()):
        labels_from_events = build_labels_from_events(df_features, events_train, events_test)
        missing_mask = labels.isna()
        filled_mask = missing_mask & labels_from_events.notna()
        if filled_mask.any():
            labels.loc[filled_mask] = labels_from_events.loc[filled_mask]
            logger.info("Filled %d missing labels from events for meta training.", int(filled_mask.sum()))
        elif bool(labels.isna().all()):
            raise ValueError(
                "No labels available in the dataset; regenerate labels via 'python -m src.label_primaire.main'."
            )

    remaining_missing = int(labels.isna().sum())
    if remaining_missing > 0:
        logger.warning("Labels remain missing for %d samples; they will be ignored.", remaining_missing)

    if labels.notna().sum() == 0:
        raise ValueError("No labeled samples found after merging events.")

    df_features["label"] = labels

    # Determine split point
    if "split" in df_features.columns:
        train_mask = df_features["split"] == "train"
        n_train = int(train_mask.sum())
    else:
        n_train = int(len(df_features) * 0.8)

    # Split features
    X = df_features[feature_cols].to_numpy()
    X_train = X[:n_train]
    X_test = X[n_train:]

    # Split labels
    y_train = labels.iloc[:n_train]
    y_test = labels.iloc[n_train:]

    logger.info("Train: %d samples, Test: %d samples", len(y_train), len(y_test))

    return X_train, y_train, X_test, y_test, events_train, events_test


def load_saved_meta_params(model_name: str) -> dict[str, Any]:
    """Load previously optimized meta-model params for a primary model."""
    params_path = LABEL_META_PARAMS_DIR / f"meta_params_{model_name}.json"
    if params_path.exists():
        params = load_json_data(params_path)
        logger.info("Loaded meta params for %s from %s", model_name, params_path)
        return params
    return {}


def load_primary_params(model_name: str) -> dict[str, Any]:
    """Load primary model params from label_primaire optimization if matching."""
    if not LABEL_PRIMAIRE_OPTIMIZATION_FILE.exists():
        logger.warning("No optimization results found, primary params empty")
        return {}

    data = load_json_data(LABEL_PRIMAIRE_OPTIMIZATION_FILE)
    optimized_model = data.get("model_name")
    if optimized_model and optimized_model.lower() != model_name.lower():
        logger.info(
            "Optimization file is for model %s, not %s. Using defaults.",
            optimized_model,
            model_name,
        )
        return {}

    primary_params = data.get("best_model_params", {})
    logger.info(
        "Loaded primary params for %s from %s: %s",
        model_name,
        LABEL_PRIMAIRE_OPTIMIZATION_FILE,
        primary_params,
    )
    return primary_params


# =============================================================================
# PATH HELPERS
# =============================================================================


def get_primary_model_path(model_name: str) -> Path:
    """Default path for a trained primary model."""
    return LABEL_PRIMAIRE_MODELS_DIR / f"{model_name}_model.joblib"


def get_meta_model_path(model_name: str) -> Path:
    """Default path for a trained meta-model linked to a primary."""
    return LABEL_META_MODELS_DIR / f"meta_model_{model_name}.joblib"


def get_meta_params_path(model_name: str) -> Path:
    """Default path for meta-model hyperparameters."""
    return LABEL_META_PARAMS_DIR / f"meta_params_{model_name}.json"


# =============================================================================
# TRAINING HELPERS
# =============================================================================


def load_primary_model_from_file(model_name: str, path: Path) -> Any:
    """Load a previously trained primary model."""
    model = joblib.load(path)
    logger.info("Loaded primary model %s from %s", model_name, path)
    return model


def train_primary_model(
    model_name: str,
    primary_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    save_path: Path | None = None,
) -> Any:
    """Train primary model on train split."""
    valid_mask = ~y_train.isna()
    X_train_valid = X_train[valid_mask.to_numpy()]
    y_train_valid = y_train[valid_mask].astype(int)

    if len(y_train_valid) == 0:
        raise ValueError("No labeled samples available to train primary model.")

    model_class = get_model_class(model_name)
    params = dict(primary_params)
    params.setdefault("random_state", DEFAULT_RANDOM_STATE)

    if model_name in ("lightgbm", "xgboost", "catboost", "rf"):
        class_weight = compute_class_weights(y_train_valid)
        if class_weight:
            params.setdefault("class_weight", class_weight)

    model = model_class(**params)
    model.fit(X_train_valid, y_train_valid)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        logger.info("Saved primary model to %s", save_path)

    return model


def get_primary_predictions(
    model: Any,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    dict[str, float],
    dict[str, float],
    pd.Index,
    pd.Index,
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
]:
    """Generate train/test predictions and metrics for a primary model."""
    valid_train_mask = ~y_train.isna()
    valid_test_mask = ~y_test.isna()

    train_indices = cast(pd.Index, y_train.index[valid_train_mask])
    test_indices = cast(pd.Index, y_test.index[valid_test_mask])

    X_train_valid = X_train[valid_train_mask.to_numpy()]
    X_test_valid = X_test[valid_test_mask.to_numpy()]

    y_train_valid = cast(pd.Series, y_train[valid_train_mask].astype(int))
    y_test_valid = cast(pd.Series, y_test[valid_test_mask].astype(int))

    if len(y_train_valid) == 0 or len(y_test_valid) == 0:
        raise ValueError("Not enough labeled samples for train/test evaluation.")

    train_pred = model.predict(X_train_valid)
    test_pred = model.predict(X_test_valid)

    train_metrics = compute_metrics(y_train_valid.to_numpy(), train_pred)
    test_metrics = compute_metrics(y_test_valid.to_numpy(), test_pred)

    train_pred_series = pd.Series(train_pred, index=train_indices)
    test_pred_series = pd.Series(test_pred, index=test_indices)

    return (
        train_pred_series,
        test_pred_series,
        train_metrics,
        test_metrics,
        train_indices,
        test_indices,
        X_train_valid,
        X_test_valid,
        y_train_valid,
        y_test_valid,
    )


def build_meta_training_data(
    primary_train_signal: pd.Series,
    events_train: pd.DataFrame,
    X_train_valid: np.ndarray,
    train_indices: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """Build meta-model training set (features + labels)."""
    events_idxed = events_train.copy()
    events_idxed["t_start"] = events_idxed["t_start"].astype(int)
    events_idxed = events_idxed.set_index("t_start")

    meta_labels = get_meta_labels(events_idxed, primary_train_signal).dropna()
    if meta_labels.empty:
        raise ValueError("No meta-labels could be generated (no trades).")

    mask_meta = train_indices.isin(meta_labels.index)
    if not mask_meta.any():
        raise ValueError("Meta-labels do not align with training indices.")

    X_meta_base = X_train_valid[mask_meta]
    primary_meta_signal = primary_train_signal.loc[train_indices[mask_meta]].to_numpy()
    y_meta = meta_labels.loc[train_indices[mask_meta]].to_numpy().astype(int)

    X_meta_train = np.column_stack([X_meta_base, primary_meta_signal])
    return X_meta_train, y_meta


def train_meta_model_for_primary(
    meta_params: dict[str, Any],
    X_meta_train: np.ndarray,
    y_meta_train: np.ndarray,
    save_path: Path,
) -> Any:
    """Train the meta-model for a given primary model and save it."""
    params = dict(meta_params)
    params.setdefault("random_state", DEFAULT_RANDOM_STATE)

    meta_model = LightGBMModel(**params)
    meta_model.fit(X_meta_train, y_meta_train)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta_model(meta_model, save_path)

    return meta_model


# =============================================================================
# RESULTS MANAGEMENT
# =============================================================================


def save_benchmark_result(result: BenchmarkResult, output_dir: Path) -> None:
    """Save benchmark result to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = result.primary_model_name

    # Save metrics
    metrics = {
        "primary_model_name": model_name,
        "primary_params": result.primary_params,
        "meta_params": result.meta_params,
        "train_metrics": result.train_metrics,
        "test_metrics": result.test_metrics,
        "final_metrics": result.final_metrics,
    }
    save_json_pretty(metrics, output_dir / f"benchmark_{model_name}.json")

    # Save predictions
    predictions_df = pd.DataFrame({
        "test_prediction": result.test_predictions,
        "meta_prediction": result.meta_predictions,
        "final_prediction": result.final_predictions,
    })
    predictions_df.to_parquet(output_dir / f"predictions_{model_name}.parquet")

    logger.info("Saved benchmark results to %s", output_dir)


def load_all_benchmark_results(output_dir: Path) -> list[dict[str, Any]]:
    """Load all benchmark results from directory."""
    results = []

    for json_file in output_dir.glob("benchmark_*.json"):
        data = load_json_data(json_file)
        results.append(data)

    return results


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print comparison table of all benchmark results."""
    if not results:
        print("No benchmark results found.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<15} {'Test MCC':>10} {'Final MCC':>10} {'Improvement':>12} {'Trades':>8}")
    print("-" * 55)

    for r in sorted(results, key=lambda x: x.get("final_metrics", {}).get("mcc", 0), reverse=True):
        model = r.get("primary_model_name", "unknown")
        test_mcc = r.get("test_metrics", {}).get("mcc", 0)
        final_mcc = r.get("final_metrics", {}).get("mcc", 0)
        improvement = final_mcc - test_mcc
        n_samples = r.get("final_metrics", {}).get("n_samples", 0)

        print(f"{model:<15} {test_mcc:>10.4f} {final_mcc:>10.4f} {improvement:>+12.4f} {n_samples:>8}")

    print("-" * 55)
    print("\n")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Main entry point for label_meta module."""
    print("\n" + "=" * 60)
    print("   LABEL META - Meta-Model Training & Application")
    print("=" * 60)
    print("\nThis module will:")
    print("  1. Let you pick a PRIMARY model (load a trained file or train now)")
    print("  2. Optimize the META-model hyperparams once per primary")
    print("  3. Train the META-model and apply it on the test split")
    print("     following De Prado's two-stage labeling")
    print("\n")

    # Create output directory
    LABEL_META_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_META_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_META_PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        model_choice = ask_model_choice()
        if model_choice is None:
            break

        print("\n" + "=" * 60)
        print(f"PRIMARY MODEL: {model_choice.upper()}")
        print("=" * 60)

        print("\nLoading data...")
        try:
            df_features, events_train, events_test = load_data(model_choice)
            X_train, y_train, X_test, y_test, events_train, events_test = prepare_splits(
                df_features, events_train, events_test
            )
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nPlease run 'python -m src.label_primaire.main' first.")
            continue
        except ValueError as e:
            print(f"\nError preparing data for {model_choice.upper()}: {e}")
            continue

        primary_model_path = get_primary_model_path(model_choice)
        primary_model: Any | None = None
        primary_params: dict[str, Any] = {}

        # Try loading an existing primary model
        if primary_model_path.exists():
            use_primary_file = ask_yes_no(
                f"Load existing trained primary model from {primary_model_path}?",
                default=True,
            )
            if use_primary_file:
                primary_model = load_primary_model_from_file(model_choice, primary_model_path)
                primary_params = {"source_path": str(primary_model_path)}

        if primary_model is None:
            custom_primary_path = input(
                f"Path to a trained primary model (.joblib) or press Enter to train now "
                f"[{primary_model_path if primary_model_path.exists() else 'train'}]: "
            ).strip()

            if custom_primary_path:
                path_obj = Path(custom_primary_path)
                primary_model = load_primary_model_from_file(model_choice, path_obj)
                primary_params = {"source_path": str(path_obj)}
            else:
                primary_params = load_primary_params(model_choice)
                print(f"Training {model_choice.upper()} with params: {primary_params or 'defaults'}")
                should_save_primary = ask_yes_no(
                    f"Save trained primary model to {primary_model_path}?", default=True
                )
                save_path = primary_model_path if should_save_primary else None
                primary_model = train_primary_model(
                    model_name=model_choice,
                    primary_params=primary_params,
                    X_train=X_train,
                    y_train=y_train,
                    save_path=save_path,
                )

        # Primary predictions and metrics
        (
            train_pred,
            test_pred,
            train_metrics,
            test_metrics,
            train_indices,
            _,  # test_indices (unused)
            X_train_valid,
            X_test_valid,
            _,
            y_test_valid,
        ) = get_primary_predictions(
            model=primary_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        print("\nPrimary model metrics:")
        print(f"  Train -> acc: {train_metrics['accuracy']:.4f}, f1: {train_metrics['f1_macro']:.4f}, mcc: {train_metrics['mcc']:.4f}")
        print(f"  Test  -> acc: {test_metrics['accuracy']:.4f}, f1: {test_metrics['f1_macro']:.4f}, mcc: {test_metrics['mcc']:.4f}")

        # Build meta training data
        try:
            X_meta_train, y_meta_train = build_meta_training_data(
                primary_train_signal=train_pred,
                events_train=events_train,
                X_train_valid=X_train_valid,
                train_indices=train_indices,
            )
        except ValueError as e:
            print(f"\nCannot train meta-model for {model_choice.upper()}: {e}")
            continue

        meta_model_path = get_meta_model_path(model_choice)
        meta_params_path = get_meta_params_path(model_choice)
        meta_model: Any | None = None
        meta_params = load_saved_meta_params(model_choice)

        # Load existing meta-model if available
        if meta_model_path.exists():
            use_existing_meta = ask_yes_no(
                f"Meta-model for {model_choice.upper()} found at {meta_model_path}. Use it?",
                default=True,
            )
            if use_existing_meta:
                meta_model = load_meta_model(meta_model_path)
                if not meta_params:
                    meta_params = load_saved_meta_params(model_choice)

        # Use primary model params for meta-model (no optimization)
        if meta_model is None:
            if not meta_params:
                # Use LightGBM params from primary model if primary is LightGBM, else use defaults
                if model_choice.lower() == "lightgbm":
                    primary_params = load_primary_params(model_choice)
                    meta_params = dict(primary_params)  # Use LightGBM params from primary
                    source_desc = "primary LightGBM params"
                else:
                    # For non-LightGBM primary models, use default LightGBM params
                    meta_params = {
                        "random_state": DEFAULT_RANDOM_STATE,
                        # Default LightGBM params for meta-model
                        "learning_rate": 0.1,
                        "num_leaves": 31,
                        "max_depth": 6,
                        "min_data_in_leaf": 20,
                        "feature_fraction": 0.8,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 1,
                        "verbosity": -1,
                    }
                    source_desc = "default LightGBM params"

                print(f"\nUsing {source_desc} for meta-model: {list(meta_params.keys())}")

                # Save meta params for consistency
                meta_params_payload = {
                    "meta_params": meta_params,
                    "source": source_desc,
                }
                save_json_pretty(meta_params_payload, meta_params_path)
                print(f"Saved meta params ({source_desc}) to {meta_params_path}")
            else:
                print(f"\nUsing saved meta params: {meta_params}")

            meta_model = cast(LightGBMModel, train_meta_model_for_primary(
                meta_params=meta_params,
                X_meta_train=X_meta_train,
                y_meta_train=y_meta_train,
                save_path=meta_model_path,
            ))
            print(f"Meta-model saved to {meta_model_path}")

        # Apply meta-model on test split
        meta_features_test = np.column_stack([X_test_valid, test_pred.to_numpy()])
        meta_predictions = meta_model.predict(meta_features_test)
        final_predictions = test_pred.to_numpy() * meta_predictions

        final_metrics = compute_metrics(y_test_valid.to_numpy(), final_predictions)

        result = BenchmarkResult(
            primary_model_name=model_choice,
            primary_params=primary_params,
            meta_params=meta_params,
            primary_model=primary_model,
            meta_model=meta_model,
            train_predictions=train_pred.to_numpy(),
            test_predictions=test_pred.to_numpy(),
            meta_predictions=np.asarray(meta_predictions),
            final_predictions=np.asarray(final_predictions),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            final_metrics=final_metrics,
        )

        save_benchmark_result(result, LABEL_META_BENCHMARKS_DIR)

        print("\nMeta-model results:")
        print(f"  Final test -> acc: {final_metrics['accuracy']:.4f}, f1: {final_metrics['f1_macro']:.4f}, mcc: {final_metrics['mcc']:.4f}")
        trades_filtered = int((np.asarray(meta_predictions) == 0).sum())
        total_meta = len(meta_predictions)
        filtered_ratio = (100 * trades_filtered / total_meta) if total_meta else 0.0
        print(
            f"  Trades filtered by meta-model: {trades_filtered} / {len(meta_predictions)} "
            f"({filtered_ratio:.1f}% filtered)"
        )

    print("\n" + "=" * 60)
    print("Meta-labeling session complete.")
    print(f"Artifacts saved under: {LABEL_META_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
