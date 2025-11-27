"""Main entry point for label_meta module.

This module provides:
1. Training the meta-model ONCE
2. Interactive benchmarking of primary models
3. Results comparison and export

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

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.config_logging import get_logger
from src.constants import DEFAULT_RANDOM_STATE
from src.label_meta.benchmark import run_benchmark_for_primary
from src.label_meta.config import BenchmarkConfig, BenchmarkResult
from src.label_meta.meta_model import (
    load_meta_model,
    save_meta_model,
    train_meta_model_once,
)
from src.path import (
    DATASET_FEATURES_LABEL_PARQUET,
    LABEL_META_DIR,
    LABEL_PRIMAIRE_DIR,
    LABEL_META_MODEL_FILE,
    LABEL_META_BENCHMARKS_DIR,
    LABEL_PRIMAIRE_OPTIMIZATION_FILE,
    LABEL_PRIMAIRE_EVENTS_TRAIN_FILE,
    LABEL_PRIMAIRE_EVENTS_TEST_FILE,
)
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features and events for meta-model training.

    Returns:
        Tuple of (features DataFrame, events_train, events_test).
    """
    # Check prerequisites
    if not DATASET_FEATURES_LABEL_PARQUET.exists():
        raise FileNotFoundError(
            f"Labeled features not found: {DATASET_FEATURES_LABEL_PARQUET}. "
            "Please run 'python -m src.label_primaire.main' first."
        )

    if not LABEL_PRIMAIRE_EVENTS_TRAIN_FILE.exists() or not LABEL_PRIMAIRE_EVENTS_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Events not found in {LABEL_PRIMAIRE_DIR}. "
            "Please run 'python -m src.label_primaire.main' first."
        )

    # Load data
    df_features = pd.read_parquet(DATASET_FEATURES_LABEL_PARQUET)
    events_train = pd.read_parquet(LABEL_PRIMAIRE_EVENTS_TRAIN_FILE)
    events_test = pd.read_parquet(LABEL_PRIMAIRE_EVENTS_TEST_FILE)

    logger.info("Loaded %d features, %d train events, %d test events",
               len(df_features), len(events_train), len(events_test))

    return df_features, events_train, events_test


def prepare_splits(
    df_features: pd.DataFrame,
    events_train: pd.DataFrame,
    events_test: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Prepare train/test splits from features and events.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, events_train, events_test).
    """
    # Get feature columns
    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    # Get labels
    labels = df_features["label"].copy()

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


def load_meta_params() -> dict[str, Any]:
    """Load LightGBM params from label_primaire optimization."""
    if not LABEL_PRIMAIRE_OPTIMIZATION_FILE.exists():
        logger.warning("No optimization results found, using default params")
        return {}

    data = load_json_data(LABEL_PRIMAIRE_OPTIMIZATION_FILE)
    meta_params = data.get("best_model_params", {})

    logger.info("Loaded meta params from %s: %s", LABEL_PRIMAIRE_OPTIMIZATION_FILE, meta_params)
    return meta_params


def load_primary_params() -> dict[str, Any]:
    """Load primary model params from label_primaire optimization."""
    if not LABEL_PRIMAIRE_OPTIMIZATION_FILE.exists():
        logger.warning("No optimization results found, primary params empty")
        return {}

    data = load_json_data(LABEL_PRIMAIRE_OPTIMIZATION_FILE)
    primary_params = data.get("best_model_params", {})

    logger.info("Loaded primary params from %s: %s", LABEL_PRIMAIRE_OPTIMIZATION_FILE, primary_params)
    return primary_params


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
    print("   LABEL META - Meta-Model Training")
    print("=" * 60)
    print("\nThis module will:")
    print("  1. Train meta-model ONCE (or load existing)")
    print("\n")

    # Create output directory
    LABEL_META_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    try:
        df_features, events_train, events_test = load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run 'python -m src.label_primaire.main' first.")
        sys.exit(1)

    # Prepare splits
    X_train, y_train, X_test, y_test, events_train, events_test = prepare_splits(
        df_features, events_train, events_test
    )

    # Load params from label_primaire
    meta_params = load_meta_params()
    primary_params = load_primary_params()

    # Check if meta-model already exists
    meta_model = None

    if LABEL_META_MODEL_FILE.exists():
        print(f"\nMeta-model found: {LABEL_META_MODEL_FILE}")
        use_existing = ask_yes_no("Use existing meta-model?", default=True)

        if use_existing:
            meta_model = load_meta_model(LABEL_META_MODEL_FILE)

    # Train meta-model if needed
    if meta_model is None:
        print("\n" + "=" * 60)
        print("Training meta-model (this happens ONCE)...")
        print("=" * 60)

        reference_model = input("Reference primary model for meta training [lightgbm]: ").strip()
        if not reference_model:
            reference_model = "lightgbm"

        meta_model = train_meta_model_once(
            X_train=X_train,
            y_train=y_train,
            events_train=events_train,
            meta_params=meta_params,
            reference_primary_name=reference_model,
            random_state=DEFAULT_RANDOM_STATE,
            verbose=True,
        )

        # Save meta-model
        save_meta_model(meta_model, LABEL_META_MODEL_FILE)
        print(f"\nMeta-model saved to {LABEL_META_MODEL_FILE}")

    print("\nMeta-model ready. Interactive benchmarking is disabled.")
    print("You can reuse the saved meta-model directly with your primary models.")
    print("=" * 60)
    print(f"Output directory: {LABEL_META_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
