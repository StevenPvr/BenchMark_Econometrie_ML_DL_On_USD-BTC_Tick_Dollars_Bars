"""Main entry point for label_meta_eval module.

This module evaluates the meta-model on the test split:
1. Loads the primary and meta models.
2. Generates predictions on the test set.
3. Filters primary trades based on meta-model confidence.
4. Computes performance metrics (Sharpe, Returns, etc.).
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
import matplotlib.pyplot as plt

from src.config_logging import get_logger
from src.label_meta.meta_labeling import (
    compute_strategy_returns,
    compute_strategy_metrics,
)
from src.path import (
    DATASET_FEATURES_LABEL_PARQUET,
    DATASET_FEATURES_LABEL_CSV,
    DATASET_FEATURES_LINEAR_LABEL_PARQUET,
    DATASET_FEATURES_LINEAR_LABEL_CSV,
    DATASET_FEATURES_LSTM_LABEL_PARQUET,
    DATASET_FEATURES_LSTM_LABEL_CSV,
    LABEL_META_TRAIN_MODELS_DIR,
    LABEL_META_EVAL_RESULTS_DIR,
    LABEL_PRIMAIRE_MODELS_DIR,
    LABEL_PRIMAIRE_EVENTS_TEST_FILE,
)
from src.utils import save_json_pretty

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
        answer = input("\nChoose a primary model to evaluate meta-model for (number or name): ").strip().lower()

        try:
            idx = int(answer)
            if 1 <= idx <= len(AVAILABLE_MODELS):
                return AVAILABLE_MODELS[idx - 1]
        except ValueError:
            pass

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


def load_models(model_name: str) -> tuple[Any, Any]:
    """Load primary and meta models."""
    primary_path = LABEL_PRIMAIRE_MODELS_DIR / f"{model_name}_model.joblib"
    meta_path = LABEL_META_TRAIN_MODELS_DIR / f"meta_model_{model_name}.joblib"

    if not primary_path.exists():
        raise FileNotFoundError(f"Primary model not found at {primary_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta model not found at {meta_path}. Run label_meta_train first.")

    logger.info("Loading primary model: %s", primary_path)
    primary_model = joblib.load(primary_path)

    logger.info("Loading meta model: %s", meta_path)
    meta_model = joblib.load(meta_path)

    return primary_model, meta_model


def load_test_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and test events."""
    dataset_parquet, _ = get_labeled_dataset_paths(model_name)

    if not dataset_parquet.exists():
        raise FileNotFoundError(f"Features file {dataset_parquet} not found.")

    if not LABEL_PRIMAIRE_EVENTS_TEST_FILE.exists():
        raise FileNotFoundError(f"Test events file {LABEL_PRIMAIRE_EVENTS_TEST_FILE} not found.")

    logger.info("Loading features from %s", dataset_parquet)
    df_features = pd.read_parquet(dataset_parquet)

    logger.info("Loading test events from %s", LABEL_PRIMAIRE_EVENTS_TEST_FILE)
    events_test = pd.read_parquet(LABEL_PRIMAIRE_EVENTS_TEST_FILE)

    return df_features, events_test


def main() -> None:
    print("\n" + "=" * 60)
    print("   LABEL META EVAL - Evaluate Meta-Model")
    print("=" * 60)

    LABEL_META_EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select Model
    model_name = ask_model_choice()

    # 2. Load Data
    try:
        df_features, events_test = load_test_data(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Load Models
    try:
        primary_model, meta_model = load_models(model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 4. Prepare Test Features
    test_indices = events_test["t_start"].values.astype(int)
    exclude_cols = ["bar_id", "datetime_close", "log_return", "label", "split"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_test = df_features.iloc[test_indices][feature_cols].to_numpy()

    # 5. Primary Predictions
    print("Generating primary predictions...")
    primary_preds = primary_model.predict(X_test)
    primary_signal = pd.Series(primary_preds, index=events_test.index)

    # 6. Meta Predictions
    # Construct Meta Features: X + primary_signal
    X_meta_test = np.column_stack([X_test, primary_preds])

    print("Generating meta predictions (probabilities)...")
    # Meta model is usually a Classifier. We want probabilities of class 1 (Correct Trade).
    if hasattr(meta_model, "predict_proba"):
        meta_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    else:
        # Fallback if model doesn't support proba (unlikely for LightGBM)
        meta_probs = meta_model.predict(X_meta_test)

    meta_probs_series = pd.Series(meta_probs, index=events_test.index)

    # 7. Apply Threshold & Compute Metrics
    threshold = 0.5  # Default, could be optimized or asked

    # Filter: Trade if primary != 0 AND meta_prob > threshold
    # Note: Primary model already filters 0s usually, but let's be safe.
    # Meta model predicts probability that the trade is "Correct".

    # Meta signal: 1 if we take the trade, 0 otherwise
    meta_decision = (meta_probs_series > threshold).astype(int)

    # Calculate Returns
    print("Computing strategy returns...")
    strategy_results = compute_strategy_returns(
        events=events_test,
        primary_signal=primary_signal,
        meta_signal=meta_decision
    )

    metrics = compute_strategy_metrics(strategy_results)

    print("\n" + "="*40)
    print("PERFORMANCE METRICS (TEST SET)")
    print("="*40)
    print(f"Primary Total Return: {metrics.get('primary_total_return', 0):.4f}")
    print(f"Primary Sharpe Ratio: {metrics.get('primary_sharpe', 0):.4f}")
    print(f"Primary Win Rate:     {metrics.get('primary_win_rate', 0):.4f}")
    print(f"Primary Trades:       {metrics.get('primary_n_trades', 0)}")
    print("-" * 40)
    print(f"Meta Total Return:    {metrics.get('meta_total_return', 0):.4f}")
    print(f"Meta Sharpe Ratio:    {metrics.get('meta_sharpe', 0):.4f}")
    print(f"Meta Win Rate:        {metrics.get('meta_win_rate', 0):.4f}")
    print(f"Meta Trades:          {metrics.get('meta_n_trades', 0)}")
    print("-" * 40)

    improvement = metrics.get('meta_sharpe', 0) - metrics.get('primary_sharpe', 0)
    print(f"Sharpe Improvement:   {improvement:+.4f}")

    # Save results
    results_file = LABEL_META_EVAL_RESULTS_DIR / f"eval_{model_name}.json"
    save_json_pretty(metrics, results_file)
    print(f"\nResults saved to {results_file}")

    # Plot equity curve
    if "primary_cum_return" in strategy_results.columns and "meta_cum_return" in strategy_results.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(strategy_results.index, strategy_results["primary_cum_return"], label="Primary Model")
        plt.plot(strategy_results.index, strategy_results["meta_cum_return"], label="Meta Model Filtered")
        plt.title(f"Equity Curve - {model_name.upper()}")
        plt.xlabel("Trade Index") # Note: events are indexed by integer t_start usually, or we can map to time
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)

        plot_path = LABEL_META_EVAL_RESULTS_DIR / f"equity_{model_name}.png"
        plt.savefig(plot_path)
        print(f"Equity plot saved to {plot_path}")

if __name__ == "__main__":
    main()
