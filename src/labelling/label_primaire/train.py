"""
Training module for Primary Label models.

This module trains a primary model using optimized parameters from the
optimization step. It:
1. Loads optimized triple barrier and model parameters
2. Generates labels on the training set
3. Trains the model
4. Saves the trained model for evaluation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_RANDOM_STATE,
    TRAIN_SPLIT_LABEL,
)
from src.labelling.label_primaire.utils import (
    MODEL_REGISTRY,
    load_model_class,
    get_dataset_for_model,
    get_daily_volatility,
)
from src.labelling.label_primaire.opti import (
    get_events_primary,
)
from src.model.base import BaseModel
from src.path import (
    DOLLAR_BARS_PARQUET,
    LABEL_PRIMAIRE_OPTI_DIR,
    LABEL_PRIMAIRE_TRAIN_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training."""

    model_name: str
    random_state: int = DEFAULT_RANDOM_STATE
    vol_window: int = 21
    use_class_weight: bool = True  # Balance classes during training


@dataclass
class TrainingResult:
    """Result of the training process."""

    model_name: str
    model_params: Dict[str, Any]
    triple_barrier_params: Dict[str, Any]
    train_samples: int
    label_distribution: Dict[str, Any]
    model_path: str
    events_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "triple_barrier_params": self.triple_barrier_params,
            "train_samples": self.train_samples,
            "label_distribution": self.label_distribution,
            "model_path": self.model_path,
            "events_path": self.events_path,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Training results saved to {path}")


# =============================================================================
# PARAMETER LOADING
# =============================================================================


def load_optimized_params(
    model_name: str,
    opti_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Load optimized parameters from the optimization step.

    Parameters
    ----------
    model_name : str
        Name of the model.
    opti_dir : Path, optional
        Directory containing optimization results.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - model_params: Model hyperparameters
        - triple_barrier_params: Triple barrier parameters
        - best_score: Best optimization score
        - metric: Optimization metric used
    """
    if opti_dir is None:
        opti_dir = LABEL_PRIMAIRE_OPTI_DIR

    opti_file = opti_dir / f"{model_name}_optimization.json"

    if not opti_file.exists():
        raise FileNotFoundError(
            f"Optimization results not found: {opti_file}\n"
            f"Run optimization first: python -m src.labelling.label_primaire.opti"
        )

    with open(opti_file, "r", encoding="utf-8") as f:
        opti_results = json.load(f)

    return {
        "model_params": opti_results["best_params"],
        "triple_barrier_params": opti_results["best_triple_barrier_params"],
        "best_score": opti_results["best_score"],
        "metric": opti_results["metric"],
    }


def get_model_output_dir(model_name: str) -> Path:
    """Get the output directory for a specific model."""
    return LABEL_PRIMAIRE_TRAIN_DIR / model_name


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train_model(
    model_name: str,
    config: TrainingConfig | None = None,
    opti_dir: Path | None = None,
    output_dir: Path | None = None,
) -> TrainingResult:
    """
    Train a primary model using optimized parameters.

    Parameters
    ----------
    model_name : str
        Name of the model to train.
    config : TrainingConfig, optional
        Training configuration.
    opti_dir : Path, optional
        Directory containing optimization results.
    output_dir : Path, optional
        Directory to save trained model and results.

    Returns
    -------
    TrainingResult
        Training results including paths to saved artifacts.
    """
    if config is None:
        config = TrainingConfig(model_name=model_name)
    else:
        config.model_name = model_name

    if output_dir is None:
        output_dir = get_model_output_dir(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {model_name} with optimized parameters")

    # Load optimized parameters
    logger.info("Loading optimized parameters...")
    optimized = load_optimized_params(model_name, opti_dir)
    model_params = optimized["model_params"]
    tb_params = optimized["triple_barrier_params"]

    logger.info(f"Best optimization score ({optimized['metric']}): {optimized['best_score']:.4f}")
    logger.info(f"Triple Barrier params: {tb_params}")
    logger.info(f"Model params: {model_params}")

    # Load model class
    model_class = load_model_class(model_name)

    # Load datasets
    logger.info("Loading datasets...")
    features_df = get_dataset_for_model(model_name)

    if not DOLLAR_BARS_PARQUET.exists():
        raise FileNotFoundError(
            f"Dollar bars file not found: {DOLLAR_BARS_PARQUET}. "
            "Please run data preparation first: python -m src.data_preparation.main"
        )

    dollar_bars = pd.read_parquet(DOLLAR_BARS_PARQUET)

    # Filter to train split only
    if "split" in features_df.columns:
        train_mask = features_df["split"] == TRAIN_SPLIT_LABEL
        features_df = features_df[train_mask].copy()
        features_df = features_df.drop(columns=["split"])

    # Align on datetime index if available (consistent with optimization)
    if "datetime_close" in features_df.columns:
        features_df = features_df.set_index("datetime_close")
    features_df = features_df.sort_index()

    # Get close prices
    close_prices = dollar_bars.set_index("datetime_close")["close"].sort_index()
    # Ensure close_prices is a pandas Series with proper index
    if not isinstance(close_prices, pd.Series):
        raise TypeError(f"Expected close_prices to be pd.Series, got {type(close_prices)}")
    # Remove duplicates and ensure it's still a Series
    close_prices = close_prices[~close_prices.index.duplicated(keep="first")]
    # Ensure it's still a Series after duplicate removal
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices, name="close")

    # Ensure close_prices is a pandas Series (type checker might infer ndarray)
    assert isinstance(close_prices, pd.Series), f"close_prices must be pd.Series, got {type(close_prices)}"

    # Align indices
    # Ensure both are pandas objects with index
    if not isinstance(close_prices, pd.Series):
        raise TypeError(f"close_prices must be pd.Series, got {type(close_prices)}")
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError(f"features_df must be pd.DataFrame, got {type(features_df)}")

    common_idx = features_df.index.intersection(close_prices.index)
    features_df = features_df.loc[common_idx]
    close_prices = close_prices.loc[common_idx]
    # Ensure close_prices remains a pandas Series after alignment
    if not isinstance(close_prices, pd.Series):
        close_prices = pd.Series(close_prices, index=common_idx, name="close")

    # Remove non-feature columns
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close", "open", "high", "low", "close",
        "volume", "cum_dollar_value", "vwap", "n_ticks", "threshold_used",
        "duration_sec", "log_return", "split", "label",
    ]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    features_df = features_df[feature_cols]

    logger.info(f"Feature dataset shape: {features_df.shape}")

    # Compute volatility using EWM (De Prado methodology)
    volatility = get_daily_volatility(close_prices, span=config.vol_window)

    # Generate labels with optimized triple barrier parameters
    logger.info("Generating triple barrier labels...")

    events = get_events_primary(
        close=close_prices,
        t_events=pd.DatetimeIndex(features_df.index),
        pt_mult=tb_params["pt_mult"],
        sl_mult=tb_params["sl_mult"],
        trgt=volatility,
        max_holding=tb_params["max_holding"],
        min_return=tb_params.get("min_return", 0.0),
    )

    logger.info(f"Generated {len(events)} events")

    # Analyze label distribution
    label_counts = events["label"].value_counts().to_dict()
    total = len(events)
    label_percentages = {k: v / total * 100 for k, v in label_counts.items()}
    label_stats = {
        "total_events": total,
        "label_counts": label_counts,
        "label_percentages": label_percentages,
    }
    logger.info(f"Label distribution: {label_counts}")

    # Align features with labels
    common_idx = features_df.index.intersection(events.index)
    X_train = features_df.loc[common_idx]
    y_train = events.loc[common_idx, "label"]

    logger.info(f"Training samples: {len(X_train)}")

    # Add random state to model params
    model_params["random_state"] = config.random_state

    # Add class weight if supported and requested
    if config.use_class_weight and model_name in ["lightgbm", "xgboost", "random_forest"]:
        # Compute class weights for imbalanced data
        class_counts = y_train.value_counts()
        total = len(y_train)
        class_weight = {
            cls: total / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        model_params["class_weight"] = class_weight
        logger.info(f"Using class weights: {class_weight}")

    # Create and train model
    logger.info("Training model...")
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    logger.info("Model training complete")

    # Save model
    model_path = output_dir / f"{model_name}_model.joblib"
    model.save(model_path)

    # Save events (labels) for later use in evaluation
    events_path = output_dir / f"{model_name}_events_train.parquet"
    events.to_parquet(events_path)
    logger.info(f"Training events saved to {events_path}")

    # Build result
    result = TrainingResult(
        model_name=model_name,
        model_params=model_params,
        triple_barrier_params=tb_params,
        train_samples=len(X_train),
        label_distribution=label_stats,
        model_path=str(model_path),
        events_path=str(events_path),
    )

    # Save training results
    results_path = output_dir / f"{model_name}_training_results.json"
    result.save(results_path)

    return result


# =============================================================================
# INTERACTIVE CLI
# =============================================================================


def get_available_optimized_models() -> list[str]:
    """Get list of models that have been optimized."""
    available = []
    for model_name in MODEL_REGISTRY.keys():
        opti_file = LABEL_PRIMAIRE_OPTI_DIR / f"{model_name}_optimization.json"
        if opti_file.exists():
            available.append(model_name)
    return available


def select_model() -> str:
    """Interactive model selection."""
    models = list(MODEL_REGISTRY.keys())
    optimized = get_available_optimized_models()

    print("\n" + "=" * 60)
    print("LABEL PRIMAIRE - TRAINING")
    print("=" * 60)
    print("\nModels disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        status = "[optimise]" if model in optimized else "[non optimise]"
        info = MODEL_REGISTRY[model]
        dataset_type = info["dataset"]
        print(f"  {i}. {model:<15} ({dataset_type}) {status}")

    print("-" * 40)

    if not optimized:
        print("\nAucun modele optimise. Lancer d'abord l'optimisation.")
        print("python -m src.labelling.label_primaire.opti")
        exit(1)

    while True:
        try:
            choice = input("\nChoisir le modele (numero ou nom): ").strip()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                else:
                    print(f"Numero invalide. Choisir entre 1 et {len(models)}")
                    continue
            elif choice.lower() in models:
                selected = choice.lower()
            else:
                print(f"Modele inconnu: {choice}")
                continue

            # Check if optimized
            if selected not in optimized:
                print(f"Le modele '{selected}' n'a pas ete optimise.")
                print("Lancer d'abord: python -m src.labelling.label_primaire.opti")
                continue

            return selected

        except KeyboardInterrupt:
            print("\nEntrainement annule.")
            exit(0)


def get_yes_no_input(prompt: str, default: bool = True) -> bool:
    """Get yes/no input with default value."""
    default_str = "O/n" if default else "o/N"
    while True:
        try:
            value = input(f"{prompt} ({default_str}): ").strip().lower()
            if value == "":
                return default
            if value in ["o", "oui", "y", "yes"]:
                return True
            if value in ["n", "non", "no"]:
                return False
            print("Repondre par O (oui) ou N (non)")
        except KeyboardInterrupt:
            print("\nAnnule.")
            exit(0)


def main() -> None:
    """Main entry point with interactive prompts."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Interactive model selection
    model_name = select_model()
    print(f"\nModele selectionne: {model_name}")

    # Load and display optimized params
    try:
        optimized = load_optimized_params(model_name)
        print(f"\nParametres optimises charges:")
        print(f"  Score ({optimized['metric']}): {optimized['best_score']:.4f}")
        print(f"  Triple Barrier: {optimized['triple_barrier_params']}")
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        exit(1)

    # Ask about class weighting
    use_class_weight = get_yes_no_input("\nUtiliser le class weighting?", default=True)

    # Confirm
    print("\n" + "-" * 40)
    print(f"Modele: {model_name}")
    print(f"Class weighting: {'Oui' if use_class_weight else 'Non'}")
    print("-" * 40)

    confirm = input("\nLancer l'entrainement? (O/n): ").strip().lower()
    if confirm == "n":
        print("Entrainement annule.")
        return

    # Create config
    config = TrainingConfig(
        model_name=model_name,
        random_state=DEFAULT_RANDOM_STATE,
        use_class_weight=use_class_weight,
    )

    # Run training
    print("\n")
    result = train_model(
        model_name=model_name,
        config=config,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"ENTRAINEMENT TERMINE: {model_name.upper()}")
    print("=" * 60)
    print(f"Echantillons d'entrainement: {result.train_samples}")
    print(f"\nDistribution des labels:")
    for label, count in result.label_distribution["label_counts"].items():
        pct = result.label_distribution["label_percentages"][label]
        print(f"  Label {label}: {count} ({pct:.1f}%)")
    print(f"\nParametres Triple Barrier:")
    for k, v in result.triple_barrier_params.items():
        print(f"  {k}: {v}")
    print(f"\nModele sauvegarde: {result.model_path}")
    print(f"Events sauvegardes: {result.events_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
