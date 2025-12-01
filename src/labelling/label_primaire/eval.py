"""
Primary Model Evaluation Module.

Evaluates trained primary models with comprehensive metrics.
Loads the model, makes predictions on train/test data, and displays metrics.

Metrics computed:
- Accuracy, Balanced Accuracy
- F1-Score (macro, weighted, per-class)
- Precision, Recall
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- AUC-ROC, Log Loss
- Confusion Matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, cast

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import shap  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL
from src.labelling.label_primaire.utils import MODEL_REGISTRY
from src.model.base import BaseModel
from src.path import (
    LABEL_PRIMAIRE_TRAIN_DIR,
    LABEL_PRIMAIRE_EVAL_DIR,
    DATASET_FEATURES_FINAL_PARQUET,
    DATASET_FEATURES_LINEAR_FINAL_PARQUET,
    DATASET_FEATURES_LSTM_FINAL_PARQUET,
)

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================


@dataclass
class ClassificationMetrics:
    """Classification metrics for model evaluation."""

    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    f1_per_class: Dict[str, float]
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    mcc: float
    cohen_kappa: float
    auc_roc_macro: float | None
    log_loss_value: float | None
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "f1_per_class": self.f1_per_class,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "mcc": self.mcc,
            "cohen_kappa": self.cohen_kappa,
            "auc_roc_macro": self.auc_roc_macro,
            "log_loss": self.log_loss_value,
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
            "n_samples": self.n_samples,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_labels = [str(c) for c in sorted(classes)]

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division="warn"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division="warn"))
    f1_per_class_arr = cast(np.ndarray, f1_score(y_true, y_pred, average=None, zero_division="warn", labels=sorted(classes)))  # type: ignore[call-overload]
    f1_per_class = {str(c): float(f) for c, f in zip(sorted(classes), f1_per_class_arr)}

    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division="warn"))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division="warn"))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division="warn"))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division="warn"))

    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    auc_roc_macro = None
    log_loss_val = None

    if y_proba is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                auc_roc_macro = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                auc_roc_macro = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                ))
            log_loss_val = float(log_loss(y_true, y_proba, labels=sorted(classes)))
        except (ValueError, IndexError):
            pass

    cm = confusion_matrix(y_true, y_pred, labels=sorted(classes))

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        f1_per_class=f1_per_class,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        mcc=mcc,
        cohen_kappa=kappa,
        auc_roc_macro=auc_roc_macro,
        log_loss_value=log_loss_val,
        confusion_matrix=cm.tolist(),
        class_labels=class_labels,
        n_samples=len(y_true),
    )


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

# Models compatible with SHAP TreeExplainer
SHAP_TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}

# Models compatible with SHAP LinearExplainer
SHAP_LINEAR_MODELS = {"ridge", "logistic"}

# Models NOT compatible with standard SHAP (would need KernelExplainer, too slow)
SHAP_INCOMPATIBLE_MODELS = {"lstm"}


def is_shap_compatible(model_name: str) -> bool:
    """Check if a model is compatible with SHAP analysis."""
    return model_name in SHAP_TREE_MODELS or model_name in SHAP_LINEAR_MODELS


def get_shap_explainer_type(model_name: str) -> str | None:
    """Get the type of SHAP explainer to use for a model."""
    if model_name in SHAP_TREE_MODELS:
        return "tree"
    elif model_name in SHAP_LINEAR_MODELS:
        return "linear"
    return None


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
    max_samples: int | None = None,
) -> tuple[shap.Explainer | None, np.ndarray | None, pd.DataFrame | None]:
    """
    Compute SHAP values for a model.

    Parameters
    ----------
    model : Any
        The trained model wrapper (must have .model attribute).
    X : pd.DataFrame
        Feature data.
    model_name : str
        Name of the model (for explainer selection).
    max_samples : int | None, default=None
        Maximum samples for SHAP computation (for performance). None means no limit.

    Returns
    -------
    tuple[shap.Explainer | None, np.ndarray | None, pd.DataFrame | None]
        (explainer, shap_values, X_sampled) or (None, None, None) if incompatible.
    """
    if not is_shap_compatible(model_name):
        logger.warning(f"Model {model_name} is not compatible with SHAP analysis")
        return None, None, None

    # Sample data if too large (None means no limit)
    if max_samples is not None and len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer_type = get_shap_explainer_type(model_name)
    underlying_model = getattr(model, "model", None)

    if underlying_model is None:
        logger.warning(f"Could not access underlying model for {model_name}")
        return None, None, None

    try:
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(underlying_model)
            shap_values = explainer.shap_values(X_sample)
        elif explainer_type == "linear":
            # For linear models, we need background data
            # Handle normalized models (ridge has scaler)
            X_transformed = X_sample.values
            if hasattr(model, "scaler") and model.scaler is not None:
                X_transformed = model.scaler.transform(X_transformed)

            explainer = shap.LinearExplainer(
                underlying_model,
                X_transformed,
                feature_perturbation="interventional",
            )
            shap_values = explainer.shap_values(X_transformed)
        else:
            return None, None, None

        return explainer, shap_values, X_sample

    except Exception as e:
        logger.warning(f"Failed to compute SHAP values for {model_name}: {e}")
        return None, None, None


def save_shap_plots(
    shap_values: np.ndarray | List[np.ndarray],
    X: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    class_names: List[str] | None = None,
    max_display: int = 20,
) -> List[Path]:
    """
    Save SHAP summary and bar plots.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from explainer.
    X : pd.DataFrame
        Feature data (must match shap_values samples).
    model_name : str
        Name of the model.
    output_dir : Path
        Directory to save plots.
    class_names : List[str] | None
        Names of classes for multi-class models.
    max_display : int, default=20
        Maximum features to display.

    Returns
    -------
    List[Path]
        List of saved plot paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # Normalize SHAP values format
    # TreeExplainer for multi-class can return:
    # - List of arrays: [array(n_samples, n_features), ...] one per class
    # - 3D array: (n_samples, n_features, n_classes) - modern SHAP format
    # - 3D array: (n_classes, n_samples, n_features) - older format

    n_features = len(X.columns)
    n_samples = len(X)

    # Convert to list format for consistent handling
    if isinstance(shap_values, list):
        # Already a list, verify shape
        shap_list = shap_values
        if len(shap_list) > 0 and hasattr(shap_list[0], 'shape'):
            logger.info(f"SHAP values format: list of {len(shap_list)} arrays, shape {shap_list[0].shape}")
    elif isinstance(shap_values, np.ndarray):
        logger.info(f"SHAP values array shape: {shap_values.shape}")
        if shap_values.ndim == 2:
            # Binary or single output: (n_samples, n_features)
            shap_list = [shap_values]
        elif shap_values.ndim == 3:
            # Multi-class 3D array - determine orientation
            # Modern SHAP: (n_samples, n_features, n_classes)
            # Older SHAP: (n_classes, n_samples, n_features)
            if shap_values.shape[0] == n_samples and shap_values.shape[1] == n_features:
                # Shape is (n_samples, n_features, n_classes)
                n_classes = shap_values.shape[2]
                shap_list = [shap_values[:, :, i] for i in range(n_classes)]
                logger.info(f"Detected modern SHAP format: (n_samples, n_features, n_classes)")
            elif shap_values.shape[1] == n_samples and shap_values.shape[2] == n_features:
                # Shape is (n_classes, n_samples, n_features)
                n_classes = shap_values.shape[0]
                shap_list = [shap_values[i] for i in range(n_classes)]
                logger.info(f"Detected older SHAP format: (n_classes, n_samples, n_features)")
            else:
                logger.warning(f"Unexpected SHAP shape: {shap_values.shape}, expected samples={n_samples}, features={n_features}")
                return saved_paths
        else:
            logger.warning(f"Unexpected SHAP dimensions: {shap_values.ndim}")
            return saved_paths
    else:
        logger.warning(f"Unexpected SHAP values type: {type(shap_values)}")
        return saved_paths

    is_multiclass = len(shap_list) > 1

    if is_multiclass:
        n_classes = len(shap_list)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        # Summary plot for each class
        for i, (sv, cn) in enumerate(zip(shap_list, class_names)):
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    sv,
                    X,
                    max_display=max_display,
                    show=False,
                    plot_type="dot",
                )
                plt.title(f"SHAP Summary - {model_name} - {cn}")
                plt.tight_layout()
                path = output_dir / f"{model_name}_shap_summary_class_{i}.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                saved_paths.append(path)
                logger.info(f"Saved SHAP summary plot: {path}")
            except Exception as e:
                logger.warning(f"Failed to save SHAP summary plot for class {i}: {e}")

        # Global bar plot (mean absolute SHAP across all classes)
        try:
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
            plt.figure(figsize=(12, 8))
            feature_importance = pd.Series(mean_abs_shap, index=X.columns)
            feature_importance = feature_importance.nlargest(max_display)
            feature_importance.plot(kind="barh")
            plt.xlabel("Mean |SHAP value| (average across classes)")
            plt.ylabel("Feature")
            plt.title(f"SHAP Feature Importance - {model_name}")
            plt.tight_layout()
            path = output_dir / f"{model_name}_shap_bar_global.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            saved_paths.append(path)
            logger.info(f"Saved SHAP bar plot: {path}")
        except Exception as e:
            logger.warning(f"Failed to save global SHAP bar plot: {e}")

    else:
        # Binary or regression: single array
        sv = shap_list[0]
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                sv,
                X,
                max_display=max_display,
                show=False,
                plot_type="dot",
            )
            plt.title(f"SHAP Summary - {model_name}")
            plt.tight_layout()
            path = output_dir / f"{model_name}_shap_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            saved_paths.append(path)
            logger.info(f"Saved SHAP summary plot: {path}")
        except Exception as e:
            logger.warning(f"Failed to save SHAP summary plot: {e}")

        # Bar plot
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                sv,
                X,
                max_display=max_display,
                show=False,
                plot_type="bar",
            )
            plt.title(f"SHAP Feature Importance - {model_name}")
            plt.tight_layout()
            path = output_dir / f"{model_name}_shap_bar.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            saved_paths.append(path)
            logger.info(f"Saved SHAP bar plot: {path}")
        except Exception as e:
            logger.warning(f"Failed to save SHAP bar plot: {e}")

    return saved_paths


# =============================================================================
# DATA LOADING
# =============================================================================


def get_features_path(model_name: str) -> Path:
    """Get the base features file path for a model type (without labels)."""
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    return path_map[dataset_type]


def get_labeled_features_path(model_name: str) -> Path:
    """
    Get the labeled features file path for a specific model.

    Each model has its own labeled features file to avoid conflicts
    when training multiple models with different triple barrier parameters.

    Example: lightgbm -> data/features/dataset_features_final_lightgbm.parquet
    """
    dataset_type = MODEL_REGISTRY[model_name]["dataset"]
    base_path_map = {
        "tree": DATASET_FEATURES_FINAL_PARQUET,
        "linear": DATASET_FEATURES_LINEAR_FINAL_PARQUET,
        "lstm": DATASET_FEATURES_LSTM_FINAL_PARQUET,
    }
    base_path = base_path_map[dataset_type]
    # Insert model name before .parquet extension
    return base_path.parent / f"{base_path.stem}_{model_name}.parquet"


def get_trained_models() -> List[str]:
    """Get list of models that have been trained."""
    trained = []
    for model_name in MODEL_REGISTRY.keys():
        model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
        model_file = model_dir / f"{model_name}_model.joblib"
        if model_file.exists():
            trained.append(model_name)
    return trained


def load_model(model_name: str) -> BaseModel:
    """Load a trained model."""
    model_dir = LABEL_PRIMAIRE_TRAIN_DIR / model_name
    model_path = model_dir / f"{model_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run training first: python -m src.labelling.label_primaire.train"
        )

    return BaseModel.load(model_path)


def load_data(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data with labels.

    Loads from the model-specific labeled features file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with features and labels
    """
    features_path = get_labeled_features_path(model_name)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Labeled features file not found: {features_path}\n"
            f"Run training first: python -m src.labelling.label_primaire.train"
        )

    df = pd.read_parquet(features_path)

    if "label" not in df.columns:
        raise ValueError(
            f"No 'label' column in {features_path}.\n"
            f"Run training first to generate labels: python -m src.labelling.label_primaire.train"
        )

    if "split" not in df.columns:
        raise ValueError(f"No 'split' column in {features_path}")

    train_df = cast(pd.DataFrame, df[df["split"] == TRAIN_SPLIT_LABEL].copy())
    test_df = cast(pd.DataFrame, df[df["split"] == TEST_SPLIT_LABEL].copy())

    return train_df, test_df


# =============================================================================
# EVALUATION
# =============================================================================


@dataclass
class EvaluationResult:
    """Evaluation result."""

    model_name: str
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    train_samples: int
    test_samples: int
    label_distribution_train: Dict[str, Any]
    label_distribution_test: Dict[str, Any]
    shap_plots: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "label_distribution_train": self.label_distribution_train,
            "label_distribution_test": self.label_distribution_test,
            "shap_plots": self.shap_plots,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {path}")


def evaluate_model(model_name: str) -> EvaluationResult:
    """
    Evaluate a trained primary model.

    Loads the model, makes predictions on train/test data,
    and computes comprehensive metrics.
    """
    logger.info(f"{'='*60}")
    logger.info(f"EVALUATION: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Load model
    logger.info("Loading model...")
    model = load_model(model_name)

    # Load data
    logger.info("Loading data...")
    train_df, test_df = load_data(model_name)

    # Prepare features
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close", "threshold_used",
        "log_return", "split", "label",
    ]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

    # Filter valid labels
    train_valid = train_df[~train_df["label"].isna()].copy()
    test_valid = test_df[~test_df["label"].isna()].copy()

    X_train = cast(pd.DataFrame, train_valid[feature_cols])
    y_train = cast(np.ndarray, cast(pd.Series, train_valid["label"]).astype(int).values)
    X_test = cast(pd.DataFrame, test_valid[feature_cols])
    y_test = cast(np.ndarray, cast(pd.Series, test_valid["label"]).astype(int).values)

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Make predictions
    logger.info("Making predictions...")
    y_train_pred = cast(np.ndarray, model.predict(X_train))
    y_test_pred = cast(np.ndarray, model.predict(X_test))

    # Get probabilities if available
    y_train_proba = None
    y_test_proba = None
    try:
        y_train_proba = cast(Any, model).predict_proba(X_train)
        y_test_proba = cast(Any, model).predict_proba(X_test)
    except Exception:
        pass

    # Compute metrics
    logger.info("Computing metrics...")
    train_metrics = compute_metrics(cast(np.ndarray, y_train), y_train_pred, y_train_proba)
    test_metrics = compute_metrics(cast(np.ndarray, y_test), y_test_pred, y_test_proba)

    # Label distributions
    train_counts = pd.Series(y_train).value_counts().to_dict()
    test_counts = pd.Series(y_test).value_counts().to_dict()

    train_dist = {
        "total": len(y_train),
        "counts": {str(k): int(v) for k, v in train_counts.items()},
        "percentages": {str(k): v / len(y_train) * 100 for k, v in train_counts.items()},
    }
    test_dist = {
        "total": len(y_test),
        "counts": {str(k): int(v) for k, v in test_counts.items()},
        "percentages": {str(k): v / len(y_test) * 100 for k, v in test_counts.items()},
    }

    return EvaluationResult(
        model_name=model_name,
        train_metrics=train_metrics.to_dict(),
        test_metrics=test_metrics.to_dict(),
        train_samples=len(y_train),
        test_samples=len(y_test),
        label_distribution_train=train_dist,
        label_distribution_test=test_dist,
    )


def run_shap_analysis(
    model_name: str,
    output_dir: Path,
    max_samples: int | None = None,
) -> List[str]:
    """
    Run SHAP analysis for a trained model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    output_dir : Path
        Directory to save SHAP plots.
    max_samples : int | None, default=None
        Maximum samples for SHAP computation (2.5% of 400k). None means no limit.

    Returns
    -------
    List[str]
        List of saved plot paths as strings.
    """
    if not is_shap_compatible(model_name):
        logger.info(f"Skipping SHAP analysis for {model_name} (not compatible)")
        return []

    logger.info(f"{'='*60}")
    logger.info(f"SHAP ANALYSIS: {model_name.upper()}")
    logger.info(f"{'='*60}")

    # Load model
    logger.info("Loading model for SHAP analysis...")
    model = load_model(model_name)

    # Load data
    logger.info("Loading test data for SHAP analysis...")
    _, test_df = load_data(model_name)

    # Prepare features
    non_feature_cols = [
        "bar_id", "timestamp_open", "timestamp_close",
        "datetime_open", "datetime_close", "threshold_used",
        "log_return", "split", "label",
    ]
    feature_cols = [c for c in test_df.columns if c not in non_feature_cols]
    test_valid = test_df[~test_df["label"].isna()].copy()
    X_test = cast(pd.DataFrame, test_valid[feature_cols])

    # Compute SHAP values (returns X_sample that was actually used)
    logger.info(f"Computing SHAP values (max {max_samples} samples)...")
    _, shap_values, X_sampled = compute_shap_values(
        model, X_test, model_name, max_samples=max_samples
    )

    if shap_values is None or X_sampled is None:
        logger.warning("Could not compute SHAP values")
        return []

    # Get class labels
    label_values = cast(pd.Series, test_valid["label"]).dropna().astype(int)
    classes = np.unique(label_values)
    class_names = [f"Class {int(c)}" for c in sorted(classes)]

    # Save SHAP plots (use X_sampled that matches shap_values exactly)
    logger.info("Saving SHAP plots...")
    shap_dir = output_dir / "shap"
    saved_paths = save_shap_plots(
        shap_values,
        X_sampled,
        model_name,
        shap_dir,
        class_names=class_names,
    )

    return [str(p) for p in saved_paths]


# =============================================================================
# DISPLAY
# =============================================================================


def print_metrics(metrics: Dict[str, Any], title: str) -> None:
    """Print metrics table."""
    print(f"\n{title}")
    print("=" * 50)

    key_metrics = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("F1 Macro", "f1_macro"),
        ("F1 Weighted", "f1_weighted"),
        ("Precision Macro", "precision_macro"),
        ("Recall Macro", "recall_macro"),
        ("MCC", "mcc"),
        ("Cohen's Kappa", "cohen_kappa"),
        ("AUC-ROC Macro", "auc_roc_macro"),
        ("Log Loss", "log_loss"),
    ]

    for name, key in key_metrics:
        value = metrics.get(key)
        if value is not None:
            print(f"  {name:<22}: {value:>8.4f}")
        else:
            print(f"  {name:<22}: {'N/A':>8}")

    if "f1_per_class" in metrics:
        print("\n  Per-Class F1:")
        for cls, f1 in metrics["f1_per_class"].items():
            print(f"    Class {cls}: {f1:.4f}")


def print_confusion_matrix(cm: List[List[int]], labels: List[str]) -> None:
    """Print confusion matrix."""
    print("\nConfusion Matrix:")
    print("-" * 40)

    header = "True\\Pred"
    for label in labels:
        header += f"  {label:>6}"
    print(header)

    for i, label in enumerate(labels):
        row = f"  {label:>6}"
        for j in range(len(labels)):
            row += f"  {cm[i][j]:>6}"
        print(row)


def print_comparison(result: EvaluationResult) -> None:
    """Print train vs test comparison."""
    print("\n" + "=" * 60)
    print("TRAIN vs TEST COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Train':>12} {'Test':>12} {'Delta':>10}")
    print("-" * 60)

    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc"]:
        train_val = result.train_metrics.get(metric, 0)
        test_val = result.test_metrics.get(metric, 0)
        delta = test_val - train_val
        print(f"{metric:<25} {train_val:>12.4f} {test_val:>12.4f} {delta:>+10.4f}")


def print_results(result: EvaluationResult) -> None:
    """Print complete evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {result.model_name.upper()}")
    print("=" * 70)

    print(f"\nSamples: Train={result.train_samples}, Test={result.test_samples}")

    print("\n--- Label Distribution ---")
    print("Train:")
    for label, count in result.label_distribution_train["counts"].items():
        pct = result.label_distribution_train["percentages"][label]
        print(f"  Class {label}: {count:>6} ({pct:>5.1f}%)")
    print("Test:")
    for label, count in result.label_distribution_test["counts"].items():
        pct = result.label_distribution_test["percentages"][label]
        print(f"  Class {label}: {count:>6} ({pct:>5.1f}%)")

    print_metrics(result.train_metrics, "TRAIN METRICS")
    print_confusion_matrix(
        result.train_metrics["confusion_matrix"],
        result.train_metrics["class_labels"]
    )

    print_metrics(result.test_metrics, "TEST METRICS")
    print_confusion_matrix(
        result.test_metrics["confusion_matrix"],
        result.test_metrics["class_labels"]
    )

    print_comparison(result)
    print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================


def select_model() -> str:
    """Interactive model selection."""
    models = list(MODEL_REGISTRY.keys())
    trained = get_trained_models()

    print("\n" + "=" * 60)
    print("LABEL PRIMAIRE - EVALUATION")
    print("=" * 60)
    print("\nEvalue le modele primaire seul (avant meta-model).")
    print("\nModeles disponibles:")
    print("-" * 40)

    for i, model in enumerate(models, 1):
        status = "[entraine]" if model in trained else "[non entraine]"
        info = MODEL_REGISTRY[model]
        dataset_type = info["dataset"]
        print(f"  {i}. {model:<15} ({dataset_type}) {status}")

    print("-" * 40)

    if not trained:
        print("\nAucun modele entraine. Lancer d'abord:")
        print("  python -m src.labelling.label_primaire.train")
        sys.exit(1)

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

            if selected not in trained:
                print(f"Le modele '{selected}' n'a pas ete entraine.")
                print("Lancer d'abord: python -m src.labelling.label_primaire.train")
                continue

            return selected

        except KeyboardInterrupt:
            print("\nAnnule.")
            sys.exit(0)


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model_name = select_model()
    print(f"\nModele selectionne: {model_name}")

    confirm = input("\nLancer l'evaluation? (O/n): ").strip().lower()
    if confirm == "n":
        print("Annule.")
        return

    print("\n")

    result = evaluate_model(model_name)
    print_results(result)

    # Save results
    output_dir = LABEL_PRIMAIRE_EVAL_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{model_name}_evaluation.json"
    result.save(results_path)

    print(f"\nResultats sauvegardes: {results_path}")

    # Run SHAP analysis if compatible
    if is_shap_compatible(model_name):
        run_shap = input("\nLancer l'analyse SHAP? (O/n): ").strip().lower()
        if run_shap != "n":
            print("\n")
            shap_plots = run_shap_analysis(model_name, output_dir)
            result.shap_plots = shap_plots
            # Update saved results with SHAP info
            result.save(results_path)

            if shap_plots:
                print(f"\n--- SHAP Plots ---")
                for plot_path in shap_plots:
                    print(f"  {plot_path}")
            else:
                print("\nAucun plot SHAP genere.")
    else:
        print(f"\n(SHAP non disponible pour {model_name})")


if __name__ == "__main__":
    main()
