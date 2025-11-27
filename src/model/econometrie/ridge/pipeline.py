"""Complete pipeline for Ridge Classifier: optimization, training, and evaluation.

This module provides a unified pipeline for:
1. Hyperparameter optimization with Optuna + walk-forward CV
2. Model training with the best parameters
3. Model evaluation with comprehensive metrics

Supports De Prado's triple-barrier labeling (-1, 0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.evaluation import (
    EvaluationResult,
    evaluate_model,
)
from src.model.econometrie.ridge_classifier.ridge_classifier import RidgeClassifierModel
from src.optimisation import (
    RidgeClassifierHyperparams,
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    create_cv_config,
)
from src.optimisation.walk_forward_cv import DEFAULT_METRIC
from src.training import (
    CrossValidationTrainingResult,
    TrainingResult,
    train_model,
    train_with_cv,
)
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================


@dataclass
class RidgePipelineConfig:
    """Configuration for the Ridge Classifier pipeline.

    Attributes:
        n_trials: Number of optimization trials.
        n_splits: Number of CV splits for optimization.
        purge_gap: Samples to purge between train and test.
        min_train_size: Minimum training set size.
        validation_split: Validation split for final training.
        metric: Metric to optimize/evaluate (default: f1_macro).
        output_dir: Directory for saving outputs.
        random_state: Random seed.
        verbose: Verbosity level.
    """

    n_trials: int = 50
    n_splits: int = 5
    purge_gap: int = 5
    min_train_size: int = 100
    validation_split: float = 0.2
    metric: str = DEFAULT_METRIC
    output_dir: Path | None = None
    random_state: int = 42
    verbose: bool = True


# ============================================================================
# PIPELINE RESULTS
# ============================================================================


@dataclass
class RidgePipelineResult:
    """Complete result from the Ridge Classifier pipeline.

    Attributes:
        optimization_result: Hyperparameter optimization results.
        training_result: Model training results.
        evaluation_result: Model evaluation results.
        best_params: Best hyperparameters found.
        model: Trained model with best parameters.
    """

    optimization_result: OptimizationResult | None
    training_result: TrainingResult | CrossValidationTrainingResult
    evaluation_result: EvaluationResult
    best_params: dict[str, Any]
    model: RidgeClassifierModel

    def summary(self) -> str:
        """Generate a summary of the pipeline results."""
        lines = [
            "=" * 60,
            "Ridge Classifier Pipeline Results",
            "=" * 60,
            "",
        ]

        if self.optimization_result is not None:
            lines.extend([
                "OPTIMIZATION:",
                f"  Trials: {self.optimization_result.n_completed}/{self.optimization_result.n_trials}",
                f"  Best CV score: {self.optimization_result.best_score:.4f}",
                "",
            ])

        lines.extend([
            "BEST PARAMETERS:",
        ])
        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "TRAINING:",
        ])
        if isinstance(self.training_result, CrossValidationTrainingResult):
            lines.extend([
                f"  CV score: {self.training_result.cv_result.mean_score:.4f} "
                f"(+/- {self.training_result.cv_result.std_score:.4f})",
                f"  Final train score: {self.training_result.final_train_score:.4f}",
            ])
        else:
            lines.extend([
                f"  Train score: {self.training_result.train_score:.4f}",
            ])
            if self.training_result.val_score is not None:
                lines.append(f"  Val score: {self.training_result.val_score:.4f}")

        # Model coefficients summary (multi-class: shape is (n_classes, n_features))
        if self.model.is_fitted:
            coefs = self.model.coef_
            if coefs.ndim == 1:
                n_features = len(coefs)
                non_zero = np.sum(coefs != 0)
            else:
                n_features = coefs.shape[1]
                non_zero = np.sum(np.any(coefs != 0, axis=0))
            lines.extend([
                "",
                "MODEL COEFFICIENTS:",
                f"  Non-zero features: {non_zero}/{n_features}",
            ])

        lines.extend([
            "",
            "EVALUATION (Test Set):",
        ])
        for metric, value in self.evaluation_result.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# RIDGE PIPELINE
# ============================================================================


class RidgePipeline:
    """Complete pipeline for Ridge Classifier: optimize, train, evaluate.

    Example:
        >>> pipeline = RidgePipeline(config=RidgePipelineConfig(
        ...     n_trials=50,
        ...     n_splits=5,
        ...     purge_gap=10,
        ... ))
        >>> result = pipeline.run(X_train, y_train, X_test, y_test)
        >>> print(result.summary())
    """

    def __init__(self, config: RidgePipelineConfig | None = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or RidgePipelineConfig()

    def optimize(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        hyperparam_space: RidgeClassifierHyperparams | None = None,
    ) -> OptimizationResult:
        """Optimize hyperparameters using Optuna with walk-forward CV.

        Args:
            X: Training features.
            y: Training target (class labels: -1, 0, 1).
            hyperparam_space: Custom hyperparameter space.

        Returns:
            OptimizationResult with best parameters.
        """
        if self.config.verbose:
            logger.info("Starting Ridge Classifier hyperparameter optimization...")

        # Create configurations
        cv_config = create_cv_config(
            n_splits=self.config.n_splits,
            purge_gap=self.config.purge_gap,
            min_train_size=self.config.min_train_size,
        )

        opt_config = OptimizationConfig(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state,
        )

        space = hyperparam_space or RidgeClassifierHyperparams()

        # Run optimization
        optimizer = OptunaOptimizer(
            model_class=RidgeClassifierModel,
            hyperparam_space=space,
            cv_config=cv_config,
            optimization_config=opt_config,
        )

        result = optimizer.optimize(
            X, y,
            metric=self.config.metric,
            verbose=self.config.verbose,
        )

        if self.config.verbose:
            logger.info("Optimization complete. Best %s: %.4f", self.config.metric, result.best_score)

        return result

    def train(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        params: dict[str, Any],
        use_cv: bool = True,
    ) -> tuple[RidgeClassifierModel, TrainingResult | CrossValidationTrainingResult]:
        """Train the model with given parameters.

        Args:
            X: Training features.
            y: Training target (class labels: -1, 0, 1).
            params: Model hyperparameters.
            use_cv: If True, use walk-forward CV for training.

        Returns:
            Tuple of (trained model, training result).
        """
        if self.config.verbose:
            logger.info("Training Ridge Classifier with optimized parameters...")

        # Create model with best params
        model = RidgeClassifierModel(**params)

        result: TrainingResult | CrossValidationTrainingResult
        if use_cv:
            result = train_with_cv(
                model, X, y,
                n_splits=self.config.n_splits,
                purge_gap=self.config.purge_gap,
                min_train_size=self.config.min_train_size,
                metric=self.config.metric,
                verbose=self.config.verbose,
            )
        else:
            result = train_model(
                model, X, y,
                validation_split=self.config.validation_split,
                metric=self.config.metric,
                verbose=self.config.verbose,
            )

        return model, result

    def evaluate(
        self,
        model: RidgeClassifierModel,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
    ) -> EvaluationResult:
        """Evaluate the model on test data.

        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target (class labels: -1, 0, 1).

        Returns:
            EvaluationResult with metrics.
        """
        if self.config.verbose:
            logger.info("Evaluating Ridge Classifier on test set...")

        result = evaluate_model(
            model, X_test, y_test,
            verbose=self.config.verbose,
        )

        return result

    def run(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        skip_optimization: bool = False,
        default_params: dict[str, Any] | None = None,
    ) -> RidgePipelineResult:
        """Run the complete pipeline: optimize, train, evaluate.

        Args:
            X_train: Training features.
            y_train: Training target (class labels: -1, 0, 1).
            X_test: Test features.
            y_test: Test target (class labels: -1, 0, 1).
            skip_optimization: If True, skip optimization and use default_params.
            default_params: Parameters to use if skipping optimization.

        Returns:
            RidgePipelineResult with all results.
        """
        if self.config.verbose:
            logger.info("=" * 60)
            logger.info("Starting Ridge Classifier Pipeline")
            logger.info("=" * 60)

        # Step 1: Optimization
        opt_result = None
        if skip_optimization:
            if default_params is None:
                default_params = {"alpha": 1.0}
            best_params = default_params
            if self.config.verbose:
                logger.info("Skipping optimization, using default parameters")
        else:
            opt_result = self.optimize(X_train, y_train)
            best_params = opt_result.best_params

        # Step 2: Training
        model, train_result = self.train(X_train, y_train, best_params)

        # Step 3: Evaluation
        eval_result = self.evaluate(model, X_test, y_test)

        # Save results if output_dir specified
        if self.config.output_dir is not None:
            self._save_results(opt_result, train_result, eval_result, best_params, model)

        result = RidgePipelineResult(
            optimization_result=opt_result,
            training_result=train_result,
            evaluation_result=eval_result,
            best_params=best_params,
            model=model,
        )

        if self.config.verbose:
            print(result.summary())

        return result

    def _save_results(
        self,
        opt_result: OptimizationResult | None,
        train_result: TrainingResult | CrossValidationTrainingResult,
        eval_result: EvaluationResult,
        best_params: dict[str, Any],
        model: RidgeClassifierModel,
    ) -> None:
        """Save pipeline results to output directory."""
        if self.config.output_dir is None:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save optimization results
        if opt_result is not None:
            opt_data = {
                "best_params": opt_result.best_params,
                "best_score": opt_result.best_score,
                "n_trials": opt_result.n_trials,
                "n_completed": opt_result.n_completed,
            }
            save_json_pretty(opt_data, output_dir / "optimization_results.json")

        # Save evaluation results
        eval_data = eval_result.to_dict()
        save_json_pretty(eval_data, output_dir / "evaluation_results.json")

        # Save best params
        save_json_pretty(best_params, output_dir / "best_params.json")

        # Save model coefficients
        if model.is_fitted:
            coef_data = {
                "coefficients": model.coef_.tolist(),
                "intercept": model.intercept_.tolist() if hasattr(model.intercept_, "tolist") else float(model.intercept_),
            }
            save_json_pretty(coef_data, output_dir / "model_coefficients.json")

        logger.info("Results saved to %s", output_dir)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def run_ridge_pipeline(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    purge_gap: int = 5,
    verbose: bool = True,
) -> RidgePipelineResult:
    """Convenience function to run the complete Ridge pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        n_trials: Number of optimization trials.
        n_splits: Number of CV splits.
        purge_gap: Samples to purge between train/test.
        verbose: Show progress.

    Returns:
        RidgePipelineResult with all results.

    Example:
        >>> result = run_ridge_pipeline(X_train, y_train, X_test, y_test)
        >>> print(f"Test MSE: {result.evaluation_result.metrics['mse']:.4f}")
    """
    config = RidgePipelineConfig(
        n_trials=n_trials,
        n_splits=n_splits,
        purge_gap=purge_gap,
        verbose=verbose,
    )
    pipeline = RidgePipeline(config)
    return pipeline.run(X_train, y_train, X_test, y_test)


def quick_ridge(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    alpha: float = 1.0,
) -> tuple[RidgeClassifierModel, dict[str, float]]:
    """Quick training and evaluation without optimization.

    Args:
        X_train: Training features.
        y_train: Training target (class labels: -1, 0, 1).
        X_test: Test features.
        y_test: Test target (class labels: -1, 0, 1).
        alpha: Regularization parameter.

    Returns:
        Tuple of (trained model, test metrics dict).
    """
    model = RidgeClassifierModel(alpha=alpha)
    model.fit(X_train, y_train)

    result = evaluate_model(model, X_test, y_test, verbose=False)
    return model, result.metrics
