"""Complete pipeline for LSTM model: optimization, training, and evaluation.

This module provides a unified pipeline for:
1. Hyperparameter optimization with Optuna + walk-forward CV
2. Model training with the best parameters
3. Model evaluation with comprehensive metrics

Note: LSTM requires sequence-based data handling, which is handled internally
by the LSTMModel class.
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
from src.model.deep_learning.lstm_model import LSTMModel
from src.optimisation import (
    LSTMHyperparams,
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    create_cv_config,
)
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
class LSTMPipelineConfig:
    """Configuration for the LSTM pipeline.

    Attributes:
        n_trials: Number of optimization trials.
        n_splits: Number of CV splits for optimization.
        purge_gap: Samples to purge between train and test.
        min_train_size: Minimum training set size.
        validation_split: Validation split for final training.
        metric: Metric to optimize/evaluate.
        output_dir: Directory for saving outputs.
        random_state: Random seed.
        verbose: Verbosity level.
        sequence_length: Default sequence length for LSTM.
        epochs: Default epochs for training.
        patience: Default patience for early stopping.
    """

    n_trials: int = 30
    n_splits: int = 3
    purge_gap: int = 10
    min_train_size: int = 300
    validation_split: float = 0.2
    metric: str = "mse"
    output_dir: Path | None = None
    random_state: int = 42
    verbose: bool = True
    sequence_length: int = 10
    epochs: int = 100
    patience: int = 10


# ============================================================================
# PIPELINE RESULTS
# ============================================================================


@dataclass
class LSTMPipelineResult:
    """Complete result from the LSTM pipeline.

    Attributes:
        optimization_result: Hyperparameter optimization results.
        training_result: Model training results.
        evaluation_result: Model evaluation results.
        best_params: Best hyperparameters found.
        model: Trained model with best parameters.
        training_history: Training loss history.
    """

    optimization_result: OptimizationResult | None
    training_result: TrainingResult | CrossValidationTrainingResult
    evaluation_result: EvaluationResult
    best_params: dict[str, Any]
    model: LSTMModel
    training_history: dict[str, list[float]] | None = None

    def summary(self) -> str:
        """Generate a summary of the pipeline results."""
        lines = [
            "=" * 60,
            "LSTM Pipeline Results",
            "=" * 60,
            "",
        ]

        if self.optimization_result is not None:
            lines.extend([
                "OPTIMIZATION:",
                f"  Trials: {self.optimization_result.n_completed}/{self.optimization_result.n_trials}",
                f"  Best CV score: {self.optimization_result.best_score:.6f}",
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
                f"  CV score: {self.training_result.cv_result.mean_score:.6f} "
                f"(+/- {self.training_result.cv_result.std_score:.6f})",
                f"  Final train score: {self.training_result.final_train_score:.6f}",
            ])
        else:
            lines.extend([
                f"  Train score: {self.training_result.train_score:.6f}",
            ])
            if self.training_result.val_score is not None:
                lines.append(f"  Val score: {self.training_result.val_score:.6f}")

        if self.training_history is not None:
            train_losses = self.training_history.get("train_loss", [])
            if train_losses:
                lines.append(f"  Final train loss: {train_losses[-1]:.6f}")
            val_losses = self.training_history.get("val_loss", [])
            if val_losses:
                lines.append(f"  Final val loss: {val_losses[-1]:.6f}")

        lines.extend([
            "",
            "EVALUATION (Test Set):",
        ])
        for metric, value in self.evaluation_result.metrics.items():
            lines.append(f"  {metric}: {value:.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# LSTM PIPELINE
# ============================================================================


class LSTMPipeline:
    """Complete pipeline for LSTM: optimize, train, evaluate.

    Example:
        >>> pipeline = LSTMPipeline(config=LSTMPipelineConfig(
        ...     n_trials=20,
        ...     n_splits=3,
        ...     purge_gap=20,
        ... ))
        >>> result = pipeline.run(X_train, y_train, X_test, y_test)
        >>> print(result.summary())
    """

    def __init__(self, config: LSTMPipelineConfig | None = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or LSTMPipelineConfig()

    def optimize(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        hyperparam_space: LSTMHyperparams | None = None,
    ) -> OptimizationResult:
        """Optimize hyperparameters using Optuna with walk-forward CV.

        Args:
            X: Training features.
            y: Training target.
            hyperparam_space: Custom hyperparameter space.

        Returns:
            OptimizationResult with best parameters.
        """
        if self.config.verbose:
            logger.info("Starting LSTM hyperparameter optimization...")

        # Create configurations - LSTM needs larger train sizes due to sequences
        cv_config = create_cv_config(
            n_splits=self.config.n_splits,
            purge_gap=self.config.purge_gap,
            min_train_size=self.config.min_train_size,
        )

        opt_config = OptimizationConfig(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state,
        )

        space = hyperparam_space or LSTMHyperparams()

        # Run optimization
        optimizer = OptunaOptimizer(
            model_class=LSTMModel,
            hyperparam_space=space,
            cv_config=cv_config,
            optimization_config=opt_config,
            fixed_model_params={
                "sequence_length": self.config.sequence_length,
                "epochs": min(self.config.epochs, 50),  # Reduce for optimization
                "patience": 5,
            },
        )

        result = optimizer.optimize(
            X, y,
            metric=self.config.metric,
            verbose=self.config.verbose,
        )

        if self.config.verbose:
            logger.info("Optimization complete. Best %s: %.6f", self.config.metric, result.best_score)

        return result

    def train(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        params: dict[str, Any],
        use_cv: bool = False,
    ) -> tuple[LSTMModel, TrainingResult | CrossValidationTrainingResult]:
        """Train the model with given parameters.

        Args:
            X: Training features.
            y: Training target.
            params: Model hyperparameters.
            use_cv: If True, use walk-forward CV for training.
                   Default False for LSTM due to training time.

        Returns:
            Tuple of (trained model, training result).
        """
        if self.config.verbose:
            logger.info("Training LSTM with optimized parameters...")

        # Create model with best params
        model = LSTMModel(
            sequence_length=self.config.sequence_length,
            epochs=self.config.epochs,
            patience=self.config.patience,
            **params,
        )

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
        model: LSTMModel,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
    ) -> EvaluationResult:
        """Evaluate the model on test data.

        Note: LSTM predictions will be shorter than y_test due to sequence creation.
        The evaluation accounts for this offset automatically.

        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target.

        Returns:
            EvaluationResult with metrics.
        """
        if self.config.verbose:
            logger.info("Evaluating LSTM on test set...")

        # LSTM predictions are offset by sequence_length
        # We need to align y_test accordingly
        y_test_arr = np.asarray(y_test).ravel()
        predictions = model.predict(X_test)

        # Align targets with predictions (skip first sequence_length samples)
        y_aligned = y_test_arr[model.sequence_length:]

        # Ensure same length
        min_len = min(len(predictions), len(y_aligned))
        predictions = predictions[:min_len]
        y_aligned = y_aligned[:min_len]

        # Use evaluator directly with aligned data
        from src.evaluation.metrics import compute_metrics, compute_residual_diagnostics

        metrics = compute_metrics(y_aligned, predictions, ["mse", "rmse", "mae", "r2", "mape"])
        residual_diag = compute_residual_diagnostics(y_aligned, predictions)

        result = EvaluationResult(
            model_name=model.name,
            metrics=metrics,
            predictions=predictions,
            actuals=y_aligned,
            residuals=y_aligned - predictions,
            residual_diagnostics=residual_diag,
            mz_r2=None,
            mz_alpha=None,
            mz_beta=None,
            evaluation_time=0.0,
            n_samples=len(y_aligned),
            model_params=model.get_params(),
        )

        if self.config.verbose:
            logger.info("Evaluation complete. MSE: %.6f", metrics.get("mse", float("nan")))

        return result

    def run(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        skip_optimization: bool = False,
        default_params: dict[str, Any] | None = None,
    ) -> LSTMPipelineResult:
        """Run the complete pipeline: optimize, train, evaluate.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            skip_optimization: If True, skip optimization and use default_params.
            default_params: Parameters to use if skipping optimization.

        Returns:
            LSTMPipelineResult with all results.
        """
        if self.config.verbose:
            logger.info("=" * 60)
            logger.info("Starting LSTM Pipeline")
            logger.info("=" * 60)

        # Step 1: Optimization
        opt_result = None
        if skip_optimization:
            if default_params is None:
                default_params = {
                    "hidden_size": 32,
                    "num_layers": 1,
                    "dropout": 0.1,
                    "learning_rate": 0.001,
                }
            best_params = default_params
            if self.config.verbose:
                logger.info("Skipping optimization, using default parameters")
        else:
            opt_result = self.optimize(X_train, y_train)
            best_params = opt_result.best_params

        # Step 2: Training
        model, train_result = self.train(X_train, y_train, best_params, use_cv=False)

        # Get training history
        training_history = model.get_history() if hasattr(model, "get_history") else None

        # Step 3: Evaluation
        eval_result = self.evaluate(model, X_test, y_test)

        # Save results if output_dir specified
        if self.config.output_dir is not None:
            self._save_results(opt_result, train_result, eval_result, best_params, training_history)

        result = LSTMPipelineResult(
            optimization_result=opt_result,
            training_result=train_result,
            evaluation_result=eval_result,
            best_params=best_params,
            model=model,
            training_history=training_history,
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
        training_history: dict[str, list[float]] | None,
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

        # Save training history
        if training_history is not None:
            save_json_pretty(training_history, output_dir / "training_history.json")

        logger.info("Results saved to %s", output_dir)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def run_lstm_pipeline(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    n_trials: int = 30,
    n_splits: int = 3,
    purge_gap: int = 10,
    sequence_length: int = 10,
    verbose: bool = True,
) -> LSTMPipelineResult:
    """Convenience function to run the complete LSTM pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        n_trials: Number of optimization trials.
        n_splits: Number of CV splits.
        purge_gap: Samples to purge between train/test.
        sequence_length: Sequence length for LSTM.
        verbose: Show progress.

    Returns:
        LSTMPipelineResult with all results.

    Example:
        >>> result = run_lstm_pipeline(X_train, y_train, X_test, y_test)
        >>> print(f"Test MSE: {result.evaluation_result.metrics['mse']:.4f}")
    """
    config = LSTMPipelineConfig(
        n_trials=n_trials,
        n_splits=n_splits,
        purge_gap=purge_gap,
        sequence_length=sequence_length,
        verbose=verbose,
    )
    pipeline = LSTMPipeline(config)
    return pipeline.run(X_train, y_train, X_test, y_test)


def quick_lstm(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    sequence_length: int = 10,
) -> tuple[LSTMModel, dict[str, float]]:
    """Quick training and evaluation without optimization.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        sequence_length: Sequence length for LSTM.

    Returns:
        Tuple of (trained model, test metrics dict).
    """
    model = LSTMModel(
        hidden_size=32,
        num_layers=1,
        sequence_length=sequence_length,
        epochs=50,
        patience=10,
    )
    model.fit(X_train, y_train, verbose=False)

    # Evaluate with aligned data
    y_test_arr = np.asarray(y_test).ravel()
    predictions = model.predict(X_test)
    y_aligned = y_test_arr[sequence_length:]

    min_len = min(len(predictions), len(y_aligned))
    predictions = predictions[:min_len]
    y_aligned = y_aligned[:min_len]

    from src.evaluation.metrics import compute_metrics
    metrics = compute_metrics(y_aligned, predictions, ["mse", "rmse", "mae", "r2"])

    return model, metrics
