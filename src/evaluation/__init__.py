"""Evaluation module for all model types.

This module provides a unified evaluation interface compatible with:
- Econometric models (Ridge, Lasso)
- Machine Learning models (XGBoost, LightGBM, CatBoost, RandomForest)
- Deep Learning models (LSTM)

Key Features:
- Comprehensive regression metrics (MSE, RMSE, MAE, R², MAPE, etc.)
- Volatility forecasting metrics (QLIKE, MZ regression)
- Residual diagnostics (normality, autocorrelation)
- Model comparison with statistical significance tests
- Easy-to-use convenience functions

Example Usage:
    >>> from src.model.machine_learning.xgboost_model import XGBoostModel
    >>> from src.evaluation import evaluate_model, compare_models
    >>>
    >>> # Evaluate a single model
    >>> model = XGBoostModel(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> result = evaluate_model(model, X_test, y_test)
    >>> print(f"MSE: {result.metrics['mse']:.4f}")
    >>> print(f"R²: {result.metrics['r2']:.4f}")
    >>>
    >>> # Compare multiple models
    >>> result = compare_models(
    ...     {"xgb": xgb_model, "lgb": lgb_model},
    ...     X_test, y_test,
    ... )
    >>> print(f"Best model: {result.best_model}")

Computing specific metrics:
    >>> from src.evaluation import mse, rmse, mae, r2_score, qlike
    >>> mse_val = mse(y_true, y_pred)
    >>> r2_val = r2_score(y_true, y_pred)
"""

from __future__ import annotations

# Metrics
from src.evaluation.metrics import (
    ALL_METRICS,
    DIRECTION_METRICS,
    REGRESSION_METRICS,
    VOLATILITY_METRICS,
    ResidualDiagnostics,
    adjusted_r2,
    aic,
    bic,
    compute_metrics,
    compute_residual_diagnostics,
    direction_accuracy,
    get_metric,
    hit_rate,
    list_available_metrics,
    mae,
    mape,
    max_error,
    median_absolute_error,
    mincer_zarnowitz_r2,
    mse,
    mse_log,
    qlike,
    r2_score,
    rmse,
    smape,
)

# Evaluator
from src.evaluation.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    PredictableModel,
    evaluate_model,
    quick_evaluate,
    save_evaluation_results,
)

# Comparison
from src.evaluation.comparison import (
    ModelComparisonResult,
    ModelComparator,
    PairwiseTestResult,
    compare_models,
    diebold_mariano_test,
    pairwise_dm_tests,
    rank_models,
    save_comparison_results,
)

__all__ = [
    # Metric functions
    "mse",
    "rmse",
    "mae",
    "mape",
    "smape",
    "r2_score",
    "adjusted_r2",
    "max_error",
    "median_absolute_error",
    "qlike",
    "mse_log",
    "direction_accuracy",
    "hit_rate",
    "aic",
    "bic",
    "mincer_zarnowitz_r2",
    # Metric utilities
    "get_metric",
    "compute_metrics",
    "list_available_metrics",
    "compute_residual_diagnostics",
    # Metric registries
    "ALL_METRICS",
    "REGRESSION_METRICS",
    "VOLATILITY_METRICS",
    "DIRECTION_METRICS",
    # Data classes
    "ResidualDiagnostics",
    "EvaluationResult",
    "EvaluationConfig",
    "ModelComparisonResult",
    "PairwiseTestResult",
    # Evaluator
    "Evaluator",
    "PredictableModel",
    "evaluate_model",
    "quick_evaluate",
    "save_evaluation_results",
    # Comparison
    "ModelComparator",
    "compare_models",
    "rank_models",
    "diebold_mariano_test",
    "pairwise_dm_tests",
    "save_comparison_results",
]
