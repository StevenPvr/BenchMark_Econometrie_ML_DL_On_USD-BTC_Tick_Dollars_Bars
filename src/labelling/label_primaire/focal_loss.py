"""
Focal Loss implementation for imbalanced multi-class classification.

Focal Loss down-weights well-classified examples and focuses on hard, misclassified ones.
This is particularly useful for the primary model where long (+1) and short (-1) classes
are minorities compared to neutral (0).

Formula: FL(p) = -alpha * (1 - p)^gamma * log(p)

Where:
- p: predicted probability for the true class
- gamma: focusing parameter (higher = more focus on hard examples)
- alpha: class weight (to handle class imbalance)

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np


def focal_loss_lgb_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Focal Loss for LightGBM binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0 or 1).
    y_pred : np.ndarray
        Raw predictions (logits).
    gamma : float, default=2.0
        Focusing parameter. Higher values increase focus on hard examples.
    alpha : float, default=0.25
        Weighting factor for the positive class.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (gradient, hessian) for LightGBM custom objective.
    """
    # Sigmoid to get probabilities
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Focal weight: (1 - p_t)^gamma
    p_t = np.where(y_true == 1, p, 1 - p)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weight
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    # Gradient and Hessian
    grad = focal_weight * alpha_t * (p - y_true)
    hess = focal_weight * alpha_t * p * (1 - p) + 1e-6

    return grad, hess


def focal_loss_lgb_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: np.ndarray | None = None,
    n_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Focal Loss for LightGBM multi-class classification.

    This is designed for triple-barrier labeling with classes: -1, 0, +1.
    Internal mapping: -1 -> 0, 0 -> 1, +1 -> 2

    Parameters
    ----------
    y_true : np.ndarray
        True labels. Can be [-1, 0, 1] or [0, 1, 2] format.
    y_pred : np.ndarray
        Raw predictions (logits) flattened in Fortran order.
        Shape: (n_samples * n_classes,)
    gamma : float, default=2.0
        Focusing parameter. Controls how much to down-weight easy examples.
        - gamma=0: Equivalent to cross-entropy (no focusing)
        - gamma=1: Moderate focusing
        - gamma=2: Standard focal loss (recommended)
        - gamma=5: Aggressive focusing
    alpha : np.ndarray, optional
        Class weights array of shape (n_classes,).
        Higher values = more importance for that class.
        Example: [2.0, 0.5, 2.0] to emphasize classes 0 and 2 (long/short).
    n_classes : int, default=3
        Number of classes.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (gradient, hessian) for LightGBM custom objective.
        Both are flattened arrays of shape (n_samples * n_classes,).
    """
    n_samples = len(y_true)

    # Handle label format: convert [-1, 0, 1] to [0, 1, 2] if needed
    y_true_int = y_true.astype(int)
    if y_true_int.min() < 0:
        y_true_int = y_true_int + 1  # Map -1,0,1 -> 0,1,2

    # Reshape predictions: (n_samples * n_classes,) -> (n_samples, n_classes)
    y_pred_2d = y_pred.reshape((n_samples, n_classes), order="F")

    # Softmax to get probabilities
    y_pred_max = np.max(y_pred_2d, axis=1, keepdims=True)
    exp_pred = np.exp(y_pred_2d - y_pred_max)  # Numerical stability
    proba = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    proba = np.clip(proba, 1e-15, 1 - 1e-15)

    # One-hot encoding of true labels
    y_true_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
    y_true_onehot[np.arange(n_samples), y_true_int] = 1.0

    # Probability of true class: p_t
    p_t = np.sum(y_true_onehot * proba, axis=1, keepdims=True)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Alpha weights (per class)
    if alpha is not None:
        alpha = np.asarray(alpha).reshape(1, n_classes)
        alpha_weight = np.sum(y_true_onehot * alpha, axis=1, keepdims=True)
        focal_weight = focal_weight * alpha_weight

    # Gradient: focal_weight * (proba - y_true_onehot)
    grad = focal_weight * (proba - y_true_onehot)

    # Hessian (approximation): focal_weight * proba * (1 - proba)
    hess = focal_weight * proba * (1 - proba) + 1e-6

    # Flatten in Fortran order (column-major) for LightGBM
    return grad.flatten("F"), hess.flatten("F")


def create_focal_loss_objective(
    gamma: float = 2.0,
    alpha: np.ndarray | None = None,
    n_classes: int = 3,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Factory function to create a focal loss objective for LightGBM.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter.
    alpha : np.ndarray, optional
        Class weights. For triple-barrier: [weight_short, weight_neutral, weight_long]
        Example: [2.0, 0.5, 2.0] to emphasize long/short.
    n_classes : int, default=3
        Number of classes.

    Returns
    -------
    Callable
        Objective function compatible with LightGBM's `objective` parameter.

    Example
    -------
    >>> from src.labelling.label_primaire.focal_loss import create_focal_loss_objective
    >>> import lightgbm as lgb
    >>>
    >>> # Create objective with gamma=2 and class weights
    >>> focal_obj = create_focal_loss_objective(
    ...     gamma=2.0,
    ...     alpha=np.array([2.0, 0.5, 2.0])  # Emphasize long/short
    ... )
    >>>
    >>> # Use in LightGBM
    >>> model = lgb.LGBMClassifier(objective=focal_obj)
    """

    def objective(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return focal_loss_lgb_multiclass(
            y_true, y_pred, gamma=gamma, alpha=alpha, n_classes=n_classes
        )

    return objective


def compute_focal_alpha_from_class_weights(
    class_weights: Dict[int, float],
    classes: tuple[int, ...] = (-1, 0, 1),
) -> np.ndarray:
    """
    Convert sklearn-style class_weight dict to alpha array for focal loss.

    Parameters
    ----------
    class_weights : Dict[int, float]
        Class weights from sklearn's compute_class_weight.
        Keys are class labels, values are weights.
    classes : tuple, default=(-1, 0, 1)
        Ordered class labels.

    Returns
    -------
    np.ndarray
        Alpha array in order [class_0, class_1, class_2]
        For triple-barrier: [weight_-1, weight_0, weight_1]
    """
    alpha = np.array([class_weights.get(c, 1.0) for c in classes])
    # Normalize to sum to n_classes (optional, keeps scale similar)
    alpha = alpha * len(classes) / alpha.sum()
    return alpha


def get_default_alpha_for_imbalanced(
    y: np.ndarray,
    minority_boost: float = 2.0,
) -> np.ndarray:
    """
    Compute default alpha weights based on class distribution.

    Parameters
    ----------
    y : np.ndarray
        Labels array (can be [-1, 0, 1] or [0, 1, 2]).
    minority_boost : float, default=2.0
        Multiplier for minority classes (long/short).

    Returns
    -------
    np.ndarray
        Alpha weights [alpha_-1, alpha_0, alpha_1] or [alpha_0, alpha_1, alpha_2].
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    proportions = counts / total

    # Inverse frequency weighting
    weights = 1.0 / (proportions + 1e-6)
    weights = weights / weights.sum() * len(unique)

    # Apply minority boost to non-majority classes
    majority_idx = np.argmax(counts)
    for i in range(len(weights)):
        if i != majority_idx:
            weights[i] *= minority_boost

    # Normalize again
    weights = weights / weights.sum() * len(unique)

    return weights
