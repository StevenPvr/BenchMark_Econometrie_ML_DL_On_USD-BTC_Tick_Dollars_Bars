"""Ridge Classifier for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import RidgeClassifier  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class RidgeClassifierModel(BaseModel):
    """
    Ridge Classifier (L2 regularized linear classifier).

    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ridge Classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength (L2 penalty).
        fit_intercept : bool, default=True
            Whether to calculate the intercept.
        normalize : bool, default=True
            Whether to normalize features before training.
        """
        super().__init__(
            name="RidgeClassifier",
            alpha=alpha,
            fit_intercept=fit_intercept,
            normalize=normalize,
            **kwargs,
        )
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scaler: StandardScaler | None = None
        self.model: RidgeClassifier | None = None
        self.classes_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
        }

    def set_params(self, **params: Any) -> "RidgeClassifierModel":
        """Update model hyperparameters."""
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "fit_intercept" in params:
            self.fit_intercept = params["fit_intercept"]
        if "normalize" in params:
            self.normalize = params["normalize"]
        return self

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "RidgeClassifierModel":
        """Train the Ridge Classifier."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)

        if self.normalize:
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        self.model = RidgeClassifier(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver="lsqr",  # Iterative solver, optimal for large dense datasets
        )
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        X_arr = np.asarray(X)
        if self.normalize and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        return np.asarray(self.model.predict(X_arr))

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for all classes.

        Uses decision function with softmax normalization.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        X_arr = np.asarray(X)
        if self.normalize and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        # Get decision function values
        decision = self.model.decision_function(X_arr)

        # Apply softmax to convert to probabilities
        if decision.ndim == 1:
            # Binary case
            exp_decision = np.exp(decision - np.max(decision))
            proba_pos = exp_decision / (1 + exp_decision)
            return np.column_stack([1 - proba_pos, proba_pos])
        else:
            # Multi-class case
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)

    @property
    def coef_(self) -> np.ndarray:
        """Return model coefficients."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return np.asarray(self.model.coef_)

    @property
    def intercept_(self) -> np.ndarray:
        """Return model intercept."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return np.asarray(self.model.intercept_)
