"""Logistic Regression Classifier (L2) for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class LogisticClassifierModel(BaseModel):
    """
    Logistic Regression Classifier (L2 regularized).

    Standard logistic regression with L2 penalty, well-suited for
    financial time series classification (dollar bars).

    Parameters tuned for financial markets:
    - C: Inverse regularization strength (0.01 to 10.0)
    - Higher C = less regularization, risk of overfitting on noisy financial data
    - Lower C = more regularization, better generalization

    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: str | dict | None = "balanced",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Logistic Regression Classifier.

        Parameters
        ----------
        C : float, default=1.0
            Inverse of regularization strength (smaller = stronger L2 penalty).
            For financial data: 0.01-10.0 range recommended.
        fit_intercept : bool, default=True
            Whether to calculate the intercept.
        normalize : bool, default=True
            Whether to normalize features before training.
            Critical for financial features with different scales.
        max_iter : int, default=1000
            Maximum iterations for convergence.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        class_weight : str or dict, default="balanced"
            Weights for classes. "balanced" adjusts weights inversely
            proportional to class frequencies (important for imbalanced
            triple-barrier labels).
        """
        super().__init__(
            name="LogisticClassifier",
            C=C,
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            **kwargs,
        )
        self.C = C
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "class_weight": self.class_weight,
        }

    def set_params(self, **params: Any) -> "LogisticClassifierModel":
        """Update model hyperparameters."""
        if "C" in params:
            self.C = params["C"]
        if "fit_intercept" in params:
            self.fit_intercept = params["fit_intercept"]
        if "normalize" in params:
            self.normalize = params["normalize"]
        if "max_iter" in params:
            self.max_iter = params["max_iter"]
        if "tol" in params:
            self.tol = params["tol"]
        if "class_weight" in params:
            self.class_weight = params["class_weight"]
        return self

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "LogisticClassifierModel":
        """Train the Logistic Regression Classifier."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)

        if self.normalize:
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        # Use lbfgs solver for L2 penalty (fast and stable)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.model = LogisticRegression(
                penalty="l2",
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                solver="lbfgs",
                class_weight=self.class_weight,
                n_jobs=-1,
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

        return self.model.predict(X_arr)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for all classes.

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

        return self.model.predict_proba(X_arr)

    @property
    def coef_(self) -> np.ndarray:
        """Return model coefficients."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Return model intercept."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return np.asarray(self.model.intercept_)
