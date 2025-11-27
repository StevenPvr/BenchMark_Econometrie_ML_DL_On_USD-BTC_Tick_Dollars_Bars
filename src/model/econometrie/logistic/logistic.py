"""Logistic Regression (no penalty) for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class LogisticModel(BaseModel):
    """
    Logistic Regression without regularization.

    Baseline classifier equivalent to OLS for regression.
    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: int = 10000,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Logistic Regression.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether to calculate the intercept.
        normalize : bool, default=True
            Whether to normalize features before training.
        max_iter : int, default=10000
            Maximum iterations for convergence.
        tol : float, default=1e-4
            Tolerance for stopping criteria.
        """
        super().__init__(
            name="Logistic",
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            **kwargs,
        )
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params: Any) -> "LogisticModel":
        """Update model hyperparameters."""
        if "fit_intercept" in params:
            self.fit_intercept = params["fit_intercept"]
        if "normalize" in params:
            self.normalize = params["normalize"]
        if "max_iter" in params:
            self.max_iter = params["max_iter"]
        if "tol" in params:
            self.tol = params["tol"]
        return self

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "LogisticModel":
        """Train the Logistic Regression model."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)

        if self.normalize:
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        # Use multinomial for multi-class, no penalty
        self.model = LogisticRegression(
            penalty=None,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            multi_class="multinomial",
            solver="lbfgs",
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
        return self.model.intercept_
