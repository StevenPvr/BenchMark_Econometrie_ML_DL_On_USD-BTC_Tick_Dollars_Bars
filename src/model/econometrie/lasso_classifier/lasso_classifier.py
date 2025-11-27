"""Lasso Classifier (L1 Logistic) for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class LassoClassifierModel(BaseModel):
    """
    Lasso Classifier (L1 regularized logistic regression).

    Uses LogisticRegression with L1 penalty for sparse feature selection.
    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: int = 10000,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Lasso Classifier.

        Parameters
        ----------
        C : float, default=1.0
            Inverse of regularization strength (smaller = stronger L1 penalty).
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
            name="LassoClassifier",
            C=C,
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            **kwargs,
        )
        self.C = C
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
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params: Any) -> "LassoClassifierModel":
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
        return self

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "LassoClassifierModel":
        """Train the Lasso Classifier."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)

        if self.normalize:
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        # Use multinomial for multi-class
        self.model = LogisticRegression(
            penalty="l1",
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            solver="saga",  # saga supports L1 and multi-class
            multi_class="multinomial",
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

    def get_selected_features(self, feature_names: list[str] | None = None) -> list[str]:
        """
        Get features selected by L1 regularization (non-zero coefficients).

        Parameters
        ----------
        feature_names : list[str], optional
            Names of features. If None, returns indices.

        Returns
        -------
        list[str]
            Names or indices of selected features.
        """
        if self.model is None:
            raise ValueError("Model not fitted.")

        # For multi-class, check if any coefficient is non-zero
        coef = self.model.coef_
        if coef.ndim == 1:
            non_zero_mask = coef != 0
        else:
            non_zero_mask = np.any(coef != 0, axis=0)

        non_zero_indices = np.where(non_zero_mask)[0]

        if feature_names is not None:
            return [feature_names[i] for i in non_zero_indices]
        return [str(i) for i in non_zero_indices]
