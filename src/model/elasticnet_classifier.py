"""Elastic Net Classifier (L1+L2) for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class ElasticNetClassifierModel(BaseModel):
    """
    Elastic Net Classifier (L1 + L2 regularized logistic regression).

    Combines L1 (sparsity) and L2 (ridge) penalties for robust feature
    selection and regularization. Well-suited for high-dimensional
    financial features from dollar bars.

    Parameters tuned for financial markets:
    - C: Inverse regularization strength (0.01 to 10.0)
    - l1_ratio: Balance between L1 and L2 (0.0 = pure L2, 1.0 = pure L1)
      - 0.1-0.3: Mostly L2, slight sparsity
      - 0.5: Equal balance
      - 0.7-0.9: Mostly L1, strong feature selection

    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
    """

    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: str | dict | None = "balanced",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Elastic Net Classifier.

        Parameters
        ----------
        C : float, default=1.0
            Inverse of regularization strength (smaller = stronger penalty).
            For financial data: 0.01-10.0 range recommended.
        l1_ratio : float, default=0.5
            The ElasticNet mixing parameter:
            - 0.0: Pure L2 penalty (Ridge)
            - 1.0: Pure L1 penalty (Lasso)
            - 0.5: Equal mix (recommended starting point)
            For financial data: 0.1-0.9 range.
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
            proportional to class frequencies.
        """
        super().__init__(
            name="ElasticNetClassifier",
            C=C,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            **kwargs,
        )
        self.C = C
        self.l1_ratio = l1_ratio
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
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "class_weight": self.class_weight,
        }

    def set_params(self, **params: Any) -> "ElasticNetClassifierModel":
        """Update model hyperparameters."""
        if "C" in params:
            self.C = params["C"]
        if "l1_ratio" in params:
            self.l1_ratio = params["l1_ratio"]
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
    ) -> "ElasticNetClassifierModel":
        """Train the Elastic Net Classifier."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)

        if self.normalize:
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        # Use saga solver for elasticnet penalty
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.model = LogisticRegression(
                penalty="elasticnet",
                C=self.C,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                solver="saga",  # Only solver supporting elasticnet
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

    def get_selected_features(self, feature_names: list[str] | None = None) -> list[str]:
        """
        Get features selected by L1 component (non-zero coefficients).

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

    def get_sparsity_ratio(self) -> float:
        """
        Get the sparsity ratio (proportion of zero coefficients).

        Returns
        -------
        float
            Sparsity ratio between 0 (dense) and 1 (fully sparse).
        """
        if self.model is None:
            raise ValueError("Model not fitted.")

        coef = self.model.coef_
        total_coefs = coef.size
        zero_coefs = np.sum(coef == 0)
        return zero_coefs / total_coefs
