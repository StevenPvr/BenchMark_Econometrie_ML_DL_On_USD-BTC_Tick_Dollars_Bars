"""Lasso regression model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import Lasso  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class LassoModel(BaseModel):
    """
    Lasso regression (L1 regularization).

    Utile pour la selection de features (sparse solutions).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele Lasso.

        Parameters
        ----------
        alpha : float, default=1.0
            Parametre de regularisation L1.
        fit_intercept : bool, default=True
            Si True, calcule l'intercept.
        normalize : bool, default=True
            Si True, normalise les features avant l'entrainement.
        max_iter : int, default=1000
            Nombre maximum d'iterations.
        tol : float, default=1e-4
            Tolerance pour la convergence.
        """
        super().__init__(name="Lasso", **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.scaler: StandardScaler | None = None
        self.model: Lasso | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "LassoModel":
        """Entraine le modele Lasso."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if self.normalize:
            self.scaler = StandardScaler()
            assert self.scaler is not None  # Type checker assurance
            X_arr = self.scaler.fit_transform(X_arr)

        self.model = Lasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        assert self.model is not None  # Type checker assurance
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele Lasso."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        X_arr = np.asarray(X)
        if self.normalize and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        return self.model.predict(X_arr)

    @property
    def coef_(self) -> np.ndarray:
        """Retourne les coefficients du modele."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.coef_

    @property
    def intercept_(self) -> float:
        """Retourne l'intercept du modele."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.intercept_

    def get_selected_features(self, feature_names: list[str] | None = None) -> list[str | int]:
        """
        Retourne les features selectionnees (coefficients non-nuls).

        Parameters
        ----------
        feature_names : list[str], optional
            Noms des features. Si None, retourne les indices.

        Returns
        -------
        list
            Features selectionnees.
        """
        if self.model is None:
            raise ValueError("Model not fitted.")

        non_zero_idx = np.where(self.model.coef_ != 0)[0]

        if feature_names is not None:
            return [feature_names[i] for i in non_zero_idx]
        return list(non_zero_idx)
