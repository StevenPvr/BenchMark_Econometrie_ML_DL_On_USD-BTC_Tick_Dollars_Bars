"""OLS (Ordinary Least Squares) regression model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.model.base import BaseModel


class OLSModel(BaseModel):
    """
    Ordinary Least Squares (OLS) regression.

    Baseline linear model without regularization.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele OLS.

        Parameters
        ----------
        fit_intercept : bool, default=True
            Si True, calcule l'intercept.
        normalize : bool, default=True
            Si True, normalise les features avant l'entrainement.
        """
        super().__init__(
            name="OLS",
            fit_intercept=fit_intercept,
            normalize=normalize,
            **kwargs,
        )
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scaler: StandardScaler | None = None
        self.model: LinearRegression | None = None

    def get_params(self) -> dict[str, Any]:
        """Retourne les hyperparametres du modele."""
        return {
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
        }

    def set_params(self, **params: Any) -> "OLSModel":
        """Met a jour les hyperparametres du modele."""
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
    ) -> "OLSModel":
        """Entraine le modele OLS."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if self.normalize:
            self.scaler = StandardScaler()
            assert self.scaler is not None  # Type checker assurance
            X_arr = self.scaler.fit_transform(X_arr)

        self.model = LinearRegression(
            fit_intercept=self.fit_intercept,
        )
        assert self.model is not None  # Type checker assurance
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele OLS."""
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
