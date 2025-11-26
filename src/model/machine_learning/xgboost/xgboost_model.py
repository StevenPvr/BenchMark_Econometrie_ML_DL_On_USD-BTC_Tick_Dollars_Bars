"""XGBoost model for regression/classification."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore

from src.model.base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost (eXtreme Gradient Boosting) model.

    Performant pour les donnees tabulaires.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "reg:squarederror",
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele XGBoost.

        Parameters
        ----------
        n_estimators : int, default=100
            Nombre d'arbres.
        max_depth : int, default=6
            Profondeur maximale des arbres.
        learning_rate : float, default=0.1
            Taux d'apprentissage.
        subsample : float, default=0.8
            Fraction des echantillons pour chaque arbre.
        colsample_bytree : float, default=0.8
            Fraction des features pour chaque arbre.
        reg_alpha : float, default=0.0
            Regularisation L1.
        reg_lambda : float, default=1.0
            Regularisation L2.
        objective : str, default="reg:squarederror"
            Fonction objectif.
        random_state : int, default=42
            Seed pour reproductibilite.
        n_jobs : int, default=-1
            Nombre de threads (-1 = tous).
        early_stopping_rounds : int, optional
            Nombre de rounds pour early stopping.
        """
        super().__init__(name="XGBoost", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.model: xgb.XGBRegressor | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: list | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
    ) -> "XGBoostModel":
        """Entraine le modele XGBoost."""
        # Use early_stopping_rounds from constructor if not provided
        effective_early_stopping = early_stopping_rounds or self.early_stopping_rounds

        # Create model with early stopping parameters if available
        model_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

        # Try to add early stopping to constructor if supported
        if effective_early_stopping is not None:
            try:
                model_params["early_stopping_rounds"] = effective_early_stopping
            except (TypeError, ValueError):
                # If not supported in constructor, we'll skip early stopping
                pass

        self.model = xgb.XGBRegressor(**model_params)

        fit_params: Dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele XGBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> np.ndarray:
        """
        Retourne l'importance des features.

        Parameters
        ----------
        importance_type : str, default="gain"
            Type d'importance ("weight", "gain", "cover").
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.feature_importances_

    @property
    def best_iteration(self) -> int:
        """Retourne le meilleur nombre d'iterations (si early stopping)."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.best_iteration if hasattr(self.model, "best_iteration") else self.n_estimators
