"""LightGBM model for regression/classification."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore
import lightgbm as lgb  # type: ignore

from ..base import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM (Light Gradient Boosting Machine) model.

    Rapide et efficace en memoire, bon pour grands datasets.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        objective: str = "regression",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele LightGBM.

        Parameters
        ----------
        n_estimators : int, default=100
            Nombre d'arbres.
        max_depth : int, default=-1
            Profondeur maximale (-1 = pas de limite).
        num_leaves : int, default=31
            Nombre maximal de feuilles par arbre.
        learning_rate : float, default=0.1
            Taux d'apprentissage.
        subsample : float, default=0.8
            Fraction des echantillons (bagging_fraction).
        colsample_bytree : float, default=0.8
            Fraction des features (feature_fraction).
        reg_alpha : float, default=0.0
            Regularisation L1.
        reg_lambda : float, default=0.0
            Regularisation L2.
        objective : str, default="regression"
            Fonction objectif.
        random_state : int, default=42
            Seed pour reproductibilite.
        n_jobs : int, default=-1
            Nombre de threads.
        """
        super().__init__(name="LightGBM", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: lgb.LGBMRegressor | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: list | None = None,
        callbacks: list | None = None,
    ) -> "LightGBMModel":
        """Entraine le modele LightGBM."""
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

        fit_params: Dict[str, Any] = {}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set
        if callbacks is not None:
            fit_params["callbacks"] = callbacks

        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele LightGBM."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        predictions = self.model.predict(X)
        return np.asarray(predictions)

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> np.ndarray:
        """
        Retourne l'importance des features.

        Parameters
        ----------
        importance_type : str, default="gain"
            Type d'importance ("split", "gain").
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.feature_importances_

    @property
    def best_iteration(self) -> int:
        """Retourne le meilleur nombre d'iterations."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.best_iteration_ if hasattr(self.model, "best_iteration_") else self.n_estimators
