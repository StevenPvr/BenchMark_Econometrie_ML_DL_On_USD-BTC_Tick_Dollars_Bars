"""XGBoost model for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore

from src.model.base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost (eXtreme Gradient Boosting) classifier.

    Performant pour les donnees tabulaires.
    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
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
        min_child_weight: int = 1,
        gamma: float = 0.0,
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
        min_child_weight : int, default=1
            Minimum sum of instance weight in a child (regularization).
        gamma : float, default=0.0
            Minimum loss reduction to make a split (regularization).
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
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.model: xgb.XGBClassifier | None = None
        self.classes_: np.ndarray | None = None
        self._label_to_xgb: Dict[int, int] = {}
        self._xgb_to_label: Dict[int, int] = {}

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: list | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "XGBoostModel":
        """Entraine le modele XGBoost classifier."""
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)

        # XGBoost requires classes to start from 0: map [-1, 0, 1] -> [0, 1, 2]
        # Create mapping from original labels to XGBoost labels
        classes_arr = self.classes_
        assert classes_arr is not None, "classes_ should not be None after np.unique"
        self._label_to_xgb = {label: idx for idx, label in enumerate(classes_arr)}
        self._xgb_to_label = {idx: label for label, idx in self._label_to_xgb.items()}
        
        # Map y to XGBoost format (0-indexed)
        y_mapped = np.array([self._label_to_xgb[label] for label in y_arr])

        # Use early_stopping_rounds from constructor if not provided
        effective_early_stopping = early_stopping_rounds or self.early_stopping_rounds

        model_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

        self.model = xgb.XGBClassifier(**model_params)

        # early_stopping_rounds must be passed to fit(), not constructor
        fit_params: Dict[str, Any] = {"verbose": verbose}
        if eval_set is not None:
            # Map eval_set labels too
            mapped_eval_set = [
                (X_eval, np.array([self._label_to_xgb[label] for label in np.asarray(y_eval).ravel()]))
                for X_eval, y_eval in eval_set
            ]
            fit_params["eval_set"] = mapped_eval_set
        if effective_early_stopping is not None:
            # Newer xgboost uses attribute-based early stopping
            self.model.set_params(early_stopping_rounds=effective_early_stopping)

        # Forward any extra kwargs (e.g., callbacks)
        fit_params.update(kwargs)

        self.model.fit(X, y_mapped, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele XGBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        # XGBoost predicts 0-indexed labels, remap to original labels
        xgb_pred = self.model.predict(X)
        return np.array([self._xgb_to_label[pred] for pred in xgb_pred])

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for all classes.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
            For triple-barrier: columns correspond to classes [-1, 0, 1].
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict_proba(X)

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
