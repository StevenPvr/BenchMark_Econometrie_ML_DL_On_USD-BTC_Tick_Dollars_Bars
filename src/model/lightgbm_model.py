"""LightGBM model for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.model.base import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM (Light Gradient Boosting Machine) classifier.

    Plus rapide que XGBoost et Random Forest sur gros datasets.
    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
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
        min_child_samples: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele LightGBM.

        Parameters
        ----------
        n_estimators : int, default=100
            Nombre d'arbres (boosting rounds).
        max_depth : int, default=-1
            Profondeur maximale (-1 = illimite).
        num_leaves : int, default=31
            Nombre maximum de feuilles par arbre.
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
        min_child_samples : int, default=20
            Minimum samples in leaf.
        random_state : int, default=42
            Seed pour reproductibilite.
        n_jobs : int, default=-1
            Nombre de threads (-1 = tous).
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
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: lgb.LGBMClassifier | None = None
        self.classes_: np.ndarray | None = None
        self._label_to_lgb: Dict[int, int] = {}
        self._lgb_to_label: Dict[int, int] = {}

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: list | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "LightGBMModel":
        """Entraine le modele LightGBM classifier."""
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)

        # LightGBM requires classes to start from 0: map [-1, 0, 1] -> [0, 1, 2]
        classes_arr = self.classes_
        assert classes_arr is not None
        self._label_to_lgb = {label: idx for idx, label in enumerate(classes_arr)}
        self._lgb_to_label = {idx: label for label, idx in self._label_to_lgb.items()}

        # Map y to LightGBM format (0-indexed)
        y_mapped = np.array([self._label_to_lgb[label] for label in y_arr])

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,  # Suppress warnings
        )

        fit_params: Dict[str, Any] = {}
        if eval_set is not None:
            mapped_eval_set = [
                (X_eval, np.array([self._label_to_lgb[label] for label in np.asarray(y_eval).ravel()]))
                for X_eval, y_eval in eval_set
            ]
            fit_params["eval_set"] = mapped_eval_set
        if early_stopping_rounds is not None:
            fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        fit_params.update(kwargs)
        self.model.fit(X, y_mapped, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele LightGBM."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        lgb_pred = self.model.predict(X)
        return np.array([self._lgb_to_label[pred] for pred in lgb_pred])

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
            Type d'importance ("split", "gain").
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.feature_importances_

    @property
    def best_iteration(self) -> int:
        """Retourne le meilleur nombre d'iterations (si early stopping)."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return getattr(self.model, "best_iteration_", self.n_estimators)
