"""CatBoost model for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd  # type: ignore
from catboost import CatBoostClassifier  # type: ignore

from src.model.base import BaseModel  # type: ignore[import-untyped]


class CatBoostModel(BaseModel):
    """
    CatBoost (Categorical Boosting) classifier.

    Excellent pour les features categoriques, robust au surapprentissage.
    Supporte classification multi-classe (triple-barrier: -1, 0, 1).
    """

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        random_seed: int = 42,
        thread_count: int = -1,
        loss_function: str = "MultiClass",
        class_weights: List[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele CatBoost.

        Parameters
        ----------
        iterations : int, default=100
            Nombre d'arbres.
        depth : int, default=6
            Profondeur maximale des arbres.
        learning_rate : float, default=0.1
            Taux d'apprentissage.
        l2_leaf_reg : float, default=3.0
            Regularisation L2 sur les feuilles.
        random_strength : float, default=1.0
            Force de la randomisation des splits.
        bagging_temperature : float, default=1.0
            Temperature pour le bayesian bagging.
        random_seed : int, default=42
            Seed pour reproductibilite.
        thread_count : int, default=-1
            Nombre de threads.
        loss_function : str, default="MultiClass"
            Fonction de perte pour l'entrainement.
        class_weights : List[float], optional
            Weights for classes in order of sorted class labels.
            Important for imbalanced triple-barrier labels.
        """
        super().__init__(name="CatBoost", **kwargs)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.loss_function = loss_function
        self.class_weights = class_weights
        self.model: CatBoostClassifier | None = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        cat_features: List[int] | List[str] | None = None,
        eval_set: tuple | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "CatBoostModel":
        """
        Entraine le modele CatBoost classifier.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features.
        y : np.ndarray | pd.Series
            Target (labels: -1, 0, 1).
        cat_features : List[int] | List[str], optional
            Indices ou noms des features categoriques.
        eval_set : tuple, optional
            Validation set (X_val, y_val).
        early_stopping_rounds : int, optional
            Nombre de rounds pour early stopping.
        verbose : bool, default=False
            Affiche les logs d'entrainement.
        **kwargs : Any
            Additional keyword arguments passed to CatBoostClassifier.fit().
        """
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)

        model_params = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_strength": self.random_strength,
            "bagging_temperature": self.bagging_temperature,
            "random_seed": self.random_seed,
            "thread_count": self.thread_count,
            "loss_function": self.loss_function,
            "verbose": verbose,
        }
        if self.class_weights is not None:
            model_params["class_weights"] = self.class_weights

        self.model = CatBoostClassifier(**model_params)

        self.model.fit(
            X,
            y_arr,
            cat_features=cat_features,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele CatBoost."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X).ravel()

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
        importance_type: str = "FeatureImportance",
    ) -> np.ndarray:
        """
        Retourne l'importance des features.

        Parameters
        ----------
        importance_type : str, default="FeatureImportance"
            Type d'importance ("FeatureImportance", "ShapValues", etc.).
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        return np.array(self.model.get_feature_importance())

    @property
    def best_iteration(self) -> int:
        """Retourne le meilleur nombre d'iterations."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        best_iter = getattr(self.model, "best_iteration_", None)
        if best_iter is not None:
            return best_iter
        return self.iterations
