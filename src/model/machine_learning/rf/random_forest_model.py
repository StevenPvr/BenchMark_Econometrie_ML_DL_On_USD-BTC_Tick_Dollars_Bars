"""Random Forest model for regression/classification."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

from src.model.base import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest model.

    Ensemble d'arbres de decision, robuste et interpretatble.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele Random Forest.

        Parameters
        ----------
        n_estimators : int, default=100
            Nombre d'arbres dans la foret.
        max_depth : int | None, default=None
            Profondeur maximale des arbres (None = pas de limite).
        min_samples_split : int, default=2
            Nombre minimum d'echantillons pour diviser un noeud.
        min_samples_leaf : int, default=1
            Nombre minimum d'echantillons dans une feuille.
        max_features : Union[str, float], default="sqrt"
            Nombre de features pour chaque split ("sqrt", "log2", ou fraction).
        bootstrap : bool, default=True
            Si True, utilise le bootstrap pour les echantillons.
        oob_score : bool, default=False
            Si True, calcule le score out-of-bag.
        random_state : int, default=42
            Seed pour reproductibilite.
        n_jobs : int, default=-1
            Nombre de threads (-1 = tous).
        """
        super().__init__(name="RandomForest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: RandomForestRegressor | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "RandomForestModel":
        """Entraine le modele Random Forest."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,  # type: ignore
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele Random Forest."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Retourne l'importance des features (impurity-based)."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.feature_importances_

    @property
    def oob_score_(self) -> float:
        """Retourne le score out-of-bag (si oob_score=True)."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        if not self.oob_score:
            raise ValueError("oob_score was not enabled during training.")
        return self.model.oob_score_
