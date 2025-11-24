"""Base model interface for all models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd  # type: ignore


class BaseModel(ABC):
    """
    Interface de base pour tous les modeles du projet.

    Tous les modeles doivent implementer:
    - fit(X, y) : entrainer le modele
    - predict(X) : faire des predictions
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialise le modele.

        Parameters
        ----------
        name : str
            Nom du modele.
        **kwargs : Any
            Hyperparametres du modele.
        """
        self.name = name
        self.params = kwargs
        self.is_fitted = False
        self.model: Any = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "BaseModel":
        """
        Entraine le modele sur les donnees.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features d'entrainement.
        y : np.ndarray | pd.Series
            Target d'entrainement.

        Returns
        -------
        BaseModel
            Le modele entraine (self).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Fait des predictions sur de nouvelles donnees.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features pour la prediction.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Retourne les hyperparametres du modele."""
        return self.params.copy()

    def set_params(self, **params: Any) -> "BaseModel":
        """Met a jour les hyperparametres du modele."""
        self.params.update(params)
        return self

    def save(self, path: Path) -> None:
        """
        Sauvegarde le modele sur disque.

        Parameters
        ----------
        path : Path
            Chemin de sauvegarde.
        """
        import joblib  # type: ignore

        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        Charge un modele depuis le disque.

        Parameters
        ----------
        path : Path
            Chemin du modele sauvegarde.

        Returns
        -------
        BaseModel
            Le modele charge.
        """
        import joblib  # type: ignore

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
