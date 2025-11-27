"""Random baseline model for multi-class classification (De Prado triple-barrier)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore

from src.model.base import BaseModel


class RandomBaseline(BaseModel):
    """
    Modele baseline aleatoire pour classification multi-classe.

    Genere des predictions aleatoires basees sur la distribution des classes
    du dataset d'entrainement (stratified random).

    Supporte triple-barrier labeling (De Prado): -1, 0, 1.
    """

    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        """
        Initialise le modele baseline.

        Parameters
        ----------
        random_state : int, default=42
            Seed pour reproductibilite.
        """
        super().__init__(name="RandomBaseline", **kwargs)
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.class_probs_: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "RandomBaseline":
        """
        Learn the class distribution from training data.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features d'entrainement (non utilise, pour compatibilite).
        y : np.ndarray | pd.Series
            Target (labels: -1, 0, 1) d'entrainement.

        Returns
        -------
        RandomBaseline
            Le modele entraine (self).
        """
        y_arr = np.asarray(y).ravel()
        self._rng = np.random.default_rng(self.random_state)

        # Calculate class probabilities
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        self.class_probs_ = counts / len(y_arr)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Genere des predictions aleatoires basees sur la distribution des classes.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features pour la prediction.

        Returns
        -------
        np.ndarray
            Predictions aleatoires (labels: -1, 0, 1).
        """
        if not self.is_fitted or self.classes_ is None or self.class_probs_ is None:
            raise ValueError("Model must be fitted before prediction.")
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_state)

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        predictions = self._rng.choice(self.classes_, size=n_samples, p=self.class_probs_)
        return predictions

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for all classes.

        Returns the class distribution learned during training
        (same for all samples).

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted or self.class_probs_ is None:
            raise ValueError("Model must be fitted before prediction.")

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        # Repeat class probabilities for all samples
        return np.tile(self.class_probs_, (n_samples, 1))

    def get_distribution_params(self) -> dict[str, Any]:
        """
        Retourne les parametres de la distribution apprise.

        Returns
        -------
        dict[str, Any]
            Dictionnaire avec 'classes' et 'probabilities'.
        """
        if not self.is_fitted or self.classes_ is None or self.class_probs_ is None:
            raise ValueError("Model must be fitted first.")
        return {
            "classes": self.classes_.tolist(),
            "probabilities": self.class_probs_.tolist(),
        }
