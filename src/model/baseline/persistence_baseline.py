"""Persistence baseline model for multi-class classification (De Prado triple-barrier)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore

from src.model.base import BaseModel


class PersistenceBaseline(BaseModel):
    """
    Modele baseline de persistance pour classification multi-classe.

    Predit que la prochaine classe sera egale a la classe precedente.
    Supporte triple-barrier labeling (De Prado): -1, 0, 1.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise le modele de persistance."""
        super().__init__(name="PersistenceBaseline", **kwargs)
        self._last_value: int | None = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "PersistenceBaseline":
        """
        Stocke la derniere valeur du train set.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features d'entrainement (non utilise).
        y : np.ndarray | pd.Series
            Target (labels: -1, 0, 1) d'entrainement.

        Returns
        -------
        PersistenceBaseline
            Le modele entraine (self).
        """
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)
        self._last_value = int(y_arr[-1])
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predit en utilisant la valeur precedente.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features pour la prediction (utilise uniquement pour la taille).

        Returns
        -------
        np.ndarray
            Predictions (toutes egales a la derniere valeur du train).
        """
        if not self.is_fitted or self._last_value is None:
            raise ValueError("Model must be fitted before prediction.")

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n_samples, self._last_value)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for all classes.

        Returns one-hot encoded probabilities based on last observed class.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted or self._last_value is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        n_classes = len(self.classes_)

        # Create one-hot probabilities
        proba = np.zeros((n_samples, n_classes))
        class_idx = np.where(self.classes_ == self._last_value)[0][0]
        proba[:, class_idx] = 1.0

        return proba

    def predict_with_actuals(
        self,
        X: np.ndarray | pd.DataFrame,
        y_actual: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Predit en utilisant les vraies valeurs precedentes (one-step ahead).

        y_pred[0] = derniere valeur du train
        y_pred[t] = y_actual[t-1] pour t > 0

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features (utilise uniquement pour la taille).
        y_actual : np.ndarray | pd.Series
            Vraies valeurs du test set.

        Returns
        -------
        np.ndarray
            Predictions one-step ahead.
        """
        if not self.is_fitted or self._last_value is None:
            raise ValueError("Model must be fitted before prediction.")

        y_arr = np.asarray(y_actual).ravel()
        n_samples = len(y_arr)

        predictions = np.empty(n_samples, dtype=int)
        predictions[0] = self._last_value
        predictions[1:] = y_arr[:-1]

        return predictions

    def get_last_value(self) -> int:
        """Retourne la derniere valeur stockee."""
        if not self.is_fitted or self._last_value is None:
            raise ValueError("Model must be fitted first.")
        return self._last_value
