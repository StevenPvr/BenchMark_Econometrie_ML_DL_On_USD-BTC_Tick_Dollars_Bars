"""AR(1) / Markov baseline model for multi-class classification (De Prado triple-barrier)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore

from src.model.base import BaseModel


class AR1Baseline(BaseModel):
    """
    Modele baseline Markov d'ordre 1 pour classification multi-classe.

    Utilise les probabilites de transition P(y[t] | y[t-1]).
    Supporte triple-barrier labeling (De Prado): -1, 0, 1.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise le modele Markov(1)."""
        super().__init__(name="AR1Baseline", **kwargs)
        self.classes_: np.ndarray | None = None
        self.transition_matrix_: np.ndarray | None = None
        self._last_value: int | None = None
        self._class_to_idx: dict | None = None
        self._idx_to_class: dict | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        **kwargs: Any,
    ) -> "AR1Baseline":
        """
        Estime les probabilites de transition Markov.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features d'entrainement (non utilise).
        y : np.ndarray | pd.Series
            Target (labels: -1, 0, 1) d'entrainement.

        Returns
        -------
        AR1Baseline
            Le modele entraine (self).
        """
        y_arr = np.asarray(y).ravel()

        # Store unique classes and create mappings
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._idx_to_class = {i: c for c, i in self._class_to_idx.items()}

        # Estimate transition matrix P[i,j] = P(y[t]=j | y[t-1]=i)
        self.transition_matrix_ = np.zeros((n_classes, n_classes))

        y_t = y_arr[1:]
        y_lag1 = y_arr[:-1]

        for i, from_class in enumerate(self.classes_):
            mask = y_lag1 == from_class
            if np.sum(mask) > 0:
                for j, to_class in enumerate(self.classes_):
                    self.transition_matrix_[i, j] = np.mean(y_t[mask] == to_class)
            else:
                # If no transitions from this class, use uniform
                self.transition_matrix_[i, :] = 1.0 / n_classes

        self._last_value = int(y_arr[-1])
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predit en utilisant les probabilites de transition.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features (utilise uniquement pour la taille).

        Returns
        -------
        np.ndarray
            Predictions multi-step (labels: -1, 0, 1).
        """
        if not self.is_fitted or self._last_value is None:
            raise ValueError("Model must be fitted before prediction.")
        if self.transition_matrix_ is None or self._class_to_idx is None or self._idx_to_class is None:
            raise ValueError("Model must be fitted before prediction.")

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        predictions = np.empty(n_samples, dtype=int)

        # Multi-step Markov prediction (deterministic: predict most likely)
        prev_idx = self._class_to_idx[self._last_value]
        for i in range(n_samples):
            # Get most likely next class
            next_idx = np.argmax(self.transition_matrix_[prev_idx])
            predictions[i] = self._idx_to_class[next_idx]
            prev_idx = next_idx

        return predictions

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for all classes.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted or self._last_value is None:
            raise ValueError("Model must be fitted before prediction.")
        if self.transition_matrix_ is None or self._class_to_idx is None:
            raise ValueError("Model must be fitted before prediction.")

        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.empty((n_samples, n_classes))

        # Multi-step probabilities
        prev_idx = self._class_to_idx[self._last_value]
        for i in range(n_samples):
            probabilities[i] = self.transition_matrix_[prev_idx]
            # For next step, use predicted class
            prev_idx = np.argmax(self.transition_matrix_[prev_idx])

        return probabilities

    def predict_with_actuals(
        self,
        X: np.ndarray | pd.DataFrame,
        y_actual: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Predit en utilisant les vraies valeurs precedentes (one-step ahead).

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
        if self.transition_matrix_ is None or self._class_to_idx is None or self._idx_to_class is None:
            raise ValueError("Model must be fitted before prediction.")

        y_arr = np.asarray(y_actual).ravel()
        n_samples = len(y_arr)

        predictions = np.empty(n_samples, dtype=int)

        # First prediction uses last training value
        prev_idx = self._class_to_idx[self._last_value]
        predictions[0] = self._idx_to_class[np.argmax(self.transition_matrix_[prev_idx])]

        # Subsequent predictions use actual previous values
        for i in range(1, n_samples):
            prev_idx = self._class_to_idx[int(y_arr[i - 1])]
            predictions[i] = self._idx_to_class[np.argmax(self.transition_matrix_[prev_idx])]

        return predictions

    def get_transition_matrix(self) -> np.ndarray:
        """
        Retourne la matrice de transition estimee.

        Returns
        -------
        np.ndarray
            Transition matrix of shape (n_classes, n_classes).
        """
        if not self.is_fitted or self.transition_matrix_ is None:
            raise ValueError("Model must be fitted first.")
        return self.transition_matrix_.copy()
