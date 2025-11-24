"""Lightweight LSTM model for time series forecasting."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore

from ..base import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM (Long Short-Term Memory) leger pour series temporelles.

    Version legere et optimisee pour la prediction financiere.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        sequence_length: int = 10,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """
        Initialise le modele LSTM.

        Parameters
        ----------
        input_size : int, default=1
            Nombre de features en entree.
        hidden_size : int, default=32
            Taille de l'etat cache du LSTM.
        num_layers : int, default=1
            Nombre de couches LSTM empilees.
        dropout : float, default=0.0
            Dropout entre les couches (si num_layers > 1).
        sequence_length : int, default=10
            Longueur des sequences d'entree.
        learning_rate : float, default=0.001
            Taux d'apprentissage.
        epochs : int, default=100
            Nombre d'epoques d'entrainement.
        batch_size : int, default=32
            Taille des batchs.
        patience : int, default=10
            Patience pour early stopping.
        device : str, default="auto"
            Device ("cpu", "cuda", "mps", "auto").
        """
        super().__init__(name="LSTM", **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device_str = device

        self.model: Any = None
        self.scaler: Any = None
        self._device: Any = None
        self._history: dict = {"train_loss": [], "val_loss": []}

    def _get_device(self) -> Any:
        """Determine le meilleur device disponible."""
        import torch  # type: ignore

        if self.device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device_str)

    def _build_model(self) -> Any:
        """Construit le modele LSTM PyTorch."""
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        class LSTMNetwork(nn.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                lstm_out, _ = self.lstm(x)
                # Prendre la derniere sortie de la sequence
                out = self.fc(lstm_out[:, -1, :])
                return out

        model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return model.to(self._device)

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Cree des sequences pour le LSTM.

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features).
        y : np.ndarray, optional
            Target.

        Returns
        -------
        Tuple
            Sequences X et y correspondantes.
        """
        X_seq: List[np.ndarray] = []
        y_seq: Optional[List[Any]] = [] if y is not None else None

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i : i + self.sequence_length])
            if y is not None and y_seq is not None:
                y_seq.append(y[i + self.sequence_length])

        X_arr = np.array(X_seq)
        y_arr = np.array(y_seq) if y_seq is not None else None

        return X_arr, y_arr

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        verbose: bool = True,
    ) -> "LSTMModel":
        """
        Entraine le modele LSTM.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features d'entrainement.
        y : np.ndarray | pd.Series
            Target d'entrainement.
        X_val : np.ndarray | pd.DataFrame, optional
            Features de validation.
        y_val : np.ndarray | pd.Series, optional
            Target de validation.
        verbose : bool, default=True
            Affiche la progression.
        """
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        # Normalisation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        # Si X est 1D, ajouter une dimension
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        self.input_size = X_scaled.shape[1]

        # Creer les sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_arr)

        if X_seq is None or len(X_seq) == 0:
            raise ValueError("Not enough data for sequence creation.")

        # Device et modele
        self._device = self._get_device()
        self.model = self._build_model()

        # Convertir en tenseurs
        X_tensor = torch.FloatTensor(X_seq).to(self._device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self._device)

        # DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation set
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_arr = np.asarray(X_val)
            y_val_arr = np.asarray(y_val).ravel()
            X_val_scaled = self.scaler.transform(X_val_arr)
            if X_val_scaled.ndim == 1:
                X_val_scaled = X_val_scaled.reshape(-1, 1)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_arr)
            if X_val_seq is not None and len(X_val_seq) > 0:
                X_val_tensor = torch.FloatTensor(X_val_seq).to(self._device)
                y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self._device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            self._history["train_loss"].append(avg_train_loss)

            # Validation phase
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                self._history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Fait des predictions avec le modele LSTM.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features pour la prediction.
            Doit avoir au moins sequence_length lignes.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        import torch  # type: ignore

        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        X_arr = np.asarray(X)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_arr)
        else:
            X_scaled = X_arr

        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        # Creer les sequences
        X_seq, _ = self._create_sequences(X_scaled, None)

        if X_seq is None or len(X_seq) == 0:
            raise ValueError("Not enough data for prediction (need >= sequence_length rows).")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self._device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions.ravel()

    def get_history(self) -> dict:
        """Retourne l'historique d'entrainement."""
        return self._history.copy()
