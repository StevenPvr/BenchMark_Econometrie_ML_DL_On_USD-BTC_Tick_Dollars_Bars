"""LightGBM model for multi-class classification (De Prado triple-barrier labeling)."""

from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
import pandas as pd  # type: ignore
import lightgbm as lgb  # type: ignore

from src.model.base import BaseModel  # type: ignore[import-untyped]


class LightGBMModel(BaseModel):
    """
    LightGBM (Light Gradient Boosting Machine) classifier.

    Rapide et efficace en memoire, bon pour grands datasets.
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
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: dict[Any, float] | str | None = None,
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
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.model: lgb.LGBMClassifier | None = None
        self.classes_: np.ndarray | None = None
        # Keep feature names to avoid sklearn warnings about missing names at prediction time
        self.feature_names: list[str] | None = None

    def _prepare_features(
        self,
        X: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert input features to a DataFrame and align columns.

        LightGBM attaches feature names during fit; passing arrays without names
        at predict time triggers sklearn warnings. This helper keeps column names
        consistent between fit and predict.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if feature_names is None:
                feature_names = list(df.columns)
            else:
                missing = [name for name in feature_names if name not in df.columns]
                if missing:
                    raise ValueError(f"Missing features in input data: {missing}")
                df = df[feature_names]  # type: ignore[assignment]
            return cast(pd.DataFrame, df), feature_names

        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        cols = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)  # type: ignore[arg-type]
        return df, cols

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_set: list | None = None,
        callbacks: list | None = None,
        **kwargs: Any,
    ) -> "LightGBMModel":
        """Entraine le modele LightGBM classifier."""
        X_df, feature_names = self._prepare_features(X)
        self.feature_names = feature_names
        y_arr = np.asarray(y).ravel()

        # Store unique classes for later use
        self.classes_ = np.unique(y_arr)

        model_params: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": -1,
        }
        if self.class_weight is not None:
            model_params["class_weight"] = cast(Any, self.class_weight)

        self.model = lgb.LGBMClassifier(**model_params)

        fit_params: Dict[str, Any] = {}
        if eval_set is not None:
            processed_eval_set = []
            for X_eval, y_eval in eval_set:
                X_eval_df, _ = self._prepare_features(X_eval, feature_names)
                processed_eval_set.append((X_eval_df, np.asarray(y_eval).ravel()))
            fit_params["eval_set"] = processed_eval_set
        if callbacks is not None:
            fit_params["callbacks"] = callbacks

        self.model.fit(X_df, y_arr, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fait des predictions avec le modele LightGBM."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        if self.feature_names is None:
            raise ValueError("Feature names not set. Fit the model before predicting.")

        X_df, _ = self._prepare_features(X, self.feature_names)
        predictions = self.model.predict(X_df)
        return np.asarray(predictions)

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
        if self.feature_names is None:
            raise ValueError("Feature names not set. Fit the model before predicting.")

        X_df, _ = self._prepare_features(X, self.feature_names)
        return np.asarray(self.model.predict_proba(X_df))

    def set_params(self, **params: Any) -> "BaseModel":
        """Met a jour les hyperparametres du modele."""
        super().set_params(**params)
        return self

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
