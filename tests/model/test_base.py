"""Tests for src/model/base.py module."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd # type: ignore[import-untyped]
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> "ConcreteModel":
        """Simple fit implementation."""
        self.model = {"mean": np.mean(y)}
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Simple predict implementation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return np.full(len(X), self.model["mean"])


class TestBaseModelInit:
    """Test cases for BaseModel initialization."""

    def test_init_with_name(self):
        """Should initialize with name."""
        model = ConcreteModel(name="TestModel")

        assert model.name == "TestModel"
        assert model.is_fitted is False
        assert model.model is None

    def test_init_with_kwargs(self):
        """Should store kwargs as params."""
        model = ConcreteModel(name="TestModel", alpha=0.1, beta=0.2)

        assert model.params == {"alpha": 0.1, "beta": 0.2}

    def test_init_default_values(self):
        """Should have default empty params."""
        model = ConcreteModel(name="TestModel")

        assert model.params == {}


class TestBaseModelGetSetParams:
    """Test cases for get_params and set_params methods."""

    def test_get_params_returns_copy(self):
        """get_params should return a copy of params."""
        model = ConcreteModel(name="TestModel", alpha=0.1)
        params = model.get_params()

        params["alpha"] = 999
        assert model.params["alpha"] == 0.1

    def test_set_params_updates(self):
        """set_params should update params."""
        model = ConcreteModel(name="TestModel", alpha=0.1)
        model.set_params(alpha=0.5, gamma=0.3)

        assert model.params["alpha"] == 0.5
        assert model.params["gamma"] == 0.3

    def test_set_params_returns_self(self):
        """set_params should return self for chaining."""
        model = ConcreteModel(name="TestModel")
        result = model.set_params(alpha=0.1)

        assert result is model


class TestBaseModelFitPredict:
    """Test cases for fit and predict methods."""

    def test_fit_sets_is_fitted(self):
        """fit should set is_fitted to True."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model.fit(X, y)

        assert model.is_fitted is True

    def test_fit_returns_self(self):
        """fit should return self."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])

        result = model.fit(X, y)

        assert result is model

    def test_predict_after_fit(self):
        """predict should work after fit."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 3
        assert predictions[0] == pytest.approx(2.0)

    def test_predict_before_fit_raises(self):
        """predict should raise if not fitted."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2]])

        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_fit_with_dataframe(self):
        """fit should work with pandas DataFrame."""
        model = ConcreteModel(name="TestModel")
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 2, 3])

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 3


class TestBaseModelSaveLoad:
    """Test cases for save and load methods."""

    def test_save_creates_file(self):
        """save should create file on disk."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            model.save(path)

            assert path.exists()

    def test_save_unfitted_raises(self):
        """save should raise if model not fitted."""
        model = ConcreteModel(name="TestModel")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"

            with pytest.raises(ValueError, match="unfitted"):
                model.save(path)

    def test_load_restores_model(self):
        """load should restore model state."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2], [3, 4]])
        y = np.array([10, 20])
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            model.save(path)

            loaded = ConcreteModel.load(path)

            assert loaded.name == "TestModel"
            assert loaded.is_fitted is True
            assert loaded.predict(X)[0] == pytest.approx(15.0)

    def test_load_nonexistent_raises(self):
        """load should raise if file not found."""
        path = Path("/nonexistent/path/model.joblib")

        with pytest.raises(FileNotFoundError):
            ConcreteModel.load(path)

    def test_save_creates_parent_dirs(self):
        """save should create parent directories."""
        model = ConcreteModel(name="TestModel")
        X = np.array([[1, 2]])
        y = np.array([1])
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "subdir" / "deep" / "model.joblib"
            model.save(path)

            assert path.exists()


class TestBaseModelRepr:
    """Test cases for __repr__ method."""

    def test_repr_unfitted(self):
        """repr should show unfitted state."""
        model = ConcreteModel(name="TestModel")

        repr_str = repr(model)

        assert "ConcreteModel" in repr_str
        assert "TestModel" in repr_str
        assert "fitted=False" in repr_str

    def test_repr_fitted(self):
        """repr should show fitted state."""
        model = ConcreteModel(name="TestModel")
        model.fit(np.array([[1]]), np.array([1]))

        repr_str = repr(model)

        assert "fitted=True" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
