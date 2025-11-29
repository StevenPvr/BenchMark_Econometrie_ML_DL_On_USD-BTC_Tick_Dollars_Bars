import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.model.base import BaseModel

class ConcreteModel(BaseModel):
    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X))

def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel("test")

def test_concrete_model_init():
    model = ConcreteModel(name="test_model", param1=10)
    assert model.name == "test_model"
    assert model.params["param1"] == 10
    assert not model.is_fitted
    assert model.model is None

def test_get_params():
    model = ConcreteModel(name="test", a=1, b=2)
    params = model.get_params()
    assert params == {"a": 1, "b": 2}

    # modify dict shouldn't affect model params
    params["a"] = 99
    assert model.params["a"] == 1

def test_set_params():
    model = ConcreteModel(name="test", a=1)
    model.set_params(a=2, b=3)
    assert model.params["a"] == 2
    assert model.params["b"] == 3

def test_save_load(tmp_path):
    model = ConcreteModel(name="test", a=1)

    # Cannot save unfitted model
    with pytest.raises(ValueError, match="Cannot save an unfitted model"):
        model.save(tmp_path / "model.pkl")

    model.fit(np.array([[1]]), np.array([1]))
    assert model.is_fitted

    save_path = tmp_path / "subdir" / "model.pkl"
    model.save(save_path)

    assert save_path.exists()

    loaded_model = ConcreteModel.load(save_path)
    assert isinstance(loaded_model, ConcreteModel)
    assert loaded_model.name == "test"
    assert loaded_model.params["a"] == 1
    assert loaded_model.is_fitted

def test_load_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BaseModel.load(tmp_path / "nonexistent.pkl")
