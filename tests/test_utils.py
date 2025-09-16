import os
import numpy as np
from src.utils import save_object, load_object


def test_save_and_load_roundtrip(tmp_path):
    obj = {"a": 1, "b": [1, 2, 3]}
    file_path = tmp_path / "obj.pkl"
    save_object(str(file_path), obj)
    loaded = load_object(str(file_path))
    assert loaded == obj
